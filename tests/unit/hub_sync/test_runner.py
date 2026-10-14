"""The runner: quiet without a login, single-flight, check-only in manual mode, prompt to stop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.identity import LinkPolicy
from local_operator.hub_sync.runner import HubSyncRunner, TickReport
from local_operator.hub_sync.settings import HubSyncSettings
from tests.unit.hub_sync.test_service import BASE, Hub, _pull_agent


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / ".local-operator"
    hub = Hub()
    monkeypatch.setattr(
        "local_operator.agents._fetch_hub_profile", lambda _c, h, **_k: hub.agents[h]
    )
    agents = AgentRegistry(root)
    cm = ConfigManager(root)
    state: dict[str, Any] = {"credential": "ok", "clients": 0}

    def build() -> svc.HubSyncContext:
        state["clients"] += 1
        return svc.HubSyncContext(
            config_dir=root,
            config_manager=cm,
            client_for_tenant=(
                (lambda _t: hub.client()) if state["credential"] == "ok" else (lambda _t: None)
            ),
            credential=state["credential"],
            agent_registry=agents,
            resolver=None,
        )

    runner = HubSyncRunner(config_manager=cm, build_ctx=build, startup_delay=0)
    return runner, hub, agents, root, cm, state


def _drift(agents, hub, root):
    row = _pull_agent(agents, hub, root)
    hub.agents["h1"] = (BASE + "\n\n## New\nAdded upstream.", "d")
    return row


@pytest.mark.asyncio
async def test_first_tick_after_upgrade_checks_but_never_applies(rig) -> None:
    runner, hub, agents, root, cm, _ = rig
    row = _drift(agents, hub, root)
    first = await runner.tick(reason="startup")
    assert first.available == 1 and first.applied == 0  # first_run_grace
    assert "Added upstream" not in agents.get_agent_system_prompt(row.id)
    second = await runner.tick(reason="timer")
    assert second.applied == 1 and "Added upstream" in agents.get_agent_system_prompt(row.id)


@pytest.mark.asyncio
async def test_manual_mode_keeps_checking_and_records_available_without_writing(rig) -> None:
    runner, hub, agents, root, cm, _ = rig
    row = _drift(agents, hub, root)
    cm.set_config_value("hub", {"auto_update": {"agents": False, "teams": False}})
    await runner.tick(reason="startup")
    report = await runner.tick(reason="timer")
    assert report.available == 1 and report.applied == 0
    assert "Added upstream" not in agents.get_agent_system_prompt(row.id)
    assert st.StatusStore(root).load()["items"][f"agent:{row.id}"]["state"] == "available"


@pytest.mark.asyncio
async def test_no_login_means_no_org_fetch_and_no_failure_noise(rig) -> None:
    runner, hub, agents, root, cm, state = rig
    row = _pull_agent(agents, hub, root)
    prov.record_agent_baseline(
        root, local_id=row.id, hub_id="h1", instructions=BASE, description="d", tenant_id="org-1"
    )
    state["credential"] = "none"
    report = await runner.tick(reason="startup")
    assert report.applied == 0 and report.failed == 0
    item = st.StatusStore(root).load()["items"][f"agent:{row.id}"]
    # The class IS recorded - it is the only way the UI can tell this user they
    # need to sign in (UX round 2, U11) - but it is not failure NOISE: no attempt
    # is counted, no retry is scheduled and nothing is described as broken.
    assert item["error_class"] == "no-credential" and item["last_error"] is None
    assert item["state"] != "failed" and item["attempts"] == 0
    assert item["next_retry_at"] is None and item["auto_retry"] is True


@pytest.mark.asyncio
async def test_no_linked_items_costs_nothing(rig) -> None:
    runner, *_ = rig
    report = await runner.tick(reason="startup")
    assert report.skipped == "no-linked-items" and report.checked == 0


@pytest.mark.asyncio
async def test_the_lease_makes_a_second_runner_skip_instead_of_double_applying(rig) -> None:
    runner, hub, agents, root, *_ = rig
    _drift(agents, hub, root)
    other = st.RunnerLease(root)
    assert other.acquire()
    try:
        report = await runner.tick(reason="timer")
        assert report.skipped == "lease-held" and report.applied == 0
    finally:
        other.release()


@pytest.mark.asyncio
async def test_ticks_are_single_flight(rig) -> None:
    runner, hub, agents, root, *_ = rig
    _drift(agents, hub, root)
    a, b = await asyncio.gather(runner.tick(reason="timer"), runner.tick(reason="timer"))
    assert sorted([a.applied, b.applied]) in ([0, 0], [0, 1])  # never both


@pytest.mark.asyncio
async def test_a_broken_tick_never_kills_the_loop_and_stop_ends_it_promptly(
    rig, monkeypatch
) -> None:
    runner, *_ = rig
    calls = {"n": 0}

    async def boom(*, reason: str):
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "tick", boom)
    monkeypatch.setattr("local_operator.hub_sync.runner.INTERVAL_JITTER", 0.0)
    runner._cm.set_config_value("hub", {"check_interval_min": 5})
    task = asyncio.create_task(runner.run_forever())
    for _ in range(50):
        await asyncio.sleep(0.01)
        if calls["n"]:
            break
    assert calls["n"] == 1 and not task.done()
    runner.stop()
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_a_tick_is_bounded_to_fifty_items_oldest_checked_first(rig) -> None:
    runner, hub, agents, root, *_ = rig
    from local_operator.hub_sync.check import Link

    links = [Link("agent", f"id{i}", f"n{i}", "h", None) for i in range(60)]
    doc = {
        "items": {
            f"agent:id{i}": {"last_checked_at": f"2026-01-01T00:{i % 60:02d}:00Z"}
            for i in range(60)
        }
    }
    picked = runner._pick(links, doc, {})
    assert len(picked["agent"]) == 50 and "id0" in picked["agent"] and "id59" not in picked["agent"]
    assert picked["team"] == set()


@pytest.mark.asyncio
async def test_a_setting_written_by_another_process_is_read_on_the_next_tick(rig) -> None:
    """The ``hub`` section is LIVE: the runner's own manager is stale, the disk is not."""

    runner, hub, agents, root, cm, _ = rig
    row = _drift(agents, hub, root)
    await runner.tick(reason="startup")  # first-run grace: checks, applies nothing
    # A second manager stands in for the TUI / CLI / another daemon writing config.yml.
    ConfigManager(root).set_config_value("hub", {"auto_update": {"agents": False}})
    report = await runner.tick(reason="timer")
    assert report.available == 1 and report.applied == 0
    assert "Added upstream" not in agents.get_agent_system_prompt(row.id)


@pytest.mark.asyncio
async def test_route_work_and_the_timer_share_one_lock(rig) -> None:
    """``run_exclusive`` queues behind a running tick instead of racing it."""

    runner, hub, agents, root, *_ = rig
    _drift(agents, hub, root)
    order: list[str] = []

    real_tick = runner._tick_locked

    async def slow_tick(reason: str):
        order.append("tick-start")
        await asyncio.sleep(0.05)
        out = await real_tick(reason)
        order.append("tick-end")
        return out

    runner._tick_locked = slow_tick  # type: ignore[method-assign]
    tick = asyncio.create_task(runner.tick(reason="timer"))
    await asyncio.sleep(0.01)
    await runner.run_exclusive(lambda _ctx: order.append("route"))
    await tick
    assert order == ["tick-start", "tick-end", "route"]


@pytest.mark.asyncio
async def test_an_apply_holds_the_cross_process_lease_and_refuses_when_it_is_taken(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI, the routes and the tool all reach ``apply_items``: it takes the lease."""

    runner, hub, agents, root, cm, _ = rig
    row = _drift(agents, hub, root)
    ctx = await runner.context()
    monkeypatch.setattr(svc, "LEASE_WAIT_S", 0.0)
    other = st.RunnerLease(root)
    assert other.acquire()
    try:
        with pytest.raises(svc.HubBusy):
            await asyncio.to_thread(svc.apply_items, ctx, kind="agent")
        assert "Added upstream" not in agents.get_agent_system_prompt(row.id)
    finally:
        other.release()
    report = await asyncio.to_thread(svc.apply_items, ctx, kind="agent")
    assert report.reports[0].applied


@pytest.mark.asyncio
async def test_a_refusal_names_when_a_dead_holders_lease_expires(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-3(b): not a vague "try again" -- the caller learns the crash-TTL bound."""

    runner, _hub, _agents, root, _cm, _ = rig
    ctx = await runner.context()
    monkeypatch.setattr(svc, "LEASE_WAIT_S", 0.0)
    other = st.RunnerLease(root)
    assert other.acquire()
    try:
        with pytest.raises(svc.HubBusy, match=r"expires in about \d+s"):
            await asyncio.to_thread(svc.apply_items, ctx, kind="agent")
        left = other.remaining_s()
        assert left is not None and 0 < left <= st.LEASE_TTL_S
    finally:
        other.release()
    assert other.remaining_s() is None  # released -> no file -> nothing to wait for


@pytest.mark.asyncio
async def test_the_lease_heartbeat_outlives_its_ttl(tmp_path: Path) -> None:
    lease = st.RunnerLease(tmp_path, ttl_s=0.3)
    assert lease.acquire()
    try:
        await asyncio.sleep(0.8)  # > 2 TTLs
        assert not st.RunnerLease(tmp_path, ttl_s=0.3).acquire()
    finally:
        lease.release()
    contender = st.RunnerLease(tmp_path, ttl_s=0.3)
    assert contender.acquire()
    contender.release()


@pytest.mark.asyncio
async def test_pick_never_mixes_agent_and_team_ids(rig) -> None:
    """A bare-id set would let an agent id select a same-string team id."""
    runner, hub, agents, root, *_ = rig
    from local_operator.hub_sync.check import Link

    links = [Link("agent", "shared", "a", "h", None), Link("team", "shared", "t", "h", None)]
    picked = runner._pick(links, {"items": {}}, {})
    assert picked["agent"] == {"shared"} and picked["team"] == {"shared"}
    # Both kinds are addressed independently; the runner hands each its own set.
    assert set(picked) == {"agent", "team", "skill"}


@pytest.mark.asyncio
async def test_pick_skips_a_both_off_link_and_orders_retry_first(rig) -> None:
    """Fairness: a both-OFF link does no periodic work; a due retry sorts ahead."""
    runner, hub, agents, root, *_ = rig
    from local_operator.hub_sync.check import Link
    from local_operator.hub_sync.identity import LinkIdentity, LinkPolicy

    ident = LinkIdentity("agent", "a-ok", "https://hub.test/v1", "public", None, "h")
    ident_off = LinkIdentity("agent", "a-off", "https://hub.test/v1", "public", None, "h2")
    links = [
        Link("agent", "a-ok", "ok", "h", None, link_id=ident.link_id),
        Link("agent", "a-off", "off", "h2", None, link_id=ident_off.link_id),
    ]
    policies = {
        ident.link_id: LinkPolicy(subscribe=True, auto_push=False),
        ident_off.link_id: LinkPolicy(subscribe=False, auto_push=False),
    }
    picked = runner._pick(links, {"items": {}}, policies)
    assert picked["agent"] == {"a-ok"}  # the both-off link does nothing


@pytest.mark.asyncio
async def test_push_phase_skips_links_without_an_outbound_receipt(rig, monkeypatch) -> None:
    """A link with no (A,P) receipt is not pushed: first publication is deliberate."""
    runner, _hub, _agents, root, cm, _ = rig
    from local_operator.hub_sync import push as push_mod
    from local_operator.hub_sync.check import ItemCheck

    identity, _record = _seed_pushable(root, cm, receipt=False)
    calls: list[dict[str, Any]] = []

    async def fake_push_item(**kw):
        calls.append(kw)
        return push_mod.PushOutcome("pushed")

    monkeypatch.setattr(push_mod, "push_item", fake_push_item)
    transport_calls: list[int] = []
    monkeypatch.setattr(
        runner,
        "_push_transport",
        lambda ctx, record: (transport_calls.append(1) or object(), None),
    )
    item = ItemCheck(
        "agent",
        "a1",
        "a",
        "h1",
        None,
        "available",
        local={"instructions": "l"},
        remote={"instructions": "r"},
        remote_revision='"rev-1"',
    )
    import dataclasses as dc

    item = dc.replace(item, link_id=identity.link_id)
    await runner._push_due(
        await runner.context(),
        [item],
        {identity.link_id: LinkPolicy(True, True)},
        HubSyncSettings.read_fresh(cm),
        False,
        st.StatusStore(root).load(),
        TickReport("timer"),
    )
    assert calls == []  # no receipt -> not pushed (the refusal precedes any outbox write)
    # The refusal is PERSISTED with its name, so the surface can explain the skip.
    assert _item(root, "agent:a1")["push_blocked_reason"] == "no-receipt"


def _seed_pushable(
    root: Path,
    cm: ConfigManager,
    *,
    cas: bool = True,
    grant_contract: int | None = 1,
    receipt: bool = True,
    source: str = "https://hub.test/v1",
    local_id: str = "a1",
    hub_id: str = "h1",
):
    """A public agent link with identity, revision, policy, grant and (A,P) receipt.

    This is the exact fact set ``_eligibility`` reads; each test then mutates ONE
    of them to reach a distinct refusal (the three QA triggers plus the control).
    """

    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkIdentity
    from local_operator.hub_sync.provenance import OutboundReceipt, fingerprint_snapshot

    identity = LinkIdentity("agent", local_id, source, "public", None, hub_id)
    record = prov.make_record("agent", local_id, hub_id, None, {"instructions": "x"}, "merge-pull")
    import dataclasses as dc

    local_fields = {"instructions": "l"}
    record = dc.replace(
        record,
        identity=identity,
        remote_revision='"rev-1"',
        lineage="clean",
        scope_known=True,
        outbound=(
            OutboundReceipt(
                local_fields=local_fields,
                accepted_fields={"instructions": "a"},
                # The stored receipt is validated on read (its fingerprint must
                # match its captured fields), so the seed must be a REAL pair.
                input_fingerprint=fingerprint_snapshot("agent", local_fields),
                remote_revision='"rev-1"',
                operation_id="op",
            )
            if receipt
            else None
        ),
    )
    prov.write_baseline(root, record)
    hub_cfg: dict[str, Any] = {
        "links": {identity.link_id: {"subscribe": True, "auto_push": True}},
        "auto_update": {"agents": True, "teams": True},
    }
    if grant_contract is not None:
        hub_cfg["public_egress_grants"] = {
            identity.link_id: {"link_id": identity.link_id, "publication_contract": grant_contract}
        }
    cm.set_config_value("hub", hub_cfg)
    if cas:
        svc.record_server_capabilities(
            svc.HubSyncContext(
                config_dir=root, config_manager=cm, client_for_tenant=lambda _t: None
            ),
            {"revision_cas": 1},
            source=source,
        )
    _seed_item(root)
    return identity, record


def _seed_item(root: Path, key: str = "agent:a1") -> None:
    """The store item the push phase keys on (empty item gates are all "due")."""

    def fold(doc: dict[str, Any]) -> None:
        doc.setdefault("items", {})[key] = {"state": "up-to-date"}

    st.StatusStore(root).mutate(fold)


def _item(root: Path, key: str) -> dict[str, Any]:
    return st.StatusStore(root).load()["items"][key]


async def _push_due_once(runner, root, cm, identity, item, monkeypatch, *, fake_push_item=None):
    """Drive ONE push phase for the seeded link; returns the recorded transport calls."""

    from local_operator.hub_sync import push as push_mod

    transport_calls: list[Any] = []
    monkeypatch.setattr(
        runner,
        "_push_transport",
        lambda ctx, record: (transport_calls.append(record) or object(), None),
    )
    calls: list[dict[str, Any]] = []

    async def default_fake(**kw):
        calls.append(kw)
        return push_mod.PushOutcome("pushed")

    monkeypatch.setattr(push_mod, "push_item", fake_push_item or default_fake)
    await runner._push_due(
        await runner.context(),
        [item],
        {identity.link_id: LinkPolicy(True, True)},
        HubSyncSettings.read_fresh(cm),
        False,
        st.StatusStore(root).load(),
        TickReport("timer"),
    )
    return transport_calls, calls


@pytest.mark.asyncio
async def test_push_recomputes_eligibility_with_no_attestation(rig, monkeypatch) -> None:
    """QA trigger 1: a hand-edited ``auto_push: true`` without an attestation never publishes."""
    runner, _hub, _agents, root, cm, _ = rig
    from local_operator.hub_sync.check import ItemCheck

    identity, _record = _seed_pushable(root, cm, cas=False)
    item = ItemCheck(
        "agent",
        "a1",
        "a",
        "h1",
        None,
        "available",
        local={"instructions": "l"},
        remote={"instructions": "r"},
        remote_revision='"rev-1"',
        link_id=identity.link_id,
    )
    transport_calls, push_calls = await _push_due_once(
        runner, root, cm, identity, item, monkeypatch
    )
    assert transport_calls == [] and push_calls == []  # no outbox write, no commit
    assert _item(root, "agent:a1")["push_blocked_reason"] == "unsupported-precondition"


@pytest.mark.asyncio
async def test_push_recomputes_eligibility_when_the_attestation_is_revoked(
    rig, monkeypatch
) -> None:
    """QA trigger 2: an attestation that no longer holds skips with the same name."""
    runner, _hub, _agents, root, cm, _ = rig
    from local_operator.hub_sync.check import ItemCheck

    identity, _record = _seed_pushable(root, cm)

    # Simulate a revoked observation: the per-source entry drops to zero.
    def fold(doc: dict[str, Any]) -> None:
        doc["server"]["sources"]["https://hub.test/v1"] = {"revision_cas": 0, "skills": 0}

    st.StatusStore(root).mutate(fold)
    item = ItemCheck(
        "agent",
        "a1",
        "a",
        "h1",
        None,
        "available",
        local={"instructions": "l"},
        remote={"instructions": "r"},
        remote_revision='"rev-1"',
        link_id=identity.link_id,
    )
    transport_calls, push_calls = await _push_due_once(
        runner, root, cm, identity, item, monkeypatch
    )
    assert transport_calls == [] and push_calls == []
    assert _item(root, "agent:a1")["push_blocked_reason"] == "unsupported-precondition"


@pytest.mark.asyncio
async def test_push_skips_when_the_public_grant_is_deleted_or_bumped(rig, monkeypatch) -> None:
    """QA trigger 3: a deleted or contract-bumped public grant is ``unsafe-publication``."""
    runner, _hub, _agents, root, cm, _ = rig
    from local_operator.hub_sync.check import ItemCheck

    identity, _record = _seed_pushable(root, cm, grant_contract=None)
    item = ItemCheck(
        "agent",
        "a1",
        "a",
        "h1",
        None,
        "available",
        local={"instructions": "l"},
        remote={"instructions": "r"},
        remote_revision='"rev-1"',
        link_id=identity.link_id,
    )
    _t, _p = await _push_due_once(runner, root, cm, identity, item, monkeypatch)
    assert _item(root, "agent:a1")["push_blocked_reason"] == "unsafe-publication"

    # Bump the grant's contract: the deliberate consent no longer covers this
    # publication contract, so it is exactly as unsafe as a missing grant.
    hub_cfg: dict[str, Any] = {
        "links": {identity.link_id: {"subscribe": True, "auto_push": True}},
        "auto_update": {"agents": True, "teams": True},
        "public_egress_grants": {
            identity.link_id: {"link_id": identity.link_id, "publication_contract": 2}
        },
    }
    cm.set_config_value("hub", hub_cfg)
    _t, _p = await _push_due_once(runner, root, cm, identity, item, monkeypatch)
    assert _item(root, "agent:a1")["push_blocked_reason"] == "unsafe-publication"


@pytest.mark.asyncio
async def test_push_phase_reaches_the_wire_when_eligible(rig, monkeypatch) -> None:
    """The control: the same fact set WITHOUT mutation publishes (push_item runs)."""
    runner, _hub, _agents, root, cm, _ = rig
    from local_operator.hub_sync.check import ItemCheck

    identity, _record = _seed_pushable(root, cm)
    item = ItemCheck(
        "agent",
        "a1",
        "a",
        "h1",
        None,
        "available",
        local={"instructions": "l"},
        remote={"instructions": "r"},
        remote_revision='"rev-1"',
        link_id=identity.link_id,
    )
    transport_calls, push_calls = await _push_due_once(
        runner, root, cm, identity, item, monkeypatch
    )
    assert len(transport_calls) == 1 and len(push_calls) == 1
    # The accepted push advances the interval and clears a stale skip mark.
    assert _item(root, "agent:a1").get("last_pushed_at") is not None
    assert "push_blocked_reason" not in _item(root, "agent:a1")


def test_push_transport_verifies_the_link_source(rig) -> None:
    """A client whose base differs from the link's recorded source is refused."""
    runner, _hub, _agents, root, cm, _ = rig
    from local_operator.hub_sync.identity import LinkIdentity

    class _Client:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def bearer_headers(self) -> dict[str, str]:
            return {"Authorization": "Bearer tok"}

    identity = LinkIdentity("agent", "a1", "https://hub.test/v1", "public", None, "h1")
    record = prov.make_record("agent", "a1", "h1", None, {"instructions": "x"}, "merge-pull")
    import dataclasses as dc

    record = dc.replace(record, identity=identity)

    def ctx_for(base: str) -> svc.HubSyncContext:
        return svc.HubSyncContext(
            config_dir=root, config_manager=cm, client_for_tenant=lambda _t: _Client(base)
        )

    transport, reason = runner._push_transport(ctx_for("https://other.test/v1"), record)
    assert transport is None and reason == "origin-conflict"
    transport, reason = runner._push_transport(ctx_for("https://hub.test/v1"), record)
    assert transport is not None and reason is None
    transport, reason = runner._push_transport(ctx_for("https://hub.test/v1/"), record)
    assert transport is not None and reason is None  # trailing slash is not a mismatch


@pytest.mark.asyncio
async def test_a_source_outage_never_burns_attempts_and_the_healthy_link_publishes(
    rig, monkeypatch
) -> None:
    """R12b: five ticks of an UNAVAILABLE check must not touch the cap.

    The unbalanced version burned five attempts on a link whose FETCH failed and
    then dead-lettered it while a healthy second link kept publishing -- i.e. the
    outage punished the wrong link and the wrong counter.
    """

    from local_operator.hub_sync import push as push_mod
    from local_operator.hub_sync.check import ItemCheck

    runner, _hub, _agents, root, cm, _ = rig
    down_identity, _r1 = _seed_pushable(root, cm, local_id="a1")
    up_identity, _r2 = _seed_pushable(root, cm, local_id="a2")
    # Both links must survive in ONE config (the second seed rewrites the hub
    # section), and both items must exist for the skip mark to be observable
    # (a real tick's check phase writes them).
    from local_operator.hub_sync.service import PUBLICATION_CONTRACT

    cm.set_config_value(
        "hub",
        {
            "links": {
                down_identity.link_id: {"subscribe": True, "auto_push": True},
                up_identity.link_id: {"subscribe": True, "auto_push": True},
            },
            "auto_update": {"agents": True, "teams": True},
            "public_egress_grants": {
                down_identity.link_id: {
                    "link_id": down_identity.link_id,
                    "publication_contract": PUBLICATION_CONTRACT,
                },
                up_identity.link_id: {
                    "link_id": up_identity.link_id,
                    "publication_contract": PUBLICATION_CONTRACT,
                },
            },
        },
    )

    def seed_items(doc: dict[str, Any]) -> None:
        items = doc.setdefault("items", {})
        items.setdefault("agent:a1", {"state": "available"})
        items.setdefault("agent:a2", {"state": "available"})

    st.StatusStore(root).mutate(seed_items)

    def item(identity, verdict, reason="hub-error"):
        return ItemCheck(
            "agent",
            identity.local_id,
            "n",
            identity.hub_id,
            None,
            verdict,
            local={"instructions": "l"},
            remote={"instructions": "r"},
            remote_revision='"rev-1"',
            link_id=identity.link_id,
            reason=reason,
        )

    calls: list[dict[str, Any]] = []

    async def fake(**kw):
        calls.append(kw)
        return push_mod.PushOutcome("pushed")

    monkeypatch.setattr(push_mod, "push_item", fake)
    transport_calls: list[Any] = []
    monkeypatch.setattr(
        runner,
        "_push_transport",
        lambda ctx, record: (transport_calls.append(record) or object(), None),
    )
    policies = {
        down_identity.link_id: LinkPolicy(True, True),
        up_identity.link_id: LinkPolicy(True, True),
    }
    for _ in range(5):
        await runner._push_due(
            await runner.context(),
            [item(down_identity, "unavailable"), item(up_identity, "available")],
            policies,
            HubSyncSettings.read_fresh(cm),
            False,
            st.StatusStore(root).load(),
            TickReport("timer"),
        )
    # The DOWN link: named skip, zero attempts, zero transport.
    assert _item(root, "agent:a1")["push_blocked_reason"] == "hub-error"
    assert [c for c in calls if c.get("local_id") == "a1"] == []
    # The HEALTHY link published (at least once -- repeats are its own pacing's
    # business, the point is it was never gated by the OTHER link's outage).
    assert len([c for c in calls if c.get("local_id") == "a2"]) >= 1
    assert "push-retries-exhausted" not in str(_item(root, "agent:a1"))

    # The source returns: the SAME (uncounted) link now publishes.
    await runner._push_due(
        await runner.context(),
        [item(down_identity, "available"), item(up_identity, "available")],
        policies,
        HubSyncSettings.read_fresh(cm),
        False,
        st.StatusStore(root).load(),
        TickReport("timer"),
    )
    assert len([c for c in calls if c.get("local_id") == "a1"]) == 1


@pytest.mark.asyncio
async def test_the_next_push_derives_its_if_match_from_the_accepted_receipt(
    rig, monkeypatch
) -> None:
    """QN2b: after ONE accepted push the next If-Match is the RECEIPT's revision.

    The check's revision is absent here (an agent check that could not read the
    detail), so the derivation must fall back to ``record.outbound.remote_revision``
    -- never the stale inbound baseline, which sent a forever-412.
    """

    import dataclasses

    from local_operator.hub_sync import push as push_mod
    from local_operator.hub_sync.check import ItemCheck
    from local_operator.hub_sync.provenance import OutboundReceipt, fingerprint_snapshot

    runner, _hub, _agents, root, cm, _ = rig

    # Seed with the accepted-receipt revision distinct from the baseline one.
    identity, _record = _seed_pushable(root, cm)
    import local_operator.hub_sync.provenance as provenance

    record = provenance.read_baseline(root, "agent", "a1")
    assert record is not None
    record = dataclasses.replace(
        record,
        remote_revision='"rev-baseline-stale"',
        outbound=OutboundReceipt(
            local_fields={"instructions": "l"},
            accepted_fields={"instructions": "a"},
            input_fingerprint=fingerprint_snapshot("agent", {"instructions": "l"}),
            remote_revision='"rev-accepted"',
            operation_id="op",
        ),
    )
    provenance.write_baseline(root, record)

    captured: list[dict[str, Any]] = []

    async def fake(**kw):
        captured.append(kw)
        return push_mod.PushOutcome("pushed")

    monkeypatch.setattr(push_mod, "push_item", fake)
    monkeypatch.setattr(runner, "_push_transport", lambda ctx, record: (object(), None))
    item = ItemCheck(
        "agent",
        "a1",
        "n",
        "h1",
        None,
        "available",
        local={"instructions": "l"},
        remote={"instructions": "r"},
        remote_revision=None,  # the fetch exposed none
        link_id=identity.link_id,
    )
    await runner._push_due(
        await runner.context(),
        [item],
        {identity.link_id: LinkPolicy(True, True)},
        HubSyncSettings.read_fresh(cm),
        False,
        st.StatusStore(root).load(),
        TickReport("timer"),
    )
    assert captured and captured[0]["expected_revision"] == '"rev-accepted"'


@pytest.mark.asyncio
async def test_end_to_end_auto_push_against_the_decoder_faithful_hub(rig, monkeypatch) -> None:
    """The round-2 acceptance row: seeded receipt, edit, REAL commit, second push.

    A served core (the runner) against a hub fake built from the server's own
    decode rules: the commit must be the BARE document with the previous
    revision's If-Match; the receipt and revision advance; a SECOND edit then
    pushes with the NEW If-Match (no forever-412); and a hub-side change merges
    without clobbering the local edit.
    """

    import dataclasses

    from local_operator.hub_sync import provenance as prov_mod
    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkIdentity
    from local_operator.hub_sync.push import receipt_for
    from tests.unit.hub_sync.fake_hub import FakeHub

    runner, hub, agents, root, cm, _ = rig
    fake = FakeHub(document={"instructions": BASE, "description": "d"})
    fake.install(monkeypatch)

    row = _pull_agent(agents, hub, root)  # local == hub listing, L == R exactly
    source = "https://hub.test/v1"
    link_id = LinkIdentity("agent", row.id, source, "public", None, "h1").link_id
    fake.revision = '"rev-1"'

    # The LINK's baseline: identity + the check-visible revision + a SEEDED receipt
    # (A = the local text, P the same remote text, revision the server's ETag).
    record = prov_mod.read_baseline(root, "agent", row.id)
    assert record is not None
    identity = LinkIdentity("agent", row.id, source, "public", None, "h1")
    record = dataclasses.replace(
        record,
        identity=identity,
        remote_revision='"rev-1"',
        lineage="clean",
        scope_known=True,
        outbound=receipt_for(
            "agent",
            {"instructions": BASE, "description": "d"},
            {"instructions": BASE, "description": "d"},
            revision='"rev-1"',
        ),
    )
    prov_mod.write_baseline(root, record)
    svc.record_server_capabilities(await runner.context(), {"revision_cas": 1}, source=source)
    from local_operator.hub_sync.service import PUBLICATION_CONTRACT

    cm.set_config_value(
        "hub",
        {
            "links": {link_id: {"subscribe": True, "auto_push": True}},
            "auto_update": {"agents": True, "teams": True},
            # Both pushes in this test are due immediately: the interval is the
            # operator's pacing knob, not part of what the test proves.
            "check_interval_min": 0,
            "public_egress_grants": {
                link_id: {"link_id": link_id, "publication_contract": PUBLICATION_CONTRACT}
            },
        },
    )
    # The check's DETAIL read mirrors the LIVE fake revision, so the second push
    # derives the new If-Match from the check (the receipt is the fallback).
    hub.get_agent = lambda hub_id, **_kw: {
        "instructions": hub.agents[hub_id][0],
        "description": hub.agents[hub_id][1],
        "revision": fake.revision,
        "sync_contract": {"revision_cas": 1},
    }

    # The first tick after an upgrade is the grace tick and never publishes.
    await runner.tick(reason="startup")

    def elapse_interval() -> None:
        # The check/push intervals are the operator's pacing knobs (clamped to
        # >= 1 min); the test elapses them by backdating the stamps instead of
        # sleeping, touching BOTH directions so neither cadence blocks the step.
        key = f"agent:{row.id}"

        def fold(doc: dict[str, Any]) -> None:
            item = doc.get("items", {}).get(key)
            if item is not None:
                item["last_pushed_at"] = "2000-01-01T00:00:00Z"
                item["last_pulled_at"] = "2000-01-01T00:00:00Z"

        st.StatusStore(root).mutate(fold)

    elapse_interval()

    # 1) a local edit, auto_push ON, one tick: the BARE document is committed.
    agents.set_agent_system_prompt(row.id, BASE + "\n\nLOCAL-ONE\n")
    tick = await runner.tick(reason="timer")
    assert tick.failed == 0
    assert len(fake.commit_calls) == 1, tick
    body = fake.bare_commit_bodies()[0]
    assert "preview_version" not in body and "document" not in body
    assert "LOCAL-ONE" in body["instructions"] and body["description"] == "d"
    assert fake.commit_calls[0].headers["If-Match"] == '"rev-1"'
    stored = prov_mod.read_baseline(root, "agent", row.id)
    assert stored is not None
    receipt = stored.outbound
    assert receipt is not None and receipt.remote_revision == fake.revision
    first_revision = fake.revision
    # The remote now HOLDS what the push accepted: keep the check's fetch in sync
    # with the accepted document (a stale remote would look like a concurrent
    # deletion of the text the push just added).
    hub.agents["h1"] = (fake.document["instructions"], fake.document["description"])

    # 2) a SECOND local edit: the push uses the revision the first push minted.
    agents.set_agent_system_prompt(row.id, BASE + "\n\nLOCAL-ONE\n\nLOCAL-TWO\n")
    elapse_interval()
    await runner.tick(reason="timer")
    assert len(fake.commit_calls) == 2
    assert fake.commit_calls[1].headers["If-Match"] == first_revision
    assert "LOCAL-TWO" in fake.bare_commit_bodies()[1]["instructions"]

    # 3) a hub-side change with the pushed text as its base merges without clobbering.
    hub.agents["h1"] = (
        fake.document["instructions"] + "\n\nHUB-SIDE\n",
        fake.document["description"],
    )
    elapse_interval()
    report = await runner.tick(reason="timer")
    merged = agents.get_agent_system_prompt(row.id)
    assert "HUB-SIDE" in merged and "LOCAL-" in merged
    assert report.failed == 0


@pytest.mark.asyncio
async def test_a_resolved_link_publishes_after_a_local_edit_from_the_seeded_receipt(
    rig, monkeypatch
) -> None:
    """QN7 end to end on the REAL entry points: pull -> resolve_source -> check
    -> up-to-date apply (seed + clean lineage + revision) -> local edit -> tick
    publishes. Nothing injects identity, lineage or a receipt."""

    from local_operator.hub_sync import provenance as prov_mod
    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.service import PUBLICATION_CONTRACT
    from tests.unit.hub_sync.fake_hub import FakeHub

    runner, hub, agents, root, cm, _ = rig
    fake = FakeHub(document={"instructions": BASE, "description": "d"})
    fake.revision = '"hub-rev-1"'  # the token the seeded receipt will condition on
    fake.install(monkeypatch)

    row = _pull_agent(agents, hub, root)
    ctx = await runner.context()
    resolved = svc.resolve_source(
        ctx,
        kind="agent",
        name=row.name,
        hub_source="https://hub.test/v1",
        visibility="public",
        tenant_id=None,
    )
    assert resolved.lineage == "unknown"
    link_id = resolved.identity.link_id if resolved.identity is not None else ""
    cm.set_config_value(
        "hub",
        {
            "links": {link_id: {"subscribe": True, "auto_push": True}},
            "auto_update": {"agents": True, "teams": True, "skills": True},
            "check_interval_min": 0,
            "public_egress_grants": {
                link_id: {"link_id": link_id, "publication_contract": PUBLICATION_CONTRACT}
            },
        },
    )

    await runner.tick(reason="startup")  # grace: checks record the attestation only
    await runner.tick(reason="timer")  # up-to-date apply: seeds, cleans, no paid call

    assert fake.commit_calls == [], "an agreed no-op must settle before any paid call"
    stored = prov_mod.read_baseline(root, "agent", row.id)
    assert stored is not None
    assert stored.lineage == "clean"
    assert stored.outbound is not None
    assert stored.outbound.remote_revision == '"hub-rev-1"'

    agents.set_agent_system_prompt(row.id, BASE + "\n\nLOCAL-ADDITION")
    await runner.tick(reason="timer")

    assert fake.commit_calls, "the seeded receipt must carry the local edit to the hub"
    body = fake.bare_commit_bodies()[-1]
    assert body["instructions"].endswith("LOCAL-ADDITION")
    assert '"hub-rev-1"' == fake.commit_calls[0].headers.get("If-Match")
    after = prov_mod.read_baseline(root, "agent", row.id)
    assert after is not None and after.outbound is not None
    assert after.outbound.remote_revision == fake.revision
