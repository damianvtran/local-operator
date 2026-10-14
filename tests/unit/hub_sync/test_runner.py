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

pytestmark = pytest.mark.asyncio


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


async def test_first_tick_after_upgrade_checks_but_never_applies(rig) -> None:
    runner, hub, agents, root, cm, _ = rig
    row = _drift(agents, hub, root)
    first = await runner.tick(reason="startup")
    assert first.available == 1 and first.applied == 0  # first_run_grace
    assert "Added upstream" not in agents.get_agent_system_prompt(row.id)
    second = await runner.tick(reason="timer")
    assert second.applied == 1 and "Added upstream" in agents.get_agent_system_prompt(row.id)


async def test_manual_mode_keeps_checking_and_records_available_without_writing(rig) -> None:
    runner, hub, agents, root, cm, _ = rig
    row = _drift(agents, hub, root)
    cm.set_config_value("hub", {"auto_update": {"agents": False, "teams": False}})
    await runner.tick(reason="startup")
    report = await runner.tick(reason="timer")
    assert report.available == 1 and report.applied == 0
    assert "Added upstream" not in agents.get_agent_system_prompt(row.id)
    assert st.StatusStore(root).load()["items"][f"agent:{row.id}"]["state"] == "available"


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


async def test_no_linked_items_costs_nothing(rig) -> None:
    runner, *_ = rig
    report = await runner.tick(reason="startup")
    assert report.skipped == "no-linked-items" and report.checked == 0


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


async def test_ticks_are_single_flight(rig) -> None:
    runner, hub, agents, root, *_ = rig
    _drift(agents, hub, root)
    a, b = await asyncio.gather(runner.tick(reason="timer"), runner.tick(reason="timer"))
    assert sorted([a.applied, b.applied]) in ([0, 0], [0, 1])  # never both


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


async def test_pick_never_mixes_agent_and_team_ids(rig) -> None:
    """A bare-id set would let an agent id select a same-string team id."""
    runner, hub, agents, root, *_ = rig
    from local_operator.hub_sync.check import Link

    links = [Link("agent", "shared", "a", "h", None), Link("team", "shared", "t", "h", None)]
    picked = runner._pick(links, {"items": {}}, {})
    assert picked["agent"] == {"shared"} and picked["team"] == {"shared"}
    # Both kinds are addressed independently; the runner hands each its own set.
    assert set(picked) == {"agent", "team", "skill"}


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
    assert _item(root, "agent:a1")["push_blocked_reason"] == "no-outbound-receipt"


def _seed_pushable(
    root: Path,
    cm: ConfigManager,
    *,
    cas: bool = True,
    grant_contract: int | None = 1,
    receipt: bool = True,
    source: str = "https://hub.test/v1",
):
    """A public agent link with identity, revision, policy, grant and (A,P) receipt.

    This is the exact fact set ``_eligibility`` reads; each test then mutates ONE
    of them to reach a distinct refusal (the three QA triggers plus the control).
    """

    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkIdentity
    from local_operator.hub_sync.provenance import OutboundReceipt, fingerprint_snapshot

    identity = LinkIdentity("agent", "a1", source, "public", None, "h1")
    record = prov.make_record("agent", "a1", "h1", None, {"instructions": "x"}, "merge-pull")
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
