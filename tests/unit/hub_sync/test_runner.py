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
from local_operator.hub_sync.runner import HubSyncRunner
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
    picked = runner._pick(links, doc)
    assert len(picked) == 50 and "id0" in picked and "id59" not in picked


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
