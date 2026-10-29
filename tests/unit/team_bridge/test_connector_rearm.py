"""D2b: the running connector re-arms on an owner change, in-process.

Diagnosis (Team Bridge post-0.90.17, D2 "second, real re-arm bug"): the entry
point resolved the committed owner ONCE and ``run_forever`` looped on that
snapshot for ever. After a re-enrolment (a generation bump, a new endpoint, a
credential rebind) every iteration failed ``pull._guard_owner`` with
``owner_mismatch`` and backed off to 60 s — for ever, because the loop never
exits and so launchd's KeepAlive never restarted it onto the new owner. And an
owner-less start exited 0, which under ``KeepAlive: true`` relaunches the whole
interpreter about every 10 s.

Exiting on a refusal is NOT the fix (review R4): ``owner_mismatch`` is also the
answer when the signed-in ACCOUNT moved against an unchanged owner, and a
restart re-reads the same owner and refuses again — a respawn storm. So the
connector stays resident, idles cheaply, and re-reads the store when its files
change (or at a bounded interval), returning to ``__main__`` only when the
committed owner actually differs.

Time is SIMULATED in the busy-loop tests: ``loop._pause`` is replaced by a
clock that advances by the requested sleep, so an hour of idling runs in
milliseconds and the call counts are exact rather than timing-dependent.
"""

from __future__ import annotations

import asyncio
import logging
import plistlib
from pathlib import Path
from typing import Any

import pytest

from local_operator.providers.auth_store import OAuthAccess
from local_operator.team_bridge.connector import install as install_mod
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.pull import PullRefused
from local_operator.team_bridge.store import OwnerReceipt, Store
from local_operator.team_bridge.types import OwnerSnapshot

HUB = "https://hub.example/v1"

OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url=HUB,
    enrollment_generation=1,
)

ACCESS = OAuthAccess(access_token="x", credential_id=1, account_id="acc_1", kind="oauth")


def _commit(config_dir: Path) -> Store:
    store = Store(config_dir)
    store.commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v1", "bep_1"), cos_session_id="s")
    return store


class _SimulatedClock:
    """Replaces ``loop._pause``: every sleep advances virtual time instantly.

    ``at`` schedules a side effect (a re-enrolment) once virtual time passes a
    mark; ``horizon`` sets ``stop`` so a loop that never returns still ends.
    """

    def __init__(self, stop: asyncio.Event | None, horizon: float) -> None:
        self.now = 0.0
        self.pauses = 0
        self._stop = stop
        self._horizon = horizon
        self._scheduled: list[tuple[float, Any]] = []

    def at(self, when: float, action: Any) -> None:
        self._scheduled.append((when, action))

    async def pause(self, stop: asyncio.Event | None, seconds: float) -> None:
        self.pauses += 1
        self.now += max(seconds, 0.0)
        for item in list(self._scheduled):
            if self.now >= item[0]:
                self._scheduled.remove(item)
                item[1]()
        if self.now >= self._horizon and self._stop is not None:
            self._stop.set()
        # Yield like the real pause, so a loop that never awaits elsewhere
        # cannot starve the event loop in the test either.
        await asyncio.sleep(0)


def _count_signal_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    reads: list[int] = []
    real = loop_mod._owner_signal

    def counted(config_dir: Path) -> Any:
        reads.append(1)
        return real(config_dir)

    monkeypatch.setattr(loop_mod, "_owner_signal", counted)
    return reads


def test_a_running_connector_rearms_after_its_enrolment_changes(tmp_path: Path) -> None:
    """Ported repro (``test_d2_a_running_connector_rearms_after_its_enrolment_changes``,
    FAILS on c69e682eb71 with "stale-owner connector still looping after 3 s").
    The live enrolment is generation 2 while the loop holds generation 1: the
    loop must hand the NEW owner back instead of refusing for ever."""

    store = _commit(tmp_path)
    store.invalidate(OWNER)  # the live enrolment is now generation 2
    stop = asyncio.Event()

    async def provider() -> OAuthAccess:
        return ACCESS

    async def drive() -> Any:
        task = asyncio.create_task(
            loop_mod.run_forever(
                OWNER, config_dir=tmp_path, auth_store=None, access_provider=provider, stop=stop
            )
        )
        done, _ = await asyncio.wait({task}, timeout=3.0)
        stop.set()
        await asyncio.wait_for(task, 70)
        assert done, "stale-owner connector still looping after 3 s"
        return task.result()

    rearmed = asyncio.run(drive())
    assert rearmed is not None and rearmed.enrollment_generation == 2
    assert rearmed.endpoint_id == OWNER.endpoint_id


def test_a_refusal_never_exits_and_never_busy_loops_then_picks_up_reenrolment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An identity refusal against an UNCHANGED owner (the signed-in account
    moved) must keep the connector resident: no return, no exit, one log line
    per state change, and at most one hub attempt per ``BACKOFF_MAX_S`` of
    simulated time. When the store then changes (re-enrolment), the file
    signal wakes the idle within one signal poll — no restart, no 60 s wait."""

    store = _commit(tmp_path)
    stop = asyncio.Event()
    clock = _SimulatedClock(stop, horizon=7200.0)
    monkeypatch.setattr(loop_mod, "_pause", clock.pause)
    signal_reads = _count_signal_reads(monkeypatch)
    attempts: list[float] = []

    async def provider() -> OAuthAccess:
        attempts.append(clock.now)
        return ACCESS

    async def refused(owner: Any, **kwargs: Any) -> dict[str, Any]:
        raise PullRefused("owner_mismatch", "pinned access account differs from owner")

    monkeypatch.setattr(loop_mod, "run_iteration", refused)
    reenrolled_at = 3600.0
    clock.at(reenrolled_at, lambda: store.invalidate(OWNER))

    async def drive() -> Any:
        return await asyncio.wait_for(
            loop_mod.run_forever(
                OWNER, config_dir=tmp_path, auth_store=None, access_provider=provider, stop=stop
            ),
            timeout=30,
        )

    with caplog.at_level(logging.INFO, logger="local_operator.team_bridge"):
        rearmed = asyncio.run(drive())

    assert not stop.is_set(), "the loop ran to the horizon: the re-enrolment was never picked up"
    assert rearmed is not None and rearmed.enrollment_generation == 2
    # Picked up by the file signal, not by the next bounded re-check.
    assert clock.now - reenrolled_at <= loop_mod.OWNER_SIGNAL_POLL_S
    # No busy loop: an identity refusal waits the full bounded interval at
    # once (no 1-2-4-8 ramp of hub attempts), so one attempt per interval.
    assert len(attempts) <= reenrolled_at / loop_mod.BACKOFF_MAX_S + 1, len(attempts)
    gaps = [b - a for a, b in zip(attempts, attempts[1:])]
    assert gaps and min(gaps) >= loop_mod.BACKOFF_MAX_S, gaps
    # The idle watch costs a stat per poll, nothing more.
    assert len(signal_reads) <= clock.now / loop_mod.OWNER_SIGNAL_POLL_S + len(attempts) + 2
    # One content-free line per STATE CHANGE, not one per tick.
    refusal_lines = [r for r in caplog.records if "code=owner_mismatch" in r.getMessage()]
    assert len(refusal_lines) == 1, [r.getMessage() for r in refusal_lines]
    assert any("stage=rearm" in r.getMessage() for r in caplog.records)


def test_an_unenrolled_running_connector_idles_and_resumes_the_same_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner disappears mid-run (no committed row): the loop stays
    resident and makes NO hub attempts while idle; when the same owner is
    committed again it resumes under its snapshot rather than returning."""

    store = _commit(tmp_path)
    stop = asyncio.Event()
    clock = _SimulatedClock(stop, horizon=3600.0)
    monkeypatch.setattr(loop_mod, "_pause", clock.pause)
    present = {"owner": False}
    attempts: list[float] = []

    def owner_source() -> Any:
        return store.active_owner() if present["owner"] else None

    async def provider() -> OAuthAccess:
        attempts.append(clock.now)
        if len(attempts) >= 2:
            stop.set()
        return ACCESS

    async def answered(owner: Any, **kwargs: Any) -> dict[str, Any]:
        return {"events": 0, "reissued": True}

    monkeypatch.setattr(loop_mod, "run_iteration", answered)
    clock.at(1800.0, lambda: present.update(owner=True))

    async def drive() -> Any:
        return await asyncio.wait_for(
            loop_mod.run_forever(
                OWNER,
                config_dir=tmp_path,
                auth_store=None,
                access_provider=provider,
                stop=stop,
                owner_source=owner_source,
            ),
            timeout=30,
        )

    assert asyncio.run(drive()) is None  # stopped, never re-armed: the owner is unchanged
    assert attempts and attempts[0] >= 1800.0, attempts  # no hub attempt while unenrolled


def test_a_no_owner_start_waits_cheaply_then_proceeds_once_enrolled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An owner-less start used to print and exit 0 — under KeepAlive a
    relaunch of the whole interpreter about every 10 s. It must WAIT
    in-process instead, re-reading the store only on a file change or at the
    bounded interval, and arm the loop once an enrolment is committed."""

    from local_operator.team_bridge.connector import __main__ as main_mod

    clock = _SimulatedClock(None, horizon=float("inf"))
    monkeypatch.setattr(loop_mod, "_pause", clock.pause)
    signal_reads = _count_signal_reads(monkeypatch)
    owner_reads: list[float] = []
    real_active_owner = Store.active_owner

    def counted_active_owner(self: Store, **kwargs: Any) -> Any:
        owner_reads.append(clock.now)
        return real_active_owner(self, **kwargs)

    monkeypatch.setattr(Store, "active_owner", counted_active_owner)
    enrolled_at = 3600.0
    clock.at(enrolled_at, lambda: _commit(tmp_path))
    armed: list[tuple[float, Any]] = []

    async def fake_run_forever(owner: Any, **kwargs: Any) -> None:
        armed.append((clock.now, owner))
        return None  # stopped: main returns

    monkeypatch.setattr(loop_mod, "run_forever", fake_run_forever)
    with caplog.at_level(logging.INFO, logger="local_operator.team_bridge"):
        assert main_mod.main(["--config-dir", str(tmp_path)]) == 0

    assert len(armed) == 1, armed
    when, owner = armed[0]
    assert owner == OWNER
    assert when - enrolled_at <= loop_mod.OWNER_SIGNAL_POLL_S
    # Cheap: SQLite is opened only at the bounded re-check (plus once per
    # file change), the idle itself is a stat per poll.
    assert len(owner_reads) <= enrolled_at / loop_mod.OWNER_RECHECK_S + 3, len(owner_reads)
    # One stat per poll plus the baseline each bounded window takes.
    windows = enrolled_at / loop_mod.OWNER_RECHECK_S
    assert len(signal_reads) <= enrolled_at / loop_mod.OWNER_SIGNAL_POLL_S + windows + 3
    idle_lines = [r for r in caplog.records if "code=no_owner" in r.getMessage()]
    assert len(idle_lines) == 1, [r.getMessage() for r in idle_lines]


def test_main_rearms_in_process_on_the_owner_the_loop_hands_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop returns the NEW owner; ``__main__`` re-arms on it under the
    SAME lock, with a client pinned to the new credential — no restart."""

    from local_operator.team_bridge import client as client_mod
    from local_operator.team_bridge.connector import __main__ as main_mod
    from local_operator.team_bridge.connector import lock as lock_mod

    _commit(tmp_path)
    moved = OwnerSnapshot(**{**OWNER.__dict__, "credential_id": 2, "enrollment_generation": 2})
    script = [moved, None]
    armed: list[Any] = []
    pinned: list[Any] = []
    acquired: list[int] = []

    async def fake_run_forever(owner: Any, **kwargs: Any) -> Any:
        armed.append(owner)
        return script.pop(0)

    real_init = client_mod.TeamBridgeClient.__init__

    def recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
        pinned.append(kwargs.get("credential_id"))
        real_init(self, *args, **kwargs)

    real_acquire = lock_mod.ConnectorLock.acquire

    def recording_acquire(self: Any) -> None:
        acquired.append(1)
        real_acquire(self)

    monkeypatch.setattr(loop_mod, "run_forever", fake_run_forever)
    monkeypatch.setattr(client_mod.TeamBridgeClient, "__init__", recording_init)
    monkeypatch.setattr(lock_mod.ConnectorLock, "acquire", recording_acquire)
    assert main_mod.main(["--config-dir", str(tmp_path)]) == 0
    assert armed == [OWNER, moved]
    assert pinned == [1, 2]
    assert acquired == [1]


def test_the_rendered_plist_throttles_relaunches(tmp_path: Path) -> None:
    """Belt and braces for anything that still exits (a lock loser, a crash):
    launchd's default 10 s throttle is a relaunch storm for an always-on job."""

    path = install_mod.write_plist(tmp_path / "cfg", tmp_path / "rendered")
    plan = plistlib.loads(path.read_bytes())
    assert plan["KeepAlive"] is True
    assert plan["ThrottleInterval"] >= loop_mod.BACKOFF_MAX_S
