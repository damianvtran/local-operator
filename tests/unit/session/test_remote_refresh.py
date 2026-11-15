"""A ``retiring`` owner is a refresh, not a death and not a stop.

The runtime retires itself when ``lop-update`` puts a newer build on disk
and it is idle (design-runtime-autorefresh §3.2). The viewer must read the
announced disconnect as "a fresh runtime is owed" — go cold at once, fire
the refresh callback so the app re-engages, and synthesise NO turn end
(nothing was interrupted; the runtime was idle by contract). Both older
outcomes keep their behaviour: ``STOPPED_REASON`` still parks the viewer in
the stopped state, and a bare owner exit still runs recovery.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import local_operator.session.attached as remote_module
from local_operator.incidents import format_cut_off_notice
from local_operator.mobile.attach_client import (
    MOVED_REASON,
    RETIRING_REASON,
    STOPPED_REASON,
)
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.session.placement import SessionPlacement


async def _never_take_over() -> Any:
    raise AssertionError("a refresh must never take over the session")


@pytest.mark.asyncio
async def test_retiring_goes_cold_now_and_fires_the_refresh_callback(tmp_path, monkeypatch):
    """The whole contract of the frame, on the viewer side.

    No recovery task: chasing the record for 8 s would end in "runtime
    exited" for a planned housekeeping event, and the successor does not
    exist yet — the app spawns it. No ``_end_turn_locally``: that is what
    would paint ``interrupted`` for a turn that never existed.
    """
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    ended: list[str] = []
    monkeypatch.setattr(
        remote, "_end_turn_locally", lambda *a, **k: ended.append("end"), raising=True
    )
    refreshed: list[str] = []
    cold: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))
    remote.set_went_cold_callback(lambda: cold.append("cold"))

    remote._on_disconnected(RETIRING_REASON)
    await asyncio.sleep(0)

    assert remote.is_cold
    assert remote._recovery_task is None, "a refresh has nothing to recover"
    assert refreshed == ["refresh"]
    assert cold == [], "the went-cold callback is the death path; a refresh is not one"
    assert ended == [], "no turn end may be synthesised for an idle runtime that retired"
    assert remote._deliberate_stop is False, "a refresh is not a stop; the next prompt engages"
    assert remote._runtime_ready.is_set(), "a cold viewer is READY: the next prompt engages"


@pytest.mark.asyncio
async def test_a_retire_while_a_turn_looked_live_drops_the_claim_without_a_verdict(
    tmp_path, monkeypatch
):
    """THE INCIDENT's repair: the claim must not outlive its own owner.

    Session 62480a153ced served ``frontend.streaming: true`` for 6+ minutes
    while idle: its runtime retired during a daemon-wide update, the retirement
    raced the turn-end delta, and the documented repair — the successor's
    snapshot — never arrived because a daemon being replaced declines to engage
    a successor. Nothing else could clear the claim, and viewers decided
    steer-vs-prompt and painted "thinking" off it (two operator presses went
    into the steer branch of a turn that had ended).

    So the facade stops claiming liveness by itself, and the canonical mirror
    loses the claim with it — WITHOUT a synthesised turn end: every shape
    ``_end_turn_locally`` can deliver is a verdict this seat cannot make for a
    turn that may still be running (the successor's snapshot, when one comes,
    re-marks whatever is actually live).
    """
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    remote._streaming = True
    remote._generation = 4
    remote._frontend_store = FrontendStateStore(
        FrontendSessionState(session_id="s1", epoch="e1", sequence=7, streaming=True)
    )
    ended: list[str] = []
    monkeypatch.setattr(
        remote, "_end_turn_locally", lambda *a, **k: ended.append("end"), raising=True
    )
    refreshed: list[str] = []
    cold: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))
    remote.set_went_cold_callback(lambda: cold.append("cold"))

    remote._on_disconnected(RETIRING_REASON)
    await asyncio.sleep(0)

    assert remote.is_streaming is False, "the facade kept claiming a live turn"
    assert remote.frontend_state.streaming is False, "the canonical mirror kept claiming it"
    assert remote.frontend_state.sequence == 7, (
        "the owner's clock must be preserved: a viewer-local mutate would consume "
        "the owner's next sequence number and break synchronization"
    )
    assert ended == [], (
        "no turn end may be synthesised for a turn the successor may still be owed: "
        "every shape of one is a verdict this seat cannot make"
    )
    assert refreshed == ["refresh"], "the app must still be told to re-engage"
    assert cold == [], "the went-cold callback is the death path; a refresh is not one"
    assert remote.is_cold
    assert remote._recovery_task is None, "a refresh has nothing to recover"


@pytest.mark.asyncio
async def test_a_retire_with_no_live_turn_leaves_canonical_state_alone(tmp_path, monkeypatch):
    """Regression guard: the drop is for a CLAIM, and an idle retire has none.

    The ordinary refresh must be byte-identical to what it always was — same
    store object, same sequence, no rewrite — which is what keeps this repair
    scoped to the anomalous arm instead of touching every ``lop-update``
    rollover.
    """
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    remote._frontend_store = FrontendStateStore(
        FrontendSessionState(session_id="s1", epoch="e1", sequence=7, streaming=False)
    )
    store = remote._frontend_store
    installed = store._state
    refreshed: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))

    remote._on_disconnected(RETIRING_REASON)
    await asyncio.sleep(0)

    assert remote._frontend_store is store
    assert store._state is installed, "an idle retire rewrote canonical state"
    assert store.state.sequence == 7
    assert refreshed == ["refresh"]


@pytest.mark.asyncio
async def test_a_move_still_ends_a_live_looking_turn_with_its_cut_off(tmp_path, monkeypatch):
    """Regression guard: the refresh drop must not leak into the MOVE arm.

    ``MOVED_REASON`` → ``_go_cold(refresh=False)`` is the death-shaped arm: the
    conversation is leaving (another device, or the viewer's own directory
    change), and a turn that looked live is closed with the owner-lost cut-off
    the departing owner will not come back to finish. That is deliberately NOT
    the refresh treatment — no drop-only repair — and this pin exists so a
    later edit cannot quietly route the two through one path.
    """
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    remote._streaming = True
    remote._frontend_store = FrontendStateStore(
        FrontendSessionState(session_id="s1", epoch="e1", streaming=True)
    )
    ends: list[Any] = []
    remote.subscribe(ends.append)
    refreshed: list[str] = []
    cold: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))
    remote.set_went_cold_callback(lambda: cold.append("cold"))

    remote._on_disconnected(MOVED_REASON)
    await asyncio.sleep(0)

    assert remote.is_streaming is False
    assert [event.type for event in ends] == ["agent_end"], ends
    assert ends[0].aborted is False
    assert ends[0].error == format_cut_off_notice("owner-lost")
    assert refreshed == [], "a move must not re-engage locally (INV-1)"
    assert cold == ["cold"], "a move goes cold through the went-cold callback"
    assert remote.moved_to == ""


@pytest.mark.asyncio
async def test_stopped_reason_still_parks_the_viewer(tmp_path, monkeypatch):
    """Regression guard: the new branch must not shadow the stop path."""
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    refreshed: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))

    remote._on_disconnected(STOPPED_REASON)
    if remote._recovery_task is not None:
        await asyncio.wait_for(remote._recovery_task, timeout=5)

    assert remote._deliberate_stop is True
    assert refreshed == []


@pytest.mark.asyncio
async def test_owner_death_still_recovers(tmp_path, monkeypatch):
    """Regression guard: an unannounced exit still runs ``_recover_runtime``."""
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    refreshed: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))

    remote._on_disconnected("owner exited")
    task = remote._recovery_task
    assert task is not None, "an unannounced exit must still start recovery"
    assert remote._recovering is True
    assert refreshed == []
    # The recovery loop's own outcomes (takeover, cold after 8 s) are covered
    # by test_remote_takeover; here only its START is the property.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_request_refresh_never_raises(tmp_path, monkeypatch):
    """Every failure is "kept": the reaper's own check is the fallback."""
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    assert (await remote.request_refresh()).startswith("kept:")

    class OldOwner:
        connected = True

        async def request_refresh(self) -> str:
            raise RuntimeError("unknown op: refresh_if_idle")

    remote._client = OldOwner()  # type: ignore[assignment]
    assert (await remote.request_refresh()).startswith("kept:")

    class Owner:
        connected = True

        async def request_refresh(self) -> str:
            return "retiring"

    remote._client = Owner()  # type: ignore[assignment]
    assert await remote.request_refresh() == "retiring"


def test_runtime_idle_reads_the_snapshot(tmp_path, monkeypatch):
    """Cold → not idle (nothing to refresh); streaming, a running job or a
    parked gate → busy; otherwise idle."""
    from types import SimpleNamespace

    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    assert remote.runtime_idle() is False, "a cold viewer has no owner to refresh"

    class Client:
        connected = True

    remote._client = Client()  # type: ignore[assignment]
    remote._ready_for_events = True
    assert not remote.is_cold

    def store(**fields: Any) -> Any:
        base = {"streaming": False, "pending_gate": None, "jobs": []}
        base.update(fields)
        return SimpleNamespace(state=SimpleNamespace(**base))

    def with_state(**fields: Any) -> None:
        remote._frontend_store = store(**fields)  # type: ignore[assignment]

    with_state()
    assert remote.runtime_idle() is True
    with_state(streaming=True)
    assert remote.runtime_idle() is False
    with_state(pending_gate=object())
    assert remote.runtime_idle() is False
    with_state(jobs=[SimpleNamespace(status="running")])
    assert remote.runtime_idle() is False
    with_state(jobs=[SimpleNamespace(status="completed")])
    assert remote.runtime_idle() is True
    remote._streaming = True
    assert remote.runtime_idle() is False


# -- a lost ``retiring`` frame (CI gate 42969069065) --------------------------
#
# The owner retired cleanly (exit 0, ``runtime-exit.json`` says "retiring for
# <build>"), but a reset overtook the frame, so the viewer heard
# ``'owner connection reset'``. It recovered, found no record, and went cold with
# refresh=False. Nothing re-engaged. The durable exit record is the evidence
# that survives a lost frame.


def _lost_run(pid: int = 4242, started_at: float = 1_000.0) -> Any:
    from local_operator.mobile.types import SessionRecord

    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id="s1",
        conversation_name="",
        cwd="/",
        model_label="",
        control_port=1,
        control_key="k",
        started_at=started_at,
    )


def _write_exit(
    tmp_path,
    *,
    reason: str,
    pid: int = 4242,
    requested: bool = False,
    started_at: float = 1_000.0,
) -> None:
    from local_operator.session.runtime import journal

    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True, exist_ok=True)
    assert journal.write_exit_record(
        "s1",
        conversation_dir=directory,
        pid=pid,
        started_at=started_at,
        reason=reason,
        requested=requested,
    )


class _PeerOwner:
    """A non-local owner seam: its exit record is not on this machine."""

    placement = SessionPlacement(mode="peer", network_id="n_1", home_device="dev_x")

    def locate(self) -> tuple[None, None]:
        return (None, None)

    async def engage(self, **kwargs: Any) -> None:  # pragma: no cover
        return None

    def make_client(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("not dialled in this test")


def _reset_viewer(
    tmp_path, monkeypatch, *, owner: Any = None, lost: Any = "default", cold_s: float = 1.0
) -> tuple[AttachedSession, list[str], list[str]]:
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    # A short cold bound keeps the crash control fast. The fix acts on the
    # FIRST pass, well inside any bound.
    monkeypatch.setattr(remote_module, "COLD_FALLBACK_S", cold_s)
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over, owner=owner
    )
    remote._runtime_record = _lost_run() if lost == "default" else lost
    refreshed: list[str] = []
    cold: list[str] = []
    remote.set_refresh_callback(lambda: refreshed.append("refresh"))
    remote.set_went_cold_callback(lambda: cold.append("cold"))
    return remote, refreshed, cold


@pytest.mark.asyncio
async def test_a_reset_that_lost_the_retiring_frame_still_refreshes(tmp_path, monkeypatch):
    """The witness's sequence: a reset in place of the frame, and a retiring record on disk."""
    _write_exit(tmp_path, reason="retiring for 0.101.44@f4a70b9")
    remote, refreshed, cold = _reset_viewer(tmp_path, monkeypatch)

    remote._on_disconnected("owner connection reset")
    task = remote._recovery_task
    assert task is not None
    await asyncio.wait_for(task, timeout=5)

    assert refreshed == ["refresh"], "a lost frame must still refresh"
    assert cold == []
    assert remote.is_cold
    assert remote._recovering is False


@pytest.mark.asyncio
async def test_a_retiring_record_written_after_the_reset_still_refreshes(tmp_path, monkeypatch):
    """The owner writes its record on the way out, which can be after the socket dropped."""
    remote, refreshed, cold = _reset_viewer(tmp_path, monkeypatch)

    remote._on_disconnected("owner connection reset")
    task = remote._recovery_task
    assert task is not None
    await asyncio.sleep(0.3)
    assert refreshed == [] and not task.done()
    _write_exit(tmp_path, reason="retiring for 0.101.44@f4a70b9")
    await asyncio.wait_for(task, timeout=5)

    assert refreshed == ["refresh"]
    assert cold == []


_RETIRING = "retiring for 0.101.44@f4a70b9"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_record", "viewer"),
    [
        (None, {}),  # a genuine crash: no record at all
        ({"reason": _RETIRING, "pid": 9999}, {}),  # a different run
        ({"reason": _RETIRING, "requested": True}, {}),  # a stop
        ({"reason": "leaving after SIGTERM"}, {}),  # not a build refresh
        # The three guards round 1 of #3797's review found untested (m3). Each
        # writes a GENUINE retiring record, so only the guard under test stands
        # between it and a silent refresh.
        ({"reason": _RETIRING, "started_at": 2_000.0}, {}),  # same pid, a later run
        ({"reason": _RETIRING}, {"owner": "peer"}),  # not this machine's record
        ({"reason": _RETIRING}, {"lost": None}),  # no run to attribute it to
    ],
    ids=[
        "crash",
        "other-run",
        "requested",
        "signal",
        "pid-reuse",
        "non-local-placement",
        "no-lost-run",
    ],
)
async def test_owner_loss_without_a_retiring_record_for_this_run_still_goes_cold(
    tmp_path, monkeypatch, exit_record, viewer
):
    """Control: real owner loss must stay a visible went-cold, never a silent refresh."""
    if exit_record is not None:
        _write_exit(tmp_path, **exit_record)
    if viewer.get("owner") == "peer":
        viewer = {**viewer, "owner": _PeerOwner()}
    remote, refreshed, cold = _reset_viewer(tmp_path, monkeypatch, **viewer)

    remote._on_disconnected("owner connection reset")
    task = remote._recovery_task
    assert task is not None
    await asyncio.wait_for(task, timeout=5)

    assert refreshed == [], "owner loss must not be read as a refresh"
    assert cold == ["cold"], "owner loss must go cold visibly"


@pytest.mark.asyncio
async def test_a_retiring_record_present_at_the_cold_deadline_is_still_read(tmp_path, monkeypatch):
    """m2 (#3797 review round 1): the deadline branch must look at the exit record once.

    A zero cold bound puts the loop's FIRST check on the deadline branch, which
    is exactly where a record written during the final paced sleep is met:
    the per-pass exit-record check sits after it and is never reached. Before
    the fix this went cold with refresh=False and nothing engaged a successor.
    """
    _write_exit(tmp_path, reason=_RETIRING)
    remote, refreshed, cold = _reset_viewer(tmp_path, monkeypatch, cold_s=0.0)

    remote._on_disconnected("owner connection reset")
    task = remote._recovery_task
    assert task is not None
    await asyncio.wait_for(task, timeout=5)

    assert refreshed == ["refresh"], "a record present at the deadline must still refresh"
    assert cold == []
    assert remote._recovering is False
