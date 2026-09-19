"""The update window: an idle handover QUEUES admissions instead of refusing them.

The incident this file answers (2026-09-19, live fleet): a session whose runtime
was still on 0.59.9 was reported by its own TUI as "it will switch to the new
version when it is next idle", and the operator's next message came back

    This session is leaving; it will not start a new turn. Your message is back in
    the composer — send it again once the session is running again.

The only recovery was ``/stop`` + ``/resume``, and the message had to be retyped.
The runtime was IDLE — it was leaving precisely because it had no work — so the
refusal protected nothing: the successor would have run the message had anyone
held it for them.

Four invariants are pinned here, in the order the spec states them:

1. an admission arriving while the window is open is SPOOLED and answered with the
   queued receipt, from every channel entry point (owner prompt, steer, peer wake);
2. the window is BOUNDED — with no heartbeat it fails open: the lock is released,
   the handover aborts, the runtime stays on the build it is running, and the
   failure is published with the incident cause token;
3. no admission path can block past ``UPDATE_LOCK_S``: they never touch the lock;
4. the successor publishes the one-shot "update applied" fact when it boots on the
   new build.

Driven the way ``test_serving_drain.py`` drives the drain latch: the production
methods over stub collaborators, so the assertion lands on the admission rather
than on a runtime boot (the e2e stage covers the boot).
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator import buildwatch
from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.inbox import (
    SOURCE_USER,
    SPOOL_RECEIPT_PROMPT,
    SPOOL_RECEIPT_WAKE,
    peek_inbox,
    read_update_window,
    write_update_window,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import (
    UPDATE_FAILED_CAUSE,
    UPDATING,
    UPDATING_DONE,
    update_phrase,
)
from local_operator.update import BuildStamp
from tests.unit.session.runtime.test_serving_drain import PromptHost, PromptSession

OLD = BuildStamp(version="0.59.9", source_ref="")
NEW = BuildStamp(version="0.59.11", source_ref="ead71b673a9a")
#: Built through the production formatter rather than typed, so the fixture carries
#: exactly the shape ``BuildStamp.label()`` produces (short ref, no brackets) — a
#: hand-typed pair would pin a string no runtime can emit.
PAIR = buildwatch.update_pair_text(OLD, NEW)


class WindowHost(PromptHost):
    """``PromptHost`` plus the window state the production ``__init__`` owns.

    The state is set here rather than reached through the real constructor
    because this file drives the admission decision, not a handle boot — the
    same split ``PromptHost`` makes for the prompt queue.
    """

    begin_update = ServingSessionHandle.begin_update
    heartbeat_update = ServingSessionHandle.heartbeat_update
    end_update = ServingSessionHandle.end_update
    # The marker's own two-hop read of the session directory, bound for the reason
    # the other production methods here are: ``begin_update`` writes the marker
    # through it, and a stubbed directory would pin nothing about which one it means.
    _session_directory = ServingSessionHandle._session_directory

    def __init__(self, session: PromptSession, *, busy: bool = True) -> None:
        super().__init__(session, busy=busy)
        self._updating = ""
        self._update_failed = ""
        self._update_lock = buildwatch.UpdateLock()
        self._applied_update = ""


def _window_host(tmp_path: Path, *, busy: bool = False) -> tuple[WindowHost, PromptSession]:
    session = PromptSession(tmp_path / "sessions" / "s1", busy=busy)
    session.transcript.directory.mkdir(parents=True, exist_ok=True)
    return WindowHost(session, busy=busy), session


# -- 1. the admission is QUEUED, from every channel ---------------------------


@pytest.mark.asyncio
async def test_a_prompt_during_the_window_is_queued_not_refused(tmp_path: Path) -> None:
    """The incident itself: the owner's message must be held, not handed back.

    The window state is set DIRECTLY rather than through ``begin_update`` so this
    cell runs on a tree without the window too — there ``_updating`` is ignored
    and the admission refuses, which is the red this asserts against.
    """
    host, session = _window_host(tmp_path)
    host._updating = PAIR

    receipt = await host.prompt("now summarise the build staleness fix", command_id="p" * 8)

    assert receipt == SPOOL_RECEIPT_PROMPT, receipt
    assert receipt != "prompt admitted"
    assert session.prompt_calls == [], "a turn was started on a runtime that is leaving"
    rows = peek_inbox(session.transcript.directory)
    assert len(rows) == 1, rows
    assert rows[0].source == SOURCE_USER, "the successor must run it as the owner's own"
    assert rows[0].wake is True, "a user prompt asks for a turn"


@pytest.mark.asyncio
async def test_a_steer_during_the_window_is_queued_not_lost(tmp_path: Path) -> None:
    """A steer has no turn to join: the runtime is idle by construction.

    Handing it to ``Session.steer`` would queue it against a turn that will never
    run in this process, and the process exits moments later — the exact
    silent-loss shape the spool exists to prevent.
    """
    host, session = _window_host(tmp_path)
    host._updating = PAIR

    receipt = await host.steer("actually, use the other provider", command_id="s" * 8)

    assert receipt == SPOOL_RECEIPT_PROMPT, receipt
    assert session.steered == [], "a steer was queued against a turn nobody will run"
    assert len(peek_inbox(session.transcript.directory)) == 1


@pytest.mark.asyncio
async def test_a_peer_wake_during_the_window_is_spooled(tmp_path: Path) -> None:
    """The peer channel is the one that already had the better answer."""
    host, session = _window_host(tmp_path)
    host._updating = PAIR

    receipt = await host.receive_peer_message("build is green", mode="wake", wake=True, sender={})

    assert receipt == SPOOL_RECEIPT_WAKE, receipt
    assert session.peer_calls == []
    assert len(peek_inbox(session.transcript.directory)) == 1


# -- 2. the bound: fail open, keep the old build, say so -----------------------


@pytest.mark.asyncio
async def test_a_stalled_window_fails_open_and_keeps_the_old_build(monkeypatch) -> None:
    """``UPDATE_LOCK_S`` with no heartbeat ends the attempt, never the runtime.

    The announce is the window's longest await (it drains each viewer's writer),
    so a viewer that never lets go is exactly the stall the bound is for. The
    assertion is the whole of the fail-open contract: the handover is ABANDONED
    (no disposal, no exit), the lock is RELEASED, the runtime keeps the build it
    is running, and the failure is published with the incident token so an
    operator can report it.
    """
    monkeypatch.setattr(child_mod, "_build_stagger_seconds", lambda: 0.0)
    monkeypatch.setattr(child_mod, "_build_changed", lambda _boot: NEW)
    monkeypatch.setattr(buildwatch, "update_lock_seconds", lambda: 0.05)
    monkeypatch.setattr(child_mod, "update_lock_seconds", lambda: 0.05, raising=False)

    handle = _WindowHandle()
    runtime = _WindowRuntime(announce_delay=0.5)
    stop = asyncio.Event()

    exited = await child_mod._refresh_for(NEW, handle, runtime, stop)

    assert exited is False, "a stalled window must not take the exit"
    assert handle.disposed is False, "the runtime stays on the build it is running"
    assert stop.is_set() is False
    assert handle.updating == "", "the window is closed on the failure path"
    assert handle.lock_held is False, "the admission lock is released"
    assert runtime.failures == [PAIR], "the failure is published"
    assert runtime.retiring == [], "the latency must not be reported as an ordinary handover"
    assert runtime.announced == [], "a stalled announce never completed, so nothing was said"


@pytest.mark.asyncio
async def test_a_window_that_finishes_in_time_hands_over(monkeypatch) -> None:
    """The green half: a healthy window is not touched by the bound."""
    monkeypatch.setattr(child_mod, "_build_stagger_seconds", lambda: 0.0)
    monkeypatch.setattr(child_mod, "_build_changed", lambda _boot: NEW)

    handle = _WindowHandle()
    runtime = _WindowRuntime(announce_delay=0.0)
    stop = asyncio.Event()

    exited = await child_mod._refresh_for(NEW, handle, runtime, stop)

    assert exited is True
    assert stop.is_set() is True
    assert runtime.failures == []
    assert handle.updating == PAIR, "the window stays open until the exit takes it"
    assert runtime.announced == ["stale-build"], "the announce must have gone out"


# -- 3. never deadlock: no admission path blocks on the lock -------------------


@pytest.mark.asyncio
async def test_an_admission_resolves_while_the_window_is_stalled(tmp_path: Path) -> None:
    """The no-deadlock invariant, exercised on the stall itself.

    ``prompt`` must answer while the window that is holding the lock is still
    stalled — not after it. ``asyncio.wait_for`` is the instrument rather than a
    wall-clock assertion: the admission coroutine either finishes inside the wake
    of the stalled task or it raised, and there is no budget to calibrate.
    """
    host, _session = _window_host(tmp_path)
    host._updating = PAIR

    async def stalled_window() -> None:
        await asyncio.sleep(10.0)

    stalled = asyncio.ensure_future(stalled_window())
    try:
        for _ in range(20):
            await asyncio.sleep(0)
            if stalled.done():  # pragma: no cover - a 10 s sleep cannot have finished
                raise AssertionError("the fixture's window did not stall")
        receipt = await asyncio.wait_for(
            host.prompt("held, not dropped", command_id="p" * 8), timeout=1.0
        )
    finally:
        stalled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stalled

    assert receipt == SPOOL_RECEIPT_PROMPT, receipt


def test_the_admission_paths_never_touch_the_update_lock() -> None:
    """The invariant's construct, pinned rather than timed.

    No portable numeric bound exists for "this never blocks" (AGENTS.md
    "Prefer a structural invariant to a numeric one"), and the property is
    structural here: the three admission paths read the window STRING and nothing
    else, so there is no lock for them to wait on. A future revision that reached
    for the lock to "check" it would reintroduce the deadlock the spec forbids,
    and no timing test would catch it on an idle box.
    """
    for method in (
        ServingSessionHandle.prompt,
        ServingSessionHandle.steer,
        ServingSessionHandle.receive_peer_message,
    ):
        source = inspect.getsource(method)
        assert "update_lock" not in source.lower(), (
            f"{method.__name__} reaches for the update lock; an admission must read the "
            f"window string and never wait on the lock that holds it"
        )


# -- 4. the completion fact ----------------------------------------------------


def test_the_successor_publishes_the_applied_update(tmp_path: Path) -> None:
    """The successor's boot is where the queued messages run, so it is where the
    "it worked" fact belongs — one read, once, and the marker is consumed."""
    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    write_update_window(directory, PAIR)
    handle = _BootHandle(directory)

    assert child_mod._consume_update_marker(handle) == PAIR

    assert handle.applied_update == PAIR
    assert read_update_window(directory) == "", "the fact is one-shot: the marker is consumed"


def test_a_boot_with_no_marker_publishes_nothing(tmp_path: Path) -> None:
    """The negative control: every ordinary boot must stay silent."""
    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    handle = _BootHandle(directory)

    assert child_mod._consume_update_marker(handle) == ""
    assert handle.applied_update == ""


def test_the_retiring_frame_carries_the_window_off_the_record() -> None:
    """The frame key is read off the RECORD, not taken as an argument.

    A parameter would be a second copy of a field the server already holds, and — the
    part that was measured — a caller whose ``announce_retiring`` predates the
    parameter takes a ``TypeError`` inside the ``except Exception`` that guards a
    viewer's writer, so the ENTIRE announcement is swallowed by the failure path meant
    for something else (``test_process_refresh``'s fake registrant lost its only frame
    exactly that way). The window is published on the record before the announce, so
    the record is the one place both ends can read it from.
    """
    import inspect

    from local_operator.session.runtime.server import RuntimeServer

    source = inspect.getsource(RuntimeServer._announce_retiring_on_loop)
    assert '"updating": self._record.updating,' in source, source


def test_the_record_seeds_the_applied_fact_from_the_handle() -> None:
    """The wire half: a session record must carry the fact the fleet surfaces read.

    Read off the source because constructing a ``RuntimeServer`` boot is the e2e
    stage's job, and the property here is that the ONE writer of the record
    consults the handle's boot fact at all.
    """
    from local_operator.session.runtime.server import RuntimeServer

    source = inspect.getsource(RuntimeServer.__init__)
    assert "applied_update" in source, (
        "the record is the only place a fleet surface can read the applied-update fact "
        "and it is built in RuntimeServer.__init__; nothing there reads the handle's"
    )


# -- the lock itself -----------------------------------------------------------


def test_the_lock_is_dead_after_the_bound_without_a_heartbeat() -> None:
    """The bound is on the HEARTBEAT, so a live window is never mistaken for one."""
    lock = buildwatch.UpdateLock()
    assert lock.acquire(PAIR, "stale-build") is True
    opened = lock.last_heartbeat()
    assert lock.expired(now=opened + buildwatch.UPDATE_LOCK_S - 0.01) is False
    assert lock.expired(now=opened + buildwatch.UPDATE_LOCK_S + 0.01) is True

    lock.heartbeat(now=opened + buildwatch.UPDATE_LOCK_S - 0.01)
    assert (
        lock.expired(now=opened + buildwatch.UPDATE_LOCK_S + 0.01) is False
    ), "a heartbeat inside the bound must move the deadline"
    assert (
        lock.heartbeat_interval() <= buildwatch.UPDATE_LOCK_S
    ), "a heartbeat interval at or above the bound can never keep a live lock alive"


def test_a_second_holder_cannot_take_the_lock() -> None:
    """One window at a time, and a dead one is recoverable rather than quarantined."""
    lock = buildwatch.UpdateLock()
    assert lock.acquire(PAIR) is True
    assert lock.acquire("other") is False
    lock.release()
    assert lock.acquire("other") is True


@pytest.mark.asyncio
async def test_the_failed_pair_is_not_retried_by_the_reaper(tmp_path: Path) -> None:
    """A failure that keeps retrying is a failure nobody can see.

    The window has no successor while it fails, so re-opening it for the same
    build would re-refuse nothing but churn the loop every check. The pair is
    remembered and the reaper's rung declines it; an explicit operator refresh
    may still ask.
    """
    host, _session = _window_host(tmp_path)
    assert host.begin_update(PAIR) is True
    host.end_update()
    host._update_failed = PAIR

    assert host.begin_update(PAIR) is False
    assert host.begin_update(PAIR, retry_failed=True) is True


# -- the record field ----------------------------------------------------------


def test_the_record_carries_the_window_and_the_failure() -> None:
    """Additive fields, and the pair the surfaces render.

    ``PROTOCOL_VERSION`` deliberately does not move: an older reader drops unknown
    keys (``SessionRecord.from_json``), which is the contract every live-state
    field on this record already keeps.
    """
    from local_operator.session.runtime.types import PROTOCOL_VERSION, SessionRecord

    base = dict(
        pid=1,
        kind="exec",
        session_id="s",
        conversation_name="c",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
    )
    record = SessionRecord(**base)
    assert record.updating == "" and record.update_failed == "" and record.updated == ""

    record.updating = PAIR
    record.update_failed = PAIR
    record.updated = PAIR
    round_tripped = SessionRecord.from_json(record.to_json())
    assert round_tripped.updating == PAIR
    assert round_tripped.update_failed == PAIR, (
        "the failure has to survive the wire: it is published on the record a runtime "
        "that STAYED owns, and a fleet surface reads it after the fact"
    )
    assert round_tripped.updated == PAIR
    assert (
        PROTOCOL_VERSION == SessionRecord(**base).protocol
    ), "an additive field must not spend the one number that gates frames"


def test_the_phrase_vocabulary_renders_the_pair_once() -> None:
    """One formatter, so the surfaces cannot disagree about the same window."""
    assert PAIR == "0.59.9 → 0.59.11@ead71b6", PAIR
    assert UPDATING in update_phrase(UPDATING, PAIR)
    assert PAIR in update_phrase(UPDATING, PAIR)
    assert PAIR in update_phrase(UPDATING_DONE, PAIR)
    # A runtime that could not read a stamp still says something true.
    assert "build on disk" in update_phrase(UPDATING, "")
    assert UPDATE_FAILED_CAUSE == "runtime-update-failed"


# -- the process-level rig -----------------------------------------------------


class _WindowHandle:
    """The window API the refresh rung drives, over no session at all.

    Deliberately NOT the production handle: these cells are about the rung's
    ORDERING (open before the announce, close on either outcome), which a boot
    would bury in a provider.
    """

    def __init__(self) -> None:
        self.updating = ""
        self.lock = buildwatch.UpdateLock()
        self.disposed = False
        self.spool_drains = 0

    def may_refresh(self) -> str:
        return ""

    def begin_update(self, pair: str, holder: str = "", *, retry_failed: bool = False) -> bool:
        if self.lock.acquire(pair, holder):
            self.updating = pair
            return True
        return False

    def heartbeat_update(self) -> None:
        self.lock.heartbeat()

    def end_update(self) -> bool:
        self.updating = ""
        return self.lock.release()

    @property
    def lock_held(self) -> bool:
        return self.lock.held

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        return True

    async def dispose(self) -> None:
        self.disposed = True


class _WindowRuntime:
    """The two things the rung talks to: the announce and the failure record."""

    _boot_build = OLD

    def __init__(self, *, announce_delay: float) -> None:
        self.announce_delay = announce_delay
        self.retiring: list[tuple[str, str, bool, str]] = []
        #: Every announce that completed, in order — so a cell can assert the frame
        #: went out at all (the window's pair rides the RECORD, not this call).
        self.announced: list[str] = []
        self.failures: list[str] = []

    async def announce_retiring(
        self,
        reason: str,
        *,
        to: str = "",
        draining: bool = False,
        leaving: str = "",
        updating: str = "",
    ) -> None:
        await asyncio.sleep(self.announce_delay)
        self.retiring.append((reason, to, draining, leaving))
        self.announced.append(reason)

    async def note_update_failed(self, pair: str, bound: float) -> None:
        self.failures.append(pair)


class _BootHandle:
    """The boot half: a session directory, and the fact the server will seed."""

    def __init__(self, directory: Path) -> None:
        self._session = SimpleNamespace(transcript=SimpleNamespace(directory=directory))
        self.applied_update = ""
