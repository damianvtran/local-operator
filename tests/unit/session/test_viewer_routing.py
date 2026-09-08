"""Viewer discovery and notification click routing.

The behaviour under test is "a notification click lands in the window the user
already has, instead of opening a new one". Its failure mode in the wild was a
silent one — an orphaned terminal per click — so the assertions here are about
observable acts (was a process spawned, was a switch issued) rather than about
internal state.

Every test that touches disk passes an explicit ``root``: the operator runs a
dozen live sessions on this machine and a test that wrote into their real
``run/`` would publish a record advertising a port pytest is about to close.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any

import pytest

from local_operator.session.runtime.viewer_client import (
    choose_viewer,
    deliver_click,
    needs_switch,
    route_click,
)
from local_operator.session.runtime.viewer_server import (
    FOCUS_WINDOW_CAPABILITY,
    ViewerServer,
)
from local_operator.session.runtime.viewers import (
    VIEWER_HEARTBEAT_TIMEOUT_S,
    ViewerRecord,
    publish_viewer,
    scan_viewers,
    viewer_record_path,
    viewer_run_dir,
)


class _Host:
    """A viewer host that records what a click asked of it.

    ``viewer_focus_window`` mirrors the production host's shape — it hops the
    blocking OS call onto a worker thread — because the property under test is
    "the OS call does not run on the loop that is serving this conversation".
    A double that recorded only its own thread could not express that: see
    ``test_focus_runs_off_the_serving_loop`` for why that shape was vacuous.
    """

    def __init__(self) -> None:
        self.resumed: list[str] = []
        #: The thread the host coroutine itself was scheduled on — i.e. the
        #: viewer endpoint's loop thread.
        self.handler_threads: list[int] = []
        #: The thread the (blocking) OS call actually executed on.
        self.focus_threads: list[int] = []
        #: Set True to drop the ``to_thread`` hop, reproducing the defect.
        self.focus_on_the_loop = False

    async def viewer_resume_session(self, session_id: str) -> str:
        self.resumed.append(session_id)
        return f"displayed {session_id}"

    async def viewer_focus_window(self) -> str:
        self.handler_threads.append(threading.get_ident())

        def activate() -> int:
            return threading.get_ident()

        if self.focus_on_the_loop:
            self.focus_threads.append(activate())
        else:
            self.focus_threads.append(await asyncio.to_thread(activate))
        return "activated"


class _NoFocusHost:
    """A viewer that cannot raise a window — ssh, Linux, an unknown emulator.

    The absence of the attribute is the point: the capability gate and the
    dispatcher both use ``hasattr``/``getattr``, so a host that merely set the
    attribute to ``None`` would still advertise and still be dispatched to.
    """

    def __init__(self) -> None:
        self.resumed: list[str] = []

    async def viewer_resume_session(self, session_id: str) -> str:
        self.resumed.append(session_id)
        return f"displayed {session_id}"


@pytest.fixture
def viewer_root(tmp_path):
    """An isolated config root. Never the operator's real one."""
    return tmp_path


def _started(host, root):
    server = ViewerServer(host, root=root)
    server.start()
    assert server.ready.wait(timeout=5.0), "viewer endpoint never bound"
    return server


def _endpoint_threads() -> int:
    """How many viewer endpoint threads this process is running.

    The listener is what leaks when ``start()`` stops being idempotent, and a
    pid-keyed record cannot show that — the second endpoint simply overwrites
    the first one's file. The thread is the observable.
    """
    return sum(1 for t in threading.enumerate() if t.name == "lop-viewer-endpoint")


def test_viewer_record_is_invisible_to_the_session_registry(viewer_root):
    """THE compatibility guarantee, and the reason viewer records live in their
    own directory.

    ``SessionRecord.from_json`` drops unknown KEYS but never validates ``kind``'s
    VALUE, so a ``kind="viewer"`` record placed in ``run/mobile`` parses as an
    ordinary session on every older build — becoming a ``lop stop --all`` target
    and a ``lop send`` destination. Older builds cannot be taught otherwise, and
    a dozen of them run on this machine at once, so isolation is the only
    mechanism available.
    """
    from local_operator.session.runtime import registry

    publish_viewer(
        ViewerRecord(pid=os.getpid(), surface="tui", control_port=1, control_key="k" * 64),
        viewer_root,
    )
    assert scan_viewers(viewer_root), "precondition: the viewer record must exist"
    # PRECONDITION: the two directories must genuinely be different ones. Without
    # this the assertion below is satisfied by an empty `run/mobile` that the
    # viewer never wrote to for some unrelated reason, and the test would keep
    # passing if the directories were merged AND the session scan happened to
    # reap the record it could not parse.
    assert (viewer_root / "run/viewers").is_dir()
    assert not (viewer_root / "run/mobile").exists() or not list(
        (viewer_root / "run/mobile").glob("*.json")
    ), "a viewer record must never be written where the session registry scans"
    # The session registry — what every older binary scans — must see nothing.
    assert registry.scan(viewer_root) == []


def test_dead_viewer_records_are_reaped_and_never_routed_to(viewer_root):
    """A ``kill -9``'d viewer leaves a file; it must not cost a click a dial."""
    dead = ViewerRecord(pid=2**22, surface="tui", control_port=1, control_key="k" * 64)
    publish_viewer(dead, viewer_root)
    path = viewer_record_path(dead.pid, viewer_root)
    assert path.exists(), "precondition: the stale record must be on disk"

    assert scan_viewers(viewer_root) == []
    assert not path.exists(), "the stale record should have been reaped"


def test_quiet_viewer_is_not_offered_as_a_target(viewer_root):
    """A wedged viewer (alive, not heartbeating) cannot answer a dial.

    It is filtered rather than surfaced: offering it would spend the click's
    dial timeout before the fallback it should have gone to directly.
    """
    record = ViewerRecord(pid=os.getpid(), surface="tui", control_port=1, control_key="k" * 64)
    publish_viewer(record, viewer_root)
    stale = json.loads(viewer_record_path(os.getpid(), viewer_root).read_text())
    stale["heartbeat_at"] = time.time() - (VIEWER_HEARTBEAT_TIMEOUT_S + 10)
    viewer_record_path(os.getpid(), viewer_root).write_text(json.dumps(stale))

    assert scan_viewers(viewer_root) == []


def test_click_switches_a_running_viewer_and_does_not_spawn(viewer_root):
    """C1 — the operator's reported bug, inverted into the expected behaviour."""
    host = _Host()
    server = _started(host, viewer_root)
    try:
        server.note_session("on-screen")
        records = scan_viewers(viewer_root)
        assert len(records) == 1, "precondition: exactly one viewer must be live"

        outcome = asyncio.run(deliver_click(records[0], "wanted"))

        assert outcome.switched is True
        assert host.resumed == ["wanted"]
    finally:
        server.close()


def test_click_for_the_displayed_session_does_not_re_resume(viewer_root):
    """C2 — focus, but no switch.

    Re-running ``/resume`` on the session already on screen would rebuild the
    view and throw away the user's scroll position for no gain.
    """
    host = _Host()
    server = _started(host, viewer_root)
    try:
        server.note_session("already-here")
        records = scan_viewers(viewer_root)
        assert needs_switch(records[0], "already-here") is False

        outcome = asyncio.run(deliver_click(records[0], "already-here"))

        assert outcome.switched is True, "the session IS displayed, so do not spawn"
        assert host.resumed == [], "no /resume may be issued for the visible session"
    finally:
        server.close()


def test_switch_succeeds_even_when_the_window_cannot_be_focused(viewer_root):
    """C3 — focus is chrome; the switch is the outcome that matters."""
    host = _NoFocusHost()
    server = _started(host, viewer_root)
    try:
        assert (
            FOCUS_WINDOW_CAPABILITY not in server.record.capabilities
        ), "precondition: a host without the method must not advertise the capability"
        server.note_session("other")
        records = scan_viewers(viewer_root)

        outcome = asyncio.run(deliver_click(records[0], "wanted"))

        assert outcome.switched is True
        assert outcome.focused is False
        assert host.resumed == ["wanted"]
    finally:
        server.close()


def test_unreachable_viewer_falls_through_without_switching(viewer_root):
    """C5 — a record pointing at a dead port must not swallow the click.

    Asserts the OUTCOME (routing declined, so the caller spawns) rather than a
    wall-clock bound: this repo has abandoned numeric timing bounds as
    unportable, and "did it decline" is the fact the caller acts on.
    """
    publish_viewer(
        # A live pid (ours) with a port nothing listens on: the wedged-viewer
        # shape, which pid liveness alone cannot detect.
        ViewerRecord(pid=os.getpid(), surface="tui", control_port=1, control_key="k" * 64),
        viewer_root,
    )
    assert scan_viewers(viewer_root), "precondition: the record must look live"

    outcome = route_click("wanted", viewer_root)

    assert outcome.switched is False


def test_no_viewer_running_declines_so_the_spawn_path_runs(viewer_root):
    """C4 — the original premise, now checked instead of assumed."""
    viewer_run_dir(viewer_root)
    outcome = route_click("wanted", viewer_root)
    assert outcome.switched is False


def test_viewer_precedence_is_deterministic():
    """C6 — several viewers could take it.

    The already-displaying viewer wins (switching it is a no-op), then the most
    recently focused, then the lowest pid. Ordering is asserted against an
    INPUT LIST IN THE WRONG ORDER, because the function must sort rather than
    inherit its collaborator's ordering.
    """
    old_focus = ViewerRecord(
        pid=1, surface="tui", control_port=1, control_key="a", current_session="x", focused_at=100.0
    )
    showing = ViewerRecord(
        pid=2, surface="tui", control_port=2, control_key="b", current_session="wanted"
    )
    recent = ViewerRecord(
        pid=3, surface="tui", control_port=3, control_key="c", current_session="y", focused_at=500.0
    )

    assert choose_viewer([old_focus, showing, recent], "wanted") is showing
    assert choose_viewer([old_focus, recent], "nobody-has-it") is recent
    assert choose_viewer([], "wanted") is None


def test_a_viewer_that_cannot_switch_is_not_chosen():
    """``can_switch`` is a capability statement, and routing must honour it."""
    fixed = ViewerRecord(
        pid=1, surface="tui", control_port=1, control_key="a", can_switch=False, focused_at=900.0
    )
    assert choose_viewer([fixed], "wanted") is None


def test_focus_runs_off_the_serving_loop(viewer_root):
    """STRUCTURAL: the OS call must not execute on the loop serving the click.

    A thread-identity assertion rather than a timing bound, per this repo's
    standing preference: it is a fact about WHERE the code ran, whereas a
    wall-clock bound on a window-server call is unportable by construction.

    **THE COMPARISON IS AGAINST THE HANDLER'S OWN THREAD, and the earlier
    version's choice of the CALLER's thread is why it was vacuous.**
    ``ViewerServer`` always serves on its dedicated ``lop-viewer-endpoint``
    thread, so the handler can never run on the caller's thread whatever the
    production code does — the assertion was satisfied by the threading model
    alone and passed with the ``to_thread`` hop deleted (found independently by
    the reviewer and by QA, both by mutation). The property that actually
    matters is that a blocking OS call does not occupy the loop that has to keep
    answering this conversation, and only handler-vs-OS-call expresses it.

    ``_Host.focus_on_the_loop`` is the defect switch, exercised below, so this
    guard's ability to go red is asserted rather than assumed.
    """
    host = _Host()
    server = _started(host, viewer_root)
    try:
        server.note_session("other")
        records = scan_viewers(viewer_root)

        asyncio.run(deliver_click(records[0], "wanted"))

        assert host.focus_threads, "precondition: focus must actually have been invoked"
        assert host.handler_threads, "precondition: the host coroutine must have run"
        # PRECONDITION for the assertion below to mean anything: the handler
        # must not have run on this test's thread, or "different from the
        # handler" would be trivially true of anything the endpoint did.
        assert threading.get_ident() not in host.handler_threads
        assert host.focus_threads[0] != host.handler_threads[0], (
            "the OS call ran on the loop serving the click; a slow window "
            "server would stall this endpoint"
        )
    finally:
        server.close()


def test_the_focus_thread_guard_goes_red_when_the_hop_is_dropped(viewer_root):
    """The mutation the guard above is meant to catch, asserted as a behaviour.

    Guard vacuity is not detectable from a passing test, so the defect is
    reproduced here in the double rather than left to a manual mutation that
    nobody re-runs: with the hop dropped, the OS call lands on the very loop
    thread the handler is running on. If this test ever stops observing that
    equality, the guard above has stopped being able to fail.
    """
    host = _Host()
    host.focus_on_the_loop = True
    server = _started(host, viewer_root)
    try:
        server.note_session("other")
        records = scan_viewers(viewer_root)

        asyncio.run(deliver_click(records[0], "wanted"))

        assert host.focus_threads and host.handler_threads, "precondition: focus must have run"
        assert host.focus_threads[0] == host.handler_threads[0]
    finally:
        server.close()


def test_the_client_ack_bound_outlasts_the_server_resume_bound():
    """Q1: the two bounds must never invert, and are derived so they cannot.

    An 8.0 s client ack against a 10.0 s server bound gave the user BOTH
    outcomes for a ``/resume`` measured at 8.5 s — the running TUI switched AND
    a duplicate terminal opened, which is the precise defect this feature
    removes. The client now derives its deadline from the server's, so an edit
    to one carries to the other; this asserts the resulting ordering, and the
    grace term's sign, rather than two hand-checked literals.
    """
    from local_operator.session.runtime import viewer_client
    from local_operator.session.runtime.viewers import (
        VIEWER_ACK_GRACE_S,
        VIEWER_RESUME_TIMEOUT_S,
    )

    assert VIEWER_ACK_GRACE_S > 0, "a non-positive grace makes the two bounds equal or inverted"
    assert viewer_client._ACK_TIMEOUT_S > VIEWER_RESUME_TIMEOUT_S, (
        "the client must never give up on a switch the server is still going "
        "to land: that delivers a switch AND a duplicate window"
    )


def test_a_viewer_speaking_an_unknown_protocol_is_not_dialed():
    """MINOR-3: the refusal ``viewers`` documents is now performed.

    Dialing a socket whose frames we cannot predict is worse than not dialing
    it — the click spends its ack timeout on a conversation neither side
    understands, when the spawn fallback works against any build.
    """
    from local_operator.session.runtime.viewers import VIEWER_PROTOCOL

    future = ViewerRecord(
        pid=1,
        surface="tui",
        control_port=1,
        control_key="a",
        protocol=VIEWER_PROTOCOL + 1,
        focused_at=900.0,
    )
    known = ViewerRecord(pid=2, surface="tui", control_port=2, control_key="b", focused_at=100.0)

    assert choose_viewer([future], "wanted") is None
    # Even when it is the freshest and already displaying the target, which is
    # rung 1 of the precedence — the protocol filter must precede both rungs.
    future.current_session = "wanted"
    assert choose_viewer([future, known], "wanted") is known


def test_focus_is_not_requested_from_a_viewer_that_cannot_do_it(viewer_root):
    """MINOR-4: the advertised capability is READ, not merely published.

    A viewer without the method answers the unknown-op error; spending a round
    trip and the focus timeout to be told what the record already said is
    latency on the click path for nothing.
    """
    host = _NoFocusHost()
    server = _started(host, viewer_root)
    # Spy on the SERVER's dispatcher rather than adding a counter to production
    # code: the observable question is "did a focus_window frame arrive", and
    # the wire is where that is answerable.
    seen_ops: list[str] = []
    original = server._dispatch

    async def spy(frame, writer):
        seen_ops.append(str(frame.get("op") or ""))
        return await original(frame, writer)

    server._dispatch = spy  # type: ignore[method-assign]
    try:
        server.note_session("other")
        records = scan_viewers(viewer_root)
        assert FOCUS_WINDOW_CAPABILITY not in records[0].capabilities, "precondition"

        outcome = asyncio.run(deliver_click(records[0], "wanted"))

        assert outcome.switched is True
        assert outcome.focused is False
        assert "resume_session" in seen_ops, "precondition: the spy must see the switch"
        assert "focus_window" not in seen_ops, "no focus_window frame should have been sent"
    finally:
        server.close()


def test_an_oversized_frame_does_not_wedge_an_authenticated_conversation(viewer_root, caplog):
    """MINOR-6: ``readline`` raises on an over-limit line WITHOUT consuming it.

    The client half hand-rolls its framing to dodge this exact defect; the
    server half did not, so an over-limit line raised ``LimitOverrunError``
    (a ``ValueError``) out of ``client_connected_cb``.

    **THE ASSERTION IS ON THE ESCAPING EXCEPTION, NOT ON THE CLOSED SOCKET, and
    that is the whole point of this test's shape.** Its first version checked
    that the conversation ended — which it does either way, since an uncaught
    exception tears the connection down just as a deliberate return does. That
    version passed with the defect reintroduced (mutation-proven), i.e. it was
    vacuous. What actually distinguishes handled from unhandled here is asyncio
    logging the traceback, so that is what is asserted.
    """
    caplog.set_level(logging.ERROR, logger="asyncio")
    server = _started(_Host(), viewer_root)
    try:
        record = scan_viewers(viewer_root)[0]

        async def probe() -> bytes:
            reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
            writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
            await writer.drain()
            # One line, comfortably over the endpoint's 64 KiB frame limit.
            writer.write(b"x" * (128 * 1024) + b"\n")
            try:
                await writer.drain()
                return await asyncio.wait_for(reader.read(1024), timeout=5.0)
            except (ConnectionResetError, BrokenPipeError):
                # A close against unread bytes may surface as RST rather than
                # FIN; both mean the conversation ended, as elsewhere here.
                return b""
            finally:
                writer.close()

        assert asyncio.run(probe()) == b"", "the conversation should end, not hang or reply"
        # The endpoint's loop logs from its own thread; give it a moment to.
        deadline = time.time() + 2.0
        while time.time() < deadline and not caplog.records:
            time.sleep(0.05)
        escaped = [
            record
            for record in caplog.records
            if "client_connected_cb" in record.getMessage()
            or "LimitOverrunError" in (record.exc_text or "")
        ]
        assert not escaped, f"the over-limit line escaped the handler: {escaped}"
    finally:
        server.close()


def test_starting_twice_binds_one_listener(viewer_root):
    """The endpoint outlives session swaps, so ``start()`` is re-entered.

    A second bind would leak a listener AND a record advertising its port —
    strictly worse than the source-leak class this codebase already documents.
    """
    host = _Host()
    server = _started(host, viewer_root)
    try:
        port = server.record.control_port
        threads_before = _endpoint_threads()
        assert threads_before == 1, "precondition: exactly one endpoint thread should exist"

        server.start()
        server.start()

        # THREAD COUNT, not the record or the port. The record is keyed by pid,
        # so a second endpoint in the same process OVERWRITES the first one's
        # file and leaves the port field looking correct — a leaked listener is
        # invisible on disk. The thread is the thing that actually leaks, so it
        # is the thing to count.
        assert _endpoint_threads() == 1
        assert server.record.control_port == port
    finally:
        server.close()


def test_close_removes_the_record(viewer_root):
    """A closed window must stop advertising a port nothing is listening on."""
    server = _started(_Host(), viewer_root)
    path = viewer_record_path(os.getpid(), viewer_root)
    assert path.exists(), "precondition: the record must have been published"

    server.close()
    server.close()  # idempotent

    # Poll rather than assert once: the serve loop notices `_closed` on its own
    # 200 ms tick and unpublishes from there, so a bare assertion races the
    # teardown it is checking and would pass on the synchronous unlink alone.
    deadline = time.time() + 5.0
    while path.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert not path.exists()


def test_unknown_op_is_answered_not_dropped(viewer_root):
    """Mid-upgrade skew: a newer client asking an older viewer for something it
    does not have must read an error and degrade, never hang."""
    server = _started(_Host(), viewer_root)
    try:
        record = scan_viewers(viewer_root)[0]

        async def probe() -> dict[str, Any]:
            reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
            writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
            writer.write(json.dumps({"op": "not_a_real_op", "req": 7}).encode() + b"\n")
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            writer.close()
            return json.loads(line)

        reply = asyncio.run(probe())
        assert reply["req"] == 7
        assert reply["op"] == "error"
    finally:
        server.close()


def test_a_bad_key_is_refused_without_a_reply(viewer_root):
    """An endpoint that answers wrong keys with errors is an oracle."""
    server = _started(_Host(), viewer_root)
    try:
        record = scan_viewers(viewer_root)[0]

        async def probe() -> bytes:
            reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
            writer.write(json.dumps({"key": "wrong"}).encode() + b"\n")
            writer.write(json.dumps({"op": "focus_window", "req": 1}).encode() + b"\n")
            try:
                await writer.drain()
                data = await asyncio.wait_for(reader.read(1024), timeout=5.0)
            except (ConnectionResetError, BrokenPipeError):
                # A close against unread bytes sends RST rather than FIN, so the
                # peer may observe a reset instead of a clean EOF. Both mean the
                # same thing here and which one occurs is a kernel timing
                # detail, so accepting only EOF made this test flaky rather than
                # strict.
                return b""
            finally:
                writer.close()
            return data

        assert asyncio.run(probe()) == b"", "an authenticated reply must never follow a bad key"
    finally:
        server.close()
