"""P3: the serving plane gets its own loop, and the handle marshals across it.

WHAT THIS PINS, and why each assertion is shaped the way it is.

`kind="daemon"` and `kind="exec"` runtimes used to serve in process
(`RuntimeServer.start_in_process`), so the listener, the welcome frame, `ping`
and the HEARTBEAT shared an event loop with the agent turn. Any synchronous step
of a turn therefore parked the entire control plane at once, and the surfaces
read that as a dead session — the operator's report, reproduced by the audit's
rig with NO client attached: the record crossed into `wedged` at t=46.2 s, and a
fresh dial connected in 0.01 s and then received no welcome within 15 s.

After the change the runtime owns a thread and a loop of its own, so the same
block leaves the welcome immediate, `ping` answering in 0.00 s, and the beat
below `HEARTBEAT_TIMEOUT_S`. A fresh beat now means THE SERVING PLANE RAN,
which is exactly the claim the surfaces already make.

AND THE HANDLE MARSHALS, because decoupling alone was measured to break the
product: without a seam, a `prompt` reaching the handle from the runtime's
thread answers with an asyncio cross-loop error while STILL running the turn on
the runtime's thread. So three of the tests below pin the seam itself — the turn
on the session's loop, the registrations on `session.py`'s own methods, and the
two mutating `_dispatch` arms whose omission from the seam left a cancel's
execution on the wrong loop with its failure swallowed — because those are the
readings that made the
un-marshalled version visible.

WHAT IT DOES NOT CLAIM. Mounting a guest mid-turn still waits for the turn's
current synchronous step — `subscribe_frontend` is hopped onto the session's
loop, so a join behind a 50 s step is answered when that step yields. The change
stops the LIE; it does not make attach instant, and no assertion here says it
does (see §4.4 of the audit).

The one expensive test is expensive on purpose. The defect is a beat that goes
stale while work is in flight, so a block shorter than `HEARTBEAT_TIMEOUT_S`
cannot produce the reading at all and would pass against the shape this change
removes. 50 s, once, is the price of a regression test that can fail.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import StreamEndEvent
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import HEARTBEAT_TIMEOUT_S
from tests.unit.session.test_session import make_session

#: Longer than ``HEARTBEAT_TIMEOUT_S`` (45 s) with margin: see the module
#: docstring for why a shorter one would not discriminate.
BLOCK_S = 50.0
#: The client's own bound, matching ``attach_client.ACK_TIMEOUT_S``; a welcome
#: inside it is the client's definition of "answering".
ACK_TIMEOUT_S = 15.0


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Home, config and cwd out of the way of the developer's real ones."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _recording_stream(calls: list[int]) -> Any:
    """A stream that records the THREAD its turn started on, then ends at once.

    The thread identity is the instrument, not decoration: "the turn ran on the
    runtime's thread" is what made the un-marshalled seam visible in the first
    place, so the regression test for the seam asserts the same reading.
    """

    def stream(request: Any, signal: Any) -> Any:
        calls.append(threading.get_ident())

        async def gen() -> Any:
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    return stream


def _dial(record: Any) -> tuple[dict[str, Any] | None, float]:
    """One blocking dial on a WORKER thread; returns (welcome, elapsed).

    A dial issued on the loop under test starves the plane being measured — the
    audit's rig had to learn that the hard way, and it is the reason every
    client here runs off the session's loop.
    """
    started = time.monotonic()
    sock = socket.create_connection(("127.0.0.1", record.control_port), timeout=ACK_TIMEOUT_S)
    sock.settimeout(ACK_TIMEOUT_S)
    sock.sendall(json.dumps({"key": record.control_key}).encode() + b"\n")
    buf = b""
    try:
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                return None, time.monotonic() - started
            buf += chunk
    except (OSError, TimeoutError):
        return None, time.monotonic() - started
    finally:
        sock.close()
    return json.loads(buf.split(b"\n", 1)[0]), time.monotonic() - started


def _request(
    record: Any,
    frame: dict[str, Any],
    *,
    timeout: float = ACK_TIMEOUT_S,
    auth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dial, send one op, return the reply for its ``req`` (or an empty dict).

    ``auth`` overrides the connection frame (the shape ``_on_connection`` reads
    to decide the client's kind and what it may subscribe to); the default is
    the daemon dial every other test in this module makes.
    """
    sock = socket.create_connection(("127.0.0.1", record.control_port), timeout=timeout)
    sock.settimeout(timeout)
    try:
        hello = {"key": record.control_key} if auth is None else auth
        sock.sendall(json.dumps(hello).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                return {}
            buf += chunk
        buf = buf.split(b"\n", 1)[1] if b"\n" in buf else b""
        sock.sendall(json.dumps(frame).encode() + b"\n")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            while b"\n" not in buf:
                sock.settimeout(max(0.1, deadline - time.monotonic()))
                chunk = sock.recv(65536)
                if not chunk:
                    return {}
                buf += chunk
            line, _, buf = buf.partition(b"\n")
            reply = json.loads(line)
            if reply.get("req") == frame.get("req"):
                return reply
        return {}
    finally:
        sock.close()


def _sample_states(samples: list[tuple[str, float]], stop: threading.Event) -> None:
    while not stop.is_set():
        found = registry.scan()
        if found:
            record, state = found[0]
            samples.append((state, max(0.0, time.time() - record.heartbeat_at)))
        time.sleep(2.0)


async def _let_the_server_settle() -> None:
    """Yield until the SERVER's side of a dial has finished with its loop.

    A reply frame is written BEFORE the connection task moves on to the rest of
    the op's epilogue — ``refresh()`` and ``_push()`` — and both of those are
    hops ONTO this loop. A client that closes as soon as it has its ack
    therefore leaves the connection task legitimately parked here, and a
    teardown that follows immediately reports it as "Task was destroyed but it
    is pending". The block these tests run through is what parks it; this is the
    yield that lets it land, and it is why the settle exists rather than an
    assertion being relaxed.
    """
    await asyncio.sleep(0.5)


async def _boot(tmp_path: Path, stream: Any, *, kind: str = "daemon") -> tuple[Any, Any, Any]:
    """A real session + real handle + thread-hosted runtime, published."""
    loop = asyncio.get_running_loop()
    session = make_session(tmp_path, stream)
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))
    runtime = RuntimeServer(handle, kind=kind)
    runtime.start()
    assert await runtime.wait_until_published(), "the boot prologue never published"
    return session, handle, runtime


@pytest.mark.asyncio
async def test_a_busy_session_never_reads_wedged_and_still_answers(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE REGRESSION TEST: the operator's sentence, asserted against.

    Three readings in one run, because they must all hold DURING the same block:
    the record never reads `wedged`, a fresh dial gets a welcome inside the
    client's own bound, and `ping` answers — while the session's loop is parked
    in a synchronous step of 50 s.
    """
    session, _handle, runtime = await _boot(tmp_path, _recording_stream([]))
    try:
        record = runtime.record
        # The structural half: two planes, two loops, two threads. Every reading
        # below follows from this and not from a timing coincidence.
        assert runtime._loop is not asyncio.get_running_loop()
        assert runtime._thread is not threading.current_thread()
        assert record.control_port != 0, "wait_until_published returned before the port was stamped"

        samples: list[tuple[str, float]] = []
        stop = threading.Event()
        sampler = threading.Thread(
            target=_sample_states, args=(samples, stop), name="p3a-sampler", daemon=True
        )
        sampler.start()

        results: dict[str, Any] = {}

        def during_the_block() -> None:
            time.sleep(1.0)
            results["welcome"] = _dial(record)
            results["ping"] = _request(record, {"op": "ping", "req": 7})

        worker = threading.Thread(target=during_the_block, name="p3a-client", daemon=True)
        worker.start()
        # The step itself: the SESSION's loop (this test's loop) is held for
        # BLOCK_S while everything above keeps running on the other plane.
        time.sleep(BLOCK_S)
        worker.join(timeout=2 * ACK_TIMEOUT_S)
        stop.set()
        sampler.join(timeout=5)

        states = {state for state, _age in samples}
        assert samples, "the sampler never read a record"
        assert "wedged" not in states, f"the beat went stale while the turn worked: {samples}"
        assert max(age for _state, age in samples) < HEARTBEAT_TIMEOUT_S

        welcome, welcome_s = results["welcome"]
        assert welcome is not None and welcome.get("op") == "projection"
        assert welcome_s < 1.0, f"a fresh dial waited {welcome_s:.2f}s for its welcome"
        assert results["ping"].get("detail") == "pong"
        await _let_the_server_settle()
    finally:
        runtime.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_prompt_over_the_control_socket_runs_on_the_sessions_loop(
    isolated_config: Path, tmp_path: Path
) -> None:
    """THE SEAM'S CONTRACT: a prompt's whole turn runs on the session's loop.

    The naive thread-hosted runtime did the opposite, and the instrument that
    showed it is the one used here — the thread the turn's own stream callback
    observes. It read ``lop-mobile-registrant`` (the RUNTIME's thread) while the
    client got an asyncio cross-loop error, and the session was then
    un-disposable (the probe had to be killed at 150 s with the session's loop
    parked in ``select()``). So this asserts: an ack, a turn that started, and
    that the turn's own callback ran on the thread holding the session's loop.
    """
    turns: list[int] = []
    session, handle, runtime = await _boot(tmp_path, _recording_stream(turns))
    try:
        session_thread = threading.get_ident()
        # On a WORKER thread: the op is hopped to the session's loop, which is
        # THIS loop, so a synchronous dial would park the very loop the runtime's
        # thread is waiting on.
        reply = await asyncio.to_thread(
            _request,
            runtime.record,
            {"op": "prompt", "req": 8, "text": "run this"},
        )
        assert reply.get("op") == "ack", reply

        deadline = time.monotonic() + 10
        while not turns and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert turns, "the prompt was acked but no turn started"
        assert turns[0] == session_thread, (
            "the turn ran on a thread that is not the session's loop — the "
            "wrong-thread execution this seam exists to prevent"
        )
        # And the drain ``prompt`` created finishes, on the session's loop.
        # The queue empties only when the DRAIN TASK pops the entry, and that
        # task is the very object the naive version created on the caller's
        # loop — so an entry that never drains is the signature of a task
        # created on the wrong one. Bounded, because a wrongly-threaded drain
        # does not raise: it simply never runs.
        deadline = time.monotonic() + 10
        while handle._prompt_queue and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert not handle._prompt_queue, "the drain task never finished"

        # And the session still disposes: bounded, so a mis-execution that wedged
        # teardown fails here instead of hanging the suite.
        await _let_the_server_settle()
        await asyncio.wait_for(session.dispose(), timeout=20)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_the_boot_registrations_run_on_the_sessions_loop(
    isolated_config: Path, tmp_path: Path
) -> None:
    """Every registration that MOVES is performed on the session's loop.

    `subscribe` folds history and seeds the projection's clocks and state;
    `subscribe_frontend` publishes. Running either on the runtime's thread would
    be a write to session state from the wrong plane, so the hop is asserted
    with the strongest instrument available — the loop the SESSION's own method
    actually ran on. Instrumenting the handle instead would prove nothing: a
    handle-side override runs wherever it was awaited, which is precisely the
    frame whose loop is NOT in question.
    """
    ran_on: list[tuple[str, Any]] = []

    loop = asyncio.get_running_loop()
    session = make_session(tmp_path, _recording_stream([]))
    real_subscribe = session.subscribe
    real_frontend = session.subscribe_frontend

    def subscribe(handler: Any) -> Any:
        ran_on.append(("session.subscribe", asyncio.get_running_loop()))
        return real_subscribe(handler)

    def subscribe_frontend(*args: Any, **kwargs: Any) -> Any:
        ran_on.append(("session.subscribe_frontend", asyncio.get_running_loop()))
        return real_frontend(*args, **kwargs)

    session.subscribe = subscribe  # type: ignore[method-assign]
    session.subscribe_frontend = subscribe_frontend  # type: ignore[method-assign]

    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    try:
        assert await runtime.wait_until_published()
        # ``subscribe`` and ``subscribe_events`` are the two BOOT registrations
        # and have already happened by the time ``start()`` settles; the
        # frontend bind is the per-connection one, so it needs a dial.
        await asyncio.to_thread(
            _request,
            runtime.record,
            {"op": "ping", "req": 1},
            auth={
                "key": runtime.record.control_key,
                "client": "attach",
                # ``frontend_state: True`` is exactly what the attach client
                # sends (``attach_client.AttachClient.connect``); it is a FLAG,
                # not a baseline — the sync frame carries the authoritative
                # state, and an empty dict here reads as falsy and would make
                # this a plain daemon dial that never binds a viewer.
                "frontend_state": True,
            },
        )
        deadline = time.monotonic() + 10
        while not any(name == "session.subscribe_frontend" for name, _ in ran_on):
            if time.monotonic() > deadline:
                break
            await asyncio.sleep(0.05)

        names = [name for name, _ in ran_on]
        assert names.count("session.subscribe") == 2, names  # subscribe + subscribe_events
        assert "session.subscribe_frontend" in names, names
        for name, ran in ran_on:
            assert ran is loop, f"{name} ran on {ran!r}, not the session's loop"
        # Same settle, same reason as the two dialling tests above: this test's
        # `ping` is acked before the connection task moves on to the epilogue
        # (`refresh()`, `_push()`), both hops onto this loop, so without it the
        # teardown below reports "Task was destroyed but it is pending". The
        # leak is a property of DIALING at all, not of the block those two run
        # through, which is why this third dialling test needs it too
        # (review round 1, MINOR-4).
        await _let_the_server_settle()
    finally:
        runtime.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_the_mutating_ops_run_on_the_sessions_loop(
    isolated_config: Path, tmp_path: Path
) -> None:
    """The two `_dispatch` arms that WRITE session state are hopped too.

    Review round 1's BLOCKER (D-1), found independently by the code reviewer and
    the design round: ``cancel_subagents_count`` and ``register_secret_redaction``
    are session-mutating ``def``s that ``_dispatch`` reached on the RUNTIME's
    loop, and neither was routed through ``_handle_call_on_session_loop``. The
    consequence was measured, not argued — on the runtime's thread
    ``Session.cancel_subagents`` created its task and aborted a loop-bound
    ``AsyncJobManager`` signal there, and the manager's
    ``RuntimeError: got Future … attached to a different loop`` was SWALLOWED at
    WARNING while the op still acked success — the plane and the swallowed
    failure, not a count of zero. The rig that found this watched the child's
    ``CancelledError`` land and the job row settle in BOTH arms, so the narrower
    claim is the one that reproduces (review round 2, QA Q3 / design D-8). This
    is the operator's second-Esc path.

    The instrument is the SESSION's own method, for the reason the registration
    test gives — a handle-side wrapper runs wherever it was awaited, which is the
    frame whose loop is not in question.
    """
    ran_on: list[tuple[str, Any]] = []

    loop = asyncio.get_running_loop()
    session = make_session(tmp_path, _recording_stream([]))

    # ``register_secret_redaction`` raises when there is no store (fail-closed),
    # so the store is attached or that op never reaches the instrument.
    from local_operator.variables import VariableStore

    # noqa: SLF001 — ``_variables`` is the property's backing field.
    session._variables = VariableStore(cwd=str(tmp_path))

    real_cancel = session.cancel_subagents
    real_register = session.variables.register_redaction

    def cancel_subagents(*args: Any, **kwargs: Any) -> Any:
        ran_on.append(("session.cancel_subagents", asyncio.get_running_loop()))
        return real_cancel(*args, **kwargs)

    def register_redaction(*args: Any, **kwargs: Any) -> Any:
        ran_on.append(("session.variables.register_redaction", asyncio.get_running_loop()))
        return real_register(*args, **kwargs)

    session.cancel_subagents = cancel_subagents  # type: ignore[method-assign]
    session.variables.register_redaction = register_redaction  # type: ignore[method-assign]

    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    try:
        assert await runtime.wait_until_published()
        # On a WORKER thread: both ops are hopped onto the session's loop, which
        # is THIS loop, so a synchronous dial would park the loop the runtime's
        # thread is waiting on.
        cancel = await asyncio.to_thread(
            _request, runtime.record, {"op": "cancel_subagents", "req": 1}
        )
        assert cancel.get("op") == "result", cancel
        register = await asyncio.to_thread(
            _request,
            runtime.record,
            {"op": "register_secret_redaction", "req": 2, "value": "probe-value"},
        )
        assert register.get("op") == "result", register

        names = [name for name, _ in ran_on]
        assert "session.cancel_subagents" in names, names
        assert "session.variables.register_redaction" in names, names
        for name, ran in ran_on:
            assert ran is loop, f"{name} ran on {ran!r}, not the session's loop"
        await _let_the_server_settle()
    finally:
        runtime.close()
        await session.dispose()


class _GuestWire:
    """One guest connection kept open for several ops — the shape a viewer dials.

    ``_request`` opens a connection per call, which cannot express "the bind
    landed between op A and op B"; the gate test needs one socket across the
    whole arc. Blocking by construction, like the other helpers in this module,
    so it runs under ``asyncio.to_thread``.
    """

    def __init__(self, record: Any, *, timeout: float = ACK_TIMEOUT_S) -> None:
        self.timeout = timeout
        self.buf = b""
        self.sock = socket.create_connection(("127.0.0.1", record.control_port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.sock.sendall(
            json.dumps(
                {
                    "key": record.control_key,
                    "client": "attach",
                    "locality": "remote",
                    "events": True,
                    "frontend_state": True,
                    "display_window": True,
                }
            ).encode()
            + b"\n"
        )

    def read(self, timeout: float | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            while b"\n" not in self.buf:
                self.sock.settimeout(max(0.05, deadline - time.monotonic()))
                chunk = self.sock.recv(1 << 20)
                if not chunk:
                    return {}
                self.buf += chunk
            line, _, self.buf = self.buf.partition(b"\n")
            try:
                return json.loads(line)
            except ValueError:
                continue
        return {}

    def read_until(self, op: str, *, timeout: float | None = None) -> dict[str, Any]:
        """The next frame carrying ``op`` — deltas may arrive before it."""
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            frame = self.read(max(0.05, deadline - time.monotonic()))
            if not frame or frame.get("op") == op:
                return frame
        return {}

    def call(self, frame: dict[str, Any]) -> dict[str, Any]:
        self.sock.sendall(json.dumps(frame).encode() + b"\n")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            reply = self.read(max(0.05, deadline - time.monotonic()))
            if not reply or reply.get("req") == frame.get("req"):
                return reply
        return {}

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


@pytest.mark.asyncio
async def test_a_binding_guest_is_live_but_not_yet_authoritative(
    isolated_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U6\'s remedy, and the price of it, in one walk of the guest wire.

    A follower used to be admitted and then DEAF: the connect handler awaited
    ``subscribe_frontend`` — a cross-thread hop onto the session's loop — before
    entering the reader loop, so a guest got its welcome and then nothing at all
    (review round 2, UX U6). The reader loop now starts first, which is what makes
    the connection reachable before it is AUTHORITATIVE; the admission gate is
    what keeps "reachable" from meaning "able to act on state it has not been told
    about".

    The BIND is what is stalled here, not the session's loop, so this is the
    gate's own window with everything else healthy: liveness is admitted, a
    session op is refused with a sentence rather than executed or dropped, and the
    same op is admitted once the sync has landed.

    THE OFF-LOOP FALLBACK IS DELIBERATELY BLINDED HERE. A stalled on-loop bind no
    longer stalls the sync at all — past ``_ONLOOP_BIND_GRACE_S`` the runtime
    binds off-loop and the guest is authoritative in ~100 ms, which is the
    improvement and is covered by its own test in ``test_server``. This test is
    about the ADMISSION window, so it keeps a handle that cannot take that path
    (the shape the TUI kind has), and the gate's window is the whole of what is
    left on that shape.
    """
    monkeypatch.delattr(ServingSessionHandle, "subscribe_frontend_nowait")
    real_subscribe_frontend = ServingSessionHandle.subscribe_frontend

    def slow_frontend(self: Any, *args: Any, **kwargs: Any) -> Any:
        async def later() -> Any:
            await asyncio.sleep(0.6)
            # AWAITED, and it matters: ``subscribe_frontend`` is an ``async def``
            # on this handle, so returning its coroutine un-awaited would leave the
            # test stalling the bind and then handing ``None``-with-a-warning up the
            # seam instead of the subscription.
            return await real_subscribe_frontend(self, *args, **kwargs)

        return later()

    monkeypatch.setattr(ServingSessionHandle, "subscribe_frontend", slow_frontend)
    session, _handle, runtime = await _boot(tmp_path, _recording_stream([]))

    def walk() -> dict[str, Any]:
        wire = _GuestWire(runtime.record)
        try:
            return {
                "welcome": wire.read(),
                "ping": wire.call({"op": "ping", "req": 1}),
                "refused": wire.call({"op": "prompt", "req": 2, "text": "act now"}),
                "sync": wire.read_until("frontend_sync"),
                "after": wire.call({"op": "snapshot", "req": 3}),
            }
        finally:
            wire.close()

    try:
        seen = await asyncio.to_thread(walk)
    finally:
        runtime.close()
        await asyncio.wait_for(session.dispose(), timeout=20)

    # The guest asks for the canonical frontend AND the event stream, so its
    # welcome is the identity-only one (``_slim_welcome_frame``); it is a welcome
    # either way, and what this walk is about is the frames after it.
    assert seen["welcome"].get("op") in ("projection", "welcome"), seen
    # LIVENESS IS ADMITTED while the bind is still in flight.
    assert seen["ping"].get("detail") == "pong", seen
    # A SESSION OP IS REFUSED, through the ordinary error frame, with the sentence
    # a person can act on — not silence, and not execution.
    assert seen["refused"].get("op") == "error", seen
    assert "still connecting" in str(seen["refused"].get("message")), seen
    # The sync is not skipped, only deferred: it is the frame the refusal's
    # "retry" waits for.
    assert seen["sync"].get("op") == "frontend_sync", seen
    # AND THE GATE OPENS once it lands — the same op is no longer refused.
    assert "still connecting" not in str(seen["after"].get("message")), seen


@pytest.mark.asyncio
async def test_wait_until_published_answers_no_when_the_bind_fails(
    isolated_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The latch is released on EVERY way out, success or failure.

    Without that, a caller waits out its whole bound for an answer `_serve`
    already has, and the two failures — "no surface" and "no surface yet" —
    become indistinguishable in the caller's timing.
    """

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("address already in use")

    monkeypatch.setattr(asyncio, "start_server", boom)
    loop = asyncio.get_running_loop()
    session = make_session(tmp_path, _recording_stream([]))
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    try:
        started = time.monotonic()
        assert await runtime.wait_until_published(timeout=30) is False
        assert time.monotonic() - started < 5, "a failed bind must release the latch, not the bound"
        assert runtime.record.control_port == 0
    finally:
        runtime.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_timed_out_publish_wait_takes_its_gate_back(
    isolated_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The TIMEOUT path leaves nothing behind (review round 2, NIT-3).

    ``_settle_publication`` empties ``_publication_gates`` wholesale, so the
    ordinary path cannot leak one — the test above drives the bind failure, which
    settles. The path that can leak is a prologue that never settles INSIDE the
    bound, and this is the only shape that reaches it: the registration hop is
    held longer than the wait, so the latch is still shut when the waiter's
    ``wait_for`` expires.
    """
    real_subscribe = ServingSessionHandle.subscribe

    def slow_subscribe(self: Any, *args: Any, **kwargs: Any) -> Any:
        async def wait_then_subscribe() -> Any:
            await asyncio.sleep(1.0)
            return real_subscribe(self, *args, **kwargs)

        return wait_then_subscribe()

    monkeypatch.setattr(ServingSessionHandle, "subscribe", slow_subscribe)
    loop = asyncio.get_running_loop()
    session = make_session(tmp_path, _recording_stream([]))
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path))
    runtime = RuntimeServer(handle, kind="daemon")
    runtime.start()
    try:
        assert await runtime.wait_until_published(timeout=0.1) is False
        assert (
            runtime._publication_gates == []
        ), "the timed-out waiter left its gate in the list for the life of the runtime"
    finally:
        # Let the held registration land before teardown, so this test is not
        # racing the very hop it held open.
        await asyncio.sleep(1.2)
        runtime.close()
        await session.dispose()


@pytest.mark.asyncio
async def test_aclose_remote_tears_down_from_a_foreign_loop(
    isolated_config: Path, tmp_path: Path
) -> None:
    """Teardown from the SESSION's loop, which is now the only teardown there is.

    `aclose` is owner-loop-only and must stay that way — a raise is the honest
    answer to a caller who would otherwise park forever — so the daemon and exec
    exit paths call `aclose_remote`, the same close awaited across a thread hop.
    Both are asserted here so the difference cannot be lost.
    """
    session, _handle, runtime = await _boot(tmp_path, _recording_stream([]))
    try:
        with pytest.raises(RuntimeError):
            await runtime.aclose()  # owner-loop-only, by design
        await runtime.aclose_remote()  # ...and safe twice, from the other loop
        assert runtime._closed.is_set()
    finally:
        runtime.close()
        await session.dispose()
