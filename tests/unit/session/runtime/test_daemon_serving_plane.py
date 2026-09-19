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
the runtime's thread, then leaves the session un-disposable. So two of the tests
below pin the seam itself — the turn on the session's loop, the registrations on
`session.py`'s own methods — because those are the readings that made the
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
    finally:
        runtime.close()
        await session.dispose()


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
