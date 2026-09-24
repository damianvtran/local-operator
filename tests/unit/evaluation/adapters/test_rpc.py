from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import re
import time
from typing import Any

import pytest

from local_operator.evaluation import deadlines
from local_operator.evaluation.adapters.api import (
    CloseParams,
    ExecuteParams,
    InspectRequirementsParams,
    ObserveParams,
)
from local_operator.evaluation.adapters.rpc import (
    MAX_RPC_BYTES,
    CancelRequest,
    IncrementalReader,
    IncrementalWriter,
    RpcClient,
    RpcProtocolError,
    RpcRequest,
    RpcResponse,
    canonical_line,
    parse_canonical_line,
)
from local_operator.evaluation.evidence.models import canonical_digest
from local_operator.evaluation.protocol import (
    MAX_TEXT_LENGTH,
    ActionBatch,
    ClickAction,
    TypeAction,
)


def request_line() -> bytes:
    return canonical_line(RpcRequest(jsonrpc="2.0", id=1, method="inspect_requirements", params={}))


@pytest.mark.parametrize(
    "payload",
    [
        b'{"id":1,"id":1,"jsonrpc":"2.0","method":"inspect_requirements","params":{}}\n',
        b'{"id":1, "jsonrpc":"2.0","method":"inspect_requirements","params":{}}\n',
        request_line().replace(b"\n", b"\r\n"),
        request_line()[:-1],
        b"{" + b"x" * MAX_RPC_BYTES + b"\n",
    ],
)
def test_malformed_noncanonical_and_oversized_lines_fail(payload: bytes) -> None:
    with pytest.raises(RpcProtocolError):
        parse_canonical_line(payload, RpcRequest)


def test_incremental_reader_handles_short_reads_and_rejects_partial_eof() -> None:
    read_fd, write_fd = os.pipe()
    try:
        encoded = request_line()
        for byte in encoded:
            os.write(write_fd, bytes([byte]))
        assert IncrementalReader(read_fd).read_line() == encoded
        os.write(write_fd, b"partial")
        os.close(write_fd)
        write_fd = -1
        with pytest.raises(RpcProtocolError, match="partial"):
            IncrementalReader(read_fd).read_line()
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def test_incremental_writer_handles_short_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks: list[bytes] = []

    def short_write(fd: int, data: memoryview) -> int:
        del fd
        chunk = bytes(data[:2])
        chunks.append(chunk)
        return len(chunk)

    monkeypatch.setattr(os, "write", short_write)
    IncrementalWriter(9).write(b"abcdef")
    assert b"".join(chunks) == b"abcdef"


@pytest.mark.asyncio
async def test_wrong_response_id_or_method_poison_and_terminate() -> None:
    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)

    async def peer() -> None:
        line = await asyncio.to_thread(IncrementalReader(requests_read).read_line)
        request = parse_canonical_line(line, RpcRequest)
        assert isinstance(request, RpcRequest)
        response = RpcResponse(
            jsonrpc="2.0",
            id=request.id + 1,
            method=request.method,
            result={"requirements": []},
        )
        IncrementalWriter(responses_write).write(canonical_line(response))

    task = asyncio.create_task(peer())
    with pytest.raises(RpcProtocolError, match="does not match the in-flight call") as excinfo:
        await client.call("inspect_requirements", InspectRequirementsParams(), timeout=1)
    await task
    assert terminated.is_set()
    # Both sides of the disagreement are named: "the response ID differs" does
    # not tell a reader of the artifact which id was expected or which arrived.
    assert "expected inspect_requirements id 1" in str(excinfo.value)
    assert "got inspect_requirements id 2" in str(excinfo.value)
    # The reply that did NOT answer is the distinguishing half of this message,
    # so it leads: the readouts cut the line at 110-160 characters, and
    # "expected inspect_requirements id 1" is the same for every call of the
    # method.
    message = str(excinfo.value)
    assert message.index("got inspect_requirements id 2") < 110
    assert message.index("got inspect_requirements id 2") < message.index("expected")
    for fd in (requests_read, requests_write, responses_read, responses_write):
        os.close(fd)


@pytest.mark.asyncio
async def test_timeout_sends_cancel_then_terminates() -> None:
    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()
    lines: list[bytes] = []

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)

    async def peer() -> None:
        reader = IncrementalReader(requests_read)
        lines.append(await asyncio.to_thread(reader.read_line))
        lines.append(await asyncio.to_thread(reader.read_line))

    task = asyncio.create_task(peer())
    with pytest.raises(TimeoutError):
        await client.call("inspect_requirements", InspectRequirementsParams(), timeout=0.01)
    await task
    assert b'"control":"cancel"' in lines[1]
    assert terminated.is_set()
    for fd in (requests_read, requests_write, responses_read, responses_write):
        os.close(fd)


def _declaring_execute(*durations_ms: int) -> ExecuteParams:
    """An ``execute`` whose batch declares exactly these waits, in order.

    The shape the campaign died on: a batch is legal up to ``MAX_BATCH_SIZE``
    (64) waits of up to ``MAX_WAIT_MS`` (60 s), and the call's budget used to
    ignore both.
    """

    batch = ActionBatch.model_validate(
        {
            "protocol_version": "1.0",
            "task_id": "task",
            "episode_id": "episode",
            "observation_id": "obs",
            "actions": [
                {"kind": "wait", "observation_id": "obs", "duration_ms": duration_ms}
                for duration_ms in durations_ms
            ],
        }
    )
    return ExecuteParams(
        operation_id="exec-declared",
        action_batch=batch,
        action_batch_id=canonical_digest("adapter-action-batch-v1", batch),
    )


def _undeclared_execute() -> ExecuteParams:
    """An ``execute`` whose batch declares no duration, as a key batch does."""

    batch = ActionBatch.model_validate(
        {
            "protocol_version": "1.0",
            "task_id": "task",
            "episode_id": "episode",
            "observation_id": "obs",
            "actions": [{"kind": "key", "observation_id": "obs", "keys": ["enter"]}],
        }
    )
    return ExecuteParams(
        operation_id="exec-undeclared",
        action_batch=batch,
        action_batch_id=canonical_digest("adapter-action-batch-v1", batch),
    )


async def _finish_peer(task: asyncio.Task[None], *fds: int) -> None:
    """Retire the peer, then close every protocol fd. ALWAYS run this first.

    A call that timed out leaves a reader thread blocked inside
    ``asyncio.to_thread`` on the response pipe, and a peer that never answers
    never releases it; closing the WRITE end is what gives that read its EOF.
    So this has to run before ANY assertion in these tests, and for every
    exception rather than the expected one: an assertion raised while the
    thread is still parked hangs the loop's own shutdown (measured: still
    hanging at 45 s) instead of reporting a clean failure, which turns a test
    failure into an infrastructure error.
    """

    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    for fd in fds:
        os.close(fd)


@pytest.mark.asyncio
async def test_a_batch_that_declares_waiting_is_funded_past_the_callers_budget() -> None:
    """A LEGAL batch is never killed by a budget that never saw it.

    The scale is smaller than the campaign's (0.6 s of declared waiting against
    a 0.2 s budget, rather than 240 s against 180 s) because the assertion is
    about which of the two numbers governs the call, not about the size of
    either; the real numbers are pinned in
    ``tests/unit/evaluation/test_deadlines.py``. The peer sleeps the declared
    time for the same reason the real worker does: ``wait`` is an
    ``asyncio.sleep`` inside the worker (``adapter.py``'s ``execute``), so a
    cancel cannot shorten it -- the parent's deadline is the only thing that
    decides whether the batch finishes.

    On the tree without the derivation this call raises ``TimeoutError`` after
    0.2 s and poisons the channel; the peer's full declared sleep then finishes
    into a channel nobody reads. That failure is the recorded defect.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    params = _declaring_execute(150, 150, 150, 150)

    async def peer() -> None:
        line = await asyncio.to_thread(IncrementalReader(requests_read).read_line)
        request = parse_canonical_line(line, RpcRequest)
        assert isinstance(request, RpcRequest)
        await asyncio.sleep(0.6)
        response = RpcResponse(
            jsonrpc="2.0", id=request.id, method=request.method, result={"applied": True}
        )
        try:
            IncrementalWriter(responses_write).write(canonical_line(response))
        except OSError:
            # The poisoned side of the BASE tree closed nothing, but a future
            # teardown that does must not turn this test into a pipe error.
            pass

    task = asyncio.create_task(peer())
    started = time.monotonic()
    timed_out: TimeoutError | None = None
    result: dict[str, Any] | None = None
    try:
        try:
            result = await client.call("execute", params, timeout=0.2)
        except TimeoutError as error:
            timed_out = error
        elapsed = time.monotonic() - started
        poisoned = terminated.is_set()
    finally:
        await _finish_peer(task, requests_read, requests_write, responses_read, responses_write)

    assert timed_out is None, (
        "BASE behaviour: the call was cut off at the caller's budget although the batch "
        f"declared {0.6} s of waiting ({timed_out!r})"
    )
    assert result == {"applied": True}
    assert elapsed >= 0.6
    # Funding the declaration is not the same as disarming the deadline: the
    # call completed because the worker did, so nothing poisoned anything.
    assert not poisoned


@pytest.mark.asyncio
async def test_a_wedged_execute_that_declares_nothing_still_times_out_and_poisons() -> None:
    """The honest failure mode, unchanged: a wedged call declares no duration.

    A batch of keys asks for no waiting, so there is nothing to fund and the
    caller's budget is the whole deadline -- exactly as before. The channel is
    then poisoned, because a late reply to a timed-out mutation must stay
    unreadable rather than be mistaken for this call's answer.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)

    async def peer() -> None:
        # Reads the request and never answers: a worker wedged inside the
        # adapter, which is the failure the timeout exists for.
        await asyncio.to_thread(IncrementalReader(requests_read).read_line)

    task = asyncio.create_task(peer())
    timed_out: TimeoutError | None = None
    try:
        try:
            await client.call("execute", _undeclared_execute(), timeout=0.05)
        except TimeoutError as error:
            timed_out = error
        poisoned = terminated.is_set()
    finally:
        # Before any assertion, and for every exception: see `_finish_peer`.
        await _finish_peer(task, requests_read, requests_write, responses_read, responses_write)
    assert timed_out is not None, "a wedged call must still time out"
    assert poisoned, "a timed-out channel must still be poisoned"
    with pytest.raises(RpcProtocolError, match="poisoned"):
        await client.call("execute", _undeclared_execute(), timeout=5.0)


@pytest.mark.asyncio
async def test_the_derived_budget_is_a_deadline_and_not_a_licence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wait-bearing call that then wedges is still cut off, at ITS budget.

    Declaring work buys time for that work, not immunity: the deadline still
    fires and still poisons. The assertion has to separate two numbers that are
    only 0.09 s apart at their defaults (a 0.01 s configured budget and a
    0.05 s + 0.05 s derived one, each buried under the timeout path's own 1 s
    cancel grace), which is why the headroom is monkeypatched to 3 s: the
    derived budget is then 3.05 s, so a LOWER bound on the elapsed time
    distinguishes "the derived number governs" from "the configured number
    governs". The UPPER bound is what keeps the derived number from simply
    borrowing the peer's 30 s; that the deadline exists at all is the
    `timed_out is not None` assertion, since an absent one would block forever
    rather than trip a bound.

    With the default headroom both bounds would be true on either tree, which
    is what makes the monkeypatch load-bearing rather than decoration.
    """

    monkeypatch.setattr(deadlines, "DECLARED_WORK_HEADROOM_S", 3.0)
    declared_s, headroom_s, configured_s = 0.05, 3.0, 0.01
    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    params = _declaring_execute(int(declared_s * 1000))

    async def peer() -> None:
        await asyncio.to_thread(IncrementalReader(requests_read).read_line)
        await asyncio.sleep(30)

    task = asyncio.create_task(peer())
    started = time.monotonic()
    timed_out: TimeoutError | None = None
    try:
        try:
            await client.call("execute", params, timeout=configured_s)
        except TimeoutError as error:
            timed_out = error
        elapsed = time.monotonic() - started
        poisoned_at_timeout = terminated.is_set()
    finally:
        # Before any assertion, and for every exception: see `_finish_peer`.
        await _finish_peer(task, requests_read, requests_write, responses_read, responses_write)
    assert timed_out is not None, "the deadline did not fire at all"
    # LOWER bound: the call survived past `declared + headroom`, which the
    # configured 0.01 s budget would not have allowed -- on the tree without the
    # derivation it raises at ~1.01 s (the cancel grace alone) and fails here.
    assert elapsed >= declared_s + headroom_s
    # UPPER bound: it did NOT wait for the peer's 30 s, so the derived budget is
    # a deadline rather than a licence.
    assert elapsed < 10.0
    assert poisoned_at_timeout


async def _silent_peer(requests_read: int, seen: list[bytes] | None = None) -> None:
    """Take requests off the wire and never answer them.

    The shape of the observed failure: the guest-side call outran its budget,
    so the parent's read never completed and the only thing it could report was
    that something timed out.
    """

    reader = IncrementalReader(requests_read)
    while True:
        try:
            line = await asyncio.to_thread(reader.read_line)
        except (EOFError, RpcProtocolError):
            return
        if seen is not None:
            seen.append(line)


@pytest.mark.asyncio
async def test_a_timeout_names_the_call_and_the_budget_it_exceeded() -> None:
    """The observed failure: 6078 s of paid episode ended on ``TimeoutError: ``.

    A guest call outran its budget, and the artifact the run left behind -- 143
    bytes, ``TimeoutError: `` plus a stderr tail -- could not say WHICH call or
    WHICH budget. The method and the budget were both in scope on this path and
    were discarded by a bare ``raise``. The behaviour the path has always had
    (cancel for this exact request, one second of grace, poison, no channel
    reuse) is asserted in the same test so legibility cannot be bought with it.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()
    seen: list[bytes] = []

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    peer = asyncio.create_task(_silent_peer(requests_read, seen))
    params = CloseParams(operation_id="close-ep-fca426e92c42", episode_id="ep-fca426e92c42")
    try:
        with pytest.raises(TimeoutError) as excinfo:
            await client.call("close", params, timeout=0.05)
        message = str(excinfo.value)
        assert "close exceeded its 0.05s budget" in message
        assert "request 1" in message
        # Cheaply available, and the only fact that distinguishes two calls of
        # the same method once the worker's replay cache has re-keyed them.
        assert "operation_id close-ep-fca426e92c42" in message
        # Method and budget lead the line: the readouts that consume the fatal
        # diagnostic truncate it at 110-160 characters.
        assert message.index("close exceeded") < message.index("request 1")
        assert len(message) < 110
        assert len(seen) == 2
        cancel = parse_canonical_line(seen[1], CancelRequest)
        assert isinstance(cancel, CancelRequest) and cancel.id == 1
        assert terminated.is_set()
    finally:
        peer.cancel()
        await asyncio.gather(peer, return_exceptions=True)
        for fd in (requests_read, requests_write, responses_read, responses_write):
            os.close(fd)


@pytest.mark.asyncio
async def test_a_funded_call_reports_the_budget_it_exceeded_not_the_callers_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declaring request's message has to name ITS deadline, not the constant.

    ``funded_timeout`` makes the caller's budget a FLOOR: for ``execute`` (an
    ``ActionBatch`` of waits) and ``cleanup`` (a selected ``CleanupPlan``) the
    deadline is the declaration plus the headroom, and the caller's constant is
    the smaller of the two. So a detail built from that constant names a budget
    the call never ran under -- and the test that pins the message cannot see it
    on any other request, because a request declaring nothing is funded to the
    byte and the two numbers coincide. The close-shaped case is pinned by
    ``test_a_timeout_names_the_call_and_the_budget_it_exceeded``; this is the
    funded one.

    The headroom is monkeypatched, the shape
    ``test_the_derived_budget_is_a_deadline_and_not_a_licence`` uses, because
    at its default 30 s a funded call cannot be made to time out inside a test.
    0.05 s of declared waiting funds to 0.5 s here against a 0.01 s caller's
    budget, so the two candidates are 50x apart and the elapsed the message
    renders separates them by a wide margin rather than a race.
    """

    monkeypatch.setattr(deadlines, "DECLARED_WORK_HEADROOM_S", 0.45)
    declared_s, headroom_s, configured_s = 0.05, 0.45, 0.01
    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    params = _declaring_execute(int(declared_s * 1000))
    peer = asyncio.create_task(_silent_peer(requests_read))
    try:
        with pytest.raises(TimeoutError) as excinfo:
            await client.call("execute", params, timeout=configured_s)
        message = str(excinfo.value)
    finally:
        peer.cancel()
        await asyncio.gather(peer, return_exceptions=True)
        for fd in (requests_read, requests_write, responses_read, responses_write):
            os.close(fd)
    funded_s = declared_s + headroom_s
    assert f"execute exceeded its {funded_s:g}s budget" in message
    # The caller's constant is the only other number in scope, and naming it was
    # the defect: a message carrying both would still tell the reader the wrong
    # budget, so its absence is asserted rather than its position.
    assert f"its {configured_s:g}s budget" not in message
    # THE WAIT IS PINNED ON THE NUMBER THE SENTENCE ITSELF RENDERS, never on a
    # stopwatch out here. Sampling `elapsed` around `client.call` measures the
    # deadline PLUS the timeout branch's 1 s cancel grace, so any floor on it is
    # satisfied by construction: this test used to assert `elapsed >= funded_s`
    # and it PASSED against a mutant whose `wait_for` stayed on the caller's
    # 0.01 s constant while only the detail was funded -- the exact split it is
    # documented to catch. The elapsed field in the message is sampled BEFORE
    # the grace (see `call`), so it is the deadline's own firing time.
    match = re.search(r"after (\d+\.\d)s \(request 1; operation_id exec-declared\)$", message)
    assert match is not None, f"the elapsed field moved out of the message contract: {message!r}"
    rendered_elapsed = float(match.group(1))
    # LOWER bound: the deadline that fired is the funded one, 50x the caller's
    # constant. A `wait_for` left on the constant fires at 0.01 s and renders
    # ~0.0s here, which is what makes this -- and not the budget string, which
    # the mutant keeps funded -- the assertion that catches that wiring.
    assert rendered_elapsed >= funded_s
    # UPPER bound, so the floor is not satisfied by a number that is merely
    # large: the sample is taken before the grace, so the rendered value is the
    # firing time rather than the firing time plus teardown. The 0.5 s allowance
    # is slack for a busy loop, comfortably inside the whole second the grace
    # would add.
    assert rendered_elapsed < funded_s + 0.5
    # Cancel for this request, one second of grace, poison, correlation ids:
    # unchanged by which number the sentence names.
    assert "(request 1; operation_id exec-declared)" in message
    assert terminated.is_set()


def _overhead_only_execute() -> ExecuteParams:
    """An ``execute`` whose ONLY declared duration is the per-action overhead.

    One mutating action and no waits, so the batch declares 0.0 s of its own: the
    one shape that separates "the rate reached ``funded_timeout``" from "the rate
    was dropped on the way". A wait-bearing batch funds the call whether or not
    the rate survives, which is why the episode-level test -- whose fake adapter
    calls ``funded_timeout`` in its own body -- could not see a dropped argument.
    """

    batch = ActionBatch(
        protocol_version="1.0",
        task_id="task",
        episode_id="episode",
        observation_id="obs",
        actions=(ClickAction(observation_id="obs", frame_id="screen", x=1, y=2),),
    )
    return ExecuteParams(
        operation_id="exec-overhead",
        action_batch=batch,
        action_batch_id=canonical_digest("adapter-action-batch-v1", batch),
    )


async def _execute_timeout_message(
    params: ExecuteParams,
    *,
    timeout: float,
    execution_overhead_seconds_per_action: float | None = None,
) -> str:
    """The ``TimeoutError`` sentence from a real ``RpcClient.call`` on ``execute``.

    The peer reads the request and never answers, so the call can only end by
    firing its deadline -- which is what makes the sentence's elapsed field a
    reading of THAT deadline rather than of a reply arriving first.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()

    async def terminate() -> None:
        return None

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    peer = asyncio.create_task(_silent_peer(requests_read))
    try:
        with pytest.raises(TimeoutError) as excinfo:
            if execution_overhead_seconds_per_action is None:
                await client.call("execute", params, timeout=timeout)
            else:
                await client.call(
                    "execute",
                    params,
                    timeout=timeout,
                    execution_overhead_seconds_per_action=(execution_overhead_seconds_per_action),
                )
        message = str(excinfo.value)
    finally:
        # Before any assertion and for every exception: see `_finish_peer`.
        await _finish_peer(peer, requests_read, requests_write, responses_read, responses_write)
    return message


@pytest.mark.asyncio
async def test_the_per_action_overhead_rate_raises_the_execute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rate has to arrive at ``funded_timeout`` from the real RPC call.

    This is the hop the episode-level test cannot see: ``FakeAdapter._call_raw``
    calls ``funded_timeout`` in its own body, so an argument lost anywhere
    between ``VerifiedAdapterSession`` and this method still funds the fake
    deadline and the fake still reports the funded number. Here the call is the
    real one and the deadline is the real ``wait_for``, so the budget the
    sentence names can only have come from the rate threaded through this call.
    The headroom is monkeypatched for the reason its sibling funded test gives:
    at the 30 s default a funded call cannot be made to time out inside a test.
    """

    monkeypatch.setattr(deadlines, "DECLARED_WORK_HEADROOM_S", 0.5)
    rate_s, configured_s = 0.5, 0.01
    message = await _execute_timeout_message(
        _overhead_only_execute(),
        timeout=configured_s,
        execution_overhead_seconds_per_action=rate_s,
    )

    funded_s = rate_s + 0.5
    assert f"execute exceeded its {funded_s:g}s budget" in message
    # The caller's constant is the only other number in scope, and naming it
    # would report a budget the call never ran under.
    assert f"its {configured_s:g}s budget" not in message
    # Pinned on the number the sentence itself renders, never on a stopwatch out
    # here: the timeout path adds its own 1 s cancel grace AFTER ``elapsed`` is
    # sampled, so a floor measured around ``client.call`` is satisfied by
    # construction and would pass against a ``wait_for`` left on the constant.
    match = re.search(r"after (\d+\.\d)s \(request 1; operation_id exec-overhead\)$", message)
    assert match is not None, f"the elapsed field moved out of the message contract: {message!r}"
    rendered_elapsed = float(match.group(1))
    assert rendered_elapsed >= funded_s
    assert rendered_elapsed < funded_s + 0.5


@pytest.mark.asyncio
async def test_the_zero_overhead_default_leaves_the_execute_deadline_unchanged() -> None:
    """The opt-in is inert by default: an unset rate funds nothing.

    Every ordinary caller -- the observation resume, the cleanup call, every
    adapter that is not the paper-settle path -- keeps the caller's constant as
    the whole deadline for a batch that declares no waiting of its own. Asserted
    on the deadline's own firing time rather than only on the sentence: a default
    that quietly funded some nominal rate would raise the rendered elapsed by the
    same order of magnitude the funded case shows.
    """

    configured_s = 0.01
    message = await _execute_timeout_message(_overhead_only_execute(), timeout=configured_s)

    assert f"execute exceeded its {configured_s:g}s budget" in message
    match = re.search(r"after (\d+\.\d)s \(request 1; operation_id exec-overhead\)$", message)
    assert match is not None, f"the elapsed field moved out of the message contract: {message!r}"
    # 0.01 s of deadline, not the 1.0 s a funded 0.5 s-per-action rate would give.
    assert float(match.group(1)) < 0.5


@pytest.mark.asyncio
async def test_the_raised_timeout_stays_a_TimeoutError() -> None:
    """Type is part of the contract: message is the defect, type is not.

    ``_diagnostic_code`` derives the bundle's ``diagnostic_code`` from the
    exception type name, and the campaign's bundles and readouts bucket on
    ``timeouterror``; a subclass would carry the same text while silently
    re-keying every historical comparison. Callers in this repo ``except
    TimeoutError`` as well.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()

    async def terminate() -> None:
        return None

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    peer = asyncio.create_task(_silent_peer(requests_read))
    try:
        with pytest.raises(TimeoutError) as excinfo:
            await client.call("inspect_requirements", InspectRequirementsParams(), timeout=0.05)
        assert type(excinfo.value) is TimeoutError
        assert str(excinfo.value) != ""
    finally:
        peer.cancel()
        await asyncio.gather(peer, return_exceptions=True)
        for fd in (requests_read, requests_write, responses_read, responses_write):
            os.close(fd)


@pytest.mark.asyncio
async def test_a_late_reply_is_never_misattributed_and_the_poison_names_its_cause() -> None:
    """The channel is dead after a timeout, and now says what killed it.

    A late reply arriving after the deadline must be unreadable by a later
    request -- the reason the poison exists -- and the SENTENCE a poisoned
    client raises is often the fatal one in a bundle, because the harness's
    teardown calls (cleanup, close, rescue) are the next to run. Naming only
    "poisoned" left that bundle unable to say which call died or why.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()
    seen: list[bytes] = []

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    peer = asyncio.create_task(_silent_peer(requests_read, seen))
    try:
        with pytest.raises(TimeoutError):
            await client.call("close", CloseParams(operation_id="close-1"), timeout=0.05)
        # The reply the timed-out call was waiting for, written AFTER the
        # deadline: a valid response for request 1 sitting in the pipe.
        assert len(seen) == 2
        stale = parse_canonical_line(seen[0], RpcRequest)
        assert isinstance(stale, RpcRequest)
        IncrementalWriter(responses_write).write(
            canonical_line(
                RpcResponse(
                    jsonrpc="2.0",
                    id=stale.id,
                    method=stale.method,
                    result={"accepted": True},
                )
            )
        )
        with pytest.raises(RpcProtocolError) as excinfo:
            await client.call("observe", ObserveParams(episode_id="ep-fca426e92c42"), timeout=0.05)
        message = str(excinfo.value)
        assert "poisoned by a timeout on close, request 1" in message
        assert "observe was not sent" in message
        assert terminated.is_set()
    finally:
        peer.cancel()
        await asyncio.gather(peer, return_exceptions=True)
        for fd in (requests_read, requests_write, responses_read, responses_write):
            os.close(fd)


@pytest.mark.asyncio
async def test_a_worker_that_dies_before_replying_names_the_in_flight_call() -> None:
    """A channel death is attributed, not just reported.

    This is the parent's view when a worker exits on a protocol fault: it
    answers with a torn channel rather than an error frame, so the read fails.
    The bare ``EOFError`` named neither the method nor the request, which is
    why the sibling rescue path in ``worker.py`` raised an adapter error
    instead of a protocol error to avoid it.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)

    async def dying_peer() -> None:
        await asyncio.to_thread(IncrementalReader(requests_read).read_line)
        os.close(responses_write)

    peer = asyncio.create_task(dying_peer())
    try:
        with pytest.raises(EOFError) as excinfo:
            await client.call("observe", ObserveParams(episode_id="episode"), timeout=5)
        assert "closed the channel before replying to observe (request 1)" in str(excinfo.value)
        await peer
        assert terminated.is_set()
    finally:
        for fd in (requests_read, requests_write, responses_read):
            os.close(fd)


@pytest.mark.asyncio
async def test_a_malformed_reply_names_both_the_cause_and_the_call() -> None:
    """The parse already named the cause; the attribution was missing.

    "RPC message is malformed" in a bundle of a hundred calls does not say
    which reply was malformed, so the operator cannot tell a worker that is
    emitting junk from one that is emitting junk for ONE method.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)

    async def malformed_peer() -> None:
        await asyncio.to_thread(IncrementalReader(requests_read).read_line)
        IncrementalWriter(responses_write).write(b"not json at all\n")

    peer = asyncio.create_task(malformed_peer())
    try:
        with pytest.raises(RpcProtocolError) as excinfo:
            await client.call("observe", ObserveParams(episode_id="episode"), timeout=5)
        message = str(excinfo.value)
        assert "malformed" in message
        assert "while reading the reply to observe, request 1" in message
        await peer
        assert terminated.is_set()
    finally:
        for fd in (requests_read, requests_write, responses_read, responses_write):
            os.close(fd)


@pytest.mark.asyncio
async def test_a_write_to_a_dead_worker_names_the_call_that_was_not_sent() -> None:
    """The WRITE side of a channel death, the mirror of the read-side fix.

    ``supervisor.launch`` closes the parent's own copies of the pipes right
    after spawn, and ``process.poll()`` is only consulted inside
    ``terminate()``, so a worker that died between steps is normally first
    observed HERE -- as EPIPE on the next write. That raise used to reach the
    artifact as ``BrokenPipeError: [Errno 32] Broken pipe``: no method and no
    request, on the most likely first observation of a dead worker, which made
    it the widest hole left in the parent-to-worker audit.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    # The worker is gone: this write end has no reader left anywhere.
    os.close(requests_read)
    try:
        with pytest.raises(BrokenPipeError) as excinfo:
            await client.call("observe", ObserveParams(episode_id="episode"), timeout=5)
        error = excinfo.value
        # The TYPE is what ``episode._diagnostic_code`` derives the bundle's
        # diagnostic_code from, so the bucket an operator greps for is
        # unchanged, and the errno is the only part of the original message
        # that named anything at all.
        assert type(error) is BrokenPipeError
        assert error.errno == errno.EPIPE
        message = str(error)
        assert message.startswith("[Errno 32]")
        assert "observe was not sent: the adapter worker closed the channel (request 1)" in message
        # Method and request id are readable inside the narrowest readout width.
        assert f"{type(error).__name__}: {message}".index("observe was not sent") < 110
        assert "\n" not in message
        assert terminated.is_set()
        # The death still poisons: a later call must not reuse the channel, and
        # the poison names what killed it and the call it did not send.
        with pytest.raises(RpcProtocolError) as followup:
            await client.call("close", CloseParams(operation_id="close-1"), timeout=5)
        assert "poisoned by BrokenPipeError on observe, request 1" in str(followup.value)
        assert "close was not sent" in str(followup.value)
    finally:
        for fd in (requests_write, responses_read, responses_write):
            os.close(fd)


@pytest.mark.asyncio
async def test_a_cancel_that_cannot_be_delivered_still_names_the_timeout() -> None:
    """The timeout branch writes too, and its worker can be gone by then.

    A worker that outran its budget is exactly the worker that may have died
    while it was busy, so the cancel frame can fail on EPIPE. The fatal then
    has to keep the timeout detail (the truth about this call, and what the
    readouts bucket on) while still naming the death -- and it must keep the
    write's own type, so which bucket records the episode does not move.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)

    async def dying_peer() -> None:
        await asyncio.to_thread(IncrementalReader(requests_read).read_line)
        # It took the request and then died: nothing will ever read the cancel.
        os.close(requests_read)

    peer = asyncio.create_task(dying_peer())
    try:
        with pytest.raises(BrokenPipeError) as excinfo:
            await client.call("close", CloseParams(operation_id="close-1"), timeout=0.25)
        error = excinfo.value
        assert type(error) is BrokenPipeError
        message = str(error)
        assert "close exceeded its 0.25s budget after" in message
        assert "(request 1; operation_id close-1)" in message
        assert "the cancel was not delivered" in message
        # The timeout detail leads for the same truncation reason as the
        # timeout-only message it replaces on this path.
        assert message.index("exceeded its 0.25s budget") < message.index("the cancel")
        assert terminated.is_set()
        await peer
        # The poison is still the timeout: the call was never answered, so a
        # later call must not read the reply the worker may still produce.
        with pytest.raises(RpcProtocolError) as followup:
            await client.call("observe", ObserveParams(episode_id="episode"), timeout=0.25)
        assert "poisoned by a timeout on close, request 1" in str(followup.value)
    finally:
        # ``requests_read`` is deliberately absent: the peer closed it, and
        # closing it twice raises out of the teardown that is running anyway.
        for fd in (requests_write, responses_read, responses_write):
            os.close(fd)


@pytest.mark.asyncio
async def test_a_funded_call_whose_cancel_cannot_be_delivered_names_its_own_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other `_timeout_detail` call site, with a request that declares work.

    The cancel-not-delivered arm is a SECOND site the reconciliation moved onto
    ``effective_budget``, and no test saw the difference: the only test that
    reaches it drives ``close``, whose request declares nothing, so
    ``funded_timeout`` returns the caller's constant there and a call site wired
    to that constant stays green. It is also the arm an operator meets most
    often, because a worker that died between taking the request and the
    deadline firing is exactly the worker that cannot read the cancel.

    Type and errno are asserted for the same reason as the close-shaped test:
    the detail is prepended to a write error whose TYPE is the bundle's
    diagnostic bucket (``brokenpipeerror``), so this arm cannot buy legibility
    with either.
    """

    monkeypatch.setattr(deadlines, "DECLARED_WORK_HEADROOM_S", 0.45)
    declared_s, headroom_s, configured_s = 0.05, 0.45, 0.01
    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    params = _declaring_execute(int(declared_s * 1000))

    async def dying_peer() -> None:
        await asyncio.to_thread(IncrementalReader(requests_read).read_line)
        # Took the request and then died: nothing will ever read the cancel.
        os.close(requests_read)

    peer = asyncio.create_task(dying_peer())
    funded_s = declared_s + headroom_s
    try:
        with pytest.raises(BrokenPipeError) as excinfo:
            await client.call("execute", params, timeout=configured_s)
        error = excinfo.value
        assert type(error) is BrokenPipeError
        assert error.errno == errno.EPIPE
        message = str(error)
        # The FUNDED number, which is the only number that distinguishes this
        # arm from its close-shaped sibling: the caller's 0.01 s constant is
        # absent for the same reason it is absent on the main raise.
        assert f"execute exceeded its {funded_s:g}s budget after" in message
        assert f"its {configured_s:g}s budget" not in message
        assert "(request 1; operation_id exec-declared)" in message
        assert "the cancel was not delivered" in message
        # The timeout detail still leads for the same truncation reason as the
        # timeout-only message it replaces on this path.
        assert message.index(f"exceeded its {funded_s:g}s budget") < message.index("the cancel")
        assert terminated.is_set()
        await peer
        # The poison is still the timeout: the call was never answered, so a
        # later call must not read the reply the worker may still produce.
        with pytest.raises(RpcProtocolError) as followup:
            await client.call("observe", ObserveParams(episode_id="episode"), timeout=configured_s)
        assert "poisoned by a timeout on execute, request 1" in str(followup.value)
    finally:
        # ``requests_read`` is deliberately absent: the peer closed it, and
        # closing it twice raises out of the teardown that is running anyway.
        for fd in (requests_write, responses_read, responses_write):
            os.close(fd)


@pytest.mark.asyncio
async def test_an_oversize_request_names_the_call_that_could_not_be_sent() -> None:
    """The request that cannot be framed is the same write that never happens.

    ``MAX_TEXT_LENGTH`` per action times ``MAX_BATCH_SIZE`` is far past
    ``MAX_RPC_BYTES``, so a LEGAL batch can exceed the frame limit and end the
    episode on a channel-killing protocol error. "RPC message exceeds one MiB"
    alone leaves the operator unable to say which call asked for it, which is
    the read side's own standard (``_read_response``) applied one line earlier.
    """

    requests_read, requests_write = os.pipe()
    responses_read, responses_write = os.pipe()
    terminated = asyncio.Event()

    async def terminate() -> None:
        terminated.set()

    client = RpcClient(requests_write, responses_read, terminate=terminate)
    batch = ActionBatch(
        protocol_version="1.0",
        task_id="task",
        episode_id="episode",
        observation_id="obs-1",
        actions=tuple(
            TypeAction(observation_id="obs-1", text="x" * MAX_TEXT_LENGTH) for _ in range(12)
        ),
    )
    params = ExecuteParams(
        operation_id="exec-1",
        action_batch=batch,
        action_batch_id=canonical_digest("adapter-action-batch-v1", batch),
    )
    try:
        # The limit is the real one and the batch is a real legal batch: the
        # frame cannot be built, which is what makes this path reachable.
        assert len(params.to_canonical_json()) > MAX_RPC_BYTES
        with pytest.raises(RpcProtocolError) as excinfo:
            await client.call("execute", params, timeout=5)
        message = str(excinfo.value)
        assert "RPC message exceeds one MiB" in message
        assert "(while sending execute, request 1)" in message
        assert terminated.is_set()
        with pytest.raises(RpcProtocolError) as followup:
            await client.call("observe", ObserveParams(episode_id="episode"), timeout=5)
        assert "poisoned by RpcProtocolError on execute, request 1" in str(followup.value)
    finally:
        for fd in (requests_read, requests_write, responses_read, responses_write):
            os.close(fd)


def test_the_longest_legal_budget_renders_in_plain_units() -> None:
    """The budget is a number an operator reads, at every magnitude the protocol admits.

    ``:g`` emits six significant digits, so the sentence's own budget leaves
    fixed notation from 1e6 s -- nine maximal cleanup actions -- and the
    protocol's ceiling is 256 of them. Both magnitudes are asserted, because
    "reachable in principle" is what a legibility NIT is called when nobody
    computes the top: nine actions is the first crossing, and 29_491_230 s is
    the largest deadline a legal call can declare. The band BELOW the crossing
    is asserted byte-identical in the same test: those strings are what the
    campaign's readouts quote and what the other tests in this file assert
    (0.05s/0.25s for ``close``, 180s/0.5s/31.5s for ``execute`` and
    ``cleanup``), so a fix that re-rendered them would churn approved evidence.
    The crossing is also asserted FROM BELOW, at values that already render an
    exponent while sitting under it, because that is the whole reason the guard
    reads the rendered text: a magnitude comparison against
    ``_EXPONENT_FORM_AT_S`` leaves those in exponent form and passes every
    other assertion here.
    """

    from local_operator.evaluation.adapters.rpc import (
        _EXPONENT_FORM_AT_S,
        _rendered_budget,
        _timeout_detail,
    )
    from local_operator.evaluation.deadlines import DECLARED_WORK_HEADROOM_S
    from local_operator.evaluation.lifecycle import (
        MAX_CLEANUP_ATTEMPTS,
        MAX_CLEANUP_TIMEOUT_MS,
    )
    from local_operator.evaluation.receipts import MAX_DECLARATIONS

    # The band the renderer's own comment states, checked rather than trusted:
    # fixed point one second below it, exponent form at it.
    assert f"{_EXPONENT_FORM_AT_S - 1.0:g}" == "999999"
    assert f"{_EXPONENT_FORM_AT_S:g}" == "1e+06"

    # The crossing seen from BELOW, which is the case the guard's rendered-text
    # comment exists for and the one a magnitude comparison slips through.
    # ``:g`` rounds to six significant digits before it decides, so these render
    # an exponent while sitting under ``_EXPONENT_FORM_AT_S``: a guard written
    # as ``if timeout < _EXPONENT_FORM_AT_S`` -- the edit
    # ``_rendered_budget``'s comment forbids -- returns the exponent form for
    # every one of them and still passes the two magnitudes above and the band
    # below. Both halves are asserted, so the failure names which half moved.
    for value, expected in (
        (999_999.5, "999999.5"),
        (999_999.9, "999999.9"),
        (999_999.99, "999999.99"),
        (999_999.999, "999999.999"),
        (999_999.999999, "999999.999999"),
    ):
        assert f"{value:g}" == "1e+06"
        assert value < _EXPONENT_FORM_AT_S
        assert _rendered_budget(value) == expected

    # One maximal cleanup action -- an hour, 32 attempts -- from the protocol's
    # own bounds rather than a number chosen here.
    per_action = MAX_CLEANUP_TIMEOUT_MS / 1000.0 * MAX_CLEANUP_ATTEMPTS
    assert per_action == 115_200.0
    nine_actions = 9 * per_action + DECLARED_WORK_HEADROOM_S
    assert nine_actions == 1_036_830.0
    assert f"{nine_actions:g}" == "1.03683e+06"
    assert _rendered_budget(nine_actions) == "1036830"
    maximal = MAX_DECLARATIONS * per_action + DECLARED_WORK_HEADROOM_S
    assert maximal == 29_491_230.0
    assert _rendered_budget(maximal) == "29491230"

    # The sentence an artifact carries, not just the field: the readouts fold
    # this whole line into the failure.
    detail = _timeout_detail(
        "cleanup",
        timeout=maximal,
        elapsed=1_036_831.0,
        request_id=42,
        operation_id="cleanup-9f2c1a4b8d3e",
    )
    assert "cleanup exceeded its 29491230s budget after 1036831.0s" in detail

    # The untouched band, at the values the existing evidence quotes.
    for value, expected in (
        (0.05, "0.05"),
        (0.25, "0.25"),
        (180.0, "180"),
        (0.5, "0.5"),
        (31.5, "31.5"),
    ):
        assert _rendered_budget(value) == expected


def test_error_detail_stays_within_the_line_framing_and_bounds() -> None:
    """Detail text must never break the transport that carries it.

    The framing is LF-delimited and rejects CR, so an adapter message holding
    either would turn an answered adapter error into a channel-killing protocol
    error -- failing loudest on precisely the path that exists to explain a
    failure. Bounds are asserted alongside because an unbounded field would let
    a worker's exception text decide the parent's allocation.
    """

    from local_operator.evaluation.adapters.rpc import (
        MAX_DETAIL_MESSAGE,
        RpcError,
        RpcErrorDetail,
        canonical_line,
    )
    from local_operator.evaluation.adapters.worker import _control_safe

    hostile = "line one\r\nline two\ttabbed\x00null " + "z" * 4000
    cleaned = _control_safe(hostile, MAX_DETAIL_MESSAGE)
    # The VALUE carries no control characters. Asserted on the string rather
    # than on the encoded line because JSON escapes CR/LF into a safe ``\r\n``
    # two-byte form -- so a framing-only assertion passes even when the value
    # is dirty, and a reader of the artifact would get the raw newlines back.
    assert not any(character in cleaned for character in "\r\n\t\x00")
    assert all(character.isprintable() or character == " " for character in cleaned)
    assert len(cleaned) <= MAX_DETAIL_MESSAGE
    detail = RpcErrorDetail(
        exception_type="RuntimeError",
        message=cleaned,
        method="execute",
        operation_id="exec-1",
    )
    line = canonical_line(
        RpcError(code="adapter_error", message="adapter operation failed", detail=detail)
    )
    assert line.endswith(b"\n") and line.count(b"\n") == 1 and b"\r" not in line


def test_unbuildable_canary_set_withholds_rather_than_assuming_no_secrets() -> None:
    """A secret that cannot be canaried must fail CLOSED, never open.

    Leaving the redaction set None after secrets were delivered would read
    identically to "this worker holds none" and skip the check on the one
    occasion something is definitely there to protect.
    """

    from local_operator.evaluation.adapters.rpc import MAX_DETAIL_MESSAGE, WITHHELD
    from local_operator.evaluation.adapters.worker import _DENY_ALL, _redacted

    assert _redacted("anything at all", MAX_DETAIL_MESSAGE, _DENY_ALL) == WITHHELD
    # No secrets delivered at all: the text is kept, because none can leak.
    assert _redacted("plain text", MAX_DETAIL_MESSAGE, None) == "plain text"


def test_a_secret_straddling_the_truncation_boundary_is_still_withheld() -> None:
    """Scan the UNBOUNDED value: truncating first severs the canary.

    ``RedactionSet.assert_clear`` is a substring check, so a secret cut by the
    field bound stops matching and its surviving prefix is returned verbatim.
    Round 1's F1: a 40-character AWS key positioned across the 512-character
    message cut emitted 25 characters of itself, and an 808-character JWT
    emitted 488. It is systematic -- ANY secret longer than its field bound can
    never match once cut -- and the parent's artifact scan cannot catch it
    either, because it applies the same substring semantics to the already
    truncated bytes.

    Both field bounds are covered because they truncate at different lengths,
    and the JWT case additionally pins a secret LONGER than the bound, which is
    the shape that can never match after cutting.
    """

    from local_operator.evaluation.adapters.rpc import (
        MAX_DETAIL_MESSAGE,
        MAX_DETAIL_NAME,
        WITHHELD,
    )
    from local_operator.evaluation.adapters.worker import _redacted
    from local_operator.evaluation.receipts import RedactionSet

    key = "AKIAIOSFODNN7EXAMPLE" + "QWERTYUIOPASDFGH1234"
    assert len(key) == 40
    jwt = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9." + "a" * 400 + "." + "b" * 370
    assert len(jwt) > MAX_DETAIL_MESSAGE

    for secret, limit in (
        (key, MAX_DETAIL_MESSAGE),
        (key, MAX_DETAIL_NAME),
        (jwt, MAX_DETAIL_MESSAGE),
    ):
        redactions = RedactionSet.from_resolved_values((secret,))
        # Place the secret so the cut lands INSIDE it rather than before it.
        filler = "adapter failed: " + "x" * max(limit - 16 - len(secret) // 2, 0)
        result = _redacted(filler + secret + " trailing context", limit, redactions)
        assert result == WITHHELD
        # No contiguous run of the secret survives -- the assertion that fails
        # if the scan is ever moved back after the truncation.
        assert not any(
            secret[:length] in result for length in range(8, len(secret) + 1)
        ), "a prefix of the secret survived truncation"
