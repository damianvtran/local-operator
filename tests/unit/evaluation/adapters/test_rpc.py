from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Any

import pytest

from local_operator.evaluation import deadlines
from local_operator.evaluation.adapters.api import (
    ExecuteParams,
    InspectRequirementsParams,
)
from local_operator.evaluation.adapters.rpc import (
    MAX_RPC_BYTES,
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
from local_operator.evaluation.protocol import ActionBatch


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
    with pytest.raises(RpcProtocolError, match="ID or method"):
        await client.call("inspect_requirements", InspectRequirementsParams(), timeout=1)
    await task
    assert terminated.is_set()
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
