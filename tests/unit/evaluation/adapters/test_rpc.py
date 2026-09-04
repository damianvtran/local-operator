from __future__ import annotations

import asyncio
import os

import pytest

from local_operator.evaluation.adapters.api import InspectRequirementsParams
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
