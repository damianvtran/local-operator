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
