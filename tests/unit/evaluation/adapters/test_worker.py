from __future__ import annotations

import asyncio
import os

import pytest

from local_operator.evaluation.adapters.rpc import (
    IncrementalReader,
    IncrementalWriter,
    RpcRequest,
    RpcResponse,
    canonical_line,
    parse_canonical_line,
)
from local_operator.evaluation.adapters.worker import Worker


async def exchange(
    writer: IncrementalWriter,
    reader: IncrementalReader,
    request: RpcRequest,
) -> RpcResponse:
    writer.write(canonical_line(request))
    raw = await asyncio.to_thread(reader.read_line)
    parsed = parse_canonical_line(raw, RpcResponse)
    assert isinstance(parsed, RpcResponse)
    return parsed


@pytest.mark.asyncio
async def test_worker_rejects_invalid_state_and_changed_request_reuse() -> None:
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    worker = Worker(request_read, response_write)
    task = asyncio.create_task(worker.run())
    writer = IncrementalWriter(request_write)
    reader = IncrementalReader(response_read)
    try:
        request = RpcRequest(jsonrpc="2.0", id=1, method="inspect_requirements", params={})
        response = await exchange(writer, reader, request)
        assert response.error is not None and response.error.code == "invalid_state"
        # Exact request replay is byte-for-byte stable.
        replay = await exchange(writer, reader, request)
        assert replay == response
        # Changed method/params under the same ID is a fatal protocol fault.
        changed = RpcRequest(
            jsonrpc="2.0",
            id=1,
            method="observe",
            params={"episode_id": "episode"},
        )
        writer.write(canonical_line(changed))
        # Protocol-fault exit abandons the blocking reader with the process; the
        # in-process test closes its peer so asyncio can join that executor call.
        os.close(request_write)
        request_write = -1
        assert await asyncio.wait_for(task, 1) == 70
    finally:
        for fd in (request_read, request_write, response_read, response_write):
            try:
                if fd >= 0:
                    os.close(fd)
            except OSError:
                pass


@pytest.mark.asyncio
async def test_worker_rejects_cancel_without_inflight_call() -> None:
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    worker = Worker(request_read, response_write)
    task = asyncio.create_task(worker.run())
    try:
        os.write(request_write, b'{"control":"cancel","id":1,"jsonrpc":"2.0"}\n')
        assert await asyncio.wait_for(task, 1) == 70
    finally:
        for fd in (request_read, request_write, response_read, response_write):
            try:
                if fd >= 0:
                    os.close(fd)
            except OSError:
                pass
