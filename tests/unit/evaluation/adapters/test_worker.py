from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import (
    AdapterState,
    PrepareParams,
    PrepareResult,
)
from local_operator.evaluation.adapters.rpc import (
    IncrementalReader,
    IncrementalWriter,
    RpcRequest,
    RpcResponse,
    canonical_line,
    parse_canonical_line,
)
from local_operator.evaluation.adapters.worker import Worker
from local_operator.evaluation.lifecycle import CleanupAction, CleanupPlan


class CountingWorker(Worker):
    def __init__(self, request_fd: int, response_fd: int, **kwargs: int) -> None:
        super().__init__(request_fd, response_fd, **kwargs)
        self.calls = 0
        self.fail = False

    def set_state(self, state: AdapterState) -> None:
        self._state = state

    async def _dispatch(self, method: Any, params: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise RuntimeError("secret adapter detail")
        assert method == "prepare" and isinstance(params, PrepareParams)
        return PrepareResult(
            cleanup_plan=CleanupPlan(
                episode_id=params.episode_id,
                actions=(
                    CleanupAction(
                        action_id="release",
                        kind="release_instance",
                        resource_ref="resource",
                        timeout_ms=100,
                        max_attempts=1,
                    ),
                ),
            )
        )


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
async def test_keyed_success_replays_under_new_request_id_without_dispatch() -> None:
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    worker = CountingWorker(request_read, response_write)
    worker.set_state("INSPECTED")
    task = asyncio.create_task(worker.run())
    writer = IncrementalWriter(request_write)
    reader = IncrementalReader(response_read)
    params = PrepareParams(
        operation_id="prepare-op", episode_id="episode", secret_refs=(), infra_values=()
    )
    try:
        first = await exchange(
            writer,
            reader,
            RpcRequest(
                jsonrpc="2.0", id=1, method="prepare", params=params.model_dump(mode="json")
            ),
        )
        replay = await exchange(
            writer,
            reader,
            RpcRequest(
                jsonrpc="2.0", id=2, method="prepare", params=params.model_dump(mode="json")
            ),
        )
        assert worker.calls == 1
        assert first.id == 1 and replay.id == 2
        assert first.result == replay.result
    finally:
        os.close(request_write)
        assert await asyncio.wait_for(task, 1) == 0
        for fd in (request_read, response_read, response_write):
            os.close(fd)


@pytest.mark.asyncio
async def test_keyed_error_replays_and_changed_params_poison() -> None:
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    worker = CountingWorker(request_read, response_write)
    worker.set_state("INSPECTED")
    worker.fail = True
    task = asyncio.create_task(worker.run())
    writer = IncrementalWriter(request_write)
    reader = IncrementalReader(response_read)
    params = PrepareParams(
        operation_id="prepare-op", episode_id="episode", secret_refs=(), infra_values=()
    )
    try:
        first = await exchange(
            writer,
            reader,
            RpcRequest(
                jsonrpc="2.0", id=1, method="prepare", params=params.model_dump(mode="json")
            ),
        )
        replay = await exchange(
            writer,
            reader,
            RpcRequest(
                jsonrpc="2.0", id=2, method="prepare", params=params.model_dump(mode="json")
            ),
        )
        assert worker.calls == 1
        assert first.error == replay.error
        changed = params.model_copy(update={"episode_id": "other"})
        writer.write(
            canonical_line(
                RpcRequest(
                    jsonrpc="2.0",
                    id=3,
                    method="prepare",
                    params=changed.model_dump(mode="json"),
                )
            )
        )
        os.close(request_write)
        request_write = -1
        assert await asyncio.wait_for(task, 1) == 70
        assert worker.calls == 1
    finally:
        for fd in (request_read, request_write, response_read, response_write):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
async def test_operation_capacity_poison_precedes_dispatch() -> None:
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    worker = CountingWorker(request_read, response_write, max_operation_records=1)
    worker.set_state("INSPECTED")
    task = asyncio.create_task(worker.run())
    writer = IncrementalWriter(request_write)
    reader = IncrementalReader(response_read)
    try:
        first = PrepareParams(
            operation_id="one", episode_id="episode", secret_refs=(), infra_values=()
        )
        await exchange(
            writer,
            reader,
            RpcRequest(jsonrpc="2.0", id=1, method="prepare", params=first.model_dump(mode="json")),
        )
        worker.set_state("INSPECTED")
        second = first.model_copy(update={"operation_id": "two"})
        writer.write(
            canonical_line(
                RpcRequest(
                    jsonrpc="2.0", id=2, method="prepare", params=second.model_dump(mode="json")
                )
            )
        )
        os.close(request_write)
        request_write = -1
        assert await asyncio.wait_for(task, 1) == 70
        assert worker.calls == 1
    finally:
        for fd in (request_read, request_write, response_read, response_write):
            if fd >= 0:
                os.close(fd)


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
