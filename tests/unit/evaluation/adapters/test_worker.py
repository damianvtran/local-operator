from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import (
    AckResult,
    AdapterCapabilities,
    AdapterMetadata,
    AdapterSelector,
    AdapterState,
    BeginRescueParams,
    CleanupOutcome,
    CleanupParams,
    CleanupResult,
    Handshake,
    HelloParams,
    PrepareParams,
    PrepareResult,
    PythonRuntime,
    RescueDescriptor,
)
from local_operator.evaluation.adapters.discovery import workspace_digest
from local_operator.evaluation.adapters.rpc import (
    IncrementalReader,
    IncrementalWriter,
    RpcProtocolError,
    RpcRequest,
    RpcResponse,
    canonical_line,
    parse_canonical_line,
)
from local_operator.evaluation.adapters.worker import Worker
from local_operator.evaluation.evidence.models import canonical_digest
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


class RescueAdapter:
    def __init__(self, metadata: AdapterMetadata) -> None:
        self.metadata = metadata
        self.cleanup_calls: list[str] = []
        self.begin_rescue_calls: list[BeginRescueParams] = []

    async def begin_rescue(self, params: BeginRescueParams) -> AckResult:
        # Recorded so the worker test can assert the handoff actually reached
        # adapter code. A rescue worker builds its teardown provider here and
        # nowhere else, so an adapter that is never called can only report
        # "could not look" for every action.
        self.begin_rescue_calls.append(params)
        return AckResult()

    async def cleanup(self, params: CleanupParams) -> CleanupResult:
        action_id = params.action_ids[0]
        self.cleanup_calls.append(action_id)
        return CleanupResult(
            outcomes=(
                CleanupOutcome(
                    action_id=action_id,
                    status="succeeded",
                    evidence_code="released",
                    duration_ms=1,
                ),
            )
        )


def rescue_selector(tmp_path: Path) -> AdapterSelector:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    release_digest = "b" * 64
    (workspace / "adapter-release.json").write_text(f'{{"release_digest":"{release_digest}"}}')
    return AdapterSelector(
        schema_version="1.4",
        adapter_id="rescue",
        distribution="rescue-adapter",
        version="1.0",
        entry_point="rescue_adapter:create",
        package_digest="a" * 64,
        release_digest=release_digest,
        python_executable=str(Path(sys.executable).resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )


def rescue_metadata() -> AdapterMetadata:
    return AdapterMetadata(
        adapter_id="rescue",
        distribution="rescue-adapter",
        version="1.0",
        entry_point="rescue_adapter:create",
        package_digest="a" * 64,
        release_digest="b" * 64,
        schema_version="1.4",
        capabilities=AdapterCapabilities(routes=("computer",), ask_user=False, scoring=False),
    )


@pytest.mark.asyncio
async def test_real_worker_rescue_invokes_each_cleanup_once_and_blocks_normal_flow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selected = rescue_selector(tmp_path)
    adapter = RescueAdapter(rescue_metadata())
    monkeypatch.setattr(
        "local_operator.evaluation.adapters.worker.load_selected_adapter", lambda _: adapter
    )
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    worker = Worker(request_read, response_write)
    task = asyncio.create_task(worker.run())
    writer = IncrementalWriter(request_write)
    reader = IncrementalReader(response_read)
    try:
        hello = await exchange(
            writer,
            reader,
            RpcRequest(
                jsonrpc="2.0",
                id=1,
                method="hello",
                params=HelloParams(selector=selected).model_dump(mode="json"),
            ),
        )
        assert hello.result is not None
        handshake = Handshake.model_validate(hello.result, strict=True)
        cleanup_plan = CleanupPlan(
            episode_id="episode",
            actions=tuple(
                CleanupAction(
                    action_id=action_id,
                    kind="release_instance",
                    resource_ref=f"resource-{action_id}",
                    timeout_ms=100,
                    max_attempts=1,
                )
                for action_id in ("one", "two")
            ),
        )
        descriptor = RescueDescriptor(
            schema_version="1.4",
            selector=selected,
            handshake=handshake,
            episode_id="episode",
            cleanup_plan=cleanup_plan,
            secret_refs=(),
            infra_values=(),
            artifact_root=str(tmp_path),
        )
        begin = BeginRescueParams(
            operation_id="begin",
            descriptor=descriptor,
            descriptor_id=descriptor.descriptor_id,
            episode_id=descriptor.episode_id,
            cleanup_plan_id=cleanup_plan.cleanup_plan_id,
            selector_digest=canonical_digest("adapter-rescue-selector-v1", selected),
            handshake_digest=canonical_digest("adapter-rescue-handshake-v1", handshake),
        )
        response = await exchange(
            writer,
            reader,
            RpcRequest(
                jsonrpc="2.0",
                id=2,
                method="begin_rescue",
                params=begin.model_dump(mode="json"),
            ),
        )
        assert response.result == AckResult().model_dump(mode="json")
        first_cleanup: CleanupParams | None = None
        for request_id, action in enumerate(cleanup_plan.actions, start=3):
            params = CleanupParams(
                operation_id=f"cleanup-{action.action_id}",
                cleanup_plan=cleanup_plan,
                action_ids=(action.action_id,),
            )
            first_cleanup = first_cleanup or params
            response = await exchange(
                writer,
                reader,
                RpcRequest(
                    jsonrpc="2.0",
                    id=request_id,
                    method="cleanup",
                    params=params.model_dump(mode="json"),
                ),
            )
            assert response.error is None
        assert first_cleanup is not None
        duplicate = first_cleanup.model_copy(update={"operation_id": "second-key-same-action"})
        response = await exchange(
            writer,
            reader,
            RpcRequest(
                jsonrpc="2.0",
                id=5,
                method="cleanup",
                params=duplicate.model_dump(mode="json"),
            ),
        )
        assert response.error is None
        alias_replay = await exchange(
            writer,
            reader,
            RpcRequest(
                jsonrpc="2.0",
                id=6,
                method="cleanup",
                params=duplicate.model_dump(mode="json"),
            ),
        )
        assert alias_replay.result == response.result
        conflicting_alias = CleanupParams(
            operation_id="second-key-same-action",
            cleanup_plan=cleanup_plan,
            action_ids=("two",),
        )
        writer.write(
            canonical_line(
                RpcRequest(
                    jsonrpc="2.0",
                    id=7,
                    method="cleanup",
                    params=conflicting_alias.model_dump(mode="json"),
                )
            )
        )
        os.close(request_write)
        request_write = -1
        assert await asyncio.wait_for(task, 1) == 70
        assert adapter.cleanup_calls == ["one", "two"]
    finally:
        if request_write >= 0:
            os.close(request_write)
        for fd in (request_read, response_read, response_write):
            os.close(fd)


def test_rescue_cleanup_rejects_forged_episode_and_action_without_losing_pin(
    tmp_path: Path,
) -> None:
    selected = rescue_selector(tmp_path)
    cleanup_plan = CleanupPlan(
        episode_id="episode",
        actions=(
            CleanupAction(
                action_id="one",
                kind="release_instance",
                resource_ref="resource-one",
                timeout_ms=100,
                max_attempts=1,
            ),
        ),
    )
    pinned_handshake = Handshake(
        selector=selected,
        metadata=rescue_metadata(),
        python=PythonRuntime.current(),
        workspace_digest=selected.workspace_digest,
        selected_route="computer",
    )
    descriptor = RescueDescriptor(
        schema_version="1.4",
        selector=selected,
        handshake=pinned_handshake,
        episode_id="episode",
        cleanup_plan=cleanup_plan,
        secret_refs=(),
        infra_values=(),
        artifact_root=str(tmp_path),
    )
    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    worker = Worker(request_read, response_write)
    worker._rescue_descriptor = descriptor
    try:
        forged_episode = CleanupPlan(episode_id="episode-b", actions=cleanup_plan.actions)
        with pytest.raises(RpcProtocolError, match="pinned rescue plan"):
            worker._validate_rescue_cleanup(
                CleanupParams(
                    operation_id="forged-episode",
                    cleanup_plan=forged_episode,
                    action_ids=("one",),
                )
            )
        modified_action = CleanupPlan(
            episode_id="episode",
            actions=(
                CleanupAction(
                    action_id="one",
                    kind="release_instance",
                    resource_ref="other-resource",
                    timeout_ms=100,
                    max_attempts=1,
                ),
            ),
        )
        with pytest.raises(RpcProtocolError, match="pinned rescue plan"):
            worker._validate_rescue_cleanup(
                CleanupParams(
                    operation_id="forged-action",
                    cleanup_plan=modified_action,
                    action_ids=("one",),
                )
            )
        assert worker._rescue_descriptor == descriptor
    finally:
        for fd in (request_read, request_write, response_read, response_write):
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


def test_error_phase_is_declared_by_the_adapter_and_only_for_execute() -> None:
    """The worker REPORTS a declared phase; it never infers one.

    The parent makes a safety decision on this field -- whether a failed
    mutating call may keep its session -- so the two ways to get it wrong are
    pinned here:

    * A phase is stamped only for an ``ObservationPhaseError``. An ordinary
      adapter exception, however much its text looks like a screenshot
      failure, stays ``unknown`` and keeps poisoning.
    * Only ``execute`` may claim it. The recovery path re-reads state after a
      committed step; there is no equivalent for ``prepare`` or ``cleanup``,
      so a claim from one of those is dropped rather than honoured.
    """

    from local_operator.evaluation.adapters.api import ObservationPhaseError
    from local_operator.evaluation.adapters.worker import _error_detail

    declared = _error_detail(
        ObservationPhaseError("environment returned no screenshot frame"),
        "execute",
        "exec-0",
        None,
    )
    assert declared is not None and declared.phase == "observation"
    # It renders, so a bundle reader can see why the harness retried.
    assert "phase=observation" in declared.render()

    # An undeclared failure whose message names the same symptom.
    undeclared = _error_detail(
        RuntimeError("environment returned no screenshot frame"),
        "execute",
        "exec-0",
        None,
    )
    assert undeclared is not None and undeclared.phase == "unknown"
    assert "phase=" not in undeclared.render()

    # The same declared exception raised from a method that cannot be resumed.
    for method in ("prepare", "reset_start", "score", "cleanup"):
        wrong_method = _error_detail(
            ObservationPhaseError("no screenshot"),
            method,  # pyright: ignore[reportArgumentType]
            "op-0",
            None,
        )
        assert wrong_method is not None
        assert wrong_method.phase == "unknown", method
