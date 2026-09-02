"""Isolated adapter worker speaking only over inherited protocol descriptors."""

from __future__ import annotations

import asyncio
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

from local_operator.evaluation.adapters.api import (
    KEYED_METHODS,
    METHOD_NEXT_STATE,
    METHOD_STATES,
    PARAM_MODELS,
    RESULT_MODELS,
    AckResult,
    AdapterMethod,
    AdapterSelector,
    AdapterState,
    BeginRescueParams,
    CleanupParams,
    EvaluationAdapter,
    Handshake,
    HelloParams,
    PythonRuntime,
    RescueDescriptor,
    canonical_params_digest,
)
from local_operator.evaluation.adapters.discovery import (
    AdapterDiscoveryError,
    load_selected_adapter,
    verify_release_manifest,
    workspace_digest,
)
from local_operator.evaluation.adapters.rpc import (
    AsyncIncrementalReader,
    CancelRequest,
    IncrementalWriter,
    RpcError,
    RpcProtocolError,
    RpcRequest,
    RpcResponse,
    canonical_line,
    parse_request_or_cancel,
)
from local_operator.evaluation.evidence.models import canonical_digest
from local_operator.evaluation.protocol import ProtocolModel

REQUEST_FD_ENV = "LO_ADAPTER_REQUEST_FD"
RESPONSE_FD_ENV = "LO_ADAPTER_RESPONSE_FD"
MAX_REPLAY = 128
# An episode may legitimately exceed the request replay window. Operation
# records are never evicted because losing one would turn a retry into a second
# side effect; bounded exhaustion poisons before dispatch instead.
MAX_OPERATION_RECORDS = 4096


@dataclass(frozen=True)
class _OperationRecord:
    method: AdapterMethod
    params_digest: str
    result: dict[str, Any] | None
    error: RpcError | None


def _workspace_digest(selector: AdapterSelector) -> str:
    verify_release_manifest(selector)
    digest = workspace_digest(selector.workspace)
    if digest != selector.workspace_digest:
        raise AdapterDiscoveryError("adapter workspace content digest differs")
    return digest


class Worker:
    def __init__(
        self,
        request_fd: int,
        response_fd: int,
        *,
        max_operation_records: int = MAX_OPERATION_RECORDS,
    ) -> None:
        self._reader = AsyncIncrementalReader(request_fd)
        self._writer = IncrementalWriter(response_fd)
        self._state: AdapterState = "NEW"
        self._adapter: EvaluationAdapter | None = None
        self._selector: AdapterSelector | None = None
        self._handshake: Handshake | None = None
        self._rescue_descriptor: RescueDescriptor | None = None
        self._rescue_actions: dict[str, _OperationRecord] = {}
        self._last_id = 0
        self._replay: OrderedDict[int, tuple[AdapterMethod, str, bytes]] = OrderedDict()
        self._operations: dict[str, _OperationRecord] = {}
        self._max_operation_records = max_operation_records
        self._pending_line: asyncio.Task[bytes] | None = None

    def _line_task(self) -> asyncio.Task[bytes]:
        if self._pending_line is None:
            self._pending_line = asyncio.create_task(self._reader.read_line())
        return self._pending_line

    async def _take_line(self) -> bytes:
        task = self._line_task()
        raw = await task
        # A completed adapter call may have left this exact task reading ahead.
        # Clear only that consumed task, never a successor installed elsewhere.
        if self._pending_line is task:
            self._pending_line = None
        return raw

    async def run(self) -> int:
        while self._state not in ("CLOSED", "POISONED"):
            try:
                raw = await self._take_line()
                message = parse_request_or_cancel(raw)
                if isinstance(message, CancelRequest):
                    raise RpcProtocolError("cancel has no in-flight application call")
                await self._handle_with_cancel(message)
            except EOFError:
                await self._cancel_pending_line()
                return 0
            except RpcProtocolError:
                self._state = "POISONED"
                await self._cancel_pending_line()
                return 70
        await self._cancel_pending_line()
        return 0 if self._state == "CLOSED" else 70

    async def _cancel_pending_line(self) -> None:
        task = self._pending_line
        self._pending_line = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _handle_with_cancel(self, request: RpcRequest) -> None:
        task = asyncio.create_task(self._handle(request))
        reader = self._line_task()
        done, _ = await asyncio.wait({task, reader}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task
            return
        raw = await reader
        if self._pending_line is reader:
            self._pending_line = None
        control = parse_request_or_cancel(raw)
        if not isinstance(control, CancelRequest) or control.id != request.id:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise RpcProtocolError("only matching cancel is valid during an application call")
        task.cancel()
        await task

    async def _handle(self, request: RpcRequest) -> None:
        if request.id < self._last_id:
            raise RpcProtocolError("request IDs must be positive monotonic values")
        params_type = PARAM_MODELS[request.method]
        try:
            params = params_type.model_validate(request.params, strict=True)
        except Exception:
            self._write_error(
                request,
                "invalid_request",
                "request parameters are invalid",
                digest=canonical_digest("adapter-invalid-request-v1", request.params),
            )
            return
        digest = canonical_params_digest(request.method, params)
        previous = self._replay.get(request.id)
        if previous is not None:
            prior_method, prior_digest, response = previous
            if prior_method != request.method or prior_digest != digest:
                raise RpcProtocolError("request ID was reused with changed content")
            self._writer.write(response)
            return
        if request.id == self._last_id:
            raise RpcProtocolError("duplicate request ID is not cached")
        self._last_id = request.id
        operation_id = getattr(params, "operation_id", None)
        if request.method in KEYED_METHODS:
            assert isinstance(operation_id, str)
            previous_operation = self._operations.get(operation_id)
            if previous_operation is not None:
                if (
                    previous_operation.method != request.method
                    or previous_operation.params_digest != digest
                ):
                    raise RpcProtocolError("operation ID was reused with changed content")
                self._replay_operation(request, digest, previous_operation)
                return
            if len(self._operations) >= self._max_operation_records:
                raise RpcProtocolError("operation replay capacity is exhausted")
        if self._state not in METHOD_STATES[request.method]:
            self._write_error(
                request,
                "invalid_state",
                "method is invalid in current state",
                digest=digest,
            )
            return
        rescue_action_id: str | None = None
        if request.method == "cleanup" and self._rescue_descriptor is not None:
            assert isinstance(params, CleanupParams)
            rescue_action_id = self._validate_rescue_cleanup(params)
            previous_action = self._rescue_actions.get(rescue_action_id)
            if previous_action is not None:
                if previous_action.params_digest != self._rescue_action_digest(params):
                    raise RpcProtocolError("rescue action was reused with changed content")
                assert isinstance(operation_id, str)
                # Reserve the alias before replying. Otherwise a second logical
                # action could reuse this new operation key after we replayed it.
                self._operations[operation_id] = _OperationRecord(
                    method=request.method,
                    params_digest=digest,
                    result=previous_action.result,
                    error=previous_action.error,
                )
                self._replay_operation(request, digest, previous_action)
                return
        try:
            result = await self._dispatch(request.method, params)
            result_type = RESULT_MODELS[request.method]
            if not isinstance(result, result_type):
                raise TypeError("adapter returned the wrong closed result")
        except asyncio.CancelledError:
            response = self._write_error(
                request,
                "cancelled",
                "adapter call was cancelled",
                digest=digest,
                operation_id=operation_id,
            )
            if rescue_action_id is not None:
                self._record_rescue_action(rescue_action_id, params, request.method, response)
            return
        except RpcProtocolError:
            raise
        except (AdapterDiscoveryError, Exception):
            # Adapter exceptions may contain reprs, paths, environment values, or
            # tracebacks.  The wire exposes only a closed code and fixed text.
            response = self._write_error(
                request,
                "adapter_error",
                "adapter operation failed",
                digest=digest,
                operation_id=operation_id,
            )
            if rescue_action_id is not None:
                self._record_rescue_action(rescue_action_id, params, request.method, response)
            return
        self._state = METHOD_NEXT_STATE[request.method]
        response = RpcResponse(
            jsonrpc="2.0",
            id=request.id,
            method=request.method,
            result=result.model_dump(mode="json"),
        )
        self._send_response(request, digest, response, operation_id=operation_id)
        if rescue_action_id is not None:
            self._record_rescue_action(rescue_action_id, params, request.method, response)

    def _record_rescue_action(
        self,
        action_id: str,
        params: ProtocolModel,
        method: AdapterMethod,
        response: RpcResponse,
    ) -> None:
        assert isinstance(params, CleanupParams)
        self._rescue_actions[action_id] = _OperationRecord(
            method=method,
            params_digest=self._rescue_action_digest(params),
            result=response.result,
            error=response.error,
        )

    def _validate_rescue_cleanup(self, params: CleanupParams) -> str:
        descriptor = self._rescue_descriptor
        assert descriptor is not None
        if (
            params.cleanup_plan.episode_id != descriptor.episode_id
            or params.cleanup_plan != descriptor.cleanup_plan
            or params.cleanup_plan.cleanup_plan_id != descriptor.cleanup_plan.cleanup_plan_id
            or len(params.action_ids) != 1
        ):
            raise RpcProtocolError("cleanup differs from the pinned rescue plan")
        action_id = params.action_ids[0]
        expected = next(
            (action for action in descriptor.cleanup_plan.actions if action.action_id == action_id),
            None,
        )
        actual = next(
            (action for action in params.cleanup_plan.actions if action.action_id == action_id),
            None,
        )
        if expected is None or actual != expected:
            raise RpcProtocolError("cleanup action differs from the pinned rescue action")
        return action_id

    @staticmethod
    def _rescue_action_digest(params: CleanupParams) -> str:
        # Operation IDs are transport idempotency keys; rescue action identity is
        # independently pinned so a new key cannot repeat the same effect.
        return canonical_digest(
            "adapter-rescue-action-call-v1",
            params.model_dump(mode="json", exclude={"operation_id"}),
        )

    async def _dispatch(self, method: AdapterMethod, params: ProtocolModel) -> ProtocolModel:
        if method == "hello":
            assert isinstance(params, HelloParams)
            self._selector = params.selector
            self._adapter = load_selected_adapter(params.selector)
            self._handshake = Handshake(
                selector=params.selector,
                metadata=self._adapter.metadata,
                python=PythonRuntime.current(),
                workspace_digest=_workspace_digest(params.selector),
                selected_route=params.selector.route_capability,
            )
            return self._handshake
        if method == "begin_rescue":
            assert isinstance(params, BeginRescueParams)
            assert self._selector is not None and self._handshake is not None
            if (
                params.descriptor.selector != self._selector
                or params.descriptor.handshake != self._handshake
                or params.selector_digest
                != canonical_digest("adapter-rescue-selector-v1", self._selector)
                or params.handshake_digest
                != canonical_digest("adapter-rescue-handshake-v1", self._handshake)
            ):
                raise RpcProtocolError("rescue pins differ from the exact handshake")
            # This transition grants cleanup routing only. Adapter code is not
            # invoked, so it cannot create resources or resolve persisted refs.
            self._rescue_descriptor = params.descriptor
            self._rescue_actions.clear()
            return AckResult()
        assert self._adapter is not None
        handler = getattr(self._adapter, method)
        result = await handler(params)
        return cast(ProtocolModel, result)

    def _write_error(
        self,
        request: RpcRequest,
        code: str,
        message: str,
        *,
        digest: str,
        operation_id: str | None = None,
    ) -> RpcResponse:
        error = RpcError.model_validate({"code": code, "message": message}, strict=True)
        response = RpcResponse(jsonrpc="2.0", id=request.id, method=request.method, error=error)
        self._send_response(request, digest, response, operation_id=operation_id)
        return response

    def _send_response(
        self,
        request: RpcRequest,
        digest: str,
        response: RpcResponse,
        *,
        operation_id: str | None,
    ) -> None:
        if operation_id is not None:
            self._operations[operation_id] = _OperationRecord(
                method=request.method,
                params_digest=digest,
                result=response.result,
                error=response.error,
            )
        encoded = canonical_line(response)
        self._cache(request.id, request.method, digest, encoded)
        self._writer.write(encoded)

    def _replay_operation(
        self,
        request: RpcRequest,
        digest: str,
        record: _OperationRecord,
    ) -> None:
        # Request IDs identify transport attempts, so durable operation replay
        # repeats the closed outcome while correlating it to this new attempt.
        response = RpcResponse(
            jsonrpc="2.0",
            id=request.id,
            method=request.method,
            result=record.result,
            error=record.error,
        )
        encoded = canonical_line(response)
        self._cache(request.id, request.method, digest, encoded)
        self._writer.write(encoded)

    def _cache(self, request_id: int, method: AdapterMethod, digest: str, response: bytes) -> None:
        if len(response) <= 1024 * 1024:
            self._replay[request_id] = (method, digest, response)
            while len(self._replay) > MAX_REPLAY:
                self._replay.popitem(last=False)


def _descriptor(name: str) -> int:
    value = os.environ.pop(name, "")
    if not value.isdecimal():
        raise RuntimeError("worker protocol descriptor is missing")
    return int(value)


def main() -> int:
    # The launcher passes ``-B``; this is the in-process guarantee of the same
    # thing, so a worker started any other way (a test harness, a debugger)
    # still never writes bytecode into the verified workspace it runs in.
    sys.dont_write_bytecode = True
    request_fd = _descriptor(REQUEST_FD_ENV)
    response_fd = _descriptor(RESPONSE_FD_ENV)
    # stdout belongs to diagnostics, never framing.  Redirect it before metadata
    # access because imports and native libraries may write to fd 1 directly.
    os.dup2(2, 1)
    return asyncio.run(Worker(request_fd, response_fd).run())


if __name__ == "__main__":
    raise SystemExit(main())
