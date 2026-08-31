"""Isolated adapter worker speaking only over inherited protocol descriptors."""

from __future__ import annotations

import asyncio
import os
from collections import OrderedDict
from typing import cast

from local_operator.evaluation.adapters.api import (
    KEYED_METHODS,
    METHOD_NEXT_STATE,
    METHOD_STATES,
    PARAM_MODELS,
    RESULT_MODELS,
    AdapterMethod,
    AdapterSelector,
    AdapterState,
    EvaluationAdapter,
    Handshake,
    HelloParams,
    PythonRuntime,
    canonical_params_digest,
)
from local_operator.evaluation.adapters.discovery import (
    AdapterDiscoveryError,
    load_selected_adapter,
)
from local_operator.evaluation.adapters.rpc import (
    IncrementalReader,
    IncrementalWriter,
    RpcError,
    RpcProtocolError,
    RpcRequest,
    RpcResponse,
    canonical_line,
    parse_canonical_line,
)
from local_operator.evaluation.evidence.models import canonical_digest
from local_operator.evaluation.protocol import ProtocolModel

REQUEST_FD_ENV = "LO_ADAPTER_REQUEST_FD"
RESPONSE_FD_ENV = "LO_ADAPTER_RESPONSE_FD"
MAX_REPLAY = 128


def _workspace_digest(selector: AdapterSelector) -> str:
    info = os.stat(selector.workspace, follow_symlinks=False)
    return canonical_digest(
        "adapter-workspace-v1",
        {"device": info.st_dev, "inode": info.st_ino, "mode": info.st_mode},
    )


class Worker:
    def __init__(self, request_fd: int, response_fd: int) -> None:
        self._reader = IncrementalReader(request_fd)
        self._writer = IncrementalWriter(response_fd)
        self._state: AdapterState = "NEW"
        self._adapter: EvaluationAdapter | None = None
        self._selector: AdapterSelector | None = None
        self._last_id = 0
        self._replay: OrderedDict[int, tuple[AdapterMethod, str, bytes]] = OrderedDict()
        self._operations: dict[tuple[AdapterMethod, str], str] = {}

    async def run(self) -> int:
        while self._state not in ("CLOSED", "POISONED"):
            try:
                raw = await asyncio.to_thread(self._reader.read_line)
                request = cast(RpcRequest, parse_canonical_line(raw, RpcRequest))
                await self._handle(request)
            except EOFError:
                return 0
            except RpcProtocolError:
                self._state = "POISONED"
                return 70
        return 0 if self._state == "CLOSED" else 70

    async def _handle(self, request: RpcRequest) -> None:
        if request.id < self._last_id:
            raise RpcProtocolError("request IDs must be positive monotonic values")
        params_type = PARAM_MODELS[request.method]
        try:
            params = params_type.model_validate(request.params, strict=True)
        except Exception:
            self._write_error(request, "invalid_request", "request parameters are invalid")
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
        if self._state not in METHOD_STATES[request.method]:
            self._write_error(request, "invalid_state", "method is invalid in current state")
            return
        operation_id = getattr(params, "operation_id", None)
        if request.method in KEYED_METHODS:
            assert isinstance(operation_id, str)
            operation_key = (request.method, operation_id)
            prior_digest = self._operations.get(operation_key)
            if prior_digest is not None and prior_digest != digest:
                raise RpcProtocolError("operation ID was reused with changed parameters")
            self._operations[operation_key] = digest
        try:
            result = await self._dispatch(request.method, params)
            result_type = RESULT_MODELS[request.method]
            if not isinstance(result, result_type):
                raise TypeError("adapter returned the wrong closed result")
        except asyncio.CancelledError:
            self._write_error(request, "cancelled", "adapter call was cancelled")
            return
        except (AdapterDiscoveryError, Exception):
            # Adapter exceptions may contain reprs, paths, environment values, or
            # tracebacks.  The wire exposes only a closed code and fixed text.
            self._write_error(request, "adapter_error", "adapter operation failed")
            return
        self._state = METHOD_NEXT_STATE[request.method]
        response = RpcResponse(
            jsonrpc="2.0",
            id=request.id,
            method=request.method,
            result=result.model_dump(mode="json"),
        )
        encoded = canonical_line(response)
        self._cache(request.id, request.method, digest, encoded)
        self._writer.write(encoded)

    async def _dispatch(self, method: AdapterMethod, params: ProtocolModel) -> ProtocolModel:
        if method == "hello":
            assert isinstance(params, HelloParams)
            self._selector = params.selector
            self._adapter = load_selected_adapter(params.selector)
            return Handshake(
                selector=params.selector,
                metadata=self._adapter.metadata,
                python=PythonRuntime.current(),
                workspace_digest=_workspace_digest(params.selector),
                selected_route=params.selector.route_capability,
            )
        assert self._adapter is not None
        handler = getattr(self._adapter, method)
        result = await handler(params)
        return cast(ProtocolModel, result)

    def _write_error(
        self,
        request: RpcRequest,
        code: str,
        message: str,
    ) -> None:
        error = RpcError.model_validate({"code": code, "message": message}, strict=True)
        response = RpcResponse(jsonrpc="2.0", id=request.id, method=request.method, error=error)
        encoded = canonical_line(response)
        self._cache(
            request.id,
            request.method,
            canonical_digest("adapter-invalid-request-v1", request.params),
            encoded,
        )
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
    request_fd = _descriptor(REQUEST_FD_ENV)
    response_fd = _descriptor(RESPONSE_FD_ENV)
    # stdout belongs to diagnostics, never framing.  Redirect it before metadata
    # access because imports and native libraries may write to fd 1 directly.
    os.dup2(2, 1)
    return asyncio.run(Worker(request_fd, response_fd).run())


if __name__ == "__main__":
    raise SystemExit(main())
