"""Canonical JSONL RPC over two inherited, one-way protocol descriptors."""

from __future__ import annotations

import asyncio
import errno
import json
import os
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, model_validator

from local_operator.evaluation.adapters.api import AdapterMethod
from local_operator.evaluation.protocol import ProtocolModel

MAX_RPC_BYTES = 1024 * 1024
MAX_ERROR_MESSAGE = 2000
MAX_SAFE_ID = 2**53 - 1


class RpcProtocolError(RuntimeError):
    pass


class RpcRemoteError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class RpcError(ProtocolModel):
    code: Literal[
        "adapter_error",
        "cancelled",
        "invalid_request",
        "invalid_state",
        "protocol_error",
        "timeout",
    ]
    message: str = Field(min_length=1, max_length=MAX_ERROR_MESSAGE)


class RpcRequest(ProtocolModel):
    jsonrpc: Literal["2.0"]
    id: int = Field(gt=0, le=MAX_SAFE_ID)
    method: AdapterMethod
    params: dict[str, Any]


class RpcResponse(ProtocolModel):
    jsonrpc: Literal["2.0"]
    id: int = Field(gt=0, le=MAX_SAFE_ID)
    method: AdapterMethod
    result: dict[str, Any] | None = None
    error: RpcError | None = None

    @model_validator(mode="after")
    def _result_xor_error(self) -> "RpcResponse":
        if (self.result is None) == (self.error is None):
            raise ValueError("RPC response requires exactly one of result and error")
        return self


class CancelRequest(ProtocolModel):
    jsonrpc: Literal["2.0"]
    control: Literal["cancel"]
    id: int = Field(gt=0, le=MAX_SAFE_ID)


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RpcProtocolError("duplicate JSON object key")
        result[key] = value
    return result


def canonical_line(model: ProtocolModel) -> bytes:
    payload = model.to_canonical_json()
    if len(payload) > MAX_RPC_BYTES:
        raise RpcProtocolError("RPC message exceeds one MiB")
    return payload + b"\n"


def parse_canonical_line(line: bytes, model: type[ProtocolModel]) -> ProtocolModel:
    if not line.endswith(b"\n") or b"\r" in line:
        raise RpcProtocolError("RPC requires LF-only complete JSON lines")
    payload = line[:-1]
    if not payload or len(payload) > MAX_RPC_BYTES:
        raise RpcProtocolError("RPC message is empty or oversized")
    try:
        decoded = json.loads(payload, object_pairs_hook=_reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, RpcProtocolError) as error:
        raise RpcProtocolError("RPC message is malformed") from error
    try:
        parsed = model.model_validate(decoded, strict=True)
    except Exception as error:
        raise RpcProtocolError("RPC message shape is invalid") from error
    if parsed.to_canonical_json() != payload:
        raise RpcProtocolError("RPC message is not canonical JSON")
    return parsed


class IncrementalReader:
    """Bound allocation before newline and reject EOF in a partial frame."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self._buffer = bytearray()

    def read_line(self) -> bytes:
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[: newline + 1])
                del self._buffer[: newline + 1]
                if len(line) - 1 > MAX_RPC_BYTES:
                    raise RpcProtocolError("RPC message exceeds one MiB")
                return line
            if len(self._buffer) > MAX_RPC_BYTES:
                raise RpcProtocolError("RPC message exceeds one MiB before newline")
            try:
                chunk = os.read(self.fd, min(65536, MAX_RPC_BYTES + 1 - len(self._buffer)))
            except InterruptedError:
                continue
            if not chunk:
                if self._buffer:
                    raise RpcProtocolError("partial RPC message at EOF")
                raise EOFError
            self._buffer.extend(chunk)


class IncrementalWriter:
    def __init__(self, fd: int) -> None:
        self.fd = fd

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            try:
                written = os.write(self.fd, view)
            except InterruptedError:
                continue
            except OSError as error:
                if error.errno == errno.EINTR:
                    continue
                raise
            if written <= 0:
                raise BrokenPipeError("protocol descriptor accepted no bytes")
            view = view[written:]


class RpcClient:
    """One-flight host RPC with strict monotonic response correlation."""

    def __init__(
        self,
        request_fd: int,
        response_fd: int,
        *,
        terminate: Callable[[], Any],
    ) -> None:
        self._reader = IncrementalReader(response_fd)
        self._writer = IncrementalWriter(request_fd)
        self._terminate = terminate
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._poisoned = False

    async def call(
        self,
        method: AdapterMethod,
        params: ProtocolModel,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._poisoned:
                raise RpcProtocolError("RPC channel is poisoned")
            request_id = self._next_id
            self._next_id += 1
            if request_id > MAX_SAFE_ID:
                await self._poison()
                raise RpcProtocolError("RPC request IDs exhausted")
            request = RpcRequest(
                jsonrpc="2.0",
                id=request_id,
                method=method,
                params=params.model_dump(mode="json"),
            )
            try:
                self._writer.write(canonical_line(request))
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._read_response, request_id, method), timeout
                )
            except TimeoutError:
                try:
                    self._writer.write(
                        canonical_line(
                            CancelRequest(jsonrpc="2.0", control="cancel", id=request_id)
                        )
                    )
                    await asyncio.sleep(1)
                finally:
                    await asyncio.shield(self._poison())
                raise
            except asyncio.CancelledError:
                await asyncio.shield(self._poison())
                raise
            except Exception:
                await asyncio.shield(self._poison())
                raise
            if response.error is not None:
                raise RpcRemoteError(response.error.code, response.error.message)
            assert response.result is not None
            return response.result

    def _read_response(self, request_id: int, method: AdapterMethod) -> RpcResponse:
        parsed = parse_canonical_line(self._reader.read_line(), RpcResponse)
        assert isinstance(parsed, RpcResponse)
        if parsed.id != request_id or parsed.method != method:
            raise RpcProtocolError("RPC response ID or method differs from the in-flight call")
        return parsed

    async def _poison(self) -> None:
        if self._poisoned:
            return
        self._poisoned = True
        result = self._terminate()
        if hasattr(result, "__await__"):
            await result
