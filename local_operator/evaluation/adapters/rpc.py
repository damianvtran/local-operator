"""Canonical JSONL RPC over two inherited, one-way protocol descriptors."""

from __future__ import annotations

import asyncio
import errno
import json
import os
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from local_operator.evaluation.adapters.api import AdapterMethod
from local_operator.evaluation.protocol import ProtocolModel

MAX_RPC_BYTES = 1024 * 1024
MAX_ERROR_MESSAGE = 2000
MAX_SAFE_ID = 2**53 - 1
# Detail bounds. Every one of these is a hard wire limit rather than a
# formatting preference: the envelope is parsed by a strict model on the far
# side, so an unbounded field would let a worker's exception text decide how
# much the parent must allocate and canonicalise.
MAX_DETAIL_MESSAGE = 512
MAX_DETAIL_TYPE = 128
MAX_DETAIL_CAUSES = 4
MAX_DETAIL_FRAMES = 8
MAX_DETAIL_NAME = 128
#: Substituted for any string the worker's own canary check rejects. Failing
#: CLOSED (drop the text, keep the structure) rather than attempting to scrub
#: keeps a partially-matched secret from being reassembled from what survived.
WITHHELD = "<withheld: matched a secret canary>"


class RpcProtocolError(RuntimeError):
    pass


class RpcRemoteError(RuntimeError):
    """A remote error the worker ANSWERED, carrying its structured cause.

    ``detail`` is optional because the closed error-code set predates it and a
    worker that cannot describe its failure must still be able to report one.
    It is folded into ``str()`` rather than left as an attribute the caller has
    to know about: the runner records a fatal error by rendering the exception
    (``episode._diagnostic``), so anything not visible through ``str`` never
    reaches the evidence bundle a paid episode leaves behind.
    """

    def __init__(self, code: str, message: str, detail: "RpcErrorDetail | None" = None) -> None:
        rendered = f"{code}: {message}"
        if detail is not None:
            rendered = f"{rendered} [{detail.render()}]"
        super().__init__(rendered)
        self.code = code
        self.detail = detail


class RpcErrorFrame(ProtocolModel):
    """One worker-side call-site: WHERE it raised, never WHAT was in scope.

    A raw traceback is refused across this boundary and that refusal is right --
    it renders source text (which can embed a literal credential) and absolute
    paths (which leak the worker's filesystem layout and the account it runs
    under). But "which line of the adapter raised" is the single most valuable
    fact for diagnosis and carries neither: a BASENAME, a line number and a
    function name are derived from the adapter's own published wheel, contain no
    runtime value, and cannot be steered by task content. Locals are absent by
    construction -- ``traceback.extract_tb`` never captures them -- rather than
    stripped afterwards, so there is no filter to get wrong.
    """

    file: str = Field(min_length=1, max_length=MAX_DETAIL_NAME)
    line: int = Field(ge=0, le=MAX_SAFE_ID)
    function: str = Field(min_length=1, max_length=MAX_DETAIL_NAME)


class RpcErrorCause(ProtocolModel):
    """One link of the ``__cause__``/``__context__`` chain.

    The chain is what actually names the fault. An adapter that wraps a cloud
    SDK failure in its own ``RuntimeError`` puts the diagnosable text one link
    down, so reporting only the outermost type reproduces the very opacity this
    envelope exists to remove.
    """

    exception_type: str = Field(min_length=1, max_length=MAX_DETAIL_TYPE)
    message: str = Field(max_length=MAX_DETAIL_MESSAGE)


class RpcErrorDetail(ProtocolModel):
    """Bounded, worker-redacted cause travelling inside the existing envelope.

    This deliberately extends ``RpcError`` instead of opening a second channel.
    The error envelope already has the properties a diagnostic needs -- it is
    correlated to the request, it is what the operation replay cache stores, and
    it is the one thing a poisoned channel still delivers -- so a parallel path
    would have to re-earn all three and would be absent on exactly the failures
    that matter. ``code`` stays a closed set and ``message`` stays a fixed
    string; the variable part is confined here, where every field is bounded and
    every string has passed the worker's canary check.
    """

    exception_type: str = Field(min_length=1, max_length=MAX_DETAIL_TYPE)
    message: str = Field(max_length=MAX_DETAIL_MESSAGE)
    method: AdapterMethod
    # The idempotency key the failure belongs to. Without it a reader holding a
    # bundle cannot tell which of several same-method calls died, and the
    # operation replay cache returns this error again under a NEW request ID,
    # so the request ID alone does not identify the originating operation.
    operation_id: str | None = Field(default=None, max_length=MAX_DETAIL_NAME)
    causes: tuple[RpcErrorCause, ...] = Field(default=(), max_length=MAX_DETAIL_CAUSES)
    frames: tuple[RpcErrorFrame, ...] = Field(default=(), max_length=MAX_DETAIL_FRAMES)

    @field_validator("causes", "frames", mode="before")
    @classmethod
    def _freeze(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    def render(self) -> str:
        """One line naming the cause, for the fatal-error evidence artifact."""

        parts = [f"{self.exception_type}: {self.message}" if self.message else self.exception_type]
        parts.append(f"method={self.method}")
        if self.operation_id is not None:
            parts.append(f"operation_id={self.operation_id}")
        for cause in self.causes:
            parts.append(
                f"caused by {cause.exception_type}: {cause.message}"
                if cause.message
                else f"caused by {cause.exception_type}"
            )
        if self.frames:
            trace = " <- ".join(
                f"{frame.file}:{frame.line} in {frame.function}" for frame in self.frames
            )
            parts.append(f"at {trace}")
        return "; ".join(parts)


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
    detail: RpcErrorDetail | None = None


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


def parse_request_or_cancel(line: bytes) -> RpcRequest | CancelRequest:
    """Decode the application and control planes without accepting extra shapes."""

    request_error: RpcProtocolError | None = None
    try:
        parsed = parse_canonical_line(line, RpcRequest)
        assert isinstance(parsed, RpcRequest)
        return parsed
    except RpcProtocolError as error:
        request_error = error
    try:
        parsed = parse_canonical_line(line, CancelRequest)
        assert isinstance(parsed, CancelRequest)
        return parsed
    except RpcProtocolError:
        raise request_error


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


class AsyncIncrementalReader:
    """Event-loop pipe reader so cancel remains readable without executor leaks."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self._buffer = bytearray()
        os.set_blocking(fd, False)

    async def read_line(self) -> bytes:
        loop = asyncio.get_running_loop()
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
            ready = asyncio.Event()
            loop.add_reader(self.fd, ready.set)
            try:
                await ready.wait()
            finally:
                loop.remove_reader(self.fd)
            try:
                chunk = os.read(self.fd, min(65536, MAX_RPC_BYTES + 1 - len(self._buffer)))
            except BlockingIOError:
                continue
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
                raise RpcRemoteError(
                    response.error.code, response.error.message, response.error.detail
                )
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
