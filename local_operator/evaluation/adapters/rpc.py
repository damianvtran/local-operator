"""Canonical JSONL RPC over two inherited, one-way protocol descriptors."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import time
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from local_operator.evaluation.adapters.api import AdapterMethod
from local_operator.evaluation.deadlines import funded_timeout
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


#: Which phase of a mutating call failed, as the ADAPTER declares it.
#:
#: ``observation`` means the mutation committed and only the read-back of the
#: resulting state failed -- the one shape of failure a repeat call cannot
#: double-apply. ``unknown`` is the default and means exactly what the boundary
#: assumed before this field existed: the call may or may not have applied, so
#: it is not safe to repeat. A worker only ever writes ``observation`` when the
#: adapter raised ``ObservationPhaseError``; nothing is inferred from an
#: exception type, a message, or a traceback.
ErrorPhase: TypeAlias = Literal["unknown", "observation"]


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
    # A CLOSED enum rather than free text, because the parent makes a safety
    # decision on it: this is the only field in this model that changes what
    # the harness DOES rather than what it records. Defaulting to "unknown"
    # keeps every adapter that does not participate on the pre-existing
    # poison-on-any-mutating-failure path.
    phase: ErrorPhase = "unknown"
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
        # Rendered only when it is load-bearing. An "unknown" phase is the
        # default and adds noise to every pre-existing diagnostic; naming the
        # observation phase explains why the harness then retried.
        if self.phase != "unknown":
            parts.append(f"phase={self.phase}")
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


#: The magnitude at which ``:g`` leaves fixed notation -- documentation and
#: test input ONLY. No production line reads it and none may: ``:g`` rounds to
#: six significant digits BEFORE it decides, so 999_999.5 s renders ``1e+06``
#: while sitting under this value, and a guard comparing against the constant
#: would leave that value in exponent form. ``_rendered_budget`` reads the
#: RENDERED text instead, and this constant only states the band in prose and
#: lets ``test_the_longest_legal_budget_renders_in_plain_units`` pin where
#: ``:g`` crosses over (999_999 -> ``1e+06``).
_EXPONENT_FORM_AT_S = 1_000_000.0


def _rendered_budget(timeout: float) -> str:
    """Render a budget in seconds in the units an operator reads: never exponent.

    WHY ``:g`` IS NOT ENOUGH HERE. This number is the only place the sentence
    says how long the deadline was, and ``:g`` drops into exponent form at the
    magnitude ``_EXPONENT_FORM_AT_S`` names -- from there and for a stretch
    below it -- which is reachable rather than theoretical now that the budget
    can be the FUNDED value: nine maximal cleanup actions declare 1_036_800 s
    and fund to ``1.03683e+06s`` in the one line an operator reads. The
    protocol's own ceiling is worse -- ``CleanupPlan``
    admits ``MAX_DECLARATIONS`` (256) actions of ``MAX_CLEANUP_TIMEOUT_MS`` x
    ``MAX_CLEANUP_ATTEMPTS`` (115_200 s) each, so the largest legal call funds
    to 29_491_230 s and would print ``2.949123e+07s``. ``execute`` cannot reach
    the band at all (its ceiling is 3_870 s of declared waiting); the renderer
    serves both arms because neither should have to know that.

    THE BAND BELOW IS DELIBERATELY UNTOUCHED. Every value already rendered
    must stay byte-identical, because those strings are what the campaign's
    readouts quote and what this module's tests assert: ``0.05s``/``0.25s`` for
    ``close``, ``180s``/``0.5s``/``31.5s`` for ``execute`` and ``cleanup``. The
    guard is therefore on the RENDERED text rather than on the magnitude --
    ``:g`` rounds to six significant digits before it decides, so 999_999.9 s
    renders an exponent too -- and only a value that already renders one is
    re-rendered, which makes the byte-identity a property of the check instead
    of a promise about it. ``_EXPONENT_FORM_AT_S`` is deliberately NOT
    consulted here: it describes the band, it is not a boundary this function
    may compare against, and the sub-band values its own comment names are
    asserted by ``test_the_longest_legal_budget_renders_in_plain_units`` so
    that a guard which does compare against it fails loudly rather than
    silently.

    Fixed point rather than ``repr`` (``1000000.0``) or a rounded integer cast:
    the trailing zeros of the six decimals are stripped, so 29_491_230.0 s
    reads ``29491230`` while a budget carrying declared milliseconds keeps
    them.
    """

    rendered = f"{timeout:g}"
    # Read the RENDERED text rather than comparing against ``_EXPONENT_FORM_AT_S``:
    # ``:g`` rounds to six significant digits before it decides, so a value just
    # below the band (999_999.9 s) renders an exponent too, and the comparison
    # would leave it in exponent form. The boundary is pinned by
    # ``test_the_longest_legal_budget_renders_in_plain_units``.
    if "e" not in rendered:
        return rendered
    return f"{timeout:.6f}".rstrip("0").rstrip(".")


def _timeout_detail(
    method: AdapterMethod,
    *,
    timeout: float,
    elapsed: float,
    request_id: int,
    operation_id: str | None,
) -> str:
    """Name which call exceeded which budget, in one line.

    WHY A MESSAGE AND NOT A NEW EXCEPTION TYPE. A fatal failure reaches the
    bundle by rendering the exception (``episode._diagnostic`` builds
    ``f"{type(error).__name__}: {error}"``), so the MESSAGE is the whole
    interface an operator holding an artifact gets; a subclass would carry the
    same text while changing ``_diagnostic_code``, which is derived from the
    type name and is what the campaign's bundles and readouts bucket on
    (``timeouterror``). The defect was the empty message: ``raise`` re-raised
    ``wait_for``'s bare ``TimeoutError``, discarding the ``method`` this frame
    already held and the ``timeout`` it had just exceeded, so a 6078 s episode
    died with a 143-byte artifact reading ``TimeoutError: ``.

    This is the opposite call from ``docs/design-aside-deadline.md``, which
    REJECTED a message-only fix for the attach-owner deadline in favour of
    ``OwnerAckTimeout(ConnectionError, TimeoutError)``. Neither of its two
    reasons holds here: there, two conditions (a wedged owner vs a slow one)
    had to stay distinguishable, and other catch sites keyed on
    ``ConnectionError`` alone. Here there is one condition -- the deadline
    expired, whoever's fault that is -- no catch arm depends on the type, and
    the type IS the bucket key, so a subclass would re-key every historical
    ``timeouterror`` comparison to buy nothing.

    ``timeout`` IS THE DEADLINE THE CALL RAN UNDER, not the caller's budget:
    ``call`` passes ``funded_timeout``'s result. The two are the same number to
    the byte for a request that declares nothing, and they are different -- the
    caller's constant is the smaller one -- for the two methods that declare
    their own duration (``execute``, ``cleanup``; see
    :mod:`local_operator.evaluation.deadlines`). Naming the constant there would
    report a budget the call never exceeded, on exactly the calls whose timeout
    an operator reads a readout for. The two candidate numbers and their
    ordering by size are pinned by
    ``test_a_funded_call_reports_the_budget_it_exceeded_not_the_callers_constant``
    and by ``deadlines``' own tests.

    ORDER IS LOAD-BEARING. The readouts that consume the recorded diagnostic
    truncate it (110-160 characters), so the method and the budget come first,
    the correlation ids last. The budget itself goes through
    :func:`_rendered_budget`, which keeps it in the plain units ``elapsed`` is
    already in at every magnitude the protocol admits. ``elapsed`` is SAMPLED
    rather than assumed equal to ``timeout``: timer granularity, a busy loop, or
    an executor in the way all push the actual firing time past the deadline,
    and a gap that is not the deadline is worth recording rather than rounding
    away. It is measured from BEFORE the request frame is written (see ``call``),
    so it is the caller's wall time for the whole call -- ``wait_for``'s own
    deadline begins a moment later -- which is the reading "exceeded its budget
    after N s" claims.
    """

    rendered_operation = f"; operation_id {operation_id[:MAX_DETAIL_NAME]}" if operation_id else ""
    return (
        f"{method} exceeded its {_rendered_budget(timeout)}s budget after {elapsed:.1f}s "
        f"(request {request_id}{rendered_operation})"
    )


def _attributed_channel_death(error: OSError, detail: str) -> OSError:
    """Rebuild a failed WRITE as its own error type, carrying ``detail``.

    WHY THE TYPE SURVIVES. ``episode._diagnostic_code`` derives the bundle's
    ``diagnostic_code`` from the exception TYPE name, so a worker death has to
    stay ``brokenpipeerror`` -- the bucket the campaign's readouts already
    group on, and the one an operator greps by when the process tree is gone.
    Only the human-readable half is replaced; the errno is rebuilt in the same
    two-argument form ``os.write`` raises it in, so ``str()`` still leads with
    the cause (``[Errno 32] ...``) rather than discarding the one part of the
    original message that named anything. A write that failed WITHOUT an errno
    (``IncrementalWriter``'s own zero-byte guard) keeps the one-argument form
    instead of rendering an errno-less ``[Errno None]`` prefix.
    """

    if error.errno is None:
        return type(error)(detail)
    return type(error)(error.errno, detail)


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
        # Set once, by the failure that took the channel down. A poisoned
        # client answers every later call with the same sentence, so without
        # this a bundle whose fatal came from the SECOND call names neither the
        # call that died nor why -- see ``_poison``.
        self._poison_reason: str | None = None

    async def call(
        self,
        method: AdapterMethod,
        params: ProtocolModel,
        *,
        timeout: float,
        execution_overhead_seconds_per_action: float = 0.0,
    ) -> dict[str, Any]:
        """One call, governed by the greater of ``timeout`` and what it declared.

        ``timeout`` is the caller's budget, and it is a FLOOR rather than the
        whole deadline: a request that declares its own duration (an
        ``ActionBatch``'s waits, a ``CleanupPlan``'s selected per-action
        timeouts and attempts) is always funded to finish what it asked for.
        A request that declares nothing is unaffected to the byte -- this is the
        same ``wait_for``, and a genuinely wedged call still times out here,
        still sends its cancel, and still poisons the channel below.

        Applied HERE, at the one deadline in the harness, rather than at each
        caller: any caller added later inherits it, and no call site can
        re-introduce the mismatch by passing a constant that never saw the
        request. See :mod:`local_operator.evaluation.deadlines` for the measured
        failure this exists to prevent and for the worst case it accepts.
        """

        async with self._lock:
            if self._poisoned:
                reason = self._poison_reason or "an earlier failure"
                raise RpcProtocolError(
                    f"RPC channel is poisoned by {reason}; {method} was not sent"
                )
            request_id = self._next_id
            self._next_id += 1
            if request_id > MAX_SAFE_ID:
                await self._poison(f"exhausted request IDs, last was {method}")
                raise RpcProtocolError("RPC request IDs exhausted")
            request = RpcRequest(
                jsonrpc="2.0",
                id=request_id,
                method=method,
                params=params.model_dump(mode="json"),
            )
            # The deadline this call actually runs under, resolved ONCE and
            # used for both the wait and the detail below. Resolving it at the
            # deadline alone (as the wait needs) and then building the detail
            # from the caller's ``timeout`` would make the message name a budget
            # the call never exceeded whenever ``funded_timeout`` lifts it --
            # false for exactly the two declaring methods, and invisible to
            # every test whose request declares nothing, where the two values
            # are equal. See :mod:`local_operator.evaluation.deadlines`.
            effective_budget = funded_timeout(
                timeout,
                params,
                execution_overhead_seconds_per_action=execution_overhead_seconds_per_action,
            )
            # Started BEFORE the request is written, deliberately: ``elapsed``
            # is the caller's wall time for the whole call -- the frame going
            # out included -- not ``wait_for``'s deadline overshoot, which
            # begins only once ``wait_for`` is entered. A write that queued
            # behind a full pipe is time this call took, so charging it is the
            # honest reading of "exceeded its budget after N s"; sampling
            # after the write would understate a call whose time went into the
            # channel.
            started = time.monotonic()
            try:
                self._write_request(request, method)
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._read_response, request_id, method),
                    effective_budget,
                )
            except TimeoutError:
                # Sampled BEFORE the cancel grace and the poison below, so the
                # number is the wait_for deadline's actual firing time rather
                # than the deadline plus teardown.
                elapsed = time.monotonic() - started
                operation_id = getattr(params, "operation_id", None)
                try:
                    self._writer.write(
                        canonical_line(
                            CancelRequest(jsonrpc="2.0", control="cancel", id=request_id)
                        )
                    )
                    await asyncio.sleep(1)
                except OSError as error:
                    # The same death seen from the timeout branch: the worker
                    # that outran its budget is GONE, so the cancel cannot be
                    # delivered. The timeout detail is what the readouts want
                    # most and it leads for the same truncation reason; the
                    # write's own type is what propagates, so the bucket an
                    # operator greps for does not move. The ``finally`` below
                    # still poisons with the timeout as the cause.
                    detail = _timeout_detail(
                        method,
                        timeout=effective_budget,
                        elapsed=elapsed,
                        request_id=request_id,
                        operation_id=operation_id,
                    )
                    raise _attributed_channel_death(
                        error, f"{detail}; the cancel was not delivered"
                    ) from error
                finally:
                    await asyncio.shield(
                        self._poison(f"a timeout on {method}, request {request_id}")
                    )
                # The cancel, the grace and the poison above are unchanged and
                # stay on this path: a timed-out call's late reply must never be
                # readable by a later request. Only the raised error changes --
                # from ``wait_for``'s bare ``TimeoutError`` to one that names
                # the call and the budget it exceeded.
                raise TimeoutError(
                    _timeout_detail(
                        method,
                        timeout=effective_budget,
                        elapsed=elapsed,
                        request_id=request_id,
                        operation_id=operation_id,
                    )
                ) from None
            except asyncio.CancelledError:
                await asyncio.shield(
                    self._poison(f"a cancellation during {method}, request {request_id}")
                )
                raise
            except Exception as error:
                await asyncio.shield(
                    self._poison(f"{type(error).__name__} on {method}, request {request_id}")
                )
                raise
            if response.error is not None:
                raise RpcRemoteError(
                    response.error.code, response.error.message, response.error.detail
                )
            assert response.result is not None
            return response.result

    def _write_request(self, request: RpcRequest, method: AdapterMethod) -> None:
        """Send one request frame, attributing a dead channel to the call.

        THE WRITE IS WHERE A BETWEEN-STEPS WORKER DEATH IS USUALLY FIRST SEEN.
        ``supervisor.launch`` closes the parent's own copies of the request and
        response pipes right after spawn, so a worker that has already exited
        makes the NEXT write raise EPIPE immediately, and the only other
        liveness check is ``process.poll()`` inside ``terminate()`` -- which
        nothing consults until a later call has failed. Re-raised raw, that
        fatal read ``BrokenPipeError: [Errno 32] Broken pipe``: the mirror
        image of the READ defect this file fixes elsewhere, naming neither the
        call nor the request, and the last thing a 6000 s bundle would say
        about the step it died on.

        The frame limit is attributed here too. It is the same write that could
        not be completed, the read side already words its own frame faults this
        way (``_read_response``), and "RPC message exceeds one MiB" alone leaves
        an operator unable to say which call asked for a frame that size --
        reachable from a legal batch, since ``MAX_TEXT_LENGTH`` per action times
        ``MAX_BATCH_SIZE`` is well past ``MAX_RPC_BYTES``.
        """

        try:
            data = canonical_line(request)
        except RpcProtocolError as error:
            raise RpcProtocolError(
                f"{error} (while sending {method}, request {request.id})"
            ) from error
        try:
            self._writer.write(data)
        except OSError as error:
            raise _attributed_channel_death(
                error,
                f"{method} was not sent: the adapter worker closed the channel "
                f"(request {request.id})",
            ) from error

    def _read_response(self, request_id: int, method: AdapterMethod) -> RpcResponse:
        """Read one reply, attributed to the call it answers.

        Every failure here kills the channel (the caller poisons on any raise),
        so this is the last place that still knows WHICH call was in flight. A
        bare ``EOFError`` out of the reader means the worker exited without
        replying -- the exit path for a worker-side protocol fault, which
        answers with a torn-down channel rather than an error frame -- and
        carries neither the method nor the request, leaving an operator unable
        to say what hung. The reason for that exit reaches the bundle through
        the worker's stderr tail (``Worker._report_protocol_error``).
        """

        try:
            line = self._reader.read_line()
        except EOFError as error:
            raise EOFError(
                "adapter worker closed the channel before replying to "
                f"{method} (request {request_id})"
            ) from error
        try:
            parsed = parse_canonical_line(line, RpcResponse)
        except RpcProtocolError as error:
            raise RpcProtocolError(
                f"{error} (while reading the reply to {method}, request {request_id})"
            ) from error
        assert isinstance(parsed, RpcResponse)
        if parsed.id != request_id or parsed.method != method:
            # What ARRIVED leads and what was expected follows, because the
            # received ids are this message's distinguishing token and the
            # readouts cut at 110-160 characters. "expected inspect_requirements
            # id 1" is shared by every call of that method, so behind the cut it
            # left a row unable to say which reply failed to answer. One rule
            # for both messages: the token that separates THIS failure from a
            # neighbouring one goes before the cut -- method and budget for a
            # timeout (``_timeout_detail``), the mismatched reply here.
            raise RpcProtocolError(
                "RPC reply does not match the in-flight call: "
                f"got {parsed.method} id {parsed.id}, expected {method} id {request_id}"
            )
        return parsed

    async def _poison(self, reason: str | None = None) -> None:
        if self._poisoned:
            return
        self._poisoned = True
        self._poison_reason = reason
        result = self._terminate()
        if hasattr(result, "__await__"):
            await result
