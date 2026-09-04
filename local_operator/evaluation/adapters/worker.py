"""Isolated adapter worker speaking only over inherited protocol descriptors."""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

from local_operator.evaluation.adapters.api import (
    KEYED_METHODS,
    METHOD_NEXT_STATE,
    METHOD_STATES,
    PARAM_MODELS,
    RESULT_MODELS,
    AdapterMethod,
    AdapterSelector,
    AdapterState,
    BeginRescueParams,
    CleanupParams,
    EvaluationAdapter,
    Handshake,
    HelloParams,
    PythonRuntime,
    RescuableAdapter,
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
    MAX_DETAIL_CAUSES,
    MAX_DETAIL_FRAMES,
    MAX_DETAIL_MESSAGE,
    MAX_DETAIL_NAME,
    MAX_DETAIL_TYPE,
    WITHHELD,
    AsyncIncrementalReader,
    CancelRequest,
    IncrementalWriter,
    RpcError,
    RpcErrorCause,
    RpcErrorDetail,
    RpcErrorFrame,
    RpcProtocolError,
    RpcRequest,
    RpcResponse,
    canonical_line,
    parse_request_or_cancel,
)
from local_operator.evaluation.evidence.models import canonical_digest
from local_operator.evaluation.protocol import ProtocolModel
from local_operator.evaluation.receipts import RedactionSet

REQUEST_FD_ENV = "LO_ADAPTER_REQUEST_FD"
RESPONSE_FD_ENV = "LO_ADAPTER_RESPONSE_FD"
MAX_REPLAY = 128
# An episode may legitimately exceed the request replay window. Operation
# records are never evicted because losing one would turn a retry into a second
# side effect; bounded exhaustion poisons before dispatch instead.
MAX_OPERATION_RECORDS = 4096


class AdapterRescueUnsupported(RuntimeError):
    """The selected adapter cannot accept the rescue handoff.

    Its own type (rather than a bare RuntimeError) so the worker's generic
    adapter-error path reports it as a normal, replied-to failure instead of
    a channel-killing protocol error: the parent must receive an answer it can
    attribute, because a rescue that silently timed out is indistinguishable
    from a slow one and hides the fact that nothing can be torn down.
    """


@dataclass(frozen=True)
class _OperationRecord:
    method: AdapterMethod
    params_digest: str
    result: dict[str, Any] | None
    error: RpcError | None


class _DenyAllRedactions(RedactionSet):
    """A canary set that refuses every non-empty string.

    Used when secret material was delivered but a canary set could not be built
    from it. The alternative -- leaving the set None -- reads identically to
    "this worker holds no secrets" and would skip the check entirely on the one
    occasion something is definitely there to protect.
    """

    def assert_clear(self, value: Any) -> None:
        raise ValueError("secret canary set is unavailable")


#: Constructed with empty canaries; ``assert_clear`` is what does the work.
_DENY_ALL = _DenyAllRedactions((), (), (), ())


def _control_safe(value: str, limit: int) -> str:
    """Collapse control characters and bound the result.

    The RPC line framing is LF-delimited and refuses CR, so an exception
    message containing either would make an otherwise-valid response
    unparseable -- the worker would have written a frame the parent rejects as
    a protocol error, killing the channel on the exact path that exists to
    explain a failure. Adapter text is arbitrary (a subprocess's captured
    stderr routinely carries newlines), so this is a wire requirement, not
    cosmetics.

    A message that is ENTIRELY control characters therefore collapses to ``""``
    once stripped. That is intended and is not a lossy edge case to guard:
    ``RpcErrorDetail.message`` permits an empty string, ``render()`` omits the
    empty half rather than emitting a dangling separator, and the exception
    TYPE -- the field that makes a failure bucketable -- is carried separately
    and is unaffected. Substituting a placeholder here would invent content the
    adapter never produced.
    """

    collapsed = "".join(character if character.isprintable() else " " for character in value)
    return collapsed.strip()[:limit]


def _redacted(value: str, limit: int, redactions: RedactionSet | None) -> str:
    """Bound and canary-check one string before it may cross the boundary.

    Fails CLOSED: a string that matches a canary is replaced wholesale rather
    than masked, because a partial mask still narrows the secret for anyone
    holding the bundle. ``redactions`` is None only when the worker holds no
    secrets at all (nothing was ever delivered to leak), which keeps the
    no-secret episode from paying for a scan that cannot match.

    ORDER IS THE SECURITY PROPERTY: scan the UNBOUNDED value, then truncate.
    ``RedactionSet.assert_clear`` is a SUBSTRING check, so truncating first
    severs the canary and the surviving prefix is returned verbatim -- a
    40-character AWS key sitting across the 512-character message cut emitted
    25 characters of itself, and an 808-character JWT emitted 488. This is
    systematic rather than a corner case: ANY secret longer than its field
    bound can never match once it has been cut. Reversing these two lines
    reintroduces that leak silently, which is why the regression test
    positions a secret to straddle the boundary.

    The parent's artifact scan is NOT a second line of defence against this
    particular failure, and must not be mistaken for one: it applies the same
    substring semantics to whatever bytes it receives, so it returns PASSED on
    an already-truncated secret for exactly the same reason. Both checks are
    blind to the same case, and this ordering is what closes it. The parent
    scan remains valuable for its own case -- values the worker never held.
    """

    if redactions is not None and value:
        try:
            # The FULL value, before any bound is applied.
            redactions.assert_clear(value)
        except ValueError:
            return WITHHELD
    return _control_safe(value, limit)


def _error_detail(
    error: BaseException,
    method: AdapterMethod,
    operation_id: str | None,
    redactions: RedactionSet | None,
) -> RpcErrorDetail | None:
    """Describe an adapter-side failure in bounded, redacted, structured form.

    Built ENTIRELY on the worker side, which is what makes the traceback
    question answerable at all: the parent never receives raw traceback text,
    only a tuple of (basename, line, function) triples this function derived and
    canary-checked. See ``RpcErrorFrame`` for why that projection carries no
    runtime value and no filesystem layout.

    Returning None rather than raising is deliberate. This runs on the failure
    path; a detail that cannot be built must degrade to the generic reply the
    parent already understood, never turn an answered adapter error into an
    unanswered channel death.
    """

    try:
        causes: list[RpcErrorCause] = []
        seen: set[int] = {id(error)}
        # ``__cause__`` first: an explicit ``raise ... from ...`` is the
        # adapter stating the real fault, while ``__context__`` is whatever
        # happened to be in flight and is only consulted when nothing was
        # stated.
        current: BaseException | None = error.__cause__ or error.__context__
        while current is not None and len(causes) < MAX_DETAIL_CAUSES:
            if id(current) in seen:
                break  # A cycle is possible through __context__; stop, never spin.
            seen.add(id(current))
            causes.append(
                RpcErrorCause(
                    exception_type=_redacted(type(current).__name__, MAX_DETAIL_TYPE, redactions)
                    or "Exception",
                    message=_redacted(str(current), MAX_DETAIL_MESSAGE, redactions),
                )
            )
            current = current.__cause__ or current.__context__
        frames: list[RpcErrorFrame] = []
        # The DEEPEST frames are the informative ones -- the harness dispatch
        # frames at the top are identical for every failure.
        for summary in traceback.extract_tb(error.__traceback__)[-MAX_DETAIL_FRAMES:]:
            frames.append(
                RpcErrorFrame(
                    # basename only: an absolute path names the worker's
                    # filesystem layout and the account it runs under.
                    file=_redacted(os.path.basename(summary.filename), MAX_DETAIL_NAME, redactions)
                    or "<unknown>",
                    line=summary.lineno or 0,
                    function=_redacted(summary.name, MAX_DETAIL_NAME, redactions) or "<unknown>",
                )
            )
        return RpcErrorDetail(
            exception_type=_redacted(type(error).__name__, MAX_DETAIL_TYPE, redactions)
            or "Exception",
            message=_redacted(str(error), MAX_DETAIL_MESSAGE, redactions),
            method=method,
            operation_id=(
                _redacted(operation_id, MAX_DETAIL_NAME, redactions) or None
                if operation_id is not None
                else None
            ),
            causes=tuple(causes),
            frames=tuple(frames),
        )
    except Exception:
        return None


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
        # Canaries over every secret THIS worker was handed, so error detail can
        # be checked before it is written to the pipe. Populated from the two
        # calls that deliver secret material (reset_start, begin_rescue) and
        # never from the environment, which is stripped by construction.
        self._redactions: RedactionSet | None = None

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
        # BEFORE dispatch, not inside it: ``reset_start`` and ``begin_rescue``
        # are themselves failure sites, and a credential delivered by the very
        # call that then raised is exactly the one whose value must not ride
        # back out in the exception text.
        self._track_secrets(params)
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
        except AdapterRescueUnsupported:
            # The ONE adapter-side failure whose reason crosses the wire. The
            # blanket redaction below exists because adapter exceptions may
            # carry reprs, paths, or environment values; this message is a
            # fixed string raised by the worker itself with no adapter-supplied
            # content, so naming it leaks nothing. It has to be named: a rescue
            # that cannot start must tell the operator WHY, or a descriptor
            # that can never be torn down looks like a transient failure.
            response = self._write_error(
                request,
                "invalid_state",
                "adapter does not implement begin_rescue",
                digest=digest,
                operation_id=operation_id,
            )
            if rescue_action_id is not None:
                self._record_rescue_action(rescue_action_id, params, request.method, response)
            return
        except (AdapterDiscoveryError, Exception) as error:
            # Adapter exceptions may contain reprs, paths, environment values, or
            # tracebacks. The wire still exposes only a closed code and fixed
            # text -- the variable part is confined to ``detail``, where every
            # field is bounded and every string has been canary-checked against
            # the secrets this worker was handed.
            #
            # Before this, the entire recorded diagnostic for a fatal failure
            # was "adapter_error: adapter operation failed": two consecutive
            # paid episodes (ep-e46c789ca818, ep-ffda3fc88f81) died here after
            # 19 and 16 billed steps, and diagnosing either cost another paid
            # run. Blanket redaction was the right instinct and the wrong
            # granularity -- it discarded the exception TYPE and the failing
            # METHOD, neither of which can carry adapter data.
            response = self._write_error(
                request,
                "adapter_error",
                "adapter operation failed",
                digest=digest,
                operation_id=operation_id,
                detail=_error_detail(error, request.method, operation_id, self._redactions),
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

    def _track_secrets(self, params: ProtocolModel) -> None:
        """Widen the canary set with any secret material this call delivered.

        Only ``ResetStartParams`` and ``BeginRescueParams`` carry resolved
        secrets (api.py's schema 1.2 note), and both are read structurally here
        rather than by method name so a future model that gains the field cannot
        silently escape the scan. Values are held only as canaries -- the set
        stores derived comparison forms and never exposes them -- and a failure
        to build one degrades to the previous behaviour of withholding all
        variable text, never to shipping unchecked text.
        """

        secrets = getattr(params, "secrets", None)
        if not secrets:
            return
        try:
            values = tuple(secret.value for secret in secrets)
            self._redactions = (
                RedactionSet.from_resolved_values(values)
                if self._redactions is None
                else self._redactions.with_values(values)
            )
        except (AttributeError, ValueError):
            # Unbuildable canaries mean nothing can be proven clean. Fail closed
            # with a set that matches nothing usable rather than leaving None,
            # which would read as "no secrets exist" and skip the check.
            self._redactions = _DENY_ALL
        return

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
            # The pin checks above are the security boundary and run FIRST:
            # nothing below may observe a descriptor whose selector/handshake
            # digests did not match the exact worker this process handshook as.
            self._rescue_descriptor = params.descriptor
            self._rescue_actions.clear()
            # The adapter MUST see begin_rescue. It is the only call that
            # carries the descriptor's infra_values plus the freshly resolved
            # ``params.secrets``, and therefore the only opportunity a rescue
            # worker has to build a teardown provider -- it never ran prepare
            # or reset_start, so it holds no credential from any other source.
            # Storing the descriptor and returning Ack without forwarding left
            # the adapter's provider None, so every cleanup action took the
            # "could not look" branch and reported attempted/terminate-
            # unconfirmed. A sweep then never confirmed teardown and never
            # discarded the descriptor -- and, far worse, a genuinely leaked
            # instance was never terminated, because no provider existed to
            # terminate it. Observed against episode ep-6ea01a117eee.
            assert self._adapter is not None
            if not isinstance(self._adapter, RescuableAdapter):
                # Loud, not silent. An adapter that cannot accept the rescue
                # handoff cannot tear anything down, and returning a clean Ack
                # here would let the sweep report an orderly "attempted" for a
                # resource nobody can release. Checked HERE rather than at
                # load: rescue is optional for an ordinary episode and
                # required only once one is actually requested.
                #
                # Raised as an ADAPTER error, not RpcProtocolError: a protocol
                # error tears the channel down without a reply, so the parent
                # only ever sees an opaque TimeoutError and the operator loses
                # the reason. This path must name itself all the way out to the
                # sweep entry, which is the whole point of failing loudly.
                raise AdapterRescueUnsupported("adapter does not implement begin_rescue")
            result = await self._adapter.begin_rescue(params)
            return cast(ProtocolModel, result)
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
        detail: RpcErrorDetail | None = None,
    ) -> RpcResponse:
        error = RpcError.model_validate(
            {"code": code, "message": message, "detail": detail}, strict=True
        )
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
