"""Opt-in SDK cache hint for one builtin; never a tool or approval grant."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.compaction.idle import CACHE_TTL_5M_S

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Sequence

    from local_operator.harness.types import AgentTool, ChatRequest, StreamEvent
    from local_operator.providers.bedrock.client import _NativeAttempt
    from local_operator.session.request_observation import RequestFinalObservation
    from local_operator.session.session import Session
    from local_operator.session.spec import SessionRoots, SessionSpec

# v1 receipts inferred authority from public usage and selectors. Never read them
# as v2, which binds an explicit native write to the actual dispatch identity.
_VERSION = 2
_FILE = "sdk-warm-tool-prefix.json"
_TOOL = "library"
_TTL_MS = int(CACHE_TTL_5M_S * 1000)
# Two 64-character hashes and two epoch/monotonic millisecond integers, plus
# framing. Bound the reader as well as the writer, including malformed files.
_MAX_BYTES = 512
_FIELDS = {"version", "scope", "publication", "wall_ms", "monotonic_ms"}


def _boot_id() -> str | None:
    """Do not interpret a previous OS boot's monotonic clock in this boot."""
    if sys.platform == "linux":
        with Path("/proc/sys/kernel/random/boot_id").open(encoding="ascii") as handle:
            return handle.read(64).strip() or None
    if sys.platform == "darwin":
        import ctypes

        buf = ctypes.create_string_buffer(64)
        size = ctypes.c_size_t(len(buf))
        sysctl = ctypes.CDLL(None).sysctlbyname
        sysctl.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        if sysctl(b"kern.bootsessionuuid", buf, ctypes.byref(size), None, 0) == 0:
            return buf.value.decode("ascii") or None
    return None


def _digest(value: Any) -> str:
    # Dict order is intentional: native Messages emits properties in that
    # order. Sorting JSON keys would false-match a changed wire prefix.
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _tool_digest(tools: Sequence[AgentTool]) -> str:
    # The same sequence of name/description/input_schema values that native
    # Messages builds from request.tools. Only the commitment ever leaves RAM.
    return _digest(
        [
            (tool.name, tool.description, tool.parameters or {"type": "object", "properties": {}})
            for tool in tools
        ]
    )


def _clock() -> tuple[int, int]:
    return time.time_ns() // 1_000_000, time.monotonic_ns() // 1_000_000


def _warm(receipt: dict[str, Any]) -> bool:
    # The wall clock covers suspend and the same-boot monotonic clock prevents
    # a wall correction, on its own, from extending the lifetime.
    return all(
        type(receipt[key]) is int and 0 <= now - receipt[key] < _TTL_MS
        for key, now in zip(("wall_ms", "monotonic_ms"), _clock(), strict=True)
    )


def _read(path: Path) -> dict[str, Any]:
    # A bad hint is a miss, not a reason to follow a symlink, block on a FIFO,
    # parse an unbounded file, or interfere with owning-session startup.
    if os.name != "posix":
        return {}
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as handle:
        meta = os.fstat(handle.fileno())
        if not stat.S_ISREG(meta.st_mode) or meta.st_uid != os.getuid() or meta.st_mode & 0o022:
            return {}
        data = handle.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        return {}
    result = json.loads(data)
    if not isinstance(result, dict) or result.keys() != _FIELDS:
        return {}
    if type(result["version"]) is not int or result["version"] != _VERSION:
        return {}
    return result


@dataclass(frozen=True)
class _Snapshot:
    scope: str
    epoch: int
    tools: list[AgentTool]


@dataclass(frozen=True)
class _NativeDispatch:
    snapshot: _Snapshot
    scope: str
    publication: str
    wall_ms: int
    monotonic_ms: int
    warm: bool

    def receipt(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "scope": self.scope,
            "publication": self.publication,
            "wall_ms": self.wall_ms,
            "monotonic_ms": self.monotonic_ms,
        }


@dataclass
class _WarmCall:
    """One owning turn request; reset proof for every physical native send."""

    hint: _WarmToolPrefix
    proof: _NativeDispatch | None = None
    attempt: _NativeAttempt | None = None

    def prepare(
        self, request: ChatRequest, region: str, namespace: str | None
    ) -> tuple[list[AgentTool] | None, _NativeDispatch | None]:
        """No awaits: revalidate only at the final native encoding/signing seam."""
        hint = self.hint
        try:
            snap = hint._snapshot()
            if (
                snap is None
                or namespace is None
                or (request.model.provider, request.model.model_id) != hint.route
            ):
                hint.candidate = None
                return None, None
            cold = hint.session._publishable(snap.tools)
            # Never override a transformed request or add a tool that has not
            # been validated in the live inventory. Real discovery stays sticky.
            publication = _tool_digest(request.tools)
            if publication != _tool_digest(cold):
                hint.candidate = None
                return None, None
            scope = _digest((snap.scope, region, namespace))
            candidate = hint.candidate
            warm_tools = None
            if candidate is not None:
                receipt, epoch = candidate
                if (
                    epoch == snap.epoch
                    and receipt["scope"] == scope
                    and _warm(receipt)
                    and not hint.session._activated_tools
                    and _TOOL in hint.session._deferred_now()
                ):
                    preview = hint.session._publishable(snap.tools, activated=frozenset({_TOOL}))
                    if _tool_digest(preview) == receipt["publication"]:
                        warm_tools = preview
                        publication = receipt["publication"]
                if warm_tools is None:
                    hint.candidate = None
            if warm_tools is None and hint.session._activated_tools != {_TOOL}:
                return None, None
            wall, mono = _clock()
            return warm_tools, _NativeDispatch(
                snap,
                scope,
                publication,
                wall,
                mono,
                warm_tools is not None,
            )
        except (OSError, ValueError, TypeError, RecursionError):
            hint.candidate = None
            return None, None

    def ready(self, dispatch: _NativeDispatch) -> bool:
        if not dispatch.warm:
            return True
        hint = self.hint
        candidate = hint.candidate
        try:
            latest = hint._snapshot()
            if (
                candidate is not None
                and latest is not None
                and latest.scope == dispatch.snapshot.scope
                and latest.epoch == candidate[1]
                and not hint.session._activated_tools
                and _warm(candidate[0])
            ):
                return True
        except (OSError, ValueError, TypeError, RecursionError):
            pass
        hint.candidate = None
        return False


class _WarmToolPrefix:
    def __init__(self, session: Session, roots: SessionRoots, world: Path, boot: str) -> None:
        self.session = session
        self.world = world
        self.manager = session.mcp_manager
        self.gate = session._request_approval
        self.route = (session.model.provider, session.model.model_id)
        self.tools = tuple(session._tools)
        self.binding = (
            session.session_id,
            str(roots.config_path),
            str(roots.agent_home_path),
            str(roots.cwd_path),
            str(world),
            boot,
            self.route,
            "sdk/refuse/builtin/library",
        )
        self.path = Path(session._session_dir() or "_") / _FILE
        self.pending: _NativeDispatch | None = None
        self.candidate: tuple[dict[str, Any], int] | None = None
        self.read_on_first_request = False
        self.unsubscribe = session.subscribe_requests(self._observe)

    def _snapshot(self) -> _Snapshot | None:
        s = self.session
        startup = s.mcp_startup
        if (
            not s._tool_deferral
            or s._declared_tools is not None
            or s._yolo
            or s._request_approval is not self.gate
            or s._confinement_root != str(self.world)
            or (s.effective_model.provider, s.effective_model.model_id) != self.route
            or self.manager is None
            or s.mcp_manager is not self.manager
            or startup is None
            or startup.settling
            or startup.failures
        ):
            return None
        epoch = self.manager._empty_catalogue_epoch()
        tools = s._tools
        if (
            epoch is None
            or len(tools) != len(self.tools)
            or any(t is not born for t, born in zip(tools, self.tools))
        ):
            return None
        # A path string alone cannot distinguish a replaced world directory.
        world_stat = self.world.stat()
        scope = _digest(
            (
                self.binding,
                world_stat.st_dev,
                world_stat.st_ino,
                sorted(s._deferral_pins),
                _tool_digest(tools),
            )
        )
        return _Snapshot(scope, epoch, tools)

    def preview(self) -> None:
        try:
            receipt = _read(self.path)
            snap = self._snapshot()
            if receipt and _warm(receipt) and snap is not None:
                self.candidate = receipt, snap.epoch
        except (OSError, ValueError, TypeError, RecursionError):
            return  # This speculative in-memory copy is never a publication.

    async def stream(self, inner: AsyncGenerator[StreamEvent, None]) -> AsyncIterator[StreamEvent]:
        from local_operator.providers.bedrock.client import _warm_call

        if self.read_on_first_request:
            self.read_on_first_request = False
            self.preview()
        call = _WarmCall(self)
        try:
            while True:
                token = _warm_call.set(call)
                try:
                    event = await anext(inner)
                except StopAsyncIteration:
                    return
                finally:
                    _warm_call.reset(token)
                yield event  # Never leave task-local authority installed across a yield.
        finally:
            await inner.aclose()

    def _observe(self, row: RequestFinalObservation) -> None:
        s = self.session
        if (
            row.purpose != "turn"
            or row.session_id != s.session_id
            or row.parent_session_id is not None
        ):
            return
        from local_operator.providers.bedrock.client import _warm_call

        call = _warm_call.get()
        # Only a successful root request with distinct proof from ITS accepted
        # native send can refresh a hint; public recorder coverage remains unchanged.
        if (
            call is not None
            and call.hint is self
            and call.proof is not None
            and row.ok
            and row.outcome == "stop"
            and (row.requested_provider, row.requested_model_id) == self.route
            and (row.served_provider, row.served_model_id) == self.route
        ):
            self.pending = call.proof

    def close(self) -> None:
        self.session._sdk_warm_prefix = None
        self.unsubscribe()
        try:
            if self.pending is None:
                return
            proof = self.pending
            receipt = proof.receipt()
            if not _warm(receipt) or not self.path.parent.is_dir():
                return
            snap = self._snapshot()
            if snap is None or (snap.scope, snap.epoch) != (
                proof.snapshot.scope,
                proof.snapshot.epoch,
            ):
                return
            if (
                self.session._activated_tools - {_TOOL}
                or _tool_digest(self.session._publishable(snap.tools, activated=frozenset({_TOOL})))
                != proof.publication
            ):
                return
            from local_operator.owned_process import _atomic_write_bytes

            _atomic_write_bytes(self.path, json.dumps(receipt, separators=(",", ":")).encode())
        except (OSError, ValueError, TypeError, RecursionError):
            return  # Do not turn a cache optimization into a failed close.


def enrol(
    session: Session, spec: SessionSpec, roots: SessionRoots, world: Path | None
) -> _WarmToolPrefix | None:
    """Admit only an isolated owner with the SDK's default refusal gate."""
    if (
        world is None
        or roots.allow_ambient
        or spec.hosting != "bedrock"
        or spec.approvals.mode != "refuse"
        or spec.tools is not None
        or spec.yolo
        or spec.team
        or spec.profile
        or spec.agent_name
        or spec.agent_id
        or not session._session_dir()
        or session.model.provider != "bedrock"
    ):
        return None
    from local_operator.providers.bedrock.models import capabilities_for_spec
    from local_operator.tools.registry import TOOL_BUILDERS

    if capabilities_for_spec(session.model).protocol != "anthropic" or any(
        tool.name not in TOOL_BUILDERS for tool in session._tools
    ):
        return None
    try:
        boot = _boot_id()
        if boot is None:
            return None
        hint = _WarmToolPrefix(session, roots, world, boot)
    except (OSError, ValueError):
        return None
    hint.read_on_first_request = bool(spec.resume)
    session._sdk_warm_prefix = hint
    return hint
