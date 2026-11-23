"""Opt-in SDK cache hint for one builtin; never a tool or approval grant."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.compaction.idle import CACHE_TTL_5M_S

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_operator.harness.types import AgentTool
    from local_operator.session.request_observation import RequestFinalObservation
    from local_operator.session.session import Session
    from local_operator.session.spec import SessionRoots, SessionSpec

# Version 1 defines the origin and activation implicitly: shipped builtin
# `library`, no other activation. There is no decoder for a name or MCP origin.
_VERSION = 1
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
    return _digest([
        (tool.name, tool.description, tool.parameters or {"type": "object", "properties": {}})
        for tool in tools
    ])


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
        self.pending: tuple[dict[str, Any], int] | None = None
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
        epoch = self.manager.empty_catalogue_epoch()
        tools = s._tools
        if epoch is None or len(tools) != len(self.tools) or any(
            t is not born for t, born in zip(tools, self.tools)
        ):
            return None
        # A path string alone cannot distinguish a replaced world directory.
        world_stat = self.world.stat()
        scope = _digest((
            self.binding,
            world_stat.st_dev,
            world_stat.st_ino,
            sorted(s._deferral_pins),
            _tool_digest(tools),
        ))
        return _Snapshot(scope, epoch, tools)

    def restore(self) -> tuple[AgentTool, ...] | None:
        """One synchronous preview+commit, before the new owner's first latch."""
        try:
            receipt = _read(self.path)
            if not receipt or not _warm(receipt) or self.session._activated_tools:
                return None
            snap = self._snapshot()
            if snap is None or receipt["scope"] != snap.scope:
                return None
            s = self.session
            if _TOOL not in s._deferred_now() or not any(t.name == _TOOL for t in snap.tools):
                return None
            preview = tuple(s._publishable(snap.tools, activated=frozenset({_TOOL})))
            if _tool_digest(preview) != receipt["publication"]:
                return None
            # No awaits, no connection work, no resolver (which could stamp a
            # use or install permissions). Check the same live generation and
            # commit only the builtin bit. Normal publication does the latch.
            latest = self._snapshot()
            if (
                latest is not None
                and latest.scope == snap.scope
                and latest.tools is snap.tools
                and latest.epoch == snap.epoch
                and not s._activated_tools
                and _warm(receipt)
            ):
                s._activated_tools.add(_TOOL)
                return preview
        except (OSError, ValueError, TypeError, RecursionError):
            return None  # Optional evidence never breaks an ordinary turn.
        return None

    def _observe(self, row: RequestFinalObservation) -> None:
        s = self.session
        if (
            row.purpose != "turn"
            or row.session_id != s.session_id
            or row.parent_session_id is not None
        ):
            return
        # The last *root turn* must have attested an explicit 5m cache write.
        # Read-only receipts, aggregate estimates, transcript usage and a
        # successful restore are not clocks that can refresh this hint.
        self.pending = None
        usage = row.usage
        if (
            not row.ok
            or row.outcome != "stop"
            or row.usage_coverage != "reported_final"
            or (row.requested_provider, row.requested_model_id) != self.route
            or (row.served_provider, row.served_model_id) != self.route
            or usage is None
            or not 0 < usage.cache_write_5m_tokens == usage.cache_write_tokens
            or usage.cache_write_1h_tokens
            or type(row.at_ms) is not int
            or not math.isfinite(row.duration_ms)
            or row.duration_ms < 0
        ):
            return
        wall, mono = _clock()
        try:
            snap = self._snapshot()
            if snap is None or s._activated_tools != {_TOOL} or not s._published_tools:
                return
            publication = _tool_digest(s._published_tools)
            if publication != _tool_digest(s._publishable(snap.tools)):
                return  # A mid-turn activation was never advertised on this request.
            elapsed = math.ceil(row.duration_ms)
            self.pending = (dict(
                version=_VERSION,
                scope=snap.scope,
                publication=publication,
                wall_ms=min(row.at_ms, wall - elapsed),
                monotonic_ms=mono - elapsed,
            ), snap.epoch)
        except (OSError, ValueError, TypeError, RecursionError):
            return

    def close(self) -> None:
        self.session._on_first_tool_publication = None
        self.unsubscribe()
        try:
            if self.pending is None:
                return
            receipt, epoch = self.pending
            if not _warm(receipt) or not self.path.parent.is_dir():
                return
            snap = self._snapshot()
            if snap is None or receipt["scope"] != snap.scope or epoch != snap.epoch:
                return
            if _tool_digest(self.session._publishable(snap.tools)) != receipt["publication"]:
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
    if spec.resume:
        session._on_first_tool_publication = hint.restore
    return hint
