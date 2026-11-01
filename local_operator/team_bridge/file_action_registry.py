"""INERT building block for a future human-file action authority.

Nothing here establishes provenance or grants access. In particular, an arbitrary
renderer-supplied binding must NEVER be commissioned. No route may use this
registry until a core-owned authenticated member-response seam, lifecycle
observation and main-process invalidation are integrated and independently
reviewed. The hub remains authoritative for current membership.

This store contains metadata only, not credentials, bytes, filenames or URLs.
Handles are process-local capabilities and must not be logged. A process restart
creates a new namespace. Main invalidation and core revocation are separate
cross-process events, not an instantaneous distributed revocation guarantee.
"""

from __future__ import annotations

import math
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

_MAX_CONTEXTS = 256
_MAX_OPERATIONS = 256
_MAX_FILES = 16
_MAX_TTL_SECONDS = 60.0
_MAX_WINDOW_TOMBSTONES = 256


class ActionUnavailable(ValueError):
    """Generic refusal: never reflect handles, resource identifiers or data."""

    def __init__(self) -> None:
        super().__init__("File action is unavailable")


@dataclass(frozen=True, repr=False)
class InertActionBinding:
    """Untrusted until commissioned by a future authenticated core-only seam."""

    credential_id: int
    account_id: str
    login_incarnation: str
    tenant_id: str
    window_id: str
    view_generation: int
    conversation_id: str
    message_id: str
    file_ids: frozenset[str]


@dataclass(frozen=True, repr=False)
class InertOperationView:
    binding: InertActionBinding
    file_id: str


@dataclass(repr=False)
class _Context:
    boot: str
    binding: InertActionBinding
    expires: float


@dataclass(repr=False)
class _Operation:
    context: str
    file_id: str
    claimed: bool = False


def _bounded_text(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256 and not any(ord(c) < 32 for c in value)


def _valid_binding(binding: object) -> bool:
    if not isinstance(binding, InertActionBinding):
        return False
    return (
        type(binding.credential_id) is int and 0 < binding.credential_id <= 2**63 - 1
        and type(binding.view_generation) is int and 0 <= binding.view_generation <= 2**63 - 1
        and all(_bounded_text(value) for value in (
            binding.account_id, binding.tenant_id, binding.window_id,
            binding.conversation_id, binding.message_id,
        ))
        and isinstance(binding.login_incarnation, str)
        and len(binding.login_incarnation) == 32
        and all(char in "0123456789abcdef" for char in binding.login_incarnation)
        and isinstance(binding.file_ids, frozenset)
        and 0 < len(binding.file_ids) <= _MAX_FILES
        and all(_bounded_text(file_id) for file_id in binding.file_ids)
    )


class InertFileActionRegistry:
    """Bounded, in-memory capability mechanics; NOT an authority or route.

    A single lock protects issue, claim, read and revoke. `current` rereads the
    registry, rather than trusting a caller's earlier snapshot; it cannot observe
    auth, membership or another process unless a future integration revokes it.
    An operation is claimed once, may be checked while in flight, and is removed
    on cancellation, completion, context expiry or revocation.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._boot = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._contexts: dict[str, _Context] = {}
        self._operations: dict[str, _Operation] = {}
        # Never evict in the active epoch: that would resurrect old generations.
        # Saturation revokes every capability and rotates the namespace instead.
        self._invalidated_through: dict[str, int] = {}
        self._sealed = False

    def _require_open(self) -> None:
        if self._sealed:
            raise ActionUnavailable()

    def current_epoch(self) -> str:
        """For the trusted scope owner only; never refresh a stale renderer scope."""
        with self._lock:
            self._require_open()
            return self._boot

    def require_epoch(self, expected_epoch: str) -> None:
        with self._lock:
            self._require_open()
            if not isinstance(expected_epoch, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", expected_epoch) is None or not secrets.compare_digest(self._boot, expected_epoch):
                raise ActionUnavailable()

    def _rotate_epoch(self) -> None:
        # Called with the insertion/invalidation lock held. Any failure leaves
        # the registry sealed; no old handle can become valid in a new epoch.
        self._sealed = True
        new_boot = secrets.token_urlsafe(32)
        if new_boot == self._boot:
            raise ActionUnavailable()
        self._contexts.clear()
        self._operations.clear()
        self._invalidated_through.clear()
        self._boot = new_boot
        self._sealed = False

    def _now(self) -> float:
        value = self._clock()
        if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not -1e15 <= value <= 1e15 or not math.isfinite(value)):
            raise ActionUnavailable()
        return float(value)

    def _drop_context(self, handle: str) -> None:
        self._contexts.pop(handle, None)
        for operation in [key for key, value in self._operations.items() if value.context == handle]:
            del self._operations[operation]

    def _prune(self, now: float) -> None:
        for handle in [key for key, value in self._contexts.items() if value.expires <= now]:
            self._drop_context(handle)

    def _fresh_handle(self) -> str:
        # Collision is fantastically unlikely but must never overwrite an existing capability.
        for _ in range(4):
            handle = f"{self._boot}.{secrets.token_urlsafe(32)}"
            if handle not in self._contexts and handle not in self._operations:
                return handle
        raise ActionUnavailable()

    def commission(self, binding: InertActionBinding, *, ttl_seconds: float, expected_epoch: str) -> str:
        """Store a candidate only; callers cannot infer member-response provenance."""
        if (not _valid_binding(binding) or isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float)) or not 0 < ttl_seconds <= _MAX_TTL_SECONDS
            or not math.isfinite(ttl_seconds)):
            raise ActionUnavailable()
        with self._lock:
            self._require_open()
            if not isinstance(expected_epoch, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", expected_epoch) is None or not secrets.compare_digest(self._boot, expected_epoch):
                raise ActionUnavailable()
            # The trusted view owner must use monotonic generations and call
            # invalidate_view before it reports invalidation complete. This
            # check and watermark publication share the same insertion lock.
            if binding.view_generation <= self._invalidated_through.get(binding.window_id, -1):
                raise ActionUnavailable()
            now = self._now()
            self._prune(now)
            if len(self._contexts) >= _MAX_CONTEXTS or not math.isfinite(now + ttl_seconds):
                raise ActionUnavailable()
            handle = self._fresh_handle()
            self._contexts[handle] = _Context(self._boot, binding, now + ttl_seconds)
            return handle

    def inspect_context(self, context: str) -> InertActionBinding:
        """Re-read a live candidate before starting an operation; not external authority."""
        if not isinstance(context, str):
            raise ActionUnavailable()
        with self._lock:
            self._require_open()
            self._prune(self._now())
            row = self._contexts.get(context)
            if row is None or row.boot != self._boot:
                raise ActionUnavailable()
            return row.binding

    def start(self, context: str, file_id: str) -> str:
        """Create one claimable operation only for a file in the candidate set."""
        if not isinstance(context, str) or not isinstance(file_id, str):
            raise ActionUnavailable()
        with self._lock:
            self._require_open()
            self._prune(self._now())
            row = self._contexts.get(context)
            if row is None or row.boot != self._boot or file_id not in row.binding.file_ids:
                raise ActionUnavailable()
            if len(self._operations) >= _MAX_OPERATIONS:
                raise ActionUnavailable()
            handle = self._fresh_handle()
            self._operations[handle] = _Operation(context, file_id)
            return handle

    def inspect_operation(self, operation: str) -> InertOperationView:
        """Read a live operation for trusted-owner cancellation without claiming it."""
        if not isinstance(operation, str):
            raise ActionUnavailable()
        with self._lock:
            self._require_open()
            self._prune(self._now())
            row = self._operations.get(operation)
            context = self._contexts.get(row.context) if row else None
            if row is None or context is None or context.boot != self._boot:
                raise ActionUnavailable()
            return InertOperationView(context.binding, row.file_id)

    def claim(self, operation: str) -> InertOperationView:
        """Consume the right to start this operation exactly once."""
        if not isinstance(operation, str):
            raise ActionUnavailable()
        with self._lock:
            self._require_open()
            self._prune(self._now())
            row = self._operations.get(operation)
            context = self._contexts.get(row.context) if row else None
            if row is None or context is None or context.boot != self._boot or row.claimed:
                raise ActionUnavailable()
            row.claimed = True
            return InertOperationView(context.binding, row.file_id)

    def current(self, operation: str) -> InertOperationView:
        """Re-read a claimed operation's current registry state; no external authority implied."""
        if not isinstance(operation, str):
            raise ActionUnavailable()
        with self._lock:
            self._require_open()
            self._prune(self._now())
            row = self._operations.get(operation)
            context = self._contexts.get(row.context) if row else None
            if row is None or not row.claimed or context is None or context.boot != self._boot:
                raise ActionUnavailable()
            return InertOperationView(context.binding, row.file_id)

    def cancel(self, operation: str) -> None:
        """Idempotently revoke a pending or in-flight operation."""
        if not isinstance(operation, str):
            raise ActionUnavailable()
        with self._lock:
            self._operations.pop(operation, None)

    def revoke(self, context: str) -> None:
        """Idempotently revoke a context and its operations."""
        if not isinstance(context, str):
            raise ActionUnavailable()
        with self._lock:
            self._drop_context(context)

    def invalidate_view(self, window_id: str, view_generation: int) -> None:
        """Retire this and older generations for a core-owned, boot-unique window.

        Generations must increase monotonically and window IDs must not be reused
        within this registry's boot. The caller must coordinate the real view
        transition with this method; no cross-process instant revocation is implied.
        A full watermark table revokes all operations and rotates the epoch,
        never evicts a live-epoch tombstone. Fresh work requires a newly
        authenticated core-owned scope bound to the new epoch.
        """
        if not _bounded_text(window_id) or type(view_generation) is not int or not 0 <= view_generation <= 2**63 - 1:
            raise ActionUnavailable()
        with self._lock:
            self._require_open()
            if window_id not in self._invalidated_through and len(self._invalidated_through) >= _MAX_WINDOW_TOMBSTONES:
                self._rotate_epoch()
            previous = self._invalidated_through.get(window_id, -1)
            self._invalidated_through[window_id] = max(previous, view_generation)
            for handle in [key for key, row in self._contexts.items()
                           if row.binding.window_id == window_id and row.binding.view_generation <= view_generation]:
                self._drop_context(handle)
