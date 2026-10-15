"""Machine-local, success-derived account leases, never a cache-hit promise.

Receipts do not travel in transcripts or mutate the credential database. A
copied session, recreated row or different auth root therefore cannot inherit
an unrelated account binding. Request construction supplies the actual wire
lifetime; selection/inspection are read-only and cannot create sliding leases.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from local_operator.harness.types import ChatRequest

logger = logging.getLogger(__name__)
_SCHEMA = 1
_MAX_TTL_MS = 60 * 60_000
# Persisted durations must stay inside the same endpoint-specific evidence
# envelope as their original wire hints, not merely a generous global cap.
_LIFETIMES = {
    "anthropic:messages": {(300_000, "explicit-wire"), (3_600_000, "explicit-wire")},
    "openai:responses": {
        (1_800_000, "documented-default"),
        (1_800_000, "bounded-heuristic"),
        (300_000, "bounded-heuristic"),
    },
    "openai:chat": {(300_000, "bounded-heuristic")},
    "codex:responses": {(300_000, "bounded-heuristic")},
}
_MAX_RECEIPT_BYTES = 64 * 1024

# Task-local selection context cannot leak an old request into inspection or a
# concurrent isolated errand. The wire hint is set only by built-in builders.
_selection_signature: ContextVar[str | None] = ContextVar(
    "account_selection_signature", default=None
)
_wire_hint: ContextVar[CacheAffinityHint | None] = ContextVar("account_wire_affinity", default=None)
_selection_policy: ContextVar[str | None] = ContextVar("account_selection_policy", default=None)
_isolated_request: ContextVar[bool] = ContextVar("account_isolated_request", default=False)


@dataclass
class AccountRequestBoundary:
    """Tentative picks for ONE active request, never successful cache evidence.

    The single session owner replaces this object at request entry; identity-
    checked cleanup cannot clear a newer request when an old generator closes.
    No timestamps slide and nothing here is written to disk.
    """

    provider: str
    model_id: str
    compatibility: str
    picks: dict[str, int] = field(default_factory=dict)


def request_compatibility(request: "ChatRequest") -> str:
    """Hash compatibility inputs, never store a prompt or replay a request body.

    This is a conservative settings/prefix identity, not exact cache-token proof.
    The initial message changes after compaction; later appended turns do not
    destroy affinity to an otherwise still-useful shared prefix.
    """
    spec = request.model
    payload = {
        "provider": spec.provider,
        "model": spec.model_id,
        "endpoint": spec.base_url,
        "responses": spec.supports_responses_api,
        "reasoning": spec.reasoning_effort,
        "fast": spec.fast_mode,
        "effort_override": request.effort_override,
        "effort_ceiling": request.effort_ceiling,
        "system": request.system_blocks,
        "tools": [tool.model_dump(mode="json") for tool in request.tools],
        "tool_choice": request.tool_choice,
        "first_message": (
            request.messages[0].model_dump(
                mode="json", include={"role", "content", "tool_calls", "tool_call_id", "tool_name"}
            )
            if request.messages
            else None
        ),
        "cache_key": request.prompt_cache_key,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@contextmanager
def selection_context(request: "ChatRequest", policy: str | None = None):
    token = _selection_signature.set(request_compatibility(request))
    policy_token = _selection_policy.set(policy)
    isolated_token = _isolated_request.set(request.isolated)
    try:
        yield
    finally:
        _selection_signature.reset(token)
        _selection_policy.reset(policy_token)
        _isolated_request.reset(isolated_token)


def current_selection_signature() -> str | None:
    return _selection_signature.get()


def current_selection_policy() -> str | None:
    return _selection_policy.get()


def current_request_isolated() -> bool:
    return _isolated_request.get()


def clear_wire_hint() -> None:
    _wire_hint.set(None)


def wire_hint() -> "CacheAffinityHint | None":
    return _wire_hint.get()


def set_wire_hint(
    request: "ChatRequest",
    endpoint_mode: str,
    ttl_ms: int,
    source: Literal["explicit-wire", "documented-default", "bounded-heuristic"],
) -> None:
    if source not in ("explicit-wire", "documented-default", "bounded-heuristic"):
        return
    _wire_hint.set(CacheAffinityHint(endpoint_mode, request_compatibility(request), ttl_ms, source))


@dataclass(frozen=True)
class CacheAffinityHint:
    """Locally derived wire facts, not provider-controlled response content."""

    endpoint_mode: str
    compatibility: str
    ttl_ms: int
    ttl_source: Literal["explicit-wire", "documented-default", "bounded-heuristic"]


@dataclass(frozen=True)
class AccountAffinityReceipt:
    schema: int
    session_id: str
    provider: str
    namespace: str
    account_id: int
    account_created_at: int
    model_id: str
    endpoint_mode: str
    compatibility: str
    sent_at_ms: int
    completed_at_ms: int
    expires_at_ms: int
    ttl_source: str

    def valid(self, now_ms: int) -> bool:
        times = (self.sent_at_ms, self.completed_at_ms, self.expires_at_ms)
        return (
            type(self.schema) is int
            and self.schema == _SCHEMA
            and all(type(value) is int and value > 0 for value in times)
            and self.sent_at_ms <= self.completed_at_ms <= now_ms
            and self.sent_at_ms <= now_ms < self.expires_at_ms
            and 0 < self.expires_at_ms - self.sent_at_ms <= _MAX_TTL_MS
            and type(self.account_id) is int
            and self.account_id > 0
            and type(self.account_created_at) is int
            and 0 < self.account_created_at <= self.sent_at_ms
            and all(
                isinstance(value, str) and 0 < len(value) <= maximum
                for value, maximum in (
                    (self.session_id, 1024),
                    (self.provider, 128),
                    (self.namespace, 128),
                    (self.model_id, 256),
                    (self.endpoint_mode, 256),
                    (self.compatibility, 128),
                )
            )
            and isinstance(self.ttl_source, str)
            and (self.expires_at_ms - self.sent_at_ms, self.ttl_source)
            in _LIFETIMES.get(self.endpoint_mode, set())
            and self.ttl_source in ("explicit-wire", "documented-default", "bounded-heuristic")
        )


class AccountAffinityStore:
    """One local owner writes its bounded provider map atomically per session.

    This deliberately does not introduce auth-DB bootstrap/locking changes.
    File failure degrades to memory, explicitly losing restart continuity; no
    failed persistence operation is allowed to replay successful inference.
    """

    def __init__(self, auth_path: Path):
        self.namespace = hashlib.sha256(str(auth_path.resolve()).encode()).hexdigest()
        self.root = auth_path.parent / "account-affinity" / self.namespace
        self._loaded: set[str] = set()
        self._receipts: dict[str, dict[str, AccountAffinityReceipt]] = {}
        self._deadlines: dict[tuple[str, str], float] = {}

    def path_for(self, session_id: str) -> Path:
        # Session ids are opaque input, not path components supplied by callers.
        digest = hashlib.sha256(session_id.encode()).hexdigest()
        return self.root / f"{digest}.json"

    def _load(self, session_id: str) -> dict[str, AccountAffinityReceipt]:
        if session_id in self._loaded:
            return self._receipts[session_id]
        self._loaded.add(session_id)
        receipts: dict[str, AccountAffinityReceipt] = {}
        self._receipts[session_id] = receipts
        try:
            path = self.path_for(session_id)
            with path.open("rb") as stream:
                raw = stream.read(_MAX_RECEIPT_BYTES + 1)
            if len(raw) > _MAX_RECEIPT_BYTES:
                return receipts
            payload = json.loads(raw)
            if (
                not isinstance(payload, dict)
                or type(payload.get("schema")) is not int
                or payload.get("schema") != _SCHEMA
            ):
                return receipts
            records = payload.get("providers")
            if not isinstance(records, dict) or len(records) > 128:
                return receipts
            for provider, record in records.items():
                if not isinstance(provider, str) or not isinstance(record, dict):
                    continue
                try:
                    receipt = AccountAffinityReceipt(**record)
                except (TypeError, ValueError):
                    continue
                if (
                    receipt.namespace == self.namespace
                    and receipt.session_id == session_id
                    and receipt.provider == provider
                    and receipt.valid(int(time.time() * 1000))
                ):
                    receipts[provider] = receipt
                    self._deadlines[(session_id, provider)] = (
                        time.monotonic() + (receipt.expires_at_ms - time.time() * 1000) / 1000
                    )
        except (OSError, ValueError, TypeError):
            pass  # Missing/corrupt receipts are a policy re-pick, not an outage.
        return receipts

    def get(
        self,
        session_id: str,
        provider: str,
        model_id: str,
        compatibility: str,
        *,
        now_ms: int | None = None,
        monotonic: float | None = None,
    ) -> AccountAffinityReceipt | None:
        receipt = self._load(session_id).get(provider)
        now = int(time.time() * 1000) if now_ms is None else now_ms
        mono = time.monotonic() if monotonic is None else monotonic
        deadline = self._deadlines.get((session_id, provider), -math.inf)
        if (
            receipt is None
            or not receipt.valid(now)
            or mono >= deadline
            or receipt.model_id != model_id
            or receipt.compatibility != compatibility
        ):
            return None
        return receipt

    def record(
        self,
        session_id: str,
        provider: str,
        account_id: int,
        account_created_at: int,
        model_id: str,
        hint: CacheAffinityHint,
        *,
        sent_at_ms: int,
        sent_monotonic: float,
        completed_at_ms: int | None = None,
    ) -> bool:
        now = int(time.time() * 1000) if completed_at_ms is None else completed_at_ms
        if type(hint.ttl_ms) is not int or not 0 < hint.ttl_ms <= _MAX_TTL_MS:
            return False
        if not math.isfinite(sent_monotonic) or not 0 <= sent_monotonic <= time.monotonic():
            return False
        receipt = AccountAffinityReceipt(
            _SCHEMA,
            session_id,
            provider,
            self.namespace,
            account_id,
            account_created_at,
            model_id,
            hint.endpoint_mode,
            hint.compatibility,
            sent_at_ms,
            now,
            sent_at_ms + hint.ttl_ms,
            hint.ttl_source,
        )
        if not receipt.valid(now):
            return False
        records = self._load(session_id)
        previous = records.get(provider)
        if previous is not None and previous.valid(now) and previous.sent_at_ms > sent_at_ms:
            # A slow old response cannot switch a newer account/prefix. For the
            # exact same compatible identity, retain a genuinely longer observed
            # wire lease rather than shortening it merely by completion order.
            same_identity = (
                previous.account_id == account_id
                and previous.account_created_at == account_created_at
                and previous.model_id == model_id
                and previous.endpoint_mode == hint.endpoint_mode
                and previous.compatibility == hint.compatibility
            )
            if not same_identity or receipt.expires_at_ms <= previous.expires_at_ms:
                return False
        if (
            previous is not None
            and previous.valid(now)
            and previous.account_id == account_id
            and previous.account_created_at == account_created_at
            and previous.model_id == model_id
            and previous.compatibility == hint.compatibility
            and previous.expires_at_ms >= receipt.expires_at_ms
        ):
            # A shorter compatible wire lease does not erase a longer useful
            # success already observed for the same immutable account.
            return False
        records[provider] = receipt
        self._deadlines[(session_id, provider)] = sent_monotonic + hint.ttl_ms / 1000
        self._save(session_id, records)
        return True

    def invalidate(
        self,
        session_id: str,
        provider: str,
        *,
        account_id: int | None = None,
        sent_at_ms: int | None = None,
    ) -> None:
        records = self._load(session_id)
        receipt = records.get(provider)
        if receipt is None:
            return
        if account_id is not None and receipt.account_id != account_id:
            return
        if sent_at_ms is not None and receipt.sent_at_ms > sent_at_ms:
            return
        if records.pop(provider, None) is not None:
            self._deadlines.pop((session_id, provider), None)
            self._save(session_id, records)

    def _save(self, session_id: str, records: dict[str, AccountAffinityReceipt]) -> None:
        temporary: str | None = None
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            payload = json.dumps(
                {
                    "schema": _SCHEMA,
                    "providers": {
                        provider: asdict(receipt) for provider, receipt in records.items()
                    },
                },
                sort_keys=True,
            )
            # Same private tempfile+replace pattern used by monitors/state.py;
            # a partial write must never turn a previous receipt into a lease.
            fd, temporary = tempfile.mkstemp(prefix=".affinity-", dir=self.root)
            with os.fdopen(fd, "w") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path_for(session_id))
            temporary = None
        except OSError:
            logger.warning("Account affinity retained in memory only; local receipt write failed")
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
