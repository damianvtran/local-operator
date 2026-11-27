"""The guardrail decision record and its sink (Radient Guard RFC v0.6, slice L1).

WHAT. Every verdict the guardrail checkpoint reaches (``guardrails/check.py``)
becomes one :class:`GuardDecisionRecord` handed to the process-wide sink. The
sink is a no-op unless something installs another one; slice L1 installs
nothing, so this module only defines the seam and the vocabulary.

WHY A CLOSED VOCABULARY. Guard's ``decisions`` release family counts
``outcome x rule_class`` per closed UTC day (RFC §10'). The record therefore
carries ONLY closed enums plus the UTC day and a random id: no command, path,
argument, target, rule name, rule text, judge reason or content of any kind.
A user rule's NAME is operator-authored free text (``no-push-to-acme-prod``),
so it is folded to :data:`RULE_CLASS_CUSTOM` rather than carried. The record's
constructor refuses any value outside the enums, so a future call site cannot
widen what leaves the checkpoint by passing a string through.

WHY IT CAN NEVER BREAK A CALL. The guardrail is a safety gate; telemetry is
not. :func:`emit_decision` swallows every exception a sink raises (and does
not log its text, which could carry whatever the sink was holding), so a
broken or slow-failing sink can neither refuse nor allow a tool call.

Granularity is the UTC day, not a timestamp: the grid needs nothing finer,
and a per-call time would let an admin line decisions up with other signals.
"""

from __future__ import annotations

import datetime as _dt
import threading
import uuid
from dataclasses import dataclass, field
from typing import Final, Protocol

#: Wire/schema generation of the record. Bump on any field change.
DECISION_SCHEMA: Final = "guard_decision_v1"

# -- outcome ---------------------------------------------------------------
#: A call the guard refused.
OUTCOME_BLOCK: Final = "block"
#: A call that runs with a warning a rule attached.
OUTCOME_WARN: Final = "warn"
#: A call a review (judge, proven evidence, operator words) let through.
OUTCOME_ALLOW: Final = "allow"
#: A call the guard ranked LOW and let through without any review.
OUTCOME_PASS: Final = "pass"
OUTCOMES: Final = frozenset({OUTCOME_BLOCK, OUTCOME_WARN, OUTCOME_ALLOW, OUTCOME_PASS})

# -- rule class ------------------------------------------------------------
RULE_CLASS_DANGEROUS_DELETE: Final = "dangerous_delete"
RULE_CLASS_POLICY_WRITE: Final = "policy_write"
#: Any operator-authored rule. Its name is free text and never leaves the host.
RULE_CLASS_CUSTOM: Final = "custom"
#: The checkpoint itself failed and refused fail-closed.
RULE_CLASS_GUARD_ERROR: Final = "guard_error"
RULE_CLASSES: Final = frozenset(
    {
        RULE_CLASS_DANGEROUS_DELETE,
        RULE_CLASS_POLICY_WRITE,
        RULE_CLASS_CUSTOM,
        RULE_CLASS_GUARD_ERROR,
    }
)

# -- tool class ------------------------------------------------------------
#: Tool names fold to a handful of classes; an unknown or MCP tool is
#: ``other`` so a tool's (possibly user-chosen) name never becomes a value.
_TOOL_CLASS_BY_NAME: Final = {
    "bash": "shell",
    "write": "file_write",
    "edit": "file_write",
    "evidence": "file_write",
    "read": "file_read",
    "grep": "file_read",
    "glob": "file_read",
    "browser": "browser",
    "web_fetch": "web",
    "web_search": "web",
}
TOOL_CLASS_OTHER: Final = "other"
TOOL_CLASSES: Final = frozenset({*_TOOL_CLASS_BY_NAME.values(), TOOL_CLASS_OTHER})


def tool_class_of(tool_name: str) -> str:
    """The closed tool class for a tool name (``other`` when not listed)."""
    return _TOOL_CLASS_BY_NAME.get(tool_name, TOOL_CLASS_OTHER)


def _utc_day() -> str:
    return _dt.datetime.now(_dt.UTC).date().isoformat()


def _new_record_id() -> str:
    # Random, never content-derived: a hash of the call would be a stable
    # fingerprint of it. The id exists only so a retried upload is idempotent.
    return uuid.uuid4().hex


@dataclass(frozen=True)
class GuardDecisionRecord:
    """One guardrail decision, reduced to closed enums.

    ``record_id`` is the idempotency key the outbox and (later) the server
    dedupe on; ``day`` is the UTC calendar day the decision was taken.
    """

    outcome: str
    rule_class: str
    tool_class: str
    day: str = field(default_factory=_utc_day)
    record_id: str = field(default_factory=_new_record_id)
    schema: str = DECISION_SCHEMA

    def __post_init__(self) -> None:
        # Refuse rather than coerce: a value outside the vocabulary is a bug at
        # the call site, and silently mapping it would hide a widening.
        if self.outcome not in OUTCOMES:
            raise ValueError("unknown guard decision outcome")
        if self.rule_class not in RULE_CLASSES:
            raise ValueError("unknown guard rule class")
        if self.tool_class not in TOOL_CLASSES:
            raise ValueError("unknown guard tool class")
        if self.schema != DECISION_SCHEMA:
            raise ValueError("unknown guard decision schema")
        _dt.date.fromisoformat(self.day)  # raises on a malformed day
        if len(self.record_id) != 32 or any(c not in "0123456789abcdef" for c in self.record_id):
            raise ValueError("guard decision record_id must be 32 lowercase hex chars")


class GuardDecisionSink(Protocol):
    """Receives each decision. Must be cheap: it runs on the tool-call path."""

    def emit(self, record: GuardDecisionRecord) -> None: ...


class NoopDecisionSink:
    """The default: drops every record. A non-Team install never leaves it."""

    def emit(self, record: GuardDecisionRecord) -> None:
        return None


NOOP_SINK: Final = NoopDecisionSink()

_lock = threading.Lock()
_sink: GuardDecisionSink = NOOP_SINK


def get_decision_sink() -> GuardDecisionSink:
    return _sink


def set_decision_sink(sink: GuardDecisionSink | None) -> GuardDecisionSink:
    """Install ``sink`` process-wide (``None`` restores the no-op); returns the previous one.

    Process-wide rather than a ContextVar on purpose: admission is a property
    of the install, not of one task, and a sink installed after consent must
    see decisions from every session and subagent the process runs.
    """
    global _sink
    with _lock:
        previous = _sink
        _sink = sink if sink is not None else NOOP_SINK
    return previous


def emit_decision(outcome: str, rule_class: str, tool_class: str) -> None:
    """Build a record and hand it to the sink; never raises.

    The record is built inside the guard too, so even a vocabulary bug at a
    call site degrades to "not recorded" instead of failing the tool call.
    """
    sink = _sink
    if sink is NOOP_SINK:
        return
    try:
        sink.emit(GuardDecisionRecord(outcome, rule_class, tool_class))
    except Exception:  # noqa: BLE001 — telemetry must never decide a call
        return


__all__ = [
    "DECISION_SCHEMA",
    "NOOP_SINK",
    "OUTCOMES",
    "OUTCOME_ALLOW",
    "OUTCOME_BLOCK",
    "OUTCOME_PASS",
    "OUTCOME_WARN",
    "RULE_CLASSES",
    "RULE_CLASS_CUSTOM",
    "RULE_CLASS_DANGEROUS_DELETE",
    "RULE_CLASS_GUARD_ERROR",
    "RULE_CLASS_POLICY_WRITE",
    "TOOL_CLASSES",
    "TOOL_CLASS_OTHER",
    "GuardDecisionRecord",
    "GuardDecisionSink",
    "NoopDecisionSink",
    "emit_decision",
    "get_decision_sink",
    "set_decision_sink",
    "tool_class_of",
]
