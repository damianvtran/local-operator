"""The delivery envelope (contract §9.1).

One material delta arrives as a ``monitor_prompt`` custom message. This module
owns the text: the envelope line (id, name, counts, skipped/held clauses), the
optional description and source, the cancel hint, and the bounded delta body
the scheduler rendered. Mirrors ``format_wake_delivery_text`` — id and name
ride the first line because an agent that cannot name its own monitor cannot
cancel it; the counts are absolute; the cancel hint is dropped in the final
delivery once ``until`` is within one interval (the monitor will not check
again, so "cancel once its goal is met" no longer applies).

Envelope rules, measured (§17): envelope ≈ 163 chars, representative body
≈ 668, worst case ≈ 1,406 — the "~≤500 tokens" budget, hit per message. One
message per material monitor: a fold could not name one id or one cancel hint,
and each message carries its own delta budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

#: The clause the source note uses when the tool is not obvious from the name.
#: "Not obvious" is a simple, testable predicate: the monitor's name contains
#: the tool's name (case-insensitive) or it does not.
SOURCE_NOTE_TEMPLATE = " (via {tool})"

CLOCK_FORMAT = "%H:%M"

#: The lifecycle notice's whole-text budget. A notice is a PUSH into the
#: conversation (unlike the list surfaces, which are read on demand), so it is
#: one bounded block: a long failure reason is clipped rather than allowed to
#: turn a disable into a wall of text, and the total is asserted in tests. The
#: live shape is ~380-560 chars.
NOTICE_MAX_CHARS = 700

#: How much of a stored ``last_error`` a notice repeats. A transport error can
#: carry a whole traceback; the first line is what names the failure.
NOTICE_ERROR_CHARS = 200


@dataclass(frozen=True)
class MonitorDelivery:
    """Everything one delivery message needs, computed by the scheduler."""

    monitor_id: str
    name: str
    tool: str
    changes: int
    checks: int
    skipped: int
    delta_text: str
    at_ms: int
    held_by_cap: int = 0
    final: bool = False
    description: str = ""
    #: §14.4: copied from the spec so the session can read it off the delivered
    #: message's ``details``. Default False (a monitor is quiet unless armed
    #: with notify), so existing construction sites keep today's behaviour.
    notify: bool = False


def format_monitor_delivery_text(delivery: MonitorDelivery) -> str:
    """The complete model-facing text of one monitor delivery."""
    clock = datetime.fromtimestamp(delivery.at_ms / 1000).strftime(CLOCK_FORMAT)
    source_note = (
        "" if delivery.tool.lower() in delivery.name.lower() else f" (via {delivery.tool})"
    )
    clauses: list[str] = []
    if delivery.skipped:
        clauses.append(f"{delivery.skipped} skipped while the session was down")
    if delivery.held_by_cap:
        clauses.append(f"{delivery.held_by_cap} earlier changes were held by the hourly cap")
    clause_text = f" ({'; '.join(clauses)})" if clauses else ""

    plural = "" if delivery.changes == 1 else "s"
    lines = [
        f"(monitor) '{delivery.name}' {delivery.monitor_id}{source_note}: "
        f"{delivery.changes} change{plural} at {clock} — check {delivery.checks}{clause_text}."
    ]
    if delivery.description:
        lines.append(f"Watching for: {delivery.description}")
    if not delivery.final:
        lines.append(
            f'Cancel with monitor({{op:"cancel",id:"{delivery.monitor_id}"}}) once its goal is met.'
        )
    lines.append("")
    lines.append("Diff vs the previous check:")
    lines.append(delivery.delta_text)
    return "\n".join(lines)


#: The lifecycle kinds a notice carries. ``delta`` is the ordinary delivery and
#: is NOT one of these: it rides ``MonitorDelivery``.
NOTICE_KINDS = ("disabled", "stalled", "restored")


@dataclass(frozen=True)
class MonitorNotice:
    """One lifecycle notice about a monitor: disabled, stalled, restored.

    Deliberately NOT a :class:`MonitorDelivery`: a notice carries no delta and
    no ``changes``, it never counts as a delivery (§9.4's rate window is for
    material changes), and it exists because the operator's live store showed
    monitors silently auto-disabling — the disable itself was durable state
    that nothing ever told anyone about.
    """

    monitor_id: str
    name: str
    tool: str
    kind: str
    at_ms: int
    checks: int = 0
    deliveries: int = 0
    failures: int = 0
    #: The failure that caused a disable, or the reason a stall is not
    #: self-healing (auth-required names its own fix).
    detail: str = ""
    #: §14.4, the delivery's control parameter: only the disable notice
    #: defaults it on, because a disabled watch is the one notice that needs
    #: the operator's attention.
    notify: bool = False


def format_monitor_notice_text(notice: MonitorNotice) -> str:
    """The complete model-facing text of one lifecycle notice."""
    clock = datetime.fromtimestamp(notice.at_ms / 1000).strftime(CLOCK_FORMAT)
    source_note = "" if notice.tool.lower() in notice.name.lower() else f" (via {notice.tool})"
    who = f"'{notice.name}' {notice.monitor_id}{source_note}"

    if notice.kind == "disabled":
        lines = [
            f"(monitor) {who} was DISABLED at {clock} after {notice.failures} consecutive "
            "failed checks — it is no longer watching."
        ]
        if notice.detail:
            lines.append(f"Last error: {_clip(notice.detail)}")
        if notice.deliveries == 0:
            lines.append(
                f"It never delivered a change since arming ({notice.checks} checks). "
                "The call may not observe what you expected."
            )
        else:
            plural = "" if notice.deliveries == 1 else "s"
            lines.append(f"{notice.deliveries} deliver{plural} of change so far.")
        lines.append(
            'To restore: re-create the same call with monitor({op:"create",…}) '
            f'(reactivates it) or cancel it with monitor({{op:"cancel",id:"{notice.monitor_id}"}}).'
        )
        lines.append("Monitors tick only while this session is open.")
        return _bounded("\n".join(lines))

    if notice.kind == "stalled":
        detail = f" ({_clip(notice.detail)})" if notice.detail else ""
        return _bounded(
            f"(monitor) {who} could not run its check at {clock}{detail} — it is retrying, "
            "without counting failures.\n"
            "Its baseline is unchanged, so the next successful check reports everything "
            "it missed as one delta. Cancel with "
            f'monitor({{op:"cancel",id:"{notice.monitor_id}"}}) if unwanted.'
        )

    # restored
    return _bounded(
        f"(monitor) {who} is running again as of {clock} — the earlier interruption has ended.\n"
        f"The next check diffs against the old baseline, so changes during the gap arrive as "
        f'one delta. Cancel with monitor({{op:"cancel",id:"{notice.monitor_id}"}}) once its '
        "goal is met."
    )


def _clip(text: str) -> str:
    clipped = " ".join(str(text or "").split())
    if len(clipped) <= NOTICE_ERROR_CHARS:
        return clipped
    return clipped[: NOTICE_ERROR_CHARS - 1] + "…"


def _bounded(text: str) -> str:
    """Enforce ``NOTICE_MAX_CHARS`` as a hard bound (tests assert it)."""
    if len(text) <= NOTICE_MAX_CHARS:
        return text
    return text[: NOTICE_MAX_CHARS - 1] + "…"
