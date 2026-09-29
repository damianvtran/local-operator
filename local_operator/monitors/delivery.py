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
