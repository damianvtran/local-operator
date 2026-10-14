"""The Hub links overlay — every link, its source/scope, and its independent policy.

Why an overlay and not a transcript block: a link inventory is REFERENCE the
user acts on in the same breath (toggle Subscribe, toggle Auto-push, retry one
direction). Appended as a block it scrolls away under the next turn, and the
controls it offers are the entire point. So it holds still like ``/usage`` and
``/goal``, opens on the ``/hub`` keystroke, and takes no rows from the
conversation.

WHY A PURE CORE. Everything above the widget is a function of
``(links, width, now)``; the widget owns only the scroll offset and which item
is focused. The sidebar requirement is specifically about what is RENDERED, so
the rendered rows are produced and tested without a running app — the same split
``usage_panel`` uses, and the reason a screenshot check can trust the rows.

ONE SLOT PER ITEM. The design freezes a single fixed status/action slot per
linked item, never a row of toggle icons plus badges. At rest a healthy link
KEEPS a quiet mark (muted ink) whose meaning is source+scope+policy; the slot
communicates the AGGREGATE state by a fixed priority (blocked > failure >
available > in-progress > applied). Gamma must not carry private/public policy
text into the collapsed name, and a poll must never change a row's height.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rich.cells import cell_len
from rich.text import Text

from local_operator.tui.widgets.tool_card import truncate_cells

#: The single status/action slot: ONE cell, reserved so a poll-driven state
#: change never reflows the row (the design's "reserve the existing slot").
SLOT_IDLE = "·"  # a healthy link still shows a quiet mark, never nothing
SLOT_AVAILABLE = "↓"  # a safe incoming update is available (direct apply)
SLOT_FAILURE = "!"  # retryable failure (retry its exact source/direction)
SLOT_BUSY = "◐"  # in progress
SLOT_APPLIED = "✓"  # transient success
SLOT_BLOCKED = "⏸"  # a decision is required (review/privacy/auth)

#: Aggregate priority: the highest wins the one slot. A spinner must not hide a
#: private/public decision, so blocked outranks busy.
_PRIORITY = {
    "blocked": 0,
    "failure": 1,
    "available": 2,
    "busy": 3,
    "applied": 4,
    "healthy": 5,
}
_SLOT_FOR = {
    "healthy": SLOT_IDLE,
    "available": SLOT_AVAILABLE,
    "failure": SLOT_FAILURE,
    "busy": SLOT_BUSY,
    "applied": SLOT_APPLIED,
    "blocked": SLOT_BLOCKED,
}


@dataclass(frozen=True)
class LinkRow:
    """One link as the sidebar needs it: identity, scope, policy, aggregate state."""

    kind: str
    local_id: str
    name: str
    link_id: str | None
    source: str | None
    visibility: str | None
    subscribe: bool | None
    auto_push: bool | None
    pull_state: str
    push_state: str
    pull_enabled: bool
    push_enabled: bool
    auto_will_apply: bool
    auto_will_push: bool
    state: str
    error_class: str | None

    @property
    def scope_label(self) -> str:
        if self.visibility == "public":
            return "Public"
        if self.visibility == "org":
            return "Private"
        return "Unknown source"

    @property
    def slot_state(self) -> str:
        """The single aggregate state the slot communicates (priority-ordered)."""

        if self.link_id is None:
            return "blocked"  # a legacy link needs resolve before any action
        if self.error_class in (
            "merge-refused",
            "unsafe-publication",
            "origin-conflict",
            "mixed-lineage",
        ):
            return "blocked"
        if self.error_class:
            return "failure"
        if self.pull_state in ("available",):
            return "available"
        if self.pull_state in ("merging", "applying") or self.push_state == "pushing":
            return "busy"
        return "healthy"

    def policy_text(self) -> str:
        """``Public; Subscribe on; Auto-push off`` — the accessible description."""

        sub = "on" if self.subscribe else "off"
        push = "on" if self.auto_push else "off"
        suffix = ""
        if self.auto_push is False and self.error_class in (
            "unsafe-publication",
            "origin-conflict",
        ):
            suffix = f", paused: {self.error_class}"
        return f"{self.scope_label}; Subscribe {sub}; Auto-push {push}{suffix}"


def rows_from_snapshot(snapshot: Mapping[str, Any]) -> list[LinkRow]:
    """Project a ``links_snapshot`` payload into render rows (pure).

    The frozen shape carries ``hub_source`` and ``pull``/``push`` as independent
    DirectionStatus objects; a legacy identity-less link is under ``unresolved``
    and gets a row with ``link_id=None`` so the sidebar offers resolve, not a
    guessed origin.
    """

    rows: list[LinkRow] = []
    for link in snapshot.get("links", []):
        policy = link.get("policy") or {}
        pull = link.get("pull") or {}
        push = link.get("push") or {}
        rows.append(
            LinkRow(
                kind=str(link.get("kind") or ""),
                local_id=str(link.get("local_id") or ""),
                name=str(link.get("name") or link.get("local_id") or ""),
                link_id=link.get("link_id"),
                source=link.get("hub_source"),
                visibility=link.get("visibility"),
                subscribe=policy.get("subscribe"),
                auto_push=policy.get("auto_push"),
                pull_state=str(pull.get("state") or "idle"),
                push_state=str(push.get("state") or "idle"),
                pull_enabled=bool(pull.get("enabled")),
                push_enabled=bool(push.get("enabled")),
                auto_will_apply=bool(link.get("auto_will_apply")),
                auto_will_push=bool(link.get("auto_will_push")),
                state=str(link.get("state") or "up-to-date"),
                error_class=pull.get("error_class"),
            )
        )
    for row in snapshot.get("unresolved", []):
        rows.append(
            LinkRow(
                kind=str(row.get("kind") or ""),
                local_id=str(row.get("local_id") or ""),
                name=str(row.get("name") or row.get("local_id") or ""),
                link_id=None,
                source=None,
                visibility=None,
                subscribe=None,
                auto_push=None,
                pull_state="idle",
                push_state="idle",
                pull_enabled=False,
                push_enabled=False,
                auto_will_apply=False,
                auto_will_push=False,
                state="up-to-date",
                error_class=None,
            )
        )
    return rows


def section_order() -> tuple[str, ...]:
    """Agents, Teams, Skills — the sibling-section order the design fixes."""

    return ("agent", "team", "skill")


_SECTION_TITLES = {"agent": "Agents", "team": "Teams", "skill": "Skills"}


def render_rows(
    rows: Sequence[LinkRow],
    width: int,
    *,
    selected: int | None = None,
    loading: bool = False,
) -> list[Text]:
    """The body: one section per kind, one reserved-slot row per link (pure).

    ``loading`` shows the reserved-card text; the caller keeps last-known rows
    across a refresh, so a poll never replaces the body with a skeleton.
    """

    if loading and not rows:
        return [Text("Loading hub links…", style="dim")]

    lines: list[Text] = []
    by_kind: dict[str, list[LinkRow]] = {kind: [] for kind in section_order()}
    for row in rows:
        by_kind.setdefault(row.kind, []).append(row)
    index = 0
    for kind in section_order():
        group = by_kind.get(kind, [])
        if not group:
            continue
        lines.append(Text(_SECTION_TITLES.get(kind, kind.title()), style="bold"))
        for row in sorted(group, key=lambda r: r.name.casefold()):
            lines.append(_render_row(row, width, selected=(selected == index)))
            index += 1
    if not lines:
        lines.append(Text("Not linked to the hub.", style="dim"))
    return lines


def _render_row(row: LinkRow, width: int, *, selected: bool) -> Text:
    """``<slot> <name> — <source> · <policy>`` fitted to ``width`` cells.

    The slot is ALWAYS one reserved cell, so a state change cannot reflow the
    row (the name carries no policy badges). At narrow widths the dim tail is
    what gives: the full source+policy is the row's accessible description
    (:meth:`LinkRow.policy_text`) and its tooltip, not a promise the collapsed
    row paints every word — a row that overflowed its box would be the louder
    defect.
    """

    slot = _SLOT_FOR.get(row.slot_state, SLOT_IDLE)
    source = row.source or "(unresolved origin)"
    prefix = f"{slot} "
    tail = f" — {source} · {row.policy_text()}"
    name_room = max(4, width - cell_len(prefix) - cell_len(tail))
    name = truncate_cells(row.name, name_room)
    remaining = width - cell_len(prefix) - cell_len(name)
    if remaining <= 0:
        tail = ""
    elif cell_len(tail) > remaining:
        # Keep the scope word visible at least: truncate from the tail's END.
        tail = truncate_cells(tail, remaining)
    line = Text()
    line.append(prefix, style="dim" if not selected else "reverse")
    line.append(name, style="reverse" if selected else "")
    line.append(tail, style="dim")
    return line


def aggregate_slot(rows: Sequence[LinkRow]) -> str:
    """The one aggregate mark for a collapsed item with several source copies."""

    if not rows:
        return SLOT_IDLE
    best = min(rows, key=lambda r: _PRIORITY.get(r.slot_state, 99))
    return _SLOT_FOR.get(best.slot_state, SLOT_IDLE)
