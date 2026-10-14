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

from local_operator.i18n.keys import tui_hub_links as _k
from local_operator.i18n.text import tr
from local_operator.tui.widgets.tool_card import truncate_cells

#: This slice owns exactly ONE catalogue, so the namespace is stated here once
#: (the `tr` contract). Every user-visible string below goes through `_t`; a
#: literal would be the English-only bypass the i18n rule forbids. The typed key
#: functions (`tui_hub_links.section_agents()`, …) are what a mistyped key fails
#: pyright on (RFC §2.4).
_t = tr("tui.hub_links")

#: The narrowest NAME a row keeps on its first line before the policy pair moves
#: to a second indented line (round-2 D8). Below this the scope/name would be
#: unreadable, so wrapping beats truncating — the namespace's own context note.
MIN_NAME_CELLS = 12

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
    tenant_name: str | None
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
    #: The policy revision the inventory reported, for the CAS on a toggle
    #: (round-2 D7). ``None`` for a legacy/unresolved row, which cannot be
    #: edited until its origin is resolved.
    policy_revision: str | None = None
    #: Service-computed auto-push eligibility and its named reason (D7): turning
    #: auto-push ON is refused locally when this is False, with the reason shown.
    can_auto_push: bool = False
    eligibility_reason: str = ""

    @property
    def scope_label(self) -> str:
        """The frozen visible scope: ``Public`` or ``<org> · Private`` (§30).

        Compact and DELIBERATELY not the API base URL: a link's origin is a
        credential-free URL, and the frozen contract requires the row to show
        the SCOPE (public, or the organization's NAME and Private) rather than
        the raw origin. The full source URL stays in the accessible description
        (:meth:`policy_text`), which is the technical detail a screen reader and
        the policy menu read; only the collapsed row is compact. Copy comes from
        the catalogue (`source.public` / `source.private`), never a literal.
        """

        if self.visibility == "public":
            return _t(_k.source_public())
        if self.visibility == "org":
            org = (self.tenant_name or "").strip()
            if org:
                return _t(_k.source_private(organization=org))
            # No resolved name: show the Private word alone rather than a bare
            # `· Private` with an empty organization slot.
            return _t(_k.source_private(organization="")).strip(" ·")
        return _t(_k.source_unavailable())

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
        """The accessible description: scope + both policies, catalogue-rendered.

        Built from the `mark.healthy` template (item; source; Subscribe <s>;
        Auto-push <p>), which the catalogue owns — the description is where the
        technical origin lives while the visible row shows the compact scope.
        """

        return _t(
            _k.mark_healthy(
                item=self.name,
                source=self.source or _t(_k.source_unavailable()),
                subscribe=self._policy_word(self.subscribe),
                auto_push=self._policy_word(self.auto_push),
            )
        )

    @staticmethod
    def _policy_word(value: bool | None) -> str:
        """``On``/``Off`` from the catalogue; an unknown policy reads ``?``."""

        if value is None:
            return "?"
        return _t(_k.policy_on()) if value else _t(_k.policy_off())


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
                tenant_name=link.get("tenant_name"),
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
                policy_revision=policy.get("revision"),
                can_auto_push=bool((link.get("eligibility") or {}).get("can_auto_push")),
                eligibility_reason=str((link.get("eligibility") or {}).get("reason") or ""),
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
                tenant_name=None,
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


def _section_title(kind: str) -> str:
    """A section heading's copy from the catalogue, never a literal."""

    return {
        "agent": _t(_k.section_agents()),
        "team": _t(_k.section_teams()),
        "skill": _t(_k.section_skills()),
    }.get(kind, kind.title())


def render_rows(
    rows: Sequence[LinkRow],
    width: int,
    *,
    selected: int | None = None,
    loading: bool = False,
    tail_style: Any = "dim",
) -> list[Text]:
    """The body: one section per kind, one reserved-slot row per link (pure).

    ``width`` is the WIDGET'S CONTENT box in cells — the panel's styles.width
    minus its padding — not the outer width. The widget computes it once
    (:meth:`HubLinksPanel._content_width`) and passes the same number it pinned
    ``styles.width`` to, so a row can never be composed wider than the box it is
    painted into (the round-2 wrap bug: rows were composed at ``screen-8`` while
    the padded content box was 4 cells narrower, so a 32-cell row wrapped).

    ``tail_style`` is the resolved ink for the source+policy tail. The WIDGET
    passes the theme's ``muted`` (D4); the default ``"dim"`` keeps the core
    renderable without a theme (a bare ``"muted"`` is a palette token, not a
    Rich style, and would raise ``MissingStyle``). ``loading`` shows the
    reserved-card text; the caller keeps last-known rows across a refresh, so a
    poll never replaces the body with a skeleton.
    """

    if loading and not rows:
        return [Text(_t(_k.state_loading()), style="dim")]

    lines: list[Text] = []
    by_kind: dict[str, list[LinkRow]] = {kind: [] for kind in section_order()}
    for row in rows:
        by_kind.setdefault(row.kind, []).append(row)
    index = 0
    for kind in section_order():
        group = by_kind.get(kind, [])
        if not group:
            continue
        lines.append(Text(_section_title(kind), style="bold"))
        for row in sorted(group, key=lambda r: r.name.casefold()):
            lines.append(
                _render_row(row, width, selected=(selected == index), tail_style=tail_style)
            )
            index += 1
    if not lines:
        lines.append(Text(_t(_k.state_empty()), style="dim"))
    return lines


def _render_row(row: LinkRow, width: int, *, selected: bool, tail_style: Any = "dim") -> Text:
    """``<slot> <name> — <scope> · <policy>`` fitted to ``width`` cells.

    The tail shows the compact SCOPE (``Public`` / ``Acme · Private``), never the
    raw API base URL (D2 / frozen §30); the full source URL is the row's
    accessible description. The slot is ALWAYS one reserved cell, so a state
    change cannot reflow the row, and neither the source/permission copy nor the
    essential policy words may be clipped away (frozen §128).

    WIDTH BEHAVIOUR (round-2 D8). At a content width too narrow for
    ``<name> — <scope> · Subscribe <s> · Auto-push <p>`` to survive with a
    readable name, the policy moves to a SECOND indented line rather than being
    truncated: the catalogue's own context note for this namespace says to wrap
    explanatory copy, never truncate it, and the policy pair is the row's whole
    point. The name keeps at least :data:`MIN_NAME_CELLS`, so the scope still
    leads the tail and the two policy words stay legible at the 40x24 floor.

    Ink: the tail uses ``muted`` (6.51:1 dark / 5.18:1 light), not ``dim``
    (3.43:1 / 2.72:1) — the projects_send precedent (D4).
    """

    slot = _SLOT_FOR.get(row.slot_state, SLOT_IDLE)
    prefix = f"{slot} "
    scope = f" — {row.scope_label}"
    policy = (
        f"{_t(_k.policy_subscribe_label())} {row._policy_word(row.subscribe)} · "
        f"{_t(_k.policy_auto_push_label())} {row._policy_word(row.auto_push)}"
    )
    tail = f"{scope} · {policy}"
    name_room = width - cell_len(prefix) - cell_len(tail)
    if name_room >= MIN_NAME_CELLS:
        # Fits on one line: name (min 4 cells) then the scope+policy tail.
        name = truncate_cells(row.name, max(4, name_room))
        remaining = width - cell_len(prefix) - cell_len(name)
        fitted_tail = truncate_cells(tail, remaining) if cell_len(tail) > remaining else tail
        line = Text()
        line.append(prefix, style="dim" if not selected else "reverse")
        line.append(name, style="reverse" if selected else "")
        line.append(fitted_tail, style=tail_style if not selected else "reverse")
        return line
    # Narrow: the policy moves to its own indented line(s) so neither policy
    # word is lost. The first line is `<slot> <name> — <scope>`; the policy is
    # indented under the name column, and when it does not fit ONE line each flag
    # gets its own line — at the 40x24 content floor `Subscribe On · Auto-push
    # Off` is 28 cells and the detail line is narrower, so a single line would
    # clip the auto-push VALUE; splitting keeps both words and both values.
    name = truncate_cells(row.name, max(4, width - cell_len(prefix) - cell_len(scope)))
    head = Text()
    head.append(prefix, style="dim" if not selected else "reverse")
    head.append(name, style="reverse" if selected else "")
    head.append(scope, style=tail_style if not selected else "reverse")
    indent = " " * cell_len(prefix)
    detail_width = max(0, width - cell_len(prefix))
    style = tail_style if not selected else "reverse"
    flags = [
        f"{_t(_k.policy_subscribe_label())} {row._policy_word(row.subscribe)}",
        f"{_t(_k.policy_auto_push_label())} {row._policy_word(row.auto_push)}",
    ]
    joined = " · ".join(flags)
    if cell_len(joined) <= detail_width:
        details = [joined]
    else:
        details = flags
    out = Text.assemble(head)
    for part in details:
        out.append("\n")
        # N1 (round-3): the INDENT carries the selection ink too, so a wrapped
        # row's highlight is a clean block rather than a ragged one with an
        # unhighlighted gap under the slot column.
        out.append(indent, style=style)
        out.append(truncate_cells(part, detail_width), style=style)
    return out


def aggregate_slot(rows: Sequence[LinkRow]) -> str:
    """The one aggregate mark for a collapsed item with several source copies."""

    if not rows:
        return SLOT_IDLE
    best = min(rows, key=lambda r: _PRIORITY.get(r.slot_state, 99))
    return _SLOT_FOR.get(best.slot_state, SLOT_IDLE)
