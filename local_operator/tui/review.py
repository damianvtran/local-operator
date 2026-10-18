"""The session's code-request ledger, derived for the TUI — React-free and pure.

WHY THIS FILE EXISTS. ``/review`` paints one ledger, and the desktop Code review
pane paints the same one. Each surface carries the same figures in a different
register (a clause, a row line, a pane line), and the design's rule is one
derived string per fact, so the two surfaces cannot state one number two ways.
This module is the TUI's half of that rule: the React-free equivalent of
``local-operator-ui``'s ``features/code-review/code-review-model.ts`` (pinned at
``76dda0304``), driven over the SAME wire rows
:func:`local_operator.code_requests.service.view_rows` produces.

WHAT IS NEVER DONE HERE. The relation is the BACKEND's vocabulary and is never
re-derived: ``unknown`` stays "possibly opened" (a script that merely printed a
PR URL is indistinguishable from one that created it without the forge's
record). A failed refresh keeps the last known data; ``refresh_error`` is the
only field that says so, and ``stale`` alone is never rendered as a failure —
"Couldn't refresh" is a claim about an ATTEMPT.

NO WIDGET IMPORTS. The view, the tests and (later) any other TUI surface read
these functions; importing Textual here would drag the runtime into a module
whose whole job is strings. Rendering vocabulary stays out too: a tone is a ROLE
(``"danger"``, ``"warning"``), and the view maps the role to a theme colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from rich.cells import cell_len

from local_operator.i18n import keys as i18n_keys
from local_operator.i18n.text import tr

_t = tr("tui.review")
_kt = i18n_keys.tui_review

#: Lane display order: Code, Design, QA, UX — then anything a newer backend
#: serves, in wire order, so an unknown lane is shown after the known four
#: rather than dropped.
LANE_ORDER = ("agent", "design", "qa", "ux")

#: The lane-name column's fixed cell budget. A MINIMUM-width format
#: (``f"{name:<6}"``) is not a clamp: a translated name wider than the budget
#: pushes its clause right, so the two lane columns stop lining up in Hindi and
#: every later locale that runs long. Measuring and padding to one budget keeps
#: the clause's start column identical everywhere.
LANE_NAME_CELLS = 8

FETCH_STATES = (
    "pending",
    "ready",
    "stale",
    "unauthenticated",
    "cooling",
    "failed",
    "untracked",
)

PillVariant = Literal["info", "outline", "success", "neutral"]
CiTone = Literal["danger", "muted", "plain"]
LaneTone = Literal["warning", "muted", "success", "outline"]


@dataclass(frozen=True, slots=True)
class Pill:
    """The state pill: the forge's word, mapped to the badge vocabulary."""

    label: str
    variant: PillVariant


@dataclass(frozen=True, slots=True)
class CiClause:
    """One CI clause and the ink role it takes."""

    text: str
    tone: CiTone


@dataclass(frozen=True, slots=True)
class LaneLine:
    """One lane's rendered line: name, round strip, and the joined clause."""

    key: str
    name: str
    clause: str
    segments: tuple[LaneTone, ...]
    tone: LaneTone


@dataclass(frozen=True, slots=True)
class Groups:
    """The row partition: the session's own requests, then its mentions."""

    opened: tuple[Mapping[str, Any], ...]
    mentioned: tuple[Mapping[str, Any], ...]


def _finite(value: Any) -> float | None:
    """A finite number from the wire, or ``None`` — the shape's null is an answer."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


# -- identity ---------------------------------------------------------------


def forge_sigil(row: Mapping[str, Any]) -> str:
    """``!N`` for a GitLab merge request, ``#N`` for everything else.

    On GitLab ``#N`` is an issue, and the operator writes ``minerva-skills!57``.
    """
    return "!" if row.get("forge") == "gitlab" else "#"


def identity_label(row: Mapping[str, Any]) -> str:
    """``damianvtran/local-operator #927`` / ``minervaai/minerva-skills !57``."""
    return f"{row.get('project') or ''} {forge_sigil(row)}{row.get('number') or ''}".strip()


def row_title(row: Mapping[str, Any]) -> str:
    """The summary's title, falling back to the URL for a link-only row."""
    summary = _mapping(row.get("summary"))
    return _text(summary.get("title")) or str(row.get("url") or "")


# -- state pill -------------------------------------------------------------


def state_pill(row: Mapping[str, Any]) -> Pill | None:
    """The state pill for a fetched summary, or ``None`` (link-only rows have none).

    ``draft`` wins over ``open`` because a draft PR is the state the operator
    must not confuse with one that can merge — and the forge reports
    ``state: "open"`` for it. An unknown state takes ``neutral`` with the raw
    word, never an invented one.
    """
    summary = _mapping(row.get("summary"))
    if not summary:
        return None
    if summary.get("draft"):
        return Pill(_t(_kt.state_draft()), "outline")
    state = _text(summary.get("state"))
    if state == "open":
        return Pill(_t(_kt.state_open()), "info")
    if state == "merged":
        return Pill(_t(_kt.state_merged()), "success")
    if state == "closed":
        return Pill(_t(_kt.state_closed()), "neutral")
    return Pill(state or _t(_kt.state_unknown()), "neutral")


# -- CI ---------------------------------------------------------------------


def ci_clause(ci: Mapping[str, Any] | None) -> CiClause | None:
    """The CI clause and its ink, exactly the reference's precedence.

    NULL COUNTS ARE A HOST'S ANSWER, NOT A MISSING ONE: GitLab pipeline state
    carries no job counts, so every clause falls back to the STATUS word when a
    count is not a finite number — ``null/null passed`` and a red pipeline read
    as passing were the two defects the desktop surface fixed. ``failed > 0``
    still outranks the status word (a run can be ``pending`` overall while
    carrying failures), and ``total == 0`` with ``status: none`` is "No checks
    yet" rather than a ``0/0`` fraction.
    """
    data = _mapping(ci)
    if not data:
        return None
    failed = _finite(data.get("failed"))
    pending = _finite(data.get("pending"))
    passed = _finite(data.get("passed"))
    total = _finite(data.get("total"))
    status = _text(data.get("status"))
    if failed is not None and failed > 0:
        return CiClause(_t(_kt.ci_failing(count=int(failed))), "danger")
    if status == "failure":
        return CiClause(_t(_kt.ci_failing_status()), "danger")
    if status == "none" or (total is not None and total == 0):
        return CiClause(_t(_kt.ci_no_checks()), "muted")
    if pending is not None and pending > 0:
        return CiClause(_t(_kt.ci_running()), "muted")
    if status == "pending":
        return CiClause(_t(_kt.ci_pending()), "muted")
    if status == "unknown":
        return CiClause(_t(_kt.ci_no_checks()), "muted")
    if total is not None and total > 0 and passed is not None:
        return CiClause(
            _t(_kt.ci_passed_count(passed=str(int(passed)), total=str(int(total)))), "plain"
        )
    if status == "success":
        return CiClause(_t(_kt.ci_passed_status()), "plain")
    # A settled run this build cannot classify: claim nothing.
    return None


def comment_clause(count: Any) -> str | None:
    """``1 comment`` / ``6 comments``, or ``None`` at 0 (the clause is omitted)."""
    value = _finite(count)
    if value is None or value <= 0:
        return None
    return _t(_kt.comments_count(count=int(value)))


# -- updated age ------------------------------------------------------------


def updated_clause(updated_at_sec: Any, now_ms: float) -> str:
    """The row's updated tail: ``updated 2 min ago``.

    The register follows the string the design fixes (minute grain, "just now"
    under a minute); a future timestamp — a clock skewed ahead of the backend's
    — reads ``just now`` rather than a negative age.
    """
    updated = _finite(updated_at_sec) or 0.0
    seconds = max(0, int(now_ms / 1000 - updated))
    if seconds < 60:
        return _t(_kt.age_just_now())
    minutes = seconds // 60
    if minutes < 60:
        return _t(_kt.age_minutes(amount=minutes))
    hours = minutes // 60
    if hours < 24:
        return _t(_kt.age_hours(amount=hours))
    return _t(_kt.age_days(amount=hours // 24))


# -- lanes ------------------------------------------------------------------


def lane_display_name(lane: str) -> str:
    """The lane's displayed name: the ``agent`` lane is this session's **Code**."""
    if lane == "agent":
        return _t(_kt.lane_code())
    if lane == "design":
        return _t(_kt.lane_design())
    if lane == "qa":
        return _t(_kt.lane_qa())
    if lane == "ux":
        return _t(_kt.lane_ux())
    return lane


def lane_name_cell(name: str) -> str:
    """A lane name in EXACTLY :data:`LANE_NAME_CELLS` cells.

    Cell-aware truncation, not a character count, because a Devanagari cluster
    and a Latin letter are different widths. No ellipsis: the name is a column
    label, and the full clause beside it is what a reader acts on. The view uses
    this for both the list row and the detail pane, so the two columns cannot
    disagree about where the clause starts.

    ``truncate_cells`` (``tool_card``) is deliberately not reused here: that
    helper lives in a widget module, and this module's whole constraint is that
    it imports no Textual — the model is read by tests and (later) any surface.
    """
    clipped = name
    while clipped and cell_len(clipped) > LANE_NAME_CELLS:
        clipped = clipped[:-1]
    return clipped + " " * (LANE_NAME_CELLS - cell_len(clipped))


def lane_state_clause(lane: Mapping[str, Any]) -> str:
    """One lane's clause: ``Round 2 · remediation posted``.

    ``awaiting_review`` has no round to place and prints the fixed "No agent
    review yet". ``remediation_posted`` carries "awaiting re-review" so the
    clause says what is being waited on. A state this build has never seen
    prints the RAW value after the round: "one fact, one mark" refuses to guess
    a word for it, and dropping the lane would hide a review that happened.
    """
    state = _text(lane.get("state"))
    if state == "awaiting_review":
        return _t(_kt.lane_no_review())
    word = {
        "findings_open": _t(_kt.lane_state_findings_open()),
        "remediation_posted": _t(_kt.lane_state_remediation_posted()),
        "clean": _t(_kt.lane_state_clean()),
        "terminal": _t(_kt.lane_state_terminal()),
        "unstated": _t(_kt.lane_state_unstated()),
    }.get(state, state)
    if not word:
        # An absent word is the same fact as `unstated` — a review happened and
        # its verdict was not recorded — and must not render as a blank clause.
        word = _t(_kt.lane_state_unstated())
    if state == "remediation_posted":
        word = _t(_kt.lane_awaiting_rereview(state=word))
    # A lane whose latest comment carried no placeable round prints the state
    # WITHOUT a number: a claim about how far along a review is must never be
    # guessed from an absence.
    round_no = _finite(lane.get("round"))
    if round_no is not None and round_no >= 1:
        return _t(_kt.lane_round(round=str(int(round_no)), state=word))
    return word


def lane_freshness_clause(lane: Mapping[str, Any], head_sha: str | None) -> str | None:
    """The freshness tail: ``stale — reviewed a3ffd2b, head 9d29452``.

    ``None`` for fresh and for an unknown head: fresh is the default and
    unmarked, and an absent SHA is never guessed.
    """
    if lane.get("freshness") != "stale":
        return None
    reviewed = _text(lane.get("reviewed_head"))[:7]
    head = (head_sha or "")[:7]
    if reviewed and head:
        return _t(_kt.lane_stale_head(reviewed=reviewed, head=head))
    if reviewed:
        return _t(_kt.lane_stale_reviewed(reviewed=reviewed))
    return _t(_kt.lane_stale())


def lane_segment_tone(state: str) -> LaneTone:
    """The tone a round's state maps to, by role and never by hex."""
    if state == "findings_open":
        return "warning"
    if state == "remediation_posted":
        return "muted"
    if state in ("clean", "terminal"):
        return "success"
    return "outline"


def round_segments(lane: Mapping[str, Any]) -> tuple[LaneTone, ...]:
    """One segment per KNOWN round, chronological.

    THE WIRE PARSES ONE STATE PER LANE — the latest convention comment's — so
    the latest segment carries the state's tone and earlier ones are NEUTRAL,
    never the latest tone repeated. A stale ``clean`` is painted ``muted``: a
    stale clean is not clean now. No round number, no segments: a segment count
    is a claim about how many rounds ran.
    """
    round_no = _finite(lane.get("round"))
    if round_no is None or round_no < 1:
        return ()
    count = int(round_no)
    latest = lane_segment_tone(_text(lane.get("state")))
    if lane.get("freshness") == "stale" and latest == "success":
        latest = "muted"
    segments: list[LaneTone] = ["muted"] * (count - 1)
    segments.append(latest)
    return tuple(segments)


def _lane_rank(lane: str) -> int:
    try:
        return LANE_ORDER.index(lane)
    except ValueError:
        return len(LANE_ORDER)


def lane_lines(row: Mapping[str, Any]) -> tuple[LaneLine, ...]:
    """Every lane the wire carries, ordered Code, Design, QA, UX, then the rest.

    THE CODE LINE IS ALWAYS PRESENT WHILE THE REQUEST IS OPEN OR A DRAFT: the
    code lane's no-comment state is ``awaiting review``, copy "No agent review
    yet", so a PR with only a design lane still shows its own line. A
    merged/closed request with NO lanes omits the strip entirely; one that has
    lanes keeps them, because a finished review history is worth the lines.
    """
    summary = _mapping(row.get("summary"))
    lanes = sorted(
        (_mapping(item) for item in _sequence(row.get("lanes"))),
        key=lambda item: _lane_rank(_text(item.get("lane"))),
    )
    has_code = any(_text(item.get("lane")) == "agent" for item in lanes)
    state = _text(summary.get("state"))
    if not has_code and (state in ("open", "draft") or summary.get("draft") is True):
        # NO ROUND NUMBER: "no agent review yet" has no round to place, and the
        # strip derives its count from this field.
        lanes.insert(
            0,
            {"lane": "agent", "round": None, "state": "awaiting_review", "freshness": "unknown"},
        )
    head_sha = _text(summary.get("head_sha"))
    out: list[LaneLine] = []
    for index, lane in enumerate(lanes):
        name = lane_display_name(_text(lane.get("lane")))
        clause = " · ".join(
            part
            for part in (lane_state_clause(lane), lane_freshness_clause(lane, head_sha))
            if part
        )
        out.append(
            LaneLine(
                key=f"{_text(lane.get('lane'))}-{index}",
                name=name,
                clause=clause,
                segments=round_segments(lane),
                tone=lane_segment_tone(_text(lane.get("state"))),
            )
        )
    return tuple(out)


# -- relation and activity --------------------------------------------------


def _mention_last_at(row: Mapping[str, Any]) -> float:
    latest = 0.0
    for item in _sequence(row.get("mentions")):
        value = _finite(_mapping(item).get("last_at"))
        if value is not None and value > latest:
            latest = value
    if latest:
        return latest
    return _finite(row.get("last_at")) or 0.0


def via_tag(via: Mapping[str, Any] | None) -> str | None:
    """``via subagent coder`` / ``via subagent coder › reviewer``, or ``None``.

    Tolerates an event that carried only the flat fields: the path is built
    from ``path`` when present, else from the agent role or label.
    """
    data = _mapping(via)
    if not data:
        return None
    path = [item for item in _sequence(data.get("path")) if isinstance(item, str) and item]
    if path:
        return _t(_kt.relation_subagent_path(path=" › ".join(path)))
    fallback = _text(data.get("agent_role")) or _text(data.get("label"))
    if fallback:
        return _t(_kt.relation_subagent_named(name=fallback))
    return _t(_kt.relation_subagent())


def relation_tag(row: Mapping[str, Any]) -> str | None:
    """The line-A relation tag, only where a row needs disambiguating.

    ``opened`` is the plain case and returns ``None``; ``mentioned`` likewise
    (its group header says it). ``unknown`` carries the design's exact words
    and never upgrades to ``opened`` — without the forge's record a printed URL
    is indistinguishable from a created one. ``inherited`` renders as the bare
    word: the frozen row shape carries no parent identity, and naming an
    invented one would be the guess ``unknown`` exists to refuse.
    """
    relation = row.get("relation")
    if relation == "opened":
        return via_tag(_mapping(row.get("via")))
    if relation == "mentioned":
        return None
    if relation == "unknown":
        return via_tag(_mapping(row.get("via"))) or _t(_kt.relation_unknown())
    if relation == "inherited":
        return _t(_kt.relation_inherited())
    return None


_ACT_WORDS = {
    "comment": _kt.activity_comment,
    "merge": _kt.activity_merge,
    "review": _kt.activity_review,
    "push": _kt.activity_push,
    "edit": _kt.activity_edit,
    "close": _kt.activity_close,
    "ready": _kt.activity_ready,
}


def acted_tag(row: Mapping[str, Any]) -> str | None:
    """The acts this session performed on the ref, as one trailing tag.

    The act words are the backend's (``comment``, ``merge``, ``review``,
    ``push``, ``edit``, ``close``, ``ready``); an unknown word is shown raw
    inside the same sentence rather than dropped, because the row must not
    claim nothing happened when something did.
    """
    acts = [item for item in _sequence(row.get("acts")) if isinstance(item, str) and item]
    if not acts:
        return None
    words = []
    for act in acts:
        factory = _ACT_WORDS.get(act)
        words.append(_t(factory()) if factory is not None else act)
    return _t(_kt.activity_you(acts=", ".join(words)))


# -- grouping ---------------------------------------------------------------


def _relation_rank(row: Mapping[str, Any]) -> int:
    relation = row.get("relation")
    if relation == "opened":
        return 0
    if relation == "unknown":
        return 1
    if relation == "inherited":
        return 2
    return 3


def group_rows(rows: Sequence[Mapping[str, Any]]) -> Groups:
    """``Opened`` = every non-mentioned relation; ``Mentioned`` = mentioned.

    The order inside each group is ONE comparator for both: a real open first,
    then the possibles, then anything this session acted on, then the freshest
    mentions — and the latest mention breaks every remaining tie.
    """
    ordered = sorted(
        rows,
        key=lambda row: (
            _relation_rank(row),
            -len([item for item in _sequence(row.get("acts")) if isinstance(item, str) and item]),
            -_mention_last_at(row),
        ),
    )
    return Groups(
        opened=tuple(row for row in ordered if row.get("relation") != "mentioned"),
        mentioned=tuple(row for row in ordered if row.get("relation") == "mentioned"),
    )


# -- notices ----------------------------------------------------------------


def link_only_remedy(row: Mapping[str, Any]) -> str:
    """The link-only row's one visible line: the backend's per-forge sentence.

    Verbatim when present — the sentence's lead is itself ``Link only``, so the
    caption and the remedy are one line — else the bare caption. The old
    derivation ("sign in with gh" for everything but GitLab) taught a remedy
    that cannot work for a host with no adapter.
    """
    hint = _text(row.get("link_only_hint"))
    return hint or _t(_kt.link_only())


def refresh_caption(row: Mapping[str, Any]) -> str | None:
    """The per-row refresh-failure caption: the backend's text VERBATIM.

    The join is a COLON rather than the house em-dash because the backend's
    sentence carries an em-dash of its own in the credential case, and two in
    one line read as a run-on.
    """
    error = _text(row.get("refresh_error"))
    return _t(_kt.notice_refresh_error(error=error)) if error else None


def fetch_state_of(row: Mapping[str, Any]) -> str | None:
    """The row's fetch state, or ``None`` for absent or unrecognised words.

    Both mean "render as if the field did not exist": absent is an older
    backend, and an unrecognised word is a newer one — the legacy path claims
    nothing either way.
    """
    value = row.get("fetch_state")
    return value if isinstance(value, str) and value in FETCH_STATES else None


def row_is_loading(row: Mapping[str, Any]) -> bool:
    """Whether the row's first state read has not resolved yet."""
    return fetch_state_of(row) == "pending"


def row_notice(row: Mapping[str, Any]) -> str | None:
    """The row's ONE quiet line under the meta figures, or ``None``.

    WITH ``fetch_state`` ON THE WIRE this is a switch on it: a ``pending`` row
    returns ``None`` — the row renders a loading line instead, and a remedy
    must never be claimed while the first read is in flight; ``stale`` leads
    with its own state word so a failed revalidation is not read as a failed
    ATTEMPT; only ``unauthenticated``/``untracked``/``cooling`` may carry the
    sign-in or "not tracked" remedy, and each reads the backend's own sentence
    (``reason``) first. Nothing here derives a CLI or a cause.

    ABSENT ``fetch_state`` (an older backend) keeps the legacy logic: a
    link-only row shows its ``reason`` when one was sent, else the per-forge
    remedy; a tracked row shows its ``refresh_error`` (prefixed) when the last
    attempt failed, else the bare ``reason``.
    """
    state = fetch_state_of(row)
    if state is None:
        if row.get("link_only"):
            return _text(row.get("reason")) or link_only_remedy(row)
        if _text(row.get("refresh_error")):
            return refresh_caption(row)
        return _text(row.get("reason")) or None
    if state == "pending":
        return None
    if state == "ready":
        return (
            refresh_caption(row)
            if _text(row.get("refresh_error"))
            else (_text(row.get("reason")) or None)
        )
    if state == "stale":
        error = _text(row.get("refresh_error"))
        return _t(_kt.notice_stale_error(error=error)) if error else _t(_kt.notice_stale())
    if state == "unauthenticated":
        return _text(row.get("reason")) or link_only_remedy(row)
    if state == "cooling":
        return _text(row.get("reason")) or _t(_kt.notice_cooling())
    if state == "failed":
        return _text(row.get("reason")) or refresh_caption(row) or _t(_kt.notice_fetch_failed())
    if state == "untracked":
        return _text(row.get("reason")) or link_only_remedy(row)
    return None


__all__ = [
    "FETCH_STATES",
    "LANE_NAME_CELLS",
    "LANE_ORDER",
    "CiClause",
    "Groups",
    "LaneLine",
    "Pill",
    "acted_tag",
    "ci_clause",
    "comment_clause",
    "fetch_state_of",
    "forge_sigil",
    "group_rows",
    "identity_label",
    "lane_display_name",
    "lane_freshness_clause",
    "lane_lines",
    "lane_name_cell",
    "lane_segment_tone",
    "lane_state_clause",
    "link_only_remedy",
    "refresh_caption",
    "relation_tag",
    "round_segments",
    "row_is_loading",
    "row_notice",
    "row_title",
    "state_pill",
    "updated_clause",
    "via_tag",
]
