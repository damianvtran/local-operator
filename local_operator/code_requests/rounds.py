"""Parse the agent-teams review-round convention out of PR/MR comments. Pure, no I/O.

WHY A PARSER AND NOT A PROMPT. Every round of every review on this fleet posts a
comment whose first line follows one convention (``### Agent review — round 2
(delta verification)``), and the merge gate is defined in terms of those comments:
"no blocker/major findings remain, and the round is fresh on head". A reader that
answers "where is this PR up to?" therefore has to read the convention, and the
alternative — asking a model — would make the answer non-deterministic and cost a
provider call per poll.

The grammar below is not invented: the spike ran it over 44 real comments (37
GitHub, 7 GitLab) and this module carries the fixtures it was measured on
(``tests/fixtures/code_requests/``). Two findings from that spike shape the code:

* **Verdicts classify by the LEADING TOKEN, never by substring search.** A
  substring version marked a real ``PASS — 0 FAIL, 0 BLOCKED`` verdict as failing
  because it contained the word FAIL. A verdict that cannot be classified is
  ``unstated``, and the copy says so; nothing is guessed.
* **The same round number can appear twice.** ``round 1`` and ``round 1 (fix
  verification)`` are two PASSES of round 1, not rounds 1 and 2. Passes are
  recorded in order, and the newest pass decides the lane's state.

WHAT IT RETURNS. Per lane (``agent``/``design``/``qa``/``ux``), the passes and
remediations in order, plus one :class:`LaneState` summarising the lane against a
known head SHA. Freshness is a prefix comparison: ``fresh`` when the head starts
with the reviewed SHA (7+ hex), ``stale`` when it does not, ``unknown`` when the
comment named no SHA. The "non-conflicting parallel fixes since review" judgement
that the merge gate also allows is deliberately NOT automated — it is a reviewer's
call, and the design says so (design §C.5).

CONSTRAINTS.

* **Comments are remote, untrusted text.** Nothing here executes or interpolates
  it; the parser only reads lines, and the returned strings are quoted data.
* **The author is never used.** The fleet shares one forge account, so a comment's
  author proves nothing about independence (operator rule). ``reviewer`` is
  reported verbatim from the ``Reviewer:`` field and no independence claim is
  derived from it.
* **Bounded work.** Fields are scanned in the first :data:`FIELD_SCAN_LINES` lines
  and every regex is linear. Fixture bodies are a few KB; a huge comment costs one
  pass, not a backtracking blowup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as _field
from typing import Any, Iterable, Mapping, Sequence

#: Lanes, in the order a UI shows them. ``agent`` is the code lane.
LANES: tuple[str, ...] = ("agent", "design", "qa", "ux")

#: The lane's state, as a word the UI can render directly (see :data:`STATE_COPY`).
STATE_AWAITING = "awaiting_review"
STATE_FINDINGS_OPEN = "findings_open"
STATE_REMEDIATION_POSTED = "remediation_posted"
STATE_CLEAN = "clean"
STATE_TERMINAL = "terminal"
STATE_UNSTATED = "unstated"

STATE_COPY: dict[str, str] = {
    STATE_AWAITING: "awaiting review",
    STATE_FINDINGS_OPEN: "findings open",
    STATE_REMEDIATION_POSTED: "remediation posted",
    STATE_CLEAN: "clean",
    STATE_TERMINAL: "terminal",
    STATE_UNSTATED: "reviewed, verdict not stated",
}

FRESHNESS_FRESH = "fresh"
FRESHNESS_STALE = "stale"
FRESHNESS_UNKNOWN = "unknown"

#: Fewer than this many hex characters is a short SHA, not a truncated one: the
#: gate's own instruction says "prefix ≥ 7", and a 4-char prefix would match too
#: many commits to be a freshness claim.
MIN_SHA_PREFIX = 7

#: Fields are read from the top of the comment. Real ones put them in the first
#: six lines; 40 leaves room for a preamble without scanning a whole review body.
FIELD_SCAN_LINES = 40

_DASH = "\u2014\u2013\\-:"
#: ``### Agent review — round 2 (delta)`` and its siblings, including
#: ``### QA remediation — round 1`` (the bare "qa" lane spelling a QA review
#: itself uses). The round number may sit later in the line, inside a
#: parenthetical: real clients write ``fold convergence (round 1 scope, fold #8)``.
_HEADER = re.compile(
    rf"^\s*#{{2,4}}\s*\**\s*"
    rf"(?P<lane>agent review|design review|qa report|qa remediation|qa|ux review|ux remediation|ux)"
    rf"(?P<remediation>\s+remediation)?"
    rf"\s*[{_DASH}]\s*"
    rf"(?P<prefix>[^\n]*?)\bround\s+(?P<round>\d+)\b(?P<tail>.*)$",
    re.IGNORECASE,
)
#: The same lane words with no round number at all: a header we recognise as a
#: convention comment but cannot place. It is reported with ``round=None`` rather
#: than dropped, because dropping it would silently hide a review.
_HEADER_NO_ROUND = re.compile(
    r"^\s*#{2,4}\s*\**\s*"
    r"(?P<lane>agent review|design review|qa report|qa remediation|qa|ux review|ux remediation|ux)"
    r"(?P<remediation>\s+remediation)?\s*$",
    re.IGNORECASE,
)

_FIELD = re.compile(
    r"^\W*(?P<name>reviewer|scope|head(?:\s+under\s+test)?|head|verdict)\W*\s*[:"
    rf"{_DASH}]\s*(?P<value>.*)$",
    re.IGNORECASE,
)
_VERDICT_HEADING = re.compile(r"^\s*#{3,4}\s*\**\s*verdict\s*\**\s*:?\s*$", re.IGNORECASE)

#: ``base..head``, with or without backticks. Both sides must be hex for the END
#: to be a reviewed head; a symbolic base (``main..72bea95``) is fine.
_SHA_RANGE = re.compile(
    r"`?\b(?P<base>[0-9a-f]{7,40}|[A-Za-z0-9._/-]+)\.\.(?P<head>[0-9a-f]{7,40})`?"
)
_SHA = re.compile(r"\b(?P<sha>[0-9a-f]{7,40})\b")

#: The verdict vocabulary, applied to the LEADING TOKEN of the verdict line after
#: ``**``, backticks and a ``Verdict:`` label are stripped.
_VERDICT_OPEN = re.compile(
    r"^(not\s+clean|not\s+terminal|not\s+safe|changes[\s-]required|fail|failed|blocked|"
    r"request(ed)?\s+changes)",
    re.IGNORECASE,
)
_VERDICT_CLEAN = re.compile(
    r"^(clean|pass|passed|approve|approved|terminal|lgtm|signs?\s+off)", re.IGNORECASE
)
_TERMINAL_WORD = re.compile(r"\bterminal\b", re.IGNORECASE)

#: Leading-token-only fallback for a verdict line with no ``Verdict:`` label — a
#: standalone bold paragraph such as ``**Clean — merge-ready.**``. Only these exact
#: leading words qualify, so prose that merely contains "clean" is never a verdict.
_BARE_VERDICT = re.compile(
    r"^(clean|not\s+clean|pass|approved?|terminal|not\s+terminal|changes[\s-]?required|"
    r"lgtm|signs?\s+off)\b",
    re.IGNORECASE,
)

_FINDING_ID = re.compile(r"\b(?P<id>[A-Z]{1,3}-?\d{1,3})(?![0-9A-Za-z])")
_DISPOSITION = re.compile(r"\b(?P<word>fixed|resolved|rejected|deferred|recorded)\b", re.IGNORECASE)
_DISPOSITION_SHA = re.compile(r"[`(](?P<sha>[0-9a-f]{7,40})[`)]")
#: A finding id and its disposition must be close together: the real forms are
#: ``**F1 — fixed (cd3a94d605).**``, ``| **R1** (MAJOR) — … | **fixed (d3d2f900).** |``
#: and ``- N1 (NIT): **deferred** — …``. The window stops at the next finding id,
#: so a disposition can never be borrowed from a neighbouring finding.
_DISPOSITION_WINDOW = 160

#: How far past a disposition word its SHA may sit. The real spellings are
#: ``fixed (`d3d2f900`)`` and ``fixed in `f0eb645f21```, so the window has to clear the
#: bracket or the word ``in`` plus eight hex characters — a tight 8-character window
#: matched neither and reported every real fix as having no commit.
_DISPOSITION_SHA_WINDOW = 24


@dataclass(frozen=True)
class Comment:
    """One normalised PR/MR comment: what the parser actually reads."""

    id: str
    body: str
    created_at: str = ""
    url: str = ""

    @staticmethod
    def from_payload(raw: object) -> "Comment | None":
        if not isinstance(raw, Mapping):
            return None
        body = raw.get("body")
        if not isinstance(body, str) or not body.strip():
            return None
        return Comment(
            id=str(raw.get("id") or ""),
            body=body,
            created_at=str(raw.get("created_at") or ""),
            url=str(raw.get("url") or ""),
        )


@dataclass(frozen=True)
class Remediation:
    """One finding's disposition inside a remediation comment."""

    finding: str
    disposition: str  # fixed | rejected | deferred | recorded
    sha: str | None = None
    verbatim: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "finding": self.finding,
            "disposition": self.disposition,
        }
        if self.sha:
            payload["sha"] = self.sha
        if self.verbatim and self.verbatim.lower() != self.disposition:
            payload["verbatim"] = self.verbatim
        return payload


@dataclass(frozen=True)
class ReviewPass:
    """One convention comment: a lane's review or remediation at some pass number."""

    lane: str
    kind: str  # "review" | "remediation"
    round: int | None
    qualifier: str = ""
    sequence: int = 0  # pass order within (lane, round, kind); 1-based
    reviewer: str = ""
    scope: str = ""
    head: str = ""
    reviewed_head: str | None = None
    verdict: str = ""
    verdict_class: str = STATE_UNSTATED  # findings_open | clean | terminal | unstated
    remediation: tuple[Remediation, ...] = ()
    comment_id: str = ""
    created_at: str = ""
    url: str = ""

    @property
    def is_verdict_clean(self) -> bool:
        return self.verdict_class in (STATE_CLEAN, STATE_TERMINAL)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lane": self.lane,
            "kind": self.kind,
            "round": self.round,
            "sequence": self.sequence,
            "verdict_class": self.verdict_class,
        }
        for name in ("qualifier", "reviewer", "reviewed_head", "verdict", "comment_id"):
            value = getattr(self, name)
            if value:
                payload[name] = value
        if self.remediation:
            payload["remediation"] = [item.to_payload() for item in self.remediation]
        return payload


@dataclass(frozen=True)
class LaneState:
    """One lane's answer for a UI: what state it is in, and how fresh that is."""

    lane: str
    state: str
    round: int | None = None
    qualifier: str = ""
    reviewed_head: str | None = None
    freshness: str = FRESHNESS_UNKNOWN
    reviewer: str = ""
    verdict: str = ""
    verdict_class: str = STATE_UNSTATED
    remediation: tuple[Remediation, ...] = ()

    @property
    def copy(self) -> str:
        """The state word plus its freshness, one derived string for tooltip and aria."""
        text = STATE_COPY.get(self.state, self.state)
        if self.freshness == FRESHNESS_FRESH:
            return f"{text}, fresh"
        if self.freshness == FRESHNESS_STALE:
            reviewed = (self.reviewed_head or "")[:7]
            return f"{text}, stale (reviewed {reviewed})"
        return f"{text}, freshness unknown"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lane": self.lane,
            "state": self.state,
            "state_copy": self.copy,
            "freshness": self.freshness,
            "verdict_class": self.verdict_class,
        }
        if self.round is not None:
            payload["round"] = self.round
        for name in ("qualifier", "reviewed_head", "reviewer", "verdict"):
            value = getattr(self, name)
            if value:
                payload[name] = value
        if self.remediation:
            payload["remediation"] = [item.to_payload() for item in self.remediation]
        return payload


@dataclass
class RoundReport:
    """Every convention comment in one code request, and the derived lane states."""

    passes: list[ReviewPass] = _field(default_factory=list)
    ignored: list[str] = _field(default_factory=list)

    def lane_passes(self, lane: str) -> list[ReviewPass]:
        return [item for item in self.passes if item.lane == lane]

    def states(self, head_sha: str | None = None, *, is_open: bool = True) -> list[LaneState]:
        """One :class:`LaneState` per lane that has a comment (plus ``agent`` on an open PR).

        The lane order is :data:`LANES`, and a lane with no comment is absent —
        except the code lane on an open code request, which is ``awaiting_review``:
        "no agent review yet" is an answer, while "no design review yet" on a PR
        that needs none is noise.
        """
        out: list[LaneState] = []
        for lane in LANES:
            items = self.lane_passes(lane)
            if not items:
                if lane == "agent" and is_open:
                    out.append(LaneState(lane=lane, state=STATE_AWAITING))
                continue
            out.append(self._state_for(lane, items, head_sha))
        return out

    @staticmethod
    def _state_for(lane: str, items: Sequence[ReviewPass], head_sha: str | None) -> LaneState:
        newest = items[-1]
        state = STATE_UNSTATED
        if newest.kind == "remediation":
            state = STATE_REMEDIATION_POSTED
        elif newest.verdict_class == STATE_FINDINGS_OPEN:
            state = STATE_FINDINGS_OPEN
        elif newest.verdict_class == STATE_TERMINAL:
            state = STATE_TERMINAL
        elif newest.verdict_class == STATE_CLEAN:
            state = STATE_CLEAN
        else:
            state = STATE_UNSTATED
        return LaneState(
            lane=lane,
            state=state,
            round=newest.round,
            qualifier=newest.qualifier,
            reviewed_head=newest.reviewed_head,
            freshness=freshness(newest.reviewed_head, head_sha),
            reviewer=newest.reviewer,
            verdict=newest.verdict,
            verdict_class=newest.verdict_class,
            remediation=newest.remediation,
        )

    def to_payload(self, head_sha: str | None = None, *, is_open: bool = True) -> dict[str, Any]:
        return {
            "lanes": [state.to_payload() for state in self.states(head_sha, is_open=is_open)],
            "passes": [item.to_payload() for item in self.passes],
            "ignored": list(self.ignored),
        }


def freshness(reviewed_head: str | None, head_sha: str | None) -> str:
    """``fresh`` / ``stale`` / ``unknown`` for a reviewed head against the live head.

    Unknown on either side is ``unknown``: a review with no SHA claims no
    freshness, and a code request whose head could not be read cannot invalidate a
    review that named one.
    """
    if not reviewed_head or not head_sha:
        return FRESHNESS_UNKNOWN
    reviewed = reviewed_head.lower()
    head = head_sha.lower()
    if len(reviewed) < MIN_SHA_PREFIX:
        return FRESHNESS_UNKNOWN
    return FRESHNESS_FRESH if head.startswith(reviewed) else FRESHNESS_STALE


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def parse_comment(comment: Comment) -> ReviewPass | None:
    """One comment as a :class:`ReviewPass`, or ``None`` when it is not a convention comment.

    A comment is a convention comment when its FIRST line is a lane heading. A
    heading with no round number is still accepted (``round=None``): the comment is
    a review, and the copy can say so without inventing a number. Everything else is
    ignored — including the real non-convention headers (``### Merge disclosure``,
    ``### v0.68.18 released``, ``### Sir Knight Lop the Second — verdict``) and the
    unheaded comments (``**Addendum to the round above …**``, ``Shipped in …``).
    """
    lines = comment.body.strip().splitlines()
    if not lines:
        return None
    header = _HEADER.match(lines[0])
    if header is not None:
        lane = _lane_of(header.group("lane"))
        kind = "remediation" if header.group("remediation") else "review"
        round_number = int(header.group("round"))
        qualifier = _qualifier(header.group("prefix"), header.group("tail"))
    else:
        plain = _HEADER_NO_ROUND.match(lines[0])
        if plain is None:
            return None
        lane = _lane_of(plain.group("lane"))
        kind = "remediation" if plain.group("remediation") else "review"
        round_number = None
        qualifier = ""
    fields, verdict_heading = _scan_fields(lines)
    verdict = fields.get("verdict", "") or verdict_heading
    head_field = fields.get("head", "")
    scope = fields.get("scope", "")
    reviewed_head = _reviewed_head(scope, head_field)
    verdict_class = classify_verdict(verdict) if kind == "review" else STATE_UNSTATED
    remediation = parse_remediation(comment.body) if kind == "remediation" else ()
    return ReviewPass(
        lane=lane,
        kind=kind,
        round=round_number,
        qualifier=qualifier,
        reviewer=fields.get("reviewer", "").strip(),
        scope=scope.strip(),
        head=head_field.strip(),
        reviewed_head=reviewed_head,
        verdict=verdict.strip(),
        verdict_class=verdict_class,
        remediation=remediation,
        comment_id=comment.id,
        created_at=comment.created_at,
        url=comment.url,
    )


def _lane_of(word: str) -> str:
    lowered = word.strip().lower()
    if lowered.startswith("agent"):
        return "agent"
    if lowered.startswith("design"):
        return "design"
    if lowered.startswith("qa"):
        return "qa"
    return "ux"


def _qualifier(prefix: str, tail: str) -> str:
    """The words around the round number, with the number itself removed.

    ``### Agent review — round 2 (delta verification)`` → ``delta verification``;
    ``### Agent review — fold convergence (round 1 scope, fold #8)`` →
    ``fold convergence (scope, fold #8)``.

    The qualifier is kept as text because it carries real meaning for a reader
    ("fix verification" is a re-review of the same round, "release bump" is a scope
    note) and no rule consults it — so it is normalised, never interpreted.
    """
    text = f"{(prefix or '').strip()} {(tail or '').strip()}".strip()
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = _strip_wrapping_parens(text.strip())
    text = re.sub(r"^[\s\u2014\u2013:,-]+", "", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def _strip_wrapping_parens(text: str) -> str:
    """Drop ONE wrapping paren pair, and only a BALANCED one.

    ``(delta)`` → ``delta``; ``fold convergence (scope, fold #8)`` keeps its
    brackets, because the opening one is not at the start. A plain ``strip("()")``
    removed the closer of the second spelling and produced ``…fold #8`` — a
    qualifier a reader would read as truncated.
    """
    if text.startswith("(") and text.endswith(")") and text.count("(") == text.count(")"):
        return text[1:-1].strip()
    return text


def _scan_fields(lines: Sequence[str]) -> tuple[dict[str, str], str]:
    """The ``Reviewer:``/``Scope:``/``Head:``/``Verdict:`` fields, first spelling wins.

    ``first`` rather than ``last`` because a review quotes its own prompt further
    down the body: a later ``Scope:`` line is usually describing something else, and
    the real comments always put the fields at the top.
    """
    fields: dict[str, str] = {}
    verdict_heading = ""
    for index, line in enumerate(lines[1:FIELD_SCAN_LINES]):
        stripped = line.strip()
        if _VERDICT_HEADING.match(stripped) is not None:
            # Handled after this loop: the heading is the one field scanned past
            # the top-lines bound (see this function's docstring).
            break
        if _VERDICT_HEADING.match(stripped) and "verdict" not in fields:
            # The heading carries no text: the verdict is the next non-empty line,
            # which in the real comments is a bold paragraph (``**Not safe to merge…**``).
            for following in lines[index + 2 : index + 6]:
                if following.strip():
                    verdict_heading = following.strip()
                    break
            fields.setdefault("verdict", verdict_heading)
            continue
        match = _FIELD.match(stripped)
        if match is None:
            continue
        name = match.group("name").lower().split()
        key = "head" if name[0] == "head" else name[0]
        value = match.group("value").strip()
        if key in fields:
            continue
        fields[key] = value
        fields.setdefault(f"_line_{key}", stripped)
    if "verdict" not in fields:
        # The ``#### Verdict`` heading, ANYWHERE in the body, and this is the one
        # field not bounded to the top: the real review on
        # ``damianvtran/local-operator#2094`` carries its heading on line 61, and a
        # 40-line bound reported that review as ``unstated``. The heading match is
        # exact-line and the classifier only reads a recognised leading token, so a
        # whole-body scan cannot turn prose into a verdict.
        for index, line in enumerate(lines):
            if _VERDICT_HEADING.match(line.strip()) is None:
                continue
            following = next((item for item in lines[index + 1 : index + 6] if item.strip()), "")
            if following.strip():
                verdict_heading = following.strip()
                fields.setdefault("verdict", verdict_heading)
            break
    if "verdict" not in fields:
        # No ``Verdict:`` label: a standalone bold verdict paragraph is the third
        # real spelling (``**Clean — merge-ready.**``). Only the recognised leading
        # words qualify, so ordinary prose never becomes a verdict.
        for line in lines[1:FIELD_SCAN_LINES]:
            stripped = line.strip()
            if not (stripped.startswith("**") and stripped.endswith("**")):
                continue
            bare = _strip_markup(stripped)
            if _BARE_VERDICT.match(bare):
                fields["verdict"] = stripped
                break
    return fields, verdict_heading


def _strip_markup(text: str) -> str:
    """A verdict line reduced to the leading words the classifier reads."""
    text = text.strip()
    text = re.sub(r"^[\s*>|\-]+", "", text)
    text = text.replace("**", "")
    text = text.strip().strip("`").strip()
    text = re.sub(r"^verdict\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def classify_verdict(verdict: str) -> str:
    """``findings_open`` / ``clean`` / ``terminal`` / ``unstated``, by LEADING TOKEN.

    The leading token decides, and the whole line is consulted for one thing only:
    the word ``terminal`` upgrades ``clean``. A line that starts with neither
    vocabulary is ``unstated`` — the honest answer for the real reviews that carry a
    fix-verification list and no verdict at all.
    """
    text = _strip_markup(verdict)
    if not text:
        return STATE_UNSTATED
    first = text.split(" ")[0] if text else ""
    if _VERDICT_OPEN.match(text):
        return STATE_FINDINGS_OPEN
    if text.lower().startswith("fail") and not re.match(r"^fail(ed)?\b.*\b0\s+fail", text, re.I):
        # ``FAIL — 3 findings`` is a failing verdict; ``PASS — 0 FAIL`` is not (its
        # leading token is PASS and it never reaches here).
        return STATE_FINDINGS_OPEN
    if _VERDICT_CLEAN.match(text):
        return STATE_TERMINAL if _TERMINAL_WORD.search(text) else STATE_CLEAN
    if not first:
        return STATE_UNSTATED
    return STATE_UNSTATED


def _reviewed_head(scope: str, head: str) -> str | None:
    """The reviewed SHA: the END of a ``sha..sha`` range in Scope, else Scope's, else Head's.

    Priority matters and comes from the gate's own rule — the round is fresh when
    the reviewed head IS the current head, and the head is the RIGHT-hand side of
    the scope range. The ``Head:``/``Head under test:`` field is consulted last
    because a QA round that quotes a range in Scope has already said the same thing
    more precisely.
    """
    for text in (scope, head):
        if not text:
            continue
        match = _SHA_RANGE.search(text)
        if match is not None:
            return match.group("head").lower()
        found = _SHA.search(text)
        if found is not None:
            return found.group("sha").lower()
    return None


def parse_remediation(body: str, limit: int = 40) -> tuple[Remediation, ...]:
    """Every finding's disposition in a remediation comment, in the order found.

    ``No findings to remediate`` (a real comment on a clean round) yields nothing,
    which is the correct answer: there is no finding to record a disposition for. A
    count of OPEN findings is never inferred here — the design forbids it, because
    prose is not a ledger.
    """
    out: list[Remediation] = []
    seen: set[tuple[str, str]] = set()
    for match in _FINDING_ID.finditer(body):
        if len(out) >= limit:
            break
        finding = match.group("id")
        window = body[match.end() : match.end() + _DISPOSITION_WINDOW]
        following = _FINDING_ID.search(window)
        if following is not None:
            window = window[: following.start()]
        disposition = _DISPOSITION.search(window)
        if disposition is None:
            continue
        sha_match = _DISPOSITION_SHA.search(window[: disposition.end() + _DISPOSITION_SHA_WINDOW])
        sha = sha_match.group("sha").lower() if sha_match is not None else None
        key = (finding, disposition.group("word").lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Remediation(
                finding=finding,
                disposition=_canonical_disposition(disposition.group("word")),
                sha=sha,
                verbatim=disposition.group("word").lower(),
            )
        )
    return tuple(out)


def _canonical_disposition(word: str) -> str:
    lowered = word.lower()
    if lowered == "resolved":
        return "fixed"
    return lowered


def parse(
    comments: Iterable[Comment] | Iterable[Mapping[str, Any]],
    *,
    head_sha: str | None = None,
    is_open: bool = True,
) -> RoundReport:
    """Parse comments **already ordered by ``created_at``** into a :class:`RoundReport`.

    Ordering is the caller's because the forge returns comments in order and a
    parser that re-sorted would have to guess at ties. Passes of the same
    (lane, round, kind) are numbered in the order given, which is what lets the
    same round appear twice (``round 1`` then ``round 1 (fix verification)``) and
    still leave the newest pass deciding the lane.
    """
    report = RoundReport()
    counters: dict[tuple[str, int | None, str], int] = {}
    for raw in comments:
        comment = raw if isinstance(raw, Comment) else Comment.from_payload(raw)
        if comment is None:
            continue
        parsed = parse_comment(comment)
        if parsed is None:
            report.ignored.append(comment.id)
            continue
        key = (parsed.lane, parsed.round, parsed.kind)
        counters[key] = counters.get(key, 0) + 1
        report.passes.append(_with_sequence(parsed, counters[key]))
    return report


def _with_sequence(item: ReviewPass, sequence: int) -> ReviewPass:
    return ReviewPass(
        lane=item.lane,
        kind=item.kind,
        round=item.round,
        qualifier=item.qualifier,
        sequence=sequence,
        reviewer=item.reviewer,
        scope=item.scope,
        head=item.head,
        reviewed_head=item.reviewed_head,
        verdict=item.verdict,
        verdict_class=item.verdict_class,
        remediation=item.remediation,
        comment_id=item.comment_id,
        created_at=item.created_at,
        url=item.url,
    )


__all__ = [
    "Comment",
    "FIELD_SCAN_LINES",
    "FRESHNESS_FRESH",
    "FRESHNESS_STALE",
    "FRESHNESS_UNKNOWN",
    "LANES",
    "LaneState",
    "MIN_SHA_PREFIX",
    "Remediation",
    "ReviewPass",
    "RoundReport",
    "STATE_AWAITING",
    "STATE_CLEAN",
    "STATE_COPY",
    "STATE_FINDINGS_OPEN",
    "STATE_REMEDIATION_POSTED",
    "STATE_TERMINAL",
    "STATE_UNSTATED",
    "classify_verdict",
    "freshness",
    "parse",
    "parse_comment",
    "parse_remediation",
]
