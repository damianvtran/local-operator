"""Deterministic, stdlib-only mining of the material that must survive a pass.

WHY THIS MODULE EXISTS
======================
A snapcompact pass keeps a bounded oldest edge, a bounded newest edge and the
newest few pages VERBATIM; everything between the edges that overflows the
frame/page budget is dropped, and each later pass folds the prior archive text
in and drops the oldest overflow AGAIN. Measured on the operator's own session
(``439818272d84``, 84 passes): ``Archive.truncated_chars`` climbs monotonically
(257,965 → 1,197,200) while the kept text stays ~110k–140k chars, so the oldest
material is re-truncated on every pass and never mined. The agent then loses the
thread and re-reads to catch up, spending back the very tokens compaction saved.

This module mines the pages a pass is ABOUT to drop and folds them into a
bounded, tier-ranked ``carry`` digest that is stored on the archive and replayed
as leading plain text. The carry is what makes the loss NON-repeating: material
that has been mined can be dropped from the pages without being lost from the
conversation, so successive passes stop re-truncating the same history.

WHY IT IS TIER-RANKED, NOT RECENCY-RANKED
=========================================
A bounded ring buffer that evicts oldest-first does NOT fix the cliff — it
delays it by ``cap / mint_rate`` passes and then resumes at the same per-pass
rate (measured by the design review: with a 8,000-char age-ordered carry on a
24-pass synthetic run, loss was 49% and the cliff returned at pass 12; with a
2,000-char carry, 74% and pass 6). The fix is to evict by IMPORTANCE, so the
loss lands on the least important material instead of the oldest:

  1. Identifiers/paths/refs/handles — never evicted inside the cap (they are
     the cheapest, highest-value recall anchors, and what makes a compacted
     session's memory externally checkable).
  2. Operator asks — the operator's own words are never paraphrased away
     (mirrors ``cutpoint.extract_preserved_user_turns``); oldest-first only
     after the identifier tier is whole.
  3. Decisions / constraints — the assistant's stated commitments.
  4. Everything else is not carried at all.

WHY IT IS A STDLIB-ONLY LEAF
============================
``snapcompact`` is consumed by hosts that are forbidden the session/model/
providers/config imports (see ``pass_.py``'s import-isolation note) and must
stay cheap on the request path. Importing the locator regex from
``memory_capture`` would drag in ``local_operator.memory`` → ``sqlite3`` plus
the redaction-shape closure (~292 modules measured). So the locator grammar is
re-stated here in ``re``-only code. It is deliberately the SAME grammar as
``memory_capture._LOCATOR``; ``tests/unit/compaction/test_mine.py`` pins the two
together so they cannot silently drift.

Nothing here opens a store, calls a model or reads a file: it is pure CPU over
strings, so a compaction pass stays local and deterministic.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

__all__ = [
    "CARRY_CHAR_BUDGET",
    "CARRY_FRESH_TOKEN_ALLOWANCE",
    "CARRY_ASK_ITEMS",
    "CARRY_DECISION_ITEMS",
    "CARRY_IDENTIFIER_ITEMS",
    "ASK_ITEM_CHARS",
    "DECISION_ITEM_CHARS",
    "CARRY_TAGS",
    "LOCATOR_PATTERN",
    "Carry",
    "carry_digest_lines",
    "carry_is_empty",
    "merge_carry",
    "mine_carry",
    "parse_carry",
    "render_carry",
]

#: Overall bound on the rendered carry, in characters. Reserved for inside the
#: archive's token budget (``CARRY_TOKEN_BUDGET``) and counted by
#: ``snapcompact.estimate_archive_tokens``, so the pass that exists to get under
#: the threshold still does. ~30k chars ≈ 7.5k tokens, well inside the
#: operator's "under 100k" bar next to the ~42k a replay already costs.
CARRY_CHAR_BUDGET = 30_000

#: The estimator's own fallback char/token density, restated here so this
#: stdlib-only leaf need not import ``compaction.tokens`` (which pulls the
#: package's tokenizer path). ``tests/unit/compaction/test_mine.py`` pins it equal
#: to ``tokens._CHARS_PER_TOKEN_FALLBACK`` so the two cannot drift.
_CARRY_CHARS_PER_TOKEN = 4

#: The token reservation the archive's own drop-budget loop honours, so the
#: carry's cost is priced BEFORE pages are kept (see ``compact_to_archive``).
#: Charged as the ACTUAL size of a carried-forward digest plus the
#: :data:`CARRY_FRESH_TOKEN_ALLOWANCE` mint ceiling. Kept as a named constant for
#: callers that want the reservation a carry-heavy archive can occupy; the budget
#: loop computes its own figure from the live prior carry.
CARRY_TOKEN_BUDGET = 8_000

#: Fresh-mint allowance, charged only when there is no carried digest to price
#: and the pass has pages to mine. It must be an UPPER BOUND on what this pass can
#: mint, because the whole point of the reservation is that a pass meant to get
#: under the line cannot commit above it once the carry is counted. A converged
#: same-session carry mints ~0.7k tokens on real segments, but a pathologically
#: instruction-dense segment mints far more, and an estimate below the true mint
#: let a small window overshoot ``window/2`` (review round 2, MAJOR B). So charge
#: the mint's own ceiling: the rendered carry cannot exceed
#: :data:`CARRY_CHAR_BUDGET` characters, i.e. ``CARRY_CHAR_BUDGET / 4`` tokens at
#: the estimator's fallback density.
CARRY_FRESH_TOKEN_ALLOWANCE = CARRY_CHAR_BUDGET // _CARRY_CHARS_PER_TOKEN

#: Per-tier item caps. Tier 1 is the widest because an identifier line is
#: short; tier 2/3 are narrower because a sentence is long.
CARRY_IDENTIFIER_ITEMS = 64
CARRY_ASK_ITEMS = 60
CARRY_DECISION_ITEMS = 40

#: Per-item truncation: an ask is carried as its own substance, not the whole
#: pasted blob that followed it.
ASK_ITEM_CHARS = 240
DECISION_ITEM_CHARS = 220

#: The tier tags the digest's rows carry (``- Asks: …``). The tagged form is the
#: digest's structural marker: it is what makes a replayed digest refuse to be
#: re-mined as history without a heading that real prose can also contain.
CARRY_TAGS = ("Asks", "Decisions", "Identifiers")

#: Tag -> the ``Carry`` attribute a tagged row feeds on :func:`parse_carry`.
_TIER_ATTRIBUTE: dict[str, str] = {
    "Asks": "asks",
    "Decisions": "decisions",
    "Identifiers": "identifiers",
}

#: Locator grammar — DELIBERATELY IDENTICAL to ``memory_capture._LOCATOR``.
#: Kept as a module constant so the drift-pin test can compare the two source
#: patterns without importing the heavier module.
LOCATOR_PATTERN = (
    r"(?<![\w/])(?P<uri>[A-Za-z][A-Za-z0-9+.-]{0,31}://[^\s<>\"'`]{1,4096})"
    r"|(?<![\w.+-])[\w.+-]{1,64}@[\w.-]{1,255}\.[A-Za-z]{2,24}"
    r"|(?<![\w:/~.-])(?:~?/|\.\.?/|(?:src|docs|tests|local_operator)/)[^\s<>\"'`]{1,400}"
    r"|\bsecret\s{1,8}[A-Z][A-Z0-9_]{2,63}\b|(?<!\w)#[\w-]{2,64}"
    r"|\b(?=[0-9a-f]{7,12}\b)[0-9a-f]*\d[0-9a-f]*\b"
    r"|(?P<reference>\b(?:branch|thread)(?:\s{0,3}:\s{0,3}|\s{1,3})[\w./-]{3,128}(?![\w./-]))"
)
_LOCATOR = re.compile(LOCATOR_PATTERN)

#: Commitment/constraint cues — a line is carried as a DECISION only when it
#: states one. Deliberately conservative: a false positive spends carry budget
#: on prose, which is the tier the eviction policy exists to shed first.
_DECISION_CUE = re.compile(
    r"\b(decided|decision|we(?:'ll| will| should| must)|i(?:'ll| will)|going with|"
    r"instead of|do not|don't|never|must not|avoid|constraint|blocker|blocked|"
    r"in flight|in-progress|next step|open question|hold off|deferred)\b",
    re.IGNORECASE,
)

#: Serialized role headers written by ``snapcompact.serialize_for_snapcompact``.
_ROLE_HEADER = re.compile(
    r"^\[(User|Assistant|Assistant tool calls|Tool result:.*|Tool ERROR:.*)\]$"
)

#: Longest prefix of any single line the locator scan examines. A tool result
#: is already truncated to 2,000 chars by the serializer and a user paste can be
#: arbitrarily long; bounding the scan keeps the pass's CPU flat and independent
#: of the biggest paste, mirroring ``memory_capture``'s bounded-window policy.
#: Nothing load-bearing is lost: the ask for a long paste is carried as its own
#: collapsed first line, and identifiers deep in a 100k-char blob are not the
#: material a resumed session cannot re-derive from the workspace. A match that
#: runs INTO this cut is refused, exactly as ``extract`` refuses one (a fragment
#: is worse than a miss).
_MAX_LINE_SCAN = 8192

#: A digest row: ``- Asks: <item>`` / ``- Decisions: <item>`` /
#: ``- Identifiers: <item>``. This TAGGED form is the carry's structural marker —
#: :func:`render_carry` emits one such row per item, and ``archive_summary``'s
#: bounded digest emits the same shape. It is what makes a replayed digest
#: distinguishable from real history: a plain ``## Identifiers`` heading in a
#: dropped report is NOT a tagged row and must still be mined (review round 2,
#: MAJOR A), while the digest's own rows always are, so re-mining it is refused
#: (review round 1, MAJOR 2).
_CARRY_TAGGED_ROW_RE = re.compile(
    r"^-[ \t]+(?P<tag>Asks|Decisions|Identifiers):[ \t]+(?P<item>.+)$"
)


def _is_carry_digest_line(line: str) -> bool:
    """True for a line belonging to a carried digest rather than real history.

    ONLY the tagged row (``- Identifiers: …`` etc.) counts — see
    :data:`_CARRY_TAGGED_ROW_RE` for why the heading alone is not a signal.
    """
    return bool(_CARRY_TAGGED_ROW_RE.match(line))


@dataclass
class Carry:
    """The tiered digest: identifiers, asks, decisions — all newest-kept at cap.

    Identifiers are the recall anchors (paths, refs, secret names), asks are the
    operator's own words and decisions are the assistant's stated commitments.
    ``merge_carry`` evicts the OLDEST row of every tier once its cap binds, so a
    resumed session keeps the material the current task depends on. That is a
    deliberate reversal of an earlier revision whose identifier tier kept the
    oldest while the other two kept the newest — the asymmetry that silently
    dropped the reference stated on the live task (review round 1, MAJOR 3).
    """

    identifiers: list[str] = field(default_factory=list)
    asks: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def mine_carry(text: str) -> Carry:
    """Extract the tiered candidate material from one serialized block.

    Walks the block's role headers so a line is attributed to the speaker that
    produced it: asks only from ``[User]`` sections, decisions only from
    ``[Assistant]`` sections. Identifiers are mined from everywhere (a path in a
    tool result is as useful as one in prose).
    """
    identifiers: Counter[str] = Counter()
    asks: list[str] = []
    decisions: list[str] = []

    role = "assistant"
    in_carry = False
    for raw in text.splitlines():
        header = _ROLE_HEADER.match(raw.strip())
        if header:
            title = header.group(1)
            # Tool blocks are their own role: a tool ERROR line reading "do not
            # delete …" is an error message, not an assistant decision, and
            # mining it as one wrote a commitment the assistant never made
            # (review round 1, NIT 9).
            role = (
                "user" if title == "User" else ("tool" if title.startswith("Tool") else "assistant")
            )
            in_carry = False  # a role header always ends a carried section
            continue
        line = raw.strip()
        if not line:
            continue
        # A carry DIGEST is never re-mined. A prior pass's carry replays as a
        # leading text block (see ``snapcompact._with_carry_block``), so it sits
        # inside the NEXT pass's ``to_summarize``; without this guard the miner
        # would re-mine its own headings and rows — every ``- `` line under the
        # replayed block would be re-classified as an operator ask and the carry
        # would echo and nest itself (``- - - <ask>``), evicting the genuine asks
        # it exists to retain (review round 1, MAJOR 2). Only TAGGED digest rows
        # latch the skip (see ``_is_carry_digest_line``); a bare section heading
        # does NOT, because a real dropped report can carry one and latching on
        # it darkened a whole section of genuine history (review round 2, MAJOR
        # A). Once a tagged row starts a carried section, the rest is skipped
        # until a role header.
        if in_carry or _is_carry_digest_line(line):
            in_carry = True
            continue
        # Identifiers: one bounded scan per line. The scan examines at most
        # ``_MAX_LINE_SCAN`` chars and, like ``memory_capture.extract``, refuses
        # to publish a token a forced cut bisected — a truncated URL prefix or a
        # half hex id is a fragment no reader can use and that lands in memory as
        # a fact (review round 1, MAJOR 4).
        forced_cut = len(line) > _MAX_LINE_SCAN and not line[_MAX_LINE_SCAN].isspace()
        scanned = line[:_MAX_LINE_SCAN]
        for match in _LOCATOR.finditer(scanned):
            if forced_cut and match.end() == len(scanned):
                continue  # the cut, not a delimiter, ended this match
            value = match.group().rstrip(".,;:)]}")
            if not value or len(value) > 200:
                continue
            # A URI the reader cannot resolve is noise, exactly as in capture.
            if match.lastgroup == "uri" and not value.lower().startswith(("http://", "https://")):
                continue
            if match.lastgroup == "reference" and ":" not in value:
                identifier = value.split(maxsplit=1)[-1]
                if "/" not in identifier and not any(c.isdigit() for c in identifier):
                    continue
            identifiers[value] += 1
        if role == "user":
            collapsed = " ".join(line.split())
            if collapsed:
                asks.append(collapsed[:ASK_ITEM_CHARS])
        elif role == "assistant" and _DECISION_CUE.search(line):
            decisions.append(" ".join(line.split())[:DECISION_ITEM_CHARS])

    # Identifiers in FIRST-SEEN order, not by frequency: merge keeps the newest
    # rows of every tier uniformly (see ``merge_carry``), and a frequency ranking
    # here would reorder them so that "newest-kept" silently kept the OLDEST
    # references instead (review round 1, MAJOR 3). Ordering is chronological so
    # the merge slice is a real recency bound; a Counter still gives the dedupe.
    return Carry(
        identifiers=list(identifiers),
        asks=_dedupe(asks),
        decisions=_dedupe(decisions),
    )


def parse_carry(text: str) -> Carry:
    """Revive a rendered carry so a later pass can merge into it rather than restart.

    Reads :func:`render_carry`'s tagged-row form; a line that is not a tagged row
    is ignored, so a caller that hands archive text alongside the digest cannot
    invent asks from it.
    """
    carry = Carry()
    for raw in text.splitlines():
        match = _CARRY_TAGGED_ROW_RE.match(raw.strip())
        if match is None:
            continue
        attribute = _TIER_ATTRIBUTE.get(match.group("tag"))
        if attribute:
            getattr(carry, attribute).append(match.group("item").strip())
    return carry


def render_carry(carry: Carry) -> str:
    """Render the digest as tagged rows (see :data:`_CARRY_TAGGED_ROW_RE`).

    One row per item, grouped asks → decisions → identifiers. Empty tiers are
    omitted entirely so an empty carry renders as the empty string. The tagged
    form is load-bearing: it is how a replayed digest is recognised and refused
    as history on the next pass.
    """
    lines: list[str] = []
    for tag, items in (
        ("Asks", carry.asks),
        ("Decisions", carry.decisions),
        ("Identifiers", carry.identifiers),
    ):
        lines.extend(f"- {tag}: {item}" for item in items)
    return "\n".join(lines)


def carry_is_empty(carry: Carry) -> bool:
    return not (carry.asks or carry.decisions or carry.identifiers)


def merge_carry(previous: Carry, fresh: Carry) -> Carry:
    """Fold a pass's fresh candidates into the prior carry, tier by tier.

    Ordering INSIDE a tier is oldest→newest so the rendered digest reads
    chronologically. When a tier overflows its cap EVICTION IS NEWEST-KEPT for
    all three tiers, uniformly: a reference, ask or decision stated on the
    CURRENT task is what a resumed session must have to continue, and the older
    material is not lost either — it still rides the archive's verbatim oldest
    edge, its frames, and (via ``memory_capture``) ``memory://session``. The
    identifier tier keeps the same direction as the other two on purpose; an
    earlier revision kept OLDEST identifiers while the other tiers kept newest,
    an asymmetry that silently dropped the newest references (review round 1,
    MAJOR 3).

    After capping, the rendered digest is trimmed from the LOWEST tier up until
    it fits :data:`CARRY_CHAR_BUDGET` INCLUDING its headings, so the character
    budget is a real bound rather than a line budget that headings overshoot
    (review round 1, MINOR 6).
    """
    identifiers = _dedupe(previous.identifiers + fresh.identifiers)[-CARRY_IDENTIFIER_ITEMS:]
    asks = _dedupe(previous.asks + fresh.asks)[-CARRY_ASK_ITEMS:]
    decisions = _dedupe(previous.decisions + fresh.decisions)[-CARRY_DECISION_ITEMS:]
    merged = Carry(identifiers=identifiers, asks=asks, decisions=decisions)
    # Trim lowest-value tier first while the RENDERED digest overshoots: the cap
    # counts characters, and the headings and bullet prefixes are part of them.
    for tier in ("decisions", "asks", "identifiers"):
        while len(render_carry(merged)) > CARRY_CHAR_BUDGET and getattr(merged, tier):
            getattr(merged, tier).pop(0)  # drop the oldest row of the lowest tier in play
    return merged


def carry_digest_lines(carry: Carry, *, max_chars: int = 2_000) -> list[str]:
    """A short, bounded digest of the carry for the archive's text slot.

    WHY: a snapcompact pass's ``archive_summary`` is READING INSTRUCTIONS, not a
    semantic digest — the model must reconstruct narrative by OCR of the frames.
    Appending the highest-tier carry lines gives the text slot real content
    without a model call, so a resumed session can recall the key asks,
    decisions and identifiers cheaply. Bounded hard: the frames are still the
    authority on history, and this must never grow the pass's cost.
    """
    lines: list[str] = []
    used = 0
    for label, items in (
        ("Asks", carry.asks),
        ("Decisions", carry.decisions),
        ("Identifiers", carry.identifiers),
    ):
        if not items:
            continue
        take: list[str] = []
        for item in items:
            cost = len(item) + 4
            if used + cost > max_chars:
                break
            take.append(item)
            used += cost
        if take:
            lines.append(f"- {label}: " + "; ".join(take))
    return lines
