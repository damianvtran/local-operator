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
    "CARRY_TOKEN_BUDGET",
    "CARRY_ASK_ITEMS",
    "CARRY_DECISION_ITEMS",
    "CARRY_IDENTIFIER_ITEMS",
    "ASK_ITEM_CHARS",
    "DECISION_ITEM_CHARS",
    "CARRY_HEADINGS",
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

#: The token reservation the archive's own drop-budget loop honours, so the
#: carry's cost is priced BEFORE pages are kept (see ``compact_to_archive``).
CARRY_TOKEN_BUDGET = 8_000

#: Per-tier item caps. Tier 1 is the widest because an identifier line is
#: short; tier 2/3 are narrower because a sentence is long.
CARRY_IDENTIFIER_ITEMS = 64
CARRY_ASK_ITEMS = 60
CARRY_DECISION_ITEMS = 40

#: Per-item truncation: an ask is carried as its own substance, not the whole
#: pasted blob that followed it.
ASK_ITEM_CHARS = 320
DECISION_ITEM_CHARS = 220

#: The carry's section headings. ``memory_capture.extract`` recognises these as
#: semantic sections (its ``_HEADINGS`` set is widened to match), which is how a
#: mined decision/ask becomes a memory://session fact without a second scan.
CARRY_HEADINGS = ("Operator asks", "Decisions / constraints", "Identifiers")

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
#: material a resumed session cannot re-derive from the workspace.
_MAX_LINE_SCAN = 8192


@dataclass
class Carry:
    """The tiered digest: identifiers (never evicted), asks, decisions."""

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
    for raw in text.splitlines():
        header = _ROLE_HEADER.match(raw.strip())
        if header:
            role = "user" if header.group(1) == "User" else "assistant"
            continue
        line = raw.strip()
        if not line:
            continue
        # Identifiers: one bounded scan per line. The scan examines at most
        # ``_MAX_LINE_SCAN`` chars and never windows mid-token, so a forced cut
        # cannot publish half a URI or hex id as its own fact.
        scanned = line[:_MAX_LINE_SCAN]
        for match in _LOCATOR.finditer(scanned):
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
        elif _DECISION_CUE.search(line):
            decisions.append(" ".join(line.split())[:DECISION_ITEM_CHARS])

    return Carry(
        identifiers=[value for value, _ in identifiers.most_common()],
        asks=_dedupe(asks),
        decisions=_dedupe(decisions),
    )


def parse_carry(text: str) -> Carry:
    """Revive a rendered carry so a later pass can merge into it rather than restart."""
    carry = Carry()
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("- "):
            continue
        value = line[2:].strip()
        if not value:
            continue
        if section == "Identifiers":
            carry.identifiers.append(value)
        elif section == "Operator asks":
            carry.asks.append(value)
        elif section == "Decisions / constraints":
            carry.decisions.append(value)
    return carry


def render_carry(carry: Carry) -> str:
    """Render the digest. Sections that are empty are omitted entirely so an
    empty carry renders as the empty string (never a heading with no rows)."""
    lines: list[str] = []
    if carry.asks:
        lines.append("## Operator asks")
        lines.extend(f"- {item}" for item in carry.asks)
    if carry.decisions:
        lines.append("## Decisions / constraints")
        lines.extend(f"- {item}" for item in carry.decisions)
    if carry.identifiers:
        lines.append("## Identifiers")
        lines.extend(f"- {item}" for item in carry.identifiers)
    return "\n".join(lines)


def carry_is_empty(carry: Carry) -> bool:
    return not (carry.asks or carry.decisions or carry.identifiers)


def _cap_tier(
    items: list[str], cap: int, *, budget: int, used: int, oldest_first: bool
) -> tuple[list[str], int]:
    """Keep at most ``cap`` items of a tier, newest-kept, inside ``budget``.

    ``oldest_first`` chooses which end survives when the tier overflows:
    ``False`` drops the oldest (asks/decisions — a live ask outranks a spent
    one); ``True`` drops the newest only for tiers where order is meaningless
    (identifiers are pre-ranked by frequency, so this is never used there).
    """
    if cap <= 0:
        return [], used
    kept: list[str] = []
    for item in reversed(items) if not oldest_first else items:
        cost = len(item) + 3
        if kept and (used + cost > budget or len(kept) >= cap):
            break
        kept.append(item)
        used += cost
    if not oldest_first:
        kept.reverse()
    return kept, used


def merge_carry(previous: Carry, fresh: Carry) -> Carry:
    """Fold a pass's fresh candidates into the prior carry, tier by tier.

    Ordering INSIDE a tier is oldest→newest so the rendered digest reads
    chronologically; EVICTION is by tier (identifiers last to go), and within
    the ask/decision tiers oldest-first, which is the architect's
    importance-not-age ranking.
    """
    identifiers = _dedupe(previous.identifiers + fresh.identifiers)[:CARRY_IDENTIFIER_ITEMS]

    # Identifiers are tier 1 and never evicted inside the cap.
    used = sum(len(item) + 3 for item in identifiers)
    asks_pool = _dedupe(previous.asks + fresh.asks)
    asks, used = _cap_tier(
        asks_pool, CARRY_ASK_ITEMS, budget=CARRY_CHAR_BUDGET, used=used, oldest_first=False
    )
    decisions_pool = _dedupe(previous.decisions + fresh.decisions)
    # Tier 3 and the first to shed: give it whatever budget remains up to its cap.
    decisions, used = _cap_tier(
        decisions_pool,
        CARRY_DECISION_ITEMS,
        budget=CARRY_CHAR_BUDGET,
        used=used,
        oldest_first=False,
    )
    return Carry(identifiers=identifiers, asks=asks, decisions=decisions)


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
