"""Deterministic segmentation and alignment (design A3/A4.1). Pure: no I/O, no model.

The push-side half of the hub merge contract imports THIS module rather than
re-implementing it: two segmenters that disagree on what an "atom" is would make
the same three texts merge differently depending on direction, which is exactly
what the shared vectors (``tests/fixtures/hub_merge/vectors.json``) exist to catch.

Two deliberate deviations from the design's letter, both forced by the
vectors it ships (recorded in the design doc: the amendment under A4.1 and
Implementation note 10):

* The similarity of two atoms is the MAX of a token ratio (words AND
  punctuation are tokens, so ``brief.`` and ``brief`` share ``brief``), a
  character ratio, and a containment rule (one atom's words are a subsequence
  of the other's). A pure whitespace-token ratio scores ``x.``/``x2.`` at 0 and
  ``Be brief.``/``Be brief and cite sources.`` at 0.29, so V01, V06, V07 and V08
  could never align and every edit would read as delete+add.
* A fenced block compares exactly (not whitespace-collapsed): indentation IS
  the content of a YAML or Python fence.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Sequence

#: Two atoms at or above this similarity are the same atom, edited (A4.1).
ALIGN_THRESHOLD = 0.60

_HEADING_RE = re.compile(r"^#{1,6}[ \t]+\S")
_HEADING_TEXT_RE = re.compile(r"^#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])[ \t]+")
_TABLE_RE = re.compile(r"^\s*\|")
# A newline straight after terminal punctuation ends a sentence whatever follows
# (hard-wrapped prose, one-statement-per-line rules); a space needs a capital,
# digit or opener after it so "e.g. this" is not split.
_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:\s+(?=[A-Z0-9\"'(\[`])|\n)")
_MASK_RE = re.compile(r"`[^`\n]*`|\]\([^)\n]*\)")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_WORD_RE = re.compile(r"\w+")


def norm(text: str | None) -> str:
    """A2.4 normalization: LF, no trailing spaces, one blank line max, trimmed."""

    if not text:
        return ""
    lines = [
        line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    out: list[str] = []
    blank = False
    for line in lines:
        if line == "":
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(line)
    return "\n".join(out).strip("\n")


def collapse_ws(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class Atom:
    """The unit of DECISION: a sentence, list item, table row or fenced block."""

    kind: str  # "sentence" | "item" | "row" | "fence"
    text: str
    #: The separator that preceded this atom in its source; reassembly reuses it
    #: so an untouched atom round-trips byte-for-byte.
    lead: str = "\n"
    norm: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        # Fences compare exactly; everything else after whitespace collapse (A3).
        object.__setattr__(
            self, "norm", self.text if self.kind == "fence" else collapse_ws(self.text)
        )


@dataclass(frozen=True)
class Region:
    """A heading section: the unit of REPORTING."""

    key: str
    name: str  # heading text without the #s ("" for the preamble)
    heading: str  # the raw heading line ("" for the preamble)
    atoms: tuple[Atom, ...]
    lead: str = "\n\n"

    @property
    def heading_norm(self) -> str:
        return collapse_ws(self.heading)


def _mask(text: str) -> str:
    return _MASK_RE.sub(lambda m: "\x00" * len(m.group()), text)


def split_sentences(paragraph: str) -> list[tuple[str, str]]:
    """``(lead, sentence)`` pairs of a paragraph; the first lead is ``""``.

    Splits on terminal punctuation followed by whitespace and a capital/digit/
    opener, never inside an inline code span or a link target (masked first).
    """

    masked = _mask(paragraph)
    out: list[tuple[str, str]] = []
    pos = 0
    lead = ""
    for m in _SENTENCE_RE.finditer(masked):
        out.append((lead, paragraph[pos : m.start()]))
        lead = paragraph[m.start() : m.end()]
        pos = m.end()
    out.append((lead, paragraph[pos:]))
    return [(lead, s) for lead, s in out if s]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _is_block_start(line: str) -> bool:
    return bool(_FENCE_RE.match(line) or _LIST_RE.match(line) or _TABLE_RE.match(line))


def _atoms_of(body: list[str], *, preamble: bool) -> tuple[Atom, ...]:
    atoms: list[Atom] = []
    i, n = 0, len(body)
    blank = False
    while i < n:
        line = body[i]
        if not line.strip():
            blank = True
            i += 1
            continue
        lead = "\n\n" if blank else "\n"
        if preamble and not atoms:
            lead = ""
        blank = False
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            j = i + 1
            while j < n:
                closing = _FENCE_RE.match(body[j])
                if (
                    closing
                    and closing.group(1)[0] == marker[0]
                    and len(closing.group(1)) >= len(marker)
                ):
                    break
                j += 1
            end = min(j, n - 1)
            atoms.append(Atom("fence", "\n".join(body[i : end + 1]), lead))
            i = end + 1
            continue
        item = _LIST_RE.match(line)
        if item:
            base = len(item.group(1))
            j = i + 1
            while j < n:
                nxt = body[j]
                if not nxt.strip():
                    # A blank line stays inside the item only when the next
                    # non-blank line is indented under it (a loose list item).
                    k = j + 1
                    while k < n and not body[k].strip():
                        k += 1
                    if k < n and _indent(body[k]) > base and not _FENCE_RE.match(body[k]):
                        j = k
                        continue
                    break
                if _indent(nxt) > base:
                    j += 1
                    continue
                if _LIST_RE.match(nxt) or _TABLE_RE.match(nxt) or _FENCE_RE.match(nxt):
                    break
                j += 1  # lazy continuation of the item's paragraph
            atoms.append(Atom("item", "\n".join(body[i:j]).rstrip(), lead))
            i = j
            continue
        if _TABLE_RE.match(line):
            atoms.append(Atom("row", line.strip(), lead))
            i += 1
            continue
        j = i
        while j < n and body[j].strip() and not _is_block_start(body[j]):
            j += 1
        paragraph = "\n".join(body[i:j])
        for k, (sep, sentence) in enumerate(split_sentences(paragraph)):
            atoms.append(Atom("sentence", sentence, lead if k == 0 else sep))
        i = j
    return tuple(atoms)


def segment(text: str | None) -> tuple[Region, ...]:
    """A3: normalize, split into heading regions, split regions into atoms."""

    lines = norm(text).split("\n") if norm(text) else []
    raw: list[tuple[str, list[str], str]] = []  # (heading line, body, lead)
    heading = ""
    body: list[str] = []
    lead = ""
    in_fence: str | None = None
    started = False
    for idx, line in enumerate(lines):
        fence = _FENCE_RE.match(line)
        if fence:
            char = fence.group(1)[0]
            in_fence = None if in_fence == char else (in_fence or char)
        if in_fence is None and not fence and _HEADING_RE.match(line):
            if started or body:
                raw.append((heading, body, lead))
            heading = line
            body = []
            started = True
            lead = "\n\n" if idx > 0 and lines[idx - 1] == "" else "\n"
            if idx == 0:
                lead = ""
            continue
        body.append(line)
    if started or body:
        raw.append((heading, body, lead))

    seen: dict[str, int] = {}
    regions: list[Region] = []
    for heading, body, lead in raw:
        if heading:
            m = _HEADING_TEXT_RE.match(heading)
            name = (m.group(1) if m else heading).strip()
            base = collapse_ws(name).casefold()
            ordinal = seen.get(base, 0)
            seen[base] = ordinal + 1
            key = f"{base}#{ordinal}"
        else:
            name, key = "", ""
        atoms = _atoms_of(body, preamble=not heading)
        regions.append(Region(key, name, heading, atoms, lead or "\n\n"))
    return tuple(regions)


def render(regions: Sequence[Region]) -> str:
    """Reassemble regions into normalized text; the inverse of :func:`segment`."""

    parts: list[str] = []
    first = True
    for region in regions:
        if not region.heading and not region.atoms:
            continue
        piece = region.heading
        prev: Atom | None = None
        for atom in region.atoms:
            lead = atom.lead
            if prev is None:
                lead = "" if not region.heading else ("\n\n" if lead == "\n\n" else "\n")
            elif lead == " " and (prev.kind != "sentence" or atom.kind != "sentence"):
                lead = "\n"
            piece += lead + atom.text
            prev = atom
        if not first:
            piece = (region.lead if region.lead in ("\n", "\n\n") else "\n\n") + piece
        parts.append(piece)
        first = False
    return norm("".join(parts))


# -- similarity and alignment ---------------------------------------------------


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def words(text: str) -> list[str]:
    """Casefolded word tokens (punctuation dropped); the reduction/regrow ruler."""

    return [w.casefold() for w in _WORD_RE.findall(text)]


def is_subsequence(short: Sequence[str], long: Sequence[str]) -> bool:
    it = iter(long)
    return all(any(tok == other for other in it) for tok in short)


def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    best = 0.0
    if ta and tb:
        best = difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()
    best = max(best, difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())
    wa, wb = words(a), words(b)
    short, long_ = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    if len(short) >= 2 and is_subsequence(short, long_):
        best = max(best, ALIGN_THRESHOLD)
    return best


@dataclass(frozen=True)
class Pairing:
    """``pairs[i] = (j, modified)`` for the i-th first-side atom matched to j."""

    pairs: dict[int, tuple[int, bool]]

    def other(self, i: int) -> int | None:
        hit = self.pairs.get(i)
        return hit[0] if hit else None


def align(first: Sequence[Atom], second: Sequence[Atom]) -> Pairing:
    """A4.1: exact match left-to-right, then greedy best-ratio >= 0.60."""

    pairs: dict[int, tuple[int, bool]] = {}
    used: set[int] = set()
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            if j not in used and a.norm == b.norm:
                pairs[i] = (j, False)
                used.add(j)
                break
    loose_a = [i for i in range(len(first)) if i not in pairs]
    loose_b = [j for j in range(len(second)) if j not in used]
    scored: list[tuple[float, int, int]] = []
    for i in loose_a:
        for j in loose_b:
            # kind must agree: a table row never "edits into" a sentence.
            if first[i].kind != second[j].kind:
                continue
            score = similarity(first[i].norm, second[j].norm)
            if score >= ALIGN_THRESHOLD:
                scored.append((score, i, j))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    taken_a: set[int] = set()
    taken_b: set[int] = set()
    for _, i, j in scored:
        if i in taken_a or j in taken_b:
            continue
        pairs[i] = (j, True)
        taken_a.add(i)
        taken_b.add(j)
    return Pairing(pairs)
