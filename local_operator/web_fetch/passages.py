"""Query-focused passage selection for oversized ``web_fetch`` previews.

Why this exists
---------------
A page too long to show whole used to come back as head + tail only: the
first and last ~4 KB of an 8 KB budget. For a research question the answer is
usually in the MIDDLE (a parameter table, a definitions list, one RFC
section), so the model's next move was a ``read spill://…?q=`` round trip to
find it — measured on 490 recent fetches, 47% were oversized and 59% of those
were followed by exactly that re-read. Each round trip is a full model turn
(~30 s). When the caller says what it is looking for (``query``), this module
picks the passages that best match so the first preview already holds them.

Constraints (all deliberate)
----------------------------
- **Pure, deterministic, offline.** No model call, no network, no new
  dependency: this runs inside a tool call on every oversized fetch, and a
  ranker that can fail, stall or vary would make the preview untrustworthy.
  Ties break on document order, so the same page and query always yield the
  same preview — which also keeps provider prompt caches warm.
- **Linear time.** Pages reach the 4 MB spill-entry cap. Every line is
  scanned once to chunk it and every chunk is tokenised once; scoring touches
  only the (few) query terms per chunk. No pairwise or sort-the-world step
  over anything larger than the chunk list.
- **Line ranges are the stored copy's.** The ranker takes the LINES the spill
  store will serve (``SpillStore.read_lines`` splits with ``str.splitlines``),
  never the pre-spill string, so a label ``lines a-b`` read back through
  ``read spill://<h> range=a-b`` returns exactly the text shown. Span-marker
  stripping and the 4 MB head+tail clip both change coordinates; ranking the
  stored copy makes both irrelevant.
- **Lexical, not semantic.** BM25 term weighting with bonuses for what
  research questions are made of — exact identifiers (``max_connections``,
  ``SETTINGS_INITIAL_WINDOW_SIZE``), numbers, and adjacent query words as a
  phrase. It cannot paraphrase; the model writes the query and can name the
  identifier it wants, and the spill handle is still there when it misses.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

#: Words that carry no retrieval signal in a question. Kept small on purpose:
#: a long list starts eating meaningful words ("default", "first", "value")
#: that are exactly what a research query is about.
_STOPWORDS = frozenset("""a an and are as at be by can do does for from has have how i in is it its
    me my of on or that the their there these this to was what when where which
    who why will with you your page find show tell need want about""".split())

#: One token: a run of word characters, optionally joined by ``.``/``-``/``:``
#: into a compound (``gc.auto``, ``utf-8``, ``h2c``). Compounds are also split
#: into their parts (see :func:`_terms`), so ``gc.auto`` matches both a query
#: for ``gc.auto`` and one for ``auto``.
_TOKEN_RE = re.compile(r"[0-9a-z_]+(?:[.\-:][0-9a-z_]+)*")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")
_DIGIT_RE = re.compile(r"\d")
#: Rendered docs decorate headings with anchor links (``[#](#guc-…)``,
#: ``[15.5.20.](#section-15.5.20)``). In a passage label they are noise the
#: model pays for on every passage, so links collapse to their text and empty
#: or ``#`` anchors disappear.
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")

#: Query words that mean "the answer is a number" — a chunk with a digit in it
#: gets a small nudge for such a query, because "what is the default X" is
#: answered by the paragraph that states a value, not the one that defines X.
_NUMERIC_INTENT = frozenset(
    """default defaults value limit limits max maximum min minimum size length
    number count code status version port timeout many much long""".split()
)

#: Chunk sizing. A prose chunk accumulates paragraphs of one section up to
#: ``_CHUNK_MIN_CHARS`` so a definition line ("``max_connections`` (integer)")
#: lands in the same chunk as the paragraph that explains it; a block (table,
#: list, code fence) is kept whole up to ``_CHUNK_MAX_CHARS`` because half a
#: table is unreadable. Lines, not chars, bound the work of a pathological
#: page of one-character lines.
_CHUNK_MIN_CHARS = 400
_CHUNK_MAX_CHARS = 2400
_CHUNK_MAX_LINES = 80

#: BM25 parameters (the textbook defaults; nothing here is tuned to a page).
_K1 = 1.2
_B = 0.75

#: Bonuses, in the same units as a BM25 term contribution (~1-6 for a rare
#: term). An exact identifier is the strongest single signal a research query
#: can carry, so it outweighs several common-word hits.
_IDENTIFIER_BONUS = 4.0
_NUMBER_BONUS = 2.0
_PHRASE_BONUS = 1.5
_NUMERIC_INTENT_BONUS = 0.75
_HEADING_WEIGHT = 2  # heading words count this many times for their chunk


@dataclass(frozen=True)
class Passage:
    """One selected passage: 1-based inclusive line range plus its text.

    ``text`` is exactly ``"\\n".join(lines[start - 1 : end])`` of the lines the
    ranker was given — the property ``read range=start-end`` relies on. The one
    exception is ``excerpt=True``: a single line too long to show whole, whose
    text is a character window of line ``start``.
    """

    start: int
    end: int
    text: str
    heading: str
    score: float
    excerpt: bool = False


@dataclass(frozen=True)
class Selection:
    """The ranker's answer: chosen passages (document order) and the rest."""

    passages: tuple[Passage, ...]
    #: Matching chunks NOT shown (budget or overlap), so the preview can say
    #: how much more there is to search for.
    more: int
    #: A ``?q=`` regex for the most specific query terms, for that hint.
    search_hint: str


@dataclass
class _Chunk:
    start: int  # 1-based, inclusive
    end: int
    heading: str
    chars: int


def _normalise(token: str) -> str:
    """Fold a plural to its stem, cheaply: ``connections`` → ``connection``.

    Deliberately not a stemmer — a real one is a dependency and over-merges
    identifiers. Only a trailing ``s`` on a plain word is dropped; compounds
    and identifiers are left exactly as written.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss") and token.isalpha():
        return token[:-1]
    return token


def _terms(text: str) -> list[str]:
    """Lowercased terms of ``text``: each compound token plus its parts."""
    out: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        out.append(_normalise(token))
        if not token.isalnum():
            for part in re.split(r"[._\-:]", token):
                if part and part != token:
                    out.append(_normalise(part))
    return out


def _is_identifier(raw: str) -> bool:
    """A query token that names a thing rather than describing it."""
    return bool(re.search(r"[_.:]", raw)) or (
        any(c.isdigit() for c in raw) and any(c.isalpha() for c in raw)
    )


def _clean_heading(text: str) -> str:
    """A heading as a short label: link text only, anchors and escapes gone."""
    text = _LINK_RE.sub(lambda m: "" if m.group(1) in ("", "#", "¶") else m.group(1), text)
    return " ".join(text.replace("\\", "").split())[:120]


def _block_kind(line: str) -> str:
    stripped = line.lstrip()
    if stripped.startswith("|"):
        return "table"
    if _LIST_RE.match(line):
        return "list"
    return "prose"


def chunk_lines(lines: list[str]) -> list[_Chunk]:
    """Split ``lines`` into retrieval chunks in one pass.

    Boundaries: a heading always starts a new chunk (and becomes the context
    label of what follows); a blank line ends a paragraph, and a paragraph ends
    the chunk once it has ``_CHUNK_MIN_CHARS`` of prose; a table/list/fenced
    block is kept whole up to the cap. Blank lines inside a fence do not split.
    """
    chunks: list[_Chunk] = []
    heading = ""
    cur_start = 0  # 0 = no open chunk
    cur_end = 0
    cur_chars = 0
    cur_kind = "prose"
    in_fence = False

    def close() -> None:
        nonlocal cur_start, cur_chars
        if cur_start:
            chunks.append(_Chunk(cur_start, cur_end, heading, cur_chars))
        cur_start = 0
        cur_chars = 0

    for number, line in enumerate(lines, start=1):
        if _FENCE_RE.match(line):
            if not in_fence:
                if cur_start and cur_kind != "code":
                    close()
                cur_kind = "code"
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING_RE.match(line)
            if match:
                close()
                heading = _clean_heading(match.group(2))
                # The heading line is its own chunk so a query naming the
                # section ("Misdirected Request") can land on it directly.
                chunks.append(_Chunk(number, number, heading, len(line)))
                continue
            if not line.strip():
                # A paragraph ended. Close only when the chunk is big enough to
                # stand alone; a short one keeps accumulating so a term and its
                # explanation stay together.
                if cur_start and cur_chars >= _CHUNK_MIN_CHARS:
                    close()
                continue
            kind = _block_kind(line)
            if cur_start and kind != cur_kind:
                # Entering or leaving a table/list: a block is its own chunk, so
                # it is ranked (and shown) whole rather than glued to the prose
                # beside it. Prose→prose never reaches here (same kind).
                close()
            cur_kind = kind
        if not cur_start:
            cur_start = number
        cur_end = number
        cur_chars += len(line) + 1
        if cur_chars >= _CHUNK_MAX_CHARS or cur_end - cur_start + 1 >= _CHUNK_MAX_LINES:
            close()
    close()
    return chunks


@dataclass(frozen=True)
class _Query:
    terms: tuple[str, ...]  # unique content terms, query order
    identifiers: tuple[str, ...]  # lowercased exact compounds to look for
    numbers: tuple[str, ...]
    bigrams: tuple[tuple[str, str], ...]
    numeric_intent: bool


def _parse_query(query: str) -> _Query | None:
    raw_tokens = _TOKEN_RE.findall(query.lower())
    content = [t for t in raw_tokens if t not in _STOPWORDS]
    terms: list[str] = []
    for term in _terms(" ".join(content)):
        if term not in _STOPWORDS and term not in terms:
            terms.append(term)
    if not terms:
        return None
    identifiers = tuple(dict.fromkeys(t for t in content if _is_identifier(t)))
    numbers = tuple(dict.fromkeys(t for t in content if t.isdigit()))
    normalised = [_normalise(t) for t in content]
    bigrams = tuple(dict.fromkeys(zip(normalised, normalised[1:])))
    return _Query(
        terms=tuple(terms),
        identifiers=identifiers,
        numbers=numbers,
        bigrams=bigrams,
        numeric_intent=any(t in _NUMERIC_INTENT for t in content),
    )


def _score_chunks(
    lines: list[str], chunks: list[_Chunk], query: _Query
) -> tuple[list[tuple[float, int]], dict[str, float]]:
    """``(score, chunk_index)`` for every matching chunk, plus each term's IDF."""
    wanted = set(query.terms)
    bigram_firsts = {a for a, _ in query.bigrams}
    per_chunk: list[tuple[Counter[str], int, str, set[tuple[str, str]]]] = []
    df: Counter[str] = Counter()
    total_len = 0
    for chunk in chunks:
        # markdownify escapes ``_`` and ``*`` (``SQLITE\\_MAX\\_COLUMN``); score
        # the unescaped form so an identifier query matches the identifier.
        # Scoring only — the passage shown is still the stored text verbatim.
        text = "\n".join(lines[chunk.start - 1 : chunk.end])
        text = text.replace("\\_", "_").replace("\\*", "*")
        terms = _terms(text)
        heading_terms = _terms(chunk.heading)
        counts = Counter(t for t in terms if t in wanted)
        for t in heading_terms:
            if t in wanted:
                counts[t] += _HEADING_WEIGHT
        # Adjacent-pair hits for the phrase bonus, computed in the same pass
        # and only for pairs whose first word is a query word (cheap filter).
        pairs = {(a, b) for a, b in zip(terms, terms[1:]) if a in bigram_firsts} & set(
            query.bigrams
        )
        per_chunk.append((counts, len(terms) + 1, text.lower(), pairs))
        df.update(counts.keys())
        total_len += len(terms) + 1
    n = len(chunks) or 1
    avg_len = total_len / n
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in query.terms}

    scored: list[tuple[float, int]] = []
    for index, (counts, length, lower, pairs) in enumerate(per_chunk):
        if not counts:
            continue
        score = 0.0
        for term, tf in counts.items():
            score += idf[term] * tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * length / avg_len))
        for ident in query.identifiers:
            if ident in lower:
                score += _IDENTIFIER_BONUS
        for number in query.numbers:
            if re.search(rf"(?<!\d){re.escape(number)}(?!\d)", lower):
                score += _NUMBER_BONUS
        score += _PHRASE_BONUS * len(pairs)
        if query.numeric_intent and _DIGIT_RE.search(lower):
            score += _NUMERIC_INTENT_BONUS
        scored.append((score, index))
    return scored, idf


def _search_hint(query: _Query, idf: dict[str, float]) -> str:
    """A ``?q=`` regex naming the most specific query words.

    Identifiers first (they are what the model is hunting), then the rarest
    remaining words. Case-insensitive because pages capitalise freely and the
    spill search is case-sensitive by default.
    """
    picks = list(query.identifiers[:2])
    # ``sorted`` is stable, so equal-IDF terms keep query order (determinism).
    for term in sorted(query.terms, key=lambda t: -idf.get(t, 0.0)):
        if len(picks) >= 3:
            break
        if term not in picks and not any(term in p for p in picks):
            picks.append(term)
    return "(?i)" + "|".join(re.escape(p) for p in picks)


def _trim_to(lines: list[str], start: int, end: int, budget: int) -> tuple[int, int]:
    """Shrink ``start..end`` from the end, by whole lines, to fit ``budget``."""
    size = 0
    last = start - 1
    for number in range(start, end + 1):
        size += len(lines[number - 1]) + 1
        if size > budget and last >= start:
            break
        last = number
    return start, last


def select_passages(
    lines: list[str],
    query: str,
    *,
    budget: int,
    skip_through: int = 0,
    max_passages: int = 6,
    passage_cap: int = 1800,
) -> Selection | None:
    """Pick the passages of ``lines`` that best answer ``query``.

    ``budget`` bounds the summed passage TEXT (the caller adds labels and
    framing and must reserve room for them). ``skip_through`` is the last line
    the caller already shows (its lead): passages never repeat those lines.
    Returns ``None`` when no chunk matches any query term — the caller then
    falls back to head/tail and says so.
    """
    parsed = _parse_query(query)
    if parsed is None or not lines:
        return None
    chunks = chunk_lines(lines)
    scored, idf = _score_chunks(lines, chunks, parsed)
    if not scored:
        return None
    # Highest score first; ties go to the earlier chunk so output is stable.
    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    chosen: list[Passage] = []
    used = 0
    shown_lines: set[int] = set()
    for score, index in scored:
        if len(chosen) >= max_passages or budget - used < 120:
            break
        chunk = chunks[index]
        start = max(chunk.start, skip_through + 1)
        end = chunk.end
        if chunk.start == chunk.end and lines[chunk.start - 1].lstrip().startswith("#"):
            # A heading hit: what the reader wants is the section under it, so
            # extend into the following chunk(s) of the same section.
            follow = index + 1
            while follow < len(chunks) and chunks[follow].heading == chunk.heading:
                end = chunks[follow].end
                if end - chunk.start > 12:
                    break
                follow += 1
        # A true interval test, not an endpoint test: a heading hit extended
        # over its section can CONTAIN a passage already chosen, and showing
        # those lines twice spends budget on a duplicate. Bounded by the chunk
        # size (≤ _CHUNK_MAX_LINES plus the heading extension), so still linear.
        if start > end or any(n in shown_lines for n in range(start, end + 1)):
            continue
        room = min(passage_cap, budget - used)
        if len(lines[start - 1]) > room:
            # One enormous line (minified text): show a window around the first
            # query hit. Labelled as an excerpt — a range read returns the whole
            # line, which is more than shown, never different from it.
            line = lines[start - 1]
            lower = line.lower()
            hit = min((p for p in (lower.find(t) for t in parsed.terms) if p >= 0), default=0)
            lo = max(0, hit - room // 3)
            text = line[lo : lo + room]
            chosen.append(Passage(start, start, text, chunk.heading, score, excerpt=True))
            used += len(text)
            shown_lines.add(start)
            continue
        start, end = _trim_to(lines, start, end, room)
        text = "\n".join(lines[start - 1 : end])
        if not text.strip():
            continue
        chosen.append(Passage(start, end, text, chunk.heading, score))
        used += len(text)
        shown_lines.update(range(start, end + 1))

    if not chosen:
        return None
    more = sum(
        1
        for _score, index in scored
        if not any(chunks[index].start <= p.end and p.start <= chunks[index].end for p in chosen)
    )
    chosen.sort(key=lambda p: p.start)
    return Selection(tuple(chosen), more, _search_hint(parsed, idf))
