"""In-thread find: per-session message search over the transcript index (BE-3).

WHY THIS EXISTS. Find-in-conversation (⌘/Ctrl+F on the desktop transcript)
searches THIS conversation's user/agent messages — the per-session analogue of
the store-wide ``sessions.search``. The documents are the message docs the
transcript index derives (``session/transcript_index.py``): one per
user/assistant message row and per injected row, in journal order. The pipeline
is the same tiered search the store already uses, scoped to one conversation
(design §D3):

1. **casefolded exact substring** over each doc's stored text;
2. a **bounded soft pass** (reusing :class:`SoftSearchIndex` — prefix /
   token-AND / edit-distance <= 2 for 4+ character tokens) when the exact hits
   are fewer than ``PRECISE_HITS_ENOUGH`` — the per-conversation analogue of
   the picker's ``soft_tier_wanted`` floor, sharing its constant;
3. **ranking**: exact tier before soft tier, injected docs after genuine ones
   within a tier (D3: injected content "stays searchable but ranks below
   genuine ones"), and journal order (oldest -> newest) inside that — so Enter
   walks the conversation forward, the order the reader is moving in.

HIDDEN CROSS-SESSION DOCS never enter ranking when ``display.hide_cross_session``
is on (``cross_session.cross_session_hidden``): the UIs stop painting peer
messages, and find must agree — the desktop overlay reveals a hit by jumping to
its row, so a returned hidden hit would be a content leak AND a jump to a row
that is no longer there (design §5.3). Only ``peer_message`` docs can arise
here: tool rows are never docs in this index (``transcript_index``'s own rule),
so the frozen set's send-tool half is structurally absent, not forgotten.

Snippets and ranges are computed at query time from the stored text: the
snippet is cut ± :data:`SNIPPET_CONTEXT` characters around the first match and
carries at most :data:`SNIPPET_MAX_RANGES` match ranges, relative to the
snippet's own start. A doc's text is capped at the index's ``DOC_TEXT_CAP``, so
a match beyond the cap is a stated miss, not a silent one (D8). A soft hit has
no literal occurrence of the query by construction (that is what makes it
soft), so it carries a head window and an empty range list; the client rings
such rows and its highlight walker is query-based (D6).

THE WARM/COLD LADDER IS THE INDEX'S OWN. The serve decision — fresh cache /
background build / first-paint wait / failure cooldown — belongs to
``transcript_index.checkpoints_view``, which is the one answer to "is the index
warm". This module rides that ladder and then reads the resident index, rather
than forking a second ladder that could disagree with the rail about freshness.
A cold conversation therefore answers ``state: "building"`` within the same
~200 ms first-paint budget the rail uses (D3), with whatever the previous scan
holds marked ``partial``; the renderer polls until ``ready`` (D6's "Indexing
conversation…").

WHY A HELD SOFT INDEX, capped and locked. ``SoftSearchIndex`` re-tokenises any
doc whose stored string changed and prunes to the docs it is handed, so ONE
held instance per conversation answers repeated typo queries in single-digit
milliseconds while a fresh instance per query would re-tokenise the whole
conversation on every keystroke (the class's own docstring measures that cost).
The instances are capped at ``_SOFT_SESSIONS`` (the resident index's own bound)
and each carries its own lock, because the pipeline runs in a worker thread and
two find requests can overlap. ``session_search``'s process-wide accelerator is
deliberately NOT reused: its cache is keyed by session and a per-conversation
sync would wipe the picker's warm store.

NOT IN SCOPE — the semantic tier (D3): no embedding backend is configured on
this machine, an API embedder would put a provider round trip in the query
path (the anti-pattern ``session_search`` documents), and the n-gram hash
embedder adds little a bounded edit-distance tier does not. D8 reserves the
schema room for it; if it ships it must run on Enter / a deliberate deep-search
action, never per keystroke.

NOT IN SCOPE — remote/peer conversations (D4): a peer's journal is not on this
disk, so the derivation and this search cannot answer for it; the route's
adapter answers ``state: "unsupported"`` for a peer, mirroring the checkpoints
manifest's degradation. There is no semantic tier here, and no LLM call
anywhere in this module.
"""

from __future__ import annotations

import asyncio
import threading
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from local_operator.cross_session import cross_session_hidden
from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.session import transcript_index
from local_operator.session.search_index import SoftSearchIndex
from local_operator.session.session_search import PRECISE_HITS_ENOUGH
from local_operator.session.transcript_index import MessageDoc

#: Characters of context kept on each side of the first match in a snippet (D3).
SNIPPET_CONTEXT = 60

#: Most match ranges a hit may carry (D3). Bounded because the client paints
#: each one and a long message can contain the query hundreds of times.
SNIPPET_MAX_RANGES = 5

#: The route's own cap (D9: ``limit`` 1..200). Mirrored here so the pipeline's
#: contract is testable without the wire.
FIND_LIMIT_MAX = 200

#: Sessions whose soft-search token cache may stay resident. Mirrors the index
#: module's resident bound: the two caches are populated per conversation and
#: an unbounded either would grow with the number of sessions searched, not
#: with the sessions the user is actually in.
_SOFT_SESSIONS = 4

#: The first-paint wait handed to the index ladder: ``building`` must come back
#: inside ~200 ms even while a 272 MB journal scans (D3). The fast paths — a
#: fresh cache, or a small append — finish well inside this.
FIRST_PAINT_WAIT_S = 0.2

EXACT = "exact"
SOFT = "soft"

#: Wire role vocabulary (D9): the stored docs say ``assistant``, the find wire
#: says ``agent`` — the same translation the store search's callers expect.
_WIRE_ROLE = {"assistant": "agent", "user": "user"}


@dataclass(frozen=True)
class FindHit:
    """One message the query matched, shaped for the wire (D9)."""

    id: str
    role: str
    ts: float
    snippet: str
    ranges: tuple[tuple[int, int], ...]
    tier: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "ts": self.ts,
            "snippet": self.snippet,
            "ranges": [[start, end] for start, end in self.ranges],
            "tier": self.tier,
        }


def _literal_occurrences(
    text: str, needle: str, *, cap: int = SNIPPET_MAX_RANGES
) -> list[tuple[int, int]]:
    """Casefolded occurrences of ``needle``, as offsets into the ORIGINAL text.

    ``needle`` is already casefolded. Non-overlapping matches, oldest first,
    capped at ``cap``: overlapping ranges would paint the same characters twice
    and fill the cap with one dense word.

    Offsets are computed against ``text.casefold()`` — casefold, not lower,
    because that is the comparison the index's docs and the store search both
    use — and then mapped back to the original string. The two agree one-to-one
    in the common case (every folded character is one character); the mapping
    pass exists for the expansions (``ß`` -> ``ss``, ``İ`` -> ``i̇`` and
    friends), where a folded offset can land mid-character and the range must
    cover exactly the characters the match touched.
    """
    folded = text.casefold()
    if len(folded) == len(text):
        found: list[tuple[int, int]] = []
        at = folded.find(needle)
        while at >= 0 and len(found) < cap:
            found.append((at, at + len(needle)))
            at = folded.find(needle, at + len(needle))
        return found

    parts: list[str] = []
    offsets: list[int] = [0]
    for character in text:
        parts.append(character.casefold())
        offsets.append(offsets[-1] + len(parts[-1]))
    folded = "".join(parts)
    found = []
    at = folded.find(needle)
    while at >= 0 and len(found) < cap:
        begin = bisect_right(offsets, at) - 1
        end = bisect_right(offsets, at + len(needle) - 1)
        # Two folded matches can collapse onto one original character (the two
        # halves of a ``ß``); keep the first, skip what its span already covers.
        if not found or begin >= found[-1][1]:
            found.append((begin, end))
        at = folded.find(needle, at + len(needle))
    return found


def _exact_snippet(
    text: str, occurrences: list[tuple[int, int]]
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """The snippet window and its in-window ranges, both snippet-relative.

    The window is ± :data:`SNIPPET_CONTEXT` around the FIRST match (D3); the
    ranges are that match and any sibling that falls inside the window whole,
    at most :data:`SNIPPET_MAX_RANGES`. A range is relative to the snippet so
    the client can mark ``snippet[start:end]`` without the window offset.
    """
    first_start, first_end = occurrences[0]
    window_start = max(0, first_start - SNIPPET_CONTEXT)
    window_end = min(len(text), first_end + SNIPPET_CONTEXT)
    snippet = text[window_start:window_end]
    ranges = tuple(
        (start - window_start, end - window_start)
        for start, end in occurrences
        if start >= window_start and end <= window_end
    )
    return snippet, ranges[:SNIPPET_MAX_RANGES]


def _snippet_for(
    doc: MessageDoc, occurrences: list[tuple[int, int]]
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """The snippet a hit carries: windowed for an exact hit, head for a soft one.

    A soft hit has no literal occurrence of the query (see the module
    docstring), so the most useful honest window is the head of the message;
    its ranges stay empty — a range whose text does not equal the query would
    be a second, weaker claim about the match.
    """
    if occurrences:
        return _exact_snippet(doc.text, occurrences)
    return doc.text[: SNIPPET_CONTEXT * 2], ()


def search_messages(
    docs: Sequence[MessageDoc],
    query: str,
    limit: int,
    *,
    soft_search: Callable[[dict[str, str]], set[str]] | None = None,
) -> tuple[list[FindHit], bool]:
    """Rank this conversation's matches for ``query``; return hits and truncation.

    The pure core of the pipeline: no I/O, no clocks, no caches it does not get
    handed. ``soft_search`` is the bounded soft pass — called ONLY when the
    exact hits fall below :data:`PRECISE_HITS_ENOUGH`, with a doc-id -> text
    mapping, returning the ids that softly matched. The caller supplies the
    cached, locked instance (see :func:`find_view`); when omitted, the
    stateless one-shot form runs, which re-tokenises every doc per call.
    """
    needle = query.strip().casefold()
    if not needle or limit <= 0:
        return [], False

    occurrences: dict[str, list[tuple[int, int]]] = {}
    for doc in docs:
        found = _literal_occurrences(doc.text, needle)
        if found:
            occurrences[doc.id] = found

    soft_ids: set[str] = set()
    if len(occurrences) < PRECISE_HITS_ENOUGH:
        digests = {doc.id: doc.text for doc in docs}
        if soft_search is not None:
            soft_ids = set(soft_search(digests))
        else:
            soft_ids = SoftSearchIndex().search(digests, query)

    # Ranking is a property of ``(tier, injected, seq)`` alone, so the entries
    # are keyed FIRST and only the survivors get a snippet: building a snippet
    # for every match wastes the whole point of ``limit`` (a common word in a
    # long conversation matches thousands of docs, each snippet ~a message long).
    # ``seq`` is unique per doc, so the order is total and deterministic.
    ranked: list[tuple[tuple[int, int, int], MessageDoc, list[tuple[int, int]]]] = []
    for doc in docs:
        found = occurrences.get(doc.id)
        if found is not None:
            tier, spans = EXACT, found
        elif doc.id in soft_ids:
            tier, spans = SOFT, []
        else:
            continue
        ranked.append(((0 if tier == EXACT else 1, 1 if doc.injected else 0, doc.seq), doc, spans))

    ranked.sort(key=lambda entry: entry[0])
    truncated = len(ranked) > limit
    hits: list[FindHit] = []
    for _key, doc, spans in ranked[:limit]:
        snippet, ranges = _snippet_for(doc, spans)
        hits.append(
            FindHit(
                id=doc.id,
                role=_WIRE_ROLE.get(doc.role, "user"),
                ts=doc.ts,
                snippet=snippet,
                ranges=ranges,
                tier=EXACT if spans else SOFT,
            )
        )
    return hits, truncated


#: Held soft-search state per conversation: the token-cache instance and the
#: lock that serialises its in-place syncs across worker threads. Guarded by
#: ``_SOFT_GUARD`` for membership only; an entry a thread holds cannot be
#: invalidated by an eviction, because the holder owns a reference.
_SOFT_LRU: "OrderedDict[tuple[str, str], tuple[SoftSearchIndex, threading.Lock]]" = OrderedDict()
_SOFT_GUARD = threading.Lock()


def _soft_for(config_dir: str | Path, session_id: str) -> tuple[SoftSearchIndex, threading.Lock]:
    """The held soft index for one conversation, creating it on first use.

    LRU-capped at :data:`_SOFT_SESSIONS`; eviction only drops the module's
    reference, never a live caller's.
    """
    key = (str(config_dir), session_id)
    with _SOFT_GUARD:
        entry = _SOFT_LRU.get(key)
        if entry is None:
            entry = (SoftSearchIndex(), threading.Lock())
            _SOFT_LRU[key] = entry
        else:
            _SOFT_LRU.move_to_end(key)
        while len(_SOFT_LRU) > _SOFT_SESSIONS:
            _SOFT_LRU.popitem(last=False)
        return entry


def _build_hits(
    config_dir: str | Path,
    session_id: str,
    docs: Sequence[MessageDoc],
    query: str,
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    """The worker-thread body: run the pipeline under the session's soft lock."""

    def soft(digests: dict[str, str]) -> set[str]:
        index, lock = _soft_for(config_dir, session_id)
        with lock:
            return index.search(digests, query)

    hits, truncated = search_messages(docs, query, limit, soft_search=soft)
    return [hit.to_payload() for hit in hits], truncated


def _response(
    query: str, state: str, *, partial: bool, hits: list[dict[str, Any]], truncated: bool
) -> dict[str, Any]:
    """The D9 answer shape, in one place so every path returns all five keys."""
    return {
        "query": query,
        "state": state,
        "partial": partial,
        "hits": hits,
        "truncated": truncated,
    }


def _visible_docs(docs: Sequence[MessageDoc]) -> Sequence[MessageDoc]:
    """The docs find may return: hidden cross-session rows dropped when asked.

    ``display.hide_cross_session`` is a view filter for the human reader:
    with it on, no surface paints a peer message, so find must not return one
    — the desktop overlay reveals a hit by jumping to the hit's id, and a
    reveal aimed at a row that no longer renders is both a content leak and a
    broken jump (design §5.3). The filter sits at the docs-selection seam of
    :func:`find_view` — applied to BOTH of its paths (the fresh ``ready``
    index and the ``building`` previous scan) so the two can never disagree
    about what find may return.

    The filtered set is exactly the frozen one, and only its ``peer_message``
    half can arise here: tool rows are never docs in this index
    (``transcript_index``'s own rule — indexing machine output would make path
    queries match everything), so a ``send`` tool row cannot reach find and
    needs no gate. When the flag is off the SAME sequence is returned — no
    copy, today's pipeline exactly — mirroring the surfaces' default-off
    contract (design §5.4).
    """
    if not cross_session_hidden():
        return docs
    return [doc for doc in docs if doc.custom_type != PEER_MESSAGE_MESSAGE_TYPE]


async def find_view(
    config_dir: str | Path,
    session_id: str,
    *,
    query: str,
    limit: int,
    wait_s: float = FIRST_PAINT_WAIT_S,
) -> dict[str, Any]:
    """The ``sessions.find`` answer for one LOCAL session (D9).

    Rides :func:`transcript_index.checkpoints_view` for the warm/cold decision
    (fresh read / background build / first-paint wait / failure cooldown), then
    queries whatever that decision leaves resident:

    * ``ready`` — the fresh index's docs, ranked; ``partial`` False.
    * ``building`` — the previous scan's docs if one exists, marked
      ``partial`` True: the tail may still be missing while the refresh runs.
    * ``error`` — the refresh failed inside its cooldown; an empty refuse.
    """
    manifest = await transcript_index.checkpoints_view(config_dir, session_id, wait_s=wait_s)
    state = str(manifest.get("index", {}).get("state", "error"))

    if state == "building":
        previous = await asyncio.to_thread(transcript_index.read_index, config_dir, session_id)
        docs = _visible_docs(previous.messages if previous is not None else [])
        hits, truncated = await asyncio.to_thread(
            _build_hits, config_dir, session_id, docs, query, limit
        )
        return _response(query, "building", partial=True, hits=hits, truncated=truncated)

    if state != "ready":
        # ``error``, and the reserved states the manifest does not emit today.
        return _response(query, "error", partial=False, hits=[], truncated=False)

    index = transcript_index.resident(config_dir, session_id)
    if index is None:
        # ``ready`` with no journal is the draft/never-written answer (empty),
        # and this fallback covers any race between the ladder and the read.
        index = await asyncio.to_thread(transcript_index.read_index, config_dir, session_id)
    docs = _visible_docs(index.messages if index is not None else [])
    hits, truncated = await asyncio.to_thread(
        _build_hits, config_dir, session_id, docs, query, limit
    )
    return _response(query, "ready", partial=False, hits=hits, truncated=truncated)


def _reset_for_tests() -> None:
    """Drop the module's held soft-search caches (tests; never production)."""
    with _SOFT_GUARD:
        _SOFT_LRU.clear()
