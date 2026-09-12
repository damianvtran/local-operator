"""One definition of "which past conversations match this query, best first".

Three surfaces ask a user's question back to the store — the TUI's ``/resume``
picker, the phone daemon's session search (``mobile.daemon``), and the desktop
catalogue's chat search (``GET /v1/desktop/sessions/search``) — and each had
grown its own copy of the rules. They had already drifted: the phone matched on
name/id and an EXACT body substring only, so a typo, a prefix or a word-order
query that the picker resolves found nothing there, and neither the phone nor
the desktop ranked what it found, so the best match sat wherever recency put
it. A surface that searches a store differently from its siblings is a bug
report waiting on the user who tries the same query twice.

So the mechanics live here, once:

* **What is admitted.** ``name`` (through :func:`local_operator.resume.fork_haystack`,
  so a row visibly tagged ``[fork]`` is findable by that tag however it is
  spelled), the 12-hex id, an exact case-insensitive substring of the
  conversation body, and — when the query is not already precisely answered — a
  bounded SOFT match (prefix, order-independent token-AND, edit distance <= 2
  on tokens of 4+ characters) over the same digest index.
* **What order.** :data:`RANK_NAME` > :data:`RANK_ID` > :data:`RANK_BODY` >
  :data:`RANK_SOFT`, with recency as the tie-break within a tier. Ranking is a
  pure function of ``(query, row, digests)`` — no memory of previous keystrokes
  — so the same query renders identically however the user reached it, which is
  the property the picker needed before it could re-home a cursor onto the top
  match.

The digest index itself is ``session/search_index.py``: this module composes
it, it does not reimplement it. Callers that hold their own rows and digests (a
keystroke-driven picker) use the pure helpers; callers that answer a one-shot
query over the store (the phone, the desktop route) use :func:`search_store`,
which does the row scan and the index build in one place.

Nothing here imports a UI framework: the daemon must not pull Textual in for a
filter, and the server must not pull it in at all.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path

from local_operator.resume import SessionRow, fork_haystack, recent_session_rows
from local_operator.session.search_index import (
    SoftSearchIndex,
    build_index,
    search_digests,
)

#: Relevance tiers, best (lowest) first, used to sort a FILTERED subset when a
#: query is active. A tier is a property of ``(query, row)`` alone — it does not
#: depend on the previous query or on the order rows arrived — which is what
#: lets ranking coexist with the picker's "no reorder under the cursor"
#: invariant: the order changes only when the query changes, and a query change
#: already re-homes the cursor to the top match.
RANK_NAME = 0  # exact substring in the visible name — the strongest signal
RANK_ID = 1  # exact substring in the id
RANK_BODY = 2  # exact substring in the body/past-name digest
RANK_SOFT = 3  # soft (prefix / token-AND / edit-distance) match only

#: Name/id matches at which the bounded soft tier is skipped entirely (see
#: :func:`soft_tier_wanted`). Three, not one: a single exact hit on a name is as
#: often incidental as deliberate — ``spit`` matches "De\ *spit*\ e" — and
#: treating it as a real answer hid every genuinely intended match behind it.
#: Measured over 517 vocabulary-drawn typos, this floor loses no rows against
#: running the tier on every keystroke while leaving the cursor exactly as
#: stable.
PRECISE_HITS_ENOUGH = 3


def name_or_id_hit(row: SessionRow, needle: str) -> bool:
    """Whether ``needle`` (already lowercased and stripped) is a PRECISE hit.

    Through :func:`fork_haystack` rather than ``row.name``, so a row admitted on
    its visible ``[fork]`` tag is a precise hit on the name tier — the tag is on
    screen, so it has to be searchable, and it must not sort as a fuzzy body
    match below every incidental digest hit.
    """
    return needle in fork_haystack(row).lower() or needle in row.id.lower()


def rank_tier(row: SessionRow, needle: str, exact_body: AbstractSet[str] | None = None) -> int:
    """The :data:`RANK_*` tier ``row`` matches ``needle`` in.

    A row that matches nothing here is still rankable: it was admitted by the
    SOFT set (it is in the already-filtered rows yet matched neither name, id,
    nor exact body), which is :data:`RANK_SOFT`. That is why this never returns
    "no match" — the caller's admission decision and this ordering decision are
    deliberately separate, so a soft hit surfaces the row exactly as an exact
    body hit does and then sorts below every exact tier.
    """
    if needle in fork_haystack(row).lower():
        return RANK_NAME
    if needle in row.id.lower():
        return RANK_ID
    if exact_body and row.id in exact_body:
        return RANK_BODY
    return RANK_SOFT


def matched_in_body(row: SessionRow, query: str, body_matches: AbstractSet[str] | None) -> bool:
    """True when ``row`` is on screen only because its CONVERSATION matched.

    Drives the body-match marker on every surface that has one: a row whose
    visible name already contains the query needs no explanation, and one that
    does not would otherwise read as the filter returning something arbitrary —
    worse than no marker at all, because it makes the whole result set look
    untrustworthy.

    ``body_matches`` is therefore the ADMITTED set (exact-body OR soft), not
    just the exact one: a row surfaced only because a PAST name or a typo
    matched is just as much "found on something other than the visible name".
    """
    needle = query.strip().lower()
    if not needle:
        return False
    if name_or_id_hit(row, needle):
        return False
    return row.id in (body_matches or frozenset())


def filter_rows(
    rows: Sequence[SessionRow],
    query: str,
    body_matches: AbstractSet[str] | None = None,
) -> list[SessionRow]:
    """Rows whose name or id contains ``query``, or whose id is in ``body_matches``.

    A pure MEMBERSHIP filter: it decides which rows are shown and preserves the
    order it was handed, so on its own it never moves a row under a cursor.
    Relevance ORDERING lives in :func:`rank_rows`, which a surface applies only
    when a query is active.

    The name/id test stays exact substring even though soft matching exists:
    those fields are a sentence the user wrote and a hex id, where an exact
    match is what a precise query expects. Soft hits arrive through
    ``body_matches`` (the caller folds the soft set in), so they surface the row
    without making the name test fuzzy.
    """
    needle = query.strip().lower()
    if not needle:
        return list(rows)
    matched = body_matches or frozenset()
    return [row for row in rows if name_or_id_hit(row, needle) or row.id in matched]


def rank_rows(
    rows: Sequence[SessionRow],
    query: str,
    body_matches: AbstractSet[str] | None = None,
) -> list[SessionRow]:
    """``rows`` ordered by relevance to ``query``, recency as the tie-break.

    * **Empty query** -> ``rows`` unchanged (recency order, newest first). A
      fixed query likewise never reorders: the key is a pure function of
      ``(query, row)``, so repeated repaints and resizes produce byte-for-byte
      the same order.
    * **Non-empty query** -> one deterministic ordering: the tier the row
      matched in (name > id > body > soft), with recency (newest first) as the
      stable tie-break WITHIN every tier. ``sorted`` is stable, so passing rows
      already in recency order makes the tie-break free.

    ``body_matches`` is the EXACT-body match set (:func:`rank_tier`), not the
    admitted one: a row that matched none of name, id or exact body was admitted
    by the soft set and takes the soft tier, which needs no membership check.
    """
    needle = query.strip().lower()
    if not needle:
        return list(rows)
    body = body_matches or frozenset()
    return sorted(rows, key=lambda row: rank_tier(row, needle, body))


def soft_tier_wanted(rows: Sequence[SessionRow], query: str) -> bool:
    """Whether the bounded soft tier should run for ``query``.

    A pure function of ``(query, rows)``: the tier runs unless the query matched
    a session's NAME or ID at least :data:`PRECISE_HITS_ENOUGH` times. No run
    history, no latch, no memory of previous keystrokes — that purity is what
    keeps the same visible query rendering identically however the user reached
    it.

    **Why name/id and not "any exact hit".** Gating on an empty exact result
    looks equivalent and silently destroys typo search. The exact tier also
    admits BODY substring hits, and on a real store almost every typed token
    appears incidentally in some conversation: ``plin`` has 8 body hits,
    ``gren`` has 1. One incidental hit anywhere in the store then silenced the
    tier for the whole query, so the typo it exists to rescue could not be
    found. Measured on typos drawn from the store's own vocabulary, that gate
    lost the target row outright on 11 of 763 queries and shed 100+ rows on 14.

    A name or id match is different in kind: those fields are a sentence the
    user wrote and a hex id they can copy, and an exact substring in either is a
    deliberate, precise hit. A body substring is not that signal — it is as
    likely to be the word appearing in passing inside an unrelated
    conversation.

    **Why a floor rather than emptiness.** ONE precise hit is not yet a useful
    answer and can easily be incidental (see :data:`PRECISE_HITS_ENOUGH`), so
    below the floor the extra recall is worth more than the precision; at or
    above it the user has a real answer and fuzzy additions would only dilute
    it.
    """
    needle = query.strip().lower()
    if not needle:
        return False
    precise = 0
    for row in rows:
        if name_or_id_hit(row, needle):
            precise += 1
            if precise >= PRECISE_HITS_ENOUGH:
                return False
    return True


@dataclass(frozen=True)
class SessionMatch:
    """One admitted row: its row, its tier, and whether its BODY is why.

    ``body_match`` is True only when the name and id did NOT explain the match,
    so a caller can say why a row surfaced without recomputing the comparison
    (and without disagreeing with the filter about what "matched on the name"
    means).
    """

    row: SessionRow
    rank: int
    body_match: bool


def rank_matches(
    rows: Sequence[SessionRow],
    query: str,
    body_matches: AbstractSet[str] | None = None,
    admitted: AbstractSet[str] | None = None,
) -> list[SessionMatch]:
    """:func:`rank_rows` with each row's tier and reason attached.

    Two sets, because they answer different questions and the picker already
    keeps them apart: ``body_matches`` (EXACT body) is what a row's TIER is
    decided against — an exact substring in the conversation is a stronger
    signal than a typo — while ``admitted`` (exact OR soft) is what the
    ``body_match`` MARKER is decided against. Marking from the exact set alone
    gets a soft-only hit silently wrong: a row found through a typo, a prefix
    or a reordered query has no visible reason for being on screen, which is
    the one case the marker exists for.

    ``admitted`` defaults to ``body_matches`` so a caller with no soft tier
    gets exactly the old behaviour.
    """
    needle = query.strip().lower()
    if not needle:
        return [SessionMatch(row, RANK_NAME, False) for row in rows]
    ordered = rank_rows(rows, needle, body_matches)
    explained = admitted if admitted is not None else body_matches
    return [
        SessionMatch(
            row, rank_tier(row, needle, body_matches), matched_in_body(row, needle, explained)
        )
        for row in ordered
    ]


#: The process-wide soft-search accelerator, built on first use.
#:
#: The phone daemon and the desktop route answer ONE-SHOT queries, so they have
#: no object of their own to hold a warm token cache — and the stateless
#: :func:`local_operator.session.search_index.soft_search_digests` re-tokenises
#: the whole store on every request (measured 21 ms at 235 sessions / ~0.7 MB of
#: digests, against ~1-8 ms warm here). :class:`SoftSearchIndex` re-syncs
#: against whatever digests it is handed, and prunes to them, so one instance
#: tracks the live store rather than growing across it.
_SHARED_SOFT = SoftSearchIndex()

#: Serializes the shared accelerator AND the exact-search memo it runs beside.
#:
#: Both are process-wide caches mutated in place, and the server calls this
#: module from worker threads (``asyncio.to_thread``). Two concurrent searches
#: through the shared soft index would interleave its token-cache sync, and
#: ``search_index._lowered``'s one-entry memo has a genuine check-then-act race:
#: thread A keys on its own digests, thread B replaces the memo, and A returns
#: B's corpus — a search answering with the WRONG rows, which is the one failure
#: this whole path exists to prevent. A search costs single-digit milliseconds,
#: so serializing them is cheaper than making either memo thread-safe, and the
#: lock is only taken on the shared path: the picker holds its own index and
#: runs on a single thread.
_SHARED_LOCK = threading.Lock()


def _search_shared(
    digests: dict[str, str],
    query: str,
    soft: SoftSearchIndex | None,
    want_soft: bool,
) -> tuple[set[str], set[str]]:
    """``(exact_body_hits, soft_hits)`` for ``query``, either index considered.

    Returns the exact hit set separately from the soft one because they answer
    different questions downstream: their union is what admits a row, while the
    exact set alone decides its tier. ``want_soft`` comes from
    :func:`soft_tier_wanted` (the caller decides, because the gate needs the
    rows), and when it is False the second set is empty and the expensive soft
    search is never run.
    """
    if soft is not None:
        # The caller owns this index (the picker keeps one for its whole life)
        # and is single-threaded: no lock, no shared cache.
        exact = search_digests(digests, query)
        return exact, soft.search(digests, query) if want_soft else set()
    with _SHARED_LOCK:
        exact = search_digests(digests, query)
        return exact, _SHARED_SOFT.search(digests, query) if want_soft else set()


def search_rows(
    rows: Sequence[SessionRow],
    query: str,
    *,
    digests: dict[str, str] | None = None,
    soft: SoftSearchIndex | None = None,
    limit: int | None = None,
) -> list[SessionMatch]:
    """Admitted rows for ``query``, best first — the whole mechanic in one call.

    ``digests`` is the caller's body index; absent (a caller with no index — a
    test, an embedder, a store that failed to digest) the search degrades to
    name and id, which is exactly what it did before body search existed rather
    than failing.

    ``soft`` is the caller's own accelerator when it has one; absent, the
    process-wide one is used under :data:`_SHARED_LOCK`.
    """
    needle = query.strip().lower()
    if not needle:
        # An empty query is not a search: every row is the answer, in recency
        # order, and nothing is "matched on the body".
        rows = list(rows)
        return [SessionMatch(row, RANK_NAME, False) for row in rows[:limit]]

    exact_body: set[str] = set()
    admitted: set[str] = set()
    if digests:
        exact_body, soft_hits = _search_shared(
            digests, needle, soft, soft_tier_wanted(rows, needle)
        )
        admitted = exact_body | soft_hits
    matched = filter_rows(rows, needle, admitted)
    ranked = rank_matches(matched, needle, exact_body, admitted=admitted)
    return ranked[:limit] if limit is not None else ranked


def search_store(
    config_dir: Path,
    query: str,
    *,
    rows: Sequence[SessionRow] | None = None,
    soft: SoftSearchIndex | None = None,
    limit: int | None = None,
) -> list[SessionMatch]:
    """One-shot store search: scan the rows, build the index, answer ``query``.

    The entry point for a caller with no rows of its own — the phone daemon and
    the desktop route. They must not each own a scan and an index build, and
    they must not disagree about either.

    ``rows`` is supplied only when the caller already has them (and knows they
    are current); otherwise the store is scanned UNCAPPED, matching the picker.
    A cap here would be the bug the picker documents at its own call site: a
    session past the cap is not merely off-screen, it is unfindable and
    indistinguishable from one that was deleted. Affordable because the scan is
    limit-independent and each row costs one bounded head read — measured at
    12.8 ms for the whole of a 235-session store on the reporting machine, plus
    ~3.3 ms for the warm index and single-digit milliseconds for the query.

    **An empty answer is not proof of an empty store.** The scan underneath
    (``resume._scan_sessions``) catches the ``OSError`` from an unreadable
    ``sessions/`` directory and returns no candidates, deliberately: every
    listing surface would otherwise fail on a store it cannot read, and a
    listing that fails is worse than one that is empty. So a store that cannot
    be walked reports zero matches rather than raising, and this function does
    not pretend otherwise — a caller that must distinguish the two has to ask
    the filesystem itself.
    """
    if rows is None:
        rows = recent_session_rows(config_dir, limit=None)
    rows = list(rows)
    middle = query.strip()
    if not middle:
        return search_rows(rows, query, limit=limit)
    digests = build_index(config_dir, [row.id for row in rows])
    return search_rows(rows, query, digests=digests, soft=soft, limit=limit)
