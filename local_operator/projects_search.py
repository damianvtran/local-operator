"""Soft search over project rows — the backend derivation the Projects views read.

WHY THIS EXISTS. The Projects views need text search with "some degree of
efficient soft match" that ranks "based on what was matched and where", and the
operator asked for the index work to happen in the backend "ahead of time ...
efficiently in an async way", not re-derived in the UI on every tab. Two facts
force the backend: ``updates[]`` is detail-only on the wire (the largest slice
of the corpus — 564 KB of the store's 651 KB when the design was measured — is
invisible to a client-side matcher), and rank must be composed where "what
matched, and where" is known, so there is ONE ranking model and one score.

**In-process derivation, never a persisted index artifact.** The server already
parses every row per request (the fresh
:class:`~local_operator.projects.ProjectRegistry`), so the only additions here
are the per-field token caches and the ranking pass; the fold and tokenisation
are re-paid only for fields whose text actually changed (see
:class:`_FieldIndex`). A persisted artifact or a write-time hook would
duplicate the source of truth and put index maintenance inside the store's
write path — neither is justified at these sizes. Measured costs live in the PR
and ``scripts/bench_projects_search.py``.

**What "soft" means here — the session-search contract, not a second
algorithm.** Matching reuses :mod:`local_operator.session.search_index`:
casefold plus diacritic fold, prefix matching for every query token, bounded
edit distance <= 2 on tokens of 4+ characters, and order-independent
token-AND across fields. It buys typos, prefixes and word order — NOT
synonyms: "billing" does not find "payments". The deferred upgrade path (a
vector slot on the index entry, cosine rerank over the fuzzy candidate set) is
deliberately not built; see the architecture note for its trigger conditions.

**Ranking — the operator's tiers, as one tunable weight dict.** name/title 8 >
description 4 = tags 4 > owner 3 = team 3 > updates 2 = progress 2 > id 1. A
hit's score is the sum over fields of ``weight * (query tokens matched in that
field)``, plus a phrase bonus (``weight`` again) when the whole folded query is
a contiguous substring of the field — so both WHAT matched and WHERE move the
number. The total order is deterministic: score desc, strongest matched field
desc, ``updated_at`` desc, display name asc (casefold), id asc. Identical
inputs produce byte-identical output (pinned in tests): a row moving under the
cursor for the SAME query is a bug; reordering as the query narrows is
relevance.

**Fields are indexed SEPARATELY** (one :class:`_FieldIndex` per field) because
attribution is the point: two query tokens may land in different fields of one
row, and the ranker must know how many matched in EACH field.

**Concurrency.** Routes run in a threadpool (``asyncio.to_thread``), so the
module-level caches are shared across request threads; every reconcile and read
happens under :data:`_LOCK`.
"""

from __future__ import annotations

import threading
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from local_operator.projects import Project, display_name
from local_operator.session.search_index import SoftSearchIndex, _tokenize

#: The indexed fields, in canonical order. One order serves three duties: the
#: weight dict's iteration order, the ``fields[]`` order on a hit ("ordered by
#: weight" — deterministic for the 4=4, 3=3, 2=2 ties), and the strongest-
#: matched-field tie-break.
_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "tags",
    "owner",
    "team",
    "updates",
    "progress",
    "id",
)

#: One dict, "tuned later against a judged fixture set" — so the RULES the
#: operator stated are the load-bearing part: title outranks description
#: outranks updates (8 > 4 > 2), and the display name's two spellings index
#: into one field (see :func:`_field_text`). Scores are comparable within one
#: answer only; changing these deliberately changes ranking (and the tests
#: that pin it).
_FIELD_WEIGHTS: dict[str, int] = {
    "name": 8,
    "description": 4,
    "tags": 4,
    "owner": 3,
    "team": 3,
    "updates": 2,
    "progress": 2,
    "id": 1,
}


def normalize(text: str) -> str:
    """The ONE fold: casefold + NFKD with combining marks stripped.

    Applied to BOTH the indexed field texts and the query, and the two MUST
    agree or a match is a coincidence — this is the function that guarantees
    they do. The session-search index has no diacritic fold; this is an
    additive improvement for project prose ("Café" is found by "cafe").

    Punctuation is NOT stripped here (the matcher's tokenizer splits on it);
    the phrase-substring tier therefore compares the folded text as written —
    the documented v1 limit: "coordination-links" and "coordination links" are
    the same tokens but not the same phrase.

    An ASCII fast path: for pure-ASCII input the fold IS casefold (NFKD
    decomposes nothing and there are no combining marks to strip), so the
    common case pays a scan plus casefold instead of a full decomposition —
    this is the cold path's dominant cost over a multi-megabyte corpus.
    """
    if text.isascii():
        return text.casefold()
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _field_text(project: Project, field: str) -> str:
    """One field's searchable text for one row — the index's input.

    ``name`` carries BOTH spellings a reader might type: the addressing
    ``name`` and the display ``title`` when set ("name (+ title)" in the
    design; the operator's "title is higher value than description"). The rest
    map one-to-one; ``progress`` also covers ``progress_reported_by`` so a
    search for a reporter lands where the reporter is recorded.
    """
    if field == "name":
        return " ".join(part for part in (project.name, project.title or "") if part)
    if field == "description":
        return project.description
    if field == "tags":
        return " ".join(project.tags)
    if field == "owner":
        return project.owner or ""
    if field == "team":
        return project.team or ""
    if field == "updates":
        return " ".join(entry.text for entry in project.updates)
    if field == "progress":
        return " ".join(part for part in (project.progress, project.progress_reported_by) if part)
    if field == "id":
        return project.id
    raise ValueError(f"unknown searchable field: {field!r}")


class _FieldIndex(SoftSearchIndex):
    """One field's token cache: per-project attribution, raw-first freshness.

    Subclasses the session-search accelerator so the tokenizer, the vocabulary
    buckets and the prefix/edit-distance tiers have ONE implementation
    (``SoftSearchIndex._resolve`` is reused as-is); this class adds the two
    things the project ranker needs on top:

    * **Raw-first freshness.** The base class stores the folded digest and
      re-folds every document it is handed. Projects are re-parsed from disk
      on every request (a fresh registry per request), so an eager fold would
      re-pay the whole-corpus fold per keystroke — measured on the live store
      at ~60-250 ms per pass under fleet load, before any matching. The
      freshness comparison here is on the RAW field text (:attr:`_raw`); the
      fold and tokenisation are paid only when a field's text actually
      changed. The folded text is still retained (in the base class's record)
      because the phrase tier reads it.
    * **Per-project token flags** (:meth:`token_flags`) instead of the base
      class's match/no-match set: the ranker needs to know WHICH query tokens
      matched in this field — to score them, and to admit a row whose tokens
      land in different fields.
    """

    def __init__(self) -> None:
        super().__init__()
        #: project id -> the RAW text the cached record was built from. A
        #: separate dict rather than a wider record tuple: the base class's
        #: ``_tokens`` shape (folded text, tokens) stays valid, so its
        #: ``_rebuild_vocab`` is reused unchanged.
        self._raw: dict[str, str] = {}

    def sync(self, docs: dict[str, str]) -> None:
        """Reconcile the cache against ``docs`` (project id -> raw field text).

        An entry whose raw text is unchanged is reused untouched — the warm
        path is one string comparison per field per request; changed and new
        entries are folded and tokenised; ids no longer present are dropped,
        so the cache tracks the live store and never grows past it (the base
        class's own bound rule).
        """
        changed = False
        for project_id in [pid for pid in self._tokens if pid not in docs]:
            del self._tokens[project_id]
            del self._raw[project_id]
            changed = True
        for project_id, raw in docs.items():
            if self._raw.get(project_id) == raw:
                continue
            folded = normalize(raw)
            self._raw[project_id] = raw
            self._tokens[project_id] = (folded, frozenset(_tokenize(folded)))
            changed = True
        if changed or self._vocab is None:
            self._rebuild_vocab()

    def resolved(self, tokens: Sequence[str]) -> list[set[str]]:
        """Each query token's softly-matching vocabulary words, resolved once."""
        return [self._resolve(token) for token in tokens]

    def token_flags(self, project_id: str, resolved: Sequence[set[str]]) -> list[bool]:
        """Per query token: does it softly match ``project_id``'s field text?"""
        cached = self._tokens.get(project_id)
        if cached is None:
            return [False] * len(resolved)
        words = cached[1]
        return [bool(words & matches) for matches in resolved]

    def folded_text(self, project_id: str) -> str:
        """The folded field text the cached record was built from (or ``""``)."""
        cached = self._tokens.get(project_id)
        return cached[0] if cached is not None else ""


@dataclass(frozen=True)
class ProjectSearchMatch:
    """One ranked hit of :func:`search_projects` — wire-neutral on purpose, so
    the desktop route and a future ``search`` op on the ``project`` tool share
    one shape.

    ``fields`` lists the matched fields in canonical weight order (the order
    :data:`_FIELDS` declares); ``name`` is the DISPLAY name (the title when
    set, else the addressing name — the fallback every surface renders);
    ``score`` is comparable within one answer, never across queries or builds
    (weights are tunable).
    """

    id: str
    name: str
    score: float
    fields: tuple[str, ...]


#: One accelerator per field, process-wide, behind :data:`_LOCK`. Per-field —
#: not one index over a joined digest — because the ranker scores "how many
#: query tokens matched in WHICH field". The lock mirrors the sessions'
#: shared-index rule: the desktop routes run in a threadpool, so every
#: reconcile and read of these caches is serialised.
_INDEXES: dict[str, _FieldIndex] = {field: _FieldIndex() for field in _FIELDS}
_LOCK = threading.Lock()


def search_projects(
    rows: Sequence[Project], query: str, *, limit: int | None = None
) -> list[ProjectSearchMatch]:
    """Ranked matches for ``query`` over ``rows``, best first — a total order.

    ``limit`` truncates AFTER ranking (``None`` returns every match). An empty
    or all-punctuation query answers ``[]``: ranking without tokens is
    undefined, and the empty-BOX behaviour (the listing's own order, untouched)
    belongs to the surface that owns the listing — the desktop route answers it
    there. See the module docstring for the match tiers and the score rules.
    """
    folded_query = normalize(query).strip()
    tokens = _tokenize(folded_query)
    if not tokens:
        return []
    with _LOCK:
        docs = {
            field: {project.id: _field_text(project, field) for project in rows}
            for field in _FIELDS
        }
        resolved: dict[str, list[set[str]]] = {}
        for field in _FIELDS:
            index = _INDEXES[field]
            index.sync(docs[field])
            resolved[field] = index.resolved(tokens)
        ranked: list[tuple[tuple[float, int, float, str, str], ProjectSearchMatch]] = []
        for project in rows:
            scored = _score_project(project, folded_query, tokens, resolved)
            if scored is not None:
                ranked.append(scored)
    ranked.sort(key=lambda pair: pair[0])
    matches = [match for _key, match in ranked]
    return matches[:limit] if limit is not None else matches


def _score_project(
    project: Project,
    folded_query: str,
    tokens: Sequence[str],
    resolved: dict[str, list[set[str]]],
) -> tuple[tuple[float, int, float, str, str], ProjectSearchMatch] | None:
    """Score one row, or ``None`` when a query token matched no field.

    The exclusion arm is the bounded-fuzzy rule: a token close to nothing in a
    row EXCLUDES that row rather than dragging it in ("excludes rather than
    ranks"). The returned key is the total order — score desc, strongest
    matched field desc, ``updated_at`` desc, display name asc (casefolded),
    id asc as the last-resort discriminator (two rows CAN share a display
    title, and the order is a tested guarantee, so it must be total).
    """
    score = 0.0
    matched_fields: list[str] = []
    admitted = [False] * len(tokens)
    for field in _FIELDS:
        flags = _INDEXES[field].token_flags(project.id, resolved[field])
        count = sum(flags)
        if not count:
            continue
        weight = _FIELD_WEIGHTS[field]
        score += weight * count
        if folded_query in _INDEXES[field].folded_text(project.id):
            # The phrase bonus: the whole folded query as one contiguous run
            # of the field (punctuation not stripped — documented v1 limit).
            score += weight
        matched_fields.append(field)
        for position, flag in enumerate(flags):
            admitted[position] = admitted[position] or flag
    if not all(admitted):
        return None
    name = display_name(project)
    strongest = _FIELD_WEIGHTS[matched_fields[0]]
    key = (-score, -strongest, -project.updated_at, name.casefold(), project.id)
    match = ProjectSearchMatch(id=project.id, name=name, score=score, fields=tuple(matched_fields))
    return key, match
