"""Make a past conversation findable by what was SAID in it, not just its opener.

Why this exists. The ``/resume`` picker filters on two fields: the session's
name and its 12-hex id. Neither is what a user remembers a week later. The id
is unmemorable by construction, and the name — before this module — was the
conversation's *opening message*, so finding a session required recalling the
first thing typed into it. The reported failure was exactly that: a session
about session naming and retention could not be found by searching for
"retention", because that word appears in the sixty turns of the conversation
and not in the sentence that opened it. A session you cannot find is lost as
surely as one that was deleted.

So the filter needs to see the conversation's BODY. That is what this module
supplies: a small, bounded, per-session digest of the text of the turns, built
once and cached, which the picker searches alongside the name and the id.

**Why a digest and not the transcripts themselves.** The store here is 204
sessions and 125 MB; a real store grows without bound. Grepping all of it costs
~190 ms even reading only the first 256 KB of each file — far too slow to run
on every keystroke of a filter, and the filter reruns per keystroke. A digest
of the first :data:`DIGEST_CHARS` characters of each conversation's prose
compresses that to ~0.7 MB, which loads in ~1 ms. Exact-substring search
(:func:`search_digests`) is then ~5 ms over the whole store; bounded soft
search (:class:`SoftSearchIndex`) is ~6-20 ms per query change once its per-
digest token cache is warm (prefix queries at the low end, a full typo-tolerant
word at the high), after a one-time ~70 ms cost on the first keystroke to build
that cache. All measured at ~640 sessions / ~180k digest tokens — see
:class:`SoftSearchIndex` for why the naive per-call form was ~75-185 ms EVERY
keystroke. The bound is per session and applied at BUILD time, so a single
pathological
transcript (a pasted 80 MB file) costs its cap and not its size.

**Why the cache lives outside the session directories.** A sidecar inside
each session directory would be simpler to invalidate, but writing one
would move the directory's mtime and make a listing that ranks by
directory recency treat an index rebuild as user activity. One file
under the cache directory touches nothing a session listing measures.

**Freshness without a full rebuild.** Each entry records the transcript's size
and mtime. A session whose transcript still matches its entry is reused as-is;
anything else — new, grown, or missing from the cache — is re-digested. A
normal picker open therefore re-reads only the handful of sessions that have
actually changed since the last one, which is why the index can be built
synchronously without the picker stalling.

**Search is lexical — substring plus BOUNDED soft matching, not semantic.** No
embedding model, no vector store: a provider call to open a picker would be
slower than the thing it is searching, would fail offline, and would spend
money per keystroke, and a local embedding model is the heaviest dependency
this offline-capable, ``pip install``-light project could add. Two lexical
tiers instead:

* **Substring** (:func:`search_digests`) — exact, case-insensitive, the
  precise-query path that never returns a confident wrong answer.
* **Bounded soft matching** (:func:`soft_search_digests`) — prefix,
  order-independent token-AND, and edit-distance-<=2 for tokens of 4+ chars, so
  ``classifer`` finds ``classifier`` and ``throughput classifier`` finds
  "Improve ADM Classifier Throughput". The earlier design warned that fuzzy
  matching produces confident nonsense; that was right about UNBOUNDED fuzzy,
  and the bound here — a small distance cap gated on token length — is exactly
  what keeps a soft match from dragging in everything. A similarity score over
  200 short digests would rank a best match for every query including nonsense;
  the AND-of-bounded-tokens rule excludes rather than ranks, so a query that
  matches nothing still matches nothing.

The digest also PREPENDS the session's title and every past name it has borne
(from the ``title.json`` sidecar), so a topic-pivot session — one whose subject
changed partway through and was re-titled — is findable by the subject it ended
on, which is the human's own summary of what it became. This is how "find by
meaning" is served honestly without a vector model.

**What this does NOT do.** It is not a full-text index. :data:`SCAN_BYTES`
bounds how much of each conversation's BODY is represented, so a phrase that
appears only deep inside a long session and never in its title may not be
found; see that constant for the measured recall and why the bound is where it
is. Full-text recall past the scan window, and true vector semantics, are both
deferred follow-ups behind a dependency-approval gate. The guarantee offered
here is "findable by what the conversation was about", not "findable by any
word ever typed in it".
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from local_operator.resume import TITLE_SIDECAR_NAME, TRANSCRIPT_NAME, read_title_names

#: Characters of prose kept per session. Enough to hold the subject matter of a
#: long conversation's opening stretch — the part that says what the session was
#: FOR — without letting one session dominate the index. At ~3.5 KB per entry a
#: thousand-session store is a ~3.5 MB index, which still loads in a frame.
DIGEST_CHARS = 4_000

#: Bytes of each transcript the digest is extracted from. Deliberately larger
#: than :data:`DIGEST_CHARS`: transcripts carry tool calls, results and base64
#: payloads between the turns, so the prose is sparse within the file and a
#: window this size typically yields the first few dozen turns.
#:
#: **This bounds RECALL, and the bound is real.** On a 204-session store, 35 of
#: 58 transcripts exceed it, and measured against a full parse the index finds
#: 3 of 5 sessions containing "retention" and 4 of 16 containing "screenshot" —
#: a word that recurs late in long sessions is the case it misses. The digest
#: is therefore "what this conversation was ABOUT", taken from its opening
#: stretch, not a full-text index of everything said.
#:
#: Deliberate rather than conceded: the picker builds this synchronously when
#: it opens, and the whole store has to be digestible inside a frame. Raising
#: the window trades that budget for recall of words that appear only deep in a
#: long conversation, which is the weaker half of the reported problem — the
#: reported failure was not being able to find a session by its SUBJECT. If
#: full-text recall is wanted later it belongs in a real index built off the
#: hot path, not in a bigger synchronous read.
SCAN_BYTES = 256_000

#: Where the index is written, under the cache directory rather than in any
#: session directory (see the module docstring: retention measures those).
INDEX_FILENAME = "session_search_index.json"

#: Bumped when the digest's SHAPE changes. An index written by an older build
#: is discarded wholesale rather than migrated — it is a derived artifact whose
#: rebuild costs about a second, and a migration path for a cache is code that
#: exists to be wrong.
#:
#: 1 -> 2: the digest now PREPENDS the session's title and every past name it
#: has borne (from the ``title.json`` sidecar), and the cache signature gains
#: the sidecar's mtime so a rename re-folds the digest even when the transcript
#: is byte-identical. Bumping discards every v1 entry once so the whole store
#: rebuilds with the new shape; ``_load`` already drops on version mismatch, so
#: this needs no migration code.
INDEX_VERSION = 2

#: Extra characters the title-fold may add on top of :data:`DIGEST_CHARS`. The
#: title and past names are the human's own summary of what the session became,
#: so they must survive the cap that a long opening stretch would otherwise
#: consume — folding them in and THEN slicing at ``DIGEST_CHARS`` could clip the
#: very title the fold exists to make searchable. A session accrues few names
#: (a rename is rare), so a small fixed headroom holds all of them without
#: letting the digest grow unbounded.
TITLE_HEADROOM = 512

#: Roles whose text enters the digest. Tool results are excluded on purpose:
#: they are machine output — directory listings, file dumps, HTTP bodies — and
#: indexing them makes every session match every path-like query, which is
#: indistinguishable from the search being broken.
_INDEXED_ROLES = frozenset({"user", "assistant"})


def index_path(config_dir: Path) -> Path:
    """Where this store's search index lives."""
    return config_dir / "cache" / INDEX_FILENAME


def digest_transcript(transcript: Path) -> str:
    """The searchable prose of one conversation, bounded and flattened.

    Returns ``""`` for anything unreadable. A session that cannot be digested
    is still listed and still searchable by name and id — losing its body from
    the index must never cost it its row.
    """
    try:
        with transcript.open("r", encoding="utf-8", errors="replace") as handle:
            # A bounded read for the same reason ``resume.session_name`` uses
            # one: iterating lines materialises each in full before any cap can
            # apply, so a transcript whose first line is a base64 image is
            # allocated whole by the very loop meant to bound it.
            head = handle.read(SCAN_BYTES)
    except OSError:
        return ""
    parts: list[str] = []
    size = 0
    for line in head.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            # Normal, not exceptional: the last line of a running session's
            # transcript is often half-written, and the window above can cut a
            # line in the middle regardless.
            continue
        if not isinstance(entry, dict) or entry.get("type") != "message":
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict) or payload.get("role") not in _INDEXED_ROLES:
            continue
        for text in _texts(payload.get("content")):
            parts.append(text)
            size += len(text)
            if size >= DIGEST_CHARS:
                # Stop reading as soon as the cap is reached rather than
                # collecting everything and slicing at the end: the slice would
                # still have paid to build the discarded remainder.
                return " ".join(" ".join(parts).split())[:DIGEST_CHARS]
    return " ".join(" ".join(parts).split())[:DIGEST_CHARS]


def _texts(content: object) -> list[str]:
    """Every text part of a persisted message's content, in order.

    Image and other non-text parts are skipped rather than described: a digest
    entry reading ``[image]`` for every screenshot would make "image" match
    most of the store.
    """
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                out.append(text)
    return out


def _load(path: Path) -> dict[str, Any]:
    """The cached index, or an empty one when it is absent, stale or corrupt.

    Every failure yields an empty index rather than raising. This is a cache:
    the worst a bad file may cost is a rebuild, never the picker.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    try:
        loaded = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(loaded, dict) or loaded.get("version") != INDEX_VERSION:
        return {}
    entries = loaded.get("entries")
    return entries if isinstance(entries, dict) else {}


def _save(path: Path, entries: dict[str, Any]) -> None:
    """Persist the index, best-effort and atomically.

    Atomic because the picker reads this file while another session may be
    writing it: a half-written JSON document would be discarded by ``_load``
    (costing a needless rebuild) on every open until someone rewrote it.
    Best-effort because an unwritable cache directory must cost the SPEED of
    the next search and never the search itself.

    The temp file carries the WRITER'S PID. A fixed name is shared by every
    process, so two sessions opening a picker at once wrote the same path and
    one ``replace``d a document the other was still filling — measured at 15
    torn reads in 1668 across three real processes. The consequence was bounded
    (a torn document is discarded and rebuilt, never read as data), but it is a
    needless cost for one interpolation.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps({"version": INDEX_VERSION, "entries": entries}),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        return


def build_index(config_dir: Path, session_ids: list[str]) -> dict[str, str]:
    """``{session_id: digest}`` for ``session_ids``, reusing what is still valid.

    Only sessions whose transcript has changed since the cached entry was
    written are re-digested, keyed on ``(size, mtime)``. That pair is what
    makes calling this on every picker open affordable: a store where nothing
    changed costs one file read, and the common case — one session appended to
    since the last open — costs one re-digest.

    Entries for sessions this call was NOT asked about are PRESERVED. The
    previous behaviour rebuilt the file from the requested ids alone, which
    made every narrow caller evict the wide caller's work: the mobile daemon
    asks for 200 ids (``mobile.daemon._search_sessions``) or 100
    (``SessionRegistry.summaries``), pruning the cache to those, and the next
    picker open then re-digested the whole store from disk — measured at
    936-2430 ms, the one production path that reached a full second.

    The pruning's original justification ("so the file cannot grow forever
    behind a store that retention keeps trimming") is OBSOLETE and its removal
    is the point of this shape. Commit 4173ec73 retired eviction; ``retention``
    never deletes a transcript, so the cache can only grow as fast as the store
    and the store is permanent by policy. The bound that replaces it is the
    honest one for that world: an entry is dropped when its session DIRECTORY
    is gone, which is now the only way a session leaves the store (explicit
    user disposal). Keep both facts together — a future reader who restores the
    prune-to-requested behaviour because the docstring sounds prudent
    reintroduces the daemon thrash.
    """
    cached = _load(index_path(config_dir))
    # Seed from the cache rather than from scratch, then update only the ids
    # this call was asked about (see the docstring): this is what preserves a
    # wide caller's entries across a narrow caller.
    sessions_root = config_dir / "sessions"
    entries: dict[str, Any] = {
        sid: entry
        for sid, entry in cached.items()
        # The replacement bound for the pruning removed above: an entry whose
        # session directory is gone can never be valid again, and a session
        # only leaves the store by explicit disposal now. One stat per cached
        # entry (~5 ms at 2700) buys a cache that tracks the live store
        # instead of growing across every session that ever existed.
        if isinstance(entry, dict) and os.path.isdir(os.path.join(sessions_root, sid))
    }
    digests: dict[str, str] = {}
    for session_id in session_ids:
        session_dir = config_dir / "sessions" / session_id
        transcript = session_dir / TRANSCRIPT_NAME
        try:
            stat = transcript.stat()
        except OSError:
            # Vanished mid-scan (retention sweeps run concurrently). Skip it
            # rather than caching an empty digest that would then be treated as
            # valid if the file came back. The seeded entry is dropped with it:
            # a transcript that cannot be stat'd cannot have its signature
            # confirmed, and keeping a stale digest under a live directory
            # would serve search results for a body that no longer exists.
            entries.pop(session_id, None)
            continue
        # The title sidecar's mtime joins the signature so a RENAME re-folds the
        # digest even when the transcript is byte-identical (a rename appends to
        # the transcript too, but the sidecar is the authoritative source of the
        # names folded below, and keying on it makes the invalidation explicit).
        # ``0`` when there is no sidecar yet, so a pre-sidecar session keys the
        # same way it did under v1 until its backfill or next rename writes one.
        try:
            title_mtime = (session_dir / TITLE_SIDECAR_NAME).stat().st_mtime
        except OSError:
            title_mtime = 0
        signature = [stat.st_size, stat.st_mtime, title_mtime]
        previous = cached.get(session_id)
        if (
            isinstance(previous, dict)
            and previous.get("signature") == signature
            and isinstance(previous.get("digest"), str)
        ):
            digest = previous["digest"]
        else:
            # Prepend the human's own summary of what the session became — its
            # title and every past name — to the body digest. This is what makes
            # a topic-pivot session findable by the subject it ended on rather
            # than only the one it opened with, WITHOUT enlarging the 256 KB scan
            # window (see SCAN_BYTES: full-transcript recall is a deferred
            # follow-up, not this change). Sliced with headroom so the folded
            # names always survive the DIGEST_CHARS cap the opening stretch would
            # otherwise fill.
            names = read_title_names(session_dir)
            body = digest_transcript(transcript)
            digest = " ".join([*names, body]).strip()[: DIGEST_CHARS + TITLE_HEADROOM]
        entries[session_id] = {"signature": signature, "digest": digest}
        digests[session_id] = digest
    if entries != cached:
        _save(index_path(config_dir), entries)
    return digests


def search_digests(digests: dict[str, str], query: str) -> set[str]:
    """Ids whose conversation body contains ``query``, case-insensitively.

    Returned as a set because the caller owns the ORDER of its rows: the picker
    must never reorder under a growing query (a row moving out from under the
    cursor is how a user resumes the wrong session), so this answers only
    "which ones match".
    """
    needle = query.strip().lower()
    if not needle:
        return set()
    # Lower the corpus ONCE per distinct digest set rather than per query. The
    # old form re-lowered every digest on every keystroke, which is 10 MB of
    # string allocation per character typed: 24 ms per keystroke at 2700
    # digests, against a 21 ms one-time cost and 5 ms per keystroke here. The
    # cache is keyed on the digests mapping's identity AND length so a
    # re-digested or resized store re-lowers rather than answering stale.
    return {sid for sid, digest in _lowered(digests).items() if needle in digest}


#: One-entry memo for the lowered corpus, keyed on the identity and size of the
#: digests mapping it was derived from. One entry because the picker holds a
#: single digest set for its whole life, and the phone daemon's calls are
#: one-shot: a larger cache would retain corpora nobody will ask for again.
_LOWERED_KEY: tuple[int, int] | None = None
_LOWERED: dict[str, str] = {}


def _lowered(digests: dict[str, str]) -> dict[str, str]:
    """``digests`` with every body lowercased, memoised across queries.

    Identity-keyed rather than content-keyed on purpose: hashing 10 MB of
    digests to decide whether to lower 10 MB of digests would cost what it
    saves. The picker builds its digest mapping once and holds it, so identity
    is a sound key there; a caller that MUTATES a mapping in place between
    queries would see a stale corpus, which is why ``build_index`` returns a
    fresh dict on every call rather than updating one.
    """
    global _LOWERED_KEY, _LOWERED
    key = (id(digests), len(digests))
    if _LOWERED_KEY != key:
        _LOWERED = {sid: digest.lower() for sid, digest in digests.items()}
        _LOWERED_KEY = key
    return _LOWERED


#: Minimum token length for edit-distance matching. Short tokens (``adm``,
#: ``svc``) are within distance 2 of far too many words, so allowing a fuzzy
#: match on them is exactly the "confident nonsense" the substring-only design
#: warned about; a prefix or exact match still handles them. Four is where the
#: false-match rate drops to something a search can trust.
_SOFT_MIN_TOKEN = 4

#: Maximum edit distance a token may be from a haystack token and still match.
#: Two covers the ordinary typo (a swap, a doubled or dropped letter,
#: ``classifer`` -> ``classifier``) without letting unrelated words of similar
#: length collide. Kept small on purpose: the bound is what makes soft matching
#: safe here, per the module docstring.
_SOFT_MAX_DISTANCE = 2

#: Splits a string into lowercase alphanumeric tokens for soft matching. Word
#: boundaries only — punctuation and whitespace separate tokens — so a query
#: matches on whole words the way a reader thinks of them.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of ``text``, in order."""
    return _TOKEN_RE.findall(text.lower())


def _within_edit_distance(a: str, b: str, max_distance: int) -> bool:
    """True when ``a`` and ``b`` are at most ``max_distance`` edits apart.

    A bounded Levenshtein: the length gap alone can exceed the cap (a cheap
    reject before any work), and the row-by-row DP short-circuits as soon as
    every cell in a row exceeds the cap, so the cost is O(len * max_distance)
    rather than O(len^2). Pure Python and dependency-free on purpose —
    ``rapidfuzz`` is a compiled wheel this project deliberately does not add
    (see the module docstring). One DP is sub-millisecond; the cost that
    mattered was running it per digest token per keystroke, which
    :class:`SoftSearchIndex` removes by resolving each query token against the
    deduplicated vocabulary once (~6-20 ms per warm query change at ~640
    sessions, vs ~75-185 ms for the naive per-digest form on every keystroke).
    """
    if abs(len(a) - len(b)) > max_distance:
        return False
    if a == b:
        return True
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + cost,  # substitution
            )
            current.append(value)
            row_min = min(row_min, value)
        # Every cell in this row already exceeds the cap, so no later row can
        # bring the final cell back under it — the distance only grows.
        if row_min > max_distance:
            return False
        previous = current
    return previous[-1] <= max_distance


def _token_soft_matches(needle_token: str, haystack_tokens: set[str]) -> bool:
    """True when ``needle_token`` softly matches any token in the haystack.

    Two tiers beyond exact equality, both bounded:

    * **Prefix** — ``class`` matches ``classifier``. Handles a partial word a
      user types before they finish it, which substring already caught inside a
      token but this states explicitly against the tokenized haystack.
    * **Edit distance <= 2, tokens >= 4 chars** — ``classifer`` (a typo)
      matches ``classifier``. The length floor is what keeps short tokens from
      matching everything; below it, only the exact/prefix tiers apply.
    """
    for token in haystack_tokens:
        if token == needle_token or token.startswith(needle_token):
            return True
        if (
            len(needle_token) >= _SOFT_MIN_TOKEN
            and len(token) >= _SOFT_MIN_TOKEN
            and _within_edit_distance(needle_token, token, _SOFT_MAX_DISTANCE)
        ):
            return True
    return False


class SoftSearchIndex:
    """A reusable soft-search accelerator that caches per-digest tokenisation.

    Why this exists as an object rather than the old stateless function. The
    picker recomputes soft matches on every KEYSTROKE (see
    ``SessionPickerScreen.visible_rows``), and the naive implementation
    re-tokenised every digest and ran the edit-distance DP against every
    digest token on each of those calls. At the real store scale (~640
    sessions, ~180k digest tokens) that cost ~75-185 ms per keystroke, measured
    (~75 ms for a short/prefix query, ~185 ms for a full typo-tolerant word) \u2014
    the "microseconds" the earlier docstrings claimed held only at the ~200
    sessions the design was first measured on. Two costs dominated and both are
    eliminated here:

    * **Re-tokenising every digest per keystroke (~55 ms).** Fixed by caching
      the token set of each digest keyed on the digest STRING, so a digest is
      tokenised once and reused until it actually changes.
    * **Running the edit-distance DP per DIGEST token (up to ~130 ms).** The 180k
      digest tokens are only ~8.5k DISTINCT words. Fixed by resolving each
      query token against that deduplicated VOCABULARY once \u2014 the DP runs
      ~8.5k times, not ~180k \u2014 then answering each digest by a cheap set
      intersection against the resolved word set. Length- and first-letter
      buckets over the vocabulary cut the DP candidate list further (edit
      distance <=2 can only reach words within two of the query token's
      length; a prefix match shares its first letter).

    Freshness and bound. ``search`` re-syncs the cache against the ``digests``
    it is handed on every call: an entry whose digest STRING changed is
    re-tokenised (a re-digested session invalidates itself), and an entry whose
    session is gone is dropped \u2014 so the cache tracks the live digest set and
    cannot grow past it across the picker's life. The derived vocabulary and
    its buckets are rebuilt only when that token cache actually changed, so a
    run of keystrokes over a fixed store rebuilds them once. Net warm cost is
    ~6-20 ms per query change (prefix cheap, full typo dearer), after a one-time
    ~70 ms first keystroke that builds the token cache and vocabulary.

    Behaviour is identical to the previous stateless implementation: same
    prefix / order-independent token-AND / edit-distance-<=2-for-tokens->=4
    bounds, same matches. Only the cost changed. See :func:`soft_search_digests`
    for the stateless one-shot wrapper tests and non-repeating callers use.
    """

    def __init__(self) -> None:
        # sid -> (digest_string, frozenset_of_tokens). Keyed on the digest
        # string so a re-digested session (new string) is re-tokenised, and
        # pruned to the live sids on each sync so it cannot grow unbounded.
        self._tokens: dict[str, tuple[str, frozenset[str]]] = {}
        # Deduplicated view of every token across the cached digests, plus the
        # buckets that bound the DP candidate list. ``None`` marks them stale
        # (a token-cache change or the first search), so they rebuild lazily.
        self._vocab: set[str] | None = None
        self._by_len: dict[int, set[str]] = {}
        self._by_first: dict[str, list[str]] = {}

    def _sync(self, digests: dict[str, str]) -> None:
        """Reconcile the token cache with ``digests``; rebuild vocab if it moved."""
        changed = False
        # Drop entries for sessions that are no longer listed. This is what
        # keeps the cache bounded by the live store rather than by the number
        # of distinct digests seen over the picker's life.
        for sid in [sid for sid in self._tokens if sid not in digests]:
            del self._tokens[sid]
            changed = True
        for sid, digest in digests.items():
            cached = self._tokens.get(sid)
            # Re-tokenise only when the digest STRING differs (new session or a
            # re-digest after a rename/append), so freshness is honoured without
            # re-tokenising unchanged digests every keystroke.
            if cached is None or cached[0] != digest:
                self._tokens[sid] = (digest, frozenset(_tokenize(digest)))
                changed = True
        if changed or self._vocab is None:
            self._rebuild_vocab()

    def _rebuild_vocab(self) -> None:
        """Recompute the deduplicated vocabulary and its DP-narrowing buckets."""
        vocab: set[str] = set()
        for _digest, tokens in self._tokens.values():
            vocab |= tokens
        by_len: dict[int, set[str]] = {}
        by_first: dict[str, list[str]] = {}
        for word in vocab:
            by_len.setdefault(len(word), set()).add(word)
            by_first.setdefault(word[0], []).append(word)
        self._vocab = vocab
        self._by_len = by_len
        self._by_first = by_first

    def _resolve(self, needle_token: str) -> set[str]:
        """Vocabulary words this query token softly matches (see :func:`_token_soft_matches`).

        The per-vocabulary equivalent of :func:`_token_soft_matches`: instead of
        asking "does this token match any word in one digest", it computes, once,
        the set of ALL vocabulary words the token matches, so every digest is then
        answered by a set intersection. Same three tiers, same bounds.
        """
        matches: set[str] = set()
        # Prefix (which subsumes exact): a vocabulary word beginning with the
        # query token shares its first letter, so only that bucket can contain a
        # prefix hit. Mirrors ``token.startswith(needle_token)``.
        for word in self._by_first.get(needle_token[0], ()):
            if word.startswith(needle_token):
                matches.add(word)
        # Edit distance <=2, both tokens >=4 chars. A word within two edits can
        # differ in length by at most two, so only those length buckets can hold
        # a fuzzy hit \u2014 this is what turns the DP from ~180k runs into ~a few
        # hundred. The length and bound checks match _token_soft_matches exactly.
        if len(needle_token) >= _SOFT_MIN_TOKEN:
            for length in range(
                len(needle_token) - _SOFT_MAX_DISTANCE,
                len(needle_token) + _SOFT_MAX_DISTANCE + 1,
            ):
                for word in self._by_len.get(length, ()):
                    if len(word) >= _SOFT_MIN_TOKEN and _within_edit_distance(
                        needle_token, word, _SOFT_MAX_DISTANCE
                    ):
                        matches.add(word)
        return matches

    def search(self, digests: dict[str, str], query: str) -> set[str]:
        """Ids whose body SOFTLY matches ``query`` \u2014 prefix, typo, or word-order.

        Identical result to :func:`soft_search_digests`; see this class's
        docstring for why it is dramatically cheaper across repeated calls over
        the same store. Re-syncs its cache against ``digests`` first, so passing
        a changed store is safe.
        """
        self._sync(digests)
        needle_tokens = _tokenize(query)
        if not needle_tokens:
            return set()
        # Resolve each query token to the vocabulary words it matches, once. A
        # query token that matches nothing in the whole store yields an empty
        # set here, which makes the AND below reject every digest \u2014 the same
        # "excludes rather than ranks" bound the stateless version relied on.
        resolved = [self._resolve(token) for token in needle_tokens]
        matched: set[str] = set()
        for sid, (_digest, tokens) in self._tokens.items():
            if not tokens:
                continue
            # Order-independent AND: every query token must have a resolved word
            # present in this digest's token set. Set intersection replaces the
            # per-token DP the stateless version ran here.
            if all(tokens & words for words in resolved):
                matched.add(sid)
        return matched


def soft_search_digests(digests: dict[str, str], query: str) -> set[str]:
    """Ids whose body SOFTLY matches ``query`` \u2014 prefix, typo, or word-order.

    The bounded soft tier that sits beside :func:`search_digests`. A query
    matches a digest when EVERY query token softly matches some token in the
    digest (order-independent AND), where "softly" is exact, prefix, or a
    bounded edit-distance typo (see :func:`_token_soft_matches`). Order-
    independent AND is what makes ``throughput classifier`` find a session
    titled "Improve ADM Classifier Throughput" \u2014 word order and exact substring
    contiguity both stop mattering.

    Returned as a SET, like :func:`search_digests`: the caller
    (:class:`SessionPickerScreen.visible_rows`) owns row order, and ranking is
    applied there only when a query is active. This function answers only
    "which ones match".

    Bounded and dependency-free by design. The old substring-only filter warned
    that fuzzy matching over free text produces confident nonsense; that was
    right about UNBOUNDED fuzzy. This is bounded \u2014 a distance cap of 2 for
    tokens of 4+ chars, plus prefix and token-AND \u2014 and the bound is precisely
    what keeps a soft match from being confident nonsense: a nonsense token
    close to nothing in the digest excludes the row rather than dragging in
    everything.

    **This is the STATELESS one-shot form.** It builds a fresh
    :class:`SoftSearchIndex` per call, so it re-tokenises the whole store every
    time \u2014 at ~640 sessions that is ~75-185 ms, fine for a single call (a test, a
    mobile-daemon query) but NOT for a per-keystroke loop. A caller that
    searches the same store repeatedly (the picker) holds a
    :class:`SoftSearchIndex` instead and pays that cost once; see that class.
    """
    return SoftSearchIndex().search(digests, query)
