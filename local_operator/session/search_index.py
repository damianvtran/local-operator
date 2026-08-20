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
compresses that to ~0.7 MB, which loads in ~1 ms and searches in microseconds.
The bound is per session and applied at BUILD time, so a single pathological
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

**Search is substring, not semantic.** No embedding model, no vector store: a
provider call to open a picker would be slower than the thing it is searching,
would fail offline, and would spend money per keystroke. Substring over the
conversation body already answers the reported case — "retention" now finds the
session — and it never returns a confident wrong answer, which a similarity
score over 200 short digests very much does.

**What this does NOT do.** It is not a full-text index. :data:`SCAN_BYTES`
bounds how much of each conversation is represented, so a phrase that appears
only deep inside a long session may not be found; see that constant for the
measured recall and why the bound is where it is. The guarantee offered here
is "findable by what the conversation was about", not "findable by any word
ever typed in it".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from local_operator.resume import TRANSCRIPT_NAME

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
INDEX_VERSION = 1

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

    Entries for sessions no longer being listed are dropped, so the file cannot
    grow forever behind a store that retention keeps trimming.
    """
    cached = _load(index_path(config_dir))
    entries: dict[str, Any] = {}
    digests: dict[str, str] = {}
    for session_id in session_ids:
        transcript = config_dir / "sessions" / session_id / TRANSCRIPT_NAME
        try:
            stat = transcript.stat()
            signature = [stat.st_size, stat.st_mtime]
        except OSError:
            # Vanished mid-scan (retention sweeps run concurrently). Skip it
            # rather than caching an empty digest that would then be treated as
            # valid if the file came back.
            continue
        previous = cached.get(session_id)
        if (
            isinstance(previous, dict)
            and previous.get("signature") == signature
            and isinstance(previous.get("digest"), str)
        ):
            digest = previous["digest"]
        else:
            digest = digest_transcript(transcript)
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
    return {sid for sid, digest in digests.items() if needle in digest.lower()}
