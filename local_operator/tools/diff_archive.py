"""Bounded, source-redacted archive of COMPLETE write/edit diffs, and its pager.

WHY THIS EXISTS. ``write``/``edit`` results carry an inline unified diff in
``details['diff']`` capped at 200 rows / 120 KiB (``tools/builtin._diff_details``)
because the transcript persists it. Every row past that cap was simply gone: a
reviewer reading a subagent's work could not see the change it was approving.
Size context (architect sample of the 2,156 most recent uncompressed session
transcripts, n=20,302 write/edit results, counting CHANGED lines -- added plus
removed, not rendered rows): median 16, p99 293, p99.9 705, max 1,847. 600
results (3.0%) had over 180 changed lines, a proximity surrogate for the
preview cap, not a measured count of diffs over 200 rendered rows.

This module keeps the WHOLE diff -- redacted, bounded, immutable -- and serves
it back by row range to the parent and child readers.

THE SECURITY PIPELINE (each step is load-bearing):

1. The producer renders the complete diff BEFORE the preview cut and hands it,
   with the complete before/after file sides, to the loop on the private
   ``ToolResult._diff_source`` carrier (``tools/builtin._diff_text_result``).
   A private attribute is outside every serializer and the carrier is dropped
   from every copy/pickle, so raw rows never reach an event, a transcript row
   or a process boundary.
2. A complete diff over :data:`ARCHIVE_MAX_ROWS` / :data:`ARCHIVE_MAX_BYTES` is
   NOT archived (``oversized``). No prefix is ever stored: a multi-line value can
   straddle any cut, and only the whole evidence can prove it is absent.
3. The loop (``AgentLoop._settle_write_diff``) redacts each complete ORIGINAL
   file side ONCE, projects the touched source lines onto the full signed diff
   (``_mask_by_source_evidence``: signs, hunk headers and counts stay exact, the
   masked side is never re-diffed), then runs the per-hunk seam with this
   module's scaled budgets. Any refusal stores nothing (``withheld``). With no
   redactor configured nothing is archived (``unavailable``): an unredacted
   artifact on disk would outlive the live process that justified showing it.
4. Only the redacted rows are persisted, in a DEDICATED :class:`SpillStore` root
   (``<config_dir>/diff-archive``): the spill store's 4 MiB entry and 64 MiB
   total LRU limits, and neither evicting nor evicted by model spill output.
   An artifact over the entry limit is refused whole (``oversized``), so the
   store's head/tail clip never fires; eviction answers ``evicted``.
5. Readers prove membership from the SESSION'S OWN transcript: the persisted
   ``role:"tool"`` row for ``call_id`` is a write/edit result whose descriptor
   names the digest, and the artifact header must name the same session and
   call. A client never supplies a digest.

NOT general file versioning: nothing here reads the current file, and rows that
were never captured stay unrecoverable (pre-archive rows answer
``not_recorded``; a row whose inline diff was complete answers ``inline``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from local_operator.paths import config_dir
from local_operator.tools.spill import SPILL_ENTRY_LIMIT_BYTES, SpillStore

#: ``details`` key of the small descriptor. Never carries diff content.
DIFF_FULL_KEY = "diff_full"

#: Descriptor/artifact format version (the UI's frozen v1 contract).
DIFF_ARCHIVE_VERSION = 1

#: Rendered rows a COMPLETE diff may have to be archived. A CHOSEN bound,
#: ~4.4x the largest observed CHANGED-line count (1,847, sample above; rendered
#: rows add hunk headers and context on top of that). It is not proved above
#: every historical rendered diff: anything past it answers ``oversized``
#: explicitly and stores nothing, never a prefix.
ARCHIVE_MAX_ROWS = 8192

#: Raw UTF-8 bytes a complete diff may have to be archived: under the spill
#: entry cap (4 MiB) with 256 KiB headroom for the JSON row encoding and header.
ARCHIVE_MAX_BYTES = 3 * 1024 * 1024 + 768 * 1024

#: The per-hunk seam's budgets for the archive pass. The inline pass runs
#: 256 rows / 128 KiB over a 200-row / 120 KiB preview; the same framing ratio
#: over the archive caps. The seam's 2R-call / 3B-input arithmetic scales with
#: them, so one call stays linear (architect measurement: 40,003 rows /
#: 3.6 MiB in 4.5 s on this laptop, inside ``asyncio.to_thread``).
ARCHIVE_REDACT_ROWS = 10240
ARCHIVE_REDACT_BYTES = 4 * 1024 * 1024

#: Page ceilings. Rows match the child transcript page; bytes match the inline
#: preview's 120 KiB so one page never costs more than the preview already does.
#: Bytes are measured on the JSON-ESCAPED form (``ensure_ascii``), the encoding
#: worst case, so the bound holds whichever encoder the route uses.
PAGE_MAX_ROWS = 200
PAGE_MAX_BYTES = 120 * 1024

#: A hunk header the producer renders is ``@@ -a,b +c,d @@`` (< 50 chars); the
#: page metadata is bounded anyway so a crafted artifact cannot inflate it.
HUNK_HEADER_MAX_CHARS = 256

DIFF_ARCHIVE_DIRNAME = "diff-archive"

STATE_ARCHIVED = "archived"
STATE_OVERSIZED = "oversized"
STATE_WITHHELD = "withheld"
STATE_UNAVAILABLE = "unavailable"
#: Reader-derived states (never stored in a descriptor).
STATE_EVICTED = "evicted"
STATE_NOT_RECORDED = "not_recorded"
STATE_INLINE = "inline"

_STORED_STATES = frozenset({STATE_ARCHIVED, STATE_OVERSIZED, STATE_WITHHELD, STATE_UNAVAILABLE})
_DIGEST = re.compile(r"[0-9a-f]{32}")
_DIFF_TOOLS = frozenset({"write", "edit"})


class DiffArchiveNotFound(LookupError):
    """Unknown session, call, or a call that is not a write/edit result.

    One class for all of them so a route's 404 cannot distinguish them.
    """


class DiffPageRangeError(ValueError):
    """``start``/``limit`` outside the artifact or the page ceilings (422)."""


class _DiffArchiveStore(SpillStore):
    """The spill store's bounded LRU machinery under its own root.

    A separate root so diff artifacts neither evict nor are evicted by model
    spill output, and so the general ``spill://`` reader cannot address them.
    ``root`` re-resolves per access for the same reason ``spill_dir`` does: an
    isolated run relocating the config dir must be followed.
    """

    __slots__ = ()

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else config_dir() / DIFF_ARCHIVE_DIRNAME


_STORE = _DiffArchiveStore()


def archive_store(config_root: Path | None = None) -> SpillStore:
    """The diff archive store: process-wide, or under an explicit config root.

    A reader that already holds its config root (the desktop host's
    ``self.root``) passes it, so it reads the same tree it proved membership
    in rather than whatever ``config_dir()`` resolves to in that process.
    """
    if config_root is not None:
        return SpillStore(config_root / DIFF_ARCHIVE_DIRNAME)
    return _STORE


def descriptor(state: str, *, rows: int | None = None, digest: str | None = None) -> dict[str, Any]:
    """The small, content-free ``details['diff_full']`` value."""
    out: dict[str, Any] = {"v": DIFF_ARCHIVE_VERSION, "state": state}
    if rows is not None:
        out["rows"] = rows
    if digest is not None:
        out["digest"] = digest
    return out


def fits_archive(rows: Sequence[str]) -> bool:
    """Whether a COMPLETE raw diff is within the archive ceilings."""
    if len(rows) > ARCHIVE_MAX_ROWS:
        return False
    total = 0
    for row in rows:
        total += len(row.encode("utf-8", errors="surrogatepass"))
        if total > ARCHIVE_MAX_BYTES:
            return False
    return True


def _encode(rows: Sequence[str], *, session_id: str, call_id: str, tool_name: str) -> str:
    # One JSON string per line: exact for every character (a row's own
    # separators, lone surrogates escaped by ``ensure_ascii``), so the reader
    # gets the stored rows back byte for byte instead of re-splitting text.
    header = json.dumps(
        {
            "v": DIFF_ARCHIVE_VERSION,
            "kind": "diff-archive",
            "session_id": session_id,
            "call_id": call_id,
            "tool": tool_name,
            "rows": len(rows),
        },
        separators=(",", ":"),
    )
    return "\n".join([header, *(json.dumps(row) for row in rows)])


def write_archive(
    rows: Sequence[str], *, session_id: str, call_id: str, tool_name: str
) -> dict[str, Any]:
    """Store ALREADY-REDACTED rows; return the descriptor.

    The encoded artifact is size-checked BEFORE the store sees it, so the
    store's head/tail clip can never produce a partial artifact. A store that
    cannot write answers ``unavailable`` rather than failing the tool call.
    """
    if not session_id or not call_id:
        return descriptor(STATE_UNAVAILABLE)
    text = _encode(rows, session_id=session_id, call_id=call_id, tool_name=tool_name)
    if len(text.encode("utf-8")) > SPILL_ENTRY_LIMIT_BYTES:
        return descriptor(STATE_OVERSIZED)
    meta = archive_store().write(text, tool_name=tool_name, session_id=session_id)
    if meta is None or not meta.complete:
        return descriptor(STATE_UNAVAILABLE)
    return descriptor(STATE_ARCHIVED, rows=len(rows), digest=meta.digest)


# -- reading ----------------------------------------------------------------


def find_tool_details(session_dir: Path, call_id: str) -> tuple[str, dict[str, Any]] | None:
    """``(tool_name, details)`` of the persisted tool row for ``call_id``.

    The newest matching row wins (the cursor locator's rule). A byte prefilter
    on the JSON-encoded id keeps the walk from parsing unrelated rows; the
    parsed payload is the authority. Read-only: never creates the journal.
    """
    from local_operator.session.transcript import (
        TranscriptEntry,
        _iter_complete_lines_backward,
        _resolved_transcript_path,
    )

    try:
        path = _resolved_transcript_path(session_dir, read_only=True)
    except Exception:  # noqa: BLE001 - an unreadable journal proves no membership
        return None
    if path is None:
        return None
    needle = json.dumps(call_id).encode("utf-8")
    try:
        handle = path.open("rb")
    except OSError:
        return None
    with handle:
        handle.seek(0, os.SEEK_END)
        for _chunk_start, lines in _iter_complete_lines_backward(handle, handle.tell()):
            for raw in lines:
                if needle not in raw:
                    continue
                entry = TranscriptEntry.from_json(raw.decode("utf-8", errors="replace"))
                if entry is None:
                    continue
                payload = entry.payload
                if payload.get("role") != "tool" or payload.get("tool_call_id") != call_id:
                    continue
                provider = payload.get("provider_payload")
                details = provider.get("details") if isinstance(provider, dict) else None
                return str(payload.get("tool_name") or ""), (
                    details if isinstance(details, dict) else {}
                )
    return None


def _json_cost(row: str) -> int:
    return len(json.dumps(row))


def _clip_row(row: str, budget: int) -> str:
    """The longest prefix of ``row`` whose JSON form fits ``budget``.

    Only ever applied to an already-redacted row, so the cut exposes nothing.
    Slicing a ``str`` cuts between code points, so the result is valid UTF-8
    once encoded; the first character (the +/-/space sign) is always kept.
    """
    low, high = 1, len(row)
    while low < high:
        mid = (low + high + 1) // 2
        if _json_cost(row[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1
    return row[:low]


def _hunk_header(rows: Sequence[str], start: int) -> str | None:
    """The ``@@`` header governing ``rows[start]`` when the page starts mid-hunk."""
    if start <= 0 or start >= len(rows) or rows[start].startswith("@@"):
        return None
    for index in range(start - 1, -1, -1):
        if rows[index].startswith("@@"):
            return rows[index][:HUNK_HEADER_MAX_CHARS]
    return None


def _empty(state: str, start: int = 0) -> dict[str, Any]:
    return {
        "state": state,
        "start": start,
        "total_rows": 0,
        "rows": [],
        "hunk_header": None,
        "next_start": None,
        "clipped_rows": [],
    }


def _load_rows(
    desc: dict[str, Any], session_id: str, call_id: str, config_root: Path | None
) -> list[str] | None:
    """The artifact's rows, or ``None`` when evicted/damaged/not this call's."""
    digest = desc.get("digest")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        return None
    store = archive_store(config_root)
    meta = store.stat(digest)
    if meta is None or not meta.complete:
        return None
    text = store._load_text(digest)  # noqa: SLF001 - same-package store
    if text is None:
        return None
    store._touch(digest)  # noqa: SLF001 - a read keeps the entry warm for LRU
    lines = text.split("\n")
    try:
        header = json.loads(lines[0])
        rows = [json.loads(line) for line in lines[1:]] if len(lines) > 1 else []
    except ValueError:
        return None
    if isinstance(header, dict) and (
        header.get("session_id") != session_id or header.get("call_id") != call_id
    ):
        # A digest that belongs to another session or call is not THIS call's
        # archive (a forged or copied descriptor): the unknown-call answer, so
        # a route cannot be used to read another conversation's artifact.
        raise DiffArchiveNotFound(call_id)
    if (
        not isinstance(header, dict)
        or header.get("kind") != "diff-archive"
        or header.get("rows") != len(rows)
        or desc.get("rows") != len(rows)
        or not all(isinstance(row, str) for row in rows)
    ):
        return None
    return rows


def load_page(
    session_dir: Path,
    session_id: str,
    call_id: str,
    *,
    start: int = 0,
    limit: int = PAGE_MAX_ROWS,
    config_root: Path | None = None,
) -> dict[str, Any]:
    """One page of the archived diff of ``call_id`` in ``session_dir``.

    The ONE membership proof every reader (desktop routes, TUI) shares. Raises
    :class:`DiffArchiveNotFound` for anything that is not a persisted
    ``write``/``edit`` row of THIS session, and :class:`DiffPageRangeError` for
    bad paging. Non-archived states answer with no rows -- they are answers,
    not errors.

    ``next_start`` is the row after the last row ACTUALLY returned (the byte
    ceiling can end a page before ``limit``), ``None`` only at the end. A
    single row larger than a whole page is returned clipped, its ABSOLUTE index
    listed in ``clipped_rows``, and the cursor still advances past it.
    """
    if start < 0 or not 1 <= limit <= PAGE_MAX_ROWS:
        raise DiffPageRangeError("start/limit out of range")
    found = find_tool_details(session_dir, call_id)
    if found is None:
        raise DiffArchiveNotFound(call_id)
    tool_name, details = found
    if tool_name not in _DIFF_TOOLS:
        raise DiffArchiveNotFound(call_id)
    desc = details.get(DIFF_FULL_KEY)
    if not isinstance(desc, dict):
        if details.get("diff_truncated_rows"):
            state = STATE_NOT_RECORDED
        else:
            state = STATE_INLINE
        if start:
            raise DiffPageRangeError("start beyond the last row")
        return _empty(state)
    state = desc.get("state")
    if state not in _STORED_STATES:
        state = STATE_UNAVAILABLE
    if state != STATE_ARCHIVED:
        if start:
            raise DiffPageRangeError("start beyond the last row")
        return _empty(state)
    rows = _load_rows(desc, session_id, call_id, config_root)
    if rows is None:
        if start:
            raise DiffPageRangeError("start beyond the last row")
        return _empty(STATE_EVICTED)
    total = len(rows)
    if start >= total and not (start == 0 and total == 0):
        raise DiffPageRangeError("start beyond the last row")
    out: list[str] = []
    clipped: list[int] = []
    spent = 0
    index = start
    while index < total and len(out) < limit:
        row = rows[index]
        cost = _json_cost(row)
        if spent + cost > PAGE_MAX_BYTES:
            if out:
                break
            row = _clip_row(row, PAGE_MAX_BYTES)
            clipped.append(index)
            cost = _json_cost(row)
        out.append(row)
        spent += cost
        index += 1
    return {
        "state": STATE_ARCHIVED,
        "start": start,
        "total_rows": total,
        "rows": out,
        "hunk_header": _hunk_header(rows, start),
        "next_start": index if index < total else None,
        "clipped_rows": clipped,
    }
