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
from typing import Any, TypedDict, cast

from local_operator.paths import config_dir
from local_operator.tools.spill import SPILL_ENTRY_LIMIT_BYTES, SpillStore

#: ``details`` key of the small descriptor. Never carries diff content.
DIFF_FULL_KEY = "diff_full"


class DiffBindingReadOptions(TypedDict, total=False):
    """The optional ``diff_bindings`` reader keyword, typed for ``**`` forwarding.

    Callers forward the flag as ``**({"diff_bindings": True} if declared else {})``
    so a reader that never declared it keeps its exact pre-flag call shape (the key
    is OMITTED, not passed as ``False``). A bare dict literal there infers
    ``dict[str, bool]``, which pyright then matches against EVERY remaining keyword
    of the callee (``order_ids``, ``before_id``, ...); naming this shape via
    ``cast`` scopes the unpack to the one key. Type-only: callers import it under
    ``TYPE_CHECKING`` and cast with a string, so nothing changes at runtime. Mirrors
    ``supplements.contract.SupplementReadOptions``.
    """

    diff_bindings: bool

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
#: worst case (never below the UTF-8 length), so the bound holds whichever
#: encoder the route uses. The byte ceiling covers the WHOLE response: rows,
#: their separators, the hunk header and :data:`PAGE_ENVELOPE_RESERVE`.
PAGE_MAX_ROWS = 200
PAGE_MAX_BYTES = 120 * 1024

#: Bytes kept for everything in the response that is not a row or the hunk
#: header: the CRUD envelope (``status``/``message``) and the page's own
#: integer/state fields. Their worst case is under 400 bytes (fixed keys,
#: ints of at most 19 digits); 1 KiB leaves margin without costing rows.
PAGE_ENVELOPE_RESERVE = 1024

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

#: Journal ``custom`` entry type binding a NESTED write/edit (``eval`` ->
#: ``tool('write', ...)``) to its archive. A nested call has no ``role:"tool"``
#: row of its own -- only the OUTER ``eval`` row is persisted -- so without
#: this the advertised ``archived`` descriptor led nowhere (QA F5 on #3762:
#: nested id GET 404 in parent and child). The HOST loop appends it, awaited,
#: through ``LoopConfig.bind_nested_diff`` BEFORE the descriptor reaches the
#: nested end event or the eval cell, so a later failed/interrupted outer
#: ``eval`` (or a crash) cannot strand a capture the UI was already offered.
#: Nothing the cell returns, nor the outer row's details, is ever read as a
#: binding: only this host-written entry is, matched on the exact nested id.
#: Custom entries never enter LLM context; unknown types are ignored by every
#: replay/projection reader.
NESTED_DIFF_BINDING_TYPE = "tool_call_diff_binding_v1"


def nested_binding(
    *, session_id: str, outer_call_id: str, call_id: str, tool_name: str, desc: dict[str, Any]
) -> dict[str, Any]:
    """The ``details`` of one :data:`NESTED_DIFF_BINDING_TYPE` entry."""
    return {
        "v": DIFF_ARCHIVE_VERSION,
        "session_id": session_id,
        "outer_call_id": outer_call_id,
        "call_id": call_id,
        "tool": tool_name,
        DIFF_FULL_KEY: dict(desc),
    }


def _binding_details(payload: dict[str, Any], call_id: str, session_id: str | None) -> Any:
    """``(tool, details)`` when ``payload`` is a valid binding for ``call_id``.

    Every field must agree: the exact nested id, THIS session, and an outer id
    that is the nested id's own ``<outer>:`` prefix (the shape the bridge
    mints). The prefix is a consistency check on the host entry, never proof
    on its own -- a request id that merely carries a prefix has no binding.
    """
    if payload.get("custom_type") != NESTED_DIFF_BINDING_TYPE:
        return None
    details = payload.get("details")
    if not isinstance(details, dict) or session_id is None:
        return None
    outer = details.get("outer_call_id")
    if (
        details.get("v") != DIFF_ARCHIVE_VERSION
        or details.get("call_id") != call_id
        or details.get("session_id") != session_id
        or not isinstance(outer, str)
        or not outer
        or not call_id.startswith(outer + ":")
        or not isinstance(details.get(DIFF_FULL_KEY), dict)
    ):
        return None
    return str(details.get("tool") or ""), {DIFF_FULL_KEY: details[DIFF_FULL_KEY]}


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
    # separators included; lone surrogates are refused by ``write_archive``
    # before this runs), so the reader
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
    try:
        for row in rows:
            row.encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate (a file decoded with ``surrogateescape``) cannot be
        # served as UTF-8 JSON, and the pager must never alter a stored
        # character to make it fit: refuse whole, honestly.
        return descriptor(STATE_WITHHELD)
    text = _encode(rows, session_id=session_id, call_id=call_id, tool_name=tool_name)
    if len(text.encode("utf-8")) > SPILL_ENTRY_LIMIT_BYTES:
        return descriptor(STATE_OVERSIZED)
    meta = archive_store().write(text, tool_name=tool_name, session_id=session_id)
    if meta is None or not meta.complete:
        return descriptor(STATE_UNAVAILABLE)
    return descriptor(STATE_ARCHIVED, rows=len(rows), digest=meta.digest)


# -- reading ----------------------------------------------------------------


class AmbiguousToolCall(LookupError):
    """Two or more DISTINCT persisted tool-result rows carry the same call id."""


def find_tool_details(
    session_dir: Path, call_id: str, session_id: str | None = None
) -> tuple[str, dict[str, Any]] | None:
    """``(tool_name, details)`` of THE persisted tool row for ``call_id``.

    Call ids are provider-minted and nothing proves them unique within one
    session (a retried turn, a second prompt, a provider that reuses ids), so
    "newest wins" could hand a reader the WRONG historical capture. The walk
    therefore fails closed: a second distinct persisted row with
    the same ``role:"tool"`` + ``tool_call_id`` raises
    :class:`AmbiguousToolCall` and it stops there, so memory stays bounded
    (chunked backward reads, at most two parsed matches). The same entry
    written twice is one row -- but only when the two lines are BYTE-IDENTICAL:
    the same entry id with different content (other details, another digest)
    is two captures as far as a reader can prove, and fails closed like two
    different ids. A byte prefilter on the JSON-encoded id keeps
    the walk from parsing unrelated rows; the parsed payload is the authority.
    Read-only: never creates the journal.

    With ``session_id``, a host-written :data:`NESTED_DIFF_BINDING_TYPE` entry
    for ``call_id`` is a match too, under the SAME duplicate rule: an ordinary
    tool row and a binding (or two bindings) for one id are two captures and
    fail closed -- the walk always reaches the journal start, so neither kind
    can shadow the other.
    """
    from local_operator.session.transcript import (
        ENTRY_CUSTOM,
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
    found: tuple[str, dict[str, Any]] | None = None
    found_raw: bytes | None = None
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
                if entry.type == ENTRY_CUSTOM:
                    match = _binding_details(payload, call_id, session_id)
                    if match is None:
                        continue
                elif payload.get("role") != "tool" or payload.get("tool_call_id") != call_id:
                    continue
                else:
                    provider = payload.get("provider_payload")
                    details = provider.get("details") if isinstance(provider, dict) else None
                    match = (
                        str(payload.get("tool_name") or ""),
                        details if isinstance(details, dict) else {},
                    )
                if found is not None:
                    if raw.rstrip(b"\r\n") == found_raw:
                        continue
                    raise AmbiguousToolCall(call_id)
                found = match
                found_raw = raw.rstrip(b"\r\n")
    return found


def _json_cost(row: str) -> int:
    return len(json.dumps(row))


def _fragment_cost(marker: str, body: str, offset: int, length: int) -> int:
    """Bytes one returned fragment costs in the rows array (``,`` included)."""
    return _json_cost(marker + body[offset : offset + length]) + 1


def _fit_length(marker: str, body: str, offset: int, budget: int) -> int:
    """Most body CODE POINTS from ``offset`` whose fragment fits ``budget``.

    Monotone in the length, so a binary search finds it. Slicing a ``str`` cuts
    between code points, so a fragment is always valid UTF-8 (a combining mark
    may land in the next fragment; the client re-joins fragments before
    rendering, so nothing is lost). Only ever applied to already-redacted rows.
    """
    low, high = 0, len(body) - offset
    while low < high:
        mid = (low + high + 1) // 2
        if _fragment_cost(marker, body, offset, mid) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def _hunk_header(rows: Sequence[str], start: int) -> str | None:
    """The ``@@`` header governing ``rows[start]`` when the page starts mid-hunk."""
    if start <= 0 or start >= len(rows) or rows[start].startswith("@@"):
        return None
    for index in range(start - 1, -1, -1):
        if rows[index].startswith("@@"):
            return rows[index][:HUNK_HEADER_MAX_CHARS]
    return None


def _empty(state: str, start: int = 0, offset: int = 0) -> dict[str, Any]:
    return {
        "state": state,
        "start": start,
        "offset": offset,
        "total_rows": 0,
        "rows": [],
        "hunk_header": None,
        "next_start": None,
        "next_offset": None,
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
    offset: int = 0,
    config_root: Path | None = None,
) -> dict[str, Any]:
    """One page of the archived diff of ``call_id`` in ``session_dir``.

    The ONE membership proof every reader (desktop routes, TUI) shares. Raises
    :class:`DiffArchiveNotFound` for anything that is not a persisted
    ``write``/``edit`` row of THIS session, and :class:`DiffPageRangeError` for
    bad paging. Non-archived states answer with no rows -- they are answers,
    not errors. Two distinct persisted rows for one call id answer
    ``unavailable``: serving either could be the wrong historical capture.

    CONTINUATION CURSOR (v1). A row is ``marker + body`` (``marker`` is its
    first character: ``+``/``-``/space, or the ``-``/``+``/``@`` of a header).
    ``offset`` is a CODE-POINT offset into the BODY of row ``start`` (server
    issued; a client never computes it). Every returned fragment starts with
    its row's original marker followed by an exact body chunk, so a row too
    large for one page is served across pages and nothing is ever discarded:
    the client re-joins fragments of the same row (the first whole, later ones
    minus their marker). When the last fragment is partial, ``next_start`` is
    its row and ``next_offset`` > 0; when it completed, ``next_start`` is the
    next row and ``next_offset`` is 0; both are ``None`` only at the end.
    ``clipped_rows`` stays for wire compatibility and is always ``[]``.
    """
    if start < 0 or offset < 0 or not 1 <= limit <= PAGE_MAX_ROWS:
        raise DiffPageRangeError("start/limit/offset out of range")
    try:
        found = find_tool_details(session_dir, call_id, session_id)
    except AmbiguousToolCall:
        found = None
        ambiguous = True
    else:
        ambiguous = False
    if ambiguous:
        if start or offset:
            raise DiffPageRangeError("start beyond the last row")
        return _empty(STATE_UNAVAILABLE)
    if found is None:
        raise DiffArchiveNotFound(call_id)
    tool_name, details = found
    if tool_name not in _DIFF_TOOLS:
        raise DiffArchiveNotFound(call_id)
    desc = details.get(DIFF_FULL_KEY)
    if not isinstance(desc, dict):
        state = STATE_NOT_RECORDED if details.get("diff_truncated_rows") else STATE_INLINE
    else:
        state = desc.get("state")
        if state not in _STORED_STATES:
            state = STATE_UNAVAILABLE
    # ``STATE_ARCHIVED`` is reachable only from the ``isinstance(desc, dict)``
    # branch above, so the cast restates that proof for the checker; no new check.
    rows = (
        _load_rows(cast("dict[str, Any]", desc), session_id, call_id, config_root)
        if state == STATE_ARCHIVED
        else None
    )
    if state == STATE_ARCHIVED and rows is None:
        state = STATE_EVICTED
    if rows is None:
        if start or offset:
            raise DiffPageRangeError("start beyond the last row")
        return _empty(state)
    total = len(rows)
    if total == 0:
        if start or offset:
            raise DiffPageRangeError("start beyond the last row")
        return {**_empty(STATE_ARCHIVED), "total_rows": 0}
    if start >= total:
        raise DiffPageRangeError("start beyond the last row")
    if offset and offset >= len(rows[start][1:]):
        # An offset is only ever issued INSIDE a body; one at or past its end
        # is not a cursor this server handed out.
        raise DiffPageRangeError("offset beyond the row body")
    header = _hunk_header(rows, start)
    budget = PAGE_MAX_BYTES - PAGE_ENVELOPE_RESERVE - (_json_cost(header) if header else 4)
    out: list[str] = []
    spent = 0
    index, at = start, offset
    next_offset = 0
    while index < total and len(out) < limit:
        row = rows[index]
        marker, body = row[:1], row[1:]
        remaining = len(body) - at
        if _fragment_cost(marker, body, at, remaining) <= budget - spent:
            out.append(marker + body[at:])
            spent += _fragment_cost(marker, body, at, remaining)
            index, at = index + 1, 0
            continue
        take = _fit_length(marker, body, at, budget - spent)
        if take > 0:
            # Partial: always forward progress (``take`` >= 1 code point).
            out.append(marker + body[at : at + take])
            next_offset = at + take
        elif not out:
            # Unreachable with the fixed budgets (a marker plus one escaped
            # code point is <= 14 bytes); refuse rather than loop forever.
            raise DiffPageRangeError("page budget cannot hold one fragment")
        break
    if next_offset:
        next_start: int | None = index
    elif index < total:
        next_start = index
    else:
        next_start = None
    return {
        "state": STATE_ARCHIVED,
        "start": start,
        "offset": offset,
        "total_rows": total,
        "rows": out,
        "hunk_header": header,
        "next_start": next_start,
        "next_offset": next_offset if next_start is not None else None,
        "clipped_rows": [],
    }
