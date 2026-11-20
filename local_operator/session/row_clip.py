"""Per-row projection for oversized transcript rows, and the opaque refs that
fetch the full row back in byte ranges.

WHY THIS EXISTS. The desktop renderer refuses any single IPC record over
8 MiB (``8_388_608`` bytes, the UI's per-record cap), and every opening read —
the open frame, the bare ``/history`` page and the in-band resync snapshot —
used to serve journal rows whole. One tool result of 12.5 MB therefore made
the conversation's opening snapshot 12.6 MB and the renderer dropped it, i.e.
the conversation could not be opened at all. The open frame deliberately
always admits its newest row (a page with no rows is a blank transcript, not a
smaller page — see ``open_frame.build_frame``); this module keeps that intent
— the row is still PRESENT, at its own position, with its own id — and only
clips its content to a budget, marking it with a ``truncated`` field so a
reader can say so and fetch the rest on demand.

THE WIRE FIELD (``docs/DESKTOP_API.md`` §"Clipped rows"): a clipped row
carries ``truncated: {bytes_total, shown_bytes, ref, reason}`` —

- ``bytes_total``: the UTF-8 byte length of the row's FULL content, i.e.
  exactly what ``GET .../history/rows/{ref}`` reassembles to
  (``canonical_row_bytes`` of the unclipped desktop wire row);
- ``shown_bytes``: the UTF-8 byte length (``canonical_row_bytes``, the same unit
  as ``bytes_total``) of the row as served, WITHOUT the ``truncated`` key
  itself — so ``shown_bytes / bytes_total`` is a true fraction. The BUDGETS
  are still measured in wire bytes (``wire_size``); UTF-8 is never larger than
  the wire spelling, so ``shown_bytes`` never exceeds the budget either;
- ``ref``: opaque, session-bound (see ``row_ref``); ``None`` only when the
  producer had no session to bind it to (pure helpers in tests);
- ``reason``: ``"open_frame_cap"`` or ``"history_cap"`` — which bound clipped
  it, so a client never has to reverse-engineer which read it came from.

MESSAGE ROWS ARE CLIPPED, OTHER ROWS ARE WITHHELD. A message row's long string
leaves (tool output, a pasted document) are cut to a shared length so the row
keeps its shape and the reader still paints the start of every part. A
``custom`` row (the frontend checkpoint, spend receipts) is NOT clipped: half a
checkpoint is not a checkpoint, and a reader that parsed one would restore a
corrupt state. Such a row keeps its envelope (id, ts, type, and its
``custom_type`` when it has one) with every other payload key removed and
``truncated.omitted: true``, and the full row is one ref fetch away. Most of
these rows are never painted, but the desktop UI DOES paint ``supplement_v1``
custom rows, so a withheld one would otherwise render as an empty block: a
reader that paints a row carrying ``truncated.omitted`` should render its own
placeholder (or fetch the row by ref) instead (``docs/DESKTOP_API.md``).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from typing import Any, Final, Literal, Mapping, Sequence

#: The per-row budget: a clipped row, its ``truncated`` marker included, is at
#: most this many WIRE bytes (``wire_size``). The same figure as
#: ``open_frame.OPEN_FRAME_MAX_BYTES`` — the measured p90 of an unstripped open
#: page — so one clipped row is never larger than a whole ordinary page, and
#: five of them still fit under the UI's 8 MiB record cap.
ROW_CLIP_MAX_BYTES: Final = 1_572_864

#: The bare ``/history`` page's TOTAL budget over its entries. Half the UI's
#: 8 MiB per-record cap, because the same entries ride inside the snapshot
#: frame next to the frontend state, which has its own size; the open frame's
#: page budget (1.5 MiB) is smaller still. Rows past this are left to the
#: next page (``has_more`` is forced true), never cut.
HISTORY_MAX_BYTES: Final = 4 * 1024 * 1024

#: The largest byte range ``GET .../history/rows/{ref}`` serves in one answer.
#: 1 MiB: below the 1.5 MiB row budget so one range plus its JSON envelope is
#: always smaller than any opening read, and a 12.5 MB row (the measured worst
#: case) reassembles in 13 requests. A larger ``limit`` is CLAMPED, not
#: refused, and the answer's ``limit`` states the clamp.
ROW_RANGE_MAX_BYTES: Final = 1_048_576

ClipReason = Literal["open_frame_cap", "history_cap"]


class RowRangeUnsatisfiable(ValueError):
    """``offset`` lies past the end of the row; the route answers 416."""


#: Envelope keys a withheld row keeps; the rest of a non-message payload goes.
_ENVELOPE_KEYS: Final = ("id", "ts", "ts_source", "type", "journal_seq")

#: THE REF KEY IS PER PROCESS, and that is a deliberate trade. A ref is a
#: capability to read one row of one session; a key that never touches disk
#: cannot leak from it, and a ref minted before a serve restart simply stops
#: resolving (the route answers 404 and the reader re-reads ``/history``,
#: which mints fresh refs). Persisting the key would buy refs that survive a
#: restart, which no reader needs.
_REF_KEY: Final = secrets.token_bytes(32)
_REF_PREFIX: Final = "r1"
_MAC_BYTES: Final = 16


def wire_json(value: Any) -> str:
    """The SSE wire spelling of ``value``: the ONE encoding every budget counts.

    WHY THIS IS THE MEASURE (review round 1, R1-1). The snapshot reaches the
    renderer as an SSE ``data:`` record built by exactly this call
    (``routes.desktop_sessions._sse_data`` and the event stream use it, so
    there is no second spelling to drift), and ``json.dumps`` escapes every
    non-ASCII character: a 2-byte UTF-8 character costs 6 bytes on the wire, a
    4-byte one (an emoji, a surrogate pair) 12. Budgets measured in UTF-8 let a
    page of Cyrillic text "bounded" at 4,000,401 B reach 12,000,401 B on the
    wire — over the 8 MiB UI cap the bound exists to respect. The result is
    pure ASCII, so its ``len`` IS its byte count; and since an escape is never
    shorter than its UTF-8 form, a row within a wire budget is within it over
    the plain-JSON HTTP routes too.
    """
    return json.dumps(value, separators=(",", ":"))


def wire_size(row: Any) -> int:
    """Bytes ``row`` occupies inside an SSE record (see :func:`wire_json`)."""
    return len(wire_json(row))


def canonical_row_bytes(row: Mapping[str, Any]) -> bytes:
    """The byte form the rows route serves a FULL row in: compact JSON, UTF-8.

    Deliberately NOT the wire spelling: the ref route ships these bytes as
    base64 ranges (never inside an SSE record), and unescaped UTF-8 is up to 3x
    smaller for non-Latin text, so a long row reassembles in fewer requests.
    ``truncated.bytes_total`` and ``shown_bytes`` count THESE bytes — the exact
    length the joined ranges reach, and one unit for both so a reader's
    "shown N of M" never reports more than the whole — while every budget
    counts wire bytes.
    """
    return json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _mac(session_id: str, row_id: str) -> bytes:
    # NUL-separated so ("a", "b\x00c") and ("a\x00b", "c") cannot collide.
    message = f"{session_id}\x00{row_id}".encode()
    return hmac.new(_REF_KEY, message, hashlib.sha256).digest()[:_MAC_BYTES]


def row_ref(session_id: str, row_id: str) -> str:
    """An opaque ref to ``row_id`` that only ``session_id`` will resolve.

    THREAT MODEL: a ref travels to renderers and, through the mesh, to other
    devices' viewers, so it must not be a capability for anything but this one
    row of this one session. The HMAC binds the pair under a server secret, so
    a ref copied to another session's route fails verification, and a ref for
    a row the holder has never been served cannot be forged (128-bit tag). The
    row id inside it is not secret — every reader of the page already holds it
    — it is carried only so the route need not keep a server-side table.
    """
    return f"{_REF_PREFIX}.{_b64(row_id.encode())}.{_b64(_mac(session_id, row_id))}"


def resolve_ref(session_id: str, ref: str) -> str | None:
    """The row id ``ref`` names for ``session_id``, or ``None`` when it is not ours.

    Every malformed or foreign ref answers the same ``None`` in constant time
    over the tag, so the route cannot be used as an oracle for which part of a
    ref was wrong.
    """
    parts = ref.split(".")
    if len(parts) != 3 or parts[0] != _REF_PREFIX:
        return None
    try:
        row_id = _unb64(parts[1]).decode()
        tag = _unb64(parts[2])
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if not hmac.compare_digest(tag, _mac(session_id, row_id)):
        return None
    return row_id


def _string_leaves(node: Any, out: list[int]) -> None:
    if isinstance(node, str):
        out.append(len(node))
    elif isinstance(node, Mapping):
        for value in node.values():
            _string_leaves(value, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _string_leaves(value, out)


def _cut_strings(node: Any, limit: int) -> Any:
    if isinstance(node, str):
        return node[:limit] if len(node) > limit else node
    if isinstance(node, Mapping):
        return {key: _cut_strings(value, limit) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_cut_strings(value, limit) for value in node]
    return node


def _clip_message(row: Mapping[str, Any], budget: int, size: int) -> dict[str, Any] | None:
    """``row`` with every string leaf cut to one shared length so it fits ``budget``.

    ONE SHARED CEILING rather than "cut the biggest string": a tool row with two
    huge parts keeps the start of both, and small strings (ids, roles, tool
    names) are untouched because they sit below the ceiling. The first guess
    assumes every string encodes at the row's average bytes per character;
    when that is optimistic it shrinks by 10% until the row fits (one or two
    dumps for the rows measured here). ``None`` when even empty strings do not fit —
    the row's bulk is structure, not text, and it is withheld instead.
    """
    lengths: list[int] = []
    _string_leaves(row, lengths)
    if not lengths:
        return None
    lengths.sort(reverse=True)
    total_chars = sum(lengths)
    # The structure's own bytes, measured with every string emptied, and the
    # strings' average encoded bytes per character (escapes, multi-byte text),
    # so the character ceiling below is sized in the budget's unit: bytes.
    fixed = wire_size(_cut_strings(row, 0))
    if fixed >= budget:
        return None
    per_char = max((size - fixed) / max(total_chars, 1), 1.0)
    room = int((budget - fixed) / per_char)
    # The largest ceiling L with sum(min(len, L)) <= room: with the ``count``
    # longest strings at L and the rest whole, the total is ``count * L +
    # rest``; the first ``count`` whose L does not reach the next string's
    # length is the one where exactly those strings are cut.
    limit = 0
    capped = 0
    for count, length in enumerate(lengths, start=1):
        capped += length
        rest = total_chars - capped
        limit = (room - rest) // count
        if count == len(lengths) or limit >= lengths[count]:
            break
    limit = max(limit, 0)
    while True:
        clipped = _cut_strings(row, limit)
        if wire_size(clipped) <= budget:
            return clipped
        if limit == 0:
            return None
        limit = int(limit * 0.9)


def _withhold(row: Mapping[str, Any]) -> dict[str, Any]:
    kept = {key: row[key] for key in _ENVELOPE_KEYS if key in row}
    payload = row.get("payload")
    custom_type = payload.get("custom_type") if isinstance(payload, Mapping) else None
    kept["payload"] = {"custom_type": custom_type} if isinstance(custom_type, str) else {}
    return kept


def clip_row(
    row: Mapping[str, Any],
    *,
    full: Mapping[str, Any] | None = None,
    session_id: str | None,
    reason: ClipReason,
    budget: int = ROW_CLIP_MAX_BYTES,
) -> Mapping[str, Any]:
    """``row`` unchanged when it fits ``budget``, else its clipped projection.

    ``full`` is the UNCLIPPED desktop wire row the ref serves, when ``row`` is
    already a projection of it (the open frame's strip): ``bytes_total`` always
    measures what the ref reassembles to, never an intermediate form.
    """
    size = wire_size(row)
    if size <= budget:
        return row
    total = len(canonical_row_bytes(full if full is not None else row))
    row_id = row.get("id")
    ref = (
        row_ref(session_id, row_id)
        if session_id is not None and isinstance(row_id, str) and row_id
        else None
    )
    # THE MARKER IS INSIDE THE BUDGET (QA round 1, Q1): a projection cut to the
    # whole budget plus its marker came to 1,572,990 B, 126 B over the
    # 1,572,864 B it was bounded by. The reserve is the marker's own wire cost
    # at its widest — ``shown_bytes`` at the budget's digit count, ``omitted``
    # present — less the ``{`` the merge into the row does not add, so the
    # served row, marker included, is <= ``budget``.
    widest = {"bytes_total": total, "shown_bytes": budget, "ref": ref, "reason": reason}
    reserve = wire_size({"truncated": {**widest, "omitted": True}}) - 1
    room = max(budget - reserve, 0)
    clipped = _clip_message(row, room, size) if row.get("type") == "message" else None
    omitted = clipped is None
    projection = _withhold(row) if clipped is None else clipped
    marker: dict[str, Any] = {
        "bytes_total": total,
        # UTF-8, the unit of ``bytes_total`` (R2-1): the wire size of a
        # non-Latin projection is up to 3x its UTF-8 size, which reported a
        # clipped Cyrillic row as showing ~2x MORE bytes than the full row.
        "shown_bytes": len(canonical_row_bytes(projection)),
        "ref": ref,
        "reason": reason,
    }
    if omitted:
        marker["omitted"] = True
    return {**projection, "truncated": marker}


def bound_history_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    session_id: str | None,
    max_bytes: int = HISTORY_MAX_BYTES,
    anchor_id: str | None = None,
) -> tuple[list[Mapping[str, Any]], bool, bool]:
    """A bare ``/history`` page clipped per row and bounded in total (wire bytes).

    Walks NEWEST-FIRST for the same reason the open frame does: the rows a
    reader cannot lose are the recent ones, so a page over budget loses its
    OLDEST rows to the next page and never its tail. The newest row is always
    kept (clipped), so the page is never empty when the journal is not.

    AN ANCHORED PAGE KEEPS ITS ANCHOR (review round 1, R1-5). ``around_id``
    with ``after > 0`` reads rows NEWER than the anchor, and a newest-first walk
    spent the budget on those and could drop the anchor itself — the one row
    the reader asked for. With ``anchor_id`` on the page the window instead
    grows outward from the anchor, older side then newer, one row at a time
    (``open_frame._window_around``'s rule), so the anchor is always present and
    the page stays contiguous around it.

    Returns ``(entries oldest-first, capped_older, capped_newer)``: rows were
    left out on that side, so the caller must answer ``has_more: true`` /
    ``has_newer: true`` respectively.
    """
    rows = [clip_row(raw, session_id=session_id, reason="history_cap") for raw in entries]
    if not rows:
        return [], False, False
    sizes = [wire_size(row) for row in rows]
    anchor = next(
        (pos for pos, row in enumerate(rows) if anchor_id and row.get("id") == anchor_id), None
    )
    if anchor is None:
        # A tail read: grow from the newest row, older only.
        anchor = len(rows) - 1
    lo = hi = anchor
    total = sizes[anchor]
    capped_older = capped_newer = False
    while True:
        grew = False
        if lo > 0 and not capped_older:
            if total + sizes[lo - 1] <= max_bytes:
                lo -= 1
                total += sizes[lo]
                grew = True
            else:
                capped_older = True
        if hi + 1 < len(rows) and not capped_newer:
            if total + sizes[hi + 1] <= max_bytes:
                hi += 1
                total += sizes[hi]
                grew = True
            else:
                capped_newer = True
        if not grew:
            break
    return rows[lo : hi + 1], capped_older, capped_newer


def row_range(content: bytes, *, offset: int, limit: int) -> dict[str, Any]:
    """One byte range of a row's full content, as the rows route answers it.

    ``data`` is base64 (standard alphabet) because a range may split a UTF-8
    character and JSON cannot carry raw bytes; the client concatenates the
    decoded ranges and decodes UTF-8 once. ``limit`` is clamped to
    ``ROW_RANGE_MAX_BYTES`` and the clamp is stated in the answer.
    """
    if offset < 0 or limit < 1:
        raise ValueError("offset must be >= 0 and limit >= 1")
    if offset > len(content):
        # ``offset == len`` is a valid empty, ``eof`` range (a reader that
        # already holds every byte); strictly past it is a reader bug.
        raise RowRangeUnsatisfiable("offset is past the end of the row")
    limit = min(limit, ROW_RANGE_MAX_BYTES)
    chunk = content[offset : offset + limit]
    return {
        "bytes_total": len(content),
        "offset": offset,
        "limit": limit,
        "length": len(chunk),
        "eof": offset + len(chunk) >= len(content),
        "data": base64.b64encode(chunk).decode("ascii"),
    }
