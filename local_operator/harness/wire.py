"""Bound the ``agent_end`` conversation frame at the wire encoders.

WHY THIS EXISTS
---------------
``AgentEndEvent.messages`` is the WHOLE turn — every message the run produced —
and it leaves as one frame. The same tool bytes already crossed the wire during
the turn as ``tool_execution_end`` events, so the aggregate ships them a second
time: measured on a 104-message conversation with 50 tool rows, the frame is
531,082 bytes and 88.7% of it is tool rows.

A frame that large is not merely wasteful on the path this module was written
for. ``lop exec --json`` writes one NDJSON line per event, and an external
supervisor reads that stream with ``bufio.Scanner`` bounded at 4 MiB
(Minerva's ``adapters/lopcli``, ``adapter.go``), returning ``scanner.Err()`` as
a RUN FAILURE. Five pasted screenshots measure a 4,667,340-byte line: the run
dies. And of the four size controls the harness already carries, none sees this
frame — the socket fitter is socket-only and caps at 1 MiB (it returns the
531 KB frame byte-identical), the SSE cap is per STRING at 16 KiB (a turn of
ordinary rows has no string over the limit and passes through untouched at
377,831 bytes), the live-event text budget bounds retained
``tool_execution_end`` rows rather than this frame, and attachment
externalisation is image-only.

WHY IT LIVES AT THE ENCODERS RATHER THAN IN THE LOOP
----------------------------------------------------
The message objects in ``AgentEndEvent.messages`` are the SAME objects the
session persists, the scheduler ledger projects into job records, and the
attention classifier filters. Bounding them at construction would either
destroy the system of record or oblige four consumers to rehydrate. There are
exactly three encoders that turn an event into viewer bytes, so the bound is
applied at each of them, on a COPY, and the loop's objects are never touched:

- :func:`local_operator.headless_print.printable_event` — the NDJSON line
  (``lop exec --json``, supervisors; the unbounded path in the incident).
- :meth:`local_operator.server.utils.operator.AgentEventBridge._raw` — the SSE
  frame, ahead of that transport's per-string cap so the two compose.
- :meth:`local_operator.session.runtime.serving.ServingSession.subscribe_events`
  — the socket relay's ``{"op": "event", ...}`` payload.

The contract, deliberately shaped like
:func:`local_operator.session.runtime.server.fit_frame_for_wire` so reviewers
recognise it: identity when the frame already fits, a documented budget, honest
stages ordered cheapest-first, and a never-raise posture.

WHAT A READER LOSES, AND HOW IT GETS IT BACK
--------------------------------------------
Tool-row CONTENT is what this bound spends: every ``usage`` receipt and all
assistant/user text survive verbatim, and so does every message's identity,
role and outcome. A row whose content is elided carries an honest marker naming
the durable transcript entry that holds it — the system of record, and the
reconstruction path every viewer already uses. The transcript keeps everything;
nothing here is a durable write.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

logger = logging.getLogger("local_operator.harness.wire")

#: Budget for one serialised ``agent_end`` frame, in bytes.
#:
#: 256 KiB is chosen against the readers that exist, not for roundness:
#:
#: - It is 4x the live-event text budget (``LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS``
#:   = 56,000 chars) that already bounds one retained tool row, so a frame
#:   bounded here is the same order of magnitude as the one the reconnect seed
#:   already ships per turn.
#: - It is 4x under the socket's own 1 MiB line (``_MAX_LINE_BYTES``), so the
#:   relay's terminal ``relay_frame_or_degraded`` stage should stop firing for
#:   conversation frames entirely.
#: - It is ~18x under the 4 MiB ``bufio.Scanner`` buffer an external supervisor
#:   reads ``lop exec --json`` with, which is the reader whose failure this
#:   bound exists to prevent — and it leaves that reader three orders of
#:   magnitude of headroom for the escape inflation a JSON re-encode adds.
#:
#: Measured at this value on the 104-message fixture: 531,082 -> 236,614 bytes,
#: every one of the 50 receipts kept and every row still carrying a ~4 KiB
#: preview. A tighter budget is one constant, but nothing here needs it: below
#: this, the preview a reader gets per row stops being legible before the
#: frame stops being large.
AGENT_END_FRAME_BUDGET_BYTES = 256 * 1024

#: How many of the NEWEST tool rows may keep a preview.
#:
#: The share below bounds what one row COSTS; it does not bound how many rows
#: there are, and "bounded per row, unbounded in total" is the defect the
#: budget exists to answer — a turn that ran 5,000 tool calls would spend the
#: share 5,000 times. Mirrors ``LIVE_EVENT_EVENT_END_ROWS_MAX``'s reason for
#: existing (``session/frontend_state.py``) and its ordering: the newest rows
#: are the ones a viewer is most likely still looking at, and a row beyond the
#: cap loses its content but never its identity, outcome or receipt.
AGENT_END_PREVIEW_ROWS_MAX = 100

#: One elided tool row's whole content. Three facts, because a reader has to be
#: able to tell three things apart: that something was cut (elided), how much
#: (the character count of what is gone), and where the real text is (the
#: durable transcript entry — an id ``transcript.read_transcript_page``
#: resolves, plus the session whose transcript holds it). The session clause is
#: dropped rather than guessed when an encoder does not know it.
_ELIDED_ROW_MARKER = (
    "[tool output elided from this event: {chars:,} chars · {tool}"
    " · full text: transcript entry {entry_id}{session}]"
)

#: Stand-in for a block the clip cannot keep whole (an image's base64, or any
#: future payload under a key this module has never heard of) inside a row that
#: otherwise keeps its text. A block that VANISHES reads as a tool that returned
#: nothing; this keeps the block's shape and says what happened. Deliberately
#: neutral about text vs media, because both reach it.
#:
#: Not the live-event module's own placeholder: that one says "dropped from the
#: reconnect snapshot", which is a different frame with a different reader.
_ELIDED_BLOCK_MARKER = "[dropped from this event — see the transcript]"


def _frame_bytes(payload: Any) -> int:
    """Encoded size of a payload, which is the unit the budget is stated in.

    ``ensure_ascii=False`` matches the NDJSON writer, which is the encoder this
    budget was set for; the socket and SSE writers escape non-ASCII, so their
    line can exceed this reading on a frame full of emoji or CJK. That is a
    known, documented divergence rather than a hole: those transports keep
    their own hard caps (1 MiB line, 16 KiB per string), and the alternative —
    measuring every frame as if it were all escapes — would elide frames the
    primary reader was never at risk on.
    """
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))


def bound_agent_end_for_wire(
    payload: dict[str, Any],
    *,
    session_id: str | None = None,
    cap_bytes: int = AGENT_END_FRAME_BUDGET_BYTES,
) -> dict[str, Any]:
    """Return ``payload`` bounded for the wire, or ``payload`` itself.

    ``payload`` is a JSON dump of an ``AgentEvent`` (``model_dump(mode="json")``,
    optionally with ``provider_payload`` already stripped); the return value is
    a payload the caller serialises as it always did.

    The contract, in order:

    1. Not an ``agent_end`` (or nothing to bound) -> returned by IDENTITY. The
       common path costs one ``json.dumps`` — which the caller was doing anyway
       — and allocates nothing beyond it.
    2. Fits ``cap_bytes`` -> returned by identity, for the same reason.
    3. Otherwise tool rows are elided on a deep copy of THE ROWS ONLY (the
       frame's other messages are shared by reference and never written to):
       the residual budget — ``cap_bytes`` minus the frame with every tool row
       emptied — is divided by the row count, floored at the live-event text
       floor, and spent through the repo's own
       ``frontend_state._bound_live_result_in_place`` so a row keeps the
       distinct, reviewed preview/clip behaviour the live seed already ships.
       Rows past ``AGENT_END_PREVIEW_ROWS_MAX`` newest lose their content in
       whole and carry the marker instead.
    4. Still over budget (assistant text alone, a compaction payload, a giant
       ``tool_calls`` argument list) -> the same clip is applied to non-tool
       text blocks, and only then are oversized strings truncated with the SSE
       transport's own ``STREAM_TRUNCATION_MARKER``. Reaching this stage is
       logged, with the size, so the next offender is named rather than guessed.
    5. Never raises and never mutates its argument: any failure returns the
       input unchanged. Fail-OPEN is the right direction for a size bound —
       ``transcript._externalize_attachments`` and
       ``server._reference_image_payloads`` both keep the inline payload when
       their own store refuses; fail-closed belongs to redaction, not to a
       frame budget.

    Two additive scalars are set on the bounded payload — ``elided_tool_rows``
    (rows whose content was elided in whole) and ``elided_bytes`` (what the
    bound removed, computed LAST so it cannot understate what a later stage
    took). They are payload keys rather than model fields: ``AgentEvent`` is
    ``extra="allow"``, so no protocol version changes and older and newer
    readers both keep working.

    ``session_id``, when the encoder knows it, is what makes the marker's
    reference resolvable — the transcript is per session. An encoder that does
    not know it degrades to naming the entry id alone; the elision is still
    reported honestly and the full text is still in the transcript.
    """
    try:
        if not isinstance(payload, dict) or payload.get("type") != "agent_end":
            return payload
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return payload
        total = _frame_bytes(payload)
        if total <= cap_bytes:
            return payload
        return _bound_agent_end(
            payload, messages, session_id=session_id, cap_bytes=cap_bytes, total=total
        )
    except Exception:  # noqa: BLE001 — a size bound must never fail a turn
        logger.warning("could not bound an agent_end frame; shipping it unbounded", exc_info=True)
        return payload


def _bound_agent_end(
    payload: dict[str, Any],
    messages: list[Any],
    *,
    session_id: str | None,
    cap_bytes: int,
    total: int,
) -> dict[str, Any]:
    """Apply the elision stages to a frame that is already over budget."""
    # Function-local, twice over, and both for the same reason: the caller is a
    # wire encoder on the CLI's own import path, and importing either at module
    # scope would put a 760 ms module on `lop exec`'s startup and on
    # `test_import_graph`'s pinned CLI closure. Neither is needed unless a
    # frame is actually over budget, which is the rare turn.
    from local_operator.session.frontend_state import (
        LIVE_EVENT_TEXT_FLOOR_CHARS,
        _bound_live_result_in_place,
    )

    # Only modified rows are copied, one deep copy each: a full deep copy of a
    # 531 KB frame is a second pass over exactly the bytes this function was
    # called to stop paying for.
    frame = dict(payload)
    bounded_messages = list(messages)
    frame["messages"] = bounded_messages

    row_positions = [
        index
        for index, message in enumerate(bounded_messages)
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    elided_rows = 0
    if row_positions:
        elided_rows = _elide_tool_rows(
            frame,
            bounded_messages,
            row_positions,
            session_id=session_id,
            cap_bytes=cap_bytes,
            floor_chars=LIVE_EVENT_TEXT_FLOOR_CHARS,
            bound_row=_bound_live_result_in_place,
        )

    if _frame_bytes(frame) > cap_bytes:
        _clip_non_tool_rows(
            frame,
            bounded_messages,
            cap_bytes=cap_bytes,
            floor_chars=LIVE_EVENT_TEXT_FLOOR_CHARS,
            bound_row=_bound_live_result_in_place,
        )
    if _frame_bytes(frame) > cap_bytes:
        _truncate_oversized_strings(frame, cap_bytes, total=total)

    frame["elided_tool_rows"] = elided_rows
    frame["elided_bytes"] = max(0, total - _frame_bytes(frame))
    return frame


def _spendable_share(
    frame: dict[str, Any],
    messages: list[Any],
    positions: list[int],
    *,
    cap_bytes: int,
    floor_chars: int,
) -> int:
    """Budget one stage may spend PER ROW, in characters.

    Measured, not assumed: the residual is the frame as this stage will LEAVE
    it with no content kept — every block still present, carrying only the clip
    marker — so the share is what is actually left once every other message
    (receipts, assistant text, identity) and the stage's own residue have been
    paid for. A frame that is over budget for reasons this stage cannot touch
    yields a floor-level share rather than a negative one.

    Counting the emptied blocks rather than deleting them is the difference
    between a share that fits and one that overshoots by the block count — the
    floor x N trap ``frontend_state`` documents for the live seed, which a row
    of 1,000 blocks reaches here. Measured before this: a 1,000-block row
    bounded to 293,199 B against a 262,144 B budget.
    """
    emptied = list(messages)
    for index in positions:
        row = messages[index]
        emptied[index] = {
            key: value for key, value in row.items() if key not in ("content", "provider_payload")
        }
        emptied[index]["content"] = [
            {"type": "text", "text": "…"}
            for block in (row.get("content") or [])
            if isinstance(block, dict)
        ]
    residual = cap_bytes - _frame_bytes({**frame, "messages": emptied})
    return max(floor_chars, residual // max(1, len(positions)))


def _elide_tool_rows(
    frame: dict[str, Any],
    messages: list[Any],
    positions: list[int],
    *,
    session_id: str | None,
    cap_bytes: int,
    floor_chars: int,
    bound_row: Any,
) -> int:
    """Spend the tool-row share, returning how many rows were elided in whole.

    The NEWEST ``AGENT_END_PREVIEW_ROWS_MAX`` rows keep a preview and the older
    ones are elided in whole — the same ordering, for the same reason, as the
    live-event row cap this module mirrors. The elided rows are written FIRST
    because their markers are part of the frame the preview share is measured
    against: at ~150 bytes each, 2,000 of them are most of a 256 KiB budget,
    and a share computed without them would hand the previews bytes that are
    already spent.
    """
    preview_count = max(0, len(positions) - AGENT_END_PREVIEW_ROWS_MAX)
    elided = 0
    for index in positions[:preview_count]:
        row = copy.deepcopy(messages[index])
        messages[index] = row
        chars = _text_chars(row)
        row["content"] = [
            {
                "type": "text",
                "text": _ELIDED_ROW_MARKER.format(
                    chars=chars,
                    tool=_tool_label(row),
                    session=f" of session {session_id}" if session_id else "",
                    entry_id=str(row.get("id") or "unknown"),
                ),
            }
        ]
        # The provider-native replay payload is dead weight once the row has
        # nothing left to replay, and the harness's own bookkeeping rides
        # inside it under ``details`` (``harness/types.py``), so one pop
        # takes both.
        row.pop("provider_payload", None)
        elided += 1

    previews = positions[preview_count:]
    if not previews:
        return elided
    share = _spendable_share(
        frame, messages, previews, cap_bytes=cap_bytes, floor_chars=floor_chars
    )
    for index in previews:
        row = copy.deepcopy(messages[index])
        messages[index] = row
        # The bound reads ``details`` off the row itself; the harness keeps it
        # under ``provider_payload["details"]`` (see ``harness/types.py``), so
        # it is moved in for the call and moved back after — on the copy only.
        provider_payload = row.get("provider_payload")
        moved_details = isinstance(provider_payload, dict) and "details" in provider_payload
        if moved_details:
            row["details"] = provider_payload.get("details")
        bound_row(row, share=share, placeholder=_ELIDED_BLOCK_MARKER)
        if moved_details:
            provider_payload["details"] = row.pop("details", None)
    return elided


def _clip_non_tool_rows(
    frame: dict[str, Any],
    messages: list[Any],
    *,
    cap_bytes: int,
    floor_chars: int,
    bound_row: Any,
) -> None:
    """Stage 4a: the same clip for text that is not a tool row's.

    Reached only when tool rows alone could not bring the frame under budget —
    a giant user paste, or an assistant message carrying the turn's prose. Text
    is clipped, never dropped: this stage must not be what makes a message's
    identity disappear from a frame a viewer is folding.
    """
    positions = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict)
        and message.get("role") != "tool"
        and isinstance(message.get("content"), list)
        and message["content"]
        and _frame_bytes(message) > floor_chars
    ]
    if not positions:
        return
    share = _spendable_share(
        frame, messages, positions, cap_bytes=cap_bytes, floor_chars=floor_chars
    )
    for index in positions:
        row = copy.deepcopy(messages[index])
        messages[index] = row
        bound_row(row, share=share, placeholder=_ELIDED_BLOCK_MARKER)


def _truncate_oversized_strings(frame: dict[str, Any], cap_bytes: int, *, total: int) -> None:
    """Stage 4b: the SSE transport's own string clip, as the last resort.

    For the payloads neither stage above can reach — a ``tool_calls`` argument
    list, a compaction ``CustomMessage`` — this clips any string longer than
    the SSE transport's per-string ceiling and says so with that transport's
    own marker, so a reader sees one truncation vocabulary rather than two.

    ``usage`` sub-trees are never descended into. A receipt is the billing
    record: every consumer reconciles per-call cost off these objects, and a
    receipt silently clipped here would be a wrong bill rather than a lost
    preview.
    """
    # Function-local: ``server.utils.operator`` is the heavy legacy adapter
    # module (yaml/dotenv/pickle) and this stage is the rare one. Re-spelling
    # its two constants here is what the import avoids.
    from local_operator.server.utils.operator import (
        STREAM_TRUNCATION_MARKER,
        STREAM_VALUE_LIMIT,
    )

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            if len(value) > STREAM_VALUE_LIMIT:
                return value[:STREAM_VALUE_LIMIT] + STREAM_TRUNCATION_MARKER
            return value
        if isinstance(value, dict):
            return {key: item if key == "usage" else walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    logger.warning(
        "agent_end frame still oversized after row elision; truncating oversized strings "
        "in a %d-byte frame against a %d-byte budget",
        total,
        cap_bytes,
    )
    truncated = walk(frame)
    frame.clear()
    frame.update(truncated)


def _text_chars(row: dict[str, Any]) -> int:
    """Characters of text the row's content blocks carry."""
    return sum(
        len(block["text"])
        for block in (row.get("content") or [])
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def _tool_label(row: dict[str, Any]) -> str:
    """The tool's name, for a marker a human reads."""
    name = row.get("tool_name")
    return str(name) if isinstance(name, str) and name else "tool"


__all__ = [
    "AGENT_END_FRAME_BUDGET_BYTES",
    "AGENT_END_PREVIEW_ROWS_MAX",
    "bound_agent_end_for_wire",
]
