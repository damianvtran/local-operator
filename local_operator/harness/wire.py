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
#: - It is 4x under the socket's own 1 MiB line (``_MAX_LINE_BYTES``), which
#:   covers every turn of ordinary size. This budget bounds CONTENT, and a
#:   frame's SKELETON — one entry per message with its id, role and
#:   ``tool_call_id`` — is not content-elidable, so a many-row turn used to
#:   cross the line regardless: measured on 500-char results, 1,400 rows
#:   (2,801 messages) made a 1,043,682 B line and fit, while 1,500 rows
#:   (3,001 messages) made 1,117,282 B and did not. Since ``AGENT_END_MESSAGE_TAIL_MAX``
#:   that skeleton is bounded too, so the terminal ``relay_frame_or_degraded``
#:   stage no longer fires for a conversation frame of any depth.
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

#: How many of the NEWEST messages survive the LAST-RESORT tail stage.
#:
#: WHY this exists at all: the per-row stages bound what one message COSTS, and
#: ``AGENT_END_PREVIEW_ROWS_MAX`` bounds how many TOOL rows keep a preview — but
#: neither bounds the message SKELETON, and the skeleton is what made a deep
#: turn cross the socket line (the measurement in the budget docstring above).
#: That frame was then replaced wholesale by a passive notice, so the viewer
#: lost the whole ``agent_end`` — the turn never settled, and its composer
#: waited forever on an acknowledgement carried by the same stream.
#:
#: 24 is "a screenful of the newest conversation": the live viewer already
#: streamed every row as it arrived, and the durable transcript holds the whole
#: turn, so the tail only has to keep what a reader is looking at when the end
#: lands. The newest-first ordering is the one ``AGENT_END_PREVIEW_ROWS_MAX``
#: already uses, for the same reason.
AGENT_END_MESSAGE_TAIL_MAX = 24

#: One elided tool row's whole content. Three facts, because a reader has to be
#: able to tell three things apart: that something was cut (elided), how much
#: (the character count of what is gone), and where the real text is. The
#: destination is filled in by :func:`_elided_row_marker`, which states a
#: reference only when it can be resolved.
_ELIDED_ROW_MARKER = "[tool output elided from this event: {chars:,} chars · {tool} · {reference}]"

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
    5. Still over budget -> the message SKELETON is the remaining cost, because
       no stage above can bound the number of messages. ``messages`` is replaced
       with the newest ``AGENT_END_MESSAGE_TAIL_MAX`` rows, and TWO additive
       integer scalars say so — ``messages_total`` and ``messages_omitted`` —
       plus ``usage_aggregate`` (an OBJECT, present only when an omitted row
       carried a usage receipt): the folded receipts of everything dropped (the
       TUI sums ``event.messages`` for its cost band, so dropping rows without
       it would undercount the turn). The two elision counters are then
       recomputed against the RETAINED rows only: a row that is no longer in the
       frame is reported by ``messages_omitted``, not as a row whose content was
       cut.
    6. Never raises and never mutates its argument: any failure returns the
       input unchanged. Fail-OPEN is the right direction for a size bound —
       ``transcript._externalize_attachments`` and
       ``server._reference_image_payloads`` both keep the inline payload when
       their own store refuses; fail-closed belongs to redaction, not to a
       frame budget.

    SIX additive payload keys describe a bounded frame, in two groups, and each
    counts a DIFFERENT thing — which is why the row counters are two numbers and
    not one:

    Always set (the content stages):
    * ``elided_tool_rows`` — tool rows elided IN WHOLE that are STILL IN the
      frame, each carrying the marker naming its transcript entry;
    * ``clipped_tool_rows`` — retained tool rows that kept a preview and lost
      part of their content (text clipped to ``…``, or a block replaced by the
      dropped-block marker);
    * ``elided_bytes`` — the byte delta between the input frame and this one,
      computed LAST so it cannot understate what an earlier stage took. It
      INCLUDES rows the tail stage dropped WHOLE, which neither row counter
      counts: a dropped row is reported by ``messages_omitted``. So the row
      counters answer "what can I see was cut", and this one answers "how much
      is gone from the conversation".

    Set only by the tail stage, which is the only stage that changes the
    message COUNT:
    * ``messages_total`` / ``messages_omitted`` — integer scalars;
    * ``usage_aggregate`` — an object, present only when an omitted row carried
      a usage receipt.

    The row counters are two numbers rather than one because a clipped row lost
    content just as surely as an elided one, and a single counter named for
    either left a consumer reading ``0`` next to a multi-kilobyte
    ``elided_bytes`` — the one reading that must not happen. Before the tail
    stage existed, ``elided_bytes`` and the row counters described the same
    messages; the distinction is stated explicitly here because they no longer
    do. They are payload keys rather than model fields: ``AgentEvent`` is
    ``extra="allow"``, so no protocol version changes and older and newer
    readers both keep working.

    ``elided_bytes`` is this bound's own arithmetic, measured against the
    payload it was handed. A transport that clips afterwards is not counted —
    the SSE encoder caps a single STRING at ``STREAM_VALUE_LIMIT`` (16 KiB)
    and does so with its own marker, so a frame whose surviving string the
    bound left above that limit loses more than this number says. The bound
    cannot know what an encoder will do next, and inventing a number for it
    would be the worse lie; the SSE path is where the two meet, and
    ``AgentEventBridge._raw`` states the ordering.

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
    # The two scalars are set at the END, but they are part of the frame the
    # stages below are measured against and must be counted from the start:
    # leaving them out is how a one-row clip landed 19 bytes over the budget
    # (262,163 against 262,144) with the payload's own report of what it had
    # removed as the only thing over it.
    frame["elided_tool_rows"] = 0
    frame["clipped_tool_rows"] = 0
    frame["elided_bytes"] = 0

    row_positions = [
        index
        for index, message in enumerate(bounded_messages)
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    elided_rows = 0
    clipped_rows = 0
    elided_positions: frozenset[int] = frozenset()
    clipped_positions: frozenset[int] = frozenset()
    if row_positions:
        elided_rows, clipped_rows, elided_positions, clipped_positions = _elide_tool_rows(
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
    if _frame_bytes(frame) > cap_bytes:
        # Read the list from the FRAME, not ``bounded_messages``:
        # ``_truncate_oversized_strings`` rebuilds it on its own walk, so the
        # local name above is stale by this point. The tail stage is the only
        # one that changes the message COUNT, and it reports what it removed
        # in scalars of its own instead of leaving a smaller list unexplained.
        kept, total_count = _retain_newest_messages(frame, frame["messages"], cap_bytes=cap_bytes)
        if len(kept) < total_count:
            # The counters describe CONTENT CUT FROM THE FRAME, and a dropped row
            # is no longer in the frame at all — it is reported by
            # ``messages_omitted``. Keeping a dropped row's count here would
            # name a row the reader cannot find, which is the same dishonesty
            # the honest markers exist to prevent.
            first_retained = total_count - len(kept)
            elided_rows = sum(1 for index in elided_positions if index >= first_retained)
            clipped_rows = sum(1 for index in clipped_positions if index >= first_retained)

    frame["elided_tool_rows"] = elided_rows
    frame["clipped_tool_rows"] = clipped_rows
    frame["elided_bytes"] = max(0, total - _frame_bytes(frame))
    return frame


def _retain_newest_messages(
    frame: dict[str, Any], messages: list[Any], *, cap_bytes: int
) -> tuple[list[Any], int]:
    """Keep the newest ``AGENT_END_MESSAGE_TAIL_MAX`` messages and say so.

    RETURNS ``(kept, total)``. The frame is rewritten in place with the tail and
    the additive scalars, and with ``usage_aggregate`` when any dropped row
    carried a receipt.

    WHY IT HALVES RATHER THAN SHIPPING 24 BLIND. 24 rows of ORDINARY content is
    far under a quarter of the budget, but the newest row is exactly the one a
    long turn ends on — often a compaction summary or a large final assistant
    block — and a synthetic call can put anything there. Halving gives at most
    five re-measurements on a path that only runs when a frame is already over
    the line, and ends at a single row that the earlier stages have bounded.
    """
    total = len(messages)
    if total <= 1:
        return list(messages), total
    tail = min(AGENT_END_MESSAGE_TAIL_MAX, total)
    while True:
        dropped = messages[: total - tail]
        kept = list(messages[total - tail :])
        candidate = dict(frame)
        candidate["messages"] = kept
        candidate["messages_total"] = total
        candidate["messages_omitted"] = total - tail
        aggregate = _usage_aggregate(dropped)
        if aggregate is not None:
            candidate["usage_aggregate"] = aggregate
        if _frame_bytes(candidate) <= cap_bytes or tail <= 1:
            frame.clear()
            frame.update(candidate)
            return kept, total
        tail = max(1, tail // 2)


def _usage_aggregate(rows: list[Any]) -> dict[str, Any] | None:
    """One folded receipt for every usage the tail stage drops, or ``None``.

    WHY THIS IS NOT OPTIONAL. ``EventController._handle_agent_end`` sums
    ``event.messages[*].usage`` to paint the status band, and a REMOTE viewer
    only ever sees what this encoder emits. Dropping rows without folding their
    receipts would quietly undercount a long turn — the band would show a
    smaller bill than the session actually incurred, with nothing on the frame
    saying why.

    The shape is the one the band already builds for itself: summed token
    buckets (so every call is counted exactly once), ``context_tokens`` from the
    newest receipt that reports one, and ``cost_components`` FOLDED by the
    session's own ``_folded_components`` rather than capped or summed by hand.
    Folding is what keeps two orders of pricing equal — see that function for
    the reported/estimated split and the per-receipt floors it normalises — and
    it is bounded by the distinct serving identities in the turn, not by the
    number of calls, so this aggregate stays a few hundred bytes even when it
    stands in for thousands of receipts.
    """
    from local_operator.harness.types import Usage
    from local_operator.session.frontend_state import _folded_components

    usages = [
        row["usage"] for row in rows if isinstance(row, dict) and isinstance(row.get("usage"), dict)
    ]
    if not usages:
        return None
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    context_tokens: int | None = None
    parsed: list[Any] = []
    for raw in usages:
        try:
            usage = Usage.model_validate(raw)
        except Exception:  # noqa: BLE001 — a malformed receipt must not fail the bound
            continue
        parsed.append(usage)
        for field in totals:
            totals[field] += int(getattr(usage, field, 0) or 0)
        if usage.context_tokens is not None:
            context_tokens = usage.context_tokens
    if not parsed:
        return None
    components = [component.model_dump(mode="json") for component in _folded_components(parsed)]
    aggregate: dict[str, Any] = {**totals, "cost_components": components}
    if context_tokens is not None:
        aggregate["context_tokens"] = context_tokens
    bases = {
        str(component.get("billing_basis"))
        for component in components
        if component.get("billing_basis")
    }
    if len(bases) == 1:
        aggregate["billing_basis"] = bases.pop()
    return aggregate


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
    it with no content kept — every block still present carrying only the clip
    marker, every other key of the row UNTOUCHED, and the frame's own scalars
    counted — so the share is what is actually left once everything the stage
    does not spend has been paid for. A frame that is over budget for reasons
    this stage cannot touch yields a floor-level share rather than a negative
    one.

    Both parts of "what the stage leaves" were learned the hard way, and each
    is worth about a quarter of the budget when it is wrong:

    - Counting the emptied BLOCKS rather than deleting them is the difference
      between a share that fits and one that overshoots by the block count —
      the floor x N trap ``frontend_state`` documents for the live seed, which
      a row of 1,000 blocks reaches here (293,199 B against a 262,144 B budget
      before this, 260,445 B after).
    - Keeping ``provider_payload`` is the same difference for a PREVIEW row: a
      row elided in whole loses it, but the clip only ever trims ``details``
      inside it (and only when that is oversized), so a residual that dropped
      the key spent its share against a frame smaller than the one emitted —
      measured at 267,793 B on the SSE and socket paths for an ordinary
      100-tool-row turn, 289,793 B with a small ``details`` per row, against a
      262,144 B budget. The overshoot is bounded by the same rule that hid it
      (a kept ``details`` may stay up to a quarter of the row's share).
    """
    emptied = list(messages)
    for index in positions:
        row = messages[index]
        # A shallow copy: every key stays as it is, including
        # ``provider_payload``, and only the content is reduced to what the clip
        # leaves of it.
        emptied[index] = {
            **row,
            "content": [
                {"type": "text", "text": "…"}
                for block in (row.get("content") or [])
                if isinstance(block, dict)
            ],
        }
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
) -> tuple[int, int, frozenset[int], frozenset[int]]:
    """Spend the tool-row share, returning the two counts and the rows they name.

    The POSITIONS travel with the counts because a later stage (the message
    tail) can drop rows, and a counter for a row that is no longer in the frame
    is a lie the reader cannot check: it names content that is not there rather
    than content that was cut. ``_bound_agent_end`` intersects these sets with
    the retained suffix instead of re-deriving them from the marker text.

    The NEWEST ``AGENT_END_PREVIEW_ROWS_MAX`` rows keep a preview and the older
    ones are elided in whole — the same ordering, for the same reason, as the
    live-event row cap this module mirrors. The elided rows are written FIRST
    because their markers are part of the frame the preview share is measured
    against: at ~150 bytes each, 2,000 of them are most of a 256 KiB budget,
    and a share computed without them would hand the previews bytes that are
    already spent.

    The two counts are returned separately rather than summed (``elided_tool_rows``
    and ``clipped_tool_rows`` on the payload): a clipped row lost content just
    as surely as an elided one, and a single counter named for one of the two
    left a consumer reading ``0`` beside a multi-kilobyte ``elided_bytes``. Each
    number is also what its row's content shows — whole-row elision leaves the
    marker naming the transcript entry, a clip leaves ``…`` — and neither count
    includes a row the bound only trimmed ``details`` on, which is a change to
    what a card renders rather than to what the row's content says.
    """
    preview_count = max(0, len(positions) - AGENT_END_PREVIEW_ROWS_MAX)
    elided = 0
    elided_positions: set[int] = set()
    for index in positions[:preview_count]:
        row = copy.deepcopy(messages[index])
        messages[index] = row
        chars = _text_chars(row)
        row["content"] = [
            {
                "type": "text",
                "text": _elided_row_marker(row, chars=chars, session_id=session_id),
            }
        ]
        # The provider-native replay payload is dead weight once the row has
        # nothing left to replay, and the harness's own bookkeeping rides
        # inside it under ``details`` (``harness/types.py``), so one pop
        # takes both.
        row.pop("provider_payload", None)
        elided += 1
        elided_positions.add(index)

    previews = positions[preview_count:]
    if not previews:
        return elided, 0, frozenset(elided_positions), frozenset()
    share = _spendable_share(
        frame, messages, previews, cap_bytes=cap_bytes, floor_chars=floor_chars
    )
    clipped = 0
    clipped_positions: set[int] = set()
    for index in previews:
        row = copy.deepcopy(messages[index])
        messages[index] = row
        before = _content_shape(row)
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
        if _content_shape(row) < before:
            clipped += 1
            clipped_positions.add(index)
    return elided, clipped, frozenset(elided_positions), frozenset(clipped_positions)


def _content_shape(row: dict[str, Any]) -> tuple[int, int]:
    """What a READER of this row sees: (characters of text, number of blocks).

    The counter it feeds (``clipped_tool_rows``) means "this row kept a preview
    and lost part of its content", so it has to be driven by the content and
    not by the row's serialized size: the bound also nulls an oversized
    ``details``, and a row whose only change is that — no text touched, no block
    dropped — counted as clipped. Measured on the shape that exposes it (100
    rows, 100-char results, a 5,000-char ``details`` each): 100 counted,
    0 ending in ``…``, 100/100 texts byte-identical to the source.
    """
    return _text_chars(row), len(
        [block for block in (row.get("content") or []) if isinstance(block, dict)]
    )


def _elided_row_marker(row: dict[str, Any], *, chars: int, session_id: str | None) -> str:
    """The marker for a row whose content was elided in whole.

    The reference is what makes the elision recoverable, so it is only stated
    when it can be resolved: a row with no entry id (never one the loop emits —
    ``harness/types.py`` generates one — but a host can hand this function
    anything) says so instead of naming an entry that does not exist.
    """
    entry_id = row.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        reference = "no transcript entry: this row carries no entry id"
    else:
        located = f"transcript entry {entry_id}"
        if session_id:
            located += f" of session {session_id}"
        reference = f"full text: {located}"
    return _ELIDED_ROW_MARKER.format(chars=chars, tool=_tool_label(row), reference=reference)


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
    "AGENT_END_MESSAGE_TAIL_MAX",
    "AGENT_END_PREVIEW_ROWS_MAX",
    "bound_agent_end_for_wire",
]
