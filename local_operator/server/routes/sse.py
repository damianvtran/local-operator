"""SSE routes: the streaming surface a UI attaches to.

Three endpoints, and the difference between the first two matters:

``GET /v1/sse/messages/{message_id}``
    Parity with the legacy WebSocket channel, keyed by the assistant *record*
    id. Use it when the id is already known (e.g. re-attaching to a message
    the transcript already lists).

``GET /v1/sse/jobs/{job_id}``
    Keyed by the job id returned from ``POST /v1/chat/.../async``. This is the
    one to prefer: it can be opened the instant the turn is submitted, before
    any record exists. The WebSocket design could not do this - the client had
    to poll job status until a record id appeared, then connect, and relied on
    frames being cumulative to catch up on whatever it missed in between. That
    race is a design defect, not a fact of life, and this endpoint removes it.

``GET /v1/sse/capabilities``
    Transport negotiation. A client hits this once; a backend too old to have
    SSE answers 404, which is the signal to fall back to WebSockets.

Resume is supported two ways because two kinds of client exist. ``EventSource``
sends ``Last-Event-ID`` automatically on reconnect; a ``fetch``-based reader
(which is what you need if you want request headers) usually tracks its own
cursor and puts it in the URL. Both are accepted, and when both are present the
*larger* wins - on a browser auto-reconnect the header is current while the URL
may hold the cursor the stream was originally opened with, and replaying from
the stale one would duplicate everything in between.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse

from local_operator.server.dependencies import get_event_broker, get_job_manager
from local_operator.server.utils.event_broker import (
    EventBroker,
    job_channel,
    message_channel,
)
from local_operator.server.utils.sse import (
    HEARTBEAT_INTERVAL_S,
    RETRY_HINT_MS,
    SSE_HEADERS,
    SSE_OPEN_COMMENT,
    TERMINAL_EVENTS,
    EventName,
    comment,
    envelope,
    frame,
    gap_payload,
    keepalive,
    open_payload,
)

router = APIRouter(prefix="/v1/sse", tags=["SSE"])
logger = logging.getLogger("local_operator.server.routes.sse")


def resolve_cursor(last_event_id: Optional[str], after_seq: Optional[int]) -> Optional[int]:
    """Pick the resume point from the header and the query parameter.

    Returns ``None`` when neither is usable, which means "send me a snapshot
    and then live events" rather than "replay everything". A malformed header
    is ignored rather than rejected: it arrives from the browser without
    application involvement, and failing the request would strand a client that
    is otherwise fine.
    """
    candidates: List[int] = []
    if last_event_id:
        try:
            candidates.append(int(last_event_id.strip()))
        except (TypeError, ValueError):
            logger.debug("ignoring unparseable Last-Event-ID: %r", last_event_id)
    if after_seq is not None:
        candidates.append(after_seq)
    if not candidates:
        return None
    # See the module docstring: the furthest-forward cursor is the correct
    # resume point, because replaying from a stale one duplicates events.
    return max(candidates)


def snapshot_from(broker: EventBroker, channel: str) -> List[Dict[str, Any]]:
    """Fold the retained buffer into the current record state.

    Record frames are cumulative, so the newest frame per record id *is* the
    state of that record. Folding rather than replaying is what makes attach
    cheap for a long turn: a 1,200-frame message collapses to one entry.
    Ordering follows first appearance, which is the order the transcript shows.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for event in broker.retained(channel):
        if event.name not in (EventName.RECORD_UPDATE, EventName.RECORD_COMPLETE):
            continue
        record = event.data.get("record")
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("id") or "")
        if not record_id:
            continue
        if record_id not in latest:
            order.append(record_id)
        latest[record_id] = record
    return [latest[rid] for rid in order]


async def _stream(
    request: Request,
    broker: EventBroker,
    channel: str,
    cursor: Optional[int],
) -> AsyncGenerator[str, None]:
    """Yield SSE frames for one subscriber until it disconnects or the turn ends.

    Structure notes:

    * The open comment and ``retry:`` hint go out before anything else so
      headers are flushed through intermediaries immediately and the browser
      learns our reconnect interval even if the turn produces nothing.
    * ``stream.open`` always carries a snapshot, so a client renders current
      state without waiting for the next delta.
    * Every yield is inside the ``try`` whose ``finally`` unsubscribes. When the
      client vanishes, Starlette throws ``GeneratorExit``/``CancelledError``
      into this generator, and without that ``finally`` the subscriber would
      leak for the channel's whole lifetime.
    """
    subscription = broker.subscribe(channel, after_seq=cursor)
    try:
        yield comment(SSE_OPEN_COMMENT.lstrip(": "))
        # `retry:` is a bare directive; attach it to the first real frame so it
        # is not a frame of its own.
        snapshot = snapshot_from(broker, channel)
        yield frame(
            EventName.OPEN,
            envelope(
                EventName.OPEN,
                open_payload(
                    channel,
                    last_seq=broker.last_sequence(channel),
                    resumed=cursor is not None,
                    snapshot=snapshot,
                ),
                channel=channel,
            ),
            retry_ms=RETRY_HINT_MS,
        )

        if subscription.resumed_with_gap:
            # The cursor predated the retained buffer. The snapshot above keeps
            # the UI correct, but anything that happened purely as a delta in
            # the evicted window is gone, so the client is told to reconcile.
            yield frame(
                EventName.GAP,
                envelope(
                    EventName.GAP,
                    gap_payload("evicted", expected_seq=(cursor or 0) + 1),
                    channel=channel,
                ),
            )

        # A channel that already ended must not leave the client hanging. Two
        # cases, and conflating them loses data: with a replay backlog queued
        # (a resume), fall through so the loop drains it and closes naturally on
        # the terminal frame it contains; with nothing queued, there is nothing
        # left to send and nothing more can arrive, so close now. The snapshot
        # in `stream.open` above is what makes the immediate close safe.
        if broker.is_terminal(channel) and subscription.pending == 0:
            retained = broker.retained(channel)
            if retained and retained[-1].name in TERMINAL_EVENTS:
                last = retained[-1]
                yield frame(last.name, last.data, event_id=last.seq)
                return

        while True:
            if await request.is_disconnected():
                return

            event = await subscription.get(timeout=HEARTBEAT_INTERVAL_S)

            if event is None:
                if subscription.is_closed:
                    # Broker shutdown. Say so rather than dying silently, so
                    # the client reconnects deliberately.
                    yield frame(
                        EventName.ERROR,
                        envelope(
                            EventName.ERROR,
                            {"error": "stream closed by server", "retryable": True},
                            channel=channel,
                        ),
                    )
                    return
                # A dispatchable keepalive, not a comment: proxies count it as
                # traffic AND the client's stall detector can re-arm on it. A
                # comment alone is invisible to EventSource, so a healthy quiet
                # turn would have read as a dead connection (review C-01).
                yield keepalive()
                continue

            dropped = subscription.take_dropped()
            if dropped:
                # Report before the event that follows it, so the client knows
                # the gap precedes this frame rather than follows it.
                yield frame(
                    EventName.GAP,
                    envelope(
                        EventName.GAP,
                        gap_payload("overflow", dropped=dropped),
                        channel=channel,
                    ),
                )

            yield frame(event.name, event.data, event_id=event.seq)

            if event.name in TERMINAL_EVENTS:
                return
    except asyncio.CancelledError:
        # Client went away mid-frame. Not an error; let `finally` clean up.
        raise
    finally:
        subscription.close()


def _sse_response(
    request: Request,
    broker: EventBroker,
    channel: str,
    cursor: Optional[int],
) -> StreamingResponse:
    return StreamingResponse(
        _stream(request, broker, channel, cursor),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get(
    "/messages/{message_id}",
    summary="Stream updates for one message record",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_message_events(
    request: Request,
    message_id: str,
    after_seq: Optional[int] = Query(
        default=None,
        ge=0,
        description="Resume after this sequence number. Alternative to the Last-Event-ID header.",
    ),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
    broker: EventBroker = Depends(get_event_broker),
) -> StreamingResponse:
    """Events for one assistant record, keyed exactly as the WebSocket path is.

    Prefer ``/jobs/{job_id}`` for new work: this endpoint requires the record
    id, which only exists once the turn has started producing.
    """
    cursor = resolve_cursor(last_event_id, after_seq)
    return _sse_response(request, broker, message_channel(message_id), cursor)


@router.get(
    "/jobs/{job_id}",
    summary="Stream every event for one job",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_job_events(
    request: Request,
    job_id: str,
    after_seq: Optional[int] = Query(
        default=None,
        ge=0,
        description="Resume after this sequence number. Alternative to the Last-Event-ID header.",
    ),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
    broker: EventBroker = Depends(get_event_broker),
) -> StreamingResponse:
    """Every event for a job, including the record ids as they are minted.

    Openable immediately after the async chat call returns, which is what lets
    a client attach with no gap and no polling.
    """
    cursor = resolve_cursor(last_event_id, after_seq)
    return _sse_response(request, broker, job_channel(job_id), cursor)


@router.get(
    "/capabilities",
    summary="Advertise the streaming transports this backend supports",
)
async def stream_capabilities(
    broker: EventBroker = Depends(get_event_broker),
    job_manager: Any = Depends(get_job_manager),
) -> Dict[str, Any]:
    """Transport negotiation for a client that may face an older backend.

    A backend without SSE has no such route and answers 404 - that is the
    signal to use WebSockets. Returning the event names as data (rather than
    making the client hardcode them) lets a newer client detect that a backend
    predates an event it wants to consume, instead of waiting forever for a
    frame that will never arrive.
    """
    del job_manager  # presence proves the app is fully wired, not read here
    return {
        "transports": ["sse", "websocket"],
        "preferred": "sse",
        "sse": {
            "version": 1,
            "channels": {
                "message": "/v1/sse/messages/{message_id}",
                "job": "/v1/sse/jobs/{job_id}",
            },
            "resume": {
                "last_event_id": True,
                "query_param": "after_seq",
                "replay_buffer": broker.stats()["replay_buffer"],
            },
            "heartbeat_interval_s": HEARTBEAT_INTERVAL_S,
            "retry_hint_ms": RETRY_HINT_MS,
            "events": sorted(
                value
                for key, value in vars(EventName).items()
                if not key.startswith("_") and isinstance(value, str)
            ),
            "legacy_record_frames": True,
        },
        "websocket": {
            "channels": {"message": "/v1/ws/messages/{message_id}"},
            "deprecated": True,
        },
    }
