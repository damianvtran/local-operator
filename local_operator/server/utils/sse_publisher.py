"""Translate pump messages into broker events.

The parent-side pump (:mod:`local_operator.server.utils.job_processor_queue`)
drains one ``multiprocessing.Queue`` per job. This module is the only place
that decides what those messages look like on the SSE wire, so the pump stays a
dispatcher and the wire contract stays reviewable in one file.

TWO CHANNELS PER EVENT, ON PURPOSE
----------------------------------
Everything is published to both the job-keyed channel and (where a record id
exists) the record-keyed channel. The record channel exists for parity with the
legacy WebSocket key; the job channel exists because a client can open it the
moment it submits a turn, before any record id has been minted. Each channel
owns its own sequence space, so a client resuming on one is unaffected by
traffic on the other.

FAILURE POSTURE
---------------
Publishing must never raise into the pump. The pump is the only consumer of the
child process's queue: if it dies, job status stops advancing and the turn is
orphaned. Every entry point here therefore swallows and logs, because a
disconnected browser is not a reason to lose a turn.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

from local_operator.jobs import JobStatus
from local_operator.server.models.schemas import WebsocketConnectionType
from local_operator.server.utils.event_broker import (
    EventBroker,
    job_channel,
    message_channel,
)
from local_operator.server.utils.sse import EventName, envelope
from local_operator.types import CodeExecutionResult

logger = logging.getLogger("local_operator.server.utils.sse_publisher")

#: Job states after which no further events can arrive, so the stream closes.
TERMINAL_JOB_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})

#: Engine event ``type`` -> SSE event name. Only events with a UI meaning are
#: mapped; anything absent here is dropped rather than forwarded under an
#: invented name, because a consumer cannot render an event it has never been
#: told about and a silently auto-named event is worse than none.
_AGENT_EVENT_NAMES: Dict[str, str] = {
    "agent_start": EventName.AGENT_START,
    "agent_end": EventName.AGENT_END,
    "turn_start": EventName.TURN_START,
    "turn_end": EventName.TURN_END,
    "message_update": EventName.MESSAGE_DELTA,
    "tool_execution_start": EventName.TOOL_START,
    "tool_execution_update": EventName.TOOL_DELTA,
    "tool_execution_end": EventName.TOOL_END,
    "notice": EventName.NOTICE,
    "compaction_start": EventName.COMPACTION_START,
    "compaction_end": EventName.COMPACTION_END,
    "retry_start": EventName.RETRY_START,
    "retry_end": EventName.RETRY_END,
}


def legacy_record_frame(record: CodeExecutionResult, message_id: str) -> Dict[str, Any]:
    """The exact dict the WebSocket transport puts on the wire.

    ``WebSocketManager.broadcast()`` dumps the record and then injects
    ``message_id`` and ``connection_type``. Reproducing both injections here is
    what makes SSE a transport swap for an existing client: its reducer sees the
    same keys it always has. Do not "clean this up" - the two extra keys are
    part of the compatibility surface.
    """
    data = record.model_dump()
    data["message_id"] = message_id
    data["connection_type"] = WebsocketConnectionType.MESSAGE.value
    return data


def publish_record(
    broker: Optional[EventBroker],
    job_id: Optional[str],
    record: CodeExecutionResult,
) -> None:
    """Publish one execution record to its record channel and the job channel.

    ``record.complete`` versus ``record.update`` is derived from the record's
    own ``is_complete``, so a client can key completion off the event name
    instead of inspecting the body - while the body still carries the flag for
    clients that already do.

    A completed record also *ends* its own channel. The record is never touched
    again once the bridge flips ``is_complete``, so a listener on that channel
    has nothing left to wait for - and a stream that stays open forever after
    the last frame is exactly the defect the legacy socket had: it never closed
    on completion, so the client had to infer the end from the payload and hang
    up itself. Here the server says so and closes. The *job* channel is not
    terminal at this point: further records (the next message, another tool)
    still follow.
    """
    if broker is None:
        return
    try:
        message_id = str(record.id)
        payload: Dict[str, Any] = {
            "record": legacy_record_frame(record, message_id),
            "message_id": message_id,
        }
        name = EventName.RECORD_COMPLETE if record.is_complete else EventName.RECORD_UPDATE
        if job_id:
            payload["job_id"] = job_id
        # The record channel keeps parity with the legacy WebSocket key; the job
        # channel is what a client can attach to before any record id exists.
        record_ch = message_channel(message_id)
        _publish(broker, record_ch, name, payload)
        if job_id:
            _publish(broker, job_channel(job_id), name, payload)

        if record.is_complete:
            _publish(
                broker,
                record_ch,
                EventName.TERMINAL,
                {"message_id": message_id, "status": "complete", "reason": "record_complete"},
                terminal=True,
            )
    except Exception:  # noqa: BLE001 - never break the pump
        logger.warning("failed to publish record to SSE broker", exc_info=True)


def _record_id_for(payload: Mapping[str, Any]) -> Any:
    """The record id an event belongs to, for the record-keyed channel.

    Engine message events carry the id nested at ``message.id`` (there is no
    top-level ``message_id``), while tool events use a top-level
    ``tool_call_id``; honour both so ``message.*`` reaches the parity endpoint
    (review B-4).
    """
    top = payload.get("message_id")
    if top:
        return top
    nested = payload.get("message")
    if isinstance(nested, Mapping):
        return nested.get("id")
    return payload.get("tool_call_id")


def publish_agent_event(
    broker: Optional[EventBroker],
    job_id: Optional[str],
    payload: Mapping[str, Any],
) -> None:
    """Publish a raw engine event (deltas, tool traces, turn boundaries).

    These are the events the legacy WebSocket bridge discards. They are additive:
    a client that only understands ``record.*`` keeps working, while one that
    wants true incremental text reads ``message.delta`` and stops re-rendering a
    whole message per frame.
    """
    if broker is None:
        return
    try:
        raw_type = str(payload.get("type") or "")
        name = _AGENT_EVENT_NAMES.get(raw_type)
        if name is None:
            return
        body = {k: v for k, v in payload.items() if k != "type"}
        if job_id:
            body["job_id"] = job_id
        channels = [job_channel(job_id)] if job_id else []
        message_id = _record_id_for(payload)
        if message_id:
            channels.append(message_channel(str(message_id)))
        for channel in channels:
            _publish(broker, channel, name, body)
    except Exception:  # noqa: BLE001 - never break the pump
        logger.warning("failed to publish agent event to SSE broker", exc_info=True)


def publish_job_status(
    broker: Optional[EventBroker],
    job_id: str,
    status: Any,
    result: Optional[Mapping[str, Any]] = None,
) -> None:
    """Publish a job status transition, and close the stream on a terminal one.

    The terminal frame is what lets a client stop cleanly rather than sit on an
    open request until a proxy times it out. It is published *after* the status
    event so the last thing a client sees is the outcome, then the close.
    """
    if broker is None:
        return
    try:
        status_value = getattr(status, "value", status)
        channel = job_channel(job_id)
        body: Dict[str, Any] = {"job_id": job_id, "status": status_value}
        if result is not None:
            # Errors are the actionable part of a result; the rest is the HTTP
            # response's job, not the stream's.
            error = result.get("error") if isinstance(result, Mapping) else None
            if error:
                body["error"] = error
        _publish(broker, channel, EventName.JOB_STATUS, body)

        if status in TERMINAL_JOB_STATUSES:
            _publish(
                broker,
                channel,
                EventName.TERMINAL,
                {"job_id": job_id, "status": status_value},
                terminal=True,
            )
    except Exception:  # noqa: BLE001 - never break the pump
        logger.warning("failed to publish job status to SSE broker", exc_info=True)


def _publish(
    broker: EventBroker,
    channel: str,
    name: str,
    payload: Mapping[str, Any],
    *,
    terminal: bool = False,
) -> None:
    """Publish with the envelope stamped to match the frame's own sequence."""
    broker.publish_with(
        channel,
        name,
        lambda seq: envelope(name, payload, seq=seq, channel=channel),
        terminal=terminal,
    )
