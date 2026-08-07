"""Server-Sent Events framing and the event taxonomy the UI consumes.

WHY SSE AND NOT WEBSOCKETS
--------------------------
Everything the browser needs from a running turn is server-to-client. The one
frame the existing UI ever sent upstream was a keepalive ``ping``; cancellation
was already an HTTP call. SSE therefore costs no capability and buys three
things a WebSocket cannot:

* **Automatic reconnection with a cursor.** ``EventSource`` retries on its own
  and replays ``Last-Event-ID``, so a dropped connection resumes at the exact
  event instead of restarting the render.
* **Plain HTTP.** It traverses proxies, CORS and dev servers without an upgrade
  handshake, and it is visible to ordinary HTTP tooling.
* **Server-driven keepalive.** A comment frame defeats idle timeouts without a
  client timer.

WHERE THE DESIGN COMES FROM
---------------------------
Four implementations were read before writing this, and each contributed a
specific decision rather than a general vibe:

* **Codex** (as normalised by Minerva's ``adapters/codexcli``): text arrives as
  *coalesced* deltas carrying both the increment and a running snapshot, so a
  late or lossy consumer can repaint from any single frame. Hence
  ``message.delta`` carries ``delta`` **and** ``snapshot``.
* **Anthropic's Messages stream**: named events plus an inner ``type``
  discriminator, and lifecycle triads (``*_start`` / ``*_delta`` / ``*_stop``).
  Hence the dual discriminator here - the SSE ``event:`` name and ``data.type``
  always agree, so a client may switch on either.
* **Minerva ``agent-runtime-svc``**: ``id:`` per frame, a ``retry:`` hint, a
  15s ``: heartbeat`` comment, an explicit terminal frame, and resume by
  cursor. Its stream is a materialised tail over a durable log, which is what
  makes resume meaningful; the bounded ring buffer in
  :mod:`local_operator.server.utils.event_broker` plays that role here.
* **omp's mobile portal**: open every stream with a full snapshot so a
  reconnecting phone renders immediately, and set ``x-accel-buffering: no``
  because a reverse proxy will otherwise buffer the stream into uselessness.

Snapshot-on-connect and cursor-replay are *both* implemented, because they fail
in opposite directions: replay is exact but bounded by the ring buffer, while a
snapshot is always available but coarse. A resuming client gets replay when its
cursor is still retained and a snapshot when it is not - and is told which,
because silently downgrading is how a UI ends up rendering a hole forever.

BACKWARD COMPATIBILITY
----------------------
``record.*`` events carry the legacy ``CodeExecutionResult`` dump verbatim,
including the ``message_id``/``connection_type`` keys the WebSocket path
injects. A client can therefore treat SSE as a transport swap and feed its
existing reducer unchanged, then adopt the richer ``message.delta`` and
``tool.*`` events at its own pace. The WebSocket publish path is untouched.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, Mapping, Optional

#: Comment sent when the stream opens. Browsers treat a comment as traffic, so
#: this both flushes response headers through any intermediary and proves
#: liveness before the first real event.
SSE_OPEN_COMMENT = ": connected"

#: Keepalive interval. Matches Minerva's 15s: comfortably inside the common
#: 30-60s proxy idle timeout while costing ~2 bytes a tick.
HEARTBEAT_INTERVAL_S = 15.0

#: Reconnection hint handed to ``EventSource``. The browser default is ~3s and
#: opaque; 1s recovers a mid-turn drop fast enough that a human does not see it.
RETRY_HINT_MS = 1000

#: Response headers. ``x-accel-buffering`` is not optional in practice - nginx
#: and several tunnels buffer ``text/event-stream`` without it, which turns a
#: live stream into one burst at the end.
SSE_HEADERS: Dict[str, str] = {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache, no-transform",
    "connection": "keep-alive",
    "x-accel-buffering": "no",
}


class EventName:
    """The wire taxonomy.

    Names are dotted and stable; they are the SSE ``event:`` names and are
    mirrored in ``data.type``. The mapping to Minerva's ``runtime.agent.*``
    vocabulary is noted per entry so a future runtime adapter is a rename
    table rather than a translation layer.
    """

    # -- stream lifecycle (transport-level, no Minerva equivalent) ---------
    #: First event on every stream: what you are attached to and where you are.
    OPEN = "stream.open"
    #: Frames were lost (subscriber overflow, or a cursor older than the
    #: buffer). The client MUST reconcile over REST rather than trust its view.
    GAP = "stream.gap"
    #: The turn reached a terminal state and the server is closing the stream.
    #: Equivalent to Minerva's ``projection.terminal``.
    TERMINAL = "stream.terminal"
    #: Transport or server failure. Equivalent to Minerva's ``error`` event.
    ERROR = "error"
    #: Dispatchable keepalive. Unlike a ``:`` comment (which proxies count as
    #: traffic but ``EventSource`` silently discards), this is a real event so a
    #: client can re-arm its stall detector on it. A healthy quiet turn must not
    #: look like a dead connection.
    KEEPALIVE = "keepalive"

    # -- legacy-compatible record frames ----------------------------------
    #: A ``CodeExecutionResult`` snapshot - byte-compatible with the WebSocket
    #: data frame. Maps to ``runtime.agent.item.updated``.
    RECORD_UPDATE = "record.update"
    #: The same shape, with ``is_complete`` set. Maps to
    #: ``runtime.agent.item.completed``.
    RECORD_COMPLETE = "record.complete"

    # -- richer engine events (dropped by the WebSocket bridge) ------------
    #: Incremental assistant text: ``delta`` plus a cumulative ``snapshot``.
    #: Maps to ``runtime.agent.item.delta``.
    MESSAGE_DELTA = "message.delta"
    #: Tool invocation announced. Maps to ``runtime.agent.item.started`` with
    #: an item type of ``command_execution``.
    TOOL_START = "tool.start"
    #: Partial tool output. Maps to ``runtime.agent.item.output_delta``.
    TOOL_DELTA = "tool.delta"
    #: Tool finished, with its result and error flag. Maps to
    #: ``runtime.agent.item.completed`` / ``failed``.
    TOOL_END = "tool.end"
    #: Turn boundaries. Map to ``runtime.agent.turn.started`` / ``completed``.
    TURN_START = "turn.start"
    TURN_END = "turn.end"
    #: Whole-run boundaries, carrying the generation counter and abort flag.
    AGENT_START = "agent.start"
    AGENT_END = "agent.end"
    #: Human-facing notice (info/warning/error) surfaced in the transcript.
    NOTICE = "notice"
    #: Context compaction began/finished - the UI may show a marker.
    COMPACTION_START = "compaction.start"
    COMPACTION_END = "compaction.end"
    #: A provider call is being retried, possibly on a fallback model.
    RETRY_START = "retry.start"
    RETRY_END = "retry.end"
    #: Job status transition. Maps to Minerva's ``run.status_changed``.
    JOB_STATUS = "job.status"
    #: Scheduled-job status. Kept a distinct name because the legacy socket
    #: pushed this onto the same channel as execution records, where a client
    #: could mistake it for one.
    SCHEDULE_STATUS = "schedule.status"


#: Events that mean the stream is finished. The route closes after sending one.
TERMINAL_EVENTS = frozenset({EventName.TERMINAL})


def envelope(
    name: str,
    payload: Mapping[str, Any] | None = None,
    *,
    seq: Optional[int] = None,
    channel: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the JSON body for one event.

    ``type`` duplicates the SSE event name on purpose: a consumer using raw
    ``fetch`` (or one that logs frames) should not have to correlate the two
    halves of the frame to know what it received. ``seq`` is the resume cursor
    and is echoed inside the body so a client that buffers events can persist
    its position without tracking ``lastEventId`` separately.
    """
    body: Dict[str, Any] = {"type": name}
    if seq is not None:
        body["seq"] = seq
    if channel is not None:
        body["channel"] = channel
    body["ts"] = time.time()
    if payload:
        body.update(payload)
    return body


def frame(
    name: str,
    data: Mapping[str, Any],
    *,
    event_id: Optional[int] = None,
    retry_ms: Optional[int] = None,
) -> str:
    """Serialise one SSE frame.

    ``data`` is emitted as a single line. ``json.dumps`` cannot produce a raw
    newline inside a string (it escapes them), so one ``data:`` line is
    sufficient and multi-line folding would only add bytes. Any newline that
    did appear would silently split the frame, so the guard below is cheap
    insurance against a future non-JSON payload.
    """
    parts = []
    if event_id is not None:
        parts.append(f"id: {event_id}")
    if retry_ms is not None:
        parts.append(f"retry: {retry_ms}")
    parts.append(f"event: {name}")
    body = json.dumps(data, default=_fallback, separators=(",", ":"))
    for line in body.split("\n"):
        parts.append(f"data: {line}")
    return "\n".join(parts) + "\n\n"


def comment(text: str) -> str:
    """An SSE comment - ignored by clients, but real traffic to proxies."""
    return f": {text}\n\n"


def heartbeat() -> str:
    """Keepalive tick as a comment - real traffic to proxies, invisible to clients."""
    return comment("heartbeat")


def keepalive() -> str:
    """Keepalive tick as a *dispatchable* event.

    A comment keeps a proxy from idling the connection out, but ``EventSource``
    discards comments without firing any handler, so a client cannot use them to
    re-arm a stall detector. This named event is the client-visible half of the
    same tick: it carries no state, only liveness.
    """
    return frame(EventName.KEEPALIVE, {"type": EventName.KEEPALIVE})


def _fallback(value: Any) -> Any:
    """Last-resort encoder.

    Records are dumped to plain dicts before they reach here, but events carry
    engine objects whose fields may include enums, datetimes or models. Failing
    the whole frame because one field is exotic would drop a real event, so
    unknown values degrade to their string form.
    """
    for attr in ("model_dump", "isoformat", "value"):
        hook = getattr(value, attr, None)
        if hook is None:
            continue
        try:
            return hook() if callable(hook) else hook
        except Exception:  # noqa: BLE001 - degrade, never drop the frame
            break
    return str(value)


def gap_payload(
    reason: str, *, dropped: int = 0, expected_seq: Optional[int] = None
) -> Dict[str, Any]:
    """Body for :data:`EventName.GAP`.

    Names the reason so the client can distinguish "you were too slow"
    (``overflow``) from "you were away too long" (``evicted``); both require a
    REST reconciliation, but only the first is a client-side bug.
    """
    payload: Dict[str, Any] = {"reason": reason, "reconcile": True}
    if dropped:
        payload["dropped"] = dropped
    if expected_seq is not None:
        payload["expected_seq"] = expected_seq
    return payload


def open_payload(
    channel: str,
    *,
    last_seq: int,
    resumed: bool,
    snapshot: Iterable[Mapping[str, Any]] | None = None,
    transport: str = "sse",
) -> Dict[str, Any]:
    """Body for :data:`EventName.OPEN`.

    Carries the snapshot inline rather than as separate events so a client
    knows the difference between "state as of attach" and "something just
    happened" - conflating them makes a reconnect look like a burst of new
    activity. ``resumed`` tells the client whether its cursor was honoured.
    """
    records = list(snapshot or [])
    return {
        "transport": transport,
        "last_seq": last_seq,
        "resumed": resumed,
        "snapshot": records,
        "snapshot_count": len(records),
    }
