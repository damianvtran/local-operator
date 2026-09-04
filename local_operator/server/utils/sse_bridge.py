"""Attach a plain harness session to an :class:`EventBroker`.

WHY THIS EXISTS
---------------
Until now the only producer of SSE events was the ``lop serve`` job pipeline:
:mod:`local_operator.server.utils.job_processor_queue` drains a
``multiprocessing.Queue`` filled by an :class:`AgentEventBridge` running in the
child process, and dispatches each tuple to
:mod:`local_operator.server.utils.sse_publisher`. Every other host of a session
- headless ``exec``, an embedded supervisor, anything driving
``Session.prompt`` directly - produced **zero** SSE events, even though the
whole transport (framing, replay buffer, ``Last-Event-ID`` resume, the
endpoints, and a UI that already speaks all of it) was sitting right there.

This module is the missing seam, and it is deliberately thin: the projection
from ``AgentEvent`` to the wire is NOT reimplemented here. A second projection
is how two hosts start disagreeing about what ``message.delta`` means, and the
one in ``AgentEventBridge`` already carries hard-won behaviour a reimplementation
would silently drop - the running ``snapshot`` on delta frames, the per-string
:data:`STREAM_VALUE_LIMIT` cap that stops a megabyte tool result from stalling
every listener, and the record shape the legacy WebSocket consumer expects.

So instead of a new projection, this supplies a new *destination* for the
existing one. ``AgentEventBridge`` talks to a ``StatusQueue`` - a protocol whose
entire surface is ``put(obj)`` - so an in-process object that implements ``put``
and publishes straight to the broker reuses the projection verbatim while
skipping the process hop the server needs and a single-process host does not.

WHAT THIS IS NOT
----------------
It does not start a server, open a socket, or own a broker. The caller supplies
the :class:`EventBroker` it already has, which keeps the dependency pointing one
way: a host decides it wants a stream, rather than the stream reaching into
hosts. A host with no broker calls nothing and pays nothing.

IMPORT COST
-----------
``tests/unit/test_import_graph.py`` pins what the CLI drags in at startup, so
the heavy import (``server.utils.operator``, which pulls the whole legacy
adapter plus yaml/dotenv/pickle) is function-local inside :func:`attach_broker`.
Importing this module costs the publisher and the broker only, and a host that
never attaches a stream never loads the adapter at all.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

from local_operator.server.utils.event_broker import EventBroker
from local_operator.server.utils.sse_publisher import (
    publish_agent_event,
    publish_job_status,
    publish_record,
)

logger = logging.getLogger("local_operator.server.utils.sse_bridge")


class BrokerStatusSink:
    """A ``StatusQueue`` that publishes to a broker instead of crossing a pipe.

    The server's queue exists because its turn runs in a *child process*; the
    parent-side pump is what turns each tuple into broker events. A host that
    runs the turn in its own process has no such boundary, so this collapses the
    pump into the queue: the same tuples arrive, and are dispatched by the same
    publisher functions the pump uses. The tuple vocabulary is therefore not a
    private detail of this class - it is the one defined by ``AgentEventBridge``
    and consumed in ``job_processor_queue``, and the two must be changed
    together.

    ``put`` never raises. It is called from inside the event handler on the
    turn's own task, so an exception here would abort the turn - the same
    failure posture the publisher documents, one step earlier in the chain.
    """

    def __init__(self, broker: EventBroker, job_id: str) -> None:
        self._broker = broker
        # Required, not optional: every frame this sink emits lands on the
        # job-keyed channel, which is the only channel a consumer can open
        # before a record id has been minted. A sink without one would publish
        # to nowhere and look like a broken stream rather than a missing id.
        self._job_id = job_id

    def put(self, obj: object, /) -> None:
        try:
            if not isinstance(obj, tuple) or len(obj) != 3:
                return
            kind, _ident, payload = obj
            if kind == "message_update":
                publish_record(self._broker, self._job_id, payload)  # type: ignore[arg-type]
            elif kind == "agent_event" and isinstance(payload, Mapping):
                publish_agent_event(self._broker, self._job_id, payload)
            # "execution_update" is deliberately dropped. It carries the legacy
            # "Thinking about my next action" placeholder record that exists to
            # drive the old UI's spinner off the job ledger; the pump routes it
            # to the JobManager and never to the broker, and inventing a stream
            # frame for it here would put a record on the wire that the server
            # path does not emit.
        except Exception:  # noqa: BLE001 - a stream must never kill the turn
            logger.warning("failed to publish session event to SSE broker", exc_info=True)


def attach_broker(
    session: Any,
    broker: Optional[EventBroker],
    job_id: str,
) -> Callable[[], None]:
    """Stream ``session``'s events to ``broker`` on ``job_channel(job_id)``.

    Returns the unsubscribe callable, which the caller MUST invoke when the run
    ends: the handler holds the bridge's per-message record state, so leaving it
    attached to a reused session leaks a record dict per message and keeps
    republishing to a channel nobody reads.

    A ``None`` broker returns a no-op, so a host can wire this unconditionally
    and let configuration decide whether a stream exists - the same shape
    ``SchedulerService`` uses for its optional broker.

    Terminal framing is the caller's job, via
    :func:`~local_operator.server.utils.sse_publisher.publish_job_status`: only
    the host knows whether the run completed, failed, or was cancelled, and a
    stream that never receives a terminal frame keepalives until the client
    gives up - the exact defect the transport was built to remove.
    """
    if broker is None:
        return lambda: None

    # Function-local: see the module docstring's IMPORT COST note. The adapter
    # is only needed once a host has actually decided to stream.
    from local_operator.server.utils.operator import AgentEventBridge

    bridge = AgentEventBridge(status_queue=BrokerStatusSink(broker, job_id), job_id=job_id)
    return session.subscribe(bridge.handle)


__all__ = ["BrokerStatusSink", "attach_broker", "publish_job_status"]
