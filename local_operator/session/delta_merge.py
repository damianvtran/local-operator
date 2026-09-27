"""The frame families a queue fold may merge, and the key that names each stream.

ONE definition of the merge taxonomy, shared by the two queues that fold:
``session/runtime/server.py::_compact_event_queue`` (the runtime's attach FIFO)
and ``server/utils/desktop_sessions.py::DesktopSessionBridge`` (the desktop
viewer's subscriber FIFO). The bridge must NOT import ``session.runtime.server``
— that would pull the whole runtime graph into the desktop daemon — so the keys
live in this zero-import leaf module instead, and neither side re-spells them.
The failure this prevents is taxonomy drift: a family renamed or re-keyed on one
side only would silently stop folding the other side's queue. What this leaf
carries is the key SPELLING — it is not family MEMBERSHIP: a family named here
still folds nothing until each folder's own fold is edited to consume its key,
so each queue keeps its own set and only the names are shared.

Families, and the RULE each key implies (the rule lives with the folder that
applies it; this module only answers "which stream is this frame a fragment of"):

* ``message_update`` — accumulates the assistant's visible text; the later frame
  already carries the earlier text in its ``message``, and ``delta`` is
  append-only, so a fold concatenates deltas onto the later frame.
* ``reasoning_delta`` — one fragment per reasoning token, no accumulated
  payload; a fold concatenates deltas in arrival order.
* ``tool_execution_update`` — a SELF-REPLACING family: every frame re-sends the
  tool's CURRENT LIVE VIEW (for ``bash`` a bounded ~128 KiB output tail, for
  ``eval`` a bounded display — not the whole transcript), and the family's
  contract is that the newest frame supersedes the earlier ones (the settled
  result rides ``tool_execution_end``), so a fold keeps the newest frame and
  concatenates nothing. Deliberately not folded by the runtime's compact pass
  (which predates this family's volume); the desktop bridge folds it, and the
  key lives here so its spelling exists once.
* ``aside_delta`` — mergeable too, but its stream identity is on the FRAME (the
  ``req``), not the payload, so ``mergeable_frame_key`` answers for it.

The key functions below were lifted from ``session/runtime/server.py``, where the
runtime's fold — and the tests that drive it — still consume
``mergeable_frame_key``. That one reads the runtime's own ``{op, data}`` frame
shape; the desktop bridge reads ``mergeable_delta_key`` / ``mergeable_snapshot_key``
off the payload its frames carry. Two readers, one taxonomy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def mergeable_delta_key(payload: Mapping[str, Any]) -> str | None:
    """The in-flight stream one queued frame carries a fragment of, or ``None``.

    Compaction folds ADJACENT frames of the same stream into one, and the only
    thing that decides "same stream" is this key. Two families are delta-grade
    and mergeable:

    * ``message_update`` — the assistant's visible text, keyed by the message it
      accumulates into;
    * ``reasoning_delta`` — the model's private reasoning, keyed by the message
      it belongs to (the field is ``message_id``, not a whole ``message``: a
      reasoning frame carries no message, which is also why it is cheap to
      merge).

    The FAMILY is part of the key, so a text fragment and a reasoning fragment
    can never fold together — merging the model's thinking into the answer
    being painted would corrupt the transcript on the viewer's screen, and the
    two arrive interleaved.

    Everything else returns ``None`` and is left alone: a frame that must not
    merge is never compared to its neighbour at all. That includes the ``op``
    around the payload — the aside's ``aside_delta`` frame is mergeable too, but
    its stream identity lives on the FRAME (its ``req``), so it is
    :func:`mergeable_frame_key` that answers for it — and the snapshot family,
    which :func:`mergeable_snapshot_key` answers for.
    """
    kind = payload.get("type")
    if kind == "message_update":
        return f"message_update:{(payload.get('message') or {}).get('id') or ''}"
    if kind == "reasoning_delta":
        return f"reasoning_delta:{payload.get('message_id') or ''}"
    return None


def mergeable_snapshot_key(payload: Mapping[str, Any]) -> str | None:
    """The stream one queued frame is a SNAPSHOT of, or ``None``.

    ``tool_execution_update`` is the family this answers for: each frame
    re-sends the tool's CURRENT LIVE VIEW — for ``bash`` a bounded ~128 KiB
    output tail, for ``eval`` a bounded display, never the whole transcript —
    and the family's self-replacing contract is that the newest frame
    supersedes the earlier ones (the settled result rides
    ``tool_execution_end``). That contract is why a fold keeps the newest frame
    of a run instead of concatenating. The frames arrive as a chunk stream
    during a tool run, so a viewer stalled behind one queues an unbroken run of
    them that a fold can collapse to the single newest frame.

    Keyed in the same namespace as :func:`mergeable_delta_key` (the family is
    part of the key), so a tool-output frame can never fold into a
    ``message_update`` or ``reasoning_delta`` beside it.

    A frame with no ``tool_call_id`` is not a stream this function can identify,
    so it is left alone rather than keyed as one nameless stream all such frames
    would share — the same rule :func:`mergeable_frame_key` applies to an
    ``aside_delta`` with no ``req``.
    """
    if payload.get("type") != "tool_execution_update":
        return None
    call_id = payload.get("tool_call_id")
    return None if not call_id else f"tool_execution_update:{call_id}"


def mergeable_frame_key(frame: Mapping[str, Any]) -> str | None:
    """The in-flight stream a queued FRAME carries a fragment of, or ``None``.

    :func:`mergeable_delta_key` reads an event's payload; this reads the frame
    around it, because the aside's stream identity is not in its payload. An
    ``aside_delta`` frame is ``{op, req, data: {delta}}`` — the request that
    asked for the aside IS the stream, and the id of the aside panel (which the
    desktop renderer knows as ``aside_id``) never crosses this wire. Two
    fragments fold only when the same ``req`` produced them, so two concurrent
    asides on one connection cannot merge into one answer.

    Keyed in the same namespace as :func:`mergeable_delta_key` (the family is
    part of the key), so an aside fragment can never fold into a
    ``message_update`` or ``reasoning_delta`` beside it: those are the
    conversation the viewer is reading, and this is a private question about it.
    """
    op = frame.get("op")
    if op == "aside_delta":
        req = frame.get("req")
        # A frame with no ``req`` is not a stream this method can identify, so it
        # is left alone rather than keyed as one nameless stream all such frames
        # would share.
        return None if req is None else f"aside_delta:{req}"
    if op == "event":
        return mergeable_delta_key(frame.get("data") or {})
    return None
