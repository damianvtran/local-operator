"""Dial a viewer and ask it to display a session, bounded at every step.

The client half of :mod:`local_operator.session.runtime.viewer_server`. Its one
production caller is the notification click handler
(:mod:`local_operator.tui.resume_click`), which runs detached with no event
loop and no user waiting on a prompt — so every failure here is an ordinary
answer that falls through to the next rung, never an exception the user sees.

BOUNDED IS THE WHOLE CONTRACT. A click that hangs is worse than a click that
does nothing: the user gets no window and no explanation, and on macOS the
30 s activation window closes while they wait. Every await below carries a
timeout, and the totals are chosen so the *fallback* spawn still happens
comfortably inside that window even when a viewer is wedged and every dial
times out.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.session.runtime.viewers import ViewerRecord

logger = logging.getLogger(__name__)

#: Connect + authenticate. Loopback, so a healthy viewer answers in
#: microseconds; this bound exists for the wedged one.
_DIAL_TIMEOUT_S = 1.0

#: How long one op may take to be acknowledged. ``resume_session`` crosses to
#: the application's loop and runs a real ``/resume``, which on a cold, long
#: transcript is genuinely slow — the architect measured ``Transcript.__init__``
#: at 3.9 s on this host's largest session. The ack is what tells us not to
#: spawn a second window, so waiting a little longer here is strictly better
#: than giving up and producing the exact duplicate window this feature exists
#: to remove.
_ACK_TIMEOUT_S = 8.0

#: ``focus_window`` is a bounded subprocess on the far side and nothing else.
#: It gets a short leash because a failure to raise a window is cosmetic —
#: the session has already been switched by then.
_FOCUS_TIMEOUT_S = 3.0

#: Bound on one frame. The viewer's replies are tiny; this stops a wedged or
#: hostile writer from growing this process's memory.
_MAX_FRAME_BYTES = 64 * 1024


@dataclass
class ViewerOutcome:
    """What the click managed to achieve. ``switched`` is the one that decides
    whether the caller may skip the spawn — focus is chrome on top of it."""

    switched: bool = False
    focused: bool = False
    detail: str = ""


async def _read_reply(
    reader: asyncio.StreamReader, req: int, timeout_s: float
) -> dict[str, Any] | None:
    """Read frames until the one answering ``req``, or give up.

    Frames are read and split HERE rather than with ``readline`` because
    ``StreamReader.readline`` raises ``LimitOverrunError`` *without consuming
    the buffer*, so one oversized line wedges every later read — the defect
    ``session/runtime/control.py`` documents for ``lop send``. This endpoint
    emits nothing large today, but the failure is silent and permanent if it
    ever does, and the fix is four lines.
    """
    buf = bytearray()

    async def _read() -> dict[str, Any] | None:
        while True:
            nl = buf.find(b"\n")
            if nl == -1:
                if len(buf) > _MAX_FRAME_BYTES:
                    return None
                chunk = await reader.read(4096)
                if not chunk:
                    return None
                buf.extend(chunk)
                continue
            line = bytes(buf[: nl + 1])
            del buf[: nl + 1]
            try:
                frame = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                continue
            if isinstance(frame, dict) and frame.get("req") == req:
                return frame

    try:
        return await asyncio.wait_for(_read(), timeout=timeout_s)
    except (TimeoutError, OSError, ConnectionError):
        return None


async def deliver_click(
    record: ViewerRecord, session_id: str, *, want_focus: bool = True
) -> ViewerOutcome:
    """Switch ``record``'s viewer to ``session_id`` and bring it forward.

    ORDER MATTERS: switch first, then focus. Raising the window before the
    switch shows the user the *previous* session for as long as the switch
    takes, which on a cold long conversation is seconds of looking at the wrong
    thing. Switching first means whatever comes forward is already correct.

    A viewer already displaying the target is switched anyway? No — the caller
    passes that case through ``needs_switch`` below, because re-running
    ``/resume`` on the session already on screen would rebuild the view and
    throw away the user's scroll position for no gain.
    """
    outcome = ViewerOutcome()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", record.control_port),
            timeout=_DIAL_TIMEOUT_S,
        )
    except (OSError, TimeoutError):
        # Refused or unreachable: the process is gone or wedged. Not an error —
        # it is the cheapest possible "route elsewhere".
        outcome.detail = "viewer unreachable"
        return outcome
    try:
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()

        req = 1
        if needs_switch(record, session_id):
            writer.write(
                json.dumps({"op": "resume_session", "req": req, "session_id": session_id}).encode()
                + b"\n"
            )
            await writer.drain()
            reply = await _read_reply(reader, req, _ACK_TIMEOUT_S)
            if reply is None or reply.get("op") != "ack":
                outcome.detail = (reply or {}).get("detail", "no answer to resume_session")
                return outcome
            outcome.switched = True
            outcome.detail = str(reply.get("detail", ""))
        else:
            # Already on screen. Nothing to switch, and saying so is what lets
            # the caller skip the spawn: the session IS displayed.
            outcome.switched = True
            outcome.detail = "already displayed"

        if want_focus:
            req = 2
            writer.write(json.dumps({"op": "focus_window", "req": req}).encode() + b"\n")
            await writer.drain()
            reply = await _read_reply(reader, req, _FOCUS_TIMEOUT_S)
            # A viewer that cannot focus answers with an error frame, and that
            # is a fine outcome: the session is switched and the user finds the
            # window themselves. Never let it undo `switched`.
            outcome.focused = bool(reply is not None and reply.get("op") == "ack")
        return outcome
    except (OSError, ConnectionError) as exc:
        outcome.detail = f"viewer connection lost: {exc}"
        return outcome
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except (OSError, ConnectionError):
            pass


def needs_switch(record: ViewerRecord, session_id: str) -> bool:
    """Whether this viewer must be told to change what it displays.

    False when it is already showing the target — the click then costs only an
    OS activation, which is both the cheapest outcome and the least surprising:
    re-resuming a session already on screen would rebuild its view and discard
    the user's scroll position.
    """
    return record.current_session != session_id


def choose_viewer(records: list[ViewerRecord], session_id: str) -> ViewerRecord | None:
    """Pick the viewer that should take this click, deterministically.

    Precedence, and each rung is a reason rather than a preference:

    1. **A viewer already displaying the target.** Switching it is a no-op, so
       this is the cheapest and least disruptive outcome available.
    2. **The most recently focused viewer that can switch.** The window the
       user was last in is the best available proxy for where they expect to
       land. Ties (two viewers never focused) break on the lowest pid, so
       repeated clicks are stable rather than alternating.

    THE ORDERING IS APPLIED HERE, not inherited from the caller. ``scan_viewers``
    happens to return records in this order already, and an earlier draft of
    this function relied on that — which made it silently wrong for any caller
    holding records from anywhere else, and the docstring promised a precedence
    the code did not implement. A function whose contract is "pick
    deterministically" has to do its own sorting; borrowing the guarantee from a
    collaborator is how it gets lost.

    Returns ``None`` when nothing can take it, which is the caller's signal to
    fall back to spawning a terminal — the behaviour that exists today and must
    keep working.
    """
    for record in records:
        if record.current_session == session_id:
            return record
    switchable = sorted(
        (rec for rec in records if rec.can_switch),
        key=lambda rec: (-rec.focused_at, rec.pid),
    )
    return switchable[0] if switchable else None


def route_click(session_id: str, root: Path | None = None) -> ViewerOutcome:
    """Resolve and deliver, synchronously, for the detached click process.

    Runs its own event loop because the click handler has none — it is a
    short-lived process macOS handed an activation. Returns an outcome whose
    ``switched`` flag is the caller's whole decision: True means a live window
    is now showing the session and nothing should be spawned.
    """
    from local_operator.session.runtime.viewers import scan_viewers

    try:
        records = scan_viewers(root)
    except OSError:
        # No viewer directory, or an unreadable one. An ordinary answer on a
        # machine where no TUI has ever run.
        return ViewerOutcome(detail="no viewer records")
    target = choose_viewer(records, session_id)
    if target is None:
        return ViewerOutcome(detail="no viewer available")
    try:
        return asyncio.run(deliver_click(target, session_id))
    except Exception:  # noqa: BLE001 — a routing failure must fall back, never raise
        logger.debug("viewer click routing failed", exc_info=True)
        return ViewerOutcome(detail="viewer routing failed")
