"""One-shot sender client for peer-to-peer session messaging (`lop send`).

A ``lop send`` invocation is a short-lived process that dials another local
session's registrant control socket, speaks exactly one ``peer_message`` op,
reads the ack, and exits. It is deliberately NOT built on ``AttachClient``:
that client authenticates as ``client: "attach"``, which makes the registrant
treat it as a follower terminal, grants it attach capabilities, and counts it
against ``ATTACH_MAX_CLIENTS``. A fire-and-forget sender wants none of that —
it dials as a plain daemon-class connection (the default when the auth frame
omits ``client``), which perturbs no attach accounting.

The trust boundary is the same one the whole mobile stack relies on: the
record's ``control_key`` lives in a 0600 file under a 0700 directory, so
holding it already proves same-account ownership. Loopback + the key is the
entire authorization story; there is no cross-account path to guard.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from local_operator.mobile.types import SessionRecord

#: Bytes we are willing to buffer for a SINGLE frame before deciding it is an
#: oversized ``welcome`` we don't care about and discarding it (see
#: ``_FrameReader``). Generous enough that any real ack frame fits with room to
#: spare, bounded so a pathological registrant cannot make the sender buffer
#: without limit. This is a memory cap, NOT a line-length ceiling: unlike
#: ``StreamReader.readline``'s limit, exceeding it discards the line and keeps
#: reading instead of raising.
_MAX_FRAME_BYTES = 1 << 23

#: One socket read. Independent of ``_MAX_FRAME_BYTES``; just the granularity
#: at which we pull from the kernel while scanning for a frame terminator.
_READ_CHUNK = 1 << 16


class _FrameReader:
    """Newline-framed reader with NO line-length ceiling.

    The reason this exists instead of ``reader.readline()``: a ``lop send``
    dials daemon-class, so the registrant's FIRST response is an unsolicited
    full-projection ``welcome`` serialized as one newline-terminated JSON line
    (``registrant._push_to``). A projection is unbounded in principle (a large
    transcript tail), so for a busy target that welcome line can exceed any
    fixed ``StreamReader`` limit, and ``readline`` then raises
    ``LimitOverrunError`` WITHOUT consuming the buffer — every subsequent
    ``readline`` re-raises on the same bytes, so the sender cannot reach its
    ack and crashes (U1).

    We instead frame lines from our own buffer over ``reader.read(n)``, which
    has no line limit. An oversized line (more than ``_MAX_FRAME_BYTES`` with
    no terminator yet) is DISCARDED as we scan, not retained, so memory stays
    bounded; crucially, bytes that arrive AFTER a line's newline in the same
    TCP read are kept, so the ack that the kernel coalesced behind the welcome
    is never thrown away.
    """

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader
        self._buf = bytearray()
        # True while we are discarding the tail of a line already judged
        # oversized: we drop bytes until its terminating newline, then resume.
        self._skipping = False

    async def next_line(self, timeout: float) -> bytes:
        """Return the next ``\\n``-terminated line, or ``b""`` on EOF.

        ``timeout`` bounds each socket read, matching the previous per-read
        deadline semantics.
        """
        while True:
            nl = self._buf.find(b"\n")
            if nl != -1:
                line = bytes(self._buf[: nl + 1])
                del self._buf[: nl + 1]
                if self._skipping:
                    # This newline ended the oversized line we were skipping;
                    # anything after it (already removed above) is the next
                    # frame and stays in the buffer for the following call.
                    self._skipping = False
                    continue
                return line
            # No terminator buffered yet. If the pending line is already over
            # the cap it is the oversized welcome: drop what we hold and keep
            # scanning incoming bytes for its newline rather than growing.
            if len(self._buf) > _MAX_FRAME_BYTES:
                self._buf.clear()
                self._skipping = True
            chunk = await asyncio.wait_for(self._reader.read(_READ_CHUNK), timeout=timeout)
            if not chunk:
                return b""
            self._buf.extend(chunk)


async def send_peer_message(
    record: SessionRecord,
    *,
    text: str,
    mode: str,
    wake: bool,
    sender: dict[str, Any],
    deadline_s: float = 5.0,
) -> str:
    """Deliver one message to ``record``'s session and return the ack detail.

    Dials as a daemon-class connection: a daemon-class dial receives an
    unsolicited ``welcome``/``projection`` push first, so we must read frames
    until the ``ack``/``error`` matching our request id, skipping intervening
    ``projection`` pushes — the same req-matching the attach client does.

    Raises ``RuntimeError`` on an ``error`` reply (e.g. an older registrant
    that does not know the op, or a handle that cannot receive), and
    ``ConnectionError`` if the session closes before acking. Both are soft
    failures the CLI surfaces as a human-readable message with a non-zero exit,
    never a traceback.

    Frames are read through ``_FrameReader`` so an oversized ``welcome``
    projection (the target's own large transcript, not our body) is tolerated
    rather than crashing the sender; the CLI still caps the body well below any
    memory limit so a huge paste can never become a silently dropped line.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
    try:
        # Auth frame: the bare key (no ``client``) => daemon-class connection.
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
        req = 1
        writer.write(
            json.dumps(
                {
                    "op": "peer_message",
                    "req": req,
                    "text": text,
                    "mode": mode,
                    "wake": wake,
                    "sender": sender,
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        frames = _FrameReader(reader)
        while True:
            line = await frames.next_line(deadline_s)
            if not line:
                raise ConnectionError("session closed the connection before acking")
            try:
                frame = json.loads(line.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                # A malformed line on an authenticated loopback socket is noise;
                # keep reading for our ack rather than crashing the sender.
                continue
            if not isinstance(frame, dict):
                continue
            # Skip the welcome/projection pushes a daemon-class dial receives;
            # only our matching ack/error terminates the read.
            if frame.get("req") == req and frame.get("op") in ("ack", "error"):
                if frame["op"] == "error":
                    raise RuntimeError(str(frame.get("message", "delivery failed")))
                return str(frame.get("detail", "delivered"))
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionError):
            pass
