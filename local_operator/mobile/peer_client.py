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

    The 1 MB read ``limit`` matches the registrant's own listener limit; the
    CLI caps the body well below that so a huge paste can never become a
    silently dropped oversized line.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port, limit=1 << 20)
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
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=deadline_s)
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
