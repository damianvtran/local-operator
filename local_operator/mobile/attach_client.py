"""The attach client: a follower terminal's half of the control socket.

Every interactive ``lop`` process hosts a :class:`~local_operator.mobile.registrant.Registrant`
whose loopback socket is the phone's window onto the session. This module is
the SAME socket seen from a second terminal: ``/resume`` of a session another
process owns dials that owner and renders its projection repaints, steering
through the same ops the phone uses. One socket, N front ends — a second
protocol would drift from the first, so there is none.

Design constraints baked in:

- **No auto-reconnect.** Owner death (socket EOF) is a DECISION POINT for the
  hosting screen — "resume here / detach" — not something to paper over by
  redialing a pid that may have been reused. The callback fires once and the
  client is dead.
- **Identity over pid trust.** ``live_session_owner`` cannot probe pids on
  Windows, and a recycled pid anywhere defeats pid trust. After auth the
  registrant sends a full projection unprompted; the client requires that
  projection's ``session_id`` to match the one the user asked for before
  declaring the attach good. A mismatch means the owner rebound away — the
  caller surfaces the graceful refusal copy.
- **Protocol gate before dialing.** A v1 registrant treats ANY authenticated
  dial as THE daemon and evicts the real one; requiring ``record.protocol
  >= 2`` turns that hazard into the graceful degradation path instead.

Stdlib-only plus the mobile wire types: the CLI imports this lazily on the
owned-resume branch only, keeping ``resume.py`` and the startup path light.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from pathlib import Path
from typing import Any, Callable

from local_operator.mobile.registry import scan
from local_operator.mobile.types import (
    PROTOCOL_VERSION,
    SessionProjection,
    SessionRecord,
    _projection_from_json,
)

#: How long to wait for an ack/error matching a request id. Mirrors the
#: daemon's ``request`` timeout: long enough for a turn-boundary op (prompt
#: acquires the turn lock) on a busy owner, short enough that a wedged owner
#: surfaces as an error rather than a hang.
ACK_TIMEOUT_S = 15.0


def find_owner_record(
    config_dir: Path, session_id: str
) -> tuple[SessionRecord | None, int | None]:
    """Locate the discovery record of the live process hosting ``session_id``.

    Returns ``(record, owner_pid)``. The normal case matches a record whose
    ``session_id`` equals the ask. The rebind-race fallback: the record's
    session_id is re-stamped only every heartbeat (15s), so when no record
    matches but the claim marker names a live pid, that pid's record is
    returned anyway and the welcome projection's identity check (in
    :meth:`AttachClient.connect`) arbitrates — a stale match costs one refused
    dial, never a wrong attach.

    ``(None, pid)`` means an owner exists but no usable record does (old
    binary, registrant failed to start): the caller degrades gracefully.
    ``(None, None)`` means no owner at all.
    """
    from local_operator.resume import live_session_owner

    owner = live_session_owner(config_dir, session_id)
    if owner is None:
        return None, None
    best: SessionRecord | None = None
    fallback: SessionRecord | None = None
    try:
        for record, state in scan(config_dir):
            if state != "live":
                continue
            if record.pid == owner and record.session_id == session_id:
                best = record
                break
            if record.pid == owner:
                fallback = record
    except OSError:
        return None, owner
    if best is not None and best.protocol >= 2:
        return best, owner
    # A v1 record is not dialable (see module docstring) — report the owner so
    # the caller can print the refusal naming it.
    if best is not None or fallback is not None:
        return None, owner
    return None, owner


class AttachClient:
    """One authenticated ``attach`` connection to a live session's registrant.

    The host (an ``AttachScreen``) supplies two callbacks — both fire on the
    client's own reader task, so the host must hop to its UI thread. The
    client is single-use: after ``on_disconnected`` it is dead by design.
    """

    def __init__(
        self,
        on_projection: Callable[[SessionProjection], None],
        on_disconnected: Callable[[str], None],
    ) -> None:
        self._on_projection = on_projection
        self._on_disconnected = on_disconnected
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[Any, asyncio.Future[dict[str, Any]]] = {}
        self._req_seq = 0
        self._session_id = ""
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self, record: SessionRecord, session_id: str) -> None:
        """Dial, authenticate as an attach client, and verify identity.

        Raises on any failure (dial refused, auth rejected, welcome identity
        mismatch, protocol too old): the caller collapses every one to the
        graceful refusal copy — a user re-running the command is the only
        retry mechanism, by design.
        """
        if record.protocol < 2:
            raise ConnectionError(
                f"owner runs protocol v{record.protocol}; attach needs >= 2"
            )
        self._session_id = session_id
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", record.control_port, limit=1 << 20
            )
        except OSError as exc:
            raise ConnectionError(f"owner socket unreachable: {exc}") from exc
        self._reader = reader
        self._writer = writer
        auth = {"key": record.control_key, "client": "attach"}
        writer.write(json.dumps(auth).encode() + b"\n")
        await writer.drain()
        # The welcome projection doubles as the identity check: it names the
        # conversation the OWNER is actually hosting right now, which is the
        # fact the user cares about and the one a pid cannot prove.
        try:
            first = await asyncio.wait_for(reader.readline(), timeout=ACK_TIMEOUT_S)
        except TimeoutError as exc:
            raise ConnectionError("owner did not send its state") from exc
        if not first:
            raise ConnectionError("owner closed the connection")
        try:
            frame = json.loads(first.decode("utf-8", "replace"))
        except ValueError as exc:
            raise ConnectionError("owner sent a malformed frame") from exc
        if frame.get("op") not in ("projection", "welcome"):
            raise ConnectionError(f"owner replied {frame.get('op')!r}, not its state")
        projection = _projection_from_json(frame.get("data") or {}, record)
        if projection.session_id != session_id:
            raise ConnectionError(
                f"owner moved to another conversation ({projection.session_id})"
            )
        self._connected = True
        self._reader_task = asyncio.get_running_loop().create_task(self._pump())
        # Deliver the welcome synchronously so the host paints before any
        # later repaint can race it.
        self._on_projection(projection)

    async def _pump(self) -> None:
        """Read frames until EOF; route projections and match acks by req id."""
        reader = self._reader
        assert reader is not None
        reason = "owner exited"
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    frame = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                op = frame.get("op")
                if op in ("projection", "welcome"):
                    try:
                        # pid=0 record: the attach screen keys nothing on the
                        # record's pid (it reads the projection's own), and
                        # building a fake record per repaint would imply the
                        # record carries truth it does not.
                        self._on_projection(
                            _projection_from_json(
                                frame.get("data") or {},
                                SessionRecord(
                                    pid=0,
                                    kind="tui",
                                    session_id="",
                                    conversation_name="",
                                    cwd="",
                                    model_label="",
                                    control_port=0,
                                    control_key="",
                                    protocol=PROTOCOL_VERSION,
                                ),
                            )
                        )
                    except Exception:  # noqa: BLE001 — a malformed push must not kill the pump
                        continue
                elif op in ("ack", "error"):
                    future = self._pending.pop(frame.get("req"), None)
                    if future is not None and not future.done():
                        future.set_result(frame)
        except (ConnectionResetError, BrokenPipeError, OSError):
            reason = "owner connection reset"
        finally:
            self._connected = False
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError(reason))
            self._pending.clear()
            try:
                self._writer.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            self._on_disconnected(reason)

    # -- requests ---------------------------------------------------------------

    async def _request(self, op: str, **fields: Any) -> str:
        """Send one op and await its ack detail (or raise its error message)."""
        if not self._connected or self._writer is None:
            raise ConnectionError("not attached")
        self._req_seq += 1
        req = self._req_seq
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req] = future
        frame = {"op": op, "req": req, **fields}
        try:
            self._writer.write(json.dumps(frame).encode() + b"\n")
            await self._writer.drain()
            reply = await asyncio.wait_for(future, timeout=ACK_TIMEOUT_S)
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            self._pending.pop(req, None)
            raise ConnectionError(f"owner connection lost: {exc}") from exc
        finally:
            self._pending.pop(req, None)
        if reply.get("op") == "error":
            raise RuntimeError(str(reply.get("message", "request failed")))
        return str(reply.get("detail", ""))

    async def prompt(self, text: str) -> str:
        return await self._request("prompt", text=text)

    async def steer(self, text: str) -> str:
        return await self._request("steer", text=text)

    async def abort(self) -> str:
        return await self._request("abort")

    async def slash(self, command: str, args: str) -> str:
        return await self._request("slash", command=command, args=args)

    async def set_model(self, provider: str, model_id: str) -> str:
        return await self._request("set_model", provider=provider, model_id=model_id)

    async def set_effort(self, effort: str) -> str:
        return await self._request("set_effort", effort=effort)

    async def approval_answer(self, request_id: str, approved: bool) -> str:
        return await self._request(
            "approval_answer", request_id=request_id, approved=approved, remember=False
        )

    async def ask_answer(self, request_id: str, value: str) -> str:
        return await self._request("ask_answer", request_id=request_id, value=value)

    async def detach(self) -> None:
        """Close the connection from our side. ``on_disconnected`` still fires
        (the pump observes EOF) so the host's teardown runs one path."""
        self._connected = False
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        """Synchronous teardown for hosts without a loop (app exit paths)."""
        self._connected = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
