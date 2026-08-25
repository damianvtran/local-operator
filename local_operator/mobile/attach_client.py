"""The attach client: a follower terminal's half of the control socket.

Every interactive ``lop`` process hosts a :class:`~local_operator.mobile.registrant.Registrant`
whose loopback socket is the phone's window onto the session. This module is
the SAME socket seen from a second terminal: ``/resume`` of a session another
process owns dials that owner and renders its projection repaints, steering
through the same ops the phone uses. One socket, N front ends — a second
protocol would drift from the first, so there is none.

Design constraints baked in:

- **No auto-reconnect.** Owner death (socket EOF) is terminal for the
  CONNECTION — never papered over by redialing a pid that may have been
  reused. The callback fires once and the client is dead. What the HOST does
  next changed in v4: ``RemoteSession`` runs a silent reattach-or-takeover
  loop (re-discover the owner, or become it through the normal resume
  factory) instead of showing a decision card, but each loop iteration still
  builds a FRESH client against a freshly discovered record.
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
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from local_operator.mobile.registry import scan
from local_operator.mobile.types import (
    PROTOCOL_VERSION,
    ContinuationCommand,
    SessionProjection,
    SessionRecord,
    _projection_from_json,
)

#: How long to wait for an ack/error matching a request id. Mirrors the
#: daemon's ``request`` timeout: long enough for a turn-boundary op (prompt
#: acquires the turn lock) on a busy owner, short enough that a wedged owner
#: surfaces as an error rather than a hang.
ACK_TIMEOUT_S = 15.0


def find_owner_record(config_dir: Path, session_id: str) -> tuple[SessionRecord | None, int | None]:
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

    The host supplies projection/disconnect callbacks (and, in v4 events mode,
    raw event + sync callbacks). All fire on the client's reader task, so a UI
    host must marshal widget work onto its message pump. The client is
    single-use: after ``on_disconnected`` it is dead by design.
    """

    def __init__(
        self,
        on_projection: Callable[[SessionProjection], None],
        on_disconnected: Callable[[str], None],
        *,
        events: bool = False,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        on_attach_sync: Callable[[dict[str, Any]], None] | None = None,
        frontend_state: bool = False,
        on_frontend_sync: Callable[[dict[str, Any]], None] | None = None,
        on_frontend_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._on_projection = on_projection
        self._on_disconnected = on_disconnected
        # v4 events mode: subscribe to the owner's raw AgentEvent relay. The
        # callbacks receive the WIRE dicts — deserialization back into concrete
        # AgentEvent subclasses is RemoteSession's job, so this transport stays
        # pydantic-free and cheap to import (module docstring contract).
        self._events = events
        self._on_event = on_event
        self._on_attach_sync = on_attach_sync
        self._frontend_state = frontend_state
        self._on_frontend_sync = on_frontend_sync
        self._on_frontend_update = on_frontend_update
        self._frontend_epoch: str | None = None
        self._frontend_sequence: int | None = None
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
            raise ConnectionError(f"owner runs protocol v{record.protocol}; attach needs >= 2")
        self._session_id = session_id
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", record.control_port, limit=1 << 20
            )
        except OSError as exc:
            raise ConnectionError(f"owner socket unreachable: {exc}") from exc
        self._reader = reader
        self._writer = writer
        auth: dict[str, Any] = {"key": record.control_key, "client": "attach"}
        if self._events:
            # v4 capability flag. A v3 owner ignores unknown auth fields and
            # simply never sends event frames — the caller gates on
            # ``record.protocol >= 4`` before relying on the relay.
            auth["events"] = True
        if self._frontend_state:
            auth["frontend_state"] = True
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
            raise ConnectionError(f"owner moved to another conversation ({projection.session_id})")
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
                elif op == "event":
                    # v4 relay frame. Deliver the raw dict; a callback failure
                    # must not kill the pump (same contract as projections).
                    if self._on_event is not None:
                        try:
                            self._on_event(frame.get("data") or {})
                        except Exception:  # noqa: BLE001
                            continue
                elif op == "attach_sync":
                    if self._on_attach_sync is not None:
                        try:
                            self._on_attach_sync(frame.get("data") or {})
                        except Exception:  # noqa: BLE001
                            continue
                elif op == "frontend_sync":
                    data = frame.get("data") or {}
                    epoch = data.get("epoch")
                    sequence = data.get("sequence")
                    if not isinstance(epoch, str) or not isinstance(sequence, int):
                        raise ConnectionError("malformed frontend sync")
                    self._frontend_epoch = epoch
                    self._frontend_sequence = sequence
                    if self._on_frontend_sync is not None:
                        self._on_frontend_sync(data)
                elif op == "frontend_update":
                    data = frame.get("data") or {}
                    epoch = data.get("epoch")
                    sequence = data.get("sequence")
                    expected = (
                        (self._frontend_sequence + 1)
                        if self._frontend_sequence is not None
                        else None
                    )
                    if epoch != self._frontend_epoch or sequence != expected:
                        # State streams are replacement-safe only when every
                        # sequence arrives. Closing forces an atomic resync; a
                        # silent skip would recreate the drift this protocol exists to prevent.
                        raise ConnectionError(
                            f"frontend state gap: expected {self._frontend_epoch}/{expected}, "
                            f"got {epoch}/{sequence}"
                        )
                    self._frontend_sequence = sequence
                    if self._on_frontend_update is not None:
                        self._on_frontend_update(data)
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

    async def prompt(
        self,
        text: str,
        *,
        command_id: str | None = None,
        images: list[dict[str, str]] | None = None,
    ) -> str:
        return await self._request(
            "prompt",
            command_id=command_id or str(uuid.uuid4()),
            text=text,
            images=list(images or []),
        )

    async def send_command(self, command: ContinuationCommand, *, streaming: bool = False) -> str:
        """Submit natural text using the latest owner projection.

        The retained command id rides both paths. If an idle projection races a
        turn start, the owner's busy rejection is reconciled once as steering;
        no second prompt is created and reconnect retries keep one identity.
        """
        if command.session_id != self._session_id:
            raise ValueError("command belongs to another conversation")
        if streaming:
            return await self.steer(
                command.text, command_id=command.command_id, images=command.images
            )
        try:
            return await self.prompt(
                command.text,
                command_id=command.command_id,
                images=command.images,
            )
        except RuntimeError as exc:
            if "already streaming" not in str(exc):
                raise
            return await self.steer(
                command.text, command_id=command.command_id, images=command.images
            )

    async def steer(
        self,
        text: str,
        *,
        command_id: str | None = None,
        images: list[dict[str, str]] | None = None,
    ) -> str:
        return await self._request(
            "steer",
            command_id=command_id or str(uuid.uuid4()),
            text=text,
            images=list(images or []),
        )

    async def abort(self) -> str:
        return await self._request("abort")

    async def slash(
        self,
        command: str,
        args: str,
        images: list[dict[str, str]] | None = None,
    ) -> str:
        return await self._request("slash", command=command, args=args, images=images or [])

    async def set_model(self, provider: str, model_id: str) -> str:
        return await self._request("set_model", provider=provider, model_id=model_id)

    async def set_effort(self, effort: str) -> str:
        return await self._request("set_effort", effort=effort)

    async def approval_answer(self, request_id: str, approved: bool) -> str:
        return await self._request(
            "approval_answer", request_id=request_id, approved=approved, remember=False
        )

    async def ask_answer(
        self, request_id: str, value: str, *, question_index: int | None = None
    ) -> str:
        fields: dict[str, Any] = {"request_id": request_id, "value": value}
        if question_index is not None:
            # The stale-answer guard (U8): name the question that was on
            # screen when the user answered, so an advanced picker refuses it.
            fields["question_index"] = question_index
        return await self._request("ask_answer", **fields)

    async def recall_steer(self, command_id: str) -> str:
        """Unsend the queued steer submitted under ``command_id`` (v4)."""
        return await self._request("recall_steer", command_id=command_id)

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


async def continue_command(
    config_dir: Path,
    command: ContinuationCommand,
    *,
    deadline_s: float = ACK_TIMEOUT_S,
    on_projection: Callable[[SessionProjection], None] | None = None,
) -> tuple[AttachClient, str]:
    """Deliver one retained command to whichever host wins the session lease.

    Every contender may start a candidate. The atomic transcript lease, never
    a check before spawning, decides authority. Losing candidates exit and the
    producer redials the published winner with the unchanged command id.
    """
    deadline = time.monotonic() + deadline_s
    spawned = False
    delay = 0.1
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        record, _ = await asyncio.to_thread(find_owner_record, config_dir, command.session_id)
        if record is not None:
            disconnected = asyncio.Event()
            client = AttachClient(
                on_projection or (lambda projection: None),
                lambda reason: disconnected.set(),
            )
            try:
                await client.connect(record, command.session_id)
                detail = await client.send_command(command)
                return client, detail
            except (ConnectionError, RuntimeError, TimeoutError) as exc:
                last_error = exc
                client.close()
        if not spawned:
            # Only routing data enters the environment. Prompt text, images and
            # command identity stay on the authenticated loopback connection.
            env = dict(os.environ)
            env["LOP_MOBILE_CHILD_CWD"] = str(Path.home())
            env["LOP_MOBILE_CHILD_RESUME"] = command.session_id
            await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "local_operator.mobile.child",
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            spawned = True
        await asyncio.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        delay = min(delay * 1.7, 1.0)
    raise TimeoutError("Couldn’t continue this conversation. Try again.") from last_error
