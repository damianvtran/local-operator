"""The mobile daemon: one process, the phone-facing control plane.

``lop mobile serve`` runs this. It is deliberately small because the design
moved the hard parts elsewhere: sessions fold themselves into projections
(:mod:`.projection`), discovery is a directory scan (:mod:`.registry`), auth
is a signed cookie (:mod:`.auth`). What remains here is plumbing with three
moving parts:

- **Adoption** — scan the record directory, dial each live registrant's
  control socket with the record's key, and relay its projection pushes into
  an in-memory table the web layer reads. Owned sessions (started from the
  phone) register through the SAME socket path — every phone-visible session
  has one shape, so the web layer never branches on ownership.
- **The web app** — a Starlette application serving the built SPA, a small
  REST surface, and the SSE stream. SSE, never WebSocket, on the phone leg:
  an identity proxy's login redirect cannot be followed by a WebSocket
  handshake, so every realtime push here is an EventSource repaint.
- **Repaint, not deltas** — every push is the whole projection for one
  session. No delta protocol means no drift; caps in the fold keep repaints
  cheap.

Threading: one asyncio loop. Registrants run their own loops in their own
processes; this loop only dials them. Blocking work (session construction,
which reads provider catalogues) goes through ``asyncio.to_thread`` so a
phone starting a session never stalls the SSE streams of the others.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from local_operator.mobile import registry
from local_operator.mobile.auth import (
    COOKIE_NAME,
    check_password,
    sign_cookie,
    verify_cookie,
)
from local_operator.mobile.types import (
    PROTOCOL_VERSION,
    SessionProjection,
    SessionRecord,
)

logger = logging.getLogger(__name__)

#: How often the record directory is scanned. Records change rarely (a
#: session starts or dies); the scan is cheap, and 2 s makes a new terminal
#: session appear on the phone before the user reaches for it.
SCAN_INTERVAL_S = 2.0

#: Backoff before re-dialing a refused control socket. A registrant whose
#: record is fresh but whose socket refuses is mid-restart; hammering it
#: buys nothing.
REDIAL_BACKOFF_S = 5.0

#: SSE keepalive cadence — under the 60 s idle cutoff of common proxies.
SSE_KEEPALIVE_S = 25.0

#: Default daemon port. Loopback only; remote access is a tunnel's job.
DEFAULT_PORT = 4098

_WEB_DIR = Path(__file__).parent / "web"
_DIST_DIR = _WEB_DIR / "dist"
#: The cropped LO mark (figure with a raised hand) served to the login page
#: and, via the same path, to the SPA. Lives next to this module so the
#: login HTML never depends on a Vite rebuild.
_STATIC_DIR = Path(__file__).parent / "static"


def _mark_data_uri() -> str:
    """The mark as a data URI — the login page and the SPA header inline it
    rather than fetch ``/mark.png``, because over an identity-proxied tunnel
    (Cloudflare Access) that fetch is itself gated: the pre-auth login page's
    <img> got a 302-to-IdP HTML body and rendered the broken-image glyph.
    An inline URI needs no request, so it renders behind Access and on the
    unauthenticated login screen alike. 7 KB; one copy in each surface.
    """
    import base64

    data = base64.b64encode((_STATIC_DIR / "mark.png").read_bytes()).decode()
    return "data:image/png;base64," + data


# ---------------------------------------------------------------------------
# Session table
# ---------------------------------------------------------------------------


class SessionEntry:
    """One phone-visible session: its latest projection plus the dial state.

    The projection arrives whole on every push; ``subscribers`` are the open
    SSE queues waiting for repaints of this session.
    """

    def __init__(self, record: SessionRecord) -> None:
        self.record = record
        self.projection: SessionProjection | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.next_dial_at: float = 0.0
        self.degraded = False
        self.ended = False
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        # Monotonic request id for control frames we originate.
        self._req_seq = 0

    def next_req(self) -> int:
        self._req_seq += 1
        return self._req_seq


class SessionTable:
    """The daemon's whole runtime state. One instance, owned by the loop."""

    def __init__(self) -> None:
        self.entries: dict[int, SessionEntry] = {}  # by pid
        self.list_subscribers: set[asyncio.Queue[None]] = set()

    def summaries(self) -> list[dict[str, Any]]:
        """The session-list payload: one summary per session, live first,
        then wedged, then ended — the order the phone list renders."""
        out = []
        for entry in self.entries.values():
            p = entry.projection
            out.append(
                {
                    "pid": entry.record.pid,
                    "kind": entry.record.kind,
                    "session_id": entry.record.session_id,
                    "conversation_name": (
                        p.conversation_name if p else entry.record.conversation_name
                    ),
                    "cwd": p.cwd if p else entry.record.cwd,
                    "model_label": p.model_label if p else entry.record.model_label,
                    "streaming": bool(p and p.streaming),
                    "ended": entry.ended,
                    "degraded": entry.degraded,
                    "needs_attention": bool(p and p.pending),
                    "pending_kind": (p.pending.kind if p and p.pending else ""),
                    "subagents_running": (
                        sum(1 for s in p.subagents if s.status == "running") if p else 0
                    ),
                    "todos_open": (sum(1 for t in p.todos if t.status == "pending") if p else 0),
                }
            )
        out.sort(key=lambda s: (s["ended"], s["degraded"], -s["pid"]))
        return out

    def notify_list_changed(self) -> None:
        for queue in self.list_subscribers:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass


# ---------------------------------------------------------------------------
# Registrant connections
# ---------------------------------------------------------------------------


async def _dial(daemon: "MobileDaemon", entry: SessionEntry) -> None:
    """Open (or re-open) the control socket to one registrant and pump its
    frames until the connection dies. One task per session."""
    record = entry.record
    try:
        # Match the registrant's 1 MB line limit. The default 64 KB
        # StreamReader cap is what made a transcript push raise
        # ValueError and leave the session stuck on "connecting…".
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
    except OSError:
        entry.degraded = True
        entry.next_dial_at = time.monotonic() + REDIAL_BACKOFF_S
        return
    entry.writer = writer
    entry.degraded = False
    try:
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
        while True:
            try:
                line = await reader.readline()
            except ValueError:
                # A frame longer than the stream limit is not a reason to
                # drop the session — a transcript push can outgrow 64 KB.
                # Skip the oversized line and keep the connection.
                logger.warning("mobile daemon: oversized control frame from pid %s", record.pid)
                continue
            if not line:
                break
            try:
                frame = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                continue
            op = frame.get("op")
            if op in ("projection", "welcome"):
                try:
                    data = frame.get("data") or {}
                    entry.projection = _projection_from_json(data, record)
                except (TypeError, ValueError, KeyError):
                    # A malformed push (mid-upgrade registrant, renamed field)
                    # must not tear the dial loop down to the reconnect path —
                    # the NEXT push is a full repaint that repairs the view.
                    logger.debug("mobile daemon: dropping malformed projection", exc_info=True)
                    continue
                entry.projection.degraded = entry.degraded
                entry.projection.ended = entry.ended
                _fan_out(entry)
            # acks/errors are matched by req id in _request's future map.
            pending = daemon._pending_reqs.pop((record.pid, frame.get("req")), None)
            if pending is not None and not pending.done():
                pending.set_result(frame)
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        if entry.writer is writer:
            entry.writer = None
        entry.next_dial_at = time.monotonic() + REDIAL_BACKOFF_S
        entry.degraded = not entry.ended
        _fan_out(entry)


def _projection_from_json(data: dict[str, Any], record: SessionRecord) -> SessionProjection:
    """Rebuild a projection from a wire payload. The registrant already
    serialized dataclasses; here we tolerate missing keys (a rolling upgrade
    mid-push) by constructing through the dataclass with defaults."""
    from dataclasses import fields

    from local_operator.mobile.types import (
        PendingRequest,
        SubagentRow,
        TodoItem,
        TranscriptEntry,
    )

    def build(cls: type, items: list[dict[str, Any]]) -> list[Any]:
        known = {f.name for f in fields(cls)}
        return [cls(**{k: v for k, v in item.items() if k in known}) for item in items]

    known = {f.name for f in fields(SessionProjection)}
    base = {
        k: v
        for k, v in data.items()
        if k in known and k not in ("transcript", "todos", "subagents", "pending")
    }
    projection = SessionProjection(**base)
    # The fold stamps pid=0 (the registrant does not know its own listen
    # pid until after the record is published). The discovery record is
    # the source of truth, and the phone keys drafts and commands on it.
    projection.pid = record.pid
    projection.transcript = build(TranscriptEntry, data.get("transcript", []))
    _pin_opening_user_message(projection, record)
    projection.todos = build(TodoItem, data.get("todos", []))
    projection.subagents = build(SubagentRow, data.get("subagents", []))
    pending = data.get("pending")
    projection.pending = (
        PendingRequest(
            **{k: v for k, v in pending.items() if k in {f.name for f in fields(PendingRequest)}}
        )
        if isinstance(pending, dict)
        else None
    )
    return projection


def _pin_opening_user_message(projection: SessionProjection, record: SessionRecord) -> None:
    """Guarantee the transcript opens with the conversation's first user
    message, even when the SESSION that folded it is running older code.

    Two independent gaps hid it: the harness never emits MessageStartEvent
    for user messages (fixed in the handle), and the 80-entry tail cap drops
    the opening prompt on any long session (fixed in the fold). Both fixes
    live in the session's own process — so a session on an older binary
    still pushes a wire projection with no user rows. The daemon can't fix
    the session's fold, but it CAN repair the view: read the opening user
    turn from the on-disk transcript (the same store /resume reads) and pin
    it at the head. Idempotent — a projection that already opens with a user
    row is left alone.
    """
    transcript = projection.transcript
    if any(e.kind == "user" for e in transcript):
        return
    try:
        from local_operator.paths import config_dir

        path = config_dir() / "sessions" / record.session_id / "transcript.jsonl"
        if not path.exists():
            return
        # Read only the head: the opening user turn is within the first few
        # entries, and a 10 MB transcript should not be replayed per repaint.
        import json as _json

        with path.open() as fh:
            # Scan a bounded head, not the whole file: the opening user turn
            # is normally within the first few entries, but if it was pruned
            # or compacted away the first surviving user message can sit
            # arbitrarily deep, and this runs on every projection reload.
            # Give up after MAX_SCAN lines — a session whose opening prompt
            # no longer exists simply has nothing to pin.
            MAX_SCAN = 400
            for i, line in enumerate(fh):
                if i >= MAX_SCAN:
                    return
                try:
                    entry = _json.loads(line)
                except ValueError:
                    continue
                if entry.get("type") != "message":
                    continue
                payload = entry.get("payload") or {}
                if payload.get("role") != "user":
                    continue
                # Transcript text blocks are stored as {"text": ...} WITHOUT a
                # "type" discriminator (the in-memory TextContent adds it), so
                # match on the text key itself rather than a type field.
                text = "".join(
                    block["text"]
                    for block in payload.get("content", [])
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                )
                if not text.strip():
                    continue
                from local_operator.mobile.types import TranscriptEntry

                projection.transcript = [
                    TranscriptEntry(
                        # The transcript persists message.id as the entry id,
                        # so it is always present — use it, no pid fallback.
                        id=entry["id"],
                        kind="user",
                        text=text,
                        final=True,
                    ),
                    *transcript,
                ]
                return
    except Exception:  # noqa: BLE001 — a missing/odd transcript must never break a repaint
        return


def _transcript_entry_json(entry: Any) -> dict[str, Any]:
    """Serialize one mobile TranscriptEntry for the history payload."""
    return entry.to_json()


def _history_page(record: SessionRecord, before: str | None, limit: int) -> tuple[list[Any], bool]:
    """Fold the session's full on-disk transcript and return the page of
    entries immediately OLDER than ``before`` (chronological within the page)
    plus whether more history exists beyond it.

    Runs off the event loop (``asyncio.to_thread`` at the call site): folding
    a long transcript rehydrates every message and is not loop-safe work.
    """
    from local_operator.mobile.projection import fold_messages_to_entries
    from local_operator.paths import config_dir
    from local_operator.session.transcript import Transcript

    directory = config_dir() / "sessions" / record.session_id
    if not (directory / "transcript.jsonl").exists():
        return [], False
    try:
        transcript = Transcript(directory)
        history = transcript.build_llm_history()
        entries = fold_messages_to_entries(history)
    except Exception:  # noqa: BLE001 — an odd transcript yields no history, not a 500
        logger.exception("history fold failed for session %s", record.session_id)
        return [], False

    if before:
        cut = next((i for i, e in enumerate(entries) if e.id == before), len(entries))
    else:
        cut = len(entries)
    older = entries[:cut]
    page = older[-limit:] if len(older) > limit else older
    has_more = len(older) > len(page)
    return page, has_more


def _fan_out(entry: SessionEntry) -> None:
    """Push the current projection to every open SSE queue for this session.
    A slow phone's queue fills; repaints supersede, so evict the oldest and
    retry until the put lands — a repaint must never be silently lost, because
    the dropped one might be the approval card."""
    if entry.projection is None:
        return
    frame = entry.projection.to_json()
    for queue in entry.subscribers:
        while True:
            try:
                queue.put_nowait(frame)
                break
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break  # racing consumer drained it; retry the put


# ---------------------------------------------------------------------------
# The daemon
# ---------------------------------------------------------------------------


class MobileDaemon:
    def __init__(self, *, port: int = DEFAULT_PORT, password: str | None = None) -> None:
        self.port = port
        self.password = password
        self.table = SessionTable()
        self._pending_reqs: dict[tuple[int, Any], asyncio.Future[dict[str, Any]]] = {}
        self._dial_tasks: dict[int, asyncio.Task[None]] = {}
        self._slash_commands: list[dict[str, Any]] | None = None

    # -- scanning --------------------------------------------------------------

    async def scan_loop(self) -> None:
        while True:
            try:
                await self._scan_once()
            except Exception:  # noqa: BLE001 — the scan must never die
                logger.warning("mobile daemon scan failed", exc_info=True)
            await asyncio.sleep(SCAN_INTERVAL_S)

    async def _scan_once(self) -> None:
        seen: set[int] = set()
        changed = False
        for record, state in await asyncio.to_thread(registry.scan):
            seen.add(record.pid)
            entry = self.table.entries.get(record.pid)
            if entry is None:
                entry = SessionEntry(record)
                self.table.entries[record.pid] = entry
                changed = True
            else:
                # Re-adopt record updates (model label, name, /resume's new
                # session id) — the socket survives them by design.
                entry.record = record
            if state == "stale":
                entry.ended = True
                changed = True
            elif state == "wedged":
                entry.degraded = True
            # Degraded is precisely "we owe this session a redial" — the only
            # gates are ended, an open socket, and the backoff clock. Excluding
            # degraded entries here was the starvation bug: one refused dial
            # meant never trying again.
            if not entry.ended and entry.writer is None:
                if time.monotonic() >= entry.next_dial_at and (
                    record.pid not in self._dial_tasks or self._dial_tasks[record.pid].done()
                ):
                    self._dial_tasks[record.pid] = asyncio.ensure_future(_dial(self, entry))
        # Reap entries whose record vanished entirely.
        for pid in list(self.table.entries):
            if pid not in seen:
                entry = self.table.entries[pid]
                if not entry.ended:
                    entry.ended = True
                    changed = True
        if changed:
            self.table.notify_list_changed()

    # -- control requests ---------------------------------------------------------

    async def request(self, pid: int, op: str, **fields: Any) -> dict[str, Any]:
        """Send one control frame to a session and await its ack/error."""
        entry = self.table.entries.get(pid)
        if entry is None or entry.writer is None:
            raise KeyError(f"session {pid} is not connected")
        req = entry.next_req()
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_reqs[(pid, req)] = future
        frame = {"op": op, "req": req, **fields}
        try:
            entry.writer.write(json.dumps(frame).encode() + b"\n")
            await entry.writer.drain()
            reply = await asyncio.wait_for(future, timeout=15.0)
        finally:
            self._pending_reqs.pop((pid, req), None)
        if reply.get("op") == "error":
            raise RuntimeError(str(reply.get("message", "request failed")))
        return reply

    # -- owned sessions ---------------------------------------------------------

    async def spawn_session(
        self, cwd: str, provider: str | None = None, model_id: str | None = None
    ) -> int:
        """Spawn a daemon-owned session in a supervised CHILD process and let
        discovery adopt it.

        A child process, not an in-process session: the daemon is supervisable
        state (launchd restarts it), and a session living inside it would die
        with every restart — taking an in-flight turn with it. A child with
        its own pid gets the same lifetime as a terminal session: the daemon
        going away costs the phone its view, never the session its work. The
        child runs the registrant standalone (``python -m
        local_operator.mobile.child``), so the record + control socket path is
        literally the same code the TUI uses.
        """
        import sys

        env = dict(os.environ)
        env["LOP_MOBILE_CHILD_CWD"] = cwd
        if provider:
            env["LOP_MOBILE_CHILD_PROVIDER"] = provider
        if model_id:
            env["LOP_MOBILE_CHILD_MODEL"] = model_id
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "local_operator.mobile.child",
            env=env,
            # Detached stdio: the child speaks through its record and socket;
            # a pipe back to the daemon would die with the daemon and take
            # the child's stdout with it.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return process.pid

    # -- slash command catalogue ----------------------------------------------------

    def slash_commands(self) -> list[dict[str, Any]]:
        """The phone's slash sheet. Imported lazily (the TUI registry pulls
        the app's command table) and cached — the registry is static."""
        if self._slash_commands is None:
            from local_operator.tui.app import SLASH_COMMANDS
            from local_operator.tui.autocomplete import ArgumentMode

            excluded = {"exit", "quit", "clear"}  # TUI chrome, meaningless on a phone
            self._slash_commands = [
                {
                    "name": cmd.name,
                    "description": cmd.description,
                    "aliases": list(cmd.aliases),
                    "arguments": (
                        cmd.arguments.name.lower()
                        if isinstance(cmd.arguments, ArgumentMode)
                        else "none"
                    ),
                }
                for cmd in SLASH_COMMANDS
                if cmd.name not in excluded
            ]
        return self._slash_commands


# ---------------------------------------------------------------------------
# Web application
# ---------------------------------------------------------------------------


def build_app(daemon: MobileDaemon):
    """The Starlette app. Imported inside the function so ``lop`` without the
    server extra never pays for starlette at startup."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
        PlainTextResponse,
        RedirectResponse,
        Response,
        StreamingResponse,
    )
    from starlette.routing import BaseRoute, Mount, Route

    # -- auth helpers -----------------------------------------------------------

    def authed(request: Request) -> bool:
        if not daemon.password:
            return False
        return verify_cookie(request.cookies.get(COOKIE_NAME), daemon.password)

    def gate(request: Request) -> Response | None:
        """None = allowed. Browsers get the login redirect, API calls a 401 —
        the split contract the health check asserts."""
        if authed(request):
            return None
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "authentication required"}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    def secure_cookie(response: Response, request: Request) -> None:
        # Secure only when the request arrived over TLS (the tunnel case);
        # plain loopback HTTP must still set the cookie or first-run dev is
        # impossible. HttpOnly + SameSite=Lax always: the cookie is never
        # read from JS and never rides a cross-site POST.
        response.headers["Cache-Control"] = "no-store"
        return None

    # -- routes -------------------------------------------------------------------

    async def healthz(request: Request) -> Response:
        return JSONResponse(
            {
                "ok": True,
                "version": PROTOCOL_VERSION,
                "sessions": len(daemon.table.entries),
                "dist": _DIST_DIR.exists(),
            }
        )

    async def login_page(request: Request) -> Response:
        if authed(request):
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(_LOGIN_HTML.replace("__MARK_DATA_URI__", _mark_data_uri()))

    async def login_submit(request: Request) -> Response:
        form = await request.form()
        candidate = str(form.get("password", ""))
        if not daemon.password or not check_password(candidate, daemon.password):
            return HTMLResponse(
                _LOGIN_HTML.replace("__MARK_DATA_URI__", _mark_data_uri()).replace(
                    "<!--ERROR-->", _LOGIN_ERROR
                ),
                status_code=401,
            )
        response = RedirectResponse("/", status_code=303)
        secure = request.url.scheme == "https"
        response.set_cookie(
            COOKIE_NAME,
            sign_cookie(daemon.password),
            max_age=30 * 24 * 3600,
            httponly=True,
            samesite="lax",
            secure=secure,
        )
        return response

    async def logout(request: Request) -> Response:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE_NAME)
        return response

    async def mark_png(request: Request) -> Response:
        """The LO mark — unauthenticated because the login page needs it
        before a cookie exists. It is a public brand asset, not a secret."""
        path = _STATIC_DIR / "mark.png"
        if not path.exists():
            return PlainTextResponse("mark missing", status_code=404)
        # no-store: a phone that loaded this while the wheel lacked the file
        # cached the 404 and kept showing a broken image after the fix. The
        # asset is tiny; the freshness guarantee is worth more than the cache.
        response = FileResponse(path, media_type="image/png")
        response.headers["Cache-Control"] = "no-store"
        return response

    async def index(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        if not _DIST_DIR.exists():
            return PlainTextResponse(
                "mobile web bundle not built — run: "
                "cd local_operator/mobile/web && pnpm install && pnpm build",
                status_code=503,
            )
        response = FileResponse(_DIST_DIR / "index.html")
        response.headers["Cache-Control"] = "no-store"  # the SPA shell; assets are hashed
        return response

    async def api_sessions(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        return JSONResponse({"sessions": daemon.table.summaries()})

    async def api_session_events(request: Request) -> Response:
        """SSE repaint stream for one session — the phone's only realtime
        channel. Opens with the current projection so a reconnecting phone
        renders immediately."""
        denied = gate(request)
        if denied is not None:
            return denied
        pid = int(request.path_params["pid"])
        entry = daemon.table.entries.get(pid)
        if entry is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=8)
        entry.subscribers.add(queue)

        async def stream():
            try:
                if entry.projection is not None:
                    yield _sse("projection", entry.projection.to_json())
                while True:
                    try:
                        frame = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_S)
                        yield _sse("projection", frame)
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                entry.subscribers.discard(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                # Identity proxies buffer SSE by default; this is the header
                # that turns buffering off at nginx-family proxies.
                "X-Accel-Buffering": "no",
            },
        )

    async def api_list_events(request: Request) -> Response:
        """SSE for the session LIST, so the home screen needs no polling."""
        denied = gate(request)
        if denied is not None:
            return denied
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=4)
        daemon.table.list_subscribers.add(queue)

        async def stream():
            try:
                yield _sse("sessions", {"sessions": daemon.table.summaries()})
                while True:
                    try:
                        await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_S)
                        yield _sse("sessions", {"sessions": daemon.table.summaries()})
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                daemon.table.list_subscribers.discard(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    async def api_session_history(request: Request) -> Response:
        """Older transcript entries for lazy loading.

        The live projection (SSE) is a tail WINDOW — the fold caps it, so a
        long session's older messages never reach the phone. This endpoint
        folds the session's FULL on-disk transcript with the same render
        semantics and serves the pages the cap dropped, so scrolling up
        back-fills history. ``before`` is the id of the oldest entry the
        phone already has; the response is the page of entries immediately
        OLDER than it (chronological within the page).
        """
        denied = gate(request)
        if denied is not None:
            return denied
        pid = int(request.path_params["pid"])
        entry = daemon.table.entries.get(pid)
        if entry is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        before = request.query_params.get("before")
        try:
            limit = max(1, min(int(request.query_params.get("limit", "80")), 200))
        except ValueError:
            limit = 80
        page, has_more = await asyncio.to_thread(_history_page, entry.record, before, limit)
        return JSONResponse(
            {
                "entries": [_transcript_entry_json(e) for e in page],
                "has_more": has_more,
            }
        )

    async def api_command(request: Request) -> Response:
        """The one mutation endpoint: {op, ...} → control frame. Keeping
        mutations on one route mirrors the registrant's dispatch and keeps
        the auth gate in one place."""
        denied = gate(request)
        if denied is not None:
            return denied
        pid = int(request.path_params["pid"])
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        op = str(body.pop("op", ""))
        try:
            reply = await daemon.request(pid, op, **body)
        except KeyError:
            return JSONResponse({"error": "session not connected"}, status_code=409)
        except TimeoutError:
            return JSONResponse({"error": "session did not answer"}, status_code=504)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse({"ok": True, "detail": reply.get("detail", "")})

    async def api_commands(request: Request) -> Response:
        denied = gate(request)
        if denied is not None:
            return denied
        return JSONResponse({"commands": daemon.slash_commands()})

    async def api_start_session(request: Request) -> Response:
        """Start a daemon-owned session in ``cwd`` and register it through
        the normal record+socket path, so it is indistinguishable from a
        terminal session to the web layer."""
        denied = gate(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        cwd_raw = str(body.get("cwd") or Path.home())
        # Resolve to a real directory under the owner's home: the picker only
        # ever sends those, and the spawn runs with the daemon's environment,
        # so the check is about fat-fingered/traversed input, not trust.
        cwd_path = Path(cwd_raw).expanduser().resolve()
        home = Path.home().resolve()
        if not cwd_path.is_dir() or not (cwd_path == home or home in cwd_path.parents):
            return JSONResponse(
                {"error": f"not a directory under home: {cwd_raw}"}, status_code=400
            )
        cwd = str(cwd_path)
        provider = body.get("provider")
        model_id = body.get("model_id")
        try:
            pid = await daemon.spawn_session(
                cwd,
                provider=str(provider) if provider else None,
                model_id=str(model_id) if model_id else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("mobile session spawn failed", exc_info=True)
            return JSONResponse({"error": str(exc)[:300]}, status_code=500)
        return JSONResponse({"ok": True, "pid": pid})

    async def api_directories(request: Request) -> Response:
        """The new-session form's cwd picker: home plus the directories of
        recent sessions (where the user has been working lately)."""
        denied = gate(request)
        if denied is not None:
            return denied
        recent = await asyncio.to_thread(_recent_directories)
        return JSONResponse({"home": str(Path.home()), "recent": recent})

    async def api_past_sessions(request: Request) -> Response:
        """Resumable past sessions — the phone's "go back to a conversation"
        list, from the same store the TUI's /resume picker reads."""
        denied = gate(request)
        if denied is not None:
            return denied
        rows = await asyncio.to_thread(_past_sessions)
        return JSONResponse({"sessions": rows})

    async def api_models(request: Request) -> Response:
        """The model sheet's catalogue: providers with stored credentials and
        their models, reusing the server's model listing so the phone and the
        desktop see the same inventory."""
        denied = gate(request)
        if denied is not None:
            return denied
        try:
            models = await asyncio.to_thread(_list_models)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)[:200]}, status_code=502)
        return JSONResponse({"models": models})

    routes: list[BaseRoute] = [
        Route("/healthz", healthz),
        Route("/login", login_page, methods=["GET"]),
        Route("/login", login_submit, methods=["POST"]),
        Route("/logout", logout),
        Route("/api/sessions", api_sessions),
        Route("/api/sessions/start", api_start_session, methods=["POST"]),
        Route("/api/sessions/events", api_list_events),
        Route("/api/directories", api_directories),
        Route("/api/sessions/past", api_past_sessions),
        Route("/api/sessions/{pid:int}/events", api_session_events),
        Route("/api/sessions/{pid:int}/history", api_session_history),
        Route("/api/sessions/{pid:int}/command", api_command, methods=["POST"]),
        Route("/api/commands", api_commands),
        Route("/api/models", api_models),
        Route("/mark.png", mark_png),
        Route("/", index),
    ]
    if _DIST_DIR.exists():
        # The mount is resolved at app build time: a rebuilt bundle needs
        # `lop mobile restart` to appear, which is the documented upgrade
        # path — per-request checks would slow every asset hit to catch a
        # once-per-upgrade event.
        routes.append(
            Mount(
                "/assets",
                app=__import__("starlette.staticfiles", fromlist=["StaticFiles"]).StaticFiles(
                    directory=_DIST_DIR / "assets"
                ),
                name="assets",
            )
        )
    return Starlette(routes=routes)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _past_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """Resumable past sessions for the phone's history list."""
    try:
        from local_operator.paths import config_dir
        from local_operator.resume import recent_session_rows

        return [
            {"id": row.id, "name": row.name, "mtime": row.mtime}
            for row in recent_session_rows(config_dir(), limit=limit)
        ]
    except Exception:  # noqa: BLE001
        return []


def _recent_directories(limit: int = 8) -> list[str]:
    """The new-session form's cwd suggestions: the working directories of
    recently active agents from the on-disk registry (the durable store both
    the CLI and server write), deduped, live directories only."""
    try:
        from local_operator.agents import AgentRegistry
        from local_operator.paths import config_dir

        registry = AgentRegistry(config_dir=config_dir())
        agents = sorted(
            registry.list_agents(),
            key=lambda agent: agent.last_message_datetime or "",
            reverse=True,
        )
        seen: list[str] = []
        for agent in agents:
            cwd = agent.current_working_directory
            if cwd and cwd not in seen and Path(cwd).is_dir():
                seen.append(cwd)
            if len(seen) >= limit:
                break
        return seen
    except Exception:  # noqa: BLE001
        return []


def _list_models() -> list[dict[str, Any]]:
    """The model sheet's rows: every model of every provider the owner can
    actually call — a provider with no stored credential is clutter in a
    picker. Credential detection consults BOTH stores, because the two
    sanctioned flows write different ones: ``lop credential update`` writes
    the legacy CredentialManager file, and ``/login`` writes the providers
    AuthStore (auth.db) — a picker reading only the first would hide every
    OAuth-logged-in provider, which on a current install is most of them.
    Runs in a thread: catalogue reads are file I/O."""
    from local_operator.credentials import CredentialManager
    from local_operator.model.registry import SupportedHostingProviders, static_models
    from local_operator.paths import config_dir
    from local_operator.providers.auth_store import AuthStore

    credential_manager = CredentialManager(config_dir=config_dir())
    store = AuthStore()
    try:
        authed_providers = {c.provider for c in store.list_credentials()}
    finally:
        store.close()

    rows: list[dict[str, Any]] = []
    for provider in SupportedHostingProviders:
        required = provider.requiredCredentials
        has_key = bool(required) and any(
            credential_manager.get_credential(key).get_secret_value() for key in required
        )
        # AuthStore login aliases: the oauth flavour of a provider logs in
        # under its own id (e.g. ``alibaba-token-plan-oauth``); the catalogue
        # key is the base id, so prefix matching covers both spellings.
        has_login = provider.id in authed_providers or any(
            p.startswith(f"{provider.id}-") for p in authed_providers
        )
        if required and not has_key and not has_login:
            continue
        if not required and not has_login:
            continue
        for model_id, info in static_models(provider.id).items():
            rows.append(
                {
                    "selector": f"{provider.id}/{model_id}",
                    "provider": provider.id,
                    "model_id": model_id,
                    "name": getattr(info, "name", "") or model_id,
                }
            )
    return rows


#: The login page is server-rendered (not part of the SPA) so the auth gate
#: has zero client-side surface: no bundle, no router state, no way for a
#: stale cached SPA to sit in front of a password form.
_LOGIN_ERROR = '<p class="error">Wrong password.</p>'

_LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#14110c">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>local operator — sign in</title>
<style>
  /* Values from local_operator/tui/theme.py BRAND_TOKENS.dark — the login
     page is server-rendered so the auth gate has zero client-side surface,
     so it tracks the TUI's own tokens by hand. Keep the two in sync: the
     TUI palette is the brand. Layout follows the TUI welcome lockup
     (welcome.py): the mark, then the letterspaced wordmark, then the
     form — no box, no accent spent on the identity. */
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    min-height: 100dvh;
    display: grid;
    place-items: center;
    padding: max(env(safe-area-inset-top), 32px) 24px max(env(safe-area-inset-bottom), 32px);
    background: #14110c;                           /* bg */
    color: #e9e5db;                                /* fg */
    font: 16px/1.5 -apple-system, "SF Pro Text", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  form {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    width: min(320px, 100%);
  }
  .lockup {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 40px;
  }
  .mark {
    width: 72px;
    height: 72px;
    display: block;
    /* The PNG is already tinted to TUI dim (#837c6d); do not fade it
       further — opacity on a white glyph is what made it read cooler
       and brighter than the TUI rest colour. */
  }
  h1 {
    margin: 20px 0 0;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: lowercase;
    color: #e9e5db;                                /* fg — the brightest row */
    text-align: center;
  }
  .field { display: flex; flex-direction: column; gap: 8px; }
  label {
    font-size: 12px;
    letter-spacing: 0.04em;
    color: #837c6d;                                /* dim */
  }
  input {
    font-size: 16px;                               /* iOS no-zoom floor */
    line-height: 1.4;
    padding: 14px 16px;
    border-radius: 10px;                           /* radius-md */
    border: 1px solid #3b3527;                     /* edge */
    background: #1e1a14;                           /* surface */
    color: #e9e5db;
    width: 100%;
    -webkit-appearance: none;
    appearance: none;
  }
  input::placeholder { color: #837c6d; }           /* dim */
  input:focus {
    outline: 2px solid #38c96a;                    /* accent — the one green */
    outline-offset: 1px;
    border-color: transparent;
  }
  button {
    margin-top: 20px;
    font-size: 15px;
    font-weight: 500;
    letter-spacing: 0.01em;
    padding: 14px 16px;
    min-height: 48px;
    border: 1px solid #3b3527;                     /* edge — not a filled bar */
    border-radius: 10px;
    background: #1e1a14;                           /* surface */
    color: #e9e5db;                                /* fg */
    cursor: pointer;
    -webkit-appearance: none;
    appearance: none;
  }
  button:active { background: #272219; }           /* raised */
  .error {
    color: #ef8078;                                /* danger */
    text-align: center;
    margin: 0 0 16px;
    font-size: 13px;
  }
</style>
</head>
<body>
<form method="post" action="/login">
  <div class="lockup">
    <img class="mark" src="__MARK_DATA_URI__" width="72" height="72" alt="">
    <h1>local operator</h1>
  </div>
  <!--ERROR-->
  <div class="field">
    <label for="password">password</label>
    <input id="password" type="password" name="password"
           autocomplete="current-password" autofocus required>
  </div>
  <button type="submit">sign in</button>
</form>
</body>
</html>
"""
