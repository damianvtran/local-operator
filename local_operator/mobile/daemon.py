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
DEFAULT_PORT = 4097

_WEB_DIR = Path(__file__).parent / "web"
_DIST_DIR = _WEB_DIR / "dist"


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
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
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
            line = await reader.readline()
            if not line:
                break
            try:
                frame = json.loads(line.decode("utf-8", "replace"))
            except ValueError:
                continue
            op = frame.get("op")
            if op in ("projection", "welcome"):
                data = frame.get("data") or {}
                entry.projection = _projection_from_json(data, record)
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
    projection.transcript = build(TranscriptEntry, data.get("transcript", []))
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


def _fan_out(entry: SessionEntry) -> None:
    """Push the current projection to every open SSE queue for this session.
    A full queue means a slow phone: drop the oldest by replacing — repaints
    supersede, so dropping stale ones loses nothing."""
    if entry.projection is None:
        return
    frame = entry.projection.to_json()
    for queue in entry.subscribers:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass


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
            if not entry.ended and entry.writer is None and not entry.degraded:
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
    from starlette.routing import Mount, Route

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
        return HTMLResponse(_LOGIN_HTML)

    async def login_submit(request: Request) -> Response:
        form = await request.form()
        candidate = str(form.get("password", ""))
        if not daemon.password or not check_password(candidate, daemon.password):
            return HTMLResponse(_LOGIN_HTML.replace("<!--ERROR-->", _LOGIN_ERROR), status_code=401)
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
        cwd = str(body.get("cwd") or Path.home())
        if not Path(cwd).is_dir():
            return JSONResponse({"error": f"not a directory: {cwd}"}, status_code=400)
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

    routes = [
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
        Route("/api/sessions/{pid:int}/command", api_command, methods=["POST"]),
        Route("/api/commands", api_commands),
        Route("/api/models", api_models),
        Route("/", index),
    ]
    if _DIST_DIR.exists():
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
    picker, so the catalogue is filtered through the credential store, the
    same gate the stream-time credential cascade applies. Runs in a thread:
    catalogue reads are file I/O."""
    from local_operator.credentials import CredentialManager
    from local_operator.model.registry import SupportedHostingProviders, static_models
    from local_operator.paths import config_dir

    credential_manager = CredentialManager(config_dir=config_dir())
    rows: list[dict[str, Any]] = []
    for provider in SupportedHostingProviders:
        required = provider.requiredCredentials
        if required and not any(credential_manager.get_credential(key) for key in required):
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
<title>local operator — sign in</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; min-height: 100dvh; display: grid; place-items: center;
    background: #14110c; color: #e9e5db;
    font: 16px/1.5 -apple-system, "SF Pro Text", system-ui, sans-serif;
  }
  form { display: grid; gap: 16px; width: min(320px, 84vw); }
  h1 { font-size: 20px; font-weight: 600; margin: 0; text-align: center; }
  .mark {
    text-align: center; font-size: 28px; color: #38c96a;
    font-family: ui-monospace, monospace;
  }
  input {
    font-size: 16px; padding: 12px 14px; border-radius: 10px;
    border: 1px solid #3a352b; background: #1d1913; color: #e9e5db;
  }
  input:focus { outline: 2px solid #38c96a; outline-offset: 1px; border-color: transparent; }
  button {
    font-size: 16px; font-weight: 600; padding: 12px; border: 0; border-radius: 10px;
    background: #38c96a; color: #0d0b08; cursor: pointer;
  }
  .error { color: #e5534b; text-align: center; margin: 0; font-size: 14px; }
</style>
</head>
<body>
<form method="post" action="/login">
  <div class="mark">▲</div>
  <h1>Local Operator</h1>
  <!--ERROR-->
  <input type="password" name="password" placeholder="Password"
         autocomplete="current-password" autofocus required>
  <button type="submit">Sign in</button>
</form>
</body>
</html>
"""
