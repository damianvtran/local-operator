"""Bounded, redacted per-request log for the tunnel gateway and the mobile daemon.

WHY THIS EXISTS. A field report (macOS, lop 0.96.5; still true on 0.102.4) had a
phone showing "Asleep" / "Not answering" while the Mac looked healthy, and nothing
on either hop recorded a single request: neither the relay gateway (the hop the
phone reaches through Radient) nor the mobile daemon (the hop the gateway relays
to) logged what arrived, what it answered, or how long that took. So the one
question that splits the fault — did the request reach this computer, and what
did each hop say? — had no evidence anywhere.

WHAT IS RECORDED, AND WHAT NEVER IS. One JSON object per request:
``method``, ``path`` (a route TEMPLATE when the app has one, otherwise the path
with long opaque segments masked), ``status``, ``reason`` (a refusal code the app
attached, for a 4xx/5xx), ``ms`` (time to the response's first byte — the number
that distinguishes "slow" from "never answered"), ``id`` and ``at``. NEVER a
query string (an auth token or pairing code can ride there), never a header
value, a cookie or a body. ``id`` is the caller's ``X-Request-Id`` only when it is
a short token of safe characters (it is attacker-controlled input written to a
file), otherwise a fresh random one; it is echoed back as ``X-Request-Id`` so a
phone-side report and this log can be joined.

BOUNDS. On disk only: a rotating file capped at :data:`FILE_MAX_BYTES` with
:data:`FILE_BACKUPS` backup, so the worst case is ~1 MiB per surface whatever the
uptime; status readers read only a bounded tail of the current file. There is
no in-memory copy: every reader is another process (review round 1, R1-5 —
an in-process ring nothing read was dead state). The file is
the cross-process channel: ``lop tunnel status`` and ``lop mobile status`` run in
another process than the server that served the requests, and the file still
answers when that server is hung — which is exactly the case being diagnosed.

The middleware is pure ASGI (no Starlette import here) so the tunnel gateway's
import graph does not grow.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

#: Rotating file cap: 512 KiB holds ~2,500 entries, hours of an active phone.
FILE_MAX_BYTES = 512 * 1024
FILE_BACKUPS = 1
#: How much of the file's tail a status reader parses. Bounded so a status
#: command stays a cheap read regardless of how full the file is.
TAIL_BYTES = 64 * 1024
#: The file's name, in each surface's own log directory (the tunnel's config
#: directory beside `service.log`; the mobile daemon's log directory).
REQUEST_LOG_NAME = "requests.log"
#: How many recent entries the status surfaces print.
STATUS_RECENT = 8

#: The scope key a handler sets (through ``note_reason``) to name why it refused.
REASON_KEY = "lop_request_reason"
REQUEST_ID_HEADER = b"x-request-id"
#: A request id with this prefix marks this package's own loopback probe (the
#: mobile auth-gate check `lop mobile status` runs): not recorded, for the same
#: reason as ``skip`` paths. Spoofing it from a phone only hides that request
#: from this diagnostic log; it changes nothing the server does.
PROBE_ID_PREFIX = "lop-probe-"
#: A caller's request id is kept only when it is this shape: short, and free of
#: anything that could forge a log line or smuggle a value.
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
#: Path segments at least this long are masked as ``:id`` when no route template
#: is available: session ids, share tokens and pairing codes are long and opaque,
#: while route words ("api", "sessions", "events") are short.
_OPAQUE_SEGMENT = 16
#: Shorter segments are masked from this length when they carry a digit (see
#: ``_opaque``); below it sit version words such as ``v1`` and ``v2``.
_ID_MIN = 6
_MAX_PATH = 160

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def note_reason(scope: Scope, reason: str) -> None:
    """Attach a refusal code to this request's log entry.

    Codes only (this package's own vocabulary, e.g. ``authorization_refused``
    or ``unknown_host``) — never a message, which could carry upstream text.
    """
    scope[REASON_KEY] = reason


def _template(scope: Scope) -> str:
    """The route template when the app resolved one, else a masked raw path.

    Starlette records the matched route on the scope; its ``path`` is the
    template (``/api/sessions/{session_id}``), which is what makes entries
    comparable and keeps identifiers out. A catch-all (``/{path:path}``, the
    gateway's only route) carries no information, so the path itself is used
    with long opaque segments masked.
    """
    route = scope.get("route")
    template = getattr(route, "path", None)
    if isinstance(template, str) and template and ":path}" not in template:
        root = scope.get("root_path") or ""
        return (root + template)[:_MAX_PATH]
    raw = str(scope.get("path") or "/")
    return "/".join(":id" if _opaque(part) else part for part in raw.split("/"))[:_MAX_PATH]


def _opaque(segment: str) -> bool:
    """Whether a raw path segment is an identifier rather than a route word.

    Length alone missed lop's own ids (review round 1, R1-4): session ids are 12
    hex characters (`/api/sessions/3b11f69df1d3/history`), as are project keys
    and tag ids, all under the long-token bound. Route words are lowercase
    letters with at most ``_``/``-`` (``sessions``, ``history``, ``push-devices``)
    and short versions (``v1``), so a segment of :data:`_ID_MIN` or more
    characters carrying a digit is treated as an id, as is anything long.
    """
    if len(segment) >= _OPAQUE_SEGMENT:
        return True
    return len(segment) >= _ID_MIN and any(c.isdigit() for c in segment)


def _request_id(scope: Scope) -> str:
    for name, value in scope.get("headers") or ():
        if name.lower() == REQUEST_ID_HEADER:
            try:
                candidate = value.decode("ascii")
            except UnicodeDecodeError:
                break
            if _SAFE_ID.fullmatch(candidate):
                return candidate
            break
    return uuid.uuid4().hex[:16]


class RequestLog:
    """The rotating file for one surface (gateway or mobile daemon)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handler: logging.Handler | None = None
        if path is not None:
            try:
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                from local_operator.logger import _PrivateRotatingFileHandler

                handler = _PrivateRotatingFileHandler(
                    path, maxBytes=FILE_MAX_BYTES, backupCount=FILE_BACKUPS, encoding="utf-8"
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._handler = handler
            except OSError:
                # No file is a degraded log, never a server that will not start.
                self._handler = None

    def record(self, entry: dict[str, Any]) -> None:
        if self._handler is not None:
            message = json.dumps(entry, separators=(",", ":"), allow_nan=False)
            # `handle`, not `emit`: it takes the handler's own lock, so two
            # threads cannot interleave a line or race a rollover.
            self._handler.handle(
                logging.LogRecord("lop.requests", logging.INFO, "", 0, message, None, None)
            )

    def close(self) -> None:
        if self._handler is not None:
            self._handler.close()
            self._handler = None

    def middleware(self, app: ASGIApp, *, skip: frozenset[str] = frozenset()) -> ASGIApp:
        """Wrap an ASGI app so every HTTP request and websocket is recorded once.

        An entry is written when the response STARTS (so a minute-long SSE
        stream is visible at once, with its time to first byte), or, for a
        request that ended without ever starting a response, when the app
        returns — recorded with ``status: null`` because "never answered" is
        the very state the phone reports as Not answering.

        ``skip`` names exact paths that are not recorded: loopback health probes
        that `status` commands and installers poll, which would otherwise crowd
        the phone's requests out of the bounded tail the status surfaces read.
        """

        async def wrapped(scope: Scope, receive: Receive, send: Send) -> None:
            kind = scope.get("type")
            if kind not in ("http", "websocket") or scope.get("path") in skip:
                await app(scope, receive, send)
                return
            started = time.monotonic()
            identifier = _request_id(scope)
            if identifier.startswith(PROBE_ID_PREFIX):
                await app(scope, receive, send)
                return
            written = False

            def write(status: int | None) -> None:
                nonlocal written
                if written:
                    return
                written = True
                reason = scope.get(REASON_KEY)
                self.record(
                    {
                        "at": round(time.time(), 3),
                        "id": identifier,
                        "method": "WS" if kind == "websocket" else str(scope.get("method", "")),
                        "path": _template(scope),
                        "status": status,
                        "reason": reason if isinstance(reason, str) else "",
                        "ms": round((time.monotonic() - started) * 1000, 1),
                    }
                )

            async def logged_send(message: MutableMapping[str, Any]) -> None:
                kind_ = message.get("type")
                if kind_ == "http.response.start":
                    headers = list(message.get("headers") or [])
                    if not any(name.lower() == REQUEST_ID_HEADER for name, _ in headers):
                        headers.append((REQUEST_ID_HEADER, identifier.encode("ascii")))
                        message["headers"] = headers
                    write(int(message.get("status", 0)))
                elif kind_ == "websocket.accept":
                    write(101)
                elif kind_ == "websocket.close" and not written:
                    # Closed before accepting: the handshake was refused.
                    write(403)
                await send(message)

            try:
                await app(scope, receive, logged_send)
            except Exception:
                # The app raised before answering: the server will turn this
                # into a 500. A CANCELLATION (the client went away) is not one,
                # and falls through to `finally` as "no response".
                write(500 if kind == "http" else None)
                raise
            finally:
                write(None)

        return wrapped


def read_tail(path: Path, *, limit: int = STATUS_RECENT) -> dict[str, Any]:
    """The last ``limit`` entries and status-class counts from a log file's tail.

    Bounded (:data:`TAIL_BYTES`); a partial first line and anything that is not
    one of this module's entries are skipped, so a status command never raises
    over the file it happens to find. Counts cover the entries in the tail,
    which is what ``window`` reports.
    """
    entries: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            tail = handle.read(TAIL_BYTES).decode("utf-8", "replace")
    except OSError:
        return {"recent": [], "counts": {}, "window": 0, "path": str(path)}
    for line in tail.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and "method" in entry and "path" in entry:
            entries.append(entry)
    counts: dict[str, int] = {}
    for entry in entries:
        status = entry.get("status")
        bucket = f"{status // 100}xx" if isinstance(status, int) and status > 0 else "none"
        counts[bucket] = counts.get(bucket, 0) + 1
    return {
        "recent": entries[-limit:],
        "counts": counts,
        "window": len(entries),
        "path": str(path),
    }


def render(summary: dict[str, Any]) -> list[str]:
    """Plain lines for a terminal: a counts line, then the recent entries.

    The rows are protocol tokens (methods, paths, status codes, reason codes);
    the heading and the no-response word are prose, so they are keyed
    (`tunnel.status`, review round 1, R1-3) and resolved at call time. Imported
    here rather than at module top: the gateway imports this module, and its
    import graph stays as pinned.
    """
    from local_operator.i18n.keys import tunnel_status
    from local_operator.i18n.text import tr

    _t = tr("tunnel.status")
    recent = summary.get("recent") or []
    if not recent:
        return []
    counts = summary.get("counts") or {}
    order = ("2xx", "3xx", "4xx", "5xx", "1xx", "none")
    tally = " ".join(f"{key}={counts[key]}" for key in order if counts.get(key))
    window = str(summary.get("window", len(recent)))
    lines = [_t(tunnel_status.requests_heading(window=window, tally=tally))]
    no_response = _t(tunnel_status.requests_no_response())
    for entry in recent:
        stamp = time.strftime("%H:%M:%S", time.localtime(float(entry.get("at") or 0)))
        status = entry.get("status")
        reason = f" {entry['reason']}" if entry.get("reason") else ""
        lines.append(
            # i18n: ignore protocol-token row (time, method, path, status, reason code, ms, id)
            f"  {stamp} {entry.get('method', '')} {entry.get('path', '')} "
            f"{status if status is not None else no_response}{reason} "
            f"{entry.get('ms', '')}ms id={entry.get('id', '')}"
        )
    return lines
