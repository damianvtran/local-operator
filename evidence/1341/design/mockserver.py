#!/usr/bin/env python3
"""Isolated sandbox server for the design review of PR #1341.

Serves a built mobile-web `dist/` plus a hand-written stand-in for the
daemon's REST/SSE surface, so the SPA can be rendered at a phone viewport
without touching the operator's live `lop mobile` daemon (port 4098).

The session payloads are synthetic. The mode (populated / empty / connecting /
error / light / dark) is read from a control file on every request so a capture
loop can switch states without restarting the server.

usage: mockserver.py <port> <dist-dir> <control-file>
"""
from __future__ import annotations

import json
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(sys.argv[1])
DIST = Path(sys.argv[2]).resolve()
CONTROL = Path(sys.argv[3])

NOW = int(time.time())


def sessions_payload() -> list[dict]:
    return [
        {
            "session_id": "01J-phone-bundle-repair",
            "section": "active",
            "conversation_name": "phone portal bundle repair",
            "cwd": "/Users/damian/local-operator",
            "model_label": "claude-sonnet-4.6 · direct",
            "streaming": True,
            "needs_attention": False,
            "pending_kind": "",
            "subagents_running": 2,
            "todos_open": 3,
            "mtime": NOW - 4,
        },
        {
            "session_id": "01J-deploy-minerva-api",
            "section": "active",
            "conversation_name": "deploy minerva-api to prod",
            "cwd": "/Users/damian/minervaai/deploy",
            "model_label": "claude-opus-4.6 · direct",
            "streaming": True,
            "needs_attention": True,
            "pending_kind": "approval",
            "subagents_running": 0,
            "todos_open": 1,
            "mtime": NOW - 30,
        },
        {
            "session_id": "01J-support-sweep",
            "section": "active",
            "conversation_name": "support inbox sweep",
            "cwd": "/Users/damian/workspace",
            "model_label": "gpt-5.1-codex · radient",
            "streaming": False,
            "needs_attention": False,
            "unseen": True,
            "pending_kind": "",
            "subagents_running": 0,
            "todos_open": 0,
            "mtime": NOW - 900,
        },
        {
            "session_id": "01J-long-name",
            "section": "active",
            "conversation_name": "migrate the snapshot installer to a generation-scoped package cache without touching the live tree",
            "cwd": "/Users/damian/workspace/deep/nested/project/directory",
            "model_label": "claude-haiku-4.5 · direct",
            "streaming": False,
            "needs_attention": False,
            "pending_kind": "",
            "subagents_running": 12,
            "todos_open": 34,
            "mtime": NOW - 3600,
        },
        {
            "session_id": "01J-usage-export",
            "section": "previous",
            "conversation_name": "quarterly usage export",
            "cwd": "/Users/damian/workspace",
            "model_label": "claude-sonnet-4.5 · direct",
            "streaming": False,
            "needs_attention": False,
            "pending_kind": "",
            "subagents_running": 0,
            "todos_open": 0,
            "mtime": NOW - 86400,
        },
    ]


def payload(mode: str) -> list[dict]:
    """`empty` = connected daemon, no sessions; `activeonly` = no previous
    sessions, which is the state that leaves the Previous heading empty."""
    if mode == "empty":
        return []
    rows = sessions_payload()
    if mode == "activeonly":
        rows = [r for r in rows if r["section"] == "active"]
    return rows


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DIST), **kw)

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _mode(self) -> str:
        try:
            return CONTROL.read_text().strip() or "populated"
        except OSError:
            return "populated"

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        mode = self._mode()

        if path == "/__frame.html":
            body = Path("/tmp/dsgn-1341/frame.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if path == "/__mode":
            return self._json({"mode": mode})
        if path == "/api/directories":
            return self._json(
                {"home": "/Users/damian", "recent": ["/Users/damian/local-operator"], "tmp": "/tmp"}
            )
        if path == "/api/commands":
            return self._json({"commands": []})
        if path == "/api/models":
            return self._json({"models": []})
        if path == "/api/sessions":
            return self._json({"sessions": payload(mode)})
        if path == "/api/sessions/events":
            return self._sse(mode)
        if path.startswith("/api/"):
            return self._json({"error": "not mocked"}, 404)
        return super().do_GET()

    def _sse(self, mode: str) -> None:
        if mode in ("error", "connecting"):
            if mode == "error":
                return self._json({"error": "daemon unavailable"}, 500)
            time.sleep(300)  # hold the socket open without ever answering
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        frame = {"sessions": payload(mode)}
        try:
            self.wfile.write(b"event: sessions\ndata: " + json.dumps(frame).encode() + b"\n\n")
            self.wfile.flush()
            for _ in range(60):
                time.sleep(5)
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
