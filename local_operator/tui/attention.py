"""Positive host attention proof, sampled only for an outstanding completion.

Textual's initial app_focus=True is not evidence. cmux additionally remembers
selected surfaces in background windows, so selection alone is insufficient.
The caller runs this bounded probe off-loop and rechecks its render afterwards.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import uuid
from collections.abc import Mapping
from functools import lru_cache
from typing import Any


def _json_command(command: list[str]) -> Any:
    result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=True)
    return json.loads(result.stdout)


@lru_cache(maxsize=1)
def _discovered_socket() -> str:
    # Let the installed CLI own fallback resolution; paths vary by build/user.
    result = _json_command(["cmux", "--json", "rpc", "system.identify", "{}"])
    return str(result.get("socket_path") or "")


def _rpc(sock: socket.socket, method: str, params: dict[str, Any]) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    sock.sendall(
        (json.dumps({"id": request_id, "method": method, "params": params}) + "\n").encode()
    )
    data = bytearray()
    while b"\n" not in data:
        chunk = sock.recv(16384)
        if not chunk or len(data) + len(chunk) > 262144:
            raise ValueError("invalid cmux response")
        data.extend(chunk)
    reply = json.loads(data.split(b"\n", 1)[0])
    if reply.get("id") != request_id or reply.get("ok") is not True:
        raise ValueError("cmux RPC refused")
    result = reply.get("result")
    if not isinstance(result, dict):
        raise ValueError("invalid cmux result")
    return result


def terminal_is_foreground(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    surface = env.get("CMUX_SURFACE_ID")
    if not surface:
        # Non-cmux terminals use positively observed terminal focus reports.
        return not any(key.startswith("CMUX_") for key in env)
    if sys.platform != "darwin":
        return False
    try:
        app = _json_command(
            [
                "osascript",
                "-l",
                "JavaScript",
                "-e",
                "ObjC.import('AppKit'); var a=$.NSWorkspace.sharedWorkspace.frontmostApplication;"
                "JSON.stringify({bundle:ObjC.unwrap(a.bundleIdentifier),"
                "pid:Number(a.processIdentifier),"
                "active:Boolean(a.active),hidden:Boolean(a.hidden)})",
            ]
        )
        if app.get("bundle") != "com.cmuxterm.app" or not app.get("active") or app.get("hidden"):
            return False
        path = env.get("CMUX_SOCKET_PATH") or _discovered_socket()
        if not path:
            return False
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            sock.connect(path)
            # Darwin SDK sys/un.h: SOL_LOCAL=0, LOCAL_PEERPID=0x002.
            # The kernel proves this socket belongs to the frontmost instance;
            # another cmux instance's remembered focused surface proves nothing.
            if sock.getsockopt(0, 0x002) != app.get("pid"):
                return False
            windows = _rpc(sock, "window.list", {})
            selected = [
                window
                for window in windows.get("windows", [])
                if window.get("key") is True and window.get("visible") is True
            ]
            if len(selected) != 1:
                return False
            window = selected[0]
            state = _rpc(sock, "system.identify", {"window_id": window["id"]}).get("focused", {})
            return bool(
                state.get("window_id") == window["id"]
                and state.get("workspace_id") == window.get("selected_workspace_id")
                and state.get("surface_id") == surface
                and state.get("surface_type") == "terminal"
            )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, AttributeError):
        _discovered_socket.cache_clear()
        return False
