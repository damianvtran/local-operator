"""Authenticated desktop adapters, separate from the legacy local API.

The Electron main process holds this process-lifetime capability. A renderer
cannot read it, and a backend launched without it must not silently expose the
new credential/configuration control plane through the legacy wildcard CORS API.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request


def require_desktop(request: Request) -> None:
    token = os.environ.get("LOCAL_OPERATOR_DESKTOP_TOKEN", "")
    if not token:
        raise HTTPException(503, "Desktop controls require a backend started by the desktop app.")
    origin = request.headers.get("origin")
    allowed = {
        item.strip()
        for item in os.environ.get("LOCAL_OPERATOR_DESKTOP_ORIGINS", "").split(",")
        if item.strip() and item.strip() != "null"
    }
    if origin is not None and origin not in allowed:
        raise HTTPException(403, "This origin cannot access desktop controls.")
    supplied = request.headers.get("authorization", "")
    if not secrets.compare_digest(supplied.encode("utf-8"), f"Bearer {token}".encode("utf-8")):
        raise HTTPException(401, "Desktop authorization is required.")
