"""Authenticated desktop adapters, separate from the legacy local API.

The Electron main process holds this process-lifetime capability. A renderer
cannot read it, and a backend launched without it must not silently expose the
new credential/configuration control plane through the legacy wildcard CORS API.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request


def _is_non_browser_caller(request: Request) -> bool:
    """Whether this request demonstrably did NOT come from a web page.

    ``Sec-Fetch-Site`` is added by the browser fetch/XHR stack itself and sits
    on the forbidden-header list, so page script can neither remove nor forge
    it. Its absence is therefore positive evidence of a native caller (Electron
    main, the dev proxy, curl), which is what lets an Origin-less request be
    admitted without reopening the allowlist bypass to a browser.
    """
    return "sec-fetch-site" not in request.headers


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
    if origin is not None:
        if origin not in allowed:
            raise HTTPException(403, "This origin cannot access desktop controls.")
    elif allowed and not _is_non_browser_caller(request):
        # An ABSENT Origin used to skip the check entirely, so a caller that
        # simply omitted the header bypassed the allowlist (code review 4).
        #
        # It cannot become an unconditional "Origin required", because the two
        # first-class callers legitimately send none: Electron main fetches
        # from the main process, and the dev proxy forwards server-side. Both
        # are non-browser agents, for whom Origin is not a security signal.
        #
        # What a BROWSER cannot forge is the absence of `Sec-Fetch-Site`: it is
        # attached by the fetch/XHR stack to every request a page makes and is
        # on the forbidden-header list, so script cannot remove or spoof it. A
        # request carrying it is browser-originated and must present an
        # allowed Origin; one without it is a native client, which the bearer
        # below still authenticates.
        raise HTTPException(403, "This origin cannot access desktop controls.")
    supplied = request.headers.get("authorization", "")
    if not secrets.compare_digest(supplied.encode("utf-8"), f"Bearer {token}".encode("utf-8")):
        raise HTTPException(401, "Desktop authorization is required.")
