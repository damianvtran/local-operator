"""The phone portal password, read for the desktop on an explicit request.

WHY THIS EXISTS. A user who brings their own tunnel (the desktop's "Using your
own tunnel?" disclosure) signs the phone in with the portal password
``lop mobile password`` prints. Until now the desktop could only tell them to
open a terminal and run that command; "Show phone password" reads the same
value here instead (UI consumer: local-operator-ui #1912).

WHY THE DESKTOP MAY READ IT. The desktop bearer is NOT an agent boundary: it is
held by the desktop app the same user runs on the same machine, and that user
can already print this password at a TTY with ``lop mobile password``. Nothing
new is disclosed to a new principal -- the route spares a terminal round-trip.
What the bearer plus the Origin rule (``require_desktop``) keep out are pages
and other origins; an agent never holds the bearer, and the CLI path keeps its
own "no TTY, no secret" rule for captured stdout, which this route does not
loosen (an HTTP response to the desktop is not an agent's context window).

CONSTRAINTS, each load-bearing:

* READ ON CLICK ONLY. The store is consulted when this request arrives and at
  no other time: no cache, no prefetch, no inclusion in any polled payload
  (``/v1/desktop/tunnel`` stays free of it). A secret that lives only in the
  response the user asked for cannot leak from a snapshot nobody asked for.
* ``no-store``. Set here explicitly (``managed_desktop_boundary`` also stamps
  the whole ``/v1/desktop/`` family) with ``Pragma: no-cache`` for HTTP/1.0
  intermediaries, so the value never lands in a cache because a middleware
  ordering changed.
* NEVER LOGGED. This module logs nothing about the value, and the server has no
  request/response body logging middleware; the access log carries the path
  only. ``tests/unit/server/test_desktop_mobile_password.py`` pins that with a
  DEBUG capture across a 200 call.
* DESKTOP ONLY. The phone relay republishes the feature map but serves no such
  route, so the capability key sits in ``DESKTOP_ONLY_FEATURES``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from local_operator.server.desktop import require_desktop

router = APIRouter(tags=["Desktop mobile"], dependencies=[Depends(require_desktop)])

#: The stable refusal code when no portal password is set on this machine. The
#: UI keys its "run `lop mobile install`" guidance off this code, never a
#: sentence, so the body carries no prose to translate.
PASSWORD_UNSET = "mobile_password_unset"

#: Belt and braces beside the family-wide middleware stamp (see module doc).
NO_STORE_HEADERS = {"Cache-Control": "no-store", "Pragma": "no-cache"}


@router.get("/v1/desktop/mobile/password")
async def mobile_password() -> JSONResponse:
    """This machine's phone portal password, or 404 ``mobile_password_unset``.

    Envelope matches the desktop family's ``CRUDResponse`` (``status``,
    ``message``, ``result``) so the UI's one reader parses it.
    """
    # Imported per request: the store module shells out to `security` /
    # `secret-tool` and has no business on every `lop serve` boot.
    from local_operator.mobile.auth import load_password

    # The macOS Keychain and Secret Service reads are subprocess calls with
    # their own timeouts; run them off the event loop so one slow keyring does
    # not stall every other desktop request.
    password = await asyncio.to_thread(load_password)
    if not password:
        raise HTTPException(404, {"code": PASSWORD_UNSET}, headers=NO_STORE_HEADERS)
    body: dict[str, Any] = {
        "status": 200,
        "message": "Mobile password.",
        "result": {"password": password},
    }
    return JSONResponse(body, headers=NO_STORE_HEADERS)
