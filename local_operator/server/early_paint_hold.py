"""Take a desktop open's paint hold before the middleware stack runs.

WHY THIS EXISTS (conversation-first-paint, residual tail R1, 2026-10-27). The
``/events`` route takes ``transcript_index.hold_paint`` as its first statement,
and #3026 made every OTHER conversation's background index scan pause while
such a hold is live. But the route runs only after the whole Starlette
``BaseHTTPMiddleware`` stack (``host_admission``, ``desktop_origin_cors``,
``managed_desktop_boundary``, ``static_response_policy``,
``desktop_supplement_response_policy``, CORS, reload admission). Each
``call_next`` hop is an await, and each await is one GIL re-acquire against
whatever CPU-bound threads are running. Measured on a real ``lop serve`` at load
33-70: the third open of a real -> legacy -> eligible switch spent 216-246 ms
in that stack while the two previous opens' deferred index scans held the GIL
in 5 ms slices, and ``events()`` itself was reached at ~+240 ms. Taking the
same hold here, before the stack, made those scans yield for that span too:
eligible 211 -> 108 ms median (spike ``early``, n=6, frames byte-identical).

THE HANDOFF. This layer's hold is not a second, longer hold. The route takes its
own hold (refcount 2) and immediately hands this one off (refcount 1) via
:func:`hand_off`, so from the route onward the accounting is exactly what it was
before: one hold, released at the snapshot frame. Releasing at response end
instead would make THIS session's own deferred cold scan wait for the whole SSE
stream (up to ``_PAINT_HOLD_CEILING_S``). The release here is idempotent, so the
``finally`` below is a no-op after a handoff and the real release on every path
that never reaches the route: a client gone mid-stack, an exception in a layer,
a 404 that refuses before ``hold_paint``... every exit of this coroutine.

WHO MAY TAKE IT. Only a request that already passes ``require_desktop`` -- the
very dependency the desktop sessions router enforces (posture, Origin rule,
constant-time bearer compare) -- so an unauthenticated local process cannot
stall another conversation's indexing by flooding opens. An authenticated open
of an unknown id holds for one request lifetime (the route's 404), which is
also all a known id's open can hold: a key nobody opens is never yielded to
past #3026's staleness cap either way.

HELD DOES NOT MEAN ADMITTED. As the outermost layer this runs BEFORE
``host_admission`` and reload admission, so a request carrying a valid bearer
but a bad ``Host`` (or arriving while reload admission would refuse it) still
takes the hold for that one request's lifetime, and is refused afterwards by
the layer that owns that rule. Only a bearer holder can do it, the request's
own lifetime and the per-batch pause budget bound it, and it is released on
that request's exit like any other (#3033 review m1).

KEYING. The hold must use the same key the route will (``pool.root``, built by
``routes.desktop_sessions.host`` from ``app.state.config_manager.config_dir``),
or the handoff would leave this one under a key nobody releases until the
``finally``. A reduced app with neither attribute takes no early hold.

Pure ASGI on purpose: a ``BaseHTTPMiddleware`` here would itself be one more
``call_next`` hop before the hold exists.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from local_operator.server.desktop import require_desktop
from local_operator.session import transcript_index

#: The scope key carrying this request's early release to the route.
SCOPE_KEY = "local_operator.early_paint_hold"

#: ``/v1/desktop/sessions/<id>/events`` -- the only route that takes a paint
#: hold. Any id shape: an id the route rejects is held for one request only.
#: The captured id is used only as a key into the hold tables, never as a path
#: component; the route resolves it (``pool.session``, whose lookup answers an
#: unknown id with 404) only after this layer has run.
_EVENTS = re.compile(r"^/v1/desktop/sessions/([^/]{1,128})/events$")


def _is_first_paint(query: bytes) -> bool:
    """``open_frame=1`` in the query, the desktop's first-paint open.

    A reconnect (``after_seq``, no ``open_frame``) also takes the route's hold,
    but has no snapshot budget at stake; it keeps today's timing.
    """
    return any(part == b"open_frame=1" for part in query.split(b"&"))


def _root(app: Any) -> Any:
    state = getattr(app, "state", None)
    pool = getattr(state, "desktop_sessions", None)
    if pool is not None:
        return pool.root
    manager = getattr(state, "config_manager", None)
    return getattr(manager, "config_dir", None) if manager is not None else None


def hand_off(scope: Scope | None) -> None:
    """Release this request's early hold. The route calls it right AFTER taking
    its own, so the refcount goes 1 -> 2 -> 1 and never touches zero (which
    would wake this session's deferred scan between the two holds)."""
    if scope is None:  # a reduced caller built without an ASGI scope
        return
    release: Callable[[], None] | None = scope.pop(SCOPE_KEY, None)
    if release is not None:
        release()


class EarlyPaintHoldMiddleware:
    """Hold the paint for an authenticated first-paint ``/events`` open."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "GET":
            await self.app(scope, receive, send)
            return
        match = _EVENTS.match(scope.get("path", ""))
        if match is None or not _is_first_paint(scope.get("query_string") or b""):
            await self.app(scope, receive, send)
            return
        root = _root(scope.get("app"))
        if root is None:
            await self.app(scope, receive, send)
            return
        try:
            require_desktop(Request(scope))
        except HTTPException:
            # Refused here and again (with the real response) by the route's
            # own dependency; this layer only declines to hold.
            await self.app(scope, receive, send)
            return
        release = transcript_index.hold_paint(root, match.group(1))
        scope[SCOPE_KEY] = release
        try:
            await self.app(scope, receive, send)
        finally:
            scope.pop(SCOPE_KEY, None)
            release()
