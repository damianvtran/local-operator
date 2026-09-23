"""Read-only tunnel health for the desktop shell.

The desktop UI can already START a Radient sign-in (``POST /v1/auth/login``) and
poll it; what it could not learn was whether this machine's remote access is
actually working, or whether the sign-in it offers would fix anything. That is
one question with one honest answer, and the answer already lives on this device
(``tunnels/report.py``, which `lop tunnel status` reads too). This route is the
read half of it.

READ-ONLY, and there is no write half: the remedy is a command an operator runs
in a terminal (`lop login radient`, `lop tunnel install`, `lop tunnel connect`),
so this reports it as a string rather than exposing an operation that would run
it on their behalf. A route that restarted a connector from a browser click is a
privilege nobody asked for, and the `actions` slot in the composer can carry the
sign-in flow the UI already has.

Poll-on-open, and refetch when a sign-in the UI started settles: this state
changes on human timescales (a login, a console edit), so an SSE frame would be
a new event kind carrying news that is minutes old by the time anyone reads it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply

router = APIRouter(tags=["Desktop tunnel"], dependencies=[Depends(require_desktop)])


@router.get("/v1/desktop/tunnel", response_model=CRUDResponse[Any])
async def tunnel_state() -> CRUDResponse[Any]:
    """This machine's tunnel and login state, as `lop tunnel status --json` sees it.

    One deliberate difference from the CLI: the cloud record is reported with
    ``cloud.source = "cached"`` and never fetched, because a route the UI polls
    on open must not wait on an upstream call to answer a question about THIS
    machine — and because the cases that matter (a parked connector, a dead
    grant) are exactly the ones where that call cannot answer.
    """
    # Imported per request, not at module scope, for the reason
    # ``routes/auth.py`` gives: the report pulls the tunnel gateway (PyJWT,
    # httpx) onto every ``lop serve`` boot (backend load report B-F10).
    from local_operator.tunnels import report

    return reply(await report.local_payload())
