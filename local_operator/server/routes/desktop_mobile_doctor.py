"""`GET /v1/desktop/mobile/doctor`: the phone-access diagnosis for the desktop.

The same redacted, read-only JSON `lop mobile doctor --json` prints
(`mobile/doctor.py`), for the desktop's "Copy diagnostics" button and for agents
that cannot run or relay shell diagnostics (Radient's WAF refuses Team Bridge
messages carrying `curl`/`lsof`/`grep`). Behind the desktop bearer like every
`/v1/desktop/*` route: the report is redacted, but it still describes this
machine, so it is not anonymous. Advertised as `features.mobile_doctor`.

The desktop may name its own version in ``X-Local-Operator-Desktop-Version``;
the doctor keeps it only if it is a plain version string, so the header cannot
inject text into a report that is meant to be pasted elsewhere.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header

from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply

router = APIRouter(tags=["Desktop mobile doctor"], dependencies=[Depends(require_desktop)])


@router.get("/v1/desktop/mobile/doctor", response_model=CRUDResponse[Any])
async def mobile_doctor(
    desktop_version: str | None = Header(default=None, alias="X-Local-Operator-Desktop-Version"),
) -> CRUDResponse[Any]:
    # Imported per request: the doctor pulls the tunnel client and the mobile
    # installer, which a `lop serve` boot must not pay for (the same reason
    # `desktop_tunnel` imports `tunnels.report` lazily).
    from local_operator.mobile import doctor

    return reply(await doctor.collect(desktop_app=desktop_version))
