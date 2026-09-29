"""Public feature negotiation contains no credentials or configuration values."""

from fastapi import APIRouter

from local_operator.server.desktop import desktop_posture
from local_operator.server.features import feature_flags
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Capabilities"])


@router.get("/v1/capabilities", response_model=CRUDResponse)
async def capabilities():
    # The map itself lives in `local_operator.server.features.feature_flags` --
    # off this route because the mobile daemon imports it too, and the deferred
    # `references` import (with the startup-cost reasoning the guard in
    # `tests/unit/test_import_graph.py` enforces) moved there with it. The route
    # stays the HTTP face of the same function.
    return CRUDResponse(
        status=200,
        message="Backend capabilities retrieved.",
        result={
            "desktop_contract": 1,
            # The signal the UI needs to choose its path WITHOUT guessing, and
            # it has to answer the claim handshake too: false while the plane
            # is closed (including a daemon nobody has claimed yet, where the
            # UI's job is to claim it), true once the app's env capability or
            # an accepted claim governs it. Reading the env variable here
            # directly was how a claimed daemon reported "no desktop" while
            # its routes had already opened — see `server/desktop.py`.
            "desktop_available": desktop_posture().enabled,
            "desktop_auth": "bearer",
            # These version the HTTP subsystems, not renderer completion or
            # third-party authorization. No aggregate "full parity" claim.
            "features": feature_flags(),
        },
    )
