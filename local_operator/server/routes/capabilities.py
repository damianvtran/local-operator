"""Public feature negotiation contains no credentials or configuration values."""

import os

from fastapi import APIRouter

from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Capabilities"])


@router.get("/v1/capabilities", response_model=CRUDResponse)
async def capabilities():
    return CRUDResponse(
        status=200,
        message="Backend capabilities retrieved.",
        result={
            "desktop_contract": 1,
            "desktop_available": bool(os.environ.get("LOCAL_OPERATOR_DESKTOP_TOKEN")),
            "desktop_auth": "bearer",
            # These version the HTTP subsystems, not renderer completion or
            # third-party authorization. No aggregate "full parity" claim.
            "features": {
                "auth": 1,
                "settings": 1,
                "commands": 1,
                "catalogues": 1,
                "lifecycle": 1,
                # Watch leases route notification delivery; they never mark read.
                # Named for this map's convention (`<subsystem>: <version>`); the
                # runtime record's capability LIST is a different namespace and
                # keeps its versioned `completion-ack-v1` string.
                "completion_ack": 1,
                "mcp": 1,
                "radient": 1,
            },
        },
    )
