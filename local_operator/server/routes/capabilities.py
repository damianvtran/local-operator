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
            "features": {"auth": 1, "settings": 1},
        },
    )
