"""
Health check endpoint for the Local Operator API.
"""

from fastapi import APIRouter

from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health Check",
    description="Returns the health status of the API server.",
    response_model=CRUDResponse[str],
)
async def health_check():
    """
    Health check endpoint.

    Returns:
        A JSON object with a "status" key indicating operational status.
    """
    return CRUDResponse(status=200, message="ok")
