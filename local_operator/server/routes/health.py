"""
Health check endpoint for the Local Operator API.
"""

import importlib.metadata

from fastapi import APIRouter

from local_operator.server.models.schemas import CRUDResponse, HealthCheckResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health Check",
    description="Returns the health status of the API server.",
    response_model=CRUDResponse[HealthCheckResponse],
    responses={
        200: {
            "description": "Successful response with version information",
            "content": {
                "application/json": {
                    "example": {"status": 200, "message": "ok", "result": {"version": "0.1.0"}}
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {"application/json": {"example": {"detail": "Internal server error"}}},
        },
    },
)
async def health_check():
    """
    Health check endpoint.

    Retrieves the current version of the Local Operator application.

    Returns:
        CRUDResponse[HealthCheckResponse]: A response object containing the application version.

    Raises:
        HTTPException: If there's an error retrieving version information.
    """
    version = importlib.metadata.version("local-operator")
    result = HealthCheckResponse(version=version)

    return CRUDResponse(status=200, message="ok", result=result)
