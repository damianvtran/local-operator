from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from local_operator.server.dependencies import get_radient_client
from local_operator.server.models.schemas import SpeechRequest

router = APIRouter()


@router.post(
    "/v1/tools/speech",
    tags=["Tools"],
    summary="Generate speech from text",
    description="""Generates speech from text using a specified provider and returns the audio data. This endpoint is protected by API key authentication and is subject to billing.""",  # noqa: E501
    responses={
        200: {
            "description": "Successful speech generation",
            "content": {"audio/mpeg": {"schema": {"type": "string", "format": "binary"}}},
        },
        400: {"description": "Bad request, such as missing required fields"},
        500: {"description": "Internal server error"},
    },
)
async def create_speech(
    speech_request: SpeechRequest,
    radient_client=Depends(get_radient_client),
):
    """
    Generates speech from text using a specified provider and returns the audio data.
    This endpoint is protected by API key authentication and is subject to billing.
    """
    try:
        audio_data = radient_client.create_speech(
            input_text=speech_request.input,
            model=speech_request.model,
            voice=speech_request.voice,
            response_format=speech_request.response_format,
            speed=speech_request.speed,
            provider=speech_request.provider,
        )

        media_type = f"audio/{speech_request.response_format}"
        return Response(content=audio_data, media_type=media_type)

    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except Exception as e:
        # Catch any other exceptions and return a 500 error
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")
