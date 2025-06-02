import os
import shutil
import tempfile
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from local_operator.clients.radient import (
    RadientClient,
    RadientTranscriptionResponseData,
)
from local_operator.server.dependencies import get_radient_client
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter()


@router.post(
    "/v1/transcriptions",
    response_model=CRUDResponse[RadientTranscriptionResponseData],
    summary="Transcribe Audio File",
    tags=["Transcription"],
)
async def create_transcription_endpoint(
    radient_client: Annotated[RadientClient, Depends(get_radient_client)],
    file: UploadFile = File(...),
    model: Optional[str] = Form("gpt-4o-transcribe"),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    language: Optional[str] = Form(None),
    provider: Optional[str] = Form("openai"),
) -> CRUDResponse[RadientTranscriptionResponseData]:
    """
    Transcribe an audio file using the specified model and parameters.

    The audio file is sent as `multipart/form-data`.
    """
    if not radient_client.api_key:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Radient API key is not configured on the server.",
        )

    # Save the uploaded file temporarily
    try:
        # Create a temporary directory to save the file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename if file.filename else "audio.tmp")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded audio file: {str(e)}",
        )
    finally:
        await file.close()

    try:
        transcription_result = radient_client.create_transcription(
            file_path=temp_file_path,
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            language=language,
            provider=provider,
        )

        return CRUDResponse(
            status=200,
            message="Transcription created successfully",
            result=transcription_result,
        )
    except FileNotFoundError:
        # This case should ideally be caught by the client if temp_file_path is wrong,
        # but good to have a catch here.
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Temporary audio file not found after saving.",
        )
    except ValueError as ve:  # For validation errors from the client
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:  # For API errors or other runtime issues from the client
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(re))
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during transcription: {str(e)}",
        )
    finally:
        # Clean up the temporary directory and its contents
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
