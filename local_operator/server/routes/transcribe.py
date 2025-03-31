import os
import os
import tempfile
import time
import uuid
import base64
import logging
import io

from fastapi import APIRouter, Request, HTTPException
import whisper

from local_operator.server.models.schemas import CRUDResponse, TranscribeResponse

router = APIRouter(tags=["Transcribe"])
logger = logging.getLogger("local_operator.server.routes.transcribe")

model = whisper.load_model("small")

@router.post(
    "/v1/transcribe",
    summary="Transcribe audio file",
    description="Transcribe audio file and return transcription",
    response_model=CRUDResponse[TranscribeResponse],
)
async def transcribe_endpoint(request: Request):
    try:
        req_json = await request.json()
        audio_base64 = req_json.get("data")

        if not audio_base64:
            raise HTTPException(status_code=400, detail="Missing audio data")

        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            logger.exception(f"Failed to decode base64 audio data: {e}")
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio data after base64 decoding")

        # Save bytes to a temporary file
        timestamp = str(int(time.time()))
        unique_id = uuid.uuid4()
        temp_audio_path = os.path.join(tempfile.gettempdir(), f"temp_{timestamp}_{unique_id}.webm")
        logger.info(f"Saving audio to temporary file: {temp_audio_path}")
        try:
            with open(temp_audio_path, "wb") as temp_audio:
                temp_audio.write(audio_bytes)
        except Exception as e:
            logger.exception(f"Failed to save temporary audio file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save temporary audio file: {e}")

        if os.path.getsize(temp_audio_path) == 0:
            raise HTTPException(status_code=500, detail="Empty audio file after saving")

        # Transcribe
        logger.info(f"Transcribing audio file: {temp_audio_path}")
        result = model.transcribe(temp_audio_path, language="en")
        logger.info(f"Transcription result: {result}")

        # Delete temporary file
        try:
            os.remove(temp_audio_path)
            logger.info(f"Deleted temporary audio file: {temp_audio_path}")
        except Exception as e:
            logger.exception(f"Failed to delete temporary audio file: {e}")

        return CRUDResponse(
            status=200,
            message="Transcription completed successfully",
            result=TranscribeResponse(text=str(result["text"])),
        )
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
