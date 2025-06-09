import logging
from typing import Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from local_operator.agents import AgentRegistry
from local_operator.clients.openrouter import OpenRouterClient
from local_operator.clients.radient import RadientClient
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig, get_env_config
from local_operator.executor import LocalCodeExecutor
from local_operator.model.configure import configure_model
from local_operator.server.dependencies import (
    get_agent_registry,
    get_config_manager,
    get_credential_manager,
    get_radient_client,
)
from local_operator.server.models.schemas import AgentSpeechRequest, SpeechRequest
from local_operator.server.utils.speech_utils import determine_voice_and_instructions

router = APIRouter()
logger = logging.getLogger("local_operator.server.routes.speech")


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
            instructions=speech_request.instructions,
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


@router.post(
    "/v1/agents/{agent_id}/speech",
    tags=["Tools"],
    summary="Generate speech from an agent's last message",
    description="""Generates speech from an agent's last message, automatically determining the voice and instructions based on the agent's profile.""",  # noqa: E501
    responses={
        200: {
            "description": "Successful speech generation",
            "content": {"audio/mpeg": {"schema": {"type": "string", "format": "binary"}}},
        },
        404: {"description": "Agent not found"},
        500: {"description": "Internal server error"},
    },
)
async def create_agent_speech(
    agent_id: str,
    speech_request: AgentSpeechRequest,
    radient_client=Depends(get_radient_client),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    credential_manager: CredentialManager = Depends(get_credential_manager),
    config_manager: ConfigManager = Depends(get_config_manager),
    env_config: EnvConfig = Depends(get_env_config),
):
    """
    Generates speech from an agent's last message.
    """
    try:
        agent = agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        hosting = agent.hosting or config_manager.get_config_value("hosting")
        model_name = agent.model or config_manager.get_config_value("model_name")

        if not hosting:
            raise ValueError("Hosting platform is not configured.")
        if not model_name:
            raise ValueError("Model name is not configured.")

        model_info_client: Optional[Union[OpenRouterClient, RadientClient]] = None
        if hosting == "openrouter":
            api_key = credential_manager.get_credential("OPENROUTER_API_KEY")
            if api_key:
                model_info_client = OpenRouterClient(api_key)
            else:
                logger.warning("OpenRouter hosting selected but OPENROUTER_API_KEY not found.")
        elif hosting == "radient":
            api_key = credential_manager.get_credential("RADIENT_API_KEY")
            if api_key:
                model_info_client = RadientClient(api_key, env_config.radient_api_base_url)
            else:
                logger.warning("Radient hosting selected but RADIENT_API_KEY not found.")

        model_config = configure_model(
            hosting=hosting,
            model_name=model_name,
            credential_manager=credential_manager,
            temperature=agent.temperature or 0.2,
            top_p=agent.top_p or 0.9,
            top_k=agent.top_k,
            max_tokens=agent.max_tokens,
            stop=agent.stop,
            frequency_penalty=agent.frequency_penalty,
            presence_penalty=agent.presence_penalty,
            seed=agent.seed,
            model_info_client=model_info_client,
            env_config=env_config,
        )
        executor = LocalCodeExecutor(model_configuration=model_config)

        voice, instructions = await determine_voice_and_instructions(agent, executor)

        audio_data = radient_client.create_speech(
            input_text=speech_request.input_text,
            instructions=instructions,
            model="gpt-4o-mini-tts",
            voice=voice,
            response_format=speech_request.response_format,
            speed=1.0,
            provider="openai",
        )

        media_type = f"audio/{speech_request.response_format}"
        return Response(content=audio_data, media_type=media_type)

    except HTTPException as http_exc:
        logger.exception(f"HTTPException: {http_exc}")
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to generate speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")
