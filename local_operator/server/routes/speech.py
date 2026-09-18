import logging
from typing import Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from local_operator.agents import AgentRegistry
from local_operator.clients._http import APIError
from local_operator.clients.openrouter import OpenRouterClient
from local_operator.clients.radient import RadientClient
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig, get_env_config
from local_operator.model.configure import configure_model
from local_operator.providers.auth_store import AuthStore
from local_operator.server.dependencies import (
    get_agent_registry,
    get_config_manager,
    get_credential_manager,
    get_provider_auth_store,
    get_radient_client,
)
from local_operator.server.models.schemas import AgentSpeechRequest, SpeechRequest
from local_operator.server.utils.operator import ServerExecutor
from local_operator.server.utils.speech_utils import determine_voice_and_instructions

router = APIRouter()
logger = logging.getLogger("local_operator.server.routes.speech")


def _upstream_failure_detail(exc: APIError) -> str:
    """Render an upstream speech failure with the upstream's own words attached.

    The sentence says what failed and the clause says what the upstream said,
    which is the text that names the real cause. Both are needed because the
    two audiences differ -- a user reads the sentence, whoever is on support
    reads the clause. The clause is safe to hand on because the body it quotes
    has already been through the shared scrubber on its way out of the client
    (``local_operator.clients._http.scrubbed_response_body`` plus this client's
    own credential), so an upstream that echoed the request back cannot put a
    credential into the client's copy.

    Args:
        exc: The upstream failure, carrying its status and scrubbed body.

    Returns:
        str: The ``detail`` to raise an ``HTTPException`` with.
    """
    if exc.status_code is not None and 200 <= exc.status_code < 300:
        # Not self-evidently a failure: Radient reports some provider failures
        # in the body of a 200, so "upstream responded 200" next to "failed"
        # reads as a contradiction unless it is worded as what it is.
        status = f"Radient reported an error (HTTP {exc.status_code})"
    elif exc.status_code is not None:
        status = f"Upstream responded {exc.status_code}"
    else:
        status = "No response from the upstream"
    if exc.body:
        return f"Speech generation failed upstream: {status}: {exc.body}"
    return f"Speech generation failed upstream: {status} with no body."


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
    radient_client: RadientClient = Depends(get_radient_client),
) -> Response:
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
    except APIError as upstream_exc:
        # A 2xx whose body is an error envelope, which the client types as an
        # upstream failure instead of returning its bytes as audio. A 200 payload
        # never reaches an exception handler, which is how a credential echoed
        # into one was served as the audio response.
        raise HTTPException(
            status_code=502, detail=_upstream_failure_detail(upstream_exc)
        ) from upstream_exc
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
    radient_client: RadientClient = Depends(get_radient_client),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    credential_manager: CredentialManager = Depends(get_credential_manager),
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
    config_manager: ConfigManager = Depends(get_config_manager),
    env_config: EnvConfig = Depends(get_env_config),
) -> Response:
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
            from local_operator.providers.radient_credentials import (
                resolve_radient_credential,
            )

            api_key = await resolve_radient_credential(
                credential_manager, env_config.radient_api_base_url, store=provider_auth_store
            )
            if api_key:
                model_info_client = RadientClient(api_key, env_config.radient_api_base_url)
            else:
                logger.warning("Radient hosting selected but RADIENT_API_KEY not found.")

        model_config = configure_model(
            hosting=hosting,
            model_name=model_name,
            credential_manager=credential_manager,
            # Pass the agent's knobs THROUGH, including "unset". The former
            # `or 0.2`/`or 0.9` re-asserted the app-wide constants that the
            # per-family sampling policy exists to stop sending, so an agent
            # with no stored preference had one invented for it here — and it
            # also silently rewrote a deliberate 0.0 into 0.2, `or` being false
            # for a legitimate zero.
            temperature=agent.temperature,
            top_p=agent.top_p,
            top_k=agent.top_k,
            max_tokens=agent.max_tokens,
            stop=agent.stop,
            frequency_penalty=agent.frequency_penalty,
            presence_penalty=agent.presence_penalty,
            seed=agent.seed,
            model_info_client=model_info_client,
            env_config=env_config,
        )
        executor = ServerExecutor(
            model_configuration=model_config,
            credential_manager=credential_manager,
            config_manager=config_manager,
            agent_registry=agent_registry,
            agent=agent,
        )

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
    except APIError as upstream_exc:
        # Same shape as the /v1/tools/speech route above: an error Radient
        # reported inside a 200 body is raised as an upstream failure rather
        # than served as audio bytes.
        logger.error(f"Upstream speech failure: {upstream_exc}")
        raise HTTPException(
            status_code=502, detail=_upstream_failure_detail(upstream_exc)
        ) from upstream_exc
    except Exception as e:
        logger.error(f"Failed to generate speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")
