import logging
from typing import Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import SecretStr

from local_operator.agents import AgentRegistry
from local_operator.clients._http import APIError, error_payload
from local_operator.clients.openrouter import OpenRouterClient
from local_operator.clients.radient import RadientClient
from local_operator.config import ConfigManager
from local_operator.env import EnvConfig, get_env_config
from local_operator.model.configure import configure_model
from local_operator.providers.auth_store import AuthStore
from local_operator.server.dependencies import (
    get_agent_registry,
    get_config_manager,
    get_provider_auth_store,
    get_radient_client,
)
from local_operator.server.models.schemas import AgentSpeechRequest, SpeechRequest
from local_operator.server.utils.operator import ServerExecutor
from local_operator.server.utils.speech_utils import determine_voice

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
        # Agent-server's designed envelope ({"error", "code", "details"}) is
        # extracted field-first: the `error` string is the only part written
        # for a client to read, and the raw body can carry framing nobody
        # should render. A body that is not that envelope falls back to its
        # scrubbed text, which is what the envelope-path tests pin.
        message, _code, _details = error_payload(exc.body)
        return f"Speech generation failed upstream: {status}: {message or exc.body}"
    # No body to quote: the client's message already IS the upstream's prose
    # (``api_error_from_response`` deliberately leaves ``body`` unset for a
    # designed envelope), so it is the clause.
    return f"Speech generation failed upstream: {status}: {exc}"


#: The refusal sentence for a missing (or refused) Radient credential. The
#: remedy is the same on both speech routes, so it is spelled once.
SPEECH_SIGN_IN_SENTENCE = "Sign in to Radient to use speaking aloud"

#: The fixed sentences for the upstream refusals a user can act on, keyed by
#: the status agent-server passes through: 401 a refused/expired credential,
#: 402 a balance the speech cannot be charged against, 429 a busy (or
#: rate-limited) upstream, 503 the provider being temporarily unavailable --
#: agent-server's own sentence for a vendor 401/402/5xx, which is why 503
#: keeps that wording here. Fixed text on purpose: the hub's envelope is
#: written for an operator ("insufficient credits for this request"), not for
#: the toast the user reads.
_SPEECH_REFUSAL_SENTENCES: Dict[int, str] = {
    401: SPEECH_SIGN_IN_SENTENCE,
    402: "Your Radient credit balance is too low for speech. Add credits to continue.",
    429: "Speech is busy right now. Try again in a moment.",
    503: "Speech is temporarily unavailable.",
}


def _require_radient_credential(radient_client: RadientClient) -> None:
    """Refuse with the sign-in remedy when no Radient credential resolved.

    The credential resolver answers "nothing found" with an empty ``SecretStr``
    rather than raising, so without this the absence would travel into the
    upstream call and come back as whatever an empty bearer produces. Both
    speech routes call this before doing any work.
    """
    if radient_client.api_key is not None and radient_client.api_key.get_secret_value():
        return
    raise HTTPException(status_code=401, detail=SPEECH_SIGN_IN_SENTENCE)


def _speech_refusal(exc: APIError) -> HTTPException:
    """Map an upstream speech failure onto this daemon's ``HTTPException``.

    The statuses in ``_SPEECH_REFUSAL_SENTENCES`` are refusals the user can act
    on; each passes through with its fixed sentence, and the upstream's body
    never reaches the response. Everything else -- other statuses, a transport
    failure, an error envelope inside a 200 -- keeps the 502 diagnostic path.
    """
    status = exc.status_code
    if status is not None:
        sentence = _SPEECH_REFUSAL_SENTENCES.get(status)
        if sentence is not None:
            return HTTPException(status_code=status, detail=sentence)
    return HTTPException(status_code=502, detail=_upstream_failure_detail(exc))


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
        _require_radient_credential(radient_client)
        audio_data = radient_client.create_speech(
            input_text=speech_request.input,
            instructions=speech_request.instructions,
            model=speech_request.model,
            voice=speech_request.voice,
            response_format=speech_request.response_format,
            speed=speech_request.speed,
            provider=speech_request.provider,
            language_code=speech_request.language_code,
        )

        media_type = f"audio/{speech_request.response_format}"
        return Response(content=audio_data, media_type=media_type)

    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except APIError as upstream_exc:
        # A classified upstream refusal: an error envelope inside a 200 (which
        # the client types as a failure instead of returning its bytes as
        # audio -- a 200 envelope never reaches the success path, which is how
        # a credential echoed into one was served as the audio response), a
        # refusal status, or a transport failure. The actionable statuses get
        # fixed sentences; everything else keeps the 502 diagnostic path. The
        # log line carries the upstream's designed `error` prose (str() of the
        # error), never the raw body.
        logger.warning(
            "Speech refused upstream (HTTP %s): %s", upstream_exc.status_code, upstream_exc
        )
        raise _speech_refusal(upstream_exc) from upstream_exc
    except Exception as e:
        # Catch any other exceptions and return a 500 error
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")


@router.post(
    "/v1/agents/{agent_id}/speech",
    tags=["Tools"],
    summary="Generate speech from an agent's last message",
    description="""Generates speech from an agent's last message, automatically determining the voice based on the agent's profile.""",  # noqa: E501
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
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
    config_manager: ConfigManager = Depends(get_config_manager),
    env_config: EnvConfig = Depends(get_env_config),
) -> Response:
    """
    Generates speech from an agent's last message.
    """
    try:
        try:
            agent = agent_registry.get_agent(agent_id)
        except KeyError:
            # ``AgentRegistry.get_agent`` raises ``KeyError`` for an unknown id
            # rather than answering None (its docstring is the contract), so
            # this is the 404 the route needs; without the catch the miss
            # surfaced as a 500 wrapping the registry's own message.
            raise HTTPException(
                status_code=404, detail=f"Agent with ID {agent_id} not found"
            ) from None
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent with ID {agent_id} not found")

        _require_radient_credential(radient_client)

        hosting = agent.hosting or config_manager.get_config_value("hosting")
        model_name = agent.model or config_manager.get_config_value("model_name")

        if not hosting:
            raise ValueError("Hosting platform is not configured.")
        if not model_name:
            raise ValueError("Model name is not configured.")

        model_info_client: Optional[Union[OpenRouterClient, RadientClient]] = None
        if hosting == "openrouter":
            # Store-first like the rest of the provider-key surface.
            from local_operator.providers.registry import provider_env_key

            raw_key = provider_env_key("openrouter", base=config_manager.config_dir)
            if raw_key:
                model_info_client = OpenRouterClient(SecretStr(raw_key))
            else:
                logger.warning("OpenRouter hosting selected but OPENROUTER_API_KEY not found.")
        elif hosting == "radient":
            from local_operator.providers.radient_credentials import (
                resolve_radient_credential,
            )

            api_key = await resolve_radient_credential(
                config_manager.config_dir,
                env_config.radient_api_base_url,
                store=provider_auth_store,
            )
            if api_key:
                model_info_client = RadientClient(api_key, env_config.radient_api_base_url)
            else:
                logger.warning("Radient hosting selected but RADIENT_API_KEY not found.")

        model_config = configure_model(
            hosting=hosting,
            model_name=model_name,
            config_dir=config_manager.config_dir,
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
            config_manager=config_manager,
            agent_registry=agent_registry,
            agent=agent,
        )

        voice = await determine_voice(agent, executor)

        # The hub's speak-aloud contract: provider named explicitly (ElevenLabs
        # primary; the OpenAI fallback is decided server-side, and only for
        # ElevenLabs unavailability), the voice travels as the female/male
        # ALIAS the hub resolves to a voice id, the model is omitted so the hub
        # owns model choice, and language_code is forwarded only when set. No
        # `instructions`: the OpenAI path's persona/delivery prompt has no
        # ElevenLabs equivalent, and the hub's fallback applies its own neutral
        # default -- a small delivery loss accepted for native multilingual
        # pronunciation.
        audio_data = radient_client.create_speech(
            input_text=speech_request.input_text,
            voice=voice,
            response_format=speech_request.response_format,
            speed=1.0,
            provider="elevenlabs",
            language_code=speech_request.language_code,
        )

        media_type = f"audio/{speech_request.response_format}"
        return Response(content=audio_data, media_type=media_type)

    except HTTPException as http_exc:
        # Routine refusals -- the no-credential 401, the unknown-agent 404 --
        # are logged as one line without a traceback; a full stack for an
        # everyday state buries the faults that need one. Unmapped 5xx keep
        # the traceback.
        if http_exc.status_code >= 500:
            logger.exception(f"HTTPException: {http_exc}")
        else:
            logger.warning(f"Speech request refused: {http_exc}")
        raise http_exc
    except APIError as upstream_exc:
        # Same classification as the /v1/tools/speech route above: an error
        # Radient reported inside a 200 body is raised as an upstream failure
        # rather than served as audio bytes, and the actionable refusal
        # statuses pass through with fixed sentences. The log line carries the
        # upstream's designed `error` prose, never the raw body.
        logger.warning(
            "Speech refused upstream (HTTP %s): %s", upstream_exc.status_code, upstream_exc
        )
        raise _speech_refusal(upstream_exc) from upstream_exc
    except Exception as e:
        logger.error(f"Failed to generate speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")
