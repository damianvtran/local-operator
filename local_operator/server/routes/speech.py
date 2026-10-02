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
from local_operator.stt.errors import PROVIDER_CREDIT_MARKERS
from local_operator.tts import VoicePath, VoicePathResolution
from local_operator.tts.adapters import Legacy
from local_operator.tts.cascade import (
    TtsUnavailable,
    resolve_voice_path,
    synthesize_speech,
)
from local_operator.tts.descriptor import DEFAULT_GENDER, descriptor_from_config

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


#: The sentence for the daemon-local no-credential refusal (the resolver found
#: nothing). It mirrors the app's own disabled-button sentence -- the speech
#: gate's speaking-aloud sign-in entry, "Sign in to Radient in Settings to
#: enable speaking aloud" (speech-gate.ts) -- so the toast and the sentence the
#: gate shows before a press agree. The app renamed the destination to
#: `Settings` and the control to `speaking aloud` when the gate module landed
#: (copy review C4 on the UI PR; the daemon side was deferred to this PR's
#: live-acceptance turn). This is NOT the state a press usually meets: the app
#: disables the speak button without a credential, so a press that fails
#: answers with the upstream 401 below instead (design round 1, D2).
SPEECH_NO_CREDENTIAL_SENTENCE = "Sign in to Radient in Settings to enable speaking aloud."

#: The sentence for an agent id that no longer exists. The raw id is an internal
#: identifier the user never chose, so it goes to the route's log line rather
#: than the customer's toast (design round 1, D1).
SPEECH_UNKNOWN_AGENT_SENTENCE = "This conversation's agent is no longer available."

#: The fixed sentences for the upstream refusals a user can act on, keyed by
#: the status agent-server passes through:
#:
#: * 401 a credential the hub refused or that expired -- the signed-in-but-
#:   broken state the app's button gate cannot catch -- so the sentence names
#:   the fix rather than claiming the user never signed in (design round 1,
#:   D2), with the destination in the app's `Settings` naming (copy review C4);
#: * 402 the hub's credit gate. Vendor 402s are absorbed into its 503 by
#:   design, so a 402 reaching the daemon is genuinely the account's balance
#:   (design round 1, D3), and it names where the top-up happens (D5);
#: * 429 a rate-limited or saturated upstream -- states the condition without
#:   attributing a mood to the service (N2);
#: * 503 the provider being temporarily unavailable -- agent-server's own
#:   sentence for a vendor 401/402/5xx, extended with its next step (D4).
#:
#: Fixed text on purpose: the hub's envelope is written for an operator
#: ("insufficient credits for this request"), not for the toast the user reads.
#: Every sentence ends with a full stop so the set reads alike in one toast (N1).
_SPEECH_REFUSAL_SENTENCES: Dict[int, str] = {
    401: "Your Radient sign-in has stopped working. Sign in again in Settings.",
    402: (
        "Your Radient credit balance is too low for speech. "
        "Add credits in the Radient Console to continue."
    ),
    429: "Speech is unavailable right now. Try again in a moment.",
    503: "Speech is temporarily unavailable. Try again in a moment.",
}

#: The same refusals for a leg the USER's own vendor key serves, parameterised
#: by the vendor's name (voicing S2 QA round 1, Q1 / security S-2). The Radient
#: The same refusals for a leg the USER's own vendor key serves. The Radient
#: sentences above name a party that is not in the exchange at all on a
#: BYO-only machine: telling someone their "Radient sign-in has stopped working"
#: when their own ElevenLabs key was refused sends them to fix the wrong thing,
#: and "Add credits in the Radient Console" sends them to a console they may
#: never have opened.
#:
#: TWO CONDITIONS, and they are not the two HTTP statuses (copy review round 1,
#: C1). ElevenLabs reports an exhausted quota as **401** (``quota_exceeded``)
#: and OpenAI reports an exhausted balance as **429**
#: (``insufficient_quota``/``credit_balance_exhausted``), so keying these on the
#: status alone told a user with a perfectly good key to replace it — churning a
#: working credential — and sent an out-of-credit user to "try again in a
#: moment". The condition is therefore classified by the response BODY, with the
#: same marker table the STT twin uses (:data:`PROVIDER_CREDIT_MARKERS`), and
#: only the leftover 401 is a key refusal.
#:
#: The remedy names the CREDENTIAL and no venue (C2): "Fix it in Settings" is
#: true of the desktop app and false of ``/login <provider>`` and
#: ``lop credential update <KEY>``, which are the other two ways a reader can
#: actually do this. No placeholders beyond ``{vendor}``, so no account or tenant
#: identity can be interpolated into a customer-visible string.
_VENDOR_REFUSAL_SENTENCES: Dict[str, str] = {
    #: ACTIVE voice, vendor as the actor (C3): "was refused" left the actor
    #: unsaid, and in an app that also refuses plenty of its own things it read
    #: as the app refusing it.
    "key": "{vendor} refused your API key. Replace it.",
    #: Parallel with the Radient sibling's "credit balance is too low for
    #: speech" and the family's plural "credits" (C4). "no speech credit left"
    #: implied a speech-specific balance neither vendor sells.
    "credit": (
        "Your {vendor} credit balance is too low for speech. "
        "Add credits with {vendor} to continue."
    ),
}

#: How each rung's vendor is named in a customer-facing sentence. The Radient
#: rung is deliberately absent: it keeps the sentences above, which are already
#: written for it.
_RUNG_VENDOR_LABELS: Dict[VoicePath, str] = {
    VoicePath.PROVIDER_TTS_ELEVENLABS: "ElevenLabs",
    VoicePath.PROVIDER_TTS_OPENAI: "OpenAI",
}


def _is_credit_refusal(exc: APIError) -> bool:
    """Whether a vendor refusal says "this account is out of credit".

    The marker table and the match (against the SCRUBBED body, lowercased) are
    the shared ones from :mod:`local_operator.stt.errors`, which is the mapper
    ``tts/clients.py`` already names as the single classifier for both speech
    directions. Two copies of this list would be two answers to the same
    vendor condition — exactly the drift the shared module exists to prevent
    (copy review round 1, C1).

    The body is the only input: a vendor's own words are what distinguishes an
    exhausted quota from a revoked key, and the status cannot do it.
    """
    body = (exc.body or "").lower()
    return any(marker in body for marker in PROVIDER_CREDIT_MARKERS)


def _require_radient_credential(radient_client: RadientClient) -> None:
    """Refuse with the sign-in remedy when no Radient credential resolved.

    The credential resolver answers "nothing found" with an empty ``SecretStr``
    rather than raising, so without this the absence would travel into the
    upstream call and come back as whatever an empty bearer produces. Both
    speech routes call this before doing any work.
    """
    if radient_client.api_key is not None and radient_client.api_key.get_secret_value():
        return
    raise HTTPException(status_code=401, detail=SPEECH_NO_CREDENTIAL_SENTENCE)


def _speech_refusal(exc: APIError, *, rung: Optional[VoicePath] = None) -> HTTPException:
    """Map an upstream speech failure onto this daemon's ``HTTPException``.

    The statuses in ``_SPEECH_REFUSAL_SENTENCES`` are refusals the user can act
    on; each passes through with its fixed sentence, and the upstream's body
    never reaches the response. Everything else -- other statuses, a transport
    failure, an error envelope inside a 200 -- keeps the 502 diagnostic path.

    ``rung`` selects the VOCABULARY. The default (``None``, and the Radient rung)
    keeps the Radient sentences the direct route has always used; a BYO rung gets
    the vendor's own words, because on that leg there may be no Radient account
    in the exchange at all (voicing S2 QA round 1, Q1 / security S-2).

    On a BYO leg the CONDITION is classified by the response BODY before the
    status arms (copy review round 1, C1): the vendors report an exhausted
    balance as 401 (ElevenLabs) and 429 (OpenAI), so the status alone cannot tell
    "your key is bad" from "your balance is empty". A credit refusal answers
    **402** — the codebase's own out-of-credit code, matching the STT twin — so a
    client branching on the status still reaches the top-up remedy.

    ``detail`` is ALWAYS one of these fixed sentences, never an upstream code or
    body: the desktop app maps exact strings (``DESIGNED_SPEECH_SENTENCES``), so
    a machine-readable code may ride alongside but must not replace the sentence.
    """
    status = exc.status_code
    if status is None:
        return HTTPException(status_code=502, detail=_upstream_failure_detail(exc))
    vendor = _RUNG_VENDOR_LABELS.get(rung) if rung is not None else None
    if vendor is None:
        sentence = _SPEECH_REFUSAL_SENTENCES.get(status)
        if sentence is not None:
            return HTTPException(status_code=status, detail=sentence)
        return HTTPException(status_code=502, detail=_upstream_failure_detail(exc))
    if _is_credit_refusal(exc):
        # 402 whatever the vendor called it: the condition is the user's, the
        # remedy is rung-independent, and it must not look retryable.
        return HTTPException(
            status_code=402,
            detail=_VENDOR_REFUSAL_SENTENCES["credit"].format(vendor=vendor),
        )
    if status == 401:
        return HTTPException(
            status_code=401,
            detail=_VENDOR_REFUSAL_SENTENCES["key"].format(vendor=vendor),
        )
    # A status with no vendor-specific sentence keeps the GENERIC one (429/503),
    # never the Radient one: those name a party that is not in this exchange.
    generic = _SPEECH_REFUSAL_SENTENCES.get(status)
    if status in (429, 503) and generic is not None:
        return HTTPException(status_code=status, detail=generic)
    return HTTPException(status_code=502, detail=_upstream_failure_detail(exc))


async def _synthesize_with_descriptor(
    text: str,
    *,
    response_format: str,
    legacy: Legacy,
    config_manager: ConfigManager,
    env_config: EnvConfig,
    provider_auth_store: AuthStore,
    radient_client: RadientClient,
    label: str,
    resolved_gender: Optional[str] = None,
    resolution: Optional[VoicePathResolution] = None,
) -> Response:
    """Build the descriptor, resolve the rung, synthesize, and report the leg.

    ONE implementation for both speech routes, because they answer the same
    question: the agent route resolves ``gender`` from its agent's classifier
    first and passes it in; the agent-LESS route (``POST /v1/tools/speech`` with
    no ``model``/``voice``) passes ``None`` and lets the descriptor's own
    ``resolved_gender`` map a configured ``auto`` onto the pool's nearest row —
    there is no agent to classify, and the hub refuses ``auto`` outright.

    ``resolution`` lets a caller that has ALREADY probed pass its answer in
    rather than pay for a second probe: the agent route resolves up front so its
    nothing-available refusal still precedes the classifier call, and a probe is
    not free (an expired OAuth row gets its refresh attempted and persisted, so
    a redundant one is a redundant network write).

    ``label`` is the caller's own name for its subject (the agent id, or the
    route's name); it rides the log lines only, never the response copy.

    Raises ``HTTPException`` for every refusal, so both routes' existing
    handlers stay in charge of the response shape: 401 when no rung can run,
    the classified refusal sentences for an upstream failure, and 500 with the
    real message (and a stack) for a defect in the serving leg.
    """
    if resolution is None:
        resolution = await resolve_voice_path(
            config_dir=config_manager.config_dir,
            base_url=env_config.radient_api_base_url,
            store=provider_auth_store,
        )
    if not resolution.servable:
        logger.warning("Speech request refused for %s: no speech path", label)
        raise HTTPException(status_code=401, detail=SPEECH_NO_CREDENTIAL_SENTENCE)

    descriptor = descriptor_from_config(config_manager, resolved_gender=resolved_gender)
    try:
        outcome = await synthesize_speech(
            text,
            config_dir=config_manager.config_dir,
            base_url=env_config.radient_api_base_url,
            store=provider_auth_store,
            descriptor=descriptor,
            legacy=legacy,
            response_format=response_format,
            resolution=resolution,
            radient_client=radient_client,
        )
    except TtsUnavailable as unavailable:
        error = unavailable.error
        if isinstance(error, APIError):
            logger.warning("Speech cascade exhausted for %s: %s", label, error)
            raise _speech_refusal(error, rung=unavailable.failed_path) from error
        if error is not None:
            logger.error(
                "Failed to generate speech: %s", error, exc_info=error, extra={"label": label}
            )
            raise HTTPException(status_code=500, detail=f"Failed to generate speech: {error}")
        logger.warning("Speech request refused for %s: no speech path", label)
        raise HTTPException(status_code=401, detail=SPEECH_NO_CREDENTIAL_SENTENCE)

    # The echoed-actual-path rule: relay the descriptors of the leg that
    # ACTUALLY served — the hub's own headers on a hub-served call, this
    # daemon's mapping headers on a BYO one — so a surface can tell what was
    # honoured without listening to the audio.
    return Response(
        content=outcome.audio,
        media_type=f"audio/{response_format}",
        headers=dict(outcome.speech_headers),
    )


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
    provider_auth_store: AuthStore = Depends(get_provider_auth_store),
    config_manager: ConfigManager = Depends(get_config_manager),
    env_config: EnvConfig = Depends(get_env_config),
) -> Response:
    """
    Generates speech from text using a specified provider and returns the audio data.
    This endpoint is protected by API key authentication and is subject to billing.

    TWO SHAPES, one route.

    * **Legacy**: the caller names ``model`` and ``voice``. Every field it sends
      is passed to the hub verbatim -- this is the pre-descriptor behaviour and
      it is unchanged, byte for byte.
    * **Descriptor-driven**: the caller sends neither. ``model``/``voice``
      therefore stop being required, and the daemon synthesizes exactly as the
      agent route does -- resolver, ``speech.voice.*`` descriptor, the same
      rung walk and the same relayed headers -- which is what the desktop UI's
      AGENT-LESS fallback posts (``{"input": "<text>"}``) when no agent
      binding is available. Before this it answered 422 on the missing fields.

    The trigger is the ABSENCE of both ``model`` and ``voice``, which no
    existing client can produce (both were required), so the new shape is
    additive rather than a behaviour change. An explicitly sent ``provider``
    is the exception: the descriptor has no field that could carry it, so a
    caller naming one is asking for the legacy pass-through and gets it.
    """
    try:
        if (
            speech_request.model is None
            and speech_request.voice is None
            and "provider" not in speech_request.model_fields_set
        ):
            # The descriptor path. ``speed`` and ``instructions`` ARE descriptor
            # fields (pace, instructions), so an explicitly sent one rides as a
            # legacy pin rather than forcing the old shape; anything the caller
            # left out pins nothing and lets the configured dial govern. No
            # agent exists here, so ``gender`` is not classified -- the helper
            # passes no resolved gender and ``auto`` maps onto the pool's own
            # nearest row.
            explicit = speech_request.model_fields_set
            return await _synthesize_with_descriptor(
                speech_request.input,
                response_format=speech_request.response_format,
                legacy=Legacy(
                    instructions=speech_request.instructions or "",
                    speed=speech_request.speed if "speed" in explicit else None,
                    language_code=speech_request.language_code or "",
                ),
                config_manager=config_manager,
                env_config=env_config,
                provider_auth_store=provider_auth_store,
                radient_client=radient_client,
                label="direct route",
            )

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
        # The stack is logged, not just the message: an unexpected fault is the
        # one class that needs it (review round 2, finding 1).
        logger.exception("Failed to generate speech: %s", e)
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

    THE CREDENTIAL POSTURE, stated because it is easy to "fix" by accident. In
    the shipped desktop posture this route is gated twice -- the ``/v1/agents``
    prefix gate and ``require_desktop`` (which arrives on
    ``get_provider_auth_store``, not on ``get_radient_client`` -- review round 2,
    N5) -- so an anonymous cross-origin caller never reaches the spend. In the
    STANDALONE posture (``lop server`` with no desktop plane) there is no such
    gate, and the daemon admits ``*`` origins, so a page the operator visits can
    drive this route. That class is pre-existing and identical for the Radient
    rung; what this route must not do is make it UNBOUNDED, which is why
    ``AgentSpeechRequest.input_text`` is capped at the hub's own 10,000
    characters (voicing S2 security round 1, S-1): the request carries no
    agent identity, so the cap is the whole bound on what one call can spend on
    the operator's own vendor key. What this route deliberately does NOT do is
    refuse a BYO-only machine until it also holds a Radient credential -- that
    would defeat the rung the feature exists for.

    WHY NO ROUTE-LOCAL ORIGIN PRECONDITION (review round 2, M3 corrected a
    wrong reason here). The app's own speak-aloud is NOT a renderer fetch: it
    is issued by the Electron main process over its typed media relay
    (``ipcRenderer.invoke("desktop-media", ...)`` -> the HTTP call in main, with
    the bearer attached there), and the browser-dev fallback posts to the
    server-side ``/__desktop/media`` proxy. Both are Origin-absent, so an
    origin precondition would NOT have taken the app's own call down -- the
    earlier ``file://``/``null`` rationale is withdrawn. The precondition is
    still rejected, on the ground that survives: refusing origins is a
    surface-wide CORS/origin-policy decision rather than one route's, the
    standalone daemon's wildcard-echo posture is the condition that would have
    to change with it, and the plane's own history records the cost of
    crash-then-tighten here. The cap bounds the per-call spend, which is what
    this delta owes (security round 2, S-6).
    """
    try:
        try:
            agent = agent_registry.get_agent(agent_id)
        except KeyError:
            # ``AgentRegistry.get_agent`` raises ``KeyError`` for an unknown id
            # rather than answering None (its docstring is the contract), so
            # this is the 404 the route needs; without the catch the miss
            # surfaced as a 500 wrapping the registry's own message. The raw
            # id stays in the log line, not the copy (design round 1, D1).
            raise HTTPException(status_code=404, detail=SPEECH_UNKNOWN_AGENT_SENTENCE) from None
        if not agent:
            raise HTTPException(status_code=404, detail=SPEECH_UNKNOWN_AGENT_SENTENCE)

        # The resolver decides which rung can run, and it runs FIRST -- before
        # any model configuration or classifier call -- so the nothing-available
        # refusal is still cheap and still happens before any work, exactly as
        # the credential gate it replaces did. Which rung is not this route's
        # decision: a signed-in Radient account goes through the hub (which owns
        # its own ElevenLabs→OpenAI cascade and maps the descriptor for
        # whichever leg serves), and a BYO-only machine maps and calls the
        # vendor itself.
        resolution = await resolve_voice_path(
            config_dir=config_manager.config_dir,
            base_url=env_config.radient_api_base_url,
            store=provider_auth_store,
        )
        if not resolution.servable:
            logger.warning("Speech request refused for agent %s: no speech path", agent_id)
            raise HTTPException(status_code=401, detail=SPEECH_NO_CREDENTIAL_SENTENCE)

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

        # The voicing descriptor: the configured dials, with ``gender``
        # resolved from the classifier ONLY when the setting says ``auto`` (the
        # hub refuses ``auto`` — it has no agent context to classify with, so
        # this is the one field the daemon must resolve before sending). A
        # fixed gender therefore skips the classifier entirely, which is both
        # faster and the point of setting one.
        configured_gender = config_manager.get_nested_value(
            ("speech", "voice", "gender"), DEFAULT_GENDER
        )
        if configured_gender == "auto":
            resolved_gender = await determine_voice(agent, executor)
        else:
            resolved_gender = str(configured_gender)

        # The descriptor build, the rung walk and the header relay live in the
        # shared helper, because the agent-LESS direct route performs exactly
        # the same synthesis (it just has no agent to classify a gender from).
        return await _synthesize_with_descriptor(
            speech_request.input_text,
            response_format=speech_request.response_format,
            # Only the CALLER's own per-call language rides the legacy field;
            # an absent one pins nothing, so the descriptor's language governs.
            legacy=Legacy(language_code=speech_request.language_code or ""),
            config_manager=config_manager,
            env_config=env_config,
            provider_auth_store=provider_auth_store,
            radient_client=radient_client,
            label=f"agent {agent_id}",
            resolved_gender=resolved_gender,
            resolution=resolution,
        )

    except HTTPException as http_exc:
        # Routine refusals -- the no-credential 401, the unknown-agent 404 --
        # are logged as one line without a traceback; a full stack for an
        # everyday state buries the faults that need one. Unmapped 5xx keep
        # the traceback. The agent id rides the warning (not the response
        # copy) so a support reader can still correlate the refusal.
        if http_exc.status_code >= 500:
            logger.exception("HTTPException: %s", http_exc)
        else:
            logger.warning("Speech request refused for agent %s: %s", agent_id, http_exc)
        raise http_exc
    except TtsUnavailable as unavailable:
        # The cascade's own refusal, in its three shapes. An ``APIError`` is an
        # upstream refusal and keeps the per-status sentences the direct route
        # uses. Anything else is a DEFECT in the leg that was serving, so it
        # surfaces as the 500 this route has always given that class — with the
        # real message and a stack, never a misleading "not signed in". And no
        # failure at all means no rung could run (the resolver reported the
        # same, so this is the belt to its braces): the sign-in sentence.
        error = unavailable.error
        if isinstance(error, APIError):
            logger.warning("Speech cascade exhausted for agent %s: %s", agent_id, error)
            raise _speech_refusal(error, rung=unavailable.failed_path) from error
        if error is not None:
            logger.error(
                "Failed to generate speech: %s", error, exc_info=error, extra={"agent_id": agent_id}
            )
            raise HTTPException(status_code=500, detail=f"Failed to generate speech: {error}")
        logger.warning("Speech request refused for agent %s: no speech path", agent_id)
        raise HTTPException(status_code=401, detail=SPEECH_NO_CREDENTIAL_SENTENCE)
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
        # The stack is logged, not just the message: an unexpected fault is the
        # one class that needs it (review round 2, finding 1).
        logger.exception("Failed to generate speech: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")
