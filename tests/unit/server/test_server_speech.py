from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import SecretStr, ValidationError

from local_operator.agents import AgentData, AgentRegistry
from local_operator.clients._http import APIError
from local_operator.config import ConfigManager
from local_operator.providers.auth_store import AuthStore
from local_operator.server.models.schemas import AgentSpeechRequest, SpeechRequest
from local_operator.server.routes.speech import create_agent_speech, create_speech
from local_operator.tts.descriptor import DEFAULT_SPEECH_INSTRUCTIONS


def _speech_store(
    tmp_path: Path, *, radient: bool = True, elevenlabs: Optional[str] = None
) -> AuthStore:
    """A REAL store on an isolated root, optionally holding logins.

    The route's rung decision comes from the store's PERSISTED rows, so a mock
    here would answer a question the resolver never asks. A real store is also
    what makes ``test_create_agent_speech_requires_a_credential_before_any_work``
    mean "nothing is stored" rather than "the mock happened to be falsy".
    """
    store = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path / "config")
    if radient:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "refresh": "r",
                "access": "a",
                "expires": int(datetime.now().timestamp() * 1000) + 3_600_000,
            },
        )
    if elevenlabs:
        store.upsert_credential(
            "elevenlabs", {"type": "api_key", "source": "login", "key": elevenlabs}
        )
    return store


@pytest.fixture
def mock_radient_client():
    """Fixture for a mocked Radient client."""
    return MagicMock()


@pytest.fixture
def speech_request_data():
    """Fixture for speech request data."""
    return {
        "input": "Hello, world!",
        "instructions": "Please speak in a friendly and engaging tone.",
        "model": "tts-1",
        "voice": "alloy",
        "response_format": "mp3",
        "speed": 1.0,
        "provider": "openai",
    }


def _speech_logged_the_stack(caplog) -> None:
    """Assert the route logged an ERROR whose stack survived to ``caplog``.

    Selected by logger name and message rather than level alone, so an
    unrelated ERROR record cannot satisfy it and a missing record reads as an
    assertion failure instead of a StopIteration (review round 3, minor).
    ``exc_text`` counts alongside ``exc_info``: ``local_operator.mcp.redaction``
    rewrites a record's exc_info into exc_text once any earlier test in the
    worker has registered a credential, and the stack is present in either
    form (the review round 3 blocker: asserting on ``exc_info`` alone made
    these cells red in exactly the shards that run MCP tests first).
    """
    records = [
        record
        for record in caplog.records
        if record.name == "local_operator.server.routes.speech"
        and record.levelname == "ERROR"
        and "Failed to generate speech" in record.getMessage()
    ]
    assert records, "no ERROR record from the speech route"
    record = records[0]
    assert record.exc_info is not None or record.exc_text is not None


@pytest.mark.asyncio
async def test_create_speech_success(speech_request_data, mock_radient_client):
    """Test successful speech creation."""
    mock_radient_client.create_speech.return_value = b"audio_data"
    speech_request = SpeechRequest(**speech_request_data)

    response = await create_speech(speech_request, mock_radient_client)

    assert response.status_code == 200
    assert response.body == b"audio_data"
    assert response.media_type == "audio/mp3"
    # An ABSENT descriptor means exactly the pre-voicing request: the pass-through
    # forwards the caller's own fields and adds no `voice_descriptor`, so a
    # client that names a voice keeps today's semantics (the agent speak-aloud
    # route is the one that builds and sends a descriptor).
    mock_radient_client.create_speech.assert_called_once_with(
        input_text="Hello, world!",
        instructions="Please speak in a friendly and engaging tone.",
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        speed=1.0,
        provider="openai",
        language_code=None,
    )


def _agent_less_request(**overrides: Any) -> SpeechRequest:
    """The UI's fallback shape: ``input`` and nothing else (see ``_speech_request``).

    Built through a typed dict for the same reason `_speech_request` is: the
    type checker treats a pydantic model's defaulted fields as required keyword
    arguments, so naming only what the caller sends does not type-check.
    """
    fields: Dict[str, Any] = {"input": "Hello"}
    fields.update(overrides)
    return SpeechRequest(**fields)


@pytest.mark.asyncio
async def test_the_agent_less_descriptor_request_synthesizes(tmp_path):
    """The UI's fallback payload -- ``{"input": ...}`` and nothing else -- speaks.

    This is the shape the desktop UI posts when no agent binding is available
    (cross-repo: its companion PR's fallback). ``model`` and ``voice`` used to be
    REQUIRED here, so that request answered 422 and the fallback had nowhere to
    go. The daemon now builds the same descriptor the agent route builds -- with
    ``gender`` mapped off ``auto`` by the descriptor itself, since there is no
    agent to classify -- and sends it with no legacy pin beside it, so the hub's
    own cascade and model choice stay server-owned.
    """
    client = _credentialed_client()

    response = await create_speech(
        _agent_less_request(),
        client,
        _speech_store(tmp_path),
        _voiced_config(tmp_path),
        _env_config(),
    )

    assert response.status_code == 200
    assert response.body == b"audio_data"
    assert response.headers["x-radient-speech-path"] == "provider_tts_radient"
    sent = client.create_speech_response.call_args.kwargs
    # No legacy pin: not provider, not speed, not an empty voice string.
    assert sent["provider"] is None
    assert sent["model"] is None
    assert sent["voice"] is None
    assert sent["speed"] is None
    # `auto` is the default configured gender and the hub refuses it, so the
    # descriptor resolves it onto the map's unknown-gender row.
    assert sent["voice_descriptor"]["gender"] == "female"


@pytest.mark.asyncio
async def test_the_agent_less_descriptor_request_runs_a_byo_key(tmp_path, monkeypatch):
    """And on a machine with no Radient account, the same payload still speaks.

    The resolver's order is Radient -> ElevenLabs -> OpenAI, so an ElevenLabs
    key alone serves the agent-less fallback through the daemon's own mapping:
    this is the follower case, and it is why the descriptor path goes through
    the cascade rather than the direct hub pass-through.
    """
    from local_operator.tts import clients as tts_clients
    from local_operator.tts.clients import TtsClientResult

    class _Serving:
        def __init__(self, _api_key, **_kwargs):
            # The key is never kept: this fake exists to prove the RESOLVER
            # routed to ElevenLabs, not to exercise transport.
            pass

        async def synthesize(self, *_args, **_kwargs):
            return TtsClientResult(
                audio=b"byo-audio", model="eleven_turbo_v2_5", provider="elevenlabs"
            )

    monkeypatch.setattr(tts_clients, "ElevenLabsTtsClient", _Serving)

    response = await create_speech(
        _agent_less_request(),
        _credentialed_client(),
        _speech_store(tmp_path, radient=False, elevenlabs="el-key"),
        _voiced_config(tmp_path),
        _env_config(),
    )

    assert response.status_code == 200
    assert response.body == b"byo-audio"
    assert response.headers["x-radient-speech-path"] == "provider_tts_elevenlabs"


@pytest.mark.asyncio
async def test_an_explicit_provider_keeps_the_direct_pass_through(tmp_path):
    """A caller naming a provider is asking for the legacy shape, and gets it.

    The descriptor has no field that could carry ``provider``, so letting this
    request take the descriptor path would silently DROP the caller's pin. It
    takes the pass-through instead, unchanged from before this slice.
    """
    client = _credentialed_client()

    response = await create_speech(
        _agent_less_request(provider="elevenlabs"),
        client,
        _speech_store(tmp_path),
        _voiced_config(tmp_path),
        _env_config(),
    )

    assert response.status_code == 200
    client.create_speech.assert_called_once_with(
        input_text="Hello",
        instructions=None,
        model=None,
        voice=None,
        response_format="mp3",
        speed=1.0,
        provider="elevenlabs",
        language_code=None,
    )


@pytest.mark.asyncio
async def test_an_explicit_speed_pins_pace_on_the_descriptor_path(tmp_path):
    """``speed`` IS a legacy field the hub can pin, so the descriptor path keeps it.

    The difference from ``provider`` above is deliberate: the hub's per-field pin
    rule has somewhere to put a speed (it WINS over the descriptor's pace) and
    nowhere to put a provider. So the speed rides as ``speed`` and is not folded
    into the descriptor.
    """
    client = _credentialed_client()

    await create_speech(
        _agent_less_request(speed=0.5),
        client,
        _speech_store(tmp_path),
        _voiced_config(tmp_path),
        _env_config(),
    )

    sent = client.create_speech_response.call_args.kwargs
    assert sent["speed"] == 0.5


@pytest.mark.asyncio
async def test_create_speech_http_exception(speech_request_data, mock_radient_client):
    """Test speech creation when Radient client raises HTTPException."""
    mock_radient_client.create_speech.side_effect = HTTPException(
        status_code=404, detail="Model not found"
    )
    speech_request = SpeechRequest(**speech_request_data)

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 404
    assert "Model not found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_speech_generic_exception(speech_request_data, mock_radient_client, caplog):
    """Test speech creation when Radient client raises a generic exception."""
    mock_radient_client.create_speech.side_effect = Exception("Something went wrong")
    speech_request = SpeechRequest(**speech_request_data)

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 500
    assert "Failed to generate speech: Something went wrong" in exc_info.value.detail
    # The one fault class that needs a stack gets one (review round 2, finding 1).
    _speech_logged_the_stack(caplog)


@pytest.mark.asyncio
async def test_create_speech_passes_language_code_through(mock_radient_client):
    """The language code is plumbing: forwarded exactly when the caller set it."""
    mock_radient_client.create_speech.return_value = b"audio_data"
    speech_request = _speech_request(input="Hola", language_code="es")

    response = await create_speech(speech_request, mock_radient_client)

    assert response.status_code == 200
    mock_radient_client.create_speech.assert_called_once_with(
        input_text="Hola",
        instructions=None,
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        speed=1.0,
        provider="openai",
        language_code="es",
    )


@pytest.mark.parametrize("missing", [None, SecretStr("")])
@pytest.mark.asyncio
async def test_create_speech_without_a_credential_answers_401(mock_radient_client, missing):
    """No resolved credential is the sign-in remedy, before any upstream work."""
    mock_radient_client.api_key = missing
    speech_request = _speech_request()

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Sign in to Radient in Settings to enable speaking aloud."
    mock_radient_client.create_speech.assert_not_called()


@pytest.mark.parametrize(
    ("status", "sentence"),
    [
        # An upstream 401 is the signed-in-but-broken state (the app's button
        # gate never lets the absent-credential state reach a press), so it
        # gets its own sentence; the daemon-local one is pinned in the
        # no-credential tests.
        (
            401,
            "Your Radient sign-in has stopped working. Sign in again in Settings.",
        ),
        (
            402,
            "Your Radient credit balance is too low for speech. "
            "Add credits in the Radient Console to continue.",
        ),
        (429, "Speech is unavailable right now. Try again in a moment."),
        (503, "Speech is temporarily unavailable. Try again in a moment."),
    ],
)
@pytest.mark.asyncio
async def test_create_speech_passes_actionable_refusals_through(
    mock_radient_client, status, sentence
):
    """The four actionable statuses keep their status and get fixed copy.

    The upstream's own envelope must not reach the response: it is written for
    an operator, and the raw frame can carry request material.
    """
    mock_radient_client.create_speech.side_effect = APIError(
        "insufficient credits for this request",
        status_code=status,
        body='{"error": "insufficient credits for this request", "code": "billing"}',
    )
    speech_request = _speech_request()

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == status
    assert exc_info.value.detail == sentence


@pytest.mark.asyncio
async def test_create_speech_keeps_the_502_path_for_an_error_envelope(mock_radient_client):
    """A 200 body that is an error envelope keeps the pre-existing 502 diagnostic."""
    mock_radient_client.create_speech.side_effect = APIError(
        "Failed to generate speech: Radient returned an error body with a 200 status",
        status_code=200,
        body='{"error": "vendor quota exceeded"}',
    )
    speech_request = _speech_request()

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 502
    assert "Radient reported an error (HTTP 200)" in exc_info.value.detail
    # The envelope's designed `error` field is extracted; the raw frame is not quoted.
    assert exc_info.value.detail.endswith("vendor quota exceeded")
    assert '{"error"' not in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_speech_extracts_the_error_field_for_other_statuses(mock_radient_client):
    """A non-actionable status keeps 502 but carries only the designed `error` prose."""
    mock_radient_client.create_speech.side_effect = APIError(
        "Invalid model",
        status_code=400,
        body='{"error": "unknown model eleven_bogus_v1", "code": "invalid_request"}',
    )
    speech_request = _speech_request()

    with pytest.raises(HTTPException) as exc_info:
        await create_speech(speech_request, mock_radient_client)

    assert exc_info.value.status_code == 502
    assert "unknown model eleven_bogus_v1" in exc_info.value.detail
    assert '{"error"' not in exc_info.value.detail


def _speech_request(**overrides: Any) -> SpeechRequest:
    """Build a valid direct-route request.

    The fields travel through a typed dict because the type checker treats a
    pydantic model's defaulted fields as required keyword arguments.
    """
    fields: Dict[str, Any] = {"input": "Hello", "model": "tts-1", "voice": "alloy"}
    fields.update(overrides)
    return SpeechRequest(**fields)


def _agent_speech_request(**overrides: Any) -> AgentSpeechRequest:
    """Build a valid agent-route request (see ``_speech_request``)."""
    fields: Dict[str, Any] = {"input_text": "Hello"}
    fields.update(overrides)
    return AgentSpeechRequest(**fields)


def _agent() -> AgentData:
    fields: Dict[str, Any] = {
        "id": "test-agent",
        "name": "Aria",
        "created_date": datetime.now(),
        "version": "1.0.0",
        "description": "A friendly assistant",
        "hosting": "openai",
        "model": "gpt-4o",
    }
    return AgentData(**fields)


def _credentialed_client() -> MagicMock:
    client = MagicMock()
    client.api_key = SecretStr("test-key")
    client.create_speech.return_value = b"audio_data"
    # The cascade calls the HEADER-CARRYING variant so it can relay the hub's
    # own ``X-Radient-Speech-Provider`` (the echoed-actual-path rule: a
    # descriptor-bearing request names no leg, so only the hub's receipt knows
    # which one served). ``create_speech`` stays wired for the untouched
    # ``/v1/tools/speech`` pass-through.
    client.create_speech_response.return_value = (b"audio_data", {})
    return client


def _env_config() -> Any:
    """The route reads exactly one field off this: the Radient base URL.

    A URL must be a string, so a MagicMock here is not a stand-in for the real
    object — it is an input the resolver will try to parse.
    """
    return SimpleNamespace(radient_api_base_url="https://api.radienthq.com/v1")


def _voiced_config(tmp_path: Path) -> ConfigManager:
    """A real config manager over an isolated root: ``speech.voice.*`` unset.

    Unset means the DEFAULTS, which is what makes ``determine_voice`` the
    patched classifier call these tests expect — the configured gender is
    ``auto``.
    """
    return ConfigManager(config_dir=tmp_path / "config")


@pytest.mark.asyncio
async def test_create_agent_speech_uses_the_elevenlabs_contract(tmp_path):
    """The descriptor is sent; no legacy provider/voice/speed/instructions ride along.

    A descriptor-bearing request names NO leg on purpose: the hub owns provider
    and model choice, and an explicit ``voice``/``speed`` sent beside the
    descriptor would PIN those fields and defeat it (the hub's per-field pin
    rule). ``language_code`` stays, because it is the CALLER's own per-call
    value and the hub's pin rule is how it survives.
    """
    radient_client = _credentialed_client()
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()

    with (
        patch("local_operator.server.routes.speech.configure_model", return_value=MagicMock()),
        patch(
            "local_operator.server.routes.speech.determine_voice",
            new_callable=AsyncMock,
            return_value="female",
        ),
    ):
        response = await create_agent_speech(
            "test-agent",
            _agent_speech_request(language_code="es"),
            radient_client,
            agent_registry,
            _speech_store(tmp_path),
            _voiced_config(tmp_path),
            _env_config(),
        )

    assert response.status_code == 200
    assert response.body == b"audio_data"
    # The daemon's own receipt names the rung that ran, even though the hub
    # named no leg back in this fake (``provider_tts_radient`` is the rung).
    assert response.headers["x-radient-speech-path"] == "provider_tts_radient"
    # assert_called_once_with pins the ABSENCE of provider/voice/speed/model too.
    radient_client.create_speech_response.assert_called_once_with(
        "Hello",
        model=None,
        voice=None,
        instructions=None,
        response_format="mp3",
        speed=None,
        provider=None,
        language_code="es",
        voice_descriptor={
            "version": 1,
            "gender": "female",
            "tone": "warm",
            "expressiveness": "medium",
            "language": "auto",
            "accent": "",
            "pace": 1.0,
            "instructions": DEFAULT_SPEECH_INSTRUCTIONS,
        },
    )


@pytest.mark.asyncio
async def test_create_agent_speech_relays_the_hubs_own_voicing_headers(tmp_path):
    """The echoed-actual-path rule: the hub's receipt is what says who spoke.

    A descriptor-bearing request names no leg, so ``X-Radient-Speech-Provider``
    on the response is the ONLY place the serving leg exists. Relaying it (with
    the map version and the degraded tokens) is what stops the daemon from
    claiming a leg the hub did not use.
    """
    radient_client = _credentialed_client()
    radient_client.create_speech_response.return_value = (
        b"audio",
        {
            "X-Radient-Speech-Map": "1.0",
            "X-Radient-Speech-Provider": "openai",
            "X-Radient-Speech-Applied": "gender,pace",
            "X-Radient-Speech-Degraded": "tone:emulated=instructions",
            # A header that is none of our business must not be copied.
            "X-Internal-Trace": "secret",
        },
    )
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()

    with (
        patch("local_operator.server.routes.speech.configure_model", return_value=MagicMock()),
        patch(
            "local_operator.server.routes.speech.determine_voice",
            new_callable=AsyncMock,
            return_value="female",
        ),
    ):
        response = await create_agent_speech(
            "test-agent",
            _agent_speech_request(),
            radient_client,
            agent_registry,
            _speech_store(tmp_path),
            _voiced_config(tmp_path),
            _env_config(),
        )

    assert response.headers["x-radient-speech-provider"] == "openai"
    assert response.headers["x-radient-speech-map"] == "1.0"
    assert response.headers["x-radient-speech-degraded"] == "tone:emulated=instructions"
    # TWO DIFFERENT FACTS, deliberately kept apart (voicing S2 review round 1,
    # M1): Path is the DAEMON RUNG that executed (the closed VoicePath
    # vocabulary), while Provider is the leg the HUB says it used. Filling Path
    # from the hub's header made "the hub used the platform's ElevenLabs" and
    # "my own ElevenLabs key ran" the same string.
    assert response.headers["x-radient-speech-path"] == "provider_tts_radient"
    assert "x-internal-trace" not in response.headers


@pytest.mark.asyncio
async def test_a_byo_refusal_names_the_users_own_vendor(tmp_path, monkeypatch):
    """Q1/C1: a vendor refusal names the user's own vendor AND its real condition.

    On a BYO-only machine there is no Radient account in the exchange, so
    "your Radient sign-in has stopped working" and "add credits in the Radient
    Console" both name the wrong system and send the user to the wrong place
    (Q1). And the CONDITION cannot be read off the status (C1): ElevenLabs
    reports an exhausted quota as 401 and OpenAI reports one as 429, so a
    status-keyed table told users with a good key to replace it. The body
    markers decide, and the two conditions have two different remedies.
    """
    from local_operator.clients._http import APIError
    from local_operator.tts import clients as tts_clients

    class _Refusing:
        #: Set per case below; declared so the type checker sees it as a class
        #: attribute rather than an assignment to an unknown name.
        status: int = 401
        body: Optional[str] = None

        def __init__(self, *_args, **_kwargs):
            pass

        async def synthesize(self, *_args, **_kwargs):
            raise APIError("vendor said no", status_code=_Refusing.status, body=_Refusing.body)

    monkeypatch.setattr(tts_clients, "ElevenLabsTtsClient", _Refusing)
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()

    key_sentence = "ElevenLabs refused your API key. Replace it."
    credit_sentence = (
        "Your ElevenLabs credit balance is too low for speech. "
        "Add credits with ElevenLabs to continue."
    )
    for status, body, expected_status, expected in (
        # A plain 401 with no credit language is the only key refusal.
        (401, '{"detail":{"status":"invalid_api_key"}}', 401, key_sentence),
        # ElevenLabs' exhausted quota, on the status it actually uses.
        (
            401,
            '{"detail":{"status":"quota_exceeded","message":"You have insufficient '
            'quota to complete the request."}}',
            402,
            credit_sentence,
        ),
        # OpenAI's exhausted balance, on the status it actually uses.
        (
            429,
            '{"error":{"code":"credit_balance_exhausted"}}',
            402,
            credit_sentence,
        ),
        (429, '{"error":{"code":"rate_limit_exceeded"}}', 429, None),
    ):
        _Refusing.status = status
        _Refusing.body = body
        with (
            patch("local_operator.server.routes.speech.configure_model", return_value=MagicMock()),
            patch(
                "local_operator.server.routes.speech.determine_voice",
                new_callable=AsyncMock,
                return_value="female",
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_agent_speech(
                    "test-agent",
                    _agent_speech_request(),
                    _credentialed_client(),
                    agent_registry,
                    _speech_store(tmp_path, radient=False, elevenlabs="el-key"),
                    _voiced_config(tmp_path),
                    _env_config(),
                )
        assert exc_info.value.status_code == expected_status
        if expected is None:
            # A rate limit is not a credit condition: it keeps the sentence
            # that names nobody, so it is not asserted as a vendor literal.
            assert "ElevenLabs" not in exc_info.value.detail
        else:
            assert exc_info.value.detail == expected
        assert "Radient" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_a_generic_vendor_refusal_keeps_the_provider_neutral_sentence(tmp_path, monkeypatch):
    """429 has no vendor sentence: the generic one already names nobody."""
    from local_operator.clients._http import APIError
    from local_operator.tts import clients as tts_clients

    class _Refusing:
        def __init__(self, *_args, **_kwargs):
            pass

        async def synthesize(self, *_args, **_kwargs):
            raise APIError("slow down", status_code=429)

    monkeypatch.setattr(tts_clients, "ElevenLabsTtsClient", _Refusing)
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()
    with (
        patch("local_operator.server.routes.speech.configure_model", return_value=MagicMock()),
        patch(
            "local_operator.server.routes.speech.determine_voice",
            new_callable=AsyncMock,
            return_value="female",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_agent_speech(
                "test-agent",
                _agent_speech_request(),
                _credentialed_client(),
                agent_registry,
                _speech_store(tmp_path, radient=False, elevenlabs="el-key"),
                _voiced_config(tmp_path),
                _env_config(),
            )

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail == "Speech is unavailable right now. Try again in a moment."


def test_an_over_long_speak_aloud_input_is_refused_at_the_schema():
    """S-1: the agent route carries no identity, so the cap IS the bound.

    It matches the hub's own 10,000-character cap, so the daemon refuses at the
    same boundary rather than forwarding a body the hub would reject — and on a
    BYO leg nothing else bounds what one call can spend of the operator's key.
    """
    from local_operator.server.models.schemas import MAX_SPEECH_INPUT_CHARS

    assert _agent_speech_request(input_text="x" * MAX_SPEECH_INPUT_CHARS).input_text
    with pytest.raises(ValidationError):
        _agent_speech_request(input_text="x" * (MAX_SPEECH_INPUT_CHARS + 1))


@pytest.mark.asyncio
async def test_create_agent_speech_404_for_an_unknown_agent(tmp_path, caplog):
    """The 404 comes from the registry's real miss path, not a mocked fiction.

    ``AgentRegistry.get_agent`` raises ``KeyError`` for an id it does not hold;
    driving the real registry here is what makes this the regression test for
    that miss having surfaced as a 500.
    """
    registry = AgentRegistry(tmp_path)
    radient_client = _credentialed_client()

    with pytest.raises(HTTPException) as exc_info:
        await create_agent_speech(
            "missing-agent",
            _agent_speech_request(),
            radient_client,
            registry,
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "This conversation's agent is no longer available."
    radient_client.create_speech.assert_not_called()
    # The raw id stays support-visible through the log line, not the copy
    # (design round 1, D1).
    assert any(
        record.levelname == "WARNING" and "missing-agent" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_create_agent_speech_requires_a_credential_before_any_work(tmp_path):
    """No STORED credential answers 401 before configuration or any model call.

    The resolver decides that, so an ambient key is not a credential: only the
    persisted rows in the store can light a rung, and an empty store lights
    none.
    """
    radient_client = MagicMock()
    radient_client.api_key = SecretStr("")
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()

    with (
        patch("local_operator.server.routes.speech.configure_model") as configure_call,
        patch("local_operator.server.routes.speech.determine_voice") as voice_call,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_agent_speech(
                "test-agent",
                _agent_speech_request(),
                radient_client,
                agent_registry,
                _speech_store(tmp_path, radient=False),
                _voiced_config(tmp_path),
                _env_config(),
            )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Sign in to Radient in Settings to enable speaking aloud."
    configure_call.assert_not_called()
    voice_call.assert_not_called()
    radient_client.create_speech.assert_not_called()


@pytest.mark.asyncio
async def test_create_agent_speech_passes_refusals_through(tmp_path):
    """The agent route shares the direct route's refusal classification."""
    radient_client = _credentialed_client()
    radient_client.create_speech_response.side_effect = APIError(
        "insufficient credits for this request", status_code=402
    )
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()

    with (
        patch("local_operator.server.routes.speech.configure_model", return_value=MagicMock()),
        patch(
            "local_operator.server.routes.speech.determine_voice",
            new_callable=AsyncMock,
            return_value="male",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_agent_speech(
                "test-agent",
                _agent_speech_request(),
                radient_client,
                agent_registry,
                _speech_store(tmp_path),
                _voiced_config(tmp_path),
                _env_config(),
            )

    assert exc_info.value.status_code == 402
    assert exc_info.value.detail == (
        "Your Radient credit balance is too low for speech. "
        "Add credits in the Radient Console to continue."
    )


@pytest.mark.asyncio
async def test_create_agent_speech_500_logs_the_stack(tmp_path, caplog):
    """The one fault class that needs a traceback gets it (review r2, f1)."""
    radient_client = _credentialed_client()
    radient_client.create_speech_response.side_effect = RuntimeError("boom")
    agent_registry = MagicMock()
    agent_registry.get_agent.return_value = _agent()

    with (
        patch("local_operator.server.routes.speech.configure_model", return_value=MagicMock()),
        patch(
            "local_operator.server.routes.speech.determine_voice",
            new_callable=AsyncMock,
            return_value="male",
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await create_agent_speech(
                "test-agent",
                _agent_speech_request(),
                radient_client,
                agent_registry,
                _speech_store(tmp_path),
                _voiced_config(tmp_path),
                _env_config(),
            )

    assert exc_info.value.status_code == 500
    assert "Failed to generate speech: boom" in exc_info.value.detail
    _speech_logged_the_stack(caplog)


@pytest.mark.parametrize("bad", ["EN", "En", "en-US", "e", "eng", "1a", " e", "éé"])
def test_language_codes_must_be_iso639_1(bad):
    """Anything but two lowercase letters is refused at the schema boundary.

    Mirrors the hub's own ``omitempty,len=2,lowercase`` on this field, so a
    code the daemon accepts cannot be one the hub refuses downstream.
    """
    with pytest.raises(ValidationError) as exc_info:
        _speech_request(language_code=bad)
    assert "two-letter ISO 639-1" in str(exc_info.value)
    with pytest.raises(ValidationError):
        _agent_speech_request(language_code=bad)


@pytest.mark.parametrize("good", ["en", "es", "zh", "ar"])
def test_language_codes_accept_iso639_1(good):
    assert _speech_request(language_code=good).language_code == good
    assert _agent_speech_request(language_code=good).language_code == good
