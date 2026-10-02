"""The STT backend table and the phone-facing availability answer.

Three layers under test:

* ``clients/stt.py`` — the table is the resolver's vocabulary (exact tokens,
  cascade order), the rows that cannot execute say so HONESTLY (SuperWhisper's
  exclusion is a research finding, not a TODO), the BYO executor probe is
  fail-closed, and the Radient adapter keeps the desktop route's temp-dir
  discipline.
* ``mobile/stt.py`` — availability: the resolver's answer is PRESERVED
  (its path, not our priority logic) and only filtered for executability —
  the SETTLED seam reads the cascade's ``AudioPathResolution`` (path-driven;
  the pre-cascade tree's first-executable-row baseline remains covered);
  the persisted-credential rule keys on stored credentials; the TTL cache
  serves repaints; the whole surface never raises.
* ``describe_stt_failure`` — the desktop route's classifier mirrored locally:
  402 for a quota refusal (Radient's own or a provider credit marker), 502
  for everything else upstream, transport failures passed through verbatim.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from local_operator.clients import stt as stt_backends
from local_operator.clients._http import APIError
from local_operator.clients.stt import (
    STT_BACKENDS,
    SttBackendUnavailable,
    SttOutcome,
    backend_ready,
    byo_executor,
    resolve_backend_key,
    transcribe_with_backend,
)
from local_operator.mobile import stt as mobile_stt
from local_operator.mobile.stt import (
    describe_stt_failure,
    path_provider_label,
    stt_availability,
    transcribe_audio,
)
from local_operator.providers.registry import store_provider_key


@pytest.fixture(autouse=True)
def _fresh_availability_cache() -> Any:
    """Every test starts from an empty cache: the TTL memo is module global."""
    mobile_stt._reset_availability_cache()
    yield
    mobile_stt._reset_availability_cache()


class _Resolution:
    """A resolver answer in the SETTLED shape (``AudioPathResolution``'s side
    of the seam): ``path`` — an ``AudioPath`` token, or the ``"none"`` no-path
    spelling — plus the reason. There is deliberately no ``available`` field:
    the reader DERIVES availability from the path, and a double that carried
    one would let a regression re-read it unnoticed.
    """

    def __init__(self, path: Optional[str], reason: str = "") -> None:
        self.path = path
        self.reason = reason


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_the_table_is_the_frozen_resolver_vocabulary_in_cascade_order() -> None:
    assert list(STT_BACKENDS) == [
        "provider_stt_radient",
        "provider_stt_elevenlabs",
        "provider_stt_openai",
        "provider_stt_superwhisper",
        "model_audio_sidecar",
    ]


def test_the_unexecutable_rows_carry_a_reason_not_a_hope() -> None:
    superwhisper = STT_BACKENDS["provider_stt_superwhisper"]
    assert superwhisper.servable is False
    # The honest sentence: research found NO audio-in/transcript-out API
    # (2026-09-28), so this row must never become "unknown path".
    assert superwhisper.reason == "SuperWhisper has no transcription API."

    sidecar = STT_BACKENDS["model_audio_sidecar"]
    assert sidecar.servable is False
    assert sidecar.reason, "a pending row still owes the user a sentence"


def test_resolve_backend_key_accepts_tokens_providers_and_enum_members() -> None:
    assert resolve_backend_key("provider_stt_radient") == "provider_stt_radient"
    assert resolve_backend_key("elevenlabs") == "provider_stt_elevenlabs"
    assert resolve_backend_key("  ElevenLabs ") == "provider_stt_elevenlabs"

    class _Member:
        value = "provider_stt_openai"

    assert resolve_backend_key(_Member()) == "provider_stt_openai"
    assert resolve_backend_key("nope") is None
    assert resolve_backend_key(None) is None


def test_backend_ready_tracks_the_executor_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    assert backend_ready("provider_stt_radient") is True
    assert backend_ready("provider_stt_superwhisper") is False

    # The BYO rungs are executable exactly when the cascade executor is in the
    # tree — and it IS here (the cascade shipped): the callable check and both
    # backend_ready answers are the same fact, read from the one probe.
    assert callable(byo_executor())
    assert backend_ready("provider_stt_elevenlabs") is True
    assert backend_ready("provider_stt_openai") is True

    # ...and the probe stays the gate: an absent executor flips them off again
    # (the fail-closed half the sibling test drives through the real probe).
    monkeypatch.setattr(stt_backends, "byo_executor", lambda: None)
    assert backend_ready("provider_stt_elevenlabs") is False
    assert backend_ready("provider_stt_openai") is False


def test_the_executor_probe_fails_closed_on_any_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stt_backends, "BYO_EXECUTOR_MODULE", "definitely.not.a.module")
    assert byo_executor() is None

    # A module that EXISTS but does not carry the symbol is absent too — the
    # probe answers "not executable", never half a rung.
    monkeypatch.setattr(stt_backends, "BYO_EXECUTOR_MODULE", "local_operator.mobile.stt")
    monkeypatch.setattr(stt_backends, "BYO_EXECUTOR_ATTR", "no_such_symbol")
    assert byo_executor() is None


@pytest.mark.asyncio
async def test_unknown_and_unservable_tokens_raise_unavailable(tmp_path: Path) -> None:
    with pytest.raises(SttBackendUnavailable, match="Unknown voice path"):
        await transcribe_with_backend("not-a-token", b"xx", "audio/wav")

    with pytest.raises(SttBackendUnavailable) as caught:
        await transcribe_with_backend("provider_stt_superwhisper", b"xx", "audio/wav")
    assert caught.value.reason == "SuperWhisper has no transcription API."

    # The cascade now DISPATCHES: with no stored key the token-targeted
    # executor refuses the rung, and the bridge re-raises that as the same
    # typed class the route answers 503 to — never the generic 500 a bare
    # cascade SttUnavailable would have produced (agents review convergence,
    # B1/Q7).
    with pytest.raises(SttBackendUnavailable) as caught:
        await transcribe_with_backend(
            "provider_stt_elevenlabs", b"xx", "audio/wav", config_root=tmp_path
        )
    assert "No ElevenLabs API key is stored" in str(caught.value)
    assert caught.value.path == "provider_stt_elevenlabs"


# ---------------------------------------------------------------------------
# The Radient adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_radient_adapter_sends_bytes_off_the_loop_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.clients import radient as radient_module
    from local_operator.providers import radient_credentials

    seen: dict[str, Any] = {}

    class _FakeRadientClient:
        def __init__(self, api_key: Any, base_url: str) -> None:
            seen["base_url"] = base_url

        def create_transcription(
            self, *, file_path: str, model: Any, prompt: Any, language: Any
        ) -> Any:
            seen["file_path"] = file_path
            seen["bytes"] = Path(file_path).read_bytes()
            seen["exists_during_call"] = Path(file_path).exists()
            return SimpleNamespace(text="hello world", provider="elevenlabs")

    async def _key(config: Any, base_url: str, *, store: Any = None) -> Any:
        seen["credential_root"] = config
        return SimpleNamespace(get_secret_value=lambda: "fixture-key")

    monkeypatch.setattr(radient_module, "RadientClient", _FakeRadientClient)
    monkeypatch.setattr(radient_credentials, "resolve_radient_credential", _key)

    outcome = await transcribe_with_backend(
        "provider_stt_radient", b"RIFFfake-bytes", "audio/wav", config_root=tmp_path
    )

    assert outcome == SttOutcome(
        text="hello world", provider="elevenlabs", model=None, path="provider_stt_radient"
    )
    assert seen["bytes"] == b"RIFFfake-bytes"
    assert seen["credential_root"] == tmp_path
    # The desktop route's temp discipline: the file exists WHILE the client
    # runs and the directory is gone after, on every path.
    assert seen["exists_during_call"] is True
    assert not Path(seen["file_path"]).exists()


@pytest.mark.asyncio
async def test_the_radient_adapter_refuses_an_empty_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator.providers import radient_credentials

    async def _no_key(config: Any, base_url: str, *, store: Any = None) -> Any:
        return SimpleNamespace(get_secret_value=lambda: "")

    monkeypatch.setattr(radient_credentials, "resolve_radient_credential", _no_key)

    with pytest.raises(SttBackendUnavailable, match="not signed in"):
        await transcribe_with_backend(
            "provider_stt_radient", b"xx", "audio/wav", config_root=tmp_path
        )


# ---------------------------------------------------------------------------
# The cascade dispatch (the settled #1734 seam)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_cascade_dispatch_passes_the_audio_and_the_token_to_the_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The settled call SHAPE: token first, then BYTES + mime; no model.

    This pins the conversion #1734 reserved to the cascade session (the old
    call passed no audio at all and the token in the ``audio_path`` slot).
    The executor's own availability re-check is its tests' subject; here a
    fake executor proves what the bridge sends and what it adopts back —
    including the provider label, which the cascade's ``SttOutcome`` does not
    carry and the dispatched row must supply.
    """
    seen: dict[str, Any] = {}

    class _CascadeOutcome:
        text = "phone text"
        path = "provider_stt_openai"

    async def fake_executor(backend: str, audio: bytes, mime: str, **kwargs: Any) -> Any:
        seen.update(backend=backend, audio=audio, mime=mime, kwargs=kwargs)
        return _CascadeOutcome()

    monkeypatch.setattr(stt_backends, "byo_executor", lambda: fake_executor)

    outcome = await transcribe_with_backend(
        "provider_stt_openai",
        b"RIFFfake",
        "audio/wav",
        language="en",
        prompt="hint",
        model="gpt-something",
        config_root=tmp_path,
    )

    assert outcome.text == "phone text"
    assert outcome.path == "provider_stt_openai"
    assert outcome.provider == "openai"
    # The phone's ``model`` form field is an STT-model hint the Radient adapter
    # honors; the cascade executor deliberately does not receive it.
    assert seen == {
        "backend": "provider_stt_openai",
        "audio": b"RIFFfake",
        "mime": "audio/wav",
        "kwargs": {
            "config_dir": tmp_path,
            "store": None,
            "language": "en",
            "prompt": "hint",
        },
    }


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_availability_uses_the_resolvers_path_and_filters_executability(
    tmp_path: Path,
) -> None:
    async def resolver(**kwargs: Any) -> _Resolution:
        return _Resolution("provider_stt_superwhisper")

    answer = await stt_availability(resolver=resolver, config_root=tmp_path)
    assert answer["available"] is False
    assert answer["reason"] == "SuperWhisper has no transcription API."

    async def resolver2(**kwargs: Any) -> _Resolution:
        return _Resolution("provider_stt_elevenlabs")

    # refresh=True: the previous answer is inside its TTL and a repaint would
    # legitimately be served from it (that is the cache test's subject, below).
    # The resolver NAMED ElevenLabs; this machine has no stored key, so
    # executability hides the mic with the credential's own sentence — not the
    # old pre-cascade "not available in this build yet".
    answer = await stt_availability(resolver=resolver2, config_root=tmp_path, refresh=True)
    assert answer["available"] is False
    assert answer["reason"] == "No ElevenLabs API key is stored on this machine."

    # With the key stored, the same resolver answer lights the mic.
    store_provider_key("ELEVENLABS_API_KEY", "stored-key", base=tmp_path)
    answer = await stt_availability(resolver=resolver2, config_root=tmp_path, refresh=True)
    assert answer["available"] is True
    assert answer["path"] == "provider_stt_elevenlabs"


@pytest.mark.asyncio
async def test_availability_requires_a_persisted_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ambient env key must not authorize a tunnel-reachable surface."""

    async def resolver(**kwargs: Any) -> _Resolution:
        return _Resolution("provider_stt_radient")

    monkeypatch.setenv("RADIENT_API_KEY", "ambient-key")
    answer = await stt_availability(resolver=resolver, config_root=tmp_path, refresh=True)
    assert answer["available"] is False

    # A PERSISTED provider row is what flips it.
    store_provider_key("RADIENT_API_KEY", "stored-key", base=tmp_path)
    answer = await stt_availability(resolver=resolver, config_root=tmp_path, refresh=True)
    assert answer["available"] is True
    assert answer["path"] == "provider_stt_radient"


@pytest.mark.asyncio
async def test_availability_serves_the_ttl_cache_and_refresh_bypasses_it(
    tmp_path: Path,
) -> None:
    calls = 0

    async def resolver(**kwargs: Any) -> _Resolution:
        nonlocal calls
        calls += 1
        # The settled no-path spelling: ``AudioPath.NONE`` as the resolver
        # emits it. It is an ANSWER (unavailable, cached, reason preserved),
        # not a resolver fault.
        return _Resolution("none", "nothing here")

    first = await stt_availability(resolver=resolver, config_root=tmp_path)
    second = await stt_availability(resolver=resolver, config_root=tmp_path)
    assert first == second
    assert calls == 1, "the repaint path must not re-ask"

    await stt_availability(resolver=resolver, config_root=tmp_path, refresh=True)
    assert calls == 2


@pytest.mark.asyncio
async def test_a_broken_resolver_degrades_to_unavailable_and_never_raises(
    tmp_path: Path,
) -> None:
    async def broken(**kwargs: Any) -> _Resolution:
        raise RuntimeError("resolver exploded")

    answer = await stt_availability(resolver=broken, config_root=tmp_path, refresh=True)
    assert answer["available"] is False
    assert answer["reason"], "an unavailable answer still owes a sentence"

    # And a resolver whose shape carries NO ``path`` at all — the
    # PRE-settlement freeze (an ``available`` boolean) — is a FAILED resolver
    # (fail-closed), not an accidental yes.
    class _OldShape:
        available = True

    async def shape_shift(**kwargs: Any) -> Any:
        return _OldShape()

    answer = await stt_availability(resolver=shape_shift, config_root=tmp_path, refresh=True)
    assert answer["available"] is False

    # A resolver that names a path THIS build cannot map is a live answer, not
    # a fault: the mic stays hidden with the mapping sentence (advertising a
    # rung nothing can run would be advertising something that appears and
    # fails).
    class _FuturePath:
        path = "provider_stt_deepgram"
        reason = "A Deepgram API key is stored."

    async def future(**kwargs: Any) -> Any:
        return _FuturePath()

    answer = await stt_availability(resolver=future, config_root=tmp_path, refresh=True)
    assert answer["available"] is False
    assert "cannot run" in answer["reason"]


@pytest.mark.asyncio
async def test_absent_resolver_falls_back_to_the_first_executable_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mic must work through the Radient leg, resolver present OR absent.

    The shipped tree carries ``local_operator.stt``: production reads the
    REAL cascade resolver, and a stored Radient credential is what lights the
    path. The transitional baseline (for a tree that predates the cascade
    module) still answers the same question through its own code path —
    monkeypatched here, because the probe now finds the module.
    """
    answer = await stt_availability(config_root=tmp_path, refresh=True)
    assert answer["available"] is False

    store_provider_key("RADIENT_API_KEY", "stored-key", base=tmp_path)
    answer = await stt_availability(config_root=tmp_path, refresh=True)
    assert answer["available"] is True
    assert answer["path"] == "provider_stt_radient"

    monkeypatch.setattr(mobile_stt, "_cascade_resolver", lambda: None)
    answer = await stt_availability(config_root=tmp_path, refresh=True)
    assert answer["available"] is True
    assert answer["path"] == "provider_stt_radient"


@pytest.mark.asyncio
async def test_dispatch_reads_availability_and_answers_503_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``transcribe_audio`` is the route's one entry point: no path, no call."""
    called = False

    async def _never(*args: Any, **kwargs: Any) -> SttOutcome:
        nonlocal called
        called = True
        raise AssertionError("must not dispatch")

    monkeypatch.setattr(mobile_stt, "transcribe_with_backend", _never)
    with pytest.raises(SttBackendUnavailable):
        await transcribe_audio(
            b"xx",
            "audio/wav",
            availability={"available": False, "path": None, "reason": "nope"},
            config_root=tmp_path,
        )
    assert called is False

    # And with an available answer the seam is reached with the resolver's path.
    seen: dict[str, Any] = {}

    async def _fake(path: str, audio: bytes, mime: str, **kwargs: Any) -> SttOutcome:
        seen.update(path=path, audio=audio, mime=mime)
        return SttOutcome(text="hi", provider="radient", path=path)

    monkeypatch.setattr(mobile_stt, "transcribe_with_backend", _fake)
    outcome = await transcribe_audio(
        b"xx",
        "audio/wav",
        availability={"available": True, "path": "provider_stt_radient", "reason": ""},
        config_root=tmp_path,
    )
    assert seen["path"] == "provider_stt_radient"
    assert outcome.path == "provider_stt_radient"


# ---------------------------------------------------------------------------
# Failure copy
# ---------------------------------------------------------------------------


def test_a_quota_refusal_maps_to_402() -> None:
    radient_credit = APIError("boom", status_code=402, body='{"detail":"insufficient credits"}')
    status, body = describe_stt_failure(radient_credit, provider="radient")
    assert status == 402
    assert body["code"] == "stt_quota"
    assert "Radient credit balance" in body["error"]

    provider_credit = APIError(
        "boom", status_code=400, body='{"error":{"message":"insufficient_quota"}}'
    )
    status, body = describe_stt_failure(provider_credit, provider="ElevenLabs")
    assert status == 402
    assert "the ElevenLabs provider has run out of credits" in body["error"]

    # The two vendor machine codes added with the desktop table (voicing S2).
    # Their condition arrives on a status that says nothing about credit, so the
    # body is the only place the classifier can read it from.
    vendor_quota = APIError(
        "boom",
        status_code=401,
        body='{"detail":{"status":"quota_exceeded","message":"You have insufficient '
        'quota to complete the request."}}',
    )
    status, body = describe_stt_failure(vendor_quota, provider="ElevenLabs")
    assert status == 402
    assert "the ElevenLabs provider has run out of credits" in body["error"]

    balance = APIError(
        "boom", status_code=429, body='{"error":{"code":"credit_balance_exhausted"}}'
    )
    status, body = describe_stt_failure(balance, provider="openai")
    assert status == 402
    assert "the openai provider has run out of credits" in body["error"]


def test_a_transport_failure_passes_the_clients_text_through() -> None:
    status, body = describe_stt_failure(APIError("Connection refused", status_code=None))
    assert status == 502
    assert body["error"] == "Connection refused"
    assert body["code"] == "stt_upstream"


def test_a_radient_edge_refusal_is_attributed_to_radient() -> None:
    exc = APIError(
        "boom",
        status_code=401,
        body='{"detail":"Invalid or expired token"}',
    )
    status, body = describe_stt_failure(exc, provider="upstream")
    assert status == 502
    assert "Radient refused this app's credentials" in body["error"]


def test_a_provider_envelope_is_attributed_to_the_provider() -> None:
    exc = APIError(
        "boom",
        status_code=400,
        body='{"error":{"message":"bad audio"}}',
    )
    status, body = describe_stt_failure(exc, provider="ElevenLabs")
    assert status == 502
    assert "The ElevenLabs provider rejected" in body["error"]


def test_path_provider_label_names_the_rung() -> None:
    assert path_provider_label("provider_stt_elevenlabs") == "ElevenLabs"
    assert path_provider_label("provider_stt_radient") == "Radient"
    assert path_provider_label(None) == "upstream"
    assert path_provider_label("nonsense") == "upstream"
