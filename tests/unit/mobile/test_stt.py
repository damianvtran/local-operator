"""The STT backend table and the phone-facing availability answer.

Three layers under test:

* ``clients/stt.py`` — the table is the resolver's vocabulary (exact tokens,
  cascade order), the rows that cannot execute say so HONESTLY (SuperWhisper's
  exclusion is a research finding, not a TODO), the BYO executor probe is
  fail-closed, and the Radient adapter keeps the desktop route's temp-dir
  discipline.
* ``mobile/stt.py`` — availability: the resolver's answer is PRESERVED
  (its path, not our priority logic) and only filtered for executability;
  the persisted-credential rule keys on stored credentials; the TTL cache
  serves repaints; the whole surface never raises; the pre-cascade tree
  falls back to the first executable row.
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
    """The resolver's answer shape the adapter reads (see mobile/stt.py)."""

    def __init__(self, available: bool, path: Optional[str], reason: str = "") -> None:
        self.available = available
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
    # tree; on this tree it is not, and both answers agree.
    assert byo_executor() is None
    assert backend_ready("provider_stt_elevenlabs") is False

    monkeypatch.setattr(stt_backends, "byo_executor", lambda: object())
    assert backend_ready("provider_stt_elevenlabs") is True
    assert backend_ready("provider_stt_openai") is True


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
async def test_unknown_and_unservable_tokens_raise_unavailable() -> None:
    with pytest.raises(SttBackendUnavailable, match="Unknown voice path"):
        await transcribe_with_backend("not-a-token", b"xx", "audio/wav")

    with pytest.raises(SttBackendUnavailable) as caught:
        await transcribe_with_backend("provider_stt_superwhisper", b"xx", "audio/wav")
    assert caught.value.reason == "SuperWhisper has no transcription API."

    with pytest.raises(SttBackendUnavailable) as caught:
        await transcribe_with_backend("provider_stt_elevenlabs", b"xx", "audio/wav")
    assert "not available in this build yet" in str(caught.value)


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
# Availability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_availability_uses_the_resolvers_path_and_filters_executability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def resolver(**kwargs: Any) -> _Resolution:
        return _Resolution(True, "provider_stt_superwhisper")

    answer = await stt_availability(resolver=resolver, config_root=tmp_path)
    assert answer["available"] is False
    assert answer["reason"] == "SuperWhisper has no transcription API."

    async def resolver2(**kwargs: Any) -> _Resolution:
        return _Resolution(True, "provider_stt_elevenlabs")

    # refresh=True: the previous answer is inside its TTL and a repaint would
    # legitimately be served from it (that is the cache test's subject, below).
    answer = await stt_availability(resolver=resolver2, config_root=tmp_path, refresh=True)
    assert answer["available"] is False
    assert "not available in this build yet" in answer["reason"]


@pytest.mark.asyncio
async def test_availability_requires_a_persisted_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ambient env key must not authorize a tunnel-reachable surface."""

    async def resolver(**kwargs: Any) -> _Resolution:
        return _Resolution(True, "provider_stt_radient")

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
        return _Resolution(False, None, "nothing here")

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

    # And a resolver answering a shape this build does not know is a FAILED
    # resolver (fail-closed), not an accidental yes.
    class _Unshaped:
        path = "provider_stt_radient"

    async def shape_shift(**kwargs: Any) -> Any:
        return _Unshaped()

    answer = await stt_availability(resolver=shape_shift, config_root=tmp_path, refresh=True)
    assert answer["available"] is False


@pytest.mark.asyncio
async def test_absent_resolver_falls_back_to_the_first_executable_row(
    tmp_path: Path,
) -> None:
    """The pre-cascade bridge: the mic must work through the Radient leg.

    On this tree ``local_operator.stt`` does not exist yet, so the probe is
    absent and the baseline runs. The credential rule still applies.
    """
    answer = await stt_availability(config_root=tmp_path, refresh=True)
    assert answer["available"] is False

    store_provider_key("RADIENT_API_KEY", "stored-key", base=tmp_path)
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
