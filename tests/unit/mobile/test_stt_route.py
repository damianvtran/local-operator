"""``POST /api/transcribe``: the gate, the bounds, the dispatch, the copy.

The route's contract in one place: auth + cross-origin first (the shared
``gate()``), then the upload bounds (413 for size, 422 for shape), then the
dispatch — 200 with the token that ACTUALLY ran, 503 when nothing can, and the
mirrored upstream sentences for 402/502. ``capabilities`` on the list payload
is asserted here too, because the web's mic visibility reads it off the LIST
THE SAME BYTES it reads sessions from.
"""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.clients._http import APIError
from local_operator.clients.stt import SttBackendUnavailable, SttOutcome
from local_operator.mobile import daemon as daemon_mod
from local_operator.mobile import stt as mobile_stt
from local_operator.mobile.daemon import MobileDaemon, build_app


@pytest.fixture(autouse=True)
def _fresh_availability_cache() -> Any:
    mobile_stt._reset_availability_cache()
    yield
    mobile_stt._reset_availability_cache()


def _client(password: str = "pw123") -> TestClient:
    return TestClient(build_app(MobileDaemon(port=0, password=password)), follow_redirects=False)


def _logged_in() -> TestClient:
    client = _client()
    client.post("/login", data={"password": "pw123"})
    return client


def _audio(blob: bytes = b"not-really-audio", mime: str = "audio/wav") -> dict[str, Any]:
    return {"files": {"audio": ("clip.wav", blob, mime)}}


def test_transcribe_requires_auth_and_refuses_cross_origin() -> None:
    client = _client()
    unauthorized = client.post("/api/transcribe", **_audio())
    assert unauthorized.status_code == 401

    client.post("/login", data={"password": "pw123"})
    cross_origin = client.post(
        "/api/transcribe",
        headers={"origin": "https://evil.example"},
        **_audio(),
    )
    assert cross_origin.status_code == 403


def test_a_missing_or_empty_part_is_422() -> None:
    client = _logged_in()
    missing = client.post("/api/transcribe", data={"language": "en"})
    assert missing.status_code == 422
    assert missing.json() == {"error": "audio file is required"}

    empty = client.post("/api/transcribe", files={"audio": ("clip.wav", b"", "audio/wav")})
    assert empty.status_code == 422
    assert empty.json() == {"error": "audio file is empty"}


def test_an_unsupported_media_type_is_422() -> None:
    client = _logged_in()
    denied = client.post(
        "/api/transcribe",
        files={"audio": ("clip.bin", b"some-bytes", "application/octet-stream")},
    )
    assert denied.status_code == 422
    assert "Unsupported audio format" in denied.json()["error"]


def test_codec_parameters_on_the_recorder_mime_are_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chrome posts ``blob.type`` verbatim: ``audio/webm;codecs=opus``."""
    client = _logged_in()
    seen: dict[str, Any] = {}

    async def fake_transcribe(audio: bytes, mime: str, **kwargs: Any) -> SttOutcome:
        seen.update(mime=mime)
        return SttOutcome(text="ok", provider="radient", path="provider_stt_radient")

    async def fake_availability(**kwargs: Any) -> dict[str, Any]:
        return {"available": True, "path": "provider_stt_radient", "reason": ""}

    # The route imports these from ``local_operator.mobile.stt`` INSIDE the
    # request, so patching the module attribute is the seam (and monkeypatch
    # undoes it, unlike a reload).
    monkeypatch.setattr(mobile_stt, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(mobile_stt, "stt_availability", fake_availability)
    response = client.post(
        "/api/transcribe",
        files={"audio": ("clip.webm", b"webm-bytes", "audio/webm;codecs=opus")},
    )

    assert response.status_code == 200
    assert seen["mime"] == "audio/webm"


def test_oversize_upload_is_413_by_the_byte_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon_mod, "STT_MAX_UPLOAD_BYTES", 32)
    client = _logged_in()
    response = client.post("/api/transcribe", **_audio(blob=b"x" * 64))
    assert response.status_code == 413
    assert "too large" in response.json()["error"]


def test_content_length_refuses_oversize_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The declared header is a fast refuse: an oversized body never parses.

    The body below is deliberately INVALID multipart — if the parse ran, this
    would be a 422; a 413 proves the header check fired first.
    """
    monkeypatch.setattr(daemon_mod, "STT_MAX_UPLOAD_BYTES", 32)
    client = _logged_in()
    body = b"x" * (32 + (1 << 20) + 16)
    response = client.post(
        "/api/transcribe",
        content=body,
        headers={"content-type": "multipart/form-data; boundary=never-closed"},
    )
    assert response.status_code == 413


def test_no_executable_path_is_503_with_the_stable_code() -> None:
    client = _logged_in()
    response = client.post("/api/transcribe", **_audio())
    # The isolated fixture has no stored credentials and no resolver module, so
    # availability is honestly unavailable and the dispatch refuses.
    assert response.status_code == 503
    assert response.json() == {
        "error": "Voice input isn't available on this machine.",
        "code": "stt_unavailable",
    }


def _stub_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outcome: Any | None = None,
    error: Exception | None = None,
) -> list[dict[str, Any]]:
    """Patch the dispatch seam (and availability) for one route call."""
    calls: list[dict[str, Any]] = []

    async def fake_availability(**kwargs: Any) -> dict[str, Any]:
        return {"available": True, "path": "provider_stt_radient", "reason": ""}

    async def fake_transcribe(audio: bytes, mime: str, **kwargs: Any) -> Any:
        calls.append({"audio": audio, "mime": mime, **kwargs})
        if error is not None:
            raise error
        return outcome

    monkeypatch.setattr(mobile_stt, "stt_availability", fake_availability)
    monkeypatch.setattr(mobile_stt, "transcribe_audio", fake_transcribe)
    return calls


def test_success_answers_with_the_path_that_actually_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_dispatch(
        monkeypatch,
        outcome=SttOutcome(
            text="hello world",
            provider="openai",
            model="gpt-transcribe",
            path="provider_stt_radient",
        ),
    )
    client = _logged_in()
    response = client.post("/api/transcribe", **_audio())
    assert response.status_code == 200
    assert response.json() == {
        "text": "hello world",
        "provider": "openai",
        "model": "gpt-transcribe",
        # The token that RAN — never re-derived from the availability answer.
        "path": "provider_stt_radient",
    }
    assert calls[0]["audio"] == b"not-really-audio"


def test_a_backend_refusal_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_dispatch(monkeypatch, error=SttBackendUnavailable("gone", path="provider_stt_radient"))
    client = _logged_in()
    response = client.post("/api/transcribe", **_audio())
    assert response.status_code == 503
    assert response.json()["code"] == "stt_unavailable"


def test_a_quota_refusal_is_402_and_an_upstream_failure_is_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_dispatch(
        monkeypatch,
        error=APIError("boom", status_code=402, body='{"detail":"insufficient credits"}'),
    )
    client = _logged_in()
    response = client.post("/api/transcribe", **_audio())
    assert response.status_code == 402
    assert response.json()["code"] == "stt_quota"
    assert "credit balance" in response.json()["error"]

    _stub_dispatch(monkeypatch, error=APIError("Connection refused", status_code=None))
    response = client.post("/api/transcribe", **_audio())
    assert response.status_code == 502
    assert response.json() == {"error": "Connection refused", "code": "stt_upstream"}


def test_a_plain_runtime_error_is_500(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_dispatch(monkeypatch, error=RuntimeError("RADIENT_API_KEY is not configured."))
    client = _logged_in()
    response = client.post("/api/transcribe", **_audio())
    assert response.status_code == 500
    assert response.json()["error"] == "RADIENT_API_KEY is not configured."


def test_the_list_payload_carries_the_capabilities_block() -> None:
    client = _logged_in()
    payload = client.get("/api/sessions").json()
    capabilities = payload.get("capabilities")
    assert isinstance(capabilities, dict)
    # features is cf9f's reader; on a pre-carriage tree it is an empty dict —
    # never a mobile-side literal (QA compares against the symbol, not a number).
    assert isinstance(capabilities.get("features"), dict)
    stt = capabilities.get("stt")
    assert isinstance(stt, dict)
    assert stt.get("available") is False, "the isolated fixture stores no credentials"
    assert isinstance(stt.get("reason"), str)


# ---------------------------------------------------------------------------
# The relay strip-gate (the daemon's half of the carriage)
# ---------------------------------------------------------------------------


class _CapabilityRecord:
    def __init__(self, capabilities: list[str] | None = None) -> None:
        self.capabilities = list(capabilities or [])


def test_the_relay_strips_the_annotation_from_an_uncapable_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fields ride only to an owner that advertised input-mode-v1."""
    token = daemon_mod.INPUT_MODE_CAPABILITY or "input-mode-v1"
    monkeypatch.setattr(daemon_mod, "INPUT_MODE_CAPABILITY", token)

    frame = {
        "op": "prompt",
        "text": "hi",
        "input_mode": "dictated",
        "input_path": "provider_stt_radient",
    }
    daemon_mod._strip_unsupported_annotation(_CapabilityRecord(), frame)
    assert "input_mode" not in frame
    assert "input_path" not in frame

    capable = _CapabilityRecord([token])
    frame = {
        "op": "prompt",
        "text": "hi",
        "input_mode": "dictated",
        "input_path": "provider_stt_radient",
    }
    daemon_mod._strip_unsupported_annotation(capable, frame)
    assert frame["input_mode"] == "dictated"
    assert frame["input_path"] == "provider_stt_radient"


def test_the_strip_gate_fails_closed_without_the_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tree that predates the capability cannot honour the annotation."""
    monkeypatch.setattr(daemon_mod, "INPUT_MODE_CAPABILITY", "")
    frame = {"op": "steer", "text": "hi", "input_mode": "mixed"}
    daemon_mod._strip_unsupported_annotation(_CapabilityRecord(["input-mode-v1"]), frame)
    assert "input_mode" not in frame
