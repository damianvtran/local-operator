"""The failure mapping: legacy Radient sentences byte-for-byte, plus the direct rungs.

The legacy expectations here are the strings the pre-move implementation
produced; the extraction's byte-compatibility was additionally verified
mechanically over a 396-case matrix (statuses × bodies × providers) against
``git show HEAD:local_operator/server/routes/transcription.py`` — zero
mismatches. These pins keep it that way per sentence.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import HTTPException

from local_operator.clients._http import APIError
from local_operator.redaction_shapes import REDACTION_MARKER
from local_operator.stt import AudioPath, errors


def _mapped(status_code, body, provider="upstream", rung=None):
    exc = APIError("upstream boom", status_code=status_code, body=body)
    return errors.failure_status_and_detail(exc, provider, rung=rung)


# -- the legacy (Radient) vocabulary, byte-for-byte -------------------------


def test_radient_402_keeps_its_sentence() -> None:
    assert _mapped(402, None) == (
        402,
        "Transcription is unavailable: your Radient credit balance is too low. "
        "Add credits to continue. Upstream responded 402 with no body.",
    )


def test_provider_quota_markers_still_map_to_402() -> None:
    assert _mapped(500, '{"error":"insufficient_quota"}', provider="openai") == (
        402,
        "Transcription is unavailable: the openai provider has run out of credits. "
        "Switch to another provider, or add credits to your Radient account. "
        'Upstream responded 500: {"error":"insufficient_quota"}',
    )


def test_the_s2_vendor_quota_codes_map_to_402_too() -> None:
    """ElevenLabs' ``quota_exceeded`` (on a 401) and OpenAI's
    ``credit_balance_exhausted`` (on a 429) name the vendor machine codes added
    for the speech direction; both map to the provider-credit 402."""
    assert _mapped(
        401,
        '{"detail":{"status":"quota_exceeded","message":"You have insufficient '
        'quota to complete the request."}}',
        provider="elevenlabs",
    ) == (
        402,
        "Transcription is unavailable: the elevenlabs provider has run out of "
        "credits. Switch to another provider, or add credits to your Radient "
        'account. Upstream responded 401: {"detail":{"status":"quota_exceeded",'
        '"message":"You have insufficient quota to complete the request."}}',
    )
    assert _mapped(
        429,
        '{"error":{"code":"credit_balance_exhausted"}}',
        provider="openai",
    ) == (
        402,
        "Transcription is unavailable: the openai provider has run out of "
        "credits. Switch to another provider, or add credits to your Radient "
        'account. Upstream responded 429: {"error":{"code":"credit_balance_exhausted"}}',
    )


def test_radient_edge_401_keeps_its_sentence() -> None:
    assert _mapped(401, '{"detail":"Invalid or expired token"}') == (
        502,
        "Transcription is unavailable: Radient refused this app's credentials. "
        "Sign in again in the app; if it keeps failing, the daemon's Radient API "
        "key is invalid or has expired. Upstream responded 401: "
        '{"detail":"Invalid or expired token"}',
    )


def test_provider_rejection_with_error_envelope_keeps_its_sentence() -> None:
    assert _mapped(422, '{"error":"bad param"}', provider="openai") == (
        502,
        "The openai provider rejected the transcription request. "
        'Upstream responded 422: {"error":"bad param"}',
    )


def test_four_xx_without_a_body_names_both_candidates() -> None:
    assert _mapped(400, None, provider="openai") == (
        502,
        "The transcription request was refused upstream (HTTP 400) and the upstream "
        "sent no body saying by whom. The daemon only authenticates to Radient, so "
        "this is either Radient's edge or the openai provider it called. "
        "Upstream responded 400 with no body.",
    )


def test_transport_failure_passes_the_client_text_through() -> None:
    exc = APIError("Connection refused", status_code=None)
    assert errors.failure_status_and_detail(exc, "upstream") == (502, "Connection refused")


def test_two_hundred_with_an_error_body_keeps_its_wording() -> None:
    assert _mapped(200, '{"error":"provider 500"}') == (
        502,
        "Transcription failed upstream. Radient reported an error "
        '(HTTP 200): {"error":"provider 500"}',
    )


def test_legacy_route_raises_the_shared_function() -> None:
    """The route module and the shared mapper are the same object, not a copy."""
    from local_operator.server.routes import transcription

    assert transcription._classify_upstream_failure is errors.classify_upstream_failure


def test_classify_wraps_the_same_status_and_detail() -> None:
    exc = APIError("upstream boom", status_code=402, body="out of credits")
    result = errors.classify_upstream_failure(exc, "upstream")
    assert isinstance(result, HTTPException)
    assert (result.status_code, result.detail) == errors.failure_status_and_detail(exc, "upstream")


# -- the direct BYO rungs ----------------------------------------------------


def test_direct_401_names_elevenlabs_and_the_key_remedy() -> None:
    assert _mapped(401, '{"detail":"x"}', rung=AudioPath.PROVIDER_STT_ELEVENLABS) == (
        502,
        "Transcription is unavailable: ElevenLabs refused the API key stored for it "
        "(HTTP 401). Check the key for this provider, or replace it and try again. "
        'Upstream responded 401: {"detail":"x"}',
    )


def test_direct_402_names_the_provider_and_the_top_up_remedy() -> None:
    assert _mapped(402, None, rung=AudioPath.PROVIDER_STT_OPENAI) == (
        402,
        "Transcription is unavailable: your OpenAI credit balance is too low. "
        "Add credits to continue. Upstream responded 402 with no body.",
    )


def test_direct_quota_marker_maps_to_402_without_radient_wording() -> None:
    status, detail = _mapped(
        500,
        '{"error":"You exceeded your current quota"}',
        rung=AudioPath.PROVIDER_STT_OPENAI,
    )
    assert status == 402
    assert detail.startswith(
        "Transcription is unavailable: your OpenAI account has run out of credits."
    )
    assert "Radient account" not in detail


def test_direct_403_says_rekeying_is_not_the_fix() -> None:
    status, detail = _mapped(403, None, rung=AudioPath.PROVIDER_STT_ELEVENLABS)
    assert status == 502
    assert detail.startswith("Transcription is unavailable: ElevenLabs refused the request (403).")
    assert "re-entering the key is not the fix" in detail


def test_direct_404_points_at_the_endpoint() -> None:
    status, detail = _mapped(404, '{"detail":"x"}', rung=AudioPath.PROVIDER_STT_OPENAI)
    assert status == 502
    assert detail.startswith(
        "Transcription is unavailable: the OpenAI transcription endpoint was not found."
    )


def test_the_radient_rung_still_gets_the_legacy_vocabulary() -> None:
    """A rung that IS Radient must not be treated as a direct BYO provider."""
    status, detail = _mapped(401, '{"detail":"x"}', rung=AudioPath.PROVIDER_STT_RADIENT)
    assert (status, detail) == _mapped(401, '{"detail":"x"}')


# -- the httpx → APIError adapter --------------------------------------------


def test_httpx_errors_keep_status_code_and_scrub_the_key() -> None:
    key = "sk-" + "a" * 40
    response = httpx.Response(401, json={"error": f"bad key {key}"})
    exc = errors.api_error_from_httpx_response(
        response, fallback_message="OpenAI refused the transcription request", secrets=[key]
    )
    assert exc.status_code == 401
    assert key not in str(exc)
    assert REDACTION_MARKER in str(exc)


def test_httpx_error_without_a_designed_message_uses_the_fallback() -> None:
    response = httpx.Response(500, text="<html>oops</html>")
    exc = errors.api_error_from_httpx_response(
        response, fallback_message="OpenAI refused the transcription request"
    )
    assert exc.status_code == 500
    assert str(exc) == "OpenAI refused the transcription request (HTTP 500)"


@pytest.mark.parametrize("rung", [None, AudioPath.PROVIDER_STT_RADIENT])
def test_radient_vocabulary_is_unaffected_by_the_new_parameter(rung) -> None:
    exc = APIError("x", status_code=503, body="service down")
    assert errors.failure_status_and_detail(exc, "upstream", rung=rung) == (
        502,
        "Transcription failed upstream. Upstream responded 503: service down",
    )
