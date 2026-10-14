"""Media extraction contracts, including helper-returned speech refusals.

Expected English bytes were derived from the frozen base 23613202e3, not from
this catalogue. Full envelope equality protects both the human sentence's tail
and the machine parameters; a prefix or either/or assertion would not detect
losing a propagated error during extraction. The shared STT classifier remains
outside this slice; only its callers' pass-through wrappers gain an envelope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.clients._http import APIError
from local_operator.i18n import catalogues
from local_operator.i18n.keys import wire_errors_media as w
from local_operator.i18n.messages import Msg
from local_operator.server.routes.speech import _speech_refusal
from local_operator.tts import VoicePath

_GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "wire_errors_media_en_golden.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("sample", _GOLDEN["samples"], ids=lambda row: row["key"])
def test_entire_english_envelope_matches_pre_extraction_bytes(sample: dict[str, Any]) -> None:
    assert Msg(sample["key"], sample["params"]).envelope() == {
        "code": sample["key"],
        "params": sample["params"],
        "text": sample["text"],
    }


def test_catalogue_and_generated_key_census_is_exact() -> None:
    expected = {sample["key"] for sample in _GOLDEN["samples"]}
    assert set(catalogues.load_catalogue("en", "wire.errors.media")) == expected
    for sample in _GOLDEN["samples"]:
        factory = getattr(w, sample["key"].removeprefix("wire.errors.media."))
        assert factory(**sample["params"]) == Msg(sample["key"], sample["params"])


def test_existing_machine_code_and_dynamic_prose_are_preserved() -> None:
    error = "provider's {raw} text — café\nsecond line"
    assert w.media_refusal(error=error).envelope(code="stt_unavailable") == {
        "code": "stt_unavailable",
        "params": {"error": error},
        "text": error,
    }
    assert w.media_refusal(error=error).envelope()["code"] == "wire.errors.media.media_refusal"


@pytest.mark.parametrize(
    "status,rung,body,key,params",
    [
        (401, None, "secret upstream diagnostic", "speech_radient_sign_in_failed", {}),
        (402, VoicePath.PROVIDER_TTS_RADIENT, "diagnostic", "speech_radient_credit_low", {}),
        (429, None, "diagnostic", "speech_unavailable", {}),
        (503, None, "diagnostic", "speech_temporarily_unavailable", {}),
        (
            401,
            VoicePath.PROVIDER_TTS_ELEVENLABS,
            "key refused",
            "speech_vendor_key_refused",
            {"vendor": "ElevenLabs"},
        ),
        (
            402,
            VoicePath.PROVIDER_TTS_ELEVENLABS,
            "markerless payment refusal",
            "speech_vendor_credit_low",
            {"vendor": "ElevenLabs"},
        ),
        (
            401,
            VoicePath.PROVIDER_TTS_ELEVENLABS,
            '{"status":"quota_exceeded"}',
            "speech_vendor_credit_low",
            {"vendor": "ElevenLabs"},
        ),
        (
            429,
            VoicePath.PROVIDER_TTS_OPENAI,
            '{"code":"credit_balance_exhausted"}',
            "speech_vendor_credit_low",
            {"vendor": "OpenAI"},
        ),
        (429, VoicePath.PROVIDER_TTS_OPENAI, "rate limited", "speech_unavailable", {}),
        (503, VoicePath.PROVIDER_TTS_OPENAI, "fault", "speech_temporarily_unavailable", {}),
        (
            200,
            None,
            '{"error":"quota tail"}',
            "speech_upstream_reported_error",
            {"status": "200", "error": "quota tail"},
        ),
        (200, None, None, "speech_upstream_reported_error", {"status": "200", "error": "failure"}),
        (
            500,
            None,
            "upstream tail",
            "speech_upstream_responded",
            {"status": "500", "error": "upstream tail"},
        ),
        (500, None, None, "speech_upstream_responded", {"status": "500", "error": "failure"}),
        (None, None, "transport tail", "speech_upstream_no_response", {"error": "transport tail"}),
        (None, None, None, "speech_upstream_no_response", {"error": "failure"}),
    ],
)
def test_speech_helper_refusals_keep_status_code_params_and_full_text(
    status: int | None,
    rung: VoicePath | None,
    body: str | None,
    key: str,
    params: dict[str, str],
) -> None:
    refusal = _speech_refusal(APIError("failure", status_code=status, body=body), rung=rung)
    assert refusal.status_code == (
        402
        if key == "speech_vendor_credit_low"
        else status if status in (401, 402, 429, 503) else 502
    )
    # The golden independently pins every template's full sentence above; here
    # the helper must choose the right template and retain every runtime param.
    assert refusal.detail == Msg("wire.errors.media." + key, params).envelope()


def test_three_route_paths_are_active_and_reject_new_prose(i18n_check) -> None:
    root = Path(__file__).resolve().parents[3]
    baseline = i18n_check.load_baseline(root / "i18n" / "baseline.json")
    allowlist = i18n_check.load_allowlist(root / "i18n" / "allowlist.toml")
    for name in ("speech", "stt", "transcription"):
        path = f"local_operator/server/routes/{name}.py"
        assert path in baseline["enforced"]
        assert not i18n_check.in_enforced_scope(path + ".neighbour", baseline["enforced"])
        assert i18n_check.allowlisted(path, allowlist) is None
        source = (root / path).read_text(encoding="utf-8")
        count = len(i18n_check.scan_source(path, source))
        assert count == baseline["files"].get(path, 0)
        injected = source + '\nraise HTTPException(400, "Media regression sentence.")\n'
        new_count = len(i18n_check.scan_source(path, injected))
        assert new_count == count + 1
        finding = i18n_check.findings({path: new_count}, baseline["files"])
        assert len(finding) == 1
        assert i18n_check.in_enforced_scope(finding[0].path, baseline["enforced"])
