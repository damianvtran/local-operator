"""The shipped ``fr`` catalogue reaches every ``wire.providers.connect`` consumer.

WHY. The slice tests in ``tests/unit/providers/test_key_check.py`` and
``tests/unit/model/test_suggested_model_copy.py`` stub the render seam, so they
prove the copy is *keyed* but not that a French user ever *sees* French. That
second half depends on the audited catalogue files, their ledger rows, and the
generated ``shipped.json`` keeping ``fr`` eligible: a missing or stale ledger row
un-ships the locale and every verdict silently falls back to English.

So this test uses the REAL shipped catalogue tree (no synthetic locale), drives
each of the five keys through its real consumer — the key-check verdicts via
``classify``/``check_api_key``/``invalid_characters_reason`` and the router name
via ``suggested_model_for`` — and compares against the shipped fr file itself.
The negative control renders the same calls under ``en`` and requires every one
to differ, so an fr assertion that passed on English fallback cannot pass.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.i18n import resolve, text
from local_operator.model import defaults
from local_operator.providers import key_check
from local_operator.providers.registry import get_provider_definition

NS = "wire.providers.connect"
CATALOGUES = Path(__file__).resolve().parents[3] / "local_operator/i18n/catalogues"


def _shipped(locale: str) -> dict[str, str]:
    return json.loads((CATALOGUES / locale / f"{NS}.json").read_text(encoding="utf-8"))


def _fill(template: str, **params: str) -> str:
    # The namespace carries only simple {name} arguments (no ICU plural/select),
    # so plain substitution reproduces the runtime render exactly.
    for name, value in params.items():
        template = template.replace("{" + name + "}", value)
    return template


@pytest.fixture
def shipped_locale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    config_dir = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("LOP_LANG", raising=False)
    setting = settings_io.resolve_key("language")
    assert setting is not None

    def select(locale: str) -> None:
        # These consumers resolve config per call, not the UI's active locale.
        # Exercise the real setting store so English fallback cannot look French.
        settings_io._store(ConfigManager(config_dir), setting.path, locale)
        assert resolve.resolve_language() == locale
        text.set_active_locale(locale)

    yield select
    text.set_active_locale("en")
    text._sources.cache_clear()


def _render_every_key() -> dict[str, str]:
    """Each key through the code path a user actually reaches."""
    google = get_provider_definition("google")
    assert google is not None

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async def offline() -> key_check.KeyCheck:
        async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as client:
            return await key_check.check_api_key("google", "sk-synthetic", client=client)

    radient = defaults.suggested_model_for("radient")
    assert radient is not None
    return {
        f"{NS}.key_invalid_characters": key_check.invalid_characters_reason(),
        f"{NS}.key_rejected": key_check.classify(google, 401, "").reason or "",
        f"{NS}.key_unverified": key_check.classify(google, 429, "").reason or "",
        f"{NS}.key_unreachable": asyncio.run(offline()).reason or "",
        f"{NS}.suggested_automatic": radient.name,
    }


def test_shipped_fr_renders_every_provider_connect_key(shipped_locale) -> None:
    assert "fr" in resolve.shipped_locales(), "fr must stay eligible via the ledger"
    fr = _shipped("fr")
    params = {
        f"{NS}.key_rejected": {"label": "Google"},
        f"{NS}.key_unverified": {"label": "Google", "status": "429"},
        f"{NS}.key_unreachable": {"label": "Google"},
    }
    expected = {key: _fill(value, **params.get(key, {})) for key, value in fr.items()}

    shipped_locale("fr")
    rendered = _render_every_key()

    assert rendered.keys() == fr.keys(), "every shipped key is exercised"
    assert rendered == expected


def test_negative_control_english_differs_for_every_key(shipped_locale) -> None:
    """Without this, the fr test above could pass on a vacuous fallback."""
    shipped_locale("en")
    english = _render_every_key()
    shipped_locale("fr")
    french = _render_every_key()

    assert english[f"{NS}.suggested_automatic"] == "Automatic"
    same = [key for key in english if english[key] == french[key]]
    assert same == [], f"fr fell back to English for {same}"
