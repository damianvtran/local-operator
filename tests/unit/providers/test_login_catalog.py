"""``providers.login_catalog``: the one presentation of the login choices.

Pins the first-run findings it closes (audit D3/U2/U9/Q3/D13/Q1/Q8): Radient
first and recommended, the desktop's group headings, a description for EVERY
shipped row, the speech/decision-only rows kept out of the setup picker, and
the chat API-key logins present.
"""

from __future__ import annotations

from local_operator.providers import login_catalog as catalog
from local_operator.providers.registry import (
    get_provider_definition,
    list_login_providers,
)


def test_radient_leads_and_is_the_one_recommendation() -> None:
    rows = catalog.ordered_rows()
    assert rows[0].id == catalog.RECOMMENDED_LOGIN == "radient"
    assert [row.id for row in rows if row.recommended] == ["radient"]
    assert catalog.RECOMMENDED_LOGIN_COMMAND == "/login radient"


def test_the_groups_are_the_desktops_in_the_desktops_order() -> None:
    headings = [heading for heading, _ in catalog.login_groups()]
    assert headings == [
        "Recommended",
        "Use a subscription",
        "Use an API key",
        "Run models on this computer",
        "Not for chat",
    ]


def test_every_shipped_login_row_has_its_own_description() -> None:
    """A new provider cannot land wordless: the fallback is for embedders."""
    missing = [d.id for d in list_login_providers() if d.id not in catalog.DESCRIPTIONS]
    assert missing == []


def test_the_setup_picker_leaves_out_what_cannot_chat() -> None:
    setup_ids = {row.id for row in catalog.ordered_rows(include_non_chat=False)}
    for speech_or_decision in ("elevenlabs", "openai-key", "typesafe"):
        assert speech_or_decision not in setup_ids
        assert catalog.is_non_chat(speech_or_decision)
    assert {"openai-api-key", "anthropic-key"} <= setup_ids


def test_the_chat_key_logins_store_under_the_chat_provider() -> None:
    for login_id, storage in (("openai-api-key", "openai"), ("anthropic-key", "anthropic")):
        definition = get_provider_definition(login_id)
        assert definition is not None and definition.store_credentials_as == storage
        assert definition.paste_prompt_required is True
        assert catalog.group_of(definition) == catalog.GROUP_API_KEY
    # The speech row says what it is for.
    speech = get_provider_definition("openai-key")
    assert speech is not None and "speech" in speech.name.lower()


def test_the_remote_hint_fires_only_where_a_browser_cannot_reach(monkeypatch) -> None:
    assert catalog.headless_display({"SSH_CONNECTION": "a b c d"}, "darwin") is True
    assert catalog.headless_display({}, "linux") is True
    assert catalog.headless_display({"WAYLAND_DISPLAY": "w"}, "linux") is False
    assert catalog.headless_display({}, "darwin") is False
    monkeypatch.setattr(catalog, "headless_display", lambda *a, **k: True)
    assert "lop login openai-device" in (catalog.remote_login_hint("openai") or "")
    assert "/login radient-key" in (catalog.remote_login_hint("radient", command="/login") or "")
    assert catalog.remote_login_hint("deepseek") is None
