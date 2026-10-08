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


def test_every_shipped_login_row_has_a_picker_description_that_fits() -> None:
    """The picker's short form is required, and budgeted (design round 1, D3).

    The long descriptions ellipsized on most picker rows — the name column, the
    description column and the state column share one terminal line — so the
    picker reads PICKER_DESCRIPTIONS and `lop login` reads DESCRIPTIONS. A row
    with no short form falls back to the long one, which is a silent return of
    the defect, so the gap is a test failure rather than a fuzzy row.
    """
    missing = [d.id for d in list_login_providers() if d.id not in catalog.PICKER_DESCRIPTIONS]
    assert missing == []
    over = {
        row_id: len(text)
        for row_id, text in catalog.PICKER_DESCRIPTIONS.items()
        if len(text) > catalog.PICKER_DESCRIPTION_BUDGET
    }
    assert over == {}, over
    # The recommended tag is one word, spelled once, and it is what the picker
    # and the CLI both say (D10).
    assert catalog.RECOMMENDED_TAG == "recommended"
    # And the recommended row's PAINTED string fits too: the picker appends the
    # tag, so the budget applies to short form + suffix. Without this the first
    # row of the setup picker ellipsized its own recommendation while every
    # short form passed the check above.
    painted = f"{catalog.picker_description(catalog.RECOMMENDED_LOGIN)} — {catalog.RECOMMENDED_TAG}"
    assert len(painted) <= catalog.PICKER_DESCRIPTION_BUDGET, (len(painted), painted)


def test_a_row_with_no_short_form_falls_back_to_the_long_one() -> None:
    """Embedders' own providers have no catalogue entry; they must not go blank."""
    assert catalog.picker_description("not-a-real-provider") == ""
    assert catalog.picker_description("openai") == catalog.PICKER_DESCRIPTIONS["openai"]


def test_every_shipped_login_row_has_a_short_unique_picker_label() -> None:
    """D3's second half: the NAME column is short, and two rows never collide.

    Painting the full registry label moved the ellipsis onto the label itself,
    because the name and description columns share one picker line. The short
    form is what makes both fit — and uniqueness is what keeps the twin rows
    (`OpenAI API key` against `OpenAI speech`) tellable apart at a glance.
    """
    shipped = [d.id for d in list_login_providers()]
    missing = [row_id for row_id in shipped if row_id not in catalog.PICKER_LABELS]
    assert missing == []
    over = {
        row_id: len(text)
        for row_id, text in catalog.PICKER_LABELS.items()
        if len(text) > catalog.PICKER_LABEL_BUDGET
    }
    assert over == {}, over
    labels = [catalog.PICKER_LABELS[row_id] for row_id in shipped]
    assert len(set(labels)) == len(labels), [x for x in labels if labels.count(x) > 1]
    # An embedder's provider has no entry and keeps its own name.
    assert catalog.picker_label("not-a-real-provider", "My Provider") == "My Provider"
