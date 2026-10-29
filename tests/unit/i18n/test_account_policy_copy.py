"""New account-policy namespaces honestly fall back to English, not key tokens.

Spanish now carries `wire.settings.account_policy` (the es W1 wave), so the
fallback pin is scoped to the locales that still do NOT carry it and a second
test proves the translated locale renders its own copy while a namespace it
still lacks keeps falling back. Keeping both halves is
the point: a locale that carries a namespace must not silently render English,
and one that does not must not render a key token.

The es relay wave moved the "still lacks" probe from `relay.account_policy` to
`relay.login`: the former is now CARRIED, so leaving it as the negative control
would have converted the guard into an assertion that Spanish falls back — the
exact regression it exists to catch. The probe follows whichever relay
namespace the locale has not been given yet; `relay.login` is en-only for es.
"""

from local_operator import settings_io
from local_operator.i18n import messages
from local_operator.i18n.keys import wire_settings_account_policy as policy


def test_registry_uses_new_child_source_and_stable_machine_enum_ids():
    row = settings_io.resolve_key("retry.accountSelectionPolicy")
    assert row is not None
    assert row.label == "account selection policy"
    assert row.label == settings_io._t(policy.label())
    assert not row.help.startswith("wire.")
    assert [choice.value for choice in row.choices] == ["reset-soonest", "headroom", "fair"]
    assert "does not guarantee a cache hit" in row.help


def test_untranslated_child_uses_english_source_under_supported_locales():
    # es moved out of this list when its catalogue landed; the locales below
    # still have no `wire.settings.account_policy` file, so the English source
    # is the correct render and a key token would be the bug.
    for locale in ("fr", "ru", "vi", "hi"):
        assert messages.render(policy.label().code, {}, locale=locale) == "account selection policy"
        assert (
            messages.render("relay.account_policy.loading", {}, locale=locale)
            == "Loading account routing…"
        )


def test_spanish_carries_the_policy_label_and_still_falls_back_elsewhere():
    # The key identity is immutable; only the value is localised.
    assert policy.label().code == "wire.settings.account_policy.label"
    assert messages.render(policy.label().code, {}, locale="es") == "Política de cuentas"
    # Same locale, a namespace the relay wave has now given it: the carried
    # copy must render, not the English source.
    assert (
        messages.render("relay.account_policy.loading", {}, locale="es")
        == "Cargando el enrutamiento de cuentas…"
    )
    # A namespace es does not carry yet must still fall back to the English
    # source rather than to a key token, in the same locale.
    assert messages.render("relay.login.error.password", {}, locale="es") == "Wrong password."
