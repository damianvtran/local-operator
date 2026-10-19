"""New account-policy namespaces honestly fall back to English, not key tokens."""

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
    for locale in ("fr", "es", "ru", "vi", "hi"):
        assert messages.render(policy.label().code, {}, locale=locale) == "account selection policy"
        assert (
            messages.render("relay.account_policy.loading", {}, locale=locale)
            == "Loading account routing…"
        )
