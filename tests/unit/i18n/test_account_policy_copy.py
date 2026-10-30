"""New account-policy namespaces honestly fall back to English, not key tokens.

Spanish now carries `wire.settings.account_policy` (the es W1 wave), so the
fallback pin is scoped to the locales that still do NOT carry it and a second
test proves the translated locale renders its own copy while a namespace it
still lacks (`relay.account_policy`) keeps falling back. Keeping both halves is
the point: a locale that carries a namespace must not silently render English,
and one that does not must not render a key token.
"""

from local_operator import settings_io
from local_operator.i18n import catalogues, messages
from local_operator.i18n.keys import wire_settings_account_policy as policy
from local_operator.i18n.runtime import render_message


def test_registry_uses_new_child_source_and_stable_machine_enum_ids():
    row = settings_io.resolve_key("retry.accountSelectionPolicy")
    assert row is not None
    assert row.label == "account selection policy"
    assert row.label == settings_io._t(policy.label())
    assert not row.help.startswith("wire.")
    assert [choice.value for choice in row.choices] == ["reset-soonest", "headroom", "fair"]
    assert "does not guarantee a cache hit" in row.help


def test_every_supported_locale_renders_its_own_copy_or_the_english_source():
    """A locale either carries the namespace or reads the English SOURCE.

    This used to name the locales it expected to fall back ("fr", "vi", "hi", …),
    which rots the moment any lane ships that catalogue — it went red when the
    Russian relay wave landed, through no fault of the code it tests. The
    property is derived from the catalogues now, so it cannot go stale: a
    locale that has no file must render the English sentence (never a key
    token), and a locale that has one must render ITS byte for byte.
    """
    for namespace, key in (
        ("relay.account_policy", "relay.account_policy.loading"),
        ("wire.settings.account_policy", policy.label().code),
    ):
        english = messages.render(key, {}, locale="en")
        for locale in ("en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"):
            rendered = messages.render(key, {}, locale=locale)
            assert f"{namespace}." not in rendered, (locale, rendered)
            try:
                catalogue = catalogues.load_catalogue(locale, namespace)
            except catalogues.CatalogueNotFound:
                assert rendered == english, (locale, namespace, rendered)
                continue
            if key in catalogue:
                assert rendered == render_message(catalogue[key], {}, locale), (locale, namespace)
            else:
                assert rendered == english, (locale, namespace, rendered)


def test_spanish_carries_the_policy_label_and_still_falls_back_elsewhere():
    # The key identity is immutable; only the value is localised.
    assert policy.label().code == "wire.settings.account_policy.label"
    assert messages.render(policy.label().code, {}, locale="es") == "Política de cuentas"
    # Spanish now CARRIES the relay account-policy chrome, so it must render its
    # own copy rather than the English source.
    assert (
        messages.render("relay.account_policy.loading", {}, locale="es")
        == "Cargando el enrutamiento de cuentas…"
    )
    # …and a namespace es still does NOT carry must fall back to the English
    # source rather than to a key token, in the same locale. `wire.errors.sdk_spec`
    # is English-pinned by caller contract, so it is a real missing namespace
    # rather than a synthetic one.
    key = "wire.errors.sdk_spec.attach_fields_unsupported"
    fallback = messages.render(key, {"fields": "x"}, locale="es")
    assert fallback == messages.render(key, {"fields": "x"}, locale="en")
    # …and the fallback must be the ENGLISH SENTENCE, never the key token — the
    # property the original "still falls back elsewhere" assertion protected.
    assert "wire.errors.sdk_spec." not in fallback
