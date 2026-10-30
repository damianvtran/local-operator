"""New account-policy namespaces honestly fall back to English, not key tokens.

Which locale carries which namespace moves with every translation wave, so both
halves of this contract are DERIVED from the committed catalogues — the files the
emitter, the checker and the runtime read — and never from a list of locale codes.
The list that used to live here is the reason: `es`, `ru` and `vi` each had to be
hand-moved out as their catalogues landed, and the `ru` relay wave still reddened
main because `relay.account_policy` had shipped while the tuple still named `ru` as
untranslated.

Both halves are the contract this file exists for: a locale that CARRIES a
namespace must render its own copy, and a locale that does NOT must render the
English SENTENCE — never a key token, and never blank. Every key of both
namespaces is checked in both directions, so a partial translation cannot hide
behind a representative key, and an `es`-only wave on one namespace cannot make
the other's assertions wrong.

`relay.account_policy` and `wire.settings.account_policy` are covered together on
purpose: they land in separate waves (the es W2 wave carried the `wire.` child; the
relay wave carried `relay.` in ru/vi/ur/zh-CN), so a per-namespace answer is what
keeps this file correct between them.

The fallback half can only prove something while some shipped locale lacks each
namespace, and `locale_coverage.require_nonempty` FAILS (it does not skip) when
that stops being true: a silent pass is exactly how this file rotted.
"""

from __future__ import annotations

import pytest

from local_operator import settings_io
from local_operator.i18n import catalogues, messages
from local_operator.i18n.keys import wire_settings_account_policy as policy
from tests.unit.i18n import locale_coverage

#: The account-policy namespaces this file covers. `wire.` is the settings
#: registry's child; `relay.` is the session surfaces' child.
POLICY_NAMESPACES = ("wire.settings.account_policy", "relay.account_policy")


def test_registry_uses_new_child_source_and_stable_machine_enum_ids():
    row = settings_io.resolve_key("retry.accountSelectionPolicy")
    assert row is not None
    assert row.label == "account selection policy"
    assert row.label == settings_io._t(policy.label())
    assert not row.help.startswith("wire.")
    assert [choice.value for choice in row.choices] == ["reset-soonest", "headroom", "fair"]
    assert "does not guarantee a cache hit" in row.help


def test_policy_key_identity_is_stable():
    """The key identity is immutable; only the value is localised."""
    assert policy.label().code == "wire.settings.account_policy.label"


@pytest.mark.parametrize("namespace", POLICY_NAMESPACES)
def test_untranslated_child_uses_english_source_under_supported_locales(namespace):
    """A locale without the namespace renders the English sentence, never the id."""
    source = catalogues.load_catalogue("en", namespace)
    assert source, f"{namespace} must carry authored copy in en"

    untranslated = locale_coverage.absent(namespace)
    locale_coverage.require_nonempty(
        untranslated,
        subject=f"{namespace} fallback",
        detail=(
            "every shipped locale carries it "
            f"(carries={list(locale_coverage.carrying(namespace))})"
        ),
    )
    for locale in untranslated:
        for key in source:
            rendered = messages.render(key, {}, locale=locale)
            assert rendered == messages.render(key, {}, locale="en"), (locale, key)
            assert rendered != key, f"{locale} rendered the key token for {key}"


@pytest.mark.parametrize("namespace", POLICY_NAMESPACES)
def test_locale_that_carries_the_namespace_renders_its_own_copy(namespace):
    """A locale WITH the namespace must not silently render the English source."""
    source = catalogues.load_catalogue("en", namespace)
    assert source, f"{namespace} must carry authored copy in en"

    carrying = locale_coverage.carrying(namespace)
    locale_coverage.require_nonempty(
        carrying,
        subject=f"{namespace} own copy",
        detail=f"no shipped locale carries it (absent={list(locale_coverage.absent(namespace))})",
    )
    for locale in carrying:
        translated = catalogues.load_catalogue(locale, namespace)
        # A shipped locale is complete for the namespace it carries; a partial file
        # would fall back per key under a translated tag.
        assert set(translated) == set(source), locale
        for key in source:
            rendered = messages.render(key, {}, locale=locale)
            assert rendered != messages.render(key, {}, locale="en"), (locale, key)
            assert rendered != key, f"{locale} rendered the key token for {key}"
