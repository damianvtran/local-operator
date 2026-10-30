"""New account-policy namespaces honestly fall back to English, not key tokens.

Which locale carries which namespace moves with every translation wave, so this
file DERIVES both sides from the committed catalogues — the same files the
emitter and the runtime read — instead of carrying a hand-maintained list of
"untranslated" locales. The list that used to live here had `es`, `ru` and `vi`
hand-moved out of it one wave at a time, and the `ru` relay wave still turned
main red: a hand list lags reality by exactly one merge, and every locale after
it (today `hi`, tomorrow `es`'s relay file) breaks it again.

Both halves are the contract this file exists for: a locale that CARRIES a
message must not silently render English, and one that does NOT must not render
a key token. Membership is decided per MESSAGE ID rather than per file, because
`messages.render_record` falls back per id — `_lookup` returns ``None`` for a
locale whose catalogue omits the key, so such a locale reads the English source
even though its file exists.

The fallback half can only prove something while some shipped locale lacks the
message, so `_skip_unless_discriminating` guards it. That guard SKIPS (with a
reason naming the derived sets) rather than failing: a fully translated corpus
is the outcome this project is working towards, not a regression, and painting
main red on the day it completes is the same false alarm the derivation removes.
The skip is reported on every run, so the coverage gap stays visible; covering
it again means pinning the fallback against a synthetic catalogue root (as
`tests/unit/i18n/test_catalogues.py` does), not re-introducing a locale list.
"""

from __future__ import annotations

import functools

import pytest

from local_operator import settings_io
from local_operator.i18n import catalogues, messages, resolve
from local_operator.i18n.keys import wire_settings_account_policy as policy

#: One message id per account-policy namespace this file covers: the settings
#: child the registry row reads, and the relay child the session surfaces read.
POLICY_MESSAGE_IDS = (policy.label().code, "relay.account_policy.loading")


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


@functools.lru_cache(maxsize=None)
def _source_namespace(code: str) -> str:
    """The one catalogue namespace that defines ``code``, decided by ``en``.

    Discovery mirrors :func:`messages.render_record` — the authored source says
    where an id lives, and the collision rule means exactly one file defines it —
    so this helper follows the runtime rather than inventing a second rule. The
    corpus is package data and immutable at runtime, hence the cache; a test that
    swaps the catalogue root must not use this helper between swaps.
    """
    defining = [
        namespace
        for namespace in catalogues.namespaces("en")
        if code in catalogues.load_catalogue("en", namespace)
    ]
    assert len(defining) == 1, f"{code} must be defined by exactly one en catalogue, got {defining}"
    return defining[0]


def _fallback_checked_locales() -> tuple[str, ...]:
    """Shipped locales other than the source — the set ``render`` can honour.

    The universe is the generated shipment ledger, not a list here, so a new
    locale needs no edit. Unshipped locales are excluded because
    :func:`resolve.eligible_locale` coerces them to ``en``, which would compare
    the source with itself and pass for the wrong reason.
    """
    return tuple(locale for locale in resolve.shipped_locales() if locale != "en")


@functools.lru_cache(maxsize=None)
def _carrying_locales(code: str) -> frozenset[str]:
    """Shipped locales whose OWN catalogues define ``code``.

    A locale whose directory or namespace file is absent is simply not carrying
    the message — the same reading the runtime takes, which lands on the English
    source rather than raising. Cached for the same reason as
    :func:`_source_namespace`.
    """
    namespace = _source_namespace(code)
    present = set(catalogues.locales())
    return frozenset(
        locale
        for locale in _fallback_checked_locales()
        if locale in present
        and namespace in catalogues.namespaces(locale)
        and code in catalogues.load_catalogue(locale, namespace)
    )


def _missing_locales(code: str) -> tuple[str, ...]:
    carrying = _carrying_locales(code)
    return tuple(locale for locale in _fallback_checked_locales() if locale not in carrying)


def _skip_unless_discriminating(code: str, what: str, remaining: tuple[str, ...]) -> None:
    """Skip a half that has nothing left to compare against, naming why.

    A guard that cannot go red is worse than no guard, and one that goes red on a
    *good* outcome (every locale translated) blocks a release for no fault. The
    reason carries the derived sets so the reader can see what stopped being
    exercised without re-deriving it.
    """
    if remaining:
        return
    pytest.skip(
        f"{code}: no shipped locale {what} — this half is vacuous on the current "
        f"corpus (carries={sorted(_carrying_locales(code))}, "
        f"shipped={list(_fallback_checked_locales())}); pin the fallback against a "
        f"synthetic catalogue root instead of restoring a locale list"
    )


@pytest.mark.parametrize("code", POLICY_MESSAGE_IDS)
def test_untranslated_child_uses_english_source_under_supported_locales(code):
    """A locale without the message renders the English source, never the id."""
    english = messages.render(code, {}, locale="en")
    assert english != code, "the authored source itself must be catalogue-backed"

    missing = _missing_locales(code)
    _skip_unless_discriminating(code, "lacks it", missing)
    for locale in missing:
        rendered = messages.render(code, {}, locale=locale)
        assert rendered == english, f"{locale} lacks {code} and must fall back to the source"
        assert rendered != code, f"{locale} rendered the key token, not English"


@pytest.mark.parametrize("code", POLICY_MESSAGE_IDS)
def test_locale_that_carries_the_namespace_does_not_render_english(code):
    """A locale WITH the message must not silently render the English source."""
    english = messages.render(code, {}, locale="en")
    assert english != code, "the authored source itself must be catalogue-backed"

    carrying = tuple(sorted(_carrying_locales(code)))
    _skip_unless_discriminating(code, "carries it", carrying)
    for locale in carrying:
        rendered = messages.render(code, {}, locale=locale)
        assert rendered != english, f"{locale} carries {code} yet rendered the English source"
        assert rendered != code, f"{locale} rendered the key token, not its translation"
