"""New account-policy namespaces render from the tree, never from a locale list.

Every supported locale must satisfy one of two halves for each account-policy
namespace: with the key in its catalogue it renders its own translation, and
without it the resolver falls back to the exact English source rather than a key
token. Which half applies is DERIVED from the catalogue files, because the
enumerated form of this test went stale three times as relay catalogues landed
(fr, then ru, then vi): "which locales are missing copy today" is a snapshot of
the tree, not a property of the system.
"""

import json
from pathlib import Path

from local_operator import settings_io
from local_operator.i18n import catalogues, messages
from local_operator.i18n.keys import wire_settings_account_policy as policy

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "i18n" / "languages.json"

#: ``(namespace, key, English source)`` for the two account-policy surfaces this
#: module pins. The English source is spelled out here because it IS the fallback
#: contract: a locale without the catalogue must render exactly this text, and
#: a locale with the catalogue must render anything but it.
TARGET_MESSAGES = (
    ("wire.settings.account_policy", policy.label().code, "account selection policy"),
    ("relay.account_policy", "relay.account_policy.loading", "Loading account routing…"),
)


def _supported_non_source_locales() -> list[str]:
    """The language manifest's locales, minus the English source itself."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [entry["code"] for entry in manifest["languages"] if entry["status"] != "source"]


def _carries(locale: str, namespace: str, code: str) -> bool:
    """Whether ``locale``'s catalogue for ``namespace`` actually carries ``code``.

    Read from the same catalogue tree the generator and the resolver read, so a
    catalogue that lands for a locale moves that locale's expectation here with
    no edit to this file.
    """
    path = catalogues.catalogue_root() / locale / f"{namespace}.json"
    if not path.is_file():
        return False
    return code in json.loads(path.read_text(encoding="utf-8"))


def test_registry_uses_new_child_source_and_stable_machine_enum_ids():
    row = settings_io.resolve_key("retry.accountSelectionPolicy")
    assert row is not None
    assert row.label == "account selection policy"
    assert row.label == settings_io._t(policy.label())
    assert not row.help.startswith("wire.")
    assert [choice.value for choice in row.choices] == ["reset-soonest", "headroom", "fair"]
    assert "does not guarantee a cache hit" in row.help


<<<<<<< HEAD
def test_untranslated_child_uses_english_source_under_supported_locales():
    # es moved out of this list when its catalogue landed; the locales below
    # still have no `wire.settings.account_policy` file, so the English source
    # is the correct render and a key token would be the bug. hi left the list
    # for `relay.account_policy` only — it now carries that namespace — so its
    # own half lives in the test below rather than being silently dropped.
    for locale in ("fr", "ru", "vi"):
        assert messages.render(policy.label().code, {}, locale=locale) == "account selection policy"
        assert (
            messages.render("relay.account_policy.loading", {}, locale=locale)
            == "Loading account routing…"
        )


def test_hindi_carries_relay_account_policy_and_still_falls_back_for_the_child():
    # The relay backfill gave hi the phone-client namespace, so this locale must
    # render its own copy there. The settings-registry child
    # (`wire.settings.account_policy`) is a DIFFERENT namespace that hi still
    # lacks, and it must keep falling back to the English source.
    assert (
        messages.render("relay.account_policy.loading", {}, locale="hi")
        == "खाता रूटिंग लोड हो रही है…"
    )
    assert messages.render(policy.label().code, {}, locale="hi") == "account selection policy"


def test_spanish_carries_the_policy_label_and_still_falls_back_elsewhere():
    # The key identity is immutable; only the value is localised.
=======
def test_account_policy_copy_follows_the_catalogue_tree():
    for locale in _supported_non_source_locales():
        for namespace, code, english in TARGET_MESSAGES:
            rendered = messages.render(code, {}, locale=locale)
            if _carries(locale, namespace, code):
                assert rendered.strip(), (
                    f"{locale} carries {namespace} but {code} renders a blank value: an empty "
                    "catalogue entry is a defect, never a fallback."
                )
                assert rendered != english, (
                    f"{locale} carries {namespace} but {code} still renders the English "
                    f"source {english!r}: a shipped catalogue holding an English-identical "
                    "value is a copy defect, not a fallback."
                )
            else:
                assert rendered == english, (
                    f"{locale} has no {namespace} catalogue carrying {code}, so the English "
                    f"source is the only correct render; got {rendered!r}. A rendered key "
                    "token means the resolver missed the fallback."
                )


def test_spanish_carries_the_policy_label_in_its_own_copy():
    # The key identity is immutable; only the value is localised. This is a
    # content assertion about Spanish, not a claim about which locales still
    # lack copy — the derived test above owns that question for every locale.
>>>>>>> origin/main
    assert policy.label().code == "wire.settings.account_policy.label"
    assert messages.render(policy.label().code, {}, locale="es") == "Política de cuentas"
