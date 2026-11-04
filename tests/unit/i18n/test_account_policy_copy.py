"""New account-policy namespaces render from the tree, never from a locale list.

Every supported locale must satisfy one of two halves for each account-policy
namespace: with the key in its catalogue it renders its own translation, and
without it the resolver falls back to the exact English source rather than a key
token. Which half applies is DERIVED from the catalogue files, because the
enumerated form of this test went stale three times as relay catalogues landed
(fr, then ru, then vi): "which locales are missing copy today" is a snapshot of
the tree, not a property of the system.

The two halves are pinned in different places for exactly that reason. CARRYING
stays on the real corpus: the wave that lands a catalogue moves it with no edit
here. FALLBACK moved to a synthetic catalogue root once this wave closed the
corpus — with `relay.account_policy` in all seven shipped locales there is no
real locale left without it, so the real corpus can no longer show the state and
iterating it would compare nothing. `locale_coverage.require_nonempty` fails
that half on purpose rather than letting it pass vacuously; the fix it prescribes
is the synthetic root below, never a restored locale list.
"""

import json
from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.i18n import catalogues, messages
from local_operator.i18n.keys import wire_settings_account_policy as policy
from tests.unit.i18n import locale_coverage

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


def test_account_policy_copy_follows_the_catalogue_tree():
    # The CARRYING half, on the real corpus: a locale that holds the catalogue
    # must render its own copy — non-blank, and never the English source. Derived
    # from the tree, so a wave moves the answer here instead of editing a list.
    # The anti-vacuity guard runs on the carrying set only (it is still non-empty:
    # a namespace with no carrying locale is a corpus-wide regression). The other
    # direction — a locale WITHOUT the catalogue renders the English source — is
    # pinned in the synthetic-root test below, because the real corpus no longer
    # contains a locale that lacks `relay.account_policy`.
    for namespace, _code, _english in TARGET_MESSAGES:
        locale_coverage.require_nonempty(
            locale_coverage.carrying(namespace),
            subject=f"{namespace} own copy",
            detail=(
                "no shipped locale carries it "
                f"(absent={list(locale_coverage.absent(namespace))})"
            ),
        )
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
                # Still worth asserting where a real absence exists (six shipped
                # locales lack `wire.settings.account_policy`), but this branch is
                # opportunistic — the guaranteed discriminator for the fallback
                # direction is the synthetic root below.
                assert rendered == english, (
                    f"{locale} has no {namespace} catalogue carrying {code}, so the English "
                    f"source is the only correct render; got {rendered!r}. A rendered key "
                    "token means the resolver missed the fallback."
                )


def test_fallback_renders_the_english_source_where_the_catalogue_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locale without the namespace renders the exact English source.

    Synthetic by necessity: this contract is evidence only while some locale
    genuinely LACKS the namespace, and the `hi` wave left every shipped locale
    carrying `relay.account_policy`. A root built here produces the absence — and
    the presence beside it — by construction, so the assertion can still fail.

    Both directions run against the SAME swapped root on purpose: a resolver that
    read the shipped corpus anyway could not satisfy the carrying assertion (it
    would return the real translation, not the marker) while faking the fallback
    one, so a swap that silently does not take cannot make this test green.
    """
    carried, lacking = "fr", "vi"
    # `resolve.eligible_locale` coerces an UNshipped locale to `en`, which would
    # turn the fallback assertion into English compared with itself.
    for locale in (carried, lacking):
        assert locale in locale_coverage.targets(), (
            f"{locale} must be a shipped locale: an unshipped one is coerced to `en`, and the "
            "fallback comparison would then be English against itself"
        )
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / carried).mkdir(parents=True)
    marker = {}
    for namespace, code, english in TARGET_MESSAGES:
        marker[namespace] = f"Synthetic {namespace} copy"
        (root / "en" / f"{namespace}.json").write_text(
            json.dumps({code: english}), encoding="utf-8"
        )
        (root / carried / f"{namespace}.json").write_text(
            json.dumps({code: marker[namespace]}), encoding="utf-8"
        )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)

    for namespace, code, english in TARGET_MESSAGES:
        assert messages.render(code, {}, locale=carried) == marker[namespace], (
            f"{carried} carries {namespace} in the synthetic root, so its own copy must render; "
            "an English render here means the root swap did not take and the fallback "
            "assertion below would be comparing English with itself"
        )
        rendered = messages.render(code, {}, locale=lacking)
        assert rendered == english, (
            f"{lacking} has no {namespace} catalogue, so the English source is the only "
            f"correct render; got {rendered!r}"
        )
        assert rendered != code, "a rendered key token means the resolver missed the fallback"


def test_spanish_carries_the_policy_label_in_its_own_copy():
    # The key identity is immutable; only the value is localised. This is a
    # content assertion about Spanish, not a claim about which locales still
    # lack copy — the derived test above owns that question for every locale.
    assert policy.label().code == "wire.settings.account_policy.label"
    assert messages.render(policy.label().code, {}, locale="es") == "Política de cuentas"
