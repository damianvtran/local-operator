"""The security-hold split: held bytes are absent, not merely unlisted.

``SECURITY_HELD_LOCALES`` in ``scripts/i18n/generate.py`` is EMPTY as of
2026-10-24, by the policy owner's explicit disposition: zh-CN's blocking
``wire.slash`` audit was remediated at current bytes and every carried
namespace now has a fresh passed ledger row, so the generated shipped set
carries eight locales. This file no longer pins Chinese's hold; it pins the
MECHANISM that made the hold meaningful, because an empty hold set is exactly
the condition under which a future edit could delete the check as "dead code":

* the active tree may only carry bytes for locales the ledger ships (a held
  locale's bytes are removed, not hidden behind an eligibility filter);
* the generator still excludes a locale that is IN the hold set even when
  every ledger row it needs is present and passed — proven with synthetic
  catalogue bytes in an isolated fixture, both for zh-CN and for a locale that
  was never held, so the check cannot regress into a zh-CN special case;
* the same fixture ships those locales when the hold is empty, which is the
  positive control that stops the negative control from passing vacuously.

The Hindi team-bridge delta at the bottom is unrelated to the disposition; it
stays here because it landed in the same wave and is still a live contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.cells import cell_len

from local_operator.i18n import catalogues, resolve, runtime, text
from local_operator.i18n.messages import Msg, envelope

REPO = Path(__file__).resolve().parents[3]


def test_active_ledger_carries_the_released_chinese_candidate() -> None:
    ledger = json.loads((REPO / "i18n" / "ledger.json").read_bytes())["entries"]
    chinese = [entry for entry in ledger if entry["locale"] == "zh-CN"]
    # The disposition was taken because every namespace the locale CARRIES has
    # a passed audit at the CURRENT en bytes; a row for a namespace with no
    # file (or a file with no row) would mean the released package carries
    # unaudited or missing copy. Keep the exact-name and freshness pins, not
    # just the cardinality: 81 was the number that satisfied the gate on the
    # disposition date, and the set equality is what actually enforces it.
    # The relay wave adds the two namespaces zh-CN now carries alongside the
    # release: `relay.approvals` (17 keys, the approval-scope copy main added
    # as literals) and the restored 77-key `relay.asks`. Both were independently
    # audited at these bytes on 2026-10-25 (four MINOR findings on relay.asks,
    # nits elsewhere, no blocker/major), so the cardinality is 83.
    assert len(chinese) == 83
    assert all(entry["status"] == "passed" for entry in chinese)
    carried = set(catalogues.namespaces("zh-CN"))
    assert {entry["namespace"] for entry in chinese} == carried
    for entry in chinese:
        assert entry["source_sha256"] == catalogues.catalogue_sha256("en", entry["namespace"]), (
            entry["namespace"],
            "stale audit against the current English bytes",
        )

    hindi = [entry for entry in ledger if entry["locale"] == "hi"]
    # TWENTY-FOUR: the union plus independently audited P3a/P3b namespaces,
    # the scratchpad-hygiene slice's hi wire.settings row, P4a's audited
    # tui.viewer row (including its keyboard-hint audit refresh), and the
    # i18n lane's audited `wire.memory` row for the namespace translated in
    # five active locales.
    # Base is twelve (the Wave-3 ten retain their audited source pins). Main
    # added a wire.settings row for the hub-setting deletion and the LOH-19
    # review slice added a tui.review row; the earlier branch added a
    # tui.transcript row for the independent re-attestation of the carried
    # content; LOH-5 slice 4 adds two wire.settings rows for the
    # transcription-cleanup setting and its D1 copy remediation. P1 adds the
    # audited `tui.schedule` row on top — SEVENTEEN from main's history plus
    # that one. LOH-19 P2 adds the audited `tui.projects_edit` row — EIGHTEEN
    # plus that one. P3a adds `tui.authoring` after its independent 40-key
    # audit — NINETEEN plus one. Keep the cardinality pin: adding an audited union must not
    # silently admit an unrelated held namespace or discard the previous audit
    # history.
    # P3b adds its 35-key tui.team_authoring namespace after independent audit.
    # The audited `hi/wire.memory` row this branch lands makes it TWENTY-FOUR;
    # the /tag slice's independently re-audited 27-key `tui.tags` row makes it
    # TWENTY-FIVE. LOH-11's independently audited `relay.session.tags` row
    # makes it TWENTY-SIX. The remote_compute six-key wire.settings delta's
    # independently audited row makes it TWENTY-SEVEN. The S3e remote-compute
    # wave adds one row per carried namespace (`cli.exec_remote_help`,
    # `wire.remote_compute`) -- TWENTY-NINE. The pin stays cardinal so an
    # unrelated held namespace cannot slip in. The session-transcript-
    # compression slice's independently audited 8-key `wire.settings` delta
    # (0 blocker, 0 major; comments carried) makes it THIRTY. The R5 stale-ask
    # wave's two independently audited namespaces (`tui.asks`, `relay.ask`)
    # make it THIRTY-TWO, and the P5 command-palette slice's independently
    # audited 7-key `tui.palette` row makes it THIRTY-THREE. The fast-mode
    # fix's refusal notices add one audited row each for `tui.notices` and
    # `wire.errors.desktop` -- THIRTY-FIVE (both newly carried, 30 and 84 keys).
    # The relay.asks re-audit adds the 77-key `hi/relay.asks` row -- THIRTY-SIX.
    # The approvals slice adds the independently audited 17-key
    # `hi/relay.approvals` row -- THIRTY-SEVEN.
    assert len(hindi) == 37
    assert all(entry["status"] == "passed" for entry in hindi)


def test_the_active_tree_holds_bytes_only_for_shipped_locales() -> None:
    # The real-tree half of the mechanism: the generator gate is what forces a
    # held locale's bytes out of the package, so with the hold empty every
    # committed locale directory must be a shipped one, and vice versa. A
    # directory that is present but not shipped is the "unlisted, not absent"
    # state the hold was created to prevent.
    shipped = resolve.shipped_locales()
    assert shipped == ("en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi")
    assert set(catalogues.locales()) == set(shipped)
    generated = json.loads((REPO / "local_operator/i18n/data/shipped.json").read_text())
    assert tuple(generated["locales"]) == shipped
    for locale in shipped:
        for namespace in catalogues.namespaces(locale):
            assert catalogues.catalogue_bytes(locale, namespace), (locale, namespace)
    chinese = set(catalogues.namespaces("zh-CN"))
    for namespace in sorted(set(catalogues.namespaces("en")) - chinese):
        # Per-namespace fallback (§2.5) covers an en-only child: the loader
        # must report absence, never serve a forged English copy as Chinese.
        with pytest.raises(catalogues.CatalogueNotFound):
            catalogues.load_catalogue("zh-CN", namespace)


def test_chinese_carries_its_own_copy_in_every_namespace() -> None:
    for namespace in catalogues.namespaces("zh-CN"):
        translated = catalogues.load_catalogue("zh-CN", namespace)
        source = catalogues.load_catalogue("en", namespace)
        if not source:
            # The three `wire.*` extraction stubs M0 shipped are empty in EVERY
            # locale until their slice populates them (RFC §5). Empty is not an
            # English fallback: there are no keys to render.
            assert translated == {}, namespace
            continue
        assert catalogues.catalogue_bytes("zh-CN", namespace) != catalogues.catalogue_bytes(
            "en", namespace
        ), f"{namespace}: shipped Chinese bytes are the English source"
        # A shipped locale must be complete for the namespace it carries; a
        # partial file would silently fall back per key under a zh-CN tag.
        assert translated.keys() == source.keys(), namespace
        for key in source:
            assert runtime.message_arguments(translated[key]) == runtime.message_arguments(
                source[key]
            ), (namespace, key)


def test_a_held_locale_is_excluded_even_with_every_row_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, i18n_generate
) -> None:
    """The hold still gates the generator, and it is not a zh-CN special case.

    Synthetic bytes only: en, fr and zh-CN each carry the SAME namespace, and
    fr/zh-CN get a passed, current-byte ledger row. Without a hold both ship
    (the positive control). Adding either to ``SECURITY_HELD_LOCALES`` — the
    locale that was held and the one that never was — excludes exactly that
    locale, so a later "dead code" deletion fails here.
    """
    root = tmp_path / "catalogues"
    for locale in ("en", "fr", "zh-CN"):
        (root / locale).mkdir(parents=True)
        (root / locale / "wire.exec.json").write_text(
            json.dumps({"wire.exec.demo": "Synthetic receipt"}), encoding="utf-8"
        )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(i18n_generate, "LEDGER", tmp_path / "ledger.json")
    en_sha = catalogues.catalogue_sha256("en", "wire.exec")
    i18n_generate.LEDGER.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "locale": locale,
                        "namespace": "wire.exec",
                        "status": "passed",
                        "source_sha256": en_sha,
                    }
                    for locale in ("fr", "zh-CN")
                ]
            }
        ),
        encoding="utf-8",
    )
    # Positive control: the disposition's real, empty hold set ships both.
    assert i18n_generate.SECURITY_HELD_LOCALES == frozenset()
    assert i18n_generate._shipped_locales() == ["en", "fr", "zh-CN"]

    monkeypatch.setattr(i18n_generate, "SECURITY_HELD_LOCALES", frozenset({"fr"}))
    assert i18n_generate._shipped_locales() == ["en", "zh-CN"]
    monkeypatch.setattr(i18n_generate, "SECURITY_HELD_LOCALES", frozenset({"zh-CN"}))
    assert i18n_generate._shipped_locales() == ["en", "fr"]
    monkeypatch.setattr(i18n_generate, "SECURITY_HELD_LOCALES", frozenset({"fr", "zh-CN"}))
    assert i18n_generate._shipped_locales() == ["en"]


@pytest.mark.parametrize("key", ("wire.slash.delete", "wire.exec.receipt_suffix"))
def test_shipped_chinese_uses_its_own_wire_copy(key: str) -> None:
    message = Msg(key)
    chinese = message.envelope(locale="zh-CN")
    english = message.envelope(locale="en")
    assert chinese["text"] != key
    assert english["text"] != key
    # The shipped tag renders the Chinese catalogue value, not the source.
    assert chinese["text"] == catalogues.load_catalogue("zh-CN", key.rsplit(".", 1)[0])[key]
    assert chinese["text"] != english["text"]
    assert envelope(key, locale="zh-CN") == chinese


@pytest.mark.parametrize("key", ("wire.slash.delete", "wire.exec.receipt_suffix"))
def test_direct_tr_uses_shipped_copy_fresh_and_after_selection(key: str) -> None:
    namespace = key.rsplit(".", 1)[0]
    message = Msg(key)
    render = text.tr(namespace)
    previous = text.active_locale()
    try:
        text.set_active_locale("zh-CN")
        assert text.active_locale() == "zh-CN"
        chinese = message.envelope(locale="zh-CN")["text"]
        assert chinese != message.envelope(locale="en")["text"]
        # Twice: the second call exercises the cached render path, where a
        # stale English entry would show up as a silent fallback.
        assert render(message) == chinese
        assert render(message) == chinese
        # Normal product selection also resolves the shipped tag before tr.
        text.set_active_locale(resolve.resolve_language("zh-CN", environ={}, system="Linux"))
        assert text.active_locale() == "zh-CN"
        assert render(message) == chinese
    finally:
        text.set_active_locale(previous)


@pytest.mark.parametrize(
    "suffix",
    (
        "team_bridge.description",
        "team_bridge.title",
        "team_bridge_enabled.help",
        "team_bridge_enabled.label",
        "team_bridge_session.help",
        "team_bridge_session.label",
    ),
)
def test_hindi_team_bridge_delta_uses_audited_copy_not_english_fallback(suffix: str) -> None:
    # The landed English source gained these six keys after the old 755-key
    # audit. Check the public consumer, not only JSON parity: a missing Hindi
    # value would otherwise silently render English while locale selection works.
    key = f"wire.settings.{suffix}"
    hindi = catalogues.load_catalogue("hi", "wire.settings")[key]
    english = catalogues.load_catalogue("en", "wire.settings")[key]
    rendered = Msg(key).envelope(locale="hi")["text"]
    assert rendered == hindi
    assert rendered != english
    assert runtime.message_arguments(hindi) == ()
    if suffix.endswith(".label"):
        # Use the real row budget; help and description have no authored cap.
        from local_operator.tui.widgets.settings_view import _LABEL_BUDGET

        assert cell_len(rendered) <= _LABEL_BUDGET
    if suffix == "team_bridge_enabled.help":
        assert "Radient" in rendered
    if suffix == "team_bridge_session.label":
        assert "CoS" in rendered
