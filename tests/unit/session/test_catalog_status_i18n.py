"""The armed-wake row label is BACKEND-generated copy, so it must be catalogued.

WHY THIS FILE EXISTS. A French census of the desktop UI found one row status
line still English in a French interface: ``Scheduled (1 wake)``. It was not a
client string — the label is built here, in ``CatalogEntry.status``, and the
sidebar prints ``row.status.label`` verbatim (its contract forbids a client
re-deriving status), so the hardcoded English literal bypassed the whole client
catalogue layer. The contract's answer for backend-originated copy is a stable
key plus params plus text rendered at the resolved language (``docs/I18N.md``,
RFC §2.6), so the assertions below are about three separate ways that can go
wrong:

* the English source must stay BYTE-IDENTICAL (every client, every English
  test and every dedupe key in the feed already carries this spelling);
* a locale that carries the namespace must get ITS OWN CLDR plural arms — an
  English ``s`` suffix would be wrong in most of the shipped set;
* a locale that does NOT carry it must fall back to English SOURCE and English
  FORMATTING rules, never to its own plural rules applied to English words
  (``docs/I18N.md``: Russian rules on English copy turn 21 into "21 job").

Plus the cost guard: ``_label_locale`` memoises the resolution because
``resolve_language`` parses ``config.yml`` when the process has no watcher, and
``status`` is a property read once per row.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from local_operator.i18n import catalogues
from local_operator.resume import SessionRow
from local_operator.session import catalog
from local_operator.session.catalog import CatalogEntry, scheduled_label, status_of

KEY = "wire.session.status.scheduled"
EN = "Scheduled ({count, plural, one {# wake} other {# wakes}})"
#: A fixture translation, NOT shipped data and not a contribution: it exists to
#: prove the key takes a locale's own arms. French is the locale the census ran
#: in, and it needs a ``many`` arm English does not have.
FR = "Planifié ({count, plural, one {# réveil} many {# de réveils} other {# réveils}})"


@pytest.fixture(autouse=True)
def english_product_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the product language, and drop the memo that remembers it.

    ``_label_locale`` caches ``resolve_language``'s answer against the config
    stamp and the two environment terms, so a test that pins a language has to
    invalidate it explicitly or it reads the previous test's answer.
    """
    monkeypatch.setenv("LOP_LANG", "en")
    monkeypatch.delenv("LOP_I18N", raising=False)
    monkeypatch.setattr(catalog, "_label_locale_memo", None)


@pytest.fixture
def translated_root(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A catalogue tree whose ``wire.session`` exists in en and fr only."""
    (tmp_path / "en").mkdir()
    (tmp_path / "fr").mkdir()
    (tmp_path / "en" / "wire.session.json").write_text(json.dumps({KEY: EN}), encoding="utf-8")
    (tmp_path / "fr" / "wire.session.json").write_text(json.dumps({KEY: FR}), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    return tmp_path


class TestTheEnglishSourceIsUnchanged:
    """The retired f-string spelled exactly this; nothing may drift."""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1, "Scheduled (1 wake)"),
            (2, "Scheduled (2 wakes)"),
            (3, "Scheduled (3 wakes)"),
            (21, "Scheduled (21 wakes)"),
        ],
    )
    def test_the_label_is_byte_identical_to_the_retired_f_string(
        self, count: int, expected: str
    ) -> None:
        assert scheduled_label(count) == expected


class TestTheLocaleSuppliesItsOwnArms:
    def test_a_carried_namespace_renders_in_its_own_language(self, translated_root) -> None:
        assert scheduled_label(1, locale="fr") == "Planifié (1 réveil)"
        assert scheduled_label(3, locale="fr") == "Planifié (3 réveils)"

    def test_a_plural_category_english_does_not_have(self, translated_root) -> None:
        # French puts an integer million in CLDR's ``many``: the arm exists only
        # because the catalogue's plural is real CLDR and not an ``s`` suffix.
        # The count is also grouped with French's own separator (U+202F), which
        # is the same point from the formatting side: both follow the LOCALE
        # the text is rendered in, not the language this file was written in.
        assert scheduled_label(1_000_000, locale="fr") == (
            "Planifié (1\u202f000\u202f000 de réveils)"
        )

    def test_an_uncarried_namespace_falls_back_to_english_source_and_rules(
        self, translated_root
    ) -> None:
        # ``ru`` carries no wire.session: English copy must be pluralised by
        # ENGLISH rules. Russian rules here would render "21 wake".
        assert scheduled_label(21, locale="ru") == "Scheduled (21 wakes)"
        assert scheduled_label(1, locale="ru") == "Scheduled (1 wake)"


class TestTheRowLabelIsTheCaller:
    """``CatalogEntry.status`` renders through the key — not a second spelling."""

    def test_the_row_reads_the_catalogue(self, translated_root, monkeypatch) -> None:
        monkeypatch.setenv("LOP_LANG", "fr")
        monkeypatch.setattr(catalog, "_label_locale_memo", None)
        row = SessionRow("x" * 12, 1.0, "a conversation", wakes=3)
        assert status_of(row, None) == ("scheduled", "Planifié (3 réveils)")

    def test_the_code_is_unchanged_beside_it(self) -> None:
        row = SessionRow("x" * 12, 1.0, "a conversation", wakes=2)
        assert CatalogEntry(row).status_code == "scheduled"
        assert CatalogEntry(row).status == "Scheduled (2 wakes)"


class TestTheLanguageIsResolvedOncePerBuild:
    """A config parse per row would be the price of the fix, so it is not paid."""

    def test_the_resolution_is_memoised_until_the_environment_moves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        from local_operator.i18n import resolve

        real = resolve.resolve_language

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr("local_operator.i18n.resolve.resolve_language", counting)
        monkeypatch.setattr(catalog, "_label_locale_memo", None)

        assert catalog._label_locale() == "en"
        for _ in range(50):
            catalog._label_locale()
        assert len(calls) == 1, "one resolution, not one per row"

        # The environment is part of the chain, so a flip is not a stale read.
        monkeypatch.setenv("LOP_LANG", "fr")
        assert catalog._label_locale() == "fr"
        assert len(calls) == 2
