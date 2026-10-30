"""Slash descriptions render in the language the process RESOLVED, not English.

The defect: the registry built every description by calling
``messages.render(code, params)`` with no locale. That argument defaults to
``None``, ``resolve.eligible_locale`` coerces it to ``"en"``, so the sentence
was English by construction — in every process, whatever ``language:``,
``LOP_LANG`` or OS locale said, and whatever ``set_active_locale`` did. The
descriptions were translated and audited all along (all seven non-English
locales carry every ``wire.slash`` key); only the render point was wrong.

The fix has TWO render points because two kinds of consumer read the copy
differently, and this module pins both:

* ``SLASH_COMMANDS``' ``description`` field, re-materialised IN PLACE by the
  registry's ``register_refresh`` hook, for the TUI (its picker and ``/help``).
* ``describe(spec, locale)``, rendered on demand, for the consumers that
  resolve a language per request and never activate one process-wide (the
  desktop catalogue and the phone sheet — nothing outside the TUI and the
  remote-compute CLI calls ``set_active_locale``).

``tests/conftest.py``'s autouse ``restore_active_locale`` puts every test back
on ``en`` through the public setter, which re-runs the refreshers, so the moves
below cannot leak into the English-pinned suites.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from local_operator.i18n import catalogues
from local_operator.i18n.messages import render
from local_operator.i18n.text import active_locale, set_active_locale
from local_operator.slash_commands import SLASH_COMMANDS, describe
from local_operator.tui.autocomplete import SlashCommand

HELP_EN = "List all commands"
HELP_FR = "Lister toutes les commandes"


def _spec(name: str) -> SlashCommand:
    return next(spec for spec in SLASH_COMMANDS if spec.name == name)


def test_every_published_row_carries_the_message_it_was_rendered_from() -> None:
    """The wire's 46 rows are all catalogued, so all 46 can render in any locale.

    The withheld rows are the ones whose copy lives outside ``wire.*`` (TUI
    pages) — they still carry their message, they are simply not served. A row
    with no message would silently stay English on every surface, which is the
    defect this change removes, so the count and the namespace are pinned.
    """
    published = [spec for spec in SLASH_COMMANDS if spec.desktop_destination]
    assert len(published) == 46
    for spec in published:
        assert spec.description_message is not None, spec.name
        assert spec.description_message.code.startswith("wire."), spec.name


def test_describe_renders_a_locale_this_process_never_activated() -> None:
    """The daemon's situation: active locale is ``en``, the row still renders French.

    Rendering must come from ``describe``'s argument here, because nothing under
    ``server/`` or ``mobile/`` ever calls ``set_active_locale`` — reading the
    field would hand every desktop user the English copy.
    """
    assert active_locale() == "en"
    assert _spec("help").description == HELP_EN
    assert describe(_spec("help"), "fr") == HELP_FR
    # And the row itself still holds English: the boundary render is a pure read.
    assert _spec("help").description == HELP_EN


def test_describe_matches_the_french_catalogue_and_differs_from_english() -> None:
    """The sentence is the CATALOGUE's, not a literal this test owns."""
    french = catalogues.load_catalogue("fr", "wire.slash")
    for name in ("help", "exit", "clear", "model"):
        spec = _spec(name)
        assert spec.description_message is not None
        expected = french[spec.description_message.code]
        assert describe(spec, "fr") == expected, name
        assert describe(spec, "en") != expected, name
        assert describe(spec, "en") == spec.description, name


def test_a_literal_description_still_answers_in_every_locale() -> None:
    """A row with no message is its own copy: never English-by-accident, never blank."""
    literal = SlashCommand("legacy", "Keep my own words")
    assert literal.description_message is None
    assert describe(literal, "fr") == "Keep my own words"
    assert describe(literal) == "Keep my own words"


def test_an_unsupported_locale_falls_back_to_english() -> None:
    """``render`` coerces through ``eligible_locale``: a bad tag is English, not raw.

    Dedicated so a future ``describe`` that bypassed the runtime's coercion would
    fail here rather than shipping a half-resolved row to a client.
    """
    assert describe(_spec("help"), "zz") == HELP_EN
    assert describe(_spec("help"), "") == HELP_EN


def test_the_kill_switch_outranks_a_warmed_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LOP_I18N=0`` forces English even after a French entry is memoised.

    The memo is keyed on the locale that will ACTUALLY render, never on the tag
    the caller asked for, so the coerce has to happen before the key: coercing
    inside ``render`` alone left a warm French entry being served straight
    through the switch (the RFC §9 control is read per call; an ``lru_cache``
    entry is not). Both directions are pinned, because a memo that misses the
    switch on the way in misses it on the way out too.
    """
    spec = _spec("help")
    # Warm French, then flip the switch: English, not the memoised French.
    assert describe(spec, "fr") == HELP_FR
    monkeypatch.setenv("LOP_I18N", "0")
    assert describe(spec, "fr") == HELP_EN
    # The other direction: warm English under the switch, then lift it.
    assert describe(spec, "fr") == HELP_EN
    monkeypatch.delenv("LOP_I18N")
    assert describe(spec, "fr") == HELP_FR


def test_a_catalogued_row_stays_hashable_and_equal_across_its_message() -> None:
    """The message is provenance, not identity: ``hash(row)`` must not raise.

    ``SlashCommand`` is frozen, so its generated ``__hash__`` hashes every
    COMPARED field — and ``Msg.params`` is a dict, so carrying the message made
    ``hash(row)`` raise ``TypeError`` for all 62 catalogued rows the moment the
    field arrived. Latent only because nothing hashes a row today; excluded from
    comparison, a set or dict key over rows keeps working, and two rows for the
    same command stay equal whether or not one of them still carries the id it
    was rendered from.
    """
    row = _spec("help")
    without_message = replace(row, description_message=None)
    assert without_message.description_message is None
    # The regression, in the order it was reported: with the message compared,
    # this line raised ``TypeError: unhashable type: 'dict'`` for every row.
    assert hash(row) == hash(without_message)
    assert without_message == row
    assert len({row, without_message}) == 1


def test_the_registry_refreshes_in_place_when_the_locale_moves() -> None:
    """``/language`` re-renders the rows without invalidating anybody's list.

    Every importer holds the MODULE's list object (the app, the pickers, the
    desktop adapter, the phone daemon). A rebind would leave each of them the
    boot rows, so the refresher must assign into the same list — pinned here by
    identity as well as by content.
    """
    registry = SLASH_COMMANDS
    assert _spec("help").description == HELP_EN
    set_active_locale("fr")
    try:
        assert SLASH_COMMANDS is registry
        assert _spec("help").description == HELP_FR
        # The message stayed put; only the rendered sentence moved.
        moved = _spec("help").description_message
        assert moved is not None
        assert moved.code == "wire.slash.help"
    finally:
        set_active_locale("en")
    assert _spec("help").description == HELP_EN


def test_describe_agrees_with_the_refreshed_field_under_the_active_locale() -> None:
    """The two render points cannot disagree about the active locale's wording.

    This is the documented hazard of having two of them (a per-request resolve
    beside an import-time constant): if ``describe(spec)`` and the refreshed
    ``spec.description`` ever diverge, the TUI and the desktop would print
    different sentences for the same command in the same language.
    """
    set_active_locale("fr")
    try:
        for spec in SLASH_COMMANDS:
            assert describe(spec) == describe(spec, "fr") == spec.description, spec.name
    finally:
        set_active_locale("en")
    for spec in SLASH_COMMANDS:
        assert describe(spec) == spec.description, spec.name


def test_the_pre_fix_render_path_is_still_english_after_a_locale_move() -> None:
    """NEGATIVE CONTROL: the seam that caused this defect still behaves that way.

    The registry used to build its rows with ``render(code, params)`` — one
    argument short. That call is unchanged in the runtime and stays English
    under a French process, which is exactly why the fix could not be "call
    ``render`` and hope": it has to pass the locale, or read the active one, on
    every path. If a future refactor routes ``describe`` back through the bare
    call, this test fails while the rest of the suite could still be green.
    """
    message = _spec("help").description_message
    assert message is not None
    set_active_locale("fr")
    try:
        assert active_locale() == "fr"
        assert render(message.code, message.params) == HELP_EN  # the old seam
        assert describe(_spec("help")) == HELP_FR  # the fixed one
    finally:
        set_active_locale("en")
