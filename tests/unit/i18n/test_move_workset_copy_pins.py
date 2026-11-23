"""Copy pins for the move-working-set surfaces (i18n audit MAJOR, review round 2).

THE DEFECT THIS REMEMBERS. ``cli.sessions_move_workset.desktop_refused`` read
``{device}`` before ``{source}`` in hi, ru, ur and zh-CN while English reads
``{source}`` first. The sentence still *parsed*, so nothing crashed — but it said the
wrong thing: on a recall the two devices swap roles, and a reader who takes "the work
on the destination needs your decision" as "the work on the machine I am looking at"
approves a move that leaves their files somewhere else. The four locales were held out
of ``shipped.json`` for it (commit a02a25449's evidence).

CHECK.PY ALREADY REFUSES A REORDER (``_arguments_problem``: "argument N is … (renamed or
reordered)"), so this file is not the gate. It is the fence: it fails in a unit run
that does not shell out to the checker, and it names the four locales that shipped with
the defect rather than any catalogue that happens to be in the tree.

WHAT IT DOES NOT DO, stated so a green run is not over-read: it pins argument ORDER and
presence, never the prose. Whether a translation still MEANS the right thing is the
independent audit's job, and the receipts for the copy this round changed are the next
audit round's work — a locale's ledger row is stale the moment its English bytes move.
"""

from __future__ import annotations

import re

import pytest

from local_operator.i18n import catalogues

#: Every locale the namespace is carried in. ``en`` is the argument source.
LOCALES = ("en", "fr", "es", "vi", "zh-CN", "ru", "ur", "hi")

NAMESPACE = "cli.sessions_move_workset"
TUI_NAMESPACE = "tui.session_move_workset"

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def _order(text: str) -> tuple[str, ...]:
    """Argument names in FIRST-occurrence order, repeats collapsed: a translation may
    legitimately repeat a placeholder ("update lop on {device}"), but it may not
    introduce one before the English does."""
    return tuple(dict.fromkeys(_PLACEHOLDER.findall(text)))


@pytest.mark.parametrize("locale", LOCALES)
def test_desktop_refused_keeps_the_source_before_the_device(locale: str) -> None:
    """The audit's MAJOR: four locales answered "{device} before {source}"."""
    english = catalogues.load_catalogue("en", NAMESPACE)[f"{NAMESPACE}.desktop_refused"]
    carried = catalogues.load_catalogue(locale, NAMESPACE)[f"{NAMESPACE}.desktop_refused"]
    assert _order(carried) == _order(english) == ("source", "device"), carried


@pytest.mark.parametrize("locale", LOCALES)
def test_the_changed_keys_carry_the_same_arguments_everywhere(locale: str) -> None:
    """Every key this round added or reworded, in all eight locales.

    A missing key is invisible until a user in that language meets the sentence, so the
    pin is a plain presence assertion over the exact keys the round touched.
    """
    english = catalogues.load_catalogue("en", NAMESPACE)
    carried = catalogues.load_catalogue(locale, NAMESPACE)
    for key in (
        f"{NAMESPACE}.unknown",
        f"{NAMESPACE}.unknown_card",
        f"{NAMESPACE}.unsupported",
        f"{NAMESPACE}.unsupported_card",
        f"{NAMESPACE}.probe_asked",
    ):
        assert key in carried, key
        assert _order(carried[key]) == _order(english[key]), (key, carried[key])
    english_tui = catalogues.load_catalogue("en", TUI_NAMESPACE)
    carried_tui = catalogues.load_catalogue(locale, TUI_NAMESPACE)
    key = f"{TUI_NAMESPACE}.title_blocked"
    assert key in carried_tui, key
    assert _order(carried_tui[key]) == _order(english_tui[key]) == ()


@pytest.mark.parametrize("locale", LOCALES)
def test_the_card_summaries_do_not_name_the_cli_flag(locale: str) -> None:
    """U12: on a card that HAS the "Move without the working files" button, the summary
    pointed at ``--without-working-set`` instead. The ``*_card`` keys are the card's own
    wording and take the button label; the flag stays in the CLI-only keys."""
    carried = catalogues.load_catalogue(locale, NAMESPACE)
    for key in (f"{NAMESPACE}.unknown_card", f"{NAMESPACE}.unsupported_card"):
        assert "--without-working-set" not in carried[key], (key, carried[key])
        assert "{button}" in carried[key], (key, carried[key])
