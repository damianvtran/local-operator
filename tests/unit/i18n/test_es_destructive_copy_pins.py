"""Pins for the Spanish copy of commands that DESTROY something.

The es W2 audit raised two blockers of the highest-severity class a translation
can produce: strings that invert a destructive command. `recuperar` ("get back")
was used for a cleanup that deletes scratchpad build/cache content and for a dry
run of runtime termination — a reader planning around that promise can lose data
they expected to keep, and a reviewer reading the Spanish alone cannot tell it is
wrong.

An audit is a moment; the catalogue is forever. These pins make the *shipped*
Spanish say what the command does, so a later wording change, a re-translation,
or a copied phrase cannot quietly restore the inversion:

* the cleanup key must carry a removal verb and must not carry a recovery verb;
* the prune key must keep the narrow restriction ("nothing is executed from it"),
  not the broader "not in use";
* the product noun stays `scratchpad` (the en sidecar forbids "pad").

What these pins do NOT do, stated so nobody over-reads them: they do not prove
the Spanish is idiomatic, and they cannot catch a *different* inversion worded
with the right verbs. They are a regression fence around a known failure, not a
substitute for the independent audit.
"""

from __future__ import annotations

import pytest

from local_operator.i18n import catalogues

#: Words that promise the reader gets something back. None may appear in a
#: string that describes a deletion.
RECOVERY_VERBS = ("recuper", "restaur", "restablec", "devolv")

#: Words that say content is removed. At least one must appear.
REMOVAL_VERBS = ("elimin", "borr", "liber", "suprim")


@pytest.fixture(scope="module")
def sessions_help():
    return catalogues.load_catalogue("es", "cli.sessions_help")


@pytest.fixture(scope="module")
def install_help():
    return catalogues.load_catalogue("es", "cli.install_help")


def test_cleanup_scratchpads_only_promises_deletion_not_recovery(sessions_help):
    value = sessions_help["cli.sessions_help.cleanup_scratchpads_only"].lower()
    assert "scratchpad" in value, "the product noun must survive"
    assert any(verb in value for verb in REMOVAL_VERBS), (
        "the cleanup key must say the content is removed; "
        f"got {sessions_help['cli.sessions_help.cleanup_scratchpads_only']!r}"
    )
    for verb in RECOVERY_VERBS:
        assert verb not in value, (
            f"{verb!r} promises recovery in a key that deletes; "
            f"got {sessions_help['cli.sessions_help.cleanup_scratchpads_only']!r}"
        )
    # The en sidecar's explicit instruction: "scratchpad" is the product noun,
    # never "pad".
    assert " pads" not in f" {value}", "the audit rejected 'pad' for 'scratchpad'"


def test_reclaim_dry_run_describes_termination_not_recovery(sessions_help):
    value = sessions_help["cli.sessions_help.reclaim_dry_run"].lower()
    assert "señal" in value, "the dry-run promise (nothing is signalled) must survive"
    for verb in RECOVERY_VERBS:
        assert verb not in value, (
            f"{verb!r} promises recovery in a dry run of termination; "
            f"got {sessions_help['cli.sessions_help.reclaim_dry_run']!r}"
        )


def test_prune_keeps_the_narrow_nothing_is_running_restriction(install_help):
    value = install_help["cli.install_help.prune"].lower()
    assert "no se ejecuta nada" in value, (
        "prune deletes exactly the generations nothing runs from; the broader "
        f"'not in use' was a blocker. got {install_help['cli.install_help.prune']!r}"
    )
    assert "en uso" not in value, (
        "'en uso' is the broader restriction the audit rejected; "
        f"got {install_help['cli.install_help.prune']!r}"
    )


def test_the_pins_can_fail():
    """A pin nobody has seen fail is a comment.

    Mirrors the three assertions above against the exact strings the audit
    rejected, so a typo in the checks themselves would show up here rather than
    silently passing forever.
    """
    inverted = "si no, recuperar solo el contenido de compilación y caché de scratchpad"
    assert any(verb in inverted for verb in RECOVERY_VERBS)
    assert not any(verb in inverted for verb in REMOVAL_VERBS)
    broadened = "eliminar generaciones de instalación que no estén en uso"
    assert "no se ejecuta nada" not in broadened
    assert "en uso" in broadened
