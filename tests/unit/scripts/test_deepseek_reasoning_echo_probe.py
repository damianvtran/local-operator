"""The live probe must keep working against a PRE-FIX tree.

``--digest`` is documented as a two-tree comparison: run it under
``PYTHONPATH=<clean origin/main worktree>`` and against this branch, and the
capability-off body must come out with the same sha256. That recipe is only
honest if the script imports nothing a pre-fix tree lacks and switches nothing
off that it does not have -- the round-1 version imported the placeholder
constant at module load and raised ``ImportError`` against ``origin/main``,
turning a comparison into a crash.
"""

from typing import cast

from local_operator.harness.types import ModelSpec
from local_operator.providers.replay import REASONING_ECHO_PLACEHOLDER
from scripts.deepseek_reasoning_echo_probe import (
    FALLBACK_ECHO_PLACEHOLDER,
    echo_placeholder,
    without_echo,
)


def test_the_literal_fallback_matches_the_shipped_constant():
    """Two spellings of one sentence: drift here would make the probe count
    placeholders it did not write (or miss ones it did)."""
    assert FALLBACK_ECHO_PLACEHOLDER == REASONING_ECHO_PLACEHOLDER
    assert echo_placeholder() == REASONING_ECHO_PLACEHOLDER


def test_without_echo_switches_off_exactly_that_field():
    capable = ModelSpec(provider="deepseek", model_id="deepseek-flash")
    capable = capable.model_copy(update={"requires_reasoning_echo": True})

    switched = without_echo(capable)

    assert switched.requires_reasoning_echo is False
    assert switched.model_id == capable.model_id
    assert switched.provider == capable.provider


def test_without_echo_is_a_no_op_for_a_tree_that_has_no_such_field():
    """The pre-fix simulation: a spec type without the field must come back
    untouched rather than gaining a key its own builder would never read.

    ``model_fields`` is the only thing the helper consults, so a stand-in that
    omits that one entry is a pre-fix class as far as the code under test can
    tell. Identity is the assertion: a no-op that returned a copy would still be
    a second way of building the pre-fix body.
    """

    class PreFixSpec:
        model_fields = {
            name: field
            for name, field in ModelSpec.model_fields.items()
            if name != "requires_reasoning_echo"
        }

    legacy = PreFixSpec()

    assert without_echo(cast(ModelSpec, legacy)) is legacy
