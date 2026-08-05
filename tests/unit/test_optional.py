"""The extras registry and its degradation messages.

`optional.py` is the module that decides what a user sees when a feature they
did not install is reached. It had no test coverage at all, which meant the
message wording, the EXTRAS/pyproject sync guard and require_extra's exception
translation were entirely unexercised — and a typo there ships an install
command that does not work.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

from local_operator.optional import (
    EXTRAS,
    MissingExtraError,
    missing_extra_error,
    require_extra,
)

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _declared_extras() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    return set(data["project"]["optional-dependencies"])


def test_every_registry_extra_is_declared_in_pyproject() -> None:
    """A hint naming an extra that does not exist is worse than no hint: the
    user runs the command, pip errors, and they distrust the next message."""
    undeclared = sorted(EXTRAS.keys() - _declared_extras())
    assert not undeclared, f"EXTRAS names extras pyproject does not declare: {undeclared}"


def test_every_installable_extra_has_a_registry_description() -> None:
    """The reverse drift: a new extra with no description here cannot produce a
    degradation message, so its failure mode is a raw ImportError. `dev` is
    excluded — it is a contributor convenience, never reached at runtime."""
    missing = sorted(_declared_extras() - EXTRAS.keys() - {"dev"})
    assert not missing, f"pyproject declares extras with no EXTRAS entry: {missing}"


def test_message_names_the_feature_and_a_runnable_command() -> None:
    message = missing_extra_error("server", "The HTTP API server")
    assert "The HTTP API server" in message
    assert '"server" extra' in message
    # The requirement MUST be quoted: zsh (the macOS default) glob-expands
    # brackets, so an unquoted `pip install local-operator[server]` fails with
    # "no matches found" and the hint is actively misleading.
    assert 'pip install "local-operator[server]"' in message


def test_unknown_extra_raises_instead_of_asserting() -> None:
    """An `assert` here would be stripped by `python -O`, removing the drift
    guard in exactly the runs where a wrong hint is hardest to spot."""
    with pytest.raises(KeyError):
        missing_extra_error("not-a-real-extra", "Something")


def test_unknown_extra_guard_survives_optimization() -> None:
    """Prove the guard is not an assert, by checking the compiled function has
    no assert opcode rather than by trusting the source."""
    import dis

    names = {
        instruction.argval
        for instruction in dis.get_instructions(missing_extra_error)
        if instruction.opname == "LOAD_GLOBAL"
    }
    assert "AssertionError" not in names


def test_require_extra_returns_the_module_when_present() -> None:
    module = require_extra("json", "server", "Anything")
    assert module is sys.modules["json"]


def test_require_extra_translates_a_missing_module() -> None:
    """The whole point: a MissingExtraError callers can distinguish from a
    genuine decode/runtime failure, carrying the actionable text."""
    with pytest.raises(MissingExtraError) as excinfo:
        require_extra("a_module_that_does_not_exist_xyz", "mcp", "MCP support")
    message = str(excinfo.value)
    assert "MCP support" in message
    assert 'pip install "local-operator[mcp]"' in message


def test_missing_extra_error_is_an_import_error() -> None:
    """Subclassing ImportError keeps `except ImportError` call sites working
    while letting newer ones catch the specific type."""
    assert issubclass(MissingExtraError, ImportError)


@pytest.mark.parametrize("extra", sorted(EXTRAS))
def test_every_message_is_one_line_and_mentions_pip(extra: str) -> None:
    """These land in a TUI notice and a CLI warning, both of which assume one
    line; an embedded newline would break the row accounting."""
    message = missing_extra_error(extra, "Feature")
    assert "\n" not in message
    assert re.search(r'pip install "local-operator\[' + re.escape(extra) + r'\]"', message)
