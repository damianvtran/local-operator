"""Arms for the escalation exemption of the guard's own source and corpus.

The module under test decides ONE thing — whether a call is a read of the guard's
own area — and these arms pin both halves of that decision. The positive half is
the operator's request; the negative half is the security requirement, and it is
the half that decides whether the exemption ships: an agent must not be able to
confer the exemption on itself by naming a path in a command, a pattern, or any
other text. Only a file-reading tool's own ``path`` argument may.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.harness.guard_area import (
    EXEMPT_SOURCES,
    READING_TOOLS,
    reads_exempt_source,
    source_is_exempt,
)
from local_operator.harness.guard_area import exempt_from_escalation as exempt

#: The two files the operator named, spelled from this checkout's root. Derived
#: from the module's own package root rather than from the CWD so the arms hold
#: whatever directory pytest was started in.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "local_operator"
SHAPE_TABLE = _PACKAGE_ROOT / "redaction_shapes.py"
CORPUS = _PACKAGE_ROOT.parent / "tests" / "unit" / "secrets" / "credential_shape_corpus.py"


def test_the_exempt_set_is_the_guards_source_and_corpus_and_nothing_else() -> None:
    assert EXEMPT_SOURCES == {SHAPE_TABLE.resolve(), CORPUS.resolve()}
    # No tree and no class rule: a sibling test module and the rest of the
    # package are NOT exempt, which is the broader version the operator declined.
    assert reads_exempt_source("read", {"path": "tests/unit/secrets/conftest.py"}) is False
    assert reads_exempt_source("read", {"path": "tests/unit/secrets/"}) is False
    assert reads_exempt_source("read", {"path": "local_operator/"}) is False
    assert reads_exempt_source("read", {"path": "local_operator/incidents.py"}) is False


@pytest.mark.parametrize("tool", sorted(READING_TOOLS))
def test_each_reading_tool_is_exempt_for_either_file(tool: str) -> None:
    assert reads_exempt_source(tool, {"path": str(SHAPE_TABLE)}) is True
    assert reads_exempt_source(tool, {"path": str(CORPUS)}) is True


def test_a_relative_spelling_resolves_the_same_way(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(SHAPE_TABLE.parent.parent)
    assert reads_exempt_source("read", {"path": "local_operator/redaction_shapes.py"}) is True
    assert reads_exempt_source("read", {"path": "./local_operator/redaction_shapes.py"}) is True


def test_a_bash_command_that_merely_names_the_path_is_not_exempt() -> None:
    """THE SECURITY ARM. Naming the path in a command confers nothing.

    ``cat``, a python one-liner and a shell pipeline all read the file in fact,
    and all of them are refused: the exemption may only be conferred by a
    file-reading tool's STRUCTURED path argument, because a command string is
    text an agent (or a file the agent read) can steer.
    """
    commands = (
        f"cat {SHAPE_TABLE}",
        f"sed -n '1,20p' {CORPUS}",
        f"python -c \"print(open('{SHAPE_TABLE}').read())\"",
        f"grep -h . {CORPUS} | head",
        f"cat {SHAPE_TABLE} > /dev/null # credential-shape corpus",
    )
    for command in commands:
        assert reads_exempt_source("bash", {"command": command}) is False, command


def test_text_that_names_the_path_elsewhere_is_not_exempt() -> None:
    """A pattern, a URL, an edit target and an excerpt are all just text."""
    assert reads_exempt_source("grep", {"pattern": str(CORPUS)}) is False
    assert reads_exempt_source("grep", {"pattern": "redaction_shapes", "path": "tests/"}) is False
    assert reads_exempt_source("read", {"path": f"skill://guard-area/{CORPUS.name}"}) is False
    assert reads_exempt_source("read", {"path": f"spill://{CORPUS.name}"}) is False
    assert reads_exempt_source("web_read", {"url": f"https://example.invalid/{CORPUS}"}) is False


def test_a_path_that_only_resembles_the_exempt_one_is_refused() -> None:
    """Exact resolved equality, not a suffix or substring match."""
    decoys = (
        SHAPE_TABLE.parent / "notes" / SHAPE_TABLE.name,
        SHAPE_TABLE.parent.parent / "vendor" / SHAPE_TABLE.name,
        Path(str(SHAPE_TABLE) + ".bak"),
        SHAPE_TABLE.parent / "redaction_shapes.pyc",
    )
    for decoy in decoys:
        assert reads_exempt_source("read", {"path": str(decoy)}) is False, decoy


def test_a_call_without_a_usable_path_is_not_exempt() -> None:
    assert reads_exempt_source("read", None) is False
    assert reads_exempt_source("read", {}) is False
    assert reads_exempt_source("read", {"path": ""}) is False
    assert reads_exempt_source("read", {"path": 42}) is False
    assert reads_exempt_source("", {"path": str(CORPUS)}) is False


def test_the_publication_is_per_call_and_resets() -> None:
    """Unset means ESCALATE: the honest default for a surface nobody published."""
    assert source_is_exempt() is False
    with exempt("read", {"path": str(CORPUS)}) as published:
        assert published is True
        assert source_is_exempt() is True
        # A nested call replaces the answer and restores it on exit, so an
        # unrelated tool reading nothing exempt cannot inherit the outer verdict.
        with exempt("bash", {"command": f"cat {CORPUS}"}) as inner:
            assert inner is False
            assert source_is_exempt() is False
        assert source_is_exempt() is True
    assert source_is_exempt() is False
