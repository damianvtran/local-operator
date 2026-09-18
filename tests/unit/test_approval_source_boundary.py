"""``source="local"`` means "an operator write through the settings facade".

Issue #1282: ``tool_approval_mode`` is LIVE, and the party the gate constrains
can write ``config.yml`` itself. The gate therefore authorises a LOOSENING on
the SOURCE of the write rather than on the value
(:func:`local_operator.harness.approval.loosening_is_authorised`), and one value
of that source carries the whole rule: ``"local"`` is a write this process made
through the operator's own facade (``settings_io`` ->
``config_watch.notify_local``), in the process holding the gate.

The equivalence is a property of the CODEBASE, not of any one call site: it
holds only while nothing reachable from a model's tool call can write through
that facade. The moment a tool, a subagent path, an MCP server surface or the
browser bridge can call ``settings_io``, ``"local"`` stops meaning "the
operator" and the gate can be loosened by the thing it gates — with no code
change anywhere near the gate.

Hence a source scan, in the idiom of
``tests/unit/test_config.py::test_the_migration_has_exactly_one_caller_and_marking_has_two``
and of ``tests/unit/test_agent_import_boundary.py``: a behavioural probe can
only catch a write path someone thought to exercise, and this one fails on the
IMPORT. The alternative spellings of the same hole are checked together —
importing the facade, and calling one of its writers through a local alias — so
the test does not depend on the import being the funnel.

The agent-facing trees are named POSITIVELY, so a new directory has to be
considered rather than silently scanned: ``tools`` and ``harness`` (what a model
call reaches), ``mcp`` (what a connected server reaches), and ``browser_bridge``
(what the shipped extension reaches on this side — it has no filesystem, and its
config access is the daemon's HTTP settings API, in a process that holds no
gate and is therefore unattributed for exactly that reason).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Trees a model-run call can reach, relative to the repository root.
_AGENT_FACING_TREES = (
    "local_operator/tools",
    "local_operator/harness",
    "local_operator/mcp",
    "local_operator/browser_bridge",
)

#: The facade's writers, public and private. ``_store``/``_delete`` are its
#: primitives and ``_notify_watcher`` is the call that MAKES a write attributed,
#: so a caller of any of them is a caller of the facade whether or not it
#: imported the module by name.
_FACADE_WRITERS = frozenset(
    {"write_setting", "reset_setting", "_store", "_delete", "_notify_watcher"}
)

_TESTS_ROOT = Path(__file__).resolve().parents[2]


def _referenced_names(tree: ast.AST) -> set[str]:
    """Every name this module imports from or calls on the settings facade.

    Both imported NAMES and the source MODULE are collected, because the two
    spellings of the same reach are different AST nodes: ``import
    local_operator.settings_io`` puts it in ``Import.alias.name``, while ``from
    local_operator import settings_io`` puts ``local_operator`` there and
    ``settings_io`` in the alias list. Collecting only the module would miss
    every call site in this codebase (``local_operator/tui/app.py`` uses the
    ``from`` form throughout), and a pin with that hole is worse than none: it
    would pass on the change it exists to catch. ``asname`` is collected too, so
    aliasing the facade does not hide it.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
            names.update(alias.asname for alias in node.names if alias.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(alias.name for alias in node.names)
            names.update(alias.asname for alias in node.names if alias.asname)
        elif isinstance(node, ast.Call):
            func = node.func
            names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
    return names


def _facade_reach(module: Path) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    names = _referenced_names(tree)
    reach = {name for name in names if name.endswith("settings_io")}
    reach |= names & _FACADE_WRITERS
    return reach


@pytest.mark.parametrize("tree", _AGENT_FACING_TREES)
def test_no_agent_facing_module_can_write_through_the_settings_facade(tree: str) -> None:
    """The gate may only loosen on the operator's own writes — enforce the "own".

    A hit here is not necessarily a bug on the day it appears (the assertion
    message says how to qualify a legitimately unrelated ``_store``); it is a
    hit on the boundary that makes ``source="local"`` trustworthy, and the fix is
    to route the write through a human-facing surface rather than to widen the
    gate.
    """
    offenders: dict[str, set[str]] = {}
    for module in sorted((_TESTS_ROOT / tree).rglob("*.py")):
        reach = _facade_reach(module)
        if reach:
            offenders[module.relative_to(_TESTS_ROOT).as_posix()] = reach
    assert offenders == {}, (
        f"{tree} can reach settings_io: {offenders}. The approval gate authorises a "
        "LOOSENING on source='local', which means 'a write the operator made through the "
        "facade in the process holding the gate' — a module under an agent-facing tree that "
        "can call the facade breaks that equivalence. If a name hit here is an unrelated "
        "object's `_store`/`_delete`, qualify the check to the settings_io module rather "
        "than deleting it."
    )
