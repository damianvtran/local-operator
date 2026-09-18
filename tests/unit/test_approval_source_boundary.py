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
IMPORT.

WHAT THIS CHECK COVERS, exactly (agent review round 1, m2 — the earlier
docstring claimed more than the code did, which is worse than a narrower pin):

* a static import of the facade, by any of its three spellings, aliased or not;
* a static call of one of its writers, including through an import alias;
* a DYNAMIC reach whose name is a string LITERAL: ``importlib.import_module(
  "local_operator.settings_io")``, ``__import__(...)``, and ``getattr(mod,
  "write_setting")`` / ``hasattr`` — with ``"a" + "b"`` folded, since that is
  the shape a deliberate obfuscation takes.

WHAT IT DOES NOT COVER, stated so that no future reader reads a green run as
more than it is: a reach through a module OUTSIDE these four trees (an inside
module importing a helper that imports the facade — the honest form of this
check is an import closure, which in this codebase would flag every tool that
transitively touches the CLI or the TUI and is therefore not what is asserted
here), and a name assembled at RUNTIME (from input, from ``globals()``). The
pin is a tripwire on the direct, literal reach — the shapes an agent-facing
write path actually takes — and the tuple of trees is asserted non-empty so a
rename cannot turn it into a vacuous pass (n1).

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

#: The facade's module name, matched by SUFFIX so ``local_operator.settings_io``
#: and a bare ``settings_io`` are the same hit.
_FACADE_MODULE = "settings_io"

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
    reach = {name for name in names if name.endswith(_FACADE_MODULE)}
    reach |= names & _FACADE_WRITERS
    reach |= _dynamic_facade_reach(tree)
    return reach


def _literal_string(node: ast.AST) -> str | None:
    """The string this expression evaluates to, when it is built of literals only.

    ``"local_operator." + "settings_io"`` is the shape a dynamic reach takes when
    it is written to get past a name-matching scan, so the concatenation is folded
    rather than treated as an unknown.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left)
        right = _literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


#: Calls that reach a module or an attribute BY NAME, so a string constant IS the
#: reach. ``importlib.util.find_spec``/``spec_from_file_location`` are deliberately
#: not here: they locate a module without importing or calling it, and a pin that
#: flagged them would fail on reflection code that never touches the facade.
_DYNAMIC_REACH_CALLS = frozenset({"import_module", "__import__", "getattr", "hasattr"})


def _dynamic_facade_reach(tree: ast.AST) -> set[str]:
    """Facade names that this module only ever spells as STRINGS (m2).

    ``importlib.import_module("local_operator.settings_io")`` followed by
    ``getattr(mod, "write_setting")`` returns no hit from the name walk, while
    being the same hole in one more step. The string is what names the target, so
    the string is what this collects — and only inside one of those call shapes,
    because a module that merely MENTIONS the facade in prose (``tools/`` has two
    such comments and a section of prose in ``shell_env``) is not a write path.
    """
    reach: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if callee not in _DYNAMIC_REACH_CALLS:
            continue
        for argument in node.args:
            literal = _literal_string(argument)
            if not literal:
                continue
            for part in (literal, literal.rpartition(".")[2]):
                if part == _FACADE_MODULE or part in _FACADE_WRITERS:
                    reach.add(literal)
    return reach


@pytest.mark.parametrize(
    "source",
    [
        "import local_operator.settings_io as sio\nsio.write_setting(m, s, v)\n",
        "from local_operator import settings_io\nsettings_io.write_setting(m, s, v)\n",
        "from local_operator.settings_io import write_setting as w\nw(m, s, v)\n",
        # The dynamic forms agent review round 1 (m2) found UNCAUGHT by the first
        # revision of this pin, which claimed them. A green run has to mean
        # something, so each is asserted rather than trusted.
        "import importlib\nmod = importlib.import_module('local_operator.settings_io')\n",
        "import importlib\nmod = importlib.import_module('local_operator.' + 'settings_io')\n",
        "mod = __import__('local_operator.settings_io')\n",
        "w = getattr(importlib.import_module('local_operator.settings_io'), 'write_setting')\n",
        "assert not hasattr(mod, 'reset_setting')\n",
    ],
)
def test_every_direct_spelling_of_the_reach_is_caught(source: str, tmp_path: Path) -> None:
    module = tmp_path / "probe.py"
    module.write_text(source, encoding="utf-8")
    assert _facade_reach(module), source


@pytest.mark.parametrize(
    "source",
    [
        # Prose is not a write path: `tools/` has comments naming the facade and
        # `tools/shell_env.py` discusses it deliberately. A scan that flagged
        # those would be red on the tree it protects.
        '"""The settings_io row is the one a user edits."""\n',
        "# `settings_io` would be the wrong module here\n",
        "import importlib\nimportlib.import_module('local_operator.config_watch')\n",
        "import importlib\nimportlib.import_module(MODULE_NAME)\n",
        "STORE = open_store()\n",
        "def f(manager):\n    return manager._rows\n",
    ],
)
def test_prose_and_unrelated_names_are_not_a_reach(source: str, tmp_path: Path) -> None:
    """The pin must not fire on the mentions that exist today.

    Without this control, "make the dynamic spelling hit" is satisfiable by
    matching the bare word anywhere — which would fail on the four trees as they
    stand, and the fix for that is never a narrower check but a deleted one.
    """
    module = tmp_path / "probe.py"
    module.write_text(source, encoding="utf-8")
    assert _facade_reach(module) == set(), source


@pytest.mark.parametrize("tree", _AGENT_FACING_TREES)
def test_no_agent_facing_module_can_write_through_the_settings_facade(tree: str) -> None:
    """The gate may only loosen on the operator's own writes — enforce the "own".

    A hit here is not necessarily a bug on the day it appears (the assertion
    message says how to qualify ``getattr``-style reflection that never touches the
    facade); it is a hit on the boundary that makes ``source="local"``
    trustworthy, and the fix is to route the write through a human-facing surface
    rather than to widen the gate.

    The tree is asserted to EXIST with modules in it: a renamed or removed
    surface used to pass this test with an empty offender set, which reads as
    coverage for a surface that is no longer being scanned at all (agent review
    round 1, n1).
    """
    modules = sorted((_TESTS_ROOT / tree).rglob("*.py"))
    assert modules, (
        f"{tree} has no Python modules — the surface moved, was renamed or was deleted, "
        "and this pin would otherwise report coverage for a tree it never read"
    )
    offenders: dict[str, set[str]] = {}
    for module in modules:
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
