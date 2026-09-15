"""Tests for the jedi-backed LSP tool.

Skipped wholesale when jedi (the ``lsp`` extra) is absent: the tool is not
advertised on such a host, so there is nothing to exercise. Everything here
runs against a real fixture tree analysed by real jedi — mocking jedi would
test our assumptions, not the integration.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.harness.types import ToolContext, ToolResult
from local_operator.tools import lsp

pytestmark = pytest.mark.skipif(lsp.jedi is None, reason="jedi extra not installed")

LIB_SOURCE = (
    "def process(items):\n"
    '    """Process items."""\n'
    "    return [i for i in items if i]\n"
    "\n"
    "\n"
    "class Handler:\n"
    "    def __init__(self, name):\n"
    "        self.name = name\n"
    "\n"
    "    def handle(self, item):\n"
    "        return process([item])\n"
)

MAIN_SOURCE = (
    "from pkg.lib import process, Handler\n"
    "\n"
    "\n"
    "def main():\n"
    '    h = Handler("x")\n'
    "    return process([1, 2]) + h.handle(3)\n"
)


@pytest.fixture()
def tree(tmp_path):
    """A two-module package: main.py imports and calls lib.py symbols."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "lib.py").write_text(LIB_SOURCE)
    (pkg / "main.py").write_text(MAIN_SOURCE)
    return tmp_path


@pytest.fixture()
def tool() -> Any:
    built = lsp.build_lsp_tool()
    assert built is not None, "jedi is installed, so the builder must advertise the tool"
    return built


def _ctx(tree) -> ToolContext:
    return ToolContext(cwd=str(tree))


def _run(tool, tree, args: dict[str, Any]) -> Any:
    return asyncio.run(tool.execute("call-1", args, context=_ctx(tree)))


# --- builder -----------------------------------------------------------------


def test_builder_shape(tool) -> None:
    assert tool.name == "lsp"
    assert tool.label == "LSP"
    assert tool.approval_tier == "read"
    assert tool.concurrency == "shared"
    assert tool.interruptible is False
    actions = tool.parameters["properties"]["action"]["enum"]
    assert set(actions) == {"definitions", "references", "symbols", "rename_preview"}


def test_builder_hides_tool_without_jedi(monkeypatch) -> None:
    monkeypatch.setattr(lsp, "jedi", None)
    assert lsp.build_lsp_tool() is None


# --- definitions -------------------------------------------------------------


def test_definitions_jump_across_modules_by_name(tool, tree) -> None:
    """The headline capability: from an IMPORTED usage, resolve the definition
    in the other module — the thing grep cannot do without lying about
    shadowing."""
    result = _run(tool, tree, {"action": "definitions", "path": "pkg/main.py", "name": "process"})
    assert not result.is_error, result.text
    assert "pkg/lib.py:1" in result.text
    assert "def process(items):" in result.text


def test_definitions_from_a_bare_line_scan_the_identifiers(tool, tree) -> None:
    """line=6 names a call site; column is unknown. Column 0 lands on
    ``return`` (a dead end — measured), so the identifier scan must rescue
    the call."""
    result = _run(tool, tree, {"action": "definitions", "path": "pkg/main.py", "line": 6})
    assert not result.is_error, result.text
    assert "pkg/lib.py:1" in result.text
    assert "def process(items):" in result.text


def test_definitions_honour_an_explicit_column(tool, tree) -> None:
    """Column 11 on line 6 is exactly the ``process`` usage; no scanning."""
    result = _run(
        tool,
        tree,
        {"action": "definitions", "path": "pkg/main.py", "line": 6, "column": 11},
    )
    assert not result.is_error, result.text
    assert "pkg/lib.py:1" in result.text


# --- references --------------------------------------------------------------


def test_references_find_call_sites_the_model_would_grep_for(tool, tree) -> None:
    result = _run(tool, tree, {"action": "references", "path": "pkg/lib.py", "name": "process"})
    assert not result.is_error, result.text
    # The definition, the same-module call, the import and the cross-module
    # call — all four rows carry file:line:text.
    assert "pkg/lib.py:1: def process(items):" in result.text
    assert "pkg/main.py:1: from pkg.lib import process, Handler" in result.text
    assert "pkg/main.py:6:     return process([1, 2]) + h.handle(3)" in result.text


def test_references_from_a_usage_line(tool, tree) -> None:
    result = _run(tool, tree, {"action": "references", "path": "pkg/main.py", "line": 6})
    assert not result.is_error, result.text
    assert "pkg/lib.py:1" in result.text
    assert "pkg/main.py:6" in result.text


def test_references_cap_display_and_spill_the_rest(tool, tree) -> None:
    """260 call sites: 200 ride the prompt, the rest sit behind a spill handle
    — the same two-cap contract as grep."""
    big = tree / "pkg" / "big.py"
    big.write_text("def tick():\n    return 1\n")
    calls = ["from pkg.big import tick", "", "", "def run():"]
    calls += [f"    assert tick() == {i}" for i in range(260)]
    (tree / "pkg" / "calls.py").write_text("\n".join(calls) + "\n")
    result = _run(tool, tree, {"action": "references", "path": "pkg/big.py", "name": "tick"})
    assert "262 reference(s)" in result.text
    # Both caps bite: the item cap holds the display at <=200 rows and the
    # char budget elides within those, while the spill keeps all 262.
    assert result.text.count("pkg/calls.py:") <= lsp.LSP_REFERENCE_LIMIT
    assert result.details and "spill" in result.details


# --- symbols -----------------------------------------------------------------


def test_symbols_outline_lists_definitions_with_line_ranges(tool, tree) -> None:
    result = _run(tool, tree, {"action": "symbols", "path": "pkg/lib.py"})
    assert not result.is_error, result.text
    assert "L1-3 function process" in result.text
    assert "L6-11 class Handler" in result.text
    # Methods nest under their class.
    assert "  L10-11 function handle" in result.text
    # Params and locals are not navigational symbols.
    assert "items" not in result.text
    assert "self" not in result.text


def test_symbols_exclude_imports_and_keep_module_definitions(tool, tree) -> None:
    """main.py imports process/Handler; an outline that listed them as local
    definitions would misdirect the very navigation it serves."""
    result = _run(tool, tree, {"action": "symbols", "path": "pkg/main.py"})
    assert not result.is_error, result.text
    assert "function main" in result.text
    assert "process" not in result.text
    assert "Handler" not in result.text


# --- rename_preview ----------------------------------------------------------


def test_rename_preview_diffs_both_files_and_writes_neither(tool, tree) -> None:
    before_lib = (tree / "pkg" / "lib.py").read_text()
    before_main = (tree / "pkg" / "main.py").read_text()
    result = _run(
        tool,
        tree,
        {
            "action": "rename_preview",
            "path": "pkg/lib.py",
            "name": "process",
            "new_name": "process_all",
        },
    )
    assert not result.is_error, result.text
    assert "--- pkg/lib.py" in result.text
    assert "+++ pkg/lib.py" in result.text
    assert "--- pkg/main.py" in result.text
    assert "+++ pkg/main.py" in result.text
    assert "+def process_all(items):" in result.text
    assert "+        return process_all([item])" in result.text
    assert "+from pkg.lib import process_all, Handler" in result.text
    assert "preview only, nothing written" in result.text
    # READ-ONLY is the contract that keeps this tool in the read tier.
    assert (tree / "pkg" / "lib.py").read_text() == before_lib
    assert (tree / "pkg" / "main.py").read_text() == before_main


def test_rename_preview_from_a_usage_line(tool, tree) -> None:
    """line=6 is a CALL site in the importing module; the rename must still
    resolve the definition in lib.py and propose edits in both files."""
    result = _run(
        tool,
        tree,
        {
            "action": "rename_preview",
            "path": "pkg/main.py",
            "line": 6,
            "new_name": "process_v2",
        },
    )
    assert not result.is_error, result.text
    assert "pkg/main.py" in result.text
    assert "pkg/lib.py" in result.text
    assert "+def process_v2(items):" in result.text


def test_rename_preview_requires_new_name(tool, tree) -> None:
    result = _run(tool, tree, {"action": "rename_preview", "path": "pkg/lib.py", "name": "process"})
    assert result.is_error
    assert "new_name" in result.text


def test_rename_preview_rejects_a_non_identifier_new_name(tool, tree) -> None:
    result = _run(
        tool,
        tree,
        {
            "action": "rename_preview",
            "path": "pkg/lib.py",
            "name": "process",
            "new_name": "not an identifier",
        },
    )
    assert result.is_error
    assert "identifier" in result.text


# --- errors ------------------------------------------------------------------


def test_unknown_symbol_names_candidates(tool, tree) -> None:
    result = _run(tool, tree, {"action": "definitions", "path": "pkg/lib.py", "name": "proces"})
    assert result.is_error
    assert "proces" in result.text
    assert "available symbols:" in result.text
    assert "process" in result.text
    assert "Handler" in result.text


def test_no_symbol_on_a_blank_line_suggests_the_recovery_routes(tool, tree) -> None:
    result = _run(tool, tree, {"action": "definitions", "path": "pkg/main.py", "line": 2})
    assert result.is_error
    assert "no symbol found at pkg/main.py:2" in result.text
    assert "name=" in result.text
    assert "action=symbols" in result.text


def test_missing_path_is_a_clean_error(tool, tree) -> None:
    result = _run(tool, tree, {"action": "symbols", "path": "pkg/nope.py"})
    assert result.is_error
    assert "does not exist" in result.text


def test_line_or_name_is_required_for_positional_actions(tool, tree) -> None:
    result = _run(tool, tree, {"action": "definitions", "path": "pkg/lib.py"})
    assert result.is_error
    assert "requires line or name" in result.text


def test_line_is_one_based(tool, tree) -> None:
    result = _run(tool, tree, {"action": "definitions", "path": "pkg/lib.py", "line": 0})
    assert result.is_error
    assert "1-based" in result.text


def test_invalid_action_is_a_validation_error(tool, tree) -> None:
    result = _run(tool, tree, {"action": "hover", "path": "pkg/lib.py"})
    assert result.is_error
    assert "invalid arguments" in result.text


def test_jedi_internal_failure_becomes_an_error_result(tool, tree, monkeypatch) -> None:
    """The loop never sees an exception from a tool — the _guard contract."""

    class ExplodingScript:
        def __init__(self, *, path=None):
            self.path = path

        def get_names(self, **_kwargs):
            raise RuntimeError("jedi exploded")

        def goto(self, *_args, **_kwargs):
            raise RuntimeError("jedi exploded")

        def get_references(self, *_args, **_kwargs):
            raise RuntimeError("jedi exploded")

        def rename(self, *_args, **_kwargs):
            raise RuntimeError("jedi exploded")

    monkeypatch.setattr(lsp.jedi, "Script", ExplodingScript)
    result = _run(tool, tree, {"action": "symbols", "path": "pkg/lib.py"})
    assert result.is_error
    assert "failed unexpectedly" in result.text
    assert "jedi exploded" in result.text


@pytest.mark.asyncio
async def test_outside_initial_path_requires_approval(tool, tree, tmp_path) -> None:
    outside = tmp_path.parent / "outside-lsp.py"
    outside.write_text("def secret():\n    return 1\n")
    requests: list[str] = []

    async def deny(tool_name: str, description: str) -> bool:
        requests.append(description)
        return False

    context = ToolContext(cwd=str(tree), request_approval=deny)
    result = await tool.execute(
        "c", {"action": "symbols", "path": str(outside)}, None, None, context
    )
    assert result.is_error is True
    assert "declined" in result.text.lower()
    assert requests and str(outside) in requests[0]


@pytest.mark.asyncio
async def test_outside_initial_path_works_only_after_approval(tool, tree, tmp_path) -> None:
    outside = tmp_path.parent / "approved-lsp.py"
    outside.write_text("def allowed():\n    return 1\n")

    async def approve(*_args: Any) -> bool:
        return True

    context = ToolContext(cwd=str(tree), request_approval=approve)
    result = await tool.execute(
        "c", {"action": "symbols", "path": str(outside)}, None, None, context
    )
    assert result.is_error is False
    assert "allowed" in result.text


def test_result_path_filter_allows_initial_or_cwd_only(tree, tmp_path) -> None:
    from local_operator.tools.lsp import _allowed_source

    initial = tree / "pkg" / "main.py"
    assert _allowed_source(initial, tree, initial) is True
    assert _allowed_source(tree / "pkg" / "lib.py", tree, initial) is True
    assert _allowed_source(tmp_path.parent / "secret.py", tree, initial) is False


# --- optional-dependency resolution ------------------------------------------


def test_availability_is_answered_without_importing(monkeypatch) -> None:
    """The builder's question is answered by ``find_spec``, not by an import.

    This is the change in one assertion. ``tools/registry.py`` imports this
    module eagerly to fill its factory table, so a module-scope ``import
    jedi`` put jedi's whole inference graph — 108 ms measured, against ~600 ms
    for ``create_session`` with its own imports pre-warmed — on every session
    construction, whether or not the model ever asked a symbol question. On the
    desktop plane a session construction is a fresh runtime child, so that is
    every attach.

    ``jedi`` absent from the module globals AFTER the probe is the part that
    distinguishes the two implementations: an import-based probe would have
    written it there (``_jedi`` caches on resolution).
    """
    monkeypatch.delattr(lsp, "jedi", raising=False)
    assert lsp._jedi_installed() is True
    assert (
        "jedi" not in lsp.__dict__
    ), "_jedi_installed executed the import; it must resolve the spec only"


def test_an_absent_extra_hides_the_tool_and_answers_in_words(monkeypatch, tree) -> None:
    """``lsp.jedi = None`` is the state a host without the extra resolves to.

    Two user-visible consequences, and the second is the one that needs the
    lazy loader rather than a bare ``if jedi is None``: the tool is not
    advertised, and a call that arrives anyway — an older tool list, a
    registry built before the extra was uninstalled — gets a sentence rather
    than an AttributeError out of the execute path.
    """
    monkeypatch.setattr(lsp, "jedi", None)
    assert lsp.build_lsp_tool() is None

    async def _call() -> ToolResult:
        # Awaited inside a coroutine rather than handed straight to
        # ``asyncio.run``: ``execute_lsp`` carries ``@_guard``, whose declared
        # return is ``Awaitable[ToolResult]`` rather than ``Coroutine``, and
        # ``asyncio.run`` only accepts the latter. This is what every
        # production call site does too.
        return await lsp.execute_lsp(
            "call-1", {"action": "symbols", "path": "pkg/lib.py"}, None, None, _ctx(tree)
        )

    result = asyncio.run(_call())
    assert result.is_error is True
    assert "jedi is not installed" in result.text


def test_the_module_attribute_stays_resolvable(monkeypatch) -> None:
    """``lsp.jedi`` and ``lsp.RefactoringError`` keep working lazily.

    PEP 562 exists here for one reason worth pinning: this module's own suite
    reads ``lsp.jedi`` at COLLECTION time (the file-level ``skipif``), and other
    callers monkeypatch both names. A lazy import that broke attribute access
    would fail the whole file at collection rather than one test, which is a
    confusing way to learn that the import moved.
    """
    monkeypatch.delattr(lsp, "jedi", raising=False)
    monkeypatch.delattr(lsp, "RefactoringError", raising=False)
    resolved = lsp.jedi
    assert resolved is not None, "jedi is installed in this suite, so the probe finds it"
    assert lsp.RefactoringError is resolved.RefactoringError
    with pytest.raises(AttributeError):
        lsp.not_a_declared_attribute


def test_the_exception_class_falls_back_when_jedi_is_absent(monkeypatch) -> None:
    """The rename path's ``except`` needs SOMETHING to name when jedi is gone.

    ``_rename_preview_result`` skips a candidate position by catching
    ``jedi.RefactoringError``. That path is unreachable without jedi, and the
    honest stand-in is a class nothing raises — never ``Exception``, which
    would silently swallow the tool's real failures if the path ever became
    reachable.
    """
    monkeypatch.setattr(lsp, "jedi", None)
    monkeypatch.delattr(lsp, "RefactoringError", raising=False)
    assert lsp._refactoring_error() is lsp._NeverRaised
    assert not issubclass(lsp._NeverRaised, (ValueError, RuntimeError))


def test_jedi_is_resolved_once(monkeypatch) -> None:
    """One import, cached in the module global, for a whole process."""
    monkeypatch.delattr(lsp, "jedi", raising=False)
    first = lsp._jedi()
    assert first is not None
    assert lsp._jedi() is first
