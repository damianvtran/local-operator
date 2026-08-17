"""LSP tool — symbol-aware Python navigation and rename previews via jedi.

Why jedi, and not a language server
-----------------------------------
A conventional LSP client needs a per-language server subprocess (pyright,
rust-analyzer, gopsl) that the harness would have to install, spawn and
babysit — the opposite of the lean-install doctrine. jedi is pure Python and
answers in-process: import it, analyse one file, return. That is also why it
ships as the optional ``lsp`` extra rather than a base dependency — a host
without jedi simply does not get the tool (``build_lsp_tool`` returns None,
the same createIf convention as ``build_browser_tool`` in builtin.py).

Every action is read-only. ``rename_preview`` computes jedi's refactoring and
returns its unified diff WITHOUT applying it: the model then makes the change
through the edit tool, which is what keeps this whole tool in the read
approval tier.

Position handling is the one place jedi's API fights the caller. jedi wants a
(line, column) pair, but a model holding a grep hit knows the LINE, not the
column — and measured against jedi 0.20, ``goto(line, 0)`` on a
``def foo(...)`` line returns nothing because column 0 lands on the ``d`` of
``def``. A line-only query therefore scans the identifiers on the line and
tries the first non-keyword one; ``name=`` skips positions entirely and
resolves through jedi's own name analysis.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.tools.builtin import (
    _capped_list_body,
    _error,
    _guard,
    _resolve_workspace_path,
    _safe_cwd,
    _text,
    _validation_error,
    spill_truncate,
)

try:
    import jedi
    from jedi import RefactoringError
except ImportError:  # pragma: no cover — the absent-extra path is covered by
    # build_lsp_tool returning None; jedi-dependent code is unreachable then.
    jedi = None  # type: ignore[assignment]
    RefactoringError = Exception  # type: ignore[assignment,misc]

#: Every action the tool accepts. One tuple so the schema text, the dispatch
#: and the tests cannot drift apart (same convention as BROWSER_ACTIONS).
LSP_ACTIONS = ("definitions", "references", "symbols", "rename_preview")

#: Displayed-reference cap. Mirrors GREP_MATCH_LIMIT: the displayed set
#: protects the prompt, the spill (written from the full set by
#: ``_capped_list_body``) keeps the rest reachable by handle.
LSP_REFERENCE_LIMIT = 200

#: Definition-target cap. ``goto`` on an ordinary symbol returns a handful; a
#: result beyond this is a pathological inference and gets elided with a count
#: rather than dumped in full.
LSP_DEFINITION_LIMIT = 20

#: Outline rows shown per ``symbols`` call. Same two-cap shape as references:
#: a generated module with thousands of definitions must not turn the outline
#: into the blob it exists to avoid; the spill holds the tail.
LSP_SYMBOL_LIMIT = 400

#: Symbols named in an unknown-symbol error. Enough to spot the right one by
#: eye without the error itself becoming output the model has to page past.
LSP_CANDIDATE_LIMIT = 30

#: One identifier, used by the line-only position scan.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class LspParams(BaseModel):
    """Arguments the model sends; field descriptions are its only docs."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["definitions", "references", "symbols", "rename_preview"] = Field(
        description="definitions | references | symbols | rename_preview."
    )
    path: str = Field(description="Python file to analyse (absolute or cwd-relative).")
    line: int | None = Field(
        default=None,
        description="1-based line of the symbol or a usage of it; optional for "
        "symbols, or give name instead.",
    )
    column: int | None = Field(
        default=None,
        description="0-based column on the line; by default the first identifier "
        "on the line is used.",
    )
    name: str | None = Field(
        default=None,
        description="Symbol to locate when the line is unknown; resolved via "
        "jedi's name analysis (definitions in this file, then imports).",
    )
    new_name: str | None = Field(
        default=None,
        description="Proposed new name (rename_preview only). The diff is "
        "returned, never applied.",
    )


def _display_path(path: Path | str, root: Path) -> str:
    """cwd-relative when inside the workspace, absolute otherwise — the same
    rendering grep uses, so the model can paste the result into a follow-up
    read or grep without translating between path styles."""
    resolved = Path(path)
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _source_lines(path: Path | str, cache: dict[str, list[str]]) -> list[str]:
    """Lines of ``path`` from the per-call cache; unreadable files read as
    empty rather than raising — a snippet is decoration, never the answer."""
    key = str(path)
    if key not in cache:
        try:
            cache[key] = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            cache[key] = []
    return cache[key]


def _is_import_line(line_no: int, lines: list[str]) -> bool:
    text = lines[line_no - 1].lstrip() if 0 < line_no <= len(lines) else ""
    return text.startswith(("import ", "from "))


def _outline_names(script: Any, lines: list[str]) -> list[Any]:
    """Definitions worth navigating to, in file order.

    jedi's public API cannot tell an imported name from one defined here — an
    imported ``process`` reports description ``def process`` while sitting on
    the import statement — so the import statement's own line text is the
    filter. Params and locals are dropped by type/indent; module-level
    assignments (column 0 statements) stay because they are the constants a
    navigation pass wants.
    """
    out: list[Any] = []
    seen: set[tuple[str, int, str]] = set()
    for name in script.get_names(all_scopes=True, definitions=True):
        if name.type == "param" or name.line is None or _is_import_line(name.line, lines):
            continue
        start = name.get_definition_start_position()
        column = start[1] if start else (name.column or 0)
        if name.type == "statement" and column > 0:
            continue
        key = (name.name, name.line, name.type)
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    out.sort(key=lambda n: n.get_definition_start_position() or (n.line, n.column or 0))
    return out


def _positions_for_name(script: Any, name: str, lines: list[str]) -> list[tuple[int, int]]:
    """Resolve a bare symbol name to a jedi position.

    Prefers a name DEFINED in this file over one merely imported (an import's
    position works too — goto follows it — but a same-named local definition
    is the likelier target when both exist).
    """
    matches = [
        n
        for n in script.get_names(all_scopes=True, definitions=True)
        if n.name == name and n.line is not None
    ]
    if not matches:
        return []
    defined_here = [n for n in matches if not _is_import_line(n.line, lines)]
    pool = defined_here or matches
    first = min(pool, key=lambda n: (n.line, n.column or 0))
    return [(first.line, first.column or 0)]


def _candidate_positions(params: LspParams, lines: list[str]) -> list[tuple[int, int]]:
    """Positions to try, best first, for a line-bearing query.

    An explicit ``column`` is honoured exactly and alone — the caller claims
    to know where the symbol is. Without one, every non-keyword identifier on
    the line is a candidate in order: column 0 usually lands on ``def`` or
    ``return`` (both dead ends, measured), while the first identifier after
    the keyword is the symbol the caller means.
    """
    if params.line is None:
        return []
    if params.column is not None:
        return [(params.line, max(params.column, 0))]
    text = lines[params.line - 1] if 0 < params.line <= len(lines) else ""
    scanned = [
        (params.line, match.start())
        for match in _IDENTIFIER_RE.finditer(text)
        if not keyword.iskeyword(match.group(0))
    ]
    return scanned or [(params.line, 0)]


def _query_positions(params: LspParams, script: Any, lines: list[str]) -> list[tuple[int, int]]:
    """All positions worth trying for this call: the line/column candidates,
    then the name-resolved position when ``name`` is given (a name rescues a
    line that names the wrong place, and is the only source when line is not
    given at all)."""
    positions = _candidate_positions(params, lines)
    if params.column is None and params.name:
        for resolved in _positions_for_name(script, params.name, lines):
            if resolved not in positions:
                positions.append(resolved)
    return positions


def _unknown_symbol_error(
    tool_call_id: str, params: LspParams, script: Any, lines: list[str], display: str
) -> ToolResult:
    """The clean 'no such symbol' answer: name the miss, then the symbols that
    DO exist so the next call can be made without a round trip through
    grep."""
    candidates = _outline_names(script, lines)
    listed = ", ".join(f"{n.name} (L{n.line}, {n.type})" for n in candidates[:LSP_CANDIDATE_LIMIT])
    more = (
        f" … and {len(candidates) - LSP_CANDIDATE_LIMIT} more"
        if len(candidates) > LSP_CANDIDATE_LIMIT
        else ""
    )
    return _error(
        tool_call_id,
        "lsp",
        f"no definition or import of '{params.name}' found in {display}\n"
        f"available symbols: {listed or '(none)'}{more}",
    )


def _no_symbol_at_position_error(tool_call_id: str, params: LspParams, display: str) -> ToolResult:
    where = f"{display}:{params.line}"
    if params.column is not None:
        where += f":{params.column}"
    return _error(
        tool_call_id,
        "lsp",
        f"no symbol found at {where}\n"
        "pass name=<symbol> to locate by name, or use action=symbols for the outline",
    )


def _definitions_result(
    tool_call_id: str,
    script: Any,
    params: LspParams,
    path: Path,
    root: Path,
    cache: dict[str, list[str]],
) -> ToolResult:
    lines = _source_lines(path, cache)
    display = _display_path(path, root)
    targets: list[Any] = []
    used: tuple[int, int] | None = None
    for line, column in _query_positions(params, script, lines):
        targets = list(script.goto(line, column, follow_imports=True))
        if targets:
            used = (line, column)
            break
    if not targets:
        if params.name and not _positions_for_name(script, params.name, lines):
            return _unknown_symbol_error(tool_call_id, params, script, lines, display)
        return _no_symbol_at_position_error(tool_call_id, params, display)

    rows: list[str] = []
    seen: set[tuple[str, int | None, str]] = set()
    label = params.name or (targets[0].name if targets else "")
    for target in targets:
        key = (str(target.module_path), target.line, target.name)
        if key in seen:
            continue
        seen.add(key)
        if target.module_path is None or target.line is None:
            # Builtins and stubs without a source position (measured: the
            # ``def print`` name carries module_path=None).
            rows.append(f"builtins: {target.name} ({target.type})")
            continue
        target_lines = _source_lines(Path(target.module_path), cache)
        snippet = (
            target_lines[target.line - 1].strip() if 0 < target.line <= len(target_lines) else ""
        )
        rows.append(f"{_display_path(Path(target.module_path), root)}:{target.line}: {snippet}")
    elided = len(rows) - LSP_DEFINITION_LIMIT
    if elided > 0:
        rows = rows[:LSP_DEFINITION_LIMIT]
        rows.append(f"… and {elided} more")
    where = f" at {display}:{used[0]}:{used[1]}" if used else ""
    return _text(
        tool_call_id,
        "lsp",
        f"definitions of '{label}'{where}:\n" + "\n".join(rows),
    )


def _references_result(
    tool_call_id: str,
    script: Any,
    params: LspParams,
    path: Path,
    root: Path,
    context: ToolContext | None,
    cache: dict[str, list[str]],
) -> ToolResult:
    lines = _source_lines(path, cache)
    display = _display_path(path, root)
    refs: list[Any] = []
    for line, column in _query_positions(params, script, lines):
        refs = list(script.get_references(line, column))
        if refs:
            break
    if not refs:
        if params.name and not _positions_for_name(script, params.name, lines):
            return _unknown_symbol_error(tool_call_id, params, script, lines, display)
        return _no_symbol_at_position_error(tool_call_id, params, display)

    label = params.name or refs[0].name
    rows: list[str] = []
    positionless = 0
    for ref in refs:
        if ref.module_path is None or ref.line is None:
            positionless += 1
            continue
        ref_lines = _source_lines(Path(ref.module_path), cache)
        text = ref_lines[ref.line - 1].rstrip() if 0 < ref.line <= len(ref_lines) else ""
        rows.append(f"{_display_path(Path(ref.module_path), root)}:{ref.line}: {text}")
    if not rows:
        return _text(
            tool_call_id,
            "lsp",
            f"No source references found for '{label}' in {display}.",
            useless=True,
            details={"useless": True},
        )
    shown = "\n".join(rows[:LSP_REFERENCE_LIMIT])
    body, details = _capped_list_body("\n".join(rows), shown, "lsp", context)
    if positionless:
        body += f"\n({positionless} reference(s) without source positions omitted)"
    return _text(
        tool_call_id,
        "lsp",
        f"{len(rows)} reference(s) of '{label}' in {display}:\n{body}",
        details=details,
    )


def _symbols_result(
    tool_call_id: str,
    script: Any,
    path: Path,
    root: Path,
    context: ToolContext | None,
    cache: dict[str, list[str]],
) -> ToolResult:
    lines = _source_lines(path, cache)
    display = _display_path(path, root)
    names = _outline_names(script, lines)
    if not names:
        return _text(
            tool_call_id,
            "lsp",
            f"No definitions found in {display} — is it a Python module?",
            useless=True,
            details={"useless": True},
        )
    rows: list[str] = []
    for name in names:
        start = name.get_definition_start_position() or (name.line, name.column or 0)
        end = name.get_definition_end_position() or start
        # The definition's start column IS its indent, so the outline nests
        # without a scope tree (jedi's Name.parent is unusable in 0.20:
        # module-level names return an object with no .name).
        depth = start[1] // 4
        rows.append(f"{'  ' * depth}L{start[0]}-{end[0]} {name.type} {name.name}")
    shown = "\n".join(rows[:LSP_SYMBOL_LIMIT])
    body, details = _capped_list_body("\n".join(rows), shown, "lsp", context)
    return _text(
        tool_call_id,
        "lsp",
        f"outline of {display} ({len(names)} symbol(s)):\n{body}",
        details=details,
    )


def _rename_preview_result(
    tool_call_id: str,
    script: Any,
    params: LspParams,
    path: Path,
    root: Path,
    context: ToolContext | None,
    cache: dict[str, list[str]],
) -> ToolResult:
    lines = _source_lines(path, cache)
    display = _display_path(path, root)
    label = params.name or ""
    diff = ""
    positions = _query_positions(params, script, lines)
    last_error: Exception | None = None
    for line, column in positions:
        # A keyword or blank position raises RefactoringError ("no name under
        # the cursor", measured) — that is the signal to try the next
        # candidate, not a failure of the call.
        try:
            refactoring = script.rename(line, column, new_name=params.new_name)
        except RefactoringError as exc:
            last_error = exc
            continue
        diff = refactoring.get_diff()
        # jedi renders the refactoring as a unified diff itself (the same
        # ``--- / +++`` shape difflib.unified_diff emits); re-diffing the
        # changed files by hand would be a worse copy of get_diff().
        if not label:
            label = _identifier_at(lines, line, column)
        break
    if not diff and last_error is not None:
        return _error(
            tool_call_id,
            "lsp",
            f"cannot rename: {last_error}\n"
            "pass name=<symbol> to locate by name, or use action=symbols for the outline",
        )
    if not diff.strip():
        return _text(
            tool_call_id,
            "lsp",
            f"rename to '{params.new_name}' would change nothing in {display}.",
            useless=True,
            details={"useless": True},
        )
    files = [row[4:] for row in diff.splitlines() if row.startswith("--- ")]
    display_diff, details = spill_truncate(diff, "lsp", context)
    return _text(
        tool_call_id,
        "lsp",
        f"rename '{label}' -> '{params.new_name}' across {len(files)} file(s) — "
        f"preview only, nothing written; apply via the edit tool:\n{display_diff}",
        details=details,
    )


def _identifier_at(lines: list[str], line: int, column: int) -> str:
    text = lines[line - 1] if 0 < line <= len(lines) else ""
    match = _IDENTIFIER_RE.match(text, column)
    return match.group(0) if match else "symbol"


@_guard("lsp")
async def execute_lsp(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Answer one symbol query. Read-only by construction; jedi's own
    failures surface as error results, never exceptions."""
    try:
        params = LspParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "lsp", exc)
    if params.action not in LSP_ACTIONS:
        return _error(
            tool_call_id,
            "lsp",
            f"unknown action: {params.action} (expected one of {', '.join(LSP_ACTIONS)})",
        )
    if params.action == "rename_preview" and not params.new_name:
        return _error(tool_call_id, "lsp", "rename_preview requires new_name")
    if params.action != "symbols" and params.line is None and not params.name:
        return _error(
            tool_call_id,
            "lsp",
            f"{params.action} requires line or name (symbols needs neither)",
        )
    if params.line is not None and params.line < 1:
        return _error(tool_call_id, "lsp", "line is 1-based; the first line is 1")
    if params.new_name and not _IDENTIFIER_RE.fullmatch(params.new_name):
        return _error(tool_call_id, "lsp", f"new_name is not an identifier: {params.new_name!r}")

    cwd = _safe_cwd(context)
    try:
        root = Path(cwd).expanduser().resolve()
    except RuntimeError:
        root = Path(cwd).resolve()
    path, _inside, _resolvable = _resolve_workspace_path(params.path, cwd)
    if not path.exists():
        return _error(tool_call_id, "lsp", f"Path does not exist: {path}")
    if not path.is_file():
        return _error(tool_call_id, "lsp", f"Not a file: {path}")

    if jedi is None:
        return _error(tool_call_id, "lsp", "jedi is not installed (install local-operator[lsp])")
    script = jedi.Script(path=str(path))
    cache: dict[str, list[str]] = {}
    if params.action == "definitions":
        return _definitions_result(tool_call_id, script, params, path, root, cache)
    if params.action == "references":
        return _references_result(tool_call_id, script, params, path, root, context, cache)
    if params.action == "symbols":
        return _symbols_result(tool_call_id, script, path, root, context, cache)
    return _rename_preview_result(tool_call_id, script, params, path, root, context, cache)


def build_lsp_tool() -> AgentTool | None:
    """Advertise the LSP tool only when jedi (the ``lsp`` extra) is importable.

    Same createIf convention as ``build_browser_tool``: an optional capability
    returns None — excluded from the inventory — when the host did not opt
    into its dependency. In-process jedi is pure Python, so unlike the browser
    tool there is no runtime probe beyond the import itself.
    """
    if jedi is None:
        return None
    return AgentTool(
        name="lsp",
        label="LSP",
        description=(
            "Symbol-aware Python navigation: definitions (jump to a definition "
            "across imports), references (usage sites with line text), symbols "
            "(file outline with line ranges) and rename_preview (unified diff of "
            "a proposed rename — never applied; make the change via the edit "
            "tool). Prefer this over grep for symbol work: it resolves imports "
            "and shadowing that text search misses."
        ),
        parameters=LspParams.model_json_schema(),
        # Every action answers a question; even rename_preview only reads.
        approval_tier="read",
        # Analysis is per-call and touches no shared state.
        concurrency="shared",
        interruptible=False,
        execute=execute_lsp,
    )
