#!/usr/bin/env python3
"""The i18n ratchet and catalogue checks (RFC §2.7).

TWO JOBS, one cheap entry point (CI job `i18n`, and a unit test):

1. **The AST ratchet.** Literals reaching user-facing sinks are counted per
   file and compared against ``i18n/baseline.json``, a per-file ceiling that
   may only stay equal or go down. Extraction slices lower their files'
   entries with ``--update``; new files start at 0, so new code must extract,
   pragma (``# i18n: ignore <reason>``) or be allowlisted (``i18n/allowlist.toml``,
   reason REQUIRED). This module deliberately flags BROADLY: a false positive
   is visible noise the allowlist records, a false negative is a hole nothing
   reports — the repository's fail-closed rule. The one known blind spot is
   documented at ``_arguments`` (string concatenation via ``+``).

   **ENFORCEMENT IS SCOPED (RFC §9: P1 ships ADVISORY).** The baseline carries
   an ``enforced`` list of path prefixes (and exact file paths). Findings
   OUTSIDE that scope are printed as ``advisory:`` lines and DO NOT fail
   (exit 0): the ratchet reports the tree's real state without turning every
   unrelated PR in a fleet of concurrent sessions red before an extraction
   slice exists to give a string a key. Findings INSIDE the scope fail, as do
   catalogue parity and generation drift EVERYWHERE (those only involve
   i18n-owned files). The list starts with the i18n package and its tooling;
   extraction slices ACTIVATE their area by adding that area's prefix in a
   reviewed change, and P5 flips the whole tree to enforced.

2. **Catalogue checks.** Every message parses in the shipped subset; keys are
   lower snake_case and namespaced; per namespace, every locale's key set,
   argument tuples (name AND kind) and plural categories match ``en``; a
   plural branch must define ``other``; the en-only context sidecar may only
   describe keys that exist, using only the shared typed field set
   (``catalogues.CONTEXT_FIELDS``) the ``read_public_copy`` reader accepts.

The ratchet's counts are a SNAPSHOT, not a truth: they pin what "the tree as
it stands" means so a diff's effect is measurable. ``--init`` is therefore
one-time by design (it refuses to overwrite an existing baseline without
``--force``); a legitimate scanner-coverage change or a full re-snapshot on a
moved base re-runs ``--init --force`` and says so in its PR body — that
preserves ``enforced`` and its note, which are REVIEW STATE, not counts.

Usage:
    python scripts/i18n/check.py                 # ratchet + parity (CI)
    python scripts/i18n/check.py --init          # capture the baseline once
    python scripts/i18n/check.py --init --force  # full re-snapshot (moved base / scanner change)
    python scripts/i18n/check.py --update <path> # lower a file's ceiling
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# Scripts under scripts/ read the tree they live in regardless of which venv
# launched them (a console script's sys.path[0] is .venv/bin, not the cwd).
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from local_operator.i18n import catalogues  # noqa: E402
from local_operator.i18n import runtime as msg_runtime  # noqa: E402
from scripts.i18n.login import check_login  # noqa: E402

PRODUCT_TREE = "local_operator"
#: The scanned corpus, repo-relative. `scripts/i18n/` is scanned because it is
#: inside the initial enforced scope: the tooling must not grow user-visible
#: prose of its own (its existing developer-diagnostic prints are frozen at
#: their baseline counts; a NEW file there starts at 0 like anywhere else).
SCAN_ROOTS = (PRODUCT_TREE, "scripts/i18n")
#: Fallback enforcement scope when the baseline carries no `enforced` field
#: (a pre-schema-2 file): the i18n package and its tooling — the code this
#: program itself owns. Slices extend the list; P5 flips the whole tree.
DEFAULT_ENFORCED = ("local_operator/i18n/", "scripts/i18n/")
BASELINE_NOTE = (
    "enforced = blocking scope (path prefixes or exact paths); findings elsewhere are "
    "advisory (report-only) until the area's extraction slice adds its prefix. "
    "files = per-file literal ceilings (shrink-only; new files start at 0). "
    "See scripts/i18n/check.py."
)
BASELINE = REPO / "i18n" / "baseline.json"
ALLOWLIST = REPO / "i18n" / "allowlist.toml"
ALLOWANCE = REPO / "i18n" / "coverage-allowance.json"
PRAGMA_RE = re.compile(r"#\s*i18n:\s*ignore\s+\S")
KEY_RE = re.compile(r"^[a-z0-9_.]+$")
#: Any Unicode letter — the line between a prose-like literal and punctuation,
#: width strings ("──"), or pure format tokens ("{}").
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    kind: str
    line: int
    text: str


def _arguments(call: ast.Call) -> list[ast.expr]:
    """Every argument expression of ``call``, positional then keyword.

    What this argument-position view cannot see on its own — copy bound to a
    NAME first (a local ``label = "..."`` or a module ``REMEDY = "..."``) and,
    in enforced files, copy selected out of a dict literal — is recovered by
    the in-function resolver at ``_Resolver``. What stays invisible is stated
    there, not implied here.
    """
    return [*call.args, *(kw.value for kw in call.keywords)]


def _literal_worthy(node: ast.expr) -> bool:
    """Whether ``node`` counts as one prose-like literal.

    A constant string with at least one letter, or an f-string with at least
    one such constant SEGMENT (`f"Session {id} failed"` counts; `f"{a}-{b}"`
    does not). Counting f-strings is deliberate: they are the dominant prose
    form in the TUI (RFC §1), and a ratchet that cannot see them would pass
    exactly the new code it exists to catch.

    Two assembled forms fold, because they reach sinks as user copy just as
    plainly as a literal does:

    * ``+`` concatenation — ``" member" + ("" if n == 1 else "s")`` is a
      BinOp whose text the pre-fold scanner could not see at all;
    * the conditional expression — ``"cmd" if darwin else "ctrl"`` is an
      IfExp selecting between two pieces of copy.

    Folding is recursive through those two shapes and no further: a literal
    merely REFERENCED by name is still invisible (see ``_arguments``).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return bool(_LETTER_RE.search(node.value))
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(part, ast.Constant)
            and isinstance(part.value, str)
            and _LETTER_RE.search(part.value)
            for part in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_worthy(node.left) or _literal_worthy(node.right)
    if isinstance(node, ast.IfExp):
        return _literal_worthy(node.body) or _literal_worthy(node.orelse)
    return False


def _call_name(call: ast.Call) -> str:
    """``print`` / ``console.print`` / ``Text`` — the syntactic callee name."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


#: Keyword names that carry user copy on the registry/dataclass constructors
#: below — the fields RFC §2.7 names as sinks (`NoticeEvent`'s text/headline
#: get their own rule above; they are not kwargs of these three).
_COPY_KWARGS = ("label", "help", "warning", "description")


#: Cell-fitting helpers whose FIRST argument is display text: their result is
#: what gets painted (``result.append(truncate_cells("Sessions", w).ljust(w))``),
#: so the literal inside never sat in a sink's own argument position and the
#: scanner read such widgets as clean. Only text-in/text-out helpers belong
#: here — ``images.fit_cells`` sizes pixels and is deliberately absent. Every
#: name is distinctive enough that a same-named machine-text helper elsewhere
#: is unlikely; a literal-worthy first argument still has to be prose-like.
_CELL_FIT_HELPERS = ("truncate_cells", "truncate_name", "truncate_row")


def _call_violations(call: ast.Call) -> list[tuple[str, ast.expr]]:
    """Sink matches for one call: ``(kind, literal node)`` pairs, deduplicated.

    The prose-shaped subset of ``_sink_expressions`` — the argument-position
    view. Names and dict selections in the remaining expressions are the
    resolver's job (``scan_source``).
    """
    return [(kind, node) for kind, node in _sink_expressions(call) if _literal_worthy(node)]


def _sink_expressions(call: ast.Call) -> list[tuple[str, ast.expr]]:
    """Every expression in a sink position of ``call``, prose-like or not.

    Deduplication matters because one literal can satisfy two rules (a
    `Setting(help=...)` is both the generic `help=` rule and the registry-copy
    rule); the count is per LITERAL, not per rule.

    `.append`/`.update` count on ANY receiver, deliberately: in this corpus
    user prose flows through list-building (`lines.append("her check-ins
    stopped")` is the status sentence), so a receiver-name allowlist would
    miss most copy while still passing machine tokens. The broad read trades
    false positives — visible, allowlist-able — for no false negatives.
    """
    name = _call_name(call)
    matched: dict[int, tuple[str, ast.expr]] = {}

    def take(kind: str, node: ast.expr | None) -> None:
        if node is not None:
            matched.setdefault(id(node), (kind, node))

    if name in _CELL_FIT_HELPERS and call.args:
        take(f"cell-fit {name}", call.args[0])
    if name == "print":
        # `print(...)` and `console.print(...)` both resolve here: `_call_name`
        # returns the attribute name for attribute calls.
        for arg in call.args:
            take("print", arg)
    if isinstance(call.func, ast.Attribute) and call.func.attr in ("append", "update"):
        for arg in call.args:
            take(f"textual.{call.func.attr}", arg)
    if name == "Text" or (
        isinstance(call.func, ast.Attribute) and call.func.attr in ("assemble", "from_markup")
    ):
        for arg in call.args:
            take("Text", arg)
    # Textual constructors and input hints are visible copy even when no
    # Rich Text/append call follows. Keep this constructor set narrow: generic
    # title/label fields also carry machine identifiers outside these widgets.
    if name in ("Static", "Label", "Input"):
        for arg in call.args:
            take(f"textual.{name}", arg)
    take("placeholder=", _keyword(call, "placeholder"))
    # SDK validation failures cross the public session boundary as human copy;
    # unrelated developer exceptions are not a product-copy sink. Match both
    # bare and qualified constructors through the same canonical name resolver.
    if name in ("SessionSpecError", "SessionIsolationError"):
        for arg in call.args:
            take("SDK session error", arg)
    take("help=", _keyword(call, "help"))
    if name == "HTTPException":
        take("HTTPException", _keyword(call, "detail"))
        if len(call.args) >= 2:
            take("HTTPException", call.args[1])
    if name == "CRUDResponse":
        take("CRUDResponse", _keyword(call, "message"))
    if name == "NoticeEvent":
        for kw in ("text", "headline"):
            take("NoticeEvent", _keyword(call, kw))
    if name in ("Setting", "SlashCommand", "Choice"):
        for kw in _COPY_KWARGS:
            take(f"{name} copy", _keyword(call, kw))
        # Positional copy fields, by the constructors' own field order:
        # SlashCommand("<name token>", "<description>", ...) and
        # Choice("<value>", "<label>", "<description>").
        if name == "SlashCommand":
            for arg in call.args[1:]:
                take("SlashCommand copy", arg)
        elif name == "Choice":
            for arg in call.args[1:3]:
                take("Choice copy", arg)
    return list(matched.values())


#: ``(first line, last line)`` of a node, inclusive — a pragma search window.
_Span = tuple[int, int]


def _span(node: ast.AST) -> _Span:
    line = getattr(node, "lineno", 1)
    return line, getattr(node, "end_lineno", line) or line


def _own_scope(node: ast.AST) -> list[ast.AST]:
    """Every node inside ``node``'s own scope — nested defs/classes/lambdas excluded.

    A nested function's assignments are ITS locals, not the enclosing
    function's; walking straight through would let an inner ``label = "x"``
    bind an outer ``print(label)`` that never sees it.
    """
    out: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        out.append(child)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(child))
    return out


class _Scope:
    """One function's (or the module's) name bindings, for the resolver.

    ``values`` maps a name to every ``(value, statement)`` a PLAIN binding
    gives it (``x = v``, ``x: T = v``, ``(x := v)``); ``local`` holds every
    name the scope stores to at all — parameters, loop/with/except targets,
    tuple unpacking, imports — so a local that merely SHADOWS a module
    constant hides it rather than inheriting its copy. Flow-insensitive on
    purpose: every plain binding counts wherever it sits, because the question
    is "can this copy reach the sink", not "does it on this path".
    """

    def __init__(self, node: ast.AST) -> None:
        self.values: dict[str, list[tuple[ast.expr, ast.stmt | ast.expr]]] = {}
        self.local: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                self.local.add(arg.arg)
            for arg in (args.vararg, args.kwarg):
                if arg is not None:
                    self.local.add(arg.arg)
        for child in _own_scope(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                self.local.add(child.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    self.local.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.local.add(child.name)
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        self.values.setdefault(target.id, []).append((child.value, child))
            elif isinstance(child, ast.AnnAssign) and child.value is not None:
                if isinstance(child.target, ast.Name):
                    self.values.setdefault(child.target.id, []).append((child.value, child))
            elif isinstance(child, ast.NamedExpr):
                self.values.setdefault(child.target.id, []).append((child.value, child))


class _Resolver:
    """Follow a sink expression back to the prose literals that can reach it.

    WHY: the argument-position scanner only saw copy whose own shape was a
    literal, so prose routed through a name — ``WEDGED_REMEDY = "lop stop if it
    stays silent"`` at module level, ``label = "Active Sessions"`` in the
    function — or picked out of a dict (``{"header:active": "Active
    Sessions"}[kind]``) read as zero while it rendered English
    (``tui/widgets/session_sidebar.py``, enforced at 0, was the measured case).

    What it follows, from a sink expression:

    * a NAME, to every plain binding in the innermost scope that stores it,
      else to a module-level binding (enclosing functions are not consulted:
      closures are rare in render code and the miss is stated, not hidden);
    * through ``+``, the conditional expression and f-string ``{fields}``;
    * in ENFORCED files only, a subscript or ``.get(...)`` on a dict literal
      (inline, or bound to a name as above) — each prose-like VALUE counts,
      and a ``.get`` default resolves like any other expression.

    The dict rule is enforcement-scoped because it is the one with real false
    positive exposure (machine maps share the shape); an enforced file has an
    owner already keying its copy, and an advisory file's counts would only
    grow a ceiling nobody acts on yet.

    STILL INVISIBLE, deliberately: values returned from another function,
    attributes (``self.title``), imported constants, closures over an
    enclosing function's locals, and container literals iterated rather than
    indexed. Each needs inter-procedural or type-aware analysis.

    Each literal is reported ONCE per file however many sinks reach it — the
    count is per literal, as in ``_call_violations`` — and the resolver never
    re-enters a binding it is already following (``x = f"{x} more"``).
    """

    def __init__(self, module: ast.Module, *, dict_values: bool) -> None:
        self.dict_values = dict_values
        self.module_scope = _Scope(module)
        self.scopes: dict[int, _Scope] = {}
        #: call id -> innermost enclosing function node (absent = module level).
        self.owner: dict[int, ast.AST] = {}
        self._index(module, None)

    def _index(self, node: ast.AST, fn: ast.AST | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call) and fn is not None:
                self.owner[id(child)] = fn
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                self._index(child, child)
            else:
                self._index(child, fn)

    def _scope(self, fn: ast.AST | None) -> _Scope:
        if fn is None:
            return self.module_scope
        scope = self.scopes.get(id(fn))
        if scope is None:
            scope = self.scopes[id(fn)] = _Scope(fn)
        return scope

    def _bindings(self, name: str, fn: ast.AST | None) -> list[tuple[ast.expr, ast.AST]]:
        scope = self._scope(fn)
        if fn is not None and name in scope.local:
            return list(scope.values.get(name, []))
        if fn is not None and name not in scope.local:
            return list(self.module_scope.values.get(name, []))
        return list(scope.values.get(name, []))

    def resolve(self, call: ast.Call, expr: ast.expr) -> list[tuple[ast.expr, _Span | None]]:
        """``(literal node, binding span or None)`` for every prose literal ``expr`` reaches.

        The span is the binding statement a pragma may also sit on; ``None``
        means the literal is inside the call itself (its span already covers it).
        """
        fn = self.owner.get(id(call))
        out: list[tuple[ast.expr, _Span | None]] = []
        self._walk(expr, fn, None, set(), out)
        return out

    def _walk(
        self,
        expr: ast.expr,
        fn: ast.AST | None,
        home: _Span | None,
        following: set[str],
        out: list[tuple[ast.expr, _Span | None]],
    ) -> None:
        if _literal_worthy(expr):
            # `home` None: the literal sits inside the call (an f-string field,
            # a `.get` default), whose span the caller's pragma rule covers.
            out.append((expr, home))
            return
        if isinstance(expr, ast.Name):
            if expr.id in following:
                return
            for value, stmt in self._bindings(expr.id, fn):
                self._walk(value, fn, _span(stmt), following | {expr.id}, out)
        elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            self._walk(expr.left, fn, home, following, out)
            self._walk(expr.right, fn, home, following, out)
        elif isinstance(expr, ast.IfExp):
            self._walk(expr.body, fn, home, following, out)
            self._walk(expr.orelse, fn, home, following, out)
        elif isinstance(expr, ast.JoinedStr):
            for part in expr.values:
                if isinstance(part, ast.FormattedValue):
                    self._walk(part.value, fn, home, following, out)
        elif self.dict_values and isinstance(expr, ast.Subscript):
            self._dict_values(expr.value, fn, home, following, out)
        elif (
            self.dict_values
            and isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "get"
        ):
            self._dict_values(expr.func.value, fn, home, following, out)
            for default in expr.args[1:2]:
                self._walk(default, fn, home, following, out)
        elif self.dict_values and isinstance(expr, ast.Dict):
            # Reached only through `_dict_values`; a dict handed straight to a
            # sink is a container, not copy.
            return

    def _dict_values(
        self,
        container: ast.expr,
        fn: ast.AST | None,
        home: _Span | None,
        following: set[str],
        out: list[tuple[ast.expr, _Span | None]],
    ) -> None:
        if isinstance(container, ast.Dict):
            for value in container.values:
                if _literal_worthy(value):
                    # An inline dict (home None) sits in the call's own span,
                    # but its VALUE lines are what a reader would annotate;
                    # the value's own span is a valid pragma site either way.
                    out.append((value, home or _span(value)))
                else:
                    self._walk(value, fn, home, following, out)
        elif isinstance(container, ast.Name) and container.id not in following:
            for value, stmt in self._bindings(container.id, fn):
                self._dict_values(value, fn, _span(stmt), following | {container.id}, out)


_ENFORCED_CACHE: dict[tuple[str, int], list[str]] = {}


def _repo_enforced() -> list[str]:
    """The ``enforced`` list of ``BASELINE`` as it is NOW.

    Keyed by path and mtime, not cached once: tests repoint ``BASELINE`` at
    fixture trees and ``_activate`` rewrites it mid-test, and a tree scan
    would otherwise re-read the file once per scanned module.
    """
    try:
        stamp = BASELINE.stat().st_mtime_ns
    except OSError:
        stamp = -1
    key = (str(BASELINE), stamp)
    if key not in _ENFORCED_CACHE:
        _ENFORCED_CACHE[key] = load_baseline(BASELINE)["enforced"]
    return _ENFORCED_CACHE[key]


def scan_source(path: str, source: str, *, enforced: list[str] | None = None) -> list[Violation]:
    """The violations of one file's source, pragmas applied.

    A ``# i18n: ignore <reason>`` comment ANYWHERE between a call's first and
    last line exempts the literals in that call — not just a comment on the
    literal's own line, which is what the allowlist docs used to claim
    (round-1 n2). Multi-line calls are the norm here, so the call span is the
    span that reads naturally; the allowlist's text now says exactly this.

    A literal the resolver reached through a binding is ALSO exempted by a
    pragma on that binding's statement (``REMEDY = "..."  # i18n: ignore
    <reason>``) — the line a reader annotating the constant writes on. The
    call-span rule keeps working for it, so an existing call pragma still
    covers whatever that call now resolves.

    ``enforced`` selects the enforced-only dict-value rule (``_Resolver``);
    ``None`` means "the repository baseline's scope", which is what every
    real-tree caller wants.
    """
    try:
        module = ast.parse(source, filename=path)
    except SyntaxError as exc:  # pragma: no cover - the repo gate parses first
        raise SystemExit(f"check.py: cannot parse {path}: {exc}") from exc
    lines = source.splitlines()
    scope = _repo_enforced() if enforced is None else enforced
    resolver = _Resolver(module, dict_values=in_enforced_scope(path, scope))

    def pragma(first: int, last: int) -> bool:
        return any(PRAGMA_RE.search(lines[i - 1]) for i in range(first, min(last, len(lines)) + 1))

    found: list[Violation] = []
    seen: set[int] = set()
    for call in (node for node in ast.walk(module) if isinstance(node, ast.Call)):
        call_span = _span(call)
        for kind, expr in _sink_expressions(call):
            if _literal_worthy(expr):
                reached: list[tuple[ast.expr, _Span | None]] = [(expr, None)]
            else:
                reached = resolver.resolve(call, expr)
            for node, home in reached:
                if id(node) in seen:
                    continue
                if home is None:
                    start = min(call.lineno, node.lineno)
                    end = max(call_span[1], node.end_lineno or node.lineno)
                    if pragma(start, end):
                        continue
                elif pragma(*call_span) or pragma(*home):
                    continue
                seen.add(id(node))
                text = node.value if isinstance(node, ast.Constant) else "<f-string>"
                found.append(Violation(kind=kind, line=node.lineno, text=str(text)[:80]))
    return found


def scan_tree(root: Path) -> dict[str, list[Violation]]:
    """Scan the corpus (``SCAN_ROOTS``) under ``root``, repo-relative keys."""
    results: dict[str, list[Violation]] = {}
    for base in SCAN_ROOTS:
        tree = root / base
        if not tree.is_dir():  # a fixture or partial tree need not carry every root
            continue
        for path in sorted(tree.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8")
            violations = scan_source(rel, source)
            if violations:
                results[rel] = violations
    return results


# ---------------------------------------------------------------------------
# Allowlist and baseline
# ---------------------------------------------------------------------------


def load_allowlist(path: Path) -> list[tuple[str, str]]:
    """``(pattern, reason)`` entries; a missing reason is a hard error."""
    if not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    parsed: list[tuple[str, str]] = []
    if not isinstance(entries, list):
        raise SystemExit(
            f"check.py: {path.name}: `entries` must be a list of [pattern, reason] pairs"
        )
    for entry in entries:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not all(isinstance(part, str) for part in entry)
        ):
            raise SystemExit(
                f'check.py: {path.name}: every allowlist entry must be ["path-pattern", "reason"]'
            )
        pattern, reason = entry
        if not reason.strip():
            raise SystemExit(f"check.py: {path.name}: allowlist entry {pattern!r} has no reason")
        parsed.append((pattern, reason))
    return parsed


def allowlisted(path: str, entries: list[tuple[str, str]]) -> str | None:
    """The reason this file is exempt, or None.

    Two forms, both exact: a repo-relative file path, or a directory prefix
    ending in ``/``. NO GLOBS on purpose — fnmatch's `*` spans path
    separators, so a "narrow-looking" glob can silently exempt a subtree;
    a directory the author means to cover is then written as one, visibly.
    """
    for pattern, reason in entries:
        if pattern.endswith("/"):
            if path.startswith(pattern):
                return reason
        elif path == pattern:
            return reason
    return None


def load_baseline(path: Path) -> dict[str, Any]:
    """The whole baseline object: ``enforced`` (scope) and ``files`` (ceilings).

    Missing or damaged shapes are tolerated as EMPTY rather than fatal: the
    file is data, and a hand-edit mistake should surface as a normal check
    failure (counted files read as "not in baseline"), not a traceback. A
    MISSING ``enforced`` key falls back to ``DEFAULT_ENFORCED`` (fail closed —
    a schema-1 file must not silently un-enforce the code this program owns);
    a PRESENT-but-empty list is respected as the deliberate "nothing is
    enforced" state; and a NON-LIST value (a hand-edit) fails closed the same
    way — silently UN-enforcing on damage is the direction this guard must not
    fail (round-4 m1).
    """
    if not path.is_file():
        return {"schema": 2, "enforced": list(DEFAULT_ENFORCED), "files": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {}
    files = data.get("files", {})
    enforced = data.get("enforced", list(DEFAULT_ENFORCED))
    return {
        "schema": data.get("schema", 1),
        "note": data.get("note", BASELINE_NOTE),
        "enforced": (
            [str(p) for p in enforced] if isinstance(enforced, list) else list(DEFAULT_ENFORCED)
        ),
        "files": {str(p): int(c) for p, c in files.items()} if isinstance(files, dict) else {},
    }


def save_baseline(path: Path, files: dict[str, int], enforced: list[str], note: str) -> None:
    payload = {
        "schema": 2,
        "note": note,
        "enforced": list(enforced),
        "files": {k: files[k] for k in sorted(files)},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _activation_state(path: Path) -> tuple[list[str], str]:
    """``(enforced, note)`` to PRESERVE across ``--init --force``.

    These are review state, not counts: a re-snapshot on a moved base must not
    walk an area's activation back. Presence of the key selects defaults, not
    truthiness — a deliberate empty list stays empty (round-1 n3's lesson,
    applied to the new field).
    """
    if not path.is_file():
        return list(DEFAULT_ENFORCED), BASELINE_NOTE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return list(DEFAULT_ENFORCED), BASELINE_NOTE
    if not isinstance(data, dict):
        return list(DEFAULT_ENFORCED), BASELINE_NOTE
    enforced = data.get("enforced", list(DEFAULT_ENFORCED))
    return (
        [str(p) for p in enforced] if isinstance(enforced, list) else list(DEFAULT_ENFORCED),
        str(data.get("note", BASELINE_NOTE)),
    )


def in_enforced_scope(rel: str, enforced: list[str]) -> bool:
    """Whether ``rel`` sits inside the blocking scope.

    A value ending in ``/`` is a path-segment prefix (``local_operator/i18n/``);
    anything else is an exact file path. No globs by design — ``fnmatch``'s
    ``*`` crosses ``/``, which is the trap the package-data guard documents
    (round-1 m4).
    """
    for entry in enforced:
        if entry.endswith("/"):
            if rel.startswith(entry):
                return True
        elif rel == entry:
            return True
    return False


@dataclass(frozen=True)
class Finding:
    """One counted file that exceeds — or is missing from — its ceiling."""

    path: str
    count: int
    ceiling: int | None  # None = not in the baseline: a new file, starts at 0

    def line(self) -> str:
        if self.ceiling is None:
            return (
                f"{self.path}: {self.count} literal(s) — not in baseline.json; "
                "a file with no recorded ceiling starts at 0"
            )
        return f"{self.path}: {self.count} literal(s) against a ceiling of {self.ceiling}"


def findings(counts: dict[str, int], files: dict[str, int]) -> list[Finding]:
    out: list[Finding] = []
    for rel in sorted(counts):
        ceiling = files.get(rel)
        count = counts[rel]
        if ceiling is None:
            out.append(Finding(rel, count, None))
        elif count > ceiling:
            out.append(Finding(rel, count, ceiling))
    return out


def _counts(scanned: dict[str, list[Violation]], entries: list[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rel, violations in scanned.items():
        if allowlisted(rel, entries) is not None:
            continue
        if violations:
            counts[rel] = len(violations)
    return counts


# ---------------------------------------------------------------------------
# Catalogue checks
# ---------------------------------------------------------------------------


def check_catalogues() -> list[str]:
    """Parity + shape problems across ``catalogues/``, as printable lines."""
    problems: list[str] = []
    root = catalogues.catalogue_root()
    if not root.is_dir():
        return [f"catalogues: root {root} does not exist"]
    plural = json.loads(
        (Path(msg_runtime.__file__).with_name("data") / "plural_rules.json").read_text(
            encoding="utf-8"
        )
    )
    en_namespaces = catalogues.namespaces("en")
    en_messages: dict[str, dict[str, str]] = {}
    for namespace in en_namespaces:
        messages = catalogues.load_catalogue("en", namespace)
        en_messages[namespace] = messages
        for key, source in messages.items():
            if not KEY_RE.match(key):
                problems.append(f"en/{namespace}: key {key!r} is not lower snake_case")
            if not key.startswith(namespace + "."):
                problems.append(f"en/{namespace}: key {key!r} does not repeat its namespace prefix")
            try:
                # Syntax gate for the en source: parse it (the cross-locale
                # argument comparison below re-reads it as the reference).
                msg_runtime.message_arguments(source)
            except msg_runtime.MessageSyntaxError as exc:
                problems.append(f"en/{namespace}: {key}: {exc}")
                continue
            _check_plural_categories(problems, "en", namespace, key, source, plural, "en")
        _check_context_sidecar(problems, "en", namespace, messages)
    # Cross-file duplicates: one message is defined by exactly ONE file (the
    # collision rule `messages._namespace_for` relies on, resolving
    # deepest-first). The per-file checks above cannot see this — a key like
    # `wire.errors.auth.x` satisfies both the `wire.errors` and the
    # `wire.errors.auth` prefix tests — so a duplicate would silently shadow
    # instead of failing. Translation-side duplicates reduce to this case or
    # to the missing/unknown-key parity checks below.
    owners: dict[str, str] = {}
    for namespace in en_namespaces:
        for key in en_messages[namespace]:
            if key in owners:
                problems.append(
                    # i18n: ignore checker diagnostic (operator tooling output, English-only)
                    f"en: key {key!r} is defined by both {owners[key]!r} and {namespace!r}"
                )
            else:
                owners[key] = namespace
    for locale in catalogues.locales():
        if locale == "en":
            continue
        if locale not in plural.get("locales", {}):
            problems.append(f"{locale}: no plural rules generated for this locale")
        for namespace in en_namespaces:
            try:
                translated = catalogues.load_catalogue(locale, namespace)
            except catalogues.CatalogueNotFound:
                # PER-NAMESPACE COVERAGE (2026-10-11 amendment, docs/design/
                # i18n.md §6): a locale that does not carry a namespace is NOT
                # an error — a namespace born after the locale's last
                # translation wave (an extraction slice's child) rides
                # en-fallback (§2.5) until its own wave lands, and the
                # per-namespace ship gate (`generate.py::_shipped_locales`)
                # decides whether the locale is offered at all. Only PRESENT
                # files are validated below: key parity, arguments and plural
                # categories against en, exactly as before.
                continue
            except catalogues.CatalogueInvalid as exc:
                problems.append(f"{locale}/{namespace}: {exc}")
                continue
            expected = en_messages[namespace]
            for key in sorted(set(expected) - set(translated)):
                problems.append(f"{locale}/{namespace}: missing key {key!r}")
            for key in sorted(set(translated) - set(expected)):
                problems.append(f"{locale}/{namespace}: unknown key {key!r}")
            for key in sorted(set(expected) & set(translated)):
                try:
                    got = msg_runtime.message_arguments(translated[key])
                except msg_runtime.MessageSyntaxError as exc:
                    problems.append(f"{locale}/{namespace}: {key}: {exc}")
                    continue
                want = msg_runtime.message_arguments(expected[key])
                problem = _arguments_problem(
                    want,
                    got,
                    translated_source=translated[key],
                    locale=locale,
                    plural=plural,
                )
                if problem is not None:
                    problems.append(
                        f"{locale}/{namespace}: {key}: arguments {got!r} != en's "
                        f"{want!r}: {problem}"
                    )
                _check_plural_categories(
                    problems, locale, namespace, key, translated[key], plural, locale
                )
    return problems


def arguments_match(
    source: tuple[tuple[str, str], ...],
    translated: tuple[tuple[str, str], ...],
    *,
    translated_source: str | None = None,
    locale: str | None = None,
    plural: dict[str, Any] | None = None,
) -> bool:
    """Whether a translation keeps en's arguments (name, order and VALUE).

    Kinds must be identical with ONE deliberate exception: a locale may render
    an en ``number`` as an ICU ``plural`` over the same argument — but ONLY
    when that upgrade is lossless (see :func:`_arguments_problem`): the
    translation's plural must carry every CLDR category of the locale, every
    branch must render the count, and no branch may be empty. Without those,
    the upgrade is a hole: a plural missing ``one`` cannot render count 1, and
    a branch without ``#``/``{count, number}`` silently drops the number.

    WHY (Urdu/Hindi duration units, 2026-10-20): en writes the countdown unit
    as a bare number — ``{count, number} h`` — because English's ``h`` does not
    inflect. A language whose counted noun DOES inflect has no invariant form
    to put there: Urdu needs ``1 گھنٹہ`` / ``3 گھنٹے`` and Hindi ``1 घंटा`` /
    ``3 घंटे``. Without the upgrade the key is wrong on nearly every render
    (the hours arm covers 1-47), and the wrongness is invisible to every
    mechanical check — the "5 минута" class. The upgrade preserves the
    argument's name, position and number VALUE; it only adds grammar.

    The reverse is NOT allowed: an en plural rendered as a bare number drops
    branches and loses the inflection, which is the defect this exists to
    catch. Every other kind change (``simple``/``number``/``date``/``select``)
    stays a hard failure — a translator must not change what a value IS.

    ``translated_source``/``locale``/``plural`` are required to judge an
    upgrade; omitting them fails CLOSED rather than skipping the guard. The
    tuple-only signature stays for callers comparing already-validated shapes;
    :func:`check_catalogues` and the parity test pass the context.
    """
    return (
        _arguments_problem(
            source,
            translated,
            translated_source=translated_source,
            locale=locale,
            plural=plural,
        )
        is None
    )


def _arguments_problem(
    source: tuple[tuple[str, str], ...],
    translated: tuple[tuple[str, str], ...],
    *,
    translated_source: str | None,
    locale: str | None,
    plural: dict[str, Any] | None,
) -> str | None:
    """Why ``translated`` breaks en's argument contract, or ``None``.

    Returning the reason rather than a bool keeps the checker's diagnostics
    specific: a reordered argument, a changed kind and a lossy ``number ->
    plural`` upgrade are different defects and must not read as one generic
    mismatch.
    """
    if len(source) != len(translated):
        return f"en has {len(source)} argument(s), translation has {len(translated)}"
    for index, ((source_name, source_kind), (translated_name, translated_kind)) in enumerate(
        zip(source, translated), start=1
    ):
        if source_name != translated_name:
            return (
                f"argument {index} is {translated_name!r}, en's is {source_name!r} "
                "(renamed or reordered)"
            )
        if source_kind == translated_kind:
            continue
        if source_kind == "number" and translated_kind == "plural":
            # Fail closed: a caller that cannot supply the message text and the
            # locale's CLDR table cannot prove the upgrade is lossless.
            if translated_source is None or locale is None or plural is None:
                return (
                    f"argument {translated_name!r} upgrades number -> plural without "
                    "the source/locale/plural table needed to validate the upgrade"
                )
            problem = _number_upgrade_problem(translated_source, translated_name, locale, plural)
            if problem is not None:
                return problem
            continue
        return f"argument {translated_name!r} changed kind {source_kind!r} -> {translated_kind!r}"
    return None


def _number_upgrade_problem(
    translated_source: str,
    argument: str,
    locale: str,
    plural: dict[str, Any],
) -> str | None:
    """Why a ``number -> plural`` upgrade is lossy, or ``None`` when sound.

    Three conditions, each a distinct real failure:

    1. The literal selectors cover the locale's FULL CLDR category set. A
       Russian plural with only ``other`` cannot render 1/2/5 correctly; a
       locale with only ``other`` (vi) is satisfied by that one category.
    2. Every branch renders the plural's value (``#`` or ``{arg, number}``).
       A branch without it silently drops the number from the sentence.
    3. No branch body is empty. ``other {}`` renders as nothing for every
       count outside the defined branches; it is a valueless branch and must
       never be accepted as grammar.
    """
    known = set(plural["locales"].get(locale, {}).get("categories", []))
    occurrences = [
        occurrence
        for occurrence in msg_runtime.plural_arguments(translated_source)
        if occurrence.argument == argument
    ]
    if not occurrences:
        return f"plural argument {argument!r} not found in the translation"
    for occurrence in occurrences:
        selectors = {
            branch.selector for branch in occurrence.branches if not branch.selector.startswith("=")
        }
        missing = sorted(known - selectors)
        if missing:
            return (
                f"plural {argument!r} is missing locale categories {missing} "
                f"(has {sorted(selectors)})"
            )
        for branch in occurrence.branches:
            if branch.empty:
                return f"plural {argument!r} branch {branch.selector!r} is empty"
            if not branch.renders_value:
                return (
                    f"plural {argument!r} branch {branch.selector!r} does not "
                    f"render the value (`#` or `{{{argument}, number}}`)"
                )
    return None


def _check_plural_categories(
    problems: list[str],
    locale: str,
    namespace: str,
    key: str,
    source: str,
    plural: dict[str, Any],
    table_locale: str,
) -> None:
    """Every literal plural selector must be a category of the locale."""
    known = set(plural["locales"].get(table_locale, {}).get("categories", []))
    try:
        selector_tuples = msg_runtime.plural_selectors(source)
    except msg_runtime.MessageSyntaxError:
        return  # reported by the argument check; not twice
    for selectors in selector_tuples:
        if "other" not in selectors:
            problems.append(f"{locale}/{namespace}: {key}: plural has no `other` branch")
        for selector in selectors:
            if selector.startswith("="):
                continue
            if selector not in known:
                categories = sorted(known)
                problems.append(
                    f"{locale}/{namespace}: {key}: plural category {selector!r} "
                    f"not in {categories}"
                )


def _check_context_sidecar(
    problems: list[str], locale: str, namespace: str, messages: dict[str, str]
) -> None:
    try:
        sidecar = catalogues.context(locale, namespace)
    except catalogues.CatalogueInvalid as exc:
        problems.append(f"{locale}/{namespace}: {exc}")
        return
    for key, entry in sidecar.items():
        if key not in messages:
            problems.append(f"{locale}/{namespace}: context sidecar describes unknown key {key!r}")
        # Authoring-time half of the shared schema (catalogues.CONTEXT_FIELDS):
        # read_public_copy refuses a WHOLE sidecar with one bad entry, so an
        # unknown or mistyped field must fail here, naming key and field,
        # before it can make the file unreadable. Values are never echoed.
        problem = catalogues.context_entry_problem(entry)
        if problem is None:
            continue
        kind, field = problem
        if kind == "not-object":
            detail = "must be an object"
        elif kind == "unsupported-field":
            detail = f"has unsupported field {field!r}"
        elif field == "maxCells":
            detail = "maxCells must be an int"
        else:
            detail = f"field {field!r} has an invalid type"
        problems.append(
            # i18n: ignore checker diagnostic (operator tooling output, English-only)
            f"{locale}/{namespace}: context sidecar {key!r} {detail}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _allowance_ids(data: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Every ``(locale, namespace, key)`` the allowance records as missing."""
    ids: set[tuple[str, str, str]] = set()
    for entry in data.get("pairs", []):
        locale = entry["locale"]
        namespace = entry["namespace"]
        for key in entry.get("missing", []):
            ids.add((locale, namespace, key))
    return ids


def _locale_keys_at_ref(ref: str, locale: str) -> set[str] | None:
    """Every key the locale carries in ANY namespace at ``ref``, or ``None``.

    Presence is asked of the whole locale, not of the namespace the debt is
    recorded under. A key moved from a nested namespace to its parent would
    otherwise look absent at base — the target namespace was never carried —
    and a dropped translation would pass as pre-existing debt.

    Unreadable returns ``None`` so the caller fails closed.
    """
    listing = _git(
        "ls-tree",
        "-r",
        "--name-only",
        ref,
        "--",
        f"local_operator/i18n/catalogues/{locale}",
    )
    if listing.returncode != 0:
        return None
    keys: set[str] = set()
    for path in listing.stdout.splitlines():
        if not path.endswith(".json") or path.endswith(catalogues.CONTEXT_SUFFIX):
            continue
        shown = _git("show", f"{ref}:{path}")
        if shown.returncode != 0:
            return None
        try:
            messages = json.loads(shown.stdout)
        except ValueError:
            return None
        if not isinstance(messages, dict):
            return None
        keys.update(messages)
    return keys


def allowance_growth(
    base: dict[str, Any],
    head: dict[str, Any],
    *,
    ref: str | None = None,
    base_english: set[str] | None = None,
    inactive_locales: set[str] | None = None,
) -> set[tuple[str, str, str]]:
    """Added allowance entries that are growth in this change.

    The shrink-only rule exists to stop untranslated copy being laundered
    into the tree. An added entry is legitimate in exactly two cases:

    1. **The debt was already real.** The ``(locale, namespace, key)`` was
       genuinely missing at the base ref — the namespace was uncarried or
       the key was absent — so this change only *records* what was already
       true. Recording it is the whole point of the allowance: without it
       the manifest gate is red about a fact it knows.
    2. **The locale cannot carry a translation at all.** It has no catalogue
       directory at the base ref, so a key it is missing is untranslatable
       rather than untranslated. Refusing this would make the two gates
       mutually unsatisfiable for every change that adds English copy — a
       deadlock, not a ratchet.

    Everything else is growth and must fail:

    - a key that **is present** at base and missing at head: the change
      deleted a translation to record debt, which is the reverse of the
      rule (``translate-then-drop``);
    - a new English key in an **active** locale: the copy is translatable,
      so it must be translated rather than allowanced.

    ``ref`` is the resolved base ref. ``None`` means the base tree could not
    be read, so nothing is provably pre-existing and every added entry is
    growth — an unverifiable comparison must not weaken the ratchet.
    ``inactive_locales`` of ``None`` is treated as empty for the same
    reason.
    """
    added = _allowance_ids(head) - _allowance_ids(base)
    if not added:
        return set()
    if ref is None or base_english is None:
        return added
    # Exempt only while the locale STILL carries no catalogue. One that gains
    # its first catalogue in this change is translatable from that moment, so
    # recording brand-new English as debt there would bypass the "new English
    # must be translated" rule through a base-ref view of inactivity.
    head_locales = set(catalogues.locales())
    inactive = (inactive_locales or set()) - head_locales
    growth: set[tuple[str, str, str]] = set()
    for locale, namespace, key in added:
        if locale in inactive:
            # Untranslatable at base and still uncarried: recording is lawful.
            continue
        if key not in base_english:
            # New English copy. An active locale must translate it.
            growth.add((locale, namespace, key))
            continue
        carried = _locale_keys_at_ref(ref, locale)
        if carried is None:
            # Unreadable base for this locale: fail rather than assume absence.
            growth.add((locale, namespace, key))
            continue
        if key in carried:
            # Present somewhere in this locale at base, missing at head: a
            # translation was removed, including via a namespace move.
            growth.add((locale, namespace, key))
    return growth


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)


def _ci_base() -> tuple[str | None, str | None]:
    """A GHA PR merge's verified first parent, never its queued event's old base.

    The event and merge ref can advance independently while a run is queued.
    Raw commit headers retain parents even at a shallow boundary; rev-list does
    not. Platform identity + second-parent linkage bind the first parent to THIS
    checkout. merge_commit_sha has the same stale race and is not required.
    """
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return None, None
    github = os.environ.get("GITHUB_ACTIONS") == "true"
    try:
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if not event_path:
            raise ValueError("missing_event")
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        request = event["pull_request"]
        base = request["base"]
        if not github:
            return base["sha"], base["ref"]
        number = event["number"]
        feature = request["head"]["sha"]
        branch = base["ref"]
        if (
            type(number) is not int
            or number <= 0
            or not isinstance(feature, str)
            or not re.fullmatch(r"[0-9a-f]{40}", feature)
            or not isinstance(branch, str)
            or not branch
        ):
            raise ValueError("invalid_event")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if github:
            raise ValueError("pr_checkout_provenance:invalid_event") from exc
        return None, None
    if os.environ.get("GITHUB_REF") != f"refs/pull/{number}/merge":
        raise ValueError("pr_checkout_provenance:merge_ref_mismatch")
    expected = os.environ.get("GITHUB_SHA", "")
    head = _git("rev-parse", "--verify", "HEAD^{commit}")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", expected)
        or head.returncode != 0
        or head.stdout.strip() != expected
    ):
        raise ValueError("pr_checkout_provenance:head_mismatch")
    raw = _git("cat-file", "-p", expected)
    if raw.returncode != 0:
        raise ValueError("pr_checkout_provenance:commit_unreadable")
    parents = [
        line[7:]
        for line in raw.stdout.split("\n\n", 1)[0].splitlines()
        if line.startswith("parent ")
    ]
    if len(parents) != 2 or any(not re.fullmatch(r"[0-9a-f]{40}", p) for p in parents):
        raise ValueError("pr_checkout_provenance:exact_two_parents_required")
    if parents[1] != feature:
        raise ValueError("pr_checkout_provenance:head_parent_mismatch")
    actual = _git("rev-parse", "--verify", f"{parents[0]}^{{commit}}")
    if actual.returncode != 0 or actual.stdout.strip() != parents[0]:
        raise ValueError("pr_checkout_provenance:first_parent_unavailable")
    return parents[0], branch


def _resolve_allowance_base(base: str | None) -> str | None:
    """The base ref to compare, or ``None`` when none is resolvable.

    An explicit ref or the GitHub PR base SHA is used when available; the PR
    base is fetched when the checkout is shallow. An isolated fixture, exported
    tarball or detached copy has no base ref, and the caller must SKIP there —
    an unknown base is not a catalogue failure.
    """
    if _git("rev-parse", "--is-inside-work-tree").returncode != 0:
        return None
    github_pr = (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("GITHUB_EVENT_NAME") == "pull_request"
    )
    ci_sha, ci_branch = _ci_base() if base is None or github_pr else (None, None)
    if github_pr:
        # _ci_base already proved a readable commit. Never replace a failed PR
        # proof with an event base, branch tip, fetch result or caller override.
        if ci_sha is None:
            raise ValueError("pr_checkout_provenance:base_unavailable")
        if base is not None:
            supplied = _git("rev-parse", "--verify", f"{base}^{{commit}}")
            if supplied.returncode != 0 or supplied.stdout.strip() != ci_sha:
                raise ValueError("pr_checkout_provenance:base_override_mismatch")
        return ci_sha
    ref = base or ci_sha or "origin/main"
    if _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode == 0:
        return ref
    # actions/checkout fetches one commit by default; the PR base is not in the
    # object database. One small fetch is enough to compare the allowance. If
    # the runner refuses a fetch by SHA, the base branch tip is the fallback.
    fetched = _git("fetch", "--no-tags", "--depth=1", "origin", ref)
    if fetched.returncode == 0:
        return "FETCH_HEAD"
    if ci_branch:
        fetched = _git("fetch", "--no-tags", "--depth=1", "origin", ci_branch)
        if fetched.returncode == 0:
            return "FETCH_HEAD"
    return None


def check_audit_history(base: str | None = None, *, structural_only: bool = False) -> list[str]:
    """Normal gate: structure plus actual-base append-only proof, fail closed in CI."""
    from scripts.i18n import ledger

    try:
        ledger.load_entries(REPO)
        if structural_only:
            if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
                raise ValueError("structural-only audit validation is forbidden in CI")
            return []
        # Repo-less checker fixtures with no audit storage exercise other gates.
        if not (REPO / "i18n/ledger.json").exists() and not (REPO / "i18n/ledger").exists():
            if _git("rev-parse", "--is-inside-work-tree").returncode != 0 and not os.environ.get(
                "CI"
            ):
                return []
        ci_sha, _ = _ci_base()
        if os.environ.get("GITHUB_EVENT_NAME") == "pull_request" and not ci_sha:
            raise ValueError("actual PR audit base is unavailable")
        ref = _resolve_allowance_base(ci_sha or base or "origin/main")
        if ref is None:
            raise ValueError(
                "readable audit base required; use --audit-base REF "
                "(isolated fixtures: --audit-structural-only)"
            )
        if ci_sha and base:
            override = _git("rev-parse", "--verify", f"{base}^{{commit}}")
            actual = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
            if override.returncode != 0 or override.stdout != actual.stdout:
                raise ValueError("--audit-base must name the actual PR base in CI")
        ledger.validate_base(REPO, ref)
    except (ValueError, OSError) as exc:
        return [f"audit history: {exc}"]
    return []


def _english_keys_at_ref(ref: str) -> set[str] | None:
    """Qualified key IDs in ``ref``'s English catalogues, or ``None``.

    Read from the committed tree rather than the working copy: the question
    is what English copy existed at the base, not what exists now. Any
    unreadable piece (missing ref, unparsable file, interrupted listing)
    returns ``None`` as a whole — a partial set would silently permit
    entries the comparison could not verify.
    """
    listing = _git("ls-tree", "-r", "--name-only", ref, "--", "local_operator/i18n/catalogues/en")
    if listing.returncode != 0:
        return None
    paths = [
        path
        for path in listing.stdout.splitlines()
        if path.endswith(".json") and not path.endswith(catalogues.CONTEXT_SUFFIX)
    ]
    if not paths:
        return None
    keys: set[str] = set()
    for path in paths:
        shown = _git("show", f"{ref}:{path}")
        if shown.returncode != 0:
            return None
        try:
            messages = json.loads(shown.stdout)
        except ValueError:
            return None
        if not isinstance(messages, dict):
            return None
        keys.update(messages)
    return keys


def _english_keys_in_tree() -> set[str] | None:
    """Qualified key IDs in the working tree's English catalogues, or ``None``.

    The head side of the comparison is the tree CI checks out, so a new key
    counts as soon as the change adds it. Unreadable or empty means the
    comparison cannot run and the caller must fail conservatively.
    """
    directory = REPO / "local_operator" / "i18n" / "catalogues" / "en"
    paths = [
        path
        for path in directory.glob("*.json")
        if not path.name.endswith(catalogues.CONTEXT_SUFFIX)
    ]
    if not paths:
        return None
    keys: set[str] = set()
    for path in paths:
        try:
            messages = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(messages, dict):
            return None
        keys.update(messages)
    return keys


def _inactive_locales_at_ref(ref: str) -> set[str] | None:
    """Language codes that carry no catalogue directory at ``ref``.

    A locale with no ``catalogues/<code>/`` at the base cannot hold a
    translation there, so a key it is missing is untranslatable rather than
    untranslated. Those are the only locales for which adding allowance debt
    in a change that also grows English copy is legitimate (see
    ``allowance_growth``).

    Read from ``i18n/languages.json`` at the same ref, skipping the source
    language. Returns ``None`` when the manifest cannot be read, so the
    caller treats the set as empty and the exemption cannot widen by
    accident.
    """
    shown = _git("show", f"{ref}:i18n/languages.json")
    if shown.returncode != 0:
        return None
    try:
        manifest = json.loads(shown.stdout)
    except ValueError:
        return None
    languages = manifest.get("languages") if isinstance(manifest, dict) else None
    if not isinstance(languages, list):
        return None
    inactive: set[str] = set()
    for entry in languages:
        if not isinstance(entry, dict):
            return None
        code = entry.get("code")
        if not isinstance(code, str) or not code:
            return None
        if code == "en":
            continue
        listing = _git(
            "ls-tree",
            "-r",
            "--name-only",
            ref,
            "--",
            f"local_operator/i18n/catalogues/{code}",
        )
        if listing.returncode != 0:
            return None
        if not listing.stdout.strip():
            inactive.add(code)
    return inactive


def check_allowance_growth(base: str | None = None) -> list[str]:
    """Fail when HEAD adds missing-key entries relative to the base branch.

    This is the automated half of the allowance's shrink-only rule. It cannot
    stop someone editing the file, but an added entry becomes a red check that
    names the debt, so growth is a deliberate reviewed act rather than a
    silent escape. A base branch without the file is the introduction case.

    An addition is permitted in two cases only (see ``allowance_growth``):
    the key was already missing at the base ref, so this change records debt
    the tree already carried; or the locale carries no catalogue at the base
    ref, so the key is untranslatable rather than untranslated. It is growth
    when a translation was removed, when the key is new English in an active
    locale, or when the base cannot be read well enough to tell.
    """
    try:
        ref = _resolve_allowance_base(base)
    except ValueError as exc:
        return [str(exc)]
    # A supplied ref is operator input, not a best-effort default: inside git,
    # a typo must fail rather than silently disable the growth comparison.
    # Repo-less fixtures still skip because no ref can resolve there.
    if ref is None and base is not None:
        if _git("rev-parse", "--is-inside-work-tree").returncode == 0:
            return [f"allowance growth: explicit base ref cannot be resolved: {base}"]
    if ref is None:
        # Fail closed on a pull request ONLY where a base was supposed to
        # resolve: a git work tree. A monkeypatched fixture, exported tarball
        # or detached copy has no base ref by construction and must skip —
        # which is the same carve-out the explicit-ref branch above already
        # makes. Without the work-tree guard, every checker test running under
        # CI's pull-request event fails on a repo-less temp directory.
        if (
            os.environ.get("GITHUB_EVENT_NAME") == "pull_request"
            and _git("rev-parse", "--is-inside-work-tree").returncode == 0
        ):
            return ["allowance growth: base ref cannot be resolved on a pull request"]
        print(
            # i18n: ignore checker diagnostic (operator tooling output, English-only)
            "allowance growth: skipped (no base ref)"
        )
        return []
    allowance_path = "i18n/coverage-allowance.json"
    present = _git("ls-tree", "--name-only", ref, "--", allowance_path)
    if present.returncode != 0:
        # Cannot tell whether the base carried an allowance. Failing open
        # here would silently disable the growth check for the whole run.
        return [f"allowance growth: base ref {ref} cannot be inspected"]
    if not present.stdout.strip():
        # The base had no allowance file: this change introduces it.
        return []
    shown = _git("show", f"{ref}:{allowance_path}")
    if shown.returncode != 0:
        return [f"allowance growth: base allowance at {ref} cannot be read"]
    try:
        base_data = json.loads(shown.stdout)
        head_data = json.loads(ALLOWANCE.read_text(encoding="utf-8"))
        if not isinstance(head_data, dict) or not isinstance(base_data, dict):
            return ["allowance growth: allowance root must be a JSON object"]
        base_english = _english_keys_at_ref(ref)
        head_english = _english_keys_in_tree()
        if base_english is None or head_english is None:
            new_english: set[str] | None = None
        else:
            new_english = head_english - base_english
        inactive = _inactive_locales_at_ref(ref)
        added = allowance_growth(
            base_data,
            head_data,
            ref=None if new_english is None else ref,
            base_english=base_english,
            inactive_locales=inactive,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"allowance growth: cannot compare allowances: {exc}"]
    if new_english is None:
        # Say so even when nothing was added: the verdict is only trustworthy
        # if a reader knows the English key sets could not be read.
        print(
            # i18n: ignore checker diagnostic (operator tooling output, English-only)
            "allowance growth: base or head English catalogues unreadable; "
            "every added entry counts as growth (conservative fallback)"
        )
    return [
        f"allowance growth: added {locale}/{namespace}: {key}"
        for locale, namespace, key in sorted(added)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--init", action="store_true", help="write the baseline from the current tree (one-time)"
    )
    mode.add_argument(
        "--update", nargs="+", metavar="PATH", help="lower the ceilings for these files"
    )
    mode.add_argument(
        "--catalogues-only",
        action="store_true",
        help="run the catalogue checks alone (used by tests and fast local runs)",
    )
    parser.add_argument(
        "--force", action="store_true", help="allow --init over an existing baseline"
    )
    parser.add_argument(
        "--allowance-base",
        metavar="REF",
        # i18n: ignore checker diagnostic (operator tooling help, English-only)
        help="fail when i18n/coverage-allowance.json adds entries relative to REF; "
        "defaults to the pull request base in CI, then origin/main locally",
    )
    parser.add_argument(
        "--audit-base",
        metavar="REF",
        # i18n: ignore developer-only CLI help; i18n tooling owns this while checkout-only
        help="actual base for append-only audit history validation",
    )
    parser.add_argument(
        "--audit-structural-only",
        action="store_true",
        # i18n: ignore developer-only CLI help; i18n tooling owns this while checkout-only
        help="isolated local fixture validation only; forbidden in CI",
    )
    args = parser.parse_args(argv)

    if args.catalogues_only:
        problems = check_catalogues()
        for line in problems:
            print(line)
        print(f"check.py: catalogue checks: {len(problems)} problem(s)")
        return 1 if problems else 0

    entries = load_allowlist(ALLOWLIST)
    scanned = scan_tree(REPO)
    counts = _counts(scanned, entries)
    baseline = load_baseline(BASELINE)
    enforced = baseline["enforced"]
    files = baseline["files"]

    if args.init:
        # ANY existing baseline refuses, empty or not: an empty file used to be
        # treated as "no baseline" by a truthiness test and silently
        # overwritten without `--force` (round-1 n3).
        if BASELINE.is_file() and not args.force:
            print(
                "check.py --init: a baseline already exists; use --force only for a reviewed "
                "scanner-coverage change (say so in the PR body)."
            )
            return 2
        preserved_enforced, note = _activation_state(BASELINE)
        save_baseline(BASELINE, counts, enforced=preserved_enforced, note=note)
        total = sum(counts.values())
        print(
            f"check.py --init: baseline written: {len(counts)} file(s), {total} literal(s); "
            f"enforced: {', '.join(preserved_enforced) or 'none'}"
        )
        return 0

    if args.update:
        for rel in args.update:
            rel = Path(rel).as_posix()
            if allowlisted(rel, entries) is not None:
                print(f"check.py --update: {rel} is allowlisted; nothing to lower")
                continue
            new = counts.get(rel, 0)
            old = files.get(rel)
            if old is None:
                if new > 0:
                    print(
                        f"check.py --update: {rel} has {new} literal(s) with no recorded "
                        "ceiling — a file with no recorded ceiling starts at 0: extract, "
                        "pragma, or allowlist instead of recording one."
                    )
                    return 1
                continue
            if new > old:
                print(
                    f"check.py --update: {rel} went {old} -> {new}; the baseline may only stay "
                    "equal or go down."
                )
                return 1
            if new:
                files[rel] = new
            else:
                files.pop(rel, None)
        save_baseline(BASELINE, files, enforced=enforced, note=baseline["note"])
        print(f"check.py --update: baseline now {len(files)} file(s)")
        return 0

    failures: list[str] = []
    advisories: list[str] = []
    for finding in findings(counts, files):
        line = finding.line()
        if in_enforced_scope(finding.path, enforced):
            failures.append(line)
        else:
            advisories.append(f"advisory: {line}")
    problems = (
        check_catalogues()
        + check_login(REPO)
        + check_allowance_growth(args.allowance_base)
        + check_audit_history(args.audit_base, structural_only=args.audit_structural_only)
    )
    for line in advisories + failures + problems:
        print(line)
    if failures or problems:
        print(
            f"check.py: FAILED — {len(failures)} ratchet finding(s) in the enforced scope, "
            f"{len(problems)} catalogue problem(s) ({len(advisories)} advisory finding(s) "
            "outside it). Extract the string, add `# i18n: ignore <reason>`, or (for a "
            "file-level exemption) record it in i18n/allowlist.toml with a reason."
        )
        return 1
    total = sum(counts.values())
    suffix = (
        f"; {len(advisories)} advisory finding(s) (report-only until activated)"
        if advisories
        else ""
    )
    print(
        f"check.py: ok — {len(scanned)} file(s) scanned, {total} literal(s) at or "
        f"below the baseline, catalogues consistent{suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
