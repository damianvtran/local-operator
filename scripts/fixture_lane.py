#!/usr/bin/env python3
"""Plan the AFFECTED lane of the PR merge gate (full-suite.yml), #4074.

WHY: the merge gate runs the whole unit suite (6 shards x 2 Pythons) for every
labelled PR. A PR that only edits the bodies of existing, self-contained unit
test modules cannot change what any other test imports or collects, so for
exactly that shape the gate may run just those files on both Python versions.
`main`, the nightly and every dispatch always run the full suite as the
backstop (this planner answers FULL for any event but `pull_request`).

THE RULE IS "ANYTHING UNKNOWN IS FULL". FULL is the stronger lane, so every
error, timeout, missing input or unrecognised shape resolves to it; the lane
is `affected` only when every condition below is proven. Nothing here widens
what a PR may skip beyond that proof, and nothing is a fail-open.

ELIGIBLE (v1): 1..10 files, every one a git status `M` of a regular 100644
file at `tests/unit/**/test_*.py`; no path on ci_scope's WINDOWS sets or
affected_tests' INFRA set, no conftest/__init__/TUI/e2e/integration/perf/bulk/
native-looking path; the module's import statements are byte-identical in AST
between base and head; every module-level statement of BOTH versions is inside
a small side-effect-free subset (see `_module_ok`); and no other tracked
test/script/workflow file mentions the module name (`git grep -w`).

NOT ELIGIBLE (v1, FULL): new test files, data fixtures, helper modules,
renames, deletions, mode changes, any product/workflow/config path.

POLICY TRADE-OFF, stated rather than hidden: this isolates SELECTION; it is
not a proof that a test body cannot leak global state or that no consumer
loads the module dynamically (`importlib` with a computed name). Those residual
cases are what the full run on `main` and the nightly exist for.

TRUST: the workflow extracts this file and its two imports (ci_scope,
affected_tests) from the TESTED BASE commit, never from the PR, so a PR that
edits the planner cannot grant itself the lane. The base lacking this file
means FULL; the PR that introduces it therefore gets a full first gate.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

MAX_FILES = 10
MAX_BYTES = 512 * 1024  # a bigger module is not a "tiny self-contained edit"
GREP_TIMEOUT_S = 120
GIT_TIMEOUT_S = 60

TEST_PATH = re.compile(r"^tests/unit/(?:[A-Za-z0-9_.\-]+/)*test_[A-Za-z0-9_.\-]*\.py$")
# Substrings that mark a path whose behaviour reaches beyond its own module
# (layout/TUI, process-bound, real-network, native, bulk, perf): FULL in v1.
DENY_WORDS = ("tui", "e2e", "integration", "perf", "bench", "bulk", "native", "layout", "windows")
# Shard manifests list every test file by path; they are weights, not importers.
MANIFESTS = ("tests/durations.json", "tests/durations-e2e.json")
GREP_PATHS = ("tests", "scripts", ".github", "conftest.py")
# Decorator/base names whose evaluation at import is known and inert.
BASES_OK = {"unittest.TestCase", "TestCase"}
FIXTURE_SCOPES = {"function", "class", "module"}
SHADOW = {"pytest", "re", "unittest", "TestCase"}


# WHOLE-TREE READERS (R1-M1). A few unit tests scan every test file's text, so
# editing ANY test body can flip them although no changed file names them. The
# affected lane therefore always adds these nodes to its targets. Source-verified
# roots: both walk the tests tree (test_ask_tool: parents[2] -> tests/, rglob;
# test_notification_isolation: parents[1] -> tests/unit, rglob). They are a
# CATALOG, not a proof that no other reader exists: a reader outside it is a
# known limit of this planner, backstopped by the full run on main/nightly.
# A catalog entry that no longer resolves to a function at HEAD is FULL, and a
# PR that edits a reader's own file is FULL (it changes the guard itself).
GLOBAL_READERS: tuple[str, ...] = (
    "tests/unit/tools/test_ask_tool.py::test_no_test_still_pins_the_retired_refusal_sentence",
    "tests/unit/test_notification_isolation.py"
    "::test_no_test_nests_a_fixture_scoped_patch_under_the_opt_in",
)


class Full(Exception):
    """Raised with the reason a diff is not eligible; the lane is FULL."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    # arg-list only: no shell, nothing from the PR is ever interpolated.
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=root,
        capture_output=True,
        timeout=GIT_TIMEOUT_S,
        check=False,
    )


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _name(node.value)
        return f"{head}.{node.attr}" if head else None
    return None


def _pure(node: ast.AST, *, annotation: bool = False) -> bool:
    """Evaluating `node` at import runs no caller-defined code."""
    if isinstance(node, (ast.Constant, ast.Name)):
        return True
    if isinstance(node, ast.Attribute):
        return _name(node) is not None
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_pure(e, annotation=annotation) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _pure(k) and _pure(v) for k, v in zip(node.keys, node.values))
    if isinstance(node, ast.UnaryOp):
        return _pure(node.operand)
    if isinstance(node, ast.BinOp) and annotation:
        return _pure(node.left, annotation=True) and _pure(node.right, annotation=True)
    if isinstance(node, ast.Subscript) and annotation:
        return _pure(node.value) and _pure(node.slice, annotation=True)
    if isinstance(node, ast.Call):
        callee = _name(node.func)
        known = (
            callee == "re.compile"
            or callee == "pytest.param"
            or (callee or "").startswith("pytest.mark.")
        )
        return (
            known
            and all(_pure(a) for a in node.args)
            and all(k.arg is not None and _pure(k.value) for k in node.keywords)
        )
    return False


def _decorator_ok(node: ast.expr) -> bool:
    call = node if isinstance(node, ast.Call) else None
    callee = _name(call.func if call else node) or ""
    if callee.startswith("pytest.mark."):
        return call is None or _pure(call)
    if callee == "pytest.fixture":
        if call is None:
            return True
        for kw in call.keywords:
            # session/package/dynamic scope reaches other modules: FULL.
            if kw.arg == "scope" and not (
                isinstance(kw.value, ast.Constant) and kw.value.value in FIXTURE_SCOPES
            ):
                return False
        return all(_pure(a) for a in call.args) and all(
            k.arg is not None and _pure(k.value) for k in call.keywords
        )
    return False


def _args_ok(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    a = fn.args
    defaults = [d for d in (*a.defaults, *a.kw_defaults) if d is not None]
    notes = [
        x.annotation
        for x in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg)
        if x and x.annotation
    ]
    if fn.returns:
        notes.append(fn.returns)
    return all(_pure(d) for d in defaults) and all(_pure(n, annotation=True) for n in notes)


def _stmt_ok(stmt: ast.stmt, *, first: bool) -> bool:
    """One module- or class-level statement. Function BODIES are not inspected."""
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return not any(al.name == "*" for al in stmt.names)
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr):
        return first and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if len(names) != len(targets) or any(n in SHADOW or n.startswith("pytest_") for n in names):
            return False  # attribute/subscript writes, plugin hooks, alias rebinding
        # The ANNOTATION is evaluated at import too (module/class scope, no
        # `from __future__ import annotations` guarantee), so a call there is
        # import-time code exactly like a value. Same annotation-aware rule as
        # function annotations. Under the future import it is only stored as a
        # string, but the planner stays conservative: a hazard is FULL anyway.
        ann_ok = not isinstance(stmt, ast.AnnAssign) or _pure(stmt.annotation, annotation=True)
        return ann_ok and (stmt.value is None or _pure(stmt.value))
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if stmt.name in SHADOW or stmt.name.startswith("pytest_"):
            return False
        return all(_decorator_ok(d) for d in stmt.decorator_list) and _args_ok(stmt)
    if isinstance(stmt, ast.ClassDef):
        bases = [_name(b) for b in stmt.bases]
        if stmt.name in SHADOW or stmt.keywords or any(b not in BASES_OK for b in bases):
            return False  # unknown inheritance or a metaclass runs foreign code
        if not all(_decorator_ok(d) for d in stmt.decorator_list):
            return False
        return all(_stmt_ok(s, first=i == 0) for i, s in enumerate(stmt.body))
    return False  # control flow, calls, with, try, ... at import time


def _module_ok(tree: ast.Module) -> bool:
    return all(_stmt_ok(s, first=i == 0) for i, s in enumerate(tree.body))


def _imports(tree: ast.Module) -> list[str]:
    return sorted(
        ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
    )


def _parse(root: Path, rev: str, rel: str) -> ast.Module:
    out = _git(root, "show", f"{rev}:{rel}")
    if out.returncode != 0 or len(out.stdout) > MAX_BYTES:
        raise Full(f"{rel}: unreadable or over {MAX_BYTES} bytes at {rev[:12]}")
    try:
        return ast.parse(out.stdout)
    except (SyntaxError, ValueError) as exc:
        raise Full(f"{rel}: does not parse at {rev[:12]} ({exc.__class__.__name__})") from None


def _trusted_sets() -> frozenset[str]:
    """The ci_scope WINDOWS and affected_tests INFRA paths, from this same base."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import affected_tests
        import ci_scope
    except Exception as exc:  # ImportError, or a base module that fails to load
        raise Full(
            f"cannot import the base's ci_scope/affected_tests ({exc.__class__.__name__})"
        ) from None
    return frozenset(
        ci_scope.WINDOWS_TEST_PATHS
        | ci_scope.WINDOWS_CONFTEST_PATHS
        | ci_scope.WINDOWS_LOADED_PATHS
        | affected_tests.INFRA_PATHS
    )


def _raw_changes(root: Path, base: str, head: str) -> list[str]:
    out = _git(
        root, "diff", "--raw", "-z", "--no-abbrev", "--find-renames", "--find-copies", base, head
    )
    if out.returncode != 0:
        raise Full("git diff --raw failed")
    fields = out.stdout.decode("utf-8", "surrogateescape").split("\0")
    paths: list[str] = []
    i = 0
    while i + 1 < len(fields) and fields[i].startswith(":"):
        meta = fields[i].split()
        if len(meta) != 5 or meta[4] != "M" or meta[0] != ":100644" or meta[1] != "100644":
            raise Full(
                f"{fields[i + 1]}: not a plain in-place edit of a regular file ({fields[i]})"
            )
        paths.append(fields[i + 1])
        i += 2
    if i < len(fields) and fields[i]:
        raise Full("unparsed raw diff output")
    return paths


def _dependents(root: Path, targets: list[str]) -> None:
    """FULL when anything else names a target module (import, path, literal)."""
    for rel in targets:
        stem = rel.rsplit("/", 1)[-1][:-3]
        try:
            out = subprocess.run(
                ["git", "grep", "-l", "-I", "-w", "-F", "-e", stem, "--", *GREP_PATHS],
                cwd=root,
                capture_output=True,
                timeout=GREP_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise Full(f"{stem}: dependents search timed out") from None
        if out.returncode not in (0, 1):  # an error is not absence
            raise Full(f"{stem}: dependents search failed (exit {out.returncode})")
        hits = [
            h
            for h in out.stdout.decode("utf-8", "replace").splitlines()
            if h != rel and h not in MANIFESTS
        ]
        if hits:
            raise Full(
                f"{stem}: named by {hits[0]}"
                + (f" (+{len(hits) - 1} more)" if len(hits) > 1 else "")
            )


def _global_readers(root: Path, head: str, paths: list[str]) -> list[str]:
    """GLOBAL_READERS nodes to add; FULL if one is edited or no longer exists."""
    out: list[str] = []
    for node in GLOBAL_READERS:
        rel, _, func = node.partition("::")
        if rel in paths:
            raise Full(f"{rel}: edits a whole-tree guard itself")
        tree = _parse(root, head, rel)
        if not any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func
            for n in ast.walk(tree)
        ):
            raise Full(f"{node}: whole-tree guard catalog entry not found at head")
        out.append(node)
    return out


def plan(root: Path, *, event: str, base: str, head: str) -> tuple[str, list[str], str]:
    """`(lane, targets, reason)`; lane is `affected` or `full`."""
    try:
        if event != "pull_request":
            raise Full(f"event {event or '?'} always runs the full suite")
        paths = _raw_changes(root, base, head)
        if not 1 <= len(paths) <= MAX_FILES:
            raise Full(f"{len(paths)} changed files (eligible: 1..{MAX_FILES})")
        denied = _trusted_sets()
        for rel in paths:
            low = rel.lower()
            if not TEST_PATH.match(rel) or rel in denied or any(w in low for w in DENY_WORDS):
                raise Full(f"{rel}: not an eligible unit test module path")
        for rel in paths:
            old, new = _parse(root, base, rel), _parse(root, head, rel)
            if _imports(old) != _imports(new):
                raise Full(f"{rel}: imports changed")
            if not (_module_ok(old) and _module_ok(new)):
                raise Full(f"{rel}: module level is outside the side-effect-free subset")
        _dependents(root, paths)
        readers = _global_readers(root, head, paths)
        return (
            "affected",
            [*paths, *readers],
            f"{len(paths)} existing unit test module(s), bodies only, plus "
            f"{len(readers)} whole-tree guard node(s)",
        )
    except Full as why:
        return "full", [], str(why)
    except Exception as exc:  # any planner bug is FULL, never a skip
        return "full", [], f"planner error: {exc.__class__.__name__}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--root", default=".")
    ap.add_argument("--planner-blob", default="")
    ap.add_argument("--github-output")
    ap.add_argument("--summary")
    ns = ap.parse_args(argv)
    lane, targets, reason = plan(
        Path(ns.root).resolve(), event=ns.event, base=ns.base, head=ns.head
    )
    lines = [
        f"lane={lane}",
        f"targets={json.dumps(targets)}",
        # One output line each: a CR/LF in a reason must not forge a key.
        f"reason={reason.replace(chr(13), ' ').replace(chr(10), ' ')}",
        f"planner_base={ns.base}",
        f"planner_blob={ns.planner_blob}",
    ]
    if ns.github_output:
        with open(ns.github_output, "a", encoding="utf-8") as fh:
            fh.write("\n".join(line.replace("\r", " ").replace("\n", " ") for line in lines) + "\n")
    if ns.summary:
        with open(ns.summary, "a", encoding="utf-8") as fh:
            fh.write(
                f"### Test lane: {lane.upper()}\n\n{reason}\n\n"
                f"base `{ns.base}` planner blob `{ns.planner_blob}`\n"
            )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
