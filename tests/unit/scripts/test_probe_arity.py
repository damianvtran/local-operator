"""Probes under ``scripts/`` are strings, so nothing type-checks their calls.

A bench drives the real harness by handing Python SOURCE to a child interpreter
(``python -c``): ``_SESSION_PROBE`` and ``_TUI_PROBE`` in
``scripts/bench_base_overhead.py`` build a real session, ``_EXEC_PROBE`` runs the
real console entry point. The call inside that string is real code at run time,
but to every static tool it is a string literal — pyright analyzes the module and
never looks inside the constant, which is why the whole-tree ``type-check`` job
cannot turn red on a signature change in the harness.

That is exactly how the repo's headline base-overhead instrument sat broken:

* #1448 (merged 2026-09-23) removed the credential manager from
  ``session_factory.create_session``, leaving ``(args, config_manager,
  agent_registry)`` — three positionals.
* the script's probes kept passing four (``args, ConfigManager(...),
  Path(config_dir), AgentRegistry(...)``), so every run died with
  ``TypeError: create_session() takes 3 positional arguments but 4 were given``
  and the benchmark could not measure anything at all.
* nothing failed loudly, because nothing executes the script in CI. Meanwhile
  its numbers are the cited source of the pinned constants in
  ``local_operator/model/configure.py``, ``compaction/tokens.py``,
  ``compaction/pruning.py``, ``helpers.py``, ``cli.py`` and ``docs/VERIFICATION.md``
  — so a silent breakage means every perf claim resting on it is unverifiable.

The guard is deliberately static and cheap (no subprocess, no interpreter boot):
it compiles every string constant under ``scripts/`` that looks like a program,
binds each call it makes to a symbol it imported from ``local_operator`` against
that symbol's live signature, and reports the ones that cannot bind. It covers
both halves of the same rot — an arity that moved, and a symbol or module that
was deleted (``from local_operator.credentials import CredentialManager``).

It cannot check a call built at run time (``getattr``, a name assembled from
data) or one that unpacks ``*args``/``**kwargs``; those are skipped rather than
guessed at. It also says nothing about what a correctly-bound probe MEASURES —
``tests/unit/scripts/test_bench_*.py`` are where a bench's numbers are pinned.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"


@dataclass(frozen=True)
class _ProbeCall:
    """One call in a probe string to a name that probe imported from the tree."""

    path: Path
    line: int
    module: str
    name: str
    call: ast.Call

    @property
    def where(self) -> str:
        # The synthetic scans that pin this guard hand in a path OUTSIDE the
        # repo, so the repo-relative form is a rendering preference and never an
        # assumption: a `relative_to` that raised here would turn "this call is
        # stale" into a crash while REPORTING it (found by the known-positive
        # control below, which is the only reason it is not in the shipped
        # version).
        try:
            path: Path | str = self.path.relative_to(REPO)
        except ValueError:
            path = self.path
        return f"{path}:{self.line} calls {self.name}()"


def _program_strings(path: Path) -> Iterator[ast.Module]:
    """Yield the string constants in ``path`` that hold a PROGRAM.

    A string is a program when it parses as a module and declares an import —
    which is what keeps prose out. A docstring sentence parses as a bare
    expression, and one that merely mentions an import carries no ``Import`` or
    ``ImportFrom`` node to find.

    ``ast.parse`` on arbitrary text is safe: it never executes anything, and the
    only requirement on a candidate is that it can hold a call at all.
    """
    try:
        source = path.read_text(errors="replace")
    except OSError:
        return
    try:
        module = ast.parse(source)
    except SyntaxError:  # an unparseable script is a different guard's problem
        return
    for node in ast.walk(module):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        # Cheap text filter before the parse: a probe that touches this tree
        # names it, and almost no docstring does.
        if "local_operator" not in node.value or "import" not in node.value:
            continue
        try:
            parsed = ast.parse(node.value)
        except SyntaxError:
            continue
        if any(isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(parsed)):
            yield parsed


def _imported_names(parsed: ast.Module) -> dict[str, str]:
    """Map each name a probe string bound to a ``local_operator`` import onto the
    dotted path it came from.

    Only names the probe itself imports are resolved, so a call to the probe's
    own helper is never checked against anything.
    """
    names: dict[str, str] = {}
    for node in ast.walk(parsed):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module != "local_operator" and not node.module.startswith("local_operator."):
            continue
        for alias in node.names:
            names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _probe_calls(root: Path = SCRIPTS) -> list[_ProbeCall]:
    """Every checkable call in every probe string under ``root``."""
    found: list[_ProbeCall] = []
    for path in sorted(root.rglob("*.py")):
        for parsed in _program_strings(path):
            names = _imported_names(parsed)
            if not names:
                continue
            for node in ast.walk(parsed):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                dotted = names.get(node.func.id)
                if dotted is None:
                    continue
                # ``*args``/``**kwargs`` hide the arity from a static scan; a
                # guess here would report a violation that does not exist.
                if any(isinstance(arg, ast.Starred) for arg in node.args):
                    continue
                if any(keyword.arg is None for keyword in node.keywords):
                    continue
                module, _, name = dotted.rpartition(".")
                found.append(_ProbeCall(path, node.lineno, module, name, node))
    return found


def _violation(found: _ProbeCall) -> str | None:
    """Why ``found`` cannot run, or ``None`` when it binds the live signature."""
    try:
        target: Any = getattr(importlib.import_module(found.module), found.name)
    except (ImportError, AttributeError) as exc:
        return f"{found.where}: {found.module}.{found.name} is gone ({exc.__class__.__name__})"
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return f"{found.where}: {found.module}.{found.name} is not callable as a function"
    keywords: dict[str, object] = {}
    for keyword in found.call.keywords:
        assert keyword.arg is not None  # filtered in _probe_calls
        keywords[keyword.arg] = object()
    try:
        signature.bind(*[object()] * len(found.call.args), **keywords)
    except TypeError as exc:
        return (
            f"{found.where}: {exc} — the live signature is "
            f"{found.module}.{found.name}{signature}"
        )
    return None


def probe_violations(root: Path = SCRIPTS) -> list[str]:
    """Every probe call under ``root`` that the live harness would reject."""
    return [message for call in _probe_calls(root) if (message := _violation(call))]


def test_every_probe_call_in_scripts_matches_the_live_harness() -> None:
    """Tree-wide: a red here means the probe string no longer runs as written.

    Fix the PROBE, not the harness signature: the signature is the live contract
    every other caller already follows, and the probe is the stale side.
    """
    violations = probe_violations()
    assert not violations, (
        "a probe string under scripts/ calls the harness in a shape it no longer "
        "accepts. pyright cannot see these calls (they are string literals handed "
        "to a child interpreter), so this list is the only thing that goes red:\n  "
        + "\n  ".join(violations)
    )


def test_the_scan_reaches_the_real_probes() -> None:
    """A scan that matched nothing would stay green forever.

    The known-positive is a probe that exists in the tree today: if
    ``create_session`` stops being seen, the guard above has lost its teeth and
    this test says so instead of quietly passing.
    """
    names = {call.name for call in _probe_calls()}
    assert "create_session" in names, f"the scan found no create_session probe; saw {sorted(names)}"
    assert "AgentRegistry" in names, f"the scan found no registry probe; saw {sorted(names)}"


#: A benchmark probe the way the real scripts spell it: a string literal handed
#: to a child interpreter. The file below is therefore CODE THAT WRAPS CODE —
#: the outer module is what pyright checks, the inner string is what runs.
_STALE = '''\
PROBE = """
import argparse
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.session_factory import create_session


async def build():
    session = await create_session(
        args, ConfigManager(Path(d)), Path(d), AgentRegistry(Path(d))
    )
"""
'''

_CURRENT = _STALE.replace("Path(d), AgentRegistry", "AgentRegistry")

_GONE = '''\
PROBE = """
from local_operator.credentials import CredentialManager

manager = CredentialManager(Path(d))
"""
'''


def _scan(tmp_path: Path, source: str) -> list[str]:
    (tmp_path / "probe_target.py").write_text(source)
    return probe_violations(tmp_path)


def test_the_scan_reports_the_four_positional_call_that_broke(tmp_path: Path) -> None:
    """The known-positive: the exact shape #1448 left behind."""
    violations = _scan(tmp_path, _STALE)
    assert len(violations) == 1, violations
    assert "calls create_session()" in violations[0]
    assert "too many positional arguments" in violations[0]


def test_the_scan_accepts_the_same_probe_with_the_live_arity(tmp_path: Path) -> None:
    """The known-negative, through the same instrument: only the stray
    positional separates the two, so a scan that always reported would fail
    here and a scan that never reported would fail above."""
    assert _scan(tmp_path, _CURRENT) == []


def test_the_scan_reports_a_probe_importing_a_symbol_that_was_deleted(tmp_path: Path) -> None:
    """The other half of the same rot: the module itself is gone (#1448)."""
    violations = _scan(tmp_path, _GONE)
    assert len(violations) == 1, violations
    assert "local_operator.credentials.CredentialManager is gone" in violations[0]
