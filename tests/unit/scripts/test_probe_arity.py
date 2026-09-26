"""Probes under ``scripts/`` are strings, so nothing type-checks their calls.

A bench drives the real harness by handing Python SOURCE to a child interpreter
(``python -c``): ``_SESSION_PROBE`` and ``_TUI_PROBE`` in
``scripts/bench_base_overhead.py`` build a real session, ``_EXEC_PROBE`` runs the
real console entry point. The call inside that string is real code at run time,
but to every static tool it is a string literal — pyright analyzes the module and
never looks inside the constant, which is why the whole-tree ``type-check`` job
cannot turn red on a signature change in the harness.

That is exactly how the base-overhead instrument sat broken:

* #1448 (merged 2026-09-23) removed the credential manager from
  ``session_factory.create_session``, leaving ``(args, config_manager,
  agent_registry)`` — three positionals.
* the script's probes kept passing four (``args, ConfigManager(...),
  Path(config_dir), AgentRegistry(...)``), so the session build died with
  ``TypeError: create_session() takes 3 positional arguments but 4 were given``
  and the benchmark could not measure anything at all.
* the TUI cell did not even die. ``_TUI_PROBE`` returned a COMPLETE, plausible
  reading — ``seconds=13.70, ru_maxrss=193 MB, modules=1440`` — while its
  factory raised: ``OperatorApp`` swallows a factory exception, so ``run_test``
  finished normally and the benchmark reported a TUI number for a session that
  never existed. With the arity fixed the same cell reads ``modules=1510`` /
  ``ru_maxrss=241 MB``, which is the direct evidence that the pre-fix number
  was false rather than merely slow (measured, review round 1). That is
  AGENTS.md's *"a dead instrument returns a reading, not an error"* happening
  inside the instrument this guard repairs, and it is why the breakage went
  three days without being noticed: a stale probe can report a number instead
  of crashing.
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

**What it binds AGAINST is checked before anything is bound**, because
the interpreter it runs under is not necessarily this tree's: see
:func:`_foreign`. A run whose ``local_operator`` resolves to another checkout is
REFUSED — reported as a failure, one line per module — instead of being used,
which is the direction that matters: binding a stale probe against a sibling
tree's signature reports success while the scanned tree is broken.

Three shapes it cannot see, listed rather than left to be discovered:

* a call built at run time (``getattr``, a name assembled from data);
* a call that unpacks ``*args``/``**kwargs`` — the arity is not in the source;
* an ATTRIBUTE call — ``import local_operator.session_factory as sf`` and then
  ``sf.create_session(...)``, or ``from local_operator import session_factory``
  and then ``session_factory.create_session(...)``. ``_probe_calls`` follows
  only ``ast.Call`` whose ``func`` is an ``ast.Name``, and both of those spell
  the call as an ``ast.Attribute``. Closing it needs a module-IDENTITY check
  first:
  ``from local_operator.agents import AgentRegistry`` binds a CLASS, so
  ``AgentRegistry.anything()`` would look exactly like a module attribute and a
  correct probe would be reported as stale. Recorded, not half-closed — an
  independent scanner covering both attribute shapes found **0** stale calls on
  this tree (review round 1), so the gap is known-empty today.

It also says nothing about what a correctly-bound probe MEASURES —
``tests/unit/scripts/test_bench_*.py`` are where a bench's numbers are pinned.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"

#: How a loaded module's own file is looked up. A parameter (defaulted to the
#: real lookup) rather than a module global, because what :func:`_foreign`
#: refuses is an ENVIRONMENT — another checkout's venv — and a test cannot
#: produce one without lying about this call.
_OriginOf = Callable[[Any], "Path | None"]


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


def _imported(dotted: str) -> Any | None:
    """``importlib.import_module(dotted)``, or ``None`` when it will not import.

    Swallowed here, because the caller renders it: a module that is GONE and a
    module that was answered from somewhere else need different messages, and
    only the first is a defect in the probe.
    """
    try:
        return importlib.import_module(dotted)
    except ImportError:
        return None


def _module_file(module: Any) -> Path | None:
    """The file ``module`` was loaded from, or ``None`` when Python cannot say.

    A seam of its own so :func:`_foreign` is testable without a second venv: the
    arrangement it guards against is an ENVIRONMENT, not an input, so a test can
    only reach it by lying about this one lookup.
    """
    try:
        return Path(inspect.getfile(module)).resolve()
    except (TypeError, OSError):
        return None


def _foreign(dotted: str, origin_of: _OriginOf = _module_file) -> str | None:
    """The refusal line when the interpreter answers ``dotted`` from OUTSIDE the
    tree under scan, else ``None``.

    ``importlib.import_module`` answers from whatever ``sys.meta_path`` reaches
    first, and the editable install a venv carries is a **MAPPING finder**: it
    answers a fully-qualified submodule name and never consults the loaded
    parent's ``__path__``. So every symbol this guard binds is only as good as
    the venv running it — under another checkout's venv it binds THAT tree's
    signature while scanning this one, which is the arrangement AGENTS.md names
    under "a dead venv fails loudly; a wrong one does not".

    Measured, both directions, on this host (a worktree at this PR's own head,
    run through the root checkout's venv, whose editable install maps
    ``local_operator`` at another tree — the root checkout also carried an
    unrelated session's uncommitted ``credentials.py``, which is why the wrong
    tree answered for a module this one does not have):

    * false RED — the tree-wide scan reported a violation against a probe that
      is correct on this tree, printing the other tree's four-positional
      signature;
    * false GREEN — the same arrangement with the trees swapped binds a STALE
      probe against the other tree's signature and reports no violation at all.

    The mirror direction is the one that decides whether this guard is worth
    having, because it reports success while the tree under scan is broken — the
    exact rot the guard exists to catch. Resolving the symbol from this tree's
    own source instead was the other option and cannot work here: importing the
    scanned tree's package still runs ITS ``import local_operator.<x>``
    statements through the same ambient finder, so one process would mix two
    trees and the result could not be trusted either.
    """
    module = _imported(dotted)
    if module is None:
        return None  # gone: `_violation` renders that as a stale probe
    origin = origin_of(module)
    if origin is not None and origin.is_relative_to(REPO):
        return None
    return (
        f"{dotted}: the interpreter running this scan resolves it to "
        f"{origin if origin is not None else 'no readable file'}, not to the tree "
        f"under scan ({REPO}) — so nothing here has been checked. Install a venv "
        'from the tree you are scanning and re-run; AGENTS.md, "Every feature '
        'worktree owns its own venv. Never symlink one."'
    )


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


def _violation(found: _ProbeCall, origin_of: _OriginOf = _module_file) -> str | None:
    """Why ``found`` cannot run, or ``None`` when it binds the live signature.

    A call in a module the interpreter answered from another checkout is not
    reported here: :func:`probe_violations` reports that once per MODULE, since
    it is one fact about the environment with one remedy, and one line per call
    would bury it under a copy of itself for every probe in the tree.
    """
    if _foreign(found.module, origin_of) is not None:
        return None
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


def probe_violations(root: Path = SCRIPTS, *, origin_of: _OriginOf = _module_file) -> list[str]:
    """Every reason ``root``'s probes and the live harness disagree.

    Refusals (:func:`_foreign`) come FIRST and are one line per module, because
    a scan that reports one has checked NOTHING — the calls below it are not
    reported at all when their module could not be resolved to this tree, since
    "stale" would be an accusation against the wrong tree's signature.
    """
    calls = _probe_calls(root)
    checks: dict[str, str | None] = {}
    for module in (call.module for call in calls):
        if module not in checks:
            checks[module] = _foreign(module, origin_of)
    refusals = [line for line in checks.values() if line is not None]
    return refusals + [
        message for call in calls if (message := _violation(call, origin_of)) is not None
    ]


def test_every_probe_call_in_scripts_matches_the_live_harness() -> None:
    """Tree-wide: a red here means the probe string no longer runs as written.

    Fix the PROBE, not the harness signature: the signature is the live contract
    every other caller already follows, and the probe is the stale side.
    """
    violations = probe_violations()
    assert not violations, (
        "a probe string under scripts/ cannot be checked, or calls the harness in "
        "a shape it no longer accepts. pyright cannot see these calls (they are "
        "string literals handed to a child interpreter), so this list is the only "
        "thing that goes red:\n  " + "\n  ".join(violations)
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


def _scan(tmp_path: Path, source: str, origin_of: _OriginOf = _module_file) -> list[str]:
    (tmp_path / "probe_target.py").write_text(source)
    return probe_violations(tmp_path, origin_of=origin_of)


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


def test_a_scan_through_another_checkout_is_refused_not_validated(
    tmp_path: Path,
) -> None:
    """The mirror direction: a scan that cannot reach THIS tree must be RED.

    This is the finding that decides whether the guard is worth having. The
    measured arrangement was a worktree at this PR's own head run through the
    root checkout's venv, whose editable install maps ``local_operator`` at
    another tree: the scan reported a violation against a probe that is correct
    here (false red) and, with the trees swapped, would have bound a stale probe
    against the wrong signature and reported success (false green). Reported
    success over a broken tree is the rot the guard exists to catch.

    Simulated by handing ``_scan`` an origin lookup that answers from elsewhere,
    because the real thing is a venv, not an input — the shipped arrangement is
    exercised end to end in the round-1 remediation evidence (the same command
    through the foreign venv, which now reports this line instead of a pass).
    """
    foreign = tmp_path / "other-checkout" / "local_operator" / "session_factory.py"
    violations = _scan(tmp_path, _CURRENT, origin_of=lambda module: foreign)
    # One line per MODULE the probe imports from `local_operator` — three of them
    # (`agents`, `config`, `session_factory`) — and not one per call, which is
    # what keeps the report readable when the whole tree is scanned.
    assert len(violations) == 3, violations
    assert len(set(violations)) == 3, violations
    for line in violations:
        assert str(foreign) in line
        assert "not to the tree under scan" in line
    assert not any("positional arguments" in line for line in violations), (
        "a refused module must not ALSO be reported as a stale call: the probe in "
        "the scanned tree is correct, the interpreter is not"
    )


def test_a_stale_probe_in_a_foreign_tree_is_not_reported_green(tmp_path: Path) -> None:
    """The false-green half of the same finding, through the same instrument.

    The scanned tree's probe is the exact shape that broke, and the interpreter
    answers the harness from elsewhere — so a guard that trusted the ambient
    import would bind it cleanly and report NO violation. Refusing is what makes
    that impossible, and it costs the honest case nothing: the same stale probe
    is still reported when the module resolves to this tree, one test above.
    """
    foreign = tmp_path / "other-checkout" / "local_operator" / "session_factory.py"
    violations = _scan(tmp_path, _STALE, origin_of=lambda module: foreign)
    assert violations, "a scan that cannot reach this tree must be RED, not a pass"
    assert all("not to the tree under scan" in line for line in violations), violations
    assert not any("positional arguments" in line for line in violations), violations
