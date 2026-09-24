"""The runner core must not drag the application into an episode.

An evaluation episode has to be reproducible from its pinned inputs. Importing
providers, config, tools, the TUI, or session code would let a benchmark result
depend on the operator's own live configuration, and the dependency would be
invisible -- nothing in a bundle records which settings file was loaded. This
mirrors the existing startup-isolation assertions in ``test_protocol.py``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]

# provider_client.py and host_secrets.py are the deliberate exceptions (the
# model client and the secret resolver are the two places a real episode must
# touch the operator's store); both defer that import, so they are absent here
# by construction rather than by luck.
#
# The predicate (:func:`_leaked`) matches a name EXACTLY or at a DOTTED
# boundary, so one entry stands for a package and everything under it, and not
# for the modules the application spells as that name's siblings: a bare
# ``local_operator.session`` never matched ``local_operator.session_factory``,
# which is the module that reads the operator's own configuration — the exact
# thing this rule exists to keep out of an episode. So "the runner may not
# import session code" was enforced by intent, not by the test: importing
# ``session_factory`` yielded an empty leak set while pulling the whole session
# package in behind it. The list below therefore spells out both: the
# application modules the runner must stay away from, and the ``_``-suffixed
# siblings that share their names, because naming the module alone leaves the
# same hole open one name over.
FORBIDDEN_PREFIXES = (
    # Packages: the entry covers every submodule underneath it.
    "local_operator.model",
    "local_operator.providers",
    "local_operator.tools",
    "local_operator.tui",
    "local_operator.mobile",
    "local_operator.session",
    "local_operator.analytics",
    # Single modules: one file each, so no other entry can stand in for them.
    "local_operator.config",
    "local_operator.config_migrations",
    "local_operator.config_watch",
    "local_operator.credentials",
    "local_operator.session_factory",
    "local_operator.session_lease",
    "local_operator.context_files",
    "local_operator.resume",
    "local_operator.paths",
    "local_operator.cli",
    "local_operator.cli_style",
    "local_operator.exec_mode",
    "local_operator.exec_session",
    "local_operator.exec_startup",
    "local_operator.exec_worker",
    "local_operator.imaging",
    "local_operator.ansi",
    "local_operator.incidents",
    # Third-party: the TUI toolkit, which no headless episode may need.
    "textual",
)


def _leaked(imported: set[str]) -> list[str]:
    """The application modules in ``imported`` an episode must not pull in.

    ONE definition of the denylist's matching rule, because a second copy of it
    is precisely how the two would come to disagree — and the rule's own
    boundary is load-bearing (see the comment on the list above).
    """
    return sorted(
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    )


def _fresh_import_modules(module: str) -> set[str]:
    """The modules a fresh interpreter holds after importing ``module``.

    A snapshot taken AFTER the import returns, which is the shape's own limit: a
    module that pops what it imported (``import local_operator.config;
    sys.modules.pop("local_operator.config")``) did execute the denied import and
    still reads clean here. That is deliberate evasion rather than the accident
    this guard is for, so it is recorded and not chased -- the deferred half
    below closes the statement form of it for free, and an import hook or
    ``sys.addaudithook`` is what would close the rest if it ever mattered.
    """
    probe = (
        "import importlib,json,sys;"
        "importlib.import_module(sys.argv[1]);"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, module],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))


# provider_client and public_reply are probed here too. Leaving them out is
# what made the widened list vacuous: no other assertion exercises its
# ``local_operator.paths`` entry, and provider_client — the runner's model
# client, whose own test only pins that it DEFERS the one configure import —
# is exactly where that leaked in, through ``local_operator.logger``.
@pytest.mark.parametrize(
    "module",
    [
        "local_operator.evaluation.runner.action_tool",
        "local_operator.evaluation.runner.episode",
        "local_operator.evaluation.runner.guards",
        "local_operator.evaluation.runner.model",
        "local_operator.evaluation.runner.provider_client",
        "local_operator.evaluation.runner.public_reply",
        "local_operator.evaluation.runner.responder",
        "local_operator.evaluation.runner.secrets",
        "local_operator.evaluation.runner.rescue_sweep",
        "local_operator.evaluation.runner.durable_root",
        "local_operator.evaluation.runner.route_ids",
    ],
)
def test_runner_core_does_not_import_the_application(module: str) -> None:
    imported = _fresh_import_modules(module)
    leaked = _leaked(imported)
    assert not leaked, f"{module} leaked application imports: {leaked}"


def _runner_core_files() -> list[Path]:
    """Every source file of the runner package, read from the tree.

    ONE traversal for both halves of the rule -- the eager half needs the module
    names a fresh interpreter can import, the deferred half needs the files to
    parse -- because two derivations of "what the runner core is" are free to
    disagree about a subpackage. Measured at cd91d7dc1, the flat ``glob("*.py")``
    these names used to come from left a new ``runner/subpkg/__init__.py`` holding
    one ``import local_operator.credentials`` at 20 passed: the same hole this
    file closes, one directory deeper. ``rglob`` is the fix, and the deferred
    half reads this same file list rather than deriving a second one.
    """
    package = REPO / "local_operator" / "evaluation" / "runner"
    return sorted(package.rglob("*.py"))


def _runner_core_module_name(path: Path) -> str:
    """``path``'s dotted name, an ``__init__`` mapped to the package it opens.

    Read from disk rather than by importing: the candidate set has to cover a
    module no test has imported yet, which is the only way the rule outlives the
    module somebody adds next. Private modules are kept -- the package
    directory, not a leading underscore, is what bounds the runner core
    (``from . import _probe`` is a legal import inside it), and a probe that
    trusts a naming convention is one rename away from silence.

    A subpackage's ``__init__`` is judged AS the subpackage rather than skipped:
    every module underneath it executes that file first, but a subpackage nothing
    imports yet has only that file to judge, which is the case a flat walk
    misses. The runner's own ``__init__`` falls out of the same rule and reads as
    the package itself, so it is no longer an exclusion to justify -- and
    ``test_runner_package_import_is_inert`` still covers it directly.
    """
    parts = list(
        path.relative_to(REPO / "local_operator" / "evaluation" / "runner").with_suffix("").parts
    )
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(["local_operator", "evaluation", "runner", *parts])


def _runner_core_modules() -> list[str]:
    """The runner core, as the names a fresh interpreter can import."""
    return [_runner_core_module_name(path) for path in _runner_core_files()]


def test_the_runner_core_reaches_no_application_machinery() -> None:
    """The eager half of the rule, over the whole runner core rather than a list.

    The verdict here is the closure each candidate EXECUTES when it is imported:
    a denied import that nothing runs at import time -- one inside a function, a
    PEP 562 ``__getattr__``, a guard that is false on this host -- is absent from
    that closure and is not judged by this assertion. That limit is stated rather
    than implied, because a claim wider than its measurement is the exact failure
    this guard exists to prevent; the deferred half is
    ``test_the_runner_core_names_no_denied_import_outside_the_pinned_deferred_seams``
    below, which reads the source instead of the closure and pins the sites that
    may defer.

    Every candidate ``test_runner_core_does_not_import_the_application`` probes
    is a hand-written entry, and the rule it enforces is about the runner core,
    which is a DIRECTORY: the pin cannot see a file. Measured at 75aa8bc45,
    adding ``local_operator/evaluation/runner/budget_window.py`` whose module
    level was one ``import local_operator.config`` left all 19 tests in this
    file green -- the #1145 shape, judged by nobody. This is the other half of
    that list: the candidates come from the package directory, so a module
    nobody remembered to name is judged anyway, as soon as anything imports it,
    and the probe itself is already transitive when it runs. (The transitivity
    is worth stating because it is what an allowed name is dangerous FOR:
    ``harness.comms``, ``harness.loop`` and ``harness.subagent`` are not
    themselves denied, and a module ON the list above that imports one of them
    fails that assertion today -- through the ``local_operator.session``,
    ``ansi``/``incidents`` and ``paths``/``resume`` those closures drag in. So a
    leaky ALLOWED module is not what this adds: what no assertion covered was a
    candidate nobody listed, which is what the derivation below is for, and the
    transitivity is stated so that no reader takes this test for the fix to a
    hole that was already closed.)

    One stricter form was considered and is not taken, for a measured reason: a
    default-deny allowlist over the closure -- the plainest reading of "an
    episode may reach the evaluation stack and nothing else" -- fails on today's
    legitimate tree, where ``action_tool`` alone reaches ``harness.types``,
    ``harness.approval``, ``harness.reply_channel`` and ``harness.wake``. The
    runner sharing the harness's vocabulary is what the hoists in #1145 and
    #1150 were FOR, so that rule would have to carry an allowlist of every
    module the two halves share -- and a second, larger list that every new
    evaluation module then has to be added to, which is the list this file
    already keeps, read from the other end.

    A static walk over the runner's import statements is NOT rejected here; it
    is taken one test below. What an UNPINNED walk cannot do is tell the
    deliberate deferral from a new one -- it fails on ``provider_client``'s
    nested ``model.configure`` and ``analytics`` imports -- so the deferred half
    carries a hand-pinned set of the sites that may defer rather than a
    relaxation of the rule. Its module-body half needs no loosening at all: the
    runner names no denied module in any module body today, which is why that
    assertion carries no pin, and it is where the environment-dependent cases
    (a ``sys.platform`` or environment guard, an ``if TYPE_CHECKING`` block)
    are judged the same way on every host.
    """
    modules = _runner_core_modules()
    # A candidate set that has collapsed is a check that cannot fail, which is
    # how the widening this file replaced went unnoticed.
    assert modules, "the runner package has no module to probe"
    offenders: dict[str, list[str]] = {}
    for module in modules:
        leaked = _leaked(_fresh_import_modules(module))
        if leaked:
            offenders[module] = leaked
    assert not offenders, f"runner-core modules reach application machinery: {offenders}"


def _denied_named(node: ast.Import | ast.ImportFrom, package: list[str]) -> list[str]:
    """The denied modules ``node`` names, or an empty list if it names none.

    ``package`` is the importing file's own package, which is what makes a
    RELATIVE import resolvable: ``from ... import config`` inside the runner
    reaches the denied ``local_operator.config``, and a walk that read only
    absolute names would call that statement clean. A ``from X import a, b``
    statement is reported as ``X`` when ``X`` itself is denied, because the
    module it names is the thing at fault -- not each alias, and not whichever
    alias happened to sort first.
    """
    if isinstance(node, ast.Import):
        named = [alias.name for alias in node.names]
    else:
        base = package[: len(package) - (node.level - 1)] if node.level else []
        root = ".".join([*base, *([node.module] if node.module else [])])
        named = (
            [root]
            if _leaked({root})
            else [f"{root}.{alias.name}" if root else alias.name for alias in node.names]
        )
    return _leaked(set(named))


def _denied_imports(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """The import statements in ``path`` naming a denied module, split by scope.

    The first half holds the statements directly in the module body, where the
    import EXECUTES on import; the second holds every other statement as
    ``(enclosing scope, denied module)``, the scope being the dotted name of the
    innermost function or class holding it -- which is what makes two deferred
    imports the same site or different ones.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = _runner_core_module_name(path).split(".")
    if path.stem != "__init__":
        package = package[:-1]
    body_ids = {id(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))}
    module_body: list[str] = []
    deferred: list[tuple[str, str]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            child_scope = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                child_scope = f"{scope}.{child.name}" if scope else child.name
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for named in _denied_named(child, package):
                    if id(child) in body_ids:
                        module_body.append(named)
                    else:
                        deferred.append((child_scope, named))
            visit(child, child_scope)

    visit(tree, "")
    return module_body, deferred


#: The deferred denied imports the runner carries ON PURPOSE, pinned by hand.
#:
#: The eager half above cannot judge these: nothing executes them at import
#: time, so they are absent from every closure it reads -- and a deferred import
#: is the shape that stays unjudged by a snapshot, which is why the runtime probe
#: cannot be widened into this half. Each entry is (module, enclosing scope,
#: denied module), so a NEW site fails even when it names a module already
#: pinned, and a rename re-declares itself here instead of sliding past. Pinning
#: sites and not just names is the point: before this set existed, only the
#: ``model.configure`` calls were named (by
#: ``test_provider_client_defers_its_configure_import``), and a fifth deferred
#: denied import passed every assertion in the file.
#:
#: Each entry is a decided exception with its own reason rather than a formality,
#: and the enumeration IS the count: a reader never has to trust a number in
#: prose, which is what drifted in this file before.
#:
#: * ``local_operator.model.configure`` is the model client's seam: an episode
#:   must reach the model, and it must do so without importing it at module
#:   level. The target is not innocent -- ``model.configure`` itself names the
#:   model package's own pieces and ``local_operator.paths`` in its module body,
#:   so calling into it puts the operator's config directory on the episode's
#:   path -- which is why the DEFERRAL is the contract and not a detail.
#: * ``local_operator.analytics`` takes a diagnostic note, inside a ``try`` that
#:   treats the ledger as never a dependency: an episode must not fail because a
#:   diagnostic could not be written.
#:
#: Adding an entry is a decision, not a formality -- the alternative is not to
#: import it -- and removing one is the Stage-5 cleanup work this guard exists to
#: keep honest (`harness.comms`, `loop` and `subagent` still leak).
PINNED_DEFERRED_DENIED_IMPORTS: tuple[tuple[str, str, str], ...] = (
    (
        "local_operator.evaluation.runner.provider_client",
        "_note_eval_session_name",
        "local_operator.analytics",
    ),
    (
        "local_operator.evaluation.runner.provider_client",
        "_table_cost_micros",
        "local_operator.model.configure",
    ),
    (
        "local_operator.evaluation.runner.provider_client",
        "_usage_from",
        "local_operator.model.configure",
    ),
    (
        "local_operator.evaluation.runner.provider_client",
        "create_provider_model_client",
        "local_operator.model.configure",
    ),
    # `_one_rung_lower` is the eval-side spelling of `harness.loop._lower_effort`
    # and MUST reach `model.effort.real_rungs` for the same reason the four above
    # reach their modules: the rule it encodes (which ladder members are ranks
    # rather than sentinels) has one owner, and a local copy would drift. The
    # import is deferred (inside the function) because `provider_client` may not
    # pull `model.effort` into its import-time closure -- `effort.py` is pure and
    # cheap, but the seam is the module boundary, not the cost. Pinned here so an
    # added deferred denied import still fails (round-2 review R2-1).
    (
        "local_operator.evaluation.runner.provider_client",
        "_one_rung_lower",
        "local_operator.model.effort",
    ),
)


def test_the_runner_core_names_no_denied_import_outside_the_pinned_deferred_seams() -> None:
    """The deferred half of the rule, read from the source instead of run.

    A runner module may reach a denied module only where this file says so. The
    eager half judges the closure an import EXECUTES, so a denied import that
    nothing executes at import time is absent from it: measured at cd91d7dc1, a
    ``runner/budget_window.py`` whose whole body was ``def f(): import
    local_operator.config`` left the file at 20 passed, and the same carrier
    behind a PEP 562 ``__getattr__`` or an ``if TYPE_CHECKING:`` block is green
    there too. The tree already carries such sites -- all in ``provider_client``,
    pinned below -- and nothing pinned the set, so a fifth one passed every
    assertion in the file. That is the gap this closes, and the docstring above
    no longer claims more than its assertion delivers.

    So every import statement of every runner module is read, at ANY scope, and
    split by where it sits:

    * in the module body, where the statement executes when the module is
      imported. Nothing is pinned there: the runner names no denied module in
      any module body today, so the assertion needs no relaxation at all -- and
      that is what makes it the environment-independent half, since it judges a
      ``sys.platform`` or environment guard, or a ``TYPE_CHECKING`` block, the
      same way on every host rather than only where the guard happens to run.
    * nested, which is deferred by construction, judged against
      ``PINNED_DEFERRED_DENIED_IMPORTS``. An added deferred denied import fails
      whether it lands in a module nobody listed or at a scope nobody pinned,
      so the site has to be justified here or not written.

    The two halves are deliberately complementary rather than either complete:
    this one is textual, so it cannot see an import built at run time
    (``importlib.import_module``), which the runtime probe does catch; the probe
    is a post-import snapshot, so it cannot see a deferred statement, which this
    one reads. Neither is allowed to describe itself as covering the other.
    """
    module_body: dict[str, list[str]] = {}
    deferred: set[tuple[str, str, str]] = set()
    for path in _runner_core_files():
        module = _runner_core_module_name(path)
        in_body, nested = _denied_imports(path)
        if in_body:
            module_body[module] = in_body
        deferred |= {(module, scope, name) for scope, name in nested}
    assert not module_body, (
        "these runner modules name a denied module in their own body, where the "
        f"import runs when the module is imported: {module_body}"
    )
    pinned = set(PINNED_DEFERRED_DENIED_IMPORTS)
    assert deferred == pinned, (
        "the runner's deferred denied imports are not the pinned set: "
        f"new={sorted(deferred - pinned)} gone={sorted(pinned - deferred)}"
    )


def test_shared_renderer_is_importable_from_an_episode() -> None:
    """The one renderer an episode calls must not drag the application in.

    The benchmark renders its transcript through the same function the TUI
    renders through (``harness/render.py``, hoisted out of
    ``session/session.py``), which puts a HARNESS module on an episode's import
    path -- nothing above covers one, and the denylist cannot notice on its own:
    this module used to reach its vocabulary through ``incidents``,
    ``session.peer``, ``tools.builtin`` and ``harness.comms`` and leaked 17
    denied modules, so importing the renderer was as barred as importing the
    session it was hoisted from. The seven markers moving into
    ``harness.message_types`` is worth exactly as much as this assertion holding:
    a single heavy import added to that module, or to the renderer, puts an
    episode back on the operator's own configuration with nothing else to show
    it. Probed here rather than in ``tests/unit/harness`` because the rule being
    enforced is this file's, and its ``FORBIDDEN_PREFIXES`` is the list it is
    enforced against.
    """
    imported = _fresh_import_modules("local_operator.harness.render")
    leaked = _leaked(imported)
    assert not leaked, f"the shared renderer leaked application imports: {leaked}"


def _public_module_names(package: Path) -> set[str]:
    """Module and package names in ``package``, read from the tree.

    Read from disk rather than by importing: this file's whole subject is that
    an episode must not import the application, and the candidate set has to
    cover a module no test has imported yet. Private modules are skipped — the
    denylist bars public entry points, and ``__init__`` is not one.
    """
    return {
        child.stem if child.suffix == ".py" else child.name
        for child in package.iterdir()
        if not child.name.startswith("_")
        and (child.suffix == ".py" or (child / "__init__.py").is_file())
    }


def test_denylist_bars_the_siblings_of_every_name_it_bars() -> None:
    """What shares a barred name's name is barred too, and the tree finds them.

    The predicate matches at a dotted boundary, so ``local_operator.session``
    does NOT cover ``local_operator.session_factory`` — the module that reads
    the operator's own configuration, and the one this widening was aimed at.
    The siblings are named in the list as entries of their own, and this test is
    what keeps that true: the candidate set comes from the package directory, so
    an eighth sibling added later fails here without anyone remembering to widen
    the list.

    Five assertions run today — ``session_factory``, ``session_lease``,
    ``config_migrations``, ``config_watch``, ``cli_style`` — and each can fail,
    which is the whole of what this derivation enforces. It is defined relative
    to the denylist, so it is blind to a *head's* removal: dropping a head
    deletes the candidate set that would have judged it. ``PINNED_ENTRIES`` is
    what catches that. The submodule half that used to sit beside this one was
    dropped rather than counted, because it could not fail: a submodule of a
    barred package is barred by the predicate itself, which made all 127 of
    those assertions true by construction.
    """
    names = _public_module_names(REPO / "local_operator")
    siblings: dict[str, list[str]] = {}
    for name in sorted(names):
        shares_name = [
            f"local_operator.{other}" for other in sorted(names) if other.startswith(f"{name}_")
        ]
        if shares_name and _leaked({f"local_operator.{name}"}):
            siblings[name] = shares_name
    # A check that has nothing to check is how the widening this replaced went
    # unnoticed: it has to be able to fail.
    assert siblings, "no barred name has a sibling to check"
    for name, shares_name in sorted(siblings.items()):
        for qualified in shares_name:
            assert _leaked({qualified}) == [qualified], (
                f"{qualified} shares a name with the barred local_operator.{name} "
                "but is not barred itself"
            )


#: The denylist as reviewed, pinned by hand because nothing else can pin it: a
#: deleted entry leaves no trace for the tree to judge, and the derivation above
#: is defined relative to the list, so it goes quiet at exactly the moment the
#: entry it would have judged disappears. Most of these names are policy rather
#: than a measured leak — nothing on disk records which modules an episode may
#: reach, and an entry that is no neighbour of another entry has no second
#: assertion standing behind it — so a removal here has to fail a test that names
#: it. The derivation this replaced made exactly that trade: it gained
#: ``session_lease`` and ``config_watch`` as derived entries but lost the four
#: entry-level pins the spot-check it replaced carried (``local_operator.session``,
#: ``local_operator.tools``, ``local_operator.exec_worker``, ``textual``), and
#: deleting any of those four then passed every test in this file.
PINNED_ENTRIES = (
    "local_operator.model",
    "local_operator.providers",
    "local_operator.tools",
    "local_operator.tui",
    "local_operator.mobile",
    "local_operator.session",
    "local_operator.analytics",
    "local_operator.config",
    "local_operator.config_migrations",
    "local_operator.config_watch",
    "local_operator.credentials",
    "local_operator.session_factory",
    "local_operator.session_lease",
    "local_operator.context_files",
    "local_operator.resume",
    "local_operator.paths",
    "local_operator.cli",
    "local_operator.cli_style",
    "local_operator.exec_mode",
    "local_operator.exec_session",
    "local_operator.exec_startup",
    "local_operator.exec_worker",
    "local_operator.imaging",
    "local_operator.ansi",
    "local_operator.incidents",
    "textual",
)


def test_denylist_still_bars_every_entry_it_was_reviewed_with() -> None:
    """No entry may leave the list in silence, whatever the tree can re-derive.

    This is a deliberate second copy of the list, and the copy is the point: the
    live list cannot test itself, and every entry in it is a decision someone
    made about what an episode may reach. Two of them show why a pin is the only
    mechanism that works here. ``local_operator.paths`` resolves the operator's
    config directory, and the runner's model client reached it through
    ``local_operator.logger`` until this PR; with that fix in place, every other
    assertion in this file still passes with the entry removed, so the entry the
    fix exists to satisfy could go unremarked. ``local_operator.session`` is the
    module the rule is named after, and its removal takes the whole derived half
    above with it — the candidates, and the assertions that would have failed for
    them.

    Adding an entry needs no edit here (the check is a subset one, so the list
    may grow); a removal, and with it the justification somebody wrote down,
    fails and names the entry.
    """
    removed = [name for name in PINNED_ENTRIES if name not in FORBIDDEN_PREFIXES]
    assert not removed, f"the denylist no longer bars {removed}"


def test_runner_package_import_is_inert() -> None:
    imported = _fresh_import_modules("local_operator.evaluation.runner")
    assert not {name for name in imported if name.startswith("local_operator.evaluation.runner.")}


def test_provider_client_defers_its_configure_import() -> None:
    """The one module allowed to reach the app must still not do it eagerly."""

    imported = _fresh_import_modules("local_operator.evaluation.runner.provider_client")
    assert "local_operator.model.configure" not in imported


def test_host_secrets_is_the_only_other_store_seam_and_takes_it_by_injection() -> None:
    """``host_secrets`` may serve the credential store but never imports it.

    It takes the config ROOT from its caller (the script that also opened the
    store for the model client), so even the lazy import is absent: importing
    the module pulls in nothing from the application.
    """

    imported = _fresh_import_modules("local_operator.evaluation.runner.host_secrets")
    leaked = _leaked(imported)
    assert not leaked, leaked


def test_run_episode_script_imports_no_session_or_tui() -> None:
    """The operator script is not a session surface; importing it stays inert.

    It reaches the store and the model configuration lazily, inside ``run``,
    so importing the module (what ``--help`` and the tests do) must pull in
    nothing from the application.
    """

    probe = (
        "import importlib.util,json,sys;"
        "spec=importlib.util.spec_from_file_location('run_episode', sys.argv[1]);"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(REPO / "scripts" / "run_episode.py")],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    imported = set(json.loads(completed.stdout.strip().splitlines()[-1]))
    leaked = _leaked(imported)
    assert not leaked, leaked


def test_episode_does_not_import_host_secrets() -> None:
    """The runner takes a resolver by injection; it never picks the store itself."""

    imported = _fresh_import_modules("local_operator.evaluation.runner.episode")
    assert "local_operator.evaluation.runner.host_secrets" not in imported
