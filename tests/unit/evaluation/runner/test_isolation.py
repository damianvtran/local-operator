"""The runner core must not drag the application into an episode.

An evaluation episode has to be reproducible from its pinned inputs. Importing
providers, config, tools, the TUI, or session code would let a benchmark result
depend on the operator's own live configuration, and the dependency would be
invisible -- nothing in a bundle records which settings file was loaded. This
mirrors the existing startup-isolation assertions in ``test_protocol.py``.
"""

from __future__ import annotations

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

    It takes a live ``CredentialManager`` from its caller (the script that
    also opened the store for the model client), so even the lazy import is
    absent: importing the module pulls in nothing from the application.
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
