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
FORBIDDEN_PREFIXES = (
    "local_operator.model",
    "local_operator.providers",
    "local_operator.config",
    "local_operator.credentials",
    "local_operator.tools",
    "local_operator.tui",
    "local_operator.mobile",
    "local_operator.session",
    "textual",
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


@pytest.mark.parametrize(
    "module",
    [
        "local_operator.evaluation.runner.episode",
        "local_operator.evaluation.runner.guards",
        "local_operator.evaluation.runner.model",
        "local_operator.evaluation.runner.responder",
        "local_operator.evaluation.runner.secrets",
        "local_operator.evaluation.runner.rescue_sweep",
        "local_operator.evaluation.runner.durable_root",
        "local_operator.evaluation.runner.route_ids",
    ],
)
def test_runner_core_does_not_import_the_application(module: str) -> None:
    imported = _fresh_import_modules(module)
    leaked = {
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    }
    assert not leaked, f"{module} leaked application imports: {sorted(leaked)}"


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
    leaked = {
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    }
    assert not leaked, sorted(leaked)


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
    leaked = {
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    }
    assert not leaked, sorted(leaked)


def test_episode_does_not_import_host_secrets() -> None:
    """The runner takes a resolver by injection; it never picks the store itself."""

    imported = _fresh_import_modules("local_operator.evaluation.runner.episode")
    assert "local_operator.evaluation.runner.host_secrets" not in imported
