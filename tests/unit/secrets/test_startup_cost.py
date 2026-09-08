"""``lop secret`` must not put the crypto stack on the CLI's import path.

``tests/unit/test_import_graph.py`` documents why this matters: one
module-level import in something ``cli.py`` touches costs EVERY invocation —
``--version``, shell completion, every scheduler tick — and nothing fails, so
the regression is invisible in review. Argument registration for this
subcommand is stdlib-only and the handlers import the crypto and storage
modules at the point of use; this pins that arrangement.

A fresh subprocess, for the reason that file gives: by the time a test body
runs, pytest has imported half the tree, so an in-process ``sys.modules``
assertion would pass on a real regression.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_PROBE = """
import importlib, json, sys
importlib.import_module("local_operator.cli")
print(json.dumps(sorted(sys.modules)))
"""


def _cli_modules() -> set[str]:
    process = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, cwd=str(REPO), timeout=120
    )
    assert process.returncode == 0, process.stderr[-3000:]
    return set(json.loads(process.stdout.strip().splitlines()[-1]))


def test_cli_import_does_not_load_the_crypto_stack() -> None:
    """``cryptography`` brings a compiled OpenSSL binding with it.

    Nothing about ``lop --version`` needs AES, and the secret handlers import
    it inside the function that uses it.
    """
    modules = _cli_modules()
    offenders = sorted(
        module
        for module in modules
        if module == "cryptography" or module.startswith("cryptography.")
    )
    assert not offenders, f"cryptography is on the CLI startup path; saw {offenders[:5]}"


def test_cli_import_does_not_load_the_store_modules() -> None:
    """Registration is stdlib-only; the store arrives with the first verb."""
    modules = _cli_modules()
    for module in (
        "local_operator.secrets.crypto",
        "local_operator.secrets.store",
        "local_operator.secrets.keys",
        "local_operator.secrets.access",
    ):
        assert module not in modules, f"{module} is on the CLI startup path"


def test_argument_registration_itself_is_stdlib_only() -> None:
    """Importing the secret CLI module must not drag the crypto stack in.

    ``cli.py`` imports this module to register arguments, so an eager
    ``from .crypto import ...`` at its top would silently undo both tests
    above. Checked directly so the failure names the real cause.
    """
    probe = (
        "import importlib, json, sys;"
        " importlib.import_module('local_operator.secrets.cli');"
        " print(json.dumps(sorted(sys.modules)))"
    )
    process = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO), timeout=120
    )
    assert process.returncode == 0, process.stderr[-3000:]
    modules = set(json.loads(process.stdout.strip().splitlines()[-1]))
    offenders = sorted(module for module in modules if module.startswith("cryptography"))
    assert not offenders, f"secrets.cli pulls cryptography at import; saw {offenders[:5]}"
