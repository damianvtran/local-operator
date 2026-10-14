"""Catch a worktree running tests with another checkout's editable interpreter.

Pytest's rootdir makes the parent import the tested checkout, but a real runtime
child imports the tree pinned in its interpreter's site-packages. On 2026-10-14,
this split produced 43 child failures in a 46-failure local runtime run: child
stderr named a different checkout and could not import ``stall_watchdog``.
Checking the interpreter's import target before collection names the cause once,
rather than letting unrelated child tests each report the symptom.

The probe uses ``-P`` like product children, so cwd cannot make a foreign venv
look correct. It resolves the package without importing application code. This
checks identity, not dependency completeness or native-runtime behaviour.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

_PROBE = (
    "import importlib.util as u; s = u.find_spec('local_operator'); "
    "print(s.origin if s and s.origin else '')"
)


def mismatch(
    rootpath: Path,
    *,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Refuse a split tree, or a probe that cannot establish child-source identity."""
    wanted = (rootpath / "local_operator").resolve()
    if not (wanted / "__init__.py").is_file():
        # Synthetic pytest roots used by harness self-tests contain no product;
        # requiring a matching install there would reject the test harness itself.
        return None
    interpreter = python or sys.executable
    try:
        done = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [interpreter, "-P", "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
            env=dict(os.environ if env is None else env),
        )
        origin = done.stdout.strip()
        seen = Path(origin).resolve().parent if done.returncode == 0 and origin else None
    except (OSError, subprocess.SubprocessError):
        seen = None
    if seen == wanted:
        return None
    return (
        "Test interpreter/source mismatch: children must import the checkout under test.\n"
        f"  expected package: {wanted}\n"
        f"  interpreter:      {interpreter}\n"
        f"  child package:    {seen if seen is not None else '(probe could not resolve it)'}\n"
        "A foreign editable venv makes pytest test one tree while real children execute another.\n"
        "Use this worktree's own venv; install it from this checkout (AGENTS.md):\n"
        "  uv venv --python 3.12 .venv\n"
        "  uv pip install -e '.[all,dev]' --python .venv/bin/python"
    )
