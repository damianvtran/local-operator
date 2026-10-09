"""Exec and the SDK cannot originate Aida: pinned by AST and by a real run.

THE PROPERTY (design 2026-10-09 §0d/§3): the automation shapes — ``lop exec``
(supervisor, agent-runtime-svc, CI), the SDK, and the session factory every
host funnels through — must never be able to CREATE her session, arm a cadence
or install a wake supervisor. That is structural, not a switch somebody may
flip: none of these modules imports ``local_operator.aida`` at all, so there is
no code path from them to her engine to begin with. A later import added in
passing would not fail anything today — which is exactly why this pin exists.

Mutating any import of hers into ``exec_mode``/``exec_worker``/``sdk``/
``session_factory`` turns the AST cell red; making an exec run create her (an
``ensure_session`` call reached from the exec path) turns the behavioural cell
red.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

#: The modules the automation paths go through. ``cli.py`` is deliberately NOT
#: here: it carries the one ``lop aida note`` import, which is a user's own
#: explicit command, not a boot that can create her.
_MODULES = (
    "local_operator/exec_mode.py",
    "local_operator/exec_worker.py",
    "local_operator/sdk.py",
    "local_operator/session_factory.py",
)


def _aida_imports_in(path: Path) -> list[str]:
    """Every import of hers the AST of ``path`` holds, spelled as the source does.

    Catches the four spellings an author might reach for — ``import
    local_operator.aida``, ``from local_operator.aida import …``, ``from
    local_operator import aida``, and a relative ``from .aida import …`` — at
    ANY nesting depth, because a function-local import is exactly how the other
    engine reads in this codebase are spelled.
    """
    found: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = {alias.name for alias in node.names}
            if module == "local_operator.aida" or module.startswith("local_operator.aida."):
                found.append(f"from {module} import … (line {node.lineno})")
            elif module == "local_operator" and "aida" in names:
                found.append(f"from local_operator import aida (line {node.lineno})")
            elif module in ("", None) and "aida" in names:
                found.append(f"from . import aida (line {node.lineno})")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "local_operator.aida" or alias.name.startswith(
                    "local_operator.aida."
                ):
                    found.append(f"import {alias.name} (line {node.lineno})")
    return found


def test_the_automation_modules_never_import_her_engine() -> None:
    offenders: dict[str, list[str]] = {}
    for relative in _MODULES:
        path = REPO / relative
        assert path.is_file(), f"the pin names a module that moved: {relative}"
        found = _aida_imports_in(path)
        if found:
            offenders[relative] = found
    assert not offenders, (
        "these modules gained an import of local_operator.aida, so an "
        "automation run could now create her session, arm a cadence or install "
        "a wake supervisor — the exact zero-footprint property R17 pins:\n"
        + json.dumps(offenders, indent=2)
    )


@pytest.mark.slow
def test_a_real_exec_run_leaves_no_aida_footprint(tmp_path: Path) -> None:
    """The behavioural half: a REAL ``lop exec`` under an isolated HOME.

    The mock hosting keeps it provider-free (the wire is irrelevant to this
    claim); what it proves is that a full exec lifecycle — config read, session
    build, turn, persist, dispose — creates nothing of hers. Mutation: any
    ``ensure_session`` reached from the exec path leaves an ``aida/`` directory
    and this goes red.
    """
    home = tmp_path / "home"
    root = home / ".local-operator"
    root.mkdir(parents=True)
    (root / "config.yml").write_text("values:\n  hosting: test\n  model_name: mock\n")
    env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(root),
        "PATH": os.environ.get("PATH", ""),
        "TERM": "dumb",
        # The suite's own kill switch, inherited by children: this run must
        # never be in a position to post anything even if it wanted to.
        "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1",
    }
    # The environment is a WHITELIST, so every ambient CMUX_*/LOP_* variable is
    # already gone: a forked run that inherits a real CMUX_WORKSPACE_ID renames
    # the operator's live cmux workspaces.
    proc = subprocess.run(
        [sys.executable, "-m", "local_operator.cli", "exec", "say hi", "--json"],
        cwd=str(home),
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert proc.returncode == 0, f"exec failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    assert (root / "sessions").is_dir(), "the exec run did not actually run a session"
    assert not (
        root / "aida"
    ).exists(), "an exec run created her engine's footprint — automation must stay free"
