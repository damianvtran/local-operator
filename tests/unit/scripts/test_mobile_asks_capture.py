"""The ask-capture rig's own contract: it must run as documented, and its
fixture flag must mean what it says.

Both assertions here exist because round 1 found the rig could not be re-run by
anyone following its own Run line (R2) and that one of its flags selected the
opposite state from the one it named (Q-2) — a capture rig whose evidence cannot
be re-derived is not evidence.

The fixture module is exercised in a SUBPROCESS rather than imported here:
`scripts/probe_isolation` refuses to run once a `local_operator` module has been
imported, and pytest's conftest has already imported plenty — the guard is the
rig's own, and a test is not the place to work around it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(extra)
    return env


def test_the_documented_run_line_reaches_main(monkeypatch):
    """`python scripts/mobile_asks_capture.py` must import, not die on `scripts`.

    R2: the module imported `scripts.mobile_overflow_capture` with no `sys.path`
    bootstrap, so the Run line in its own docstring failed with
    `ModuleNotFoundError: No module named 'scripts'` before `main()` was reached.
    Invoked exactly as documented, from the repository root, with PYTHONPATH
    cleared — the module's own bootstrap is what has to make it work.
    """
    monkeypatch.delenv("PYTHONPATH", raising=False)
    proc = subprocess.run(
        [sys.executable, "scripts/mobile_asks_capture.py"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr


def test_the_fixture_flag_and_the_new_sessions_behave_as_named():
    """Q-2 and design round 1 (D6) in one subprocess.

    `LOP_ASK_FIXTURE_EMPTY_INDEX=0` used to select the EMPTY index — a truthiness
    test on the raw string made the off-spelling mean on, so a round that meant
    to photograph a populated sheet photographed an empty one. The two sessions
    the new frames come from are asserted present for the same reason: evidence a
    later round cannot regenerate is not evidence.
    """
    program = (
        "import json, os;"
        "from scripts.mobile_overflow_fixture import ("
        " _empty_index_requested, _foreign_ask_projection, _asks_stacked_projection);"
        "print(json.dumps({"
        "'flag': _empty_index_requested(),"
        "'foreign': _foreign_ask_projection().session_id,"
        "'stacked': _asks_stacked_projection().session_id}))"
    )
    results = {}
    for value in ["1", "true", "TRUE", " 1 ", "0", "false", "", "no", "off"]:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPO_ROOT,
            env=_clean_env(LOP_ASK_FIXTURE_EMPTY_INDEX=value),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr
        results[value] = __import__("json").loads(proc.stdout.strip().splitlines()[-1])

    for value in ["1", "true", "TRUE", " 1 "]:
        assert results[value]["flag"] is True, value
    for value in ["0", "false", "", "no", "off"]:
        assert results[value]["flag"] is False, value
    assert results["1"]["foreign"] == "asks-foreign"
    assert results["1"]["stacked"] == "asks-stacked"
