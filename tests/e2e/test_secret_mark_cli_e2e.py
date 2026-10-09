"""The mark verb as a REAL operator types it: subprocesses, one isolated root.

What this covers that the in-process cells cannot: the whole CLI process
boundary — environment resolution for an isolated root, a stdin-carried secret
value, exit codes, and the receipt's exact stdout — for the S4 surface whose
path never needs a relay (``lop secret set`` → ``lop network init --no-start``
→ ``lop network credential mark``). Both relays in the delivery cells are the
fixtures'; the operator's commands here are real processes.

Every process is run to completion (``subprocess.run``) — nothing is spawned in
the background, so there is nothing to reap.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.harness import NO_NOTIFY_ENV

pytestmark = pytest.mark.e2e

SECRET_NAME = "E2E_MARK_TOKEN"
SECRET_VALUE = "e2e-mark-value"
KEY = f"secret:{SECRET_NAME}"


def _env(config_dir: Path, home: Path) -> dict[str, str]:
    """The scratch environment: CMUX_*/LOP_* stripped, HOME and config redirected.

    The strip matters beyond hygiene: an inherited ``CMUX_*`` variable once let a
    headless run rename the operator's real cmux workspaces, and a scratch HOME
    is what keeps the run off the operator's own store.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    env.update(NO_NOTIFY_ENV)
    env["HOME"] = str(home)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    return env


def _lop(
    config_dir: Path, home: Path, *argv: str, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            sys.executable,
            "-c",
            "import sys; from local_operator.cli import main; sys.exit(main())",
            *argv,
        ],
        env=_env(config_dir, home),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_mark_round_trips_through_a_real_cli(tmp_path: Path) -> None:
    config = tmp_path / "config"
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    config.mkdir(parents=True, exist_ok=True)

    stored = _lop(config, home, "secret", "set", SECRET_NAME, stdin=SECRET_VALUE)
    assert stored.returncode == 0, stored.stderr

    created = _lop(
        config, home, "network", "init", "e2e-mark", "--no-start", "--listen-address", "127.0.0.1"
    )
    assert created.returncode == 0, created.stderr

    marked = _lop(config, home, "network", "credential", "mark", SECRET_NAME, "sync")
    assert marked.returncode == 0, marked.stderr
    assert f"'{SECRET_NAME}' is marked sync" in marked.stdout

    state_files = sorted((config / "network" / "credentials").glob("*/sync.json"))
    assert len(state_files) == 1, state_files
    document = json.loads(state_files[0].read_text(encoding="utf-8"))
    assert document["marks"][KEY]["mark"] == "sync"

    cleared = _lop(config, home, "network", "credential", "mark", SECRET_NAME, "default")
    assert cleared.returncode == 0, cleared.stderr
    document = json.loads(state_files[0].read_text(encoding="utf-8"))
    assert KEY not in document["marks"]

    refused = _lop(config, home, "network", "credential", "mark", "NO_SUCH_SECRET", "sync")
    assert refused.returncode == 1
    assert "holds no secret named 'NO_SUCH_SECRET'" in refused.stdout + refused.stderr
    assert "nothing to mark" in refused.stdout + refused.stderr
