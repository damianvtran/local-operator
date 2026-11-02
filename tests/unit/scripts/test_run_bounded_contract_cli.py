"""Real CLI/receipt boundary; every command is short-lived and owns no descendants."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator import bounded_contract, memory_guard

RUNNER = Path(__file__).resolve().parents[3] / "scripts" / "run_bounded.py"


def invoke(tmp_path, args, *, inherited=None):
    # The zero-grant cases are isolated test policy, never an availability
    # fallback. The positive-grant cells explicitly restore inherited admission.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CMUX_", "LOCAL_OPERATOR_", "LOP_"))
    }
    env.update(
        HOME=str(tmp_path),
        LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / "config"),
        LOCAL_OPERATOR_SCRATCHPAD=str(tmp_path),
        NODE_OPTIONS="--trace-warnings",
    )
    if inherited is not None:
        env[memory_guard.RESOLVED_MEMORY_MB_ENV] = str(inherited)
    done = subprocess.run(
        [sys.executable, str(RUNNER), *args],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    records = list((tmp_path / "run-bounded-runs").glob("*.completion.json"))
    assert len(records) == 1, done.stderr
    return done, json.loads(records[0].read_text())


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="Linux has a strict backend")
def test_real_default_refuses_before_command_side_effect(tmp_path):
    marker = tmp_path / "must-not-exist"
    done, row = invoke(
        tmp_path,
        ["--", sys.executable, "-c", f"open({str(marker)!r}, 'w').write('spawned')"],
    )
    assert done.returncode == 125, done.stderr
    assert not marker.exists()
    assert row["requested_contract"] == "cleanup-required"
    assert row["termination_cause"] == "unsupported-backend"
    assert row["command_exit_code"] is None
    assert row["cleanup_scope"] == "owned-process-group"
    assert row["cleanup_status"] == row["descendant_cleanup"] == "unknown"
    assert not list((tmp_path / "run-bounded-runs").glob("*.spawn.json"))


@pytest.mark.skipif(
    not bounded_contract.DirectChild.available(),
    reason=bounded_contract.DirectChild.unavailable_reason() or "Boundary available",
)
@pytest.mark.parametrize("status", [0, 7, -signal.SIGTERM])
def test_real_command_status_preserves_child_outcome_and_environment(tmp_path, status):
    source = (
        "import json,os,signal,sys; "
        f"print(json.dumps([os.environ[{memory_guard.RESOLVED_MEMORY_MB_ENV!r}], "
        "os.environ['NODE_OPTIONS'], os.environ['LOCAL_OPERATOR_NO_NOTIFICATIONS']]),flush=True); "
        + (f"os.kill(os.getpid(), {-status})" if status < 0 else f"sys.exit({status})")
    )
    done, row = invoke(
        tmp_path,
        ["--contract=command-status", "--memory-mb=0", "--", sys.executable, "-c", source],
    )
    expected = 128 - status if status < 0 else status
    assert done.returncode == expected, done.stderr
    assert json.loads(done.stdout) == ["0", "--trace-warnings --max-old-space-size=6144", "1"]
    assert row["command_exit_code"] == row["wrapper_exit_code"] == expected
    assert row["termination_cause"] == ("child-signal" if status < 0 else "child-exit")
    assert row["signal_scope"] == "direct-child"
    assert row["cleanup_scope"] == "owned-process-group"
    assert row["cleanup_status"] == row["descendant_cleanup"] == "unknown"


@pytest.mark.parametrize("memory_args", [[], ["--memory-mb=0"], ["--memory-mb=129"]])
def test_real_opt_in_cannot_bypass_positive_inherited_grant(tmp_path, memory_args):
    marker = tmp_path / "must-not-exist"
    done, row = invoke(
        tmp_path,
        [
            "--contract=command-status",
            *memory_args,
            "--",
            sys.executable,
            "-c",
            f"open({str(marker)!r}, 'w').write('spawned')",
        ],
        inherited=128,
    )
    assert done.returncode == 125, done.stderr
    assert not marker.exists()
    assert row["termination_cause"] == (
        "memory-policy-refused" if memory_args else "unsupported-policy"
    )
    assert row["command_exit_code"] is None
    assert row["cleanup_status"] == "unknown"


@pytest.mark.skipif(hasattr(os, "waitid"), reason="This Python exposes os.waitid")
def test_real_missing_waitid_is_named_in_refusal_receipt(tmp_path):
    done, row = invoke(
        tmp_path,
        [
            "--contract=command-status",
            "--memory-mb=0",
            "--",
            sys.executable,
            "-c",
            "print('must-not-run')",
        ],
    )
    assert done.returncode == 125
    assert done.stdout == ""
    assert row["termination_cause"] == "unsupported-boundary"
    assert "missing os.waitid" in row["cleanup_reason"]
    assert row["command_exit_code"] is None
    assert row["cleanup_status"] == "unknown"
