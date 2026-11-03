"""Real CLI/receipt boundary; every command is short-lived and owns no descendants."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
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
    not bounded_contract.ReservedLeader.available(),
    reason=bounded_contract.ReservedLeader.unavailable_reason() or "Boundary available",
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
    assert row["signal_scope"] == "reserved-leader-group"
    assert row["cleanup_scope"] == "owned-process-group"
    assert row["cleanup_status"] == row["descendant_cleanup"] == "unknown"
    # 3.12 on macOS has no os.waitid: it must RUN on the sysctl observer, not skip.
    assert row["terminal_observer"] == bounded_contract.ReservedLeader.select_observer()
    assert "memory guard disabled (explicit)" in done.stderr


def _alive(token):
    found = subprocess.run(["pgrep", "-f", token], capture_output=True, text=True, check=False)
    return [int(pid) for pid in found.stdout.split()]


def _reap_token(token):
    # Exact-pid cleanup of anything a failing cell leaves behind.
    for pid in _alive(token):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(
    not bounded_contract.ReservedLeader.available(),
    reason=bounded_contract.ReservedLeader.unavailable_reason() or "Boundary available",
)
def test_real_command_status_enforces_tightened_inherited_grant_on_grandchild(tmp_path):
    token = f"run-bounded-alloc-{uuid.uuid4().hex}"
    grandchild = "b = bytearray(400 * 1024 * 1024); import time; time.sleep(300)"
    leader = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}, {token!r}]).wait()"
    )
    try:
        done, row = invoke(
            tmp_path,
            [
                "--contract=command-status",
                "--memory-mb=64",
                "--timeout=60",
                "--",
                sys.executable,
                "-c",
                leader,
            ],
            inherited=128,
        )
        assert done.returncode == 137, done.stderr
        assert "MEMORY LIMIT EXCEEDED" in done.stderr
        assert "memory budget 64 MiB (explicit, capped by inherited)" in done.stderr
        assert row["termination_cause"] == "memory-policy"
        assert row["cleanup_status"] == "unknown"
        assert row["memory_ceiling_mb"] == 64
        assert row["memory_budget_source"] == "explicit, capped by inherited"
        deadline = time.monotonic() + 5
        while _alive(token) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _alive(token) == [], "the allocating grandchild survived the breach"
    finally:
        _reap_token(token)


@pytest.mark.skipif(
    not bounded_contract.ReservedLeader.available(),
    reason=bounded_contract.ReservedLeader.unavailable_reason() or "Boundary available",
)
def test_real_leader_exit_sweeps_left_behind_sleeper(tmp_path):
    token = f"run-bounded-sweep-{uuid.uuid4().hex}"
    try:
        done, row = invoke(
            tmp_path,
            [
                "--contract=command-status",
                "--memory-mb=0",
                "--",
                "/bin/sh",
                "-c",
                f"exec 1>&- 2>&-; (exec -a {token} /bin/sleep 300) & exit 0",
            ],
        )
        assert done.returncode == 0, done.stderr
        assert row["termination_cause"] == "child-exit"
        assert row["cleanup_status"] == "unknown"
        assert "leader exited with members left" in done.stderr
        deadline = time.monotonic() + 5
        while _alive(token) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert _alive(token) == [], "the sweep left the sleeper running"
    finally:
        _reap_token(token)


@pytest.mark.parametrize("memory_args", [["--memory-mb=0"], ["--memory-mb=129"]])
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
    assert row["termination_cause"] == "memory-policy-refused"
    assert row["command_exit_code"] is None
    assert row["cleanup_status"] == "unknown"
    assert "refused before spawn: memory budget refused" in done.stderr
    assert "spawned=no" in done.stderr and f"invocation_id={row['invocation_id']}" in done.stderr


def test_real_refusals_explain_themselves(tmp_path):
    done, row = invoke(tmp_path, ["--contract=command-status", "--memory-mb=0", "--"])
    assert done.returncode == 125
    assert "no command was given" in done.stderr and "spawned=no" in done.stderr
    assert row["termination_cause"] == "invalid-request"


@pytest.mark.skipif(
    not bounded_contract.ReservedLeader.available(),
    reason=bounded_contract.ReservedLeader.unavailable_reason() or "Boundary available",
)
def test_real_command_that_cannot_start_names_argv0(tmp_path):
    done, row = invoke(
        tmp_path, ["--contract=command-status", "--memory-mb=0", "--", "/nonexistent/bin-xyz"]
    )
    assert done.returncode == 125
    assert "could not start '/nonexistent/bin-xyz': No such file or directory" in done.stderr
    assert "spawned=no" in done.stderr
    assert row["termination_cause"] == "spawn-refused"


def test_real_python_without_waitid_still_runs_on_darwin(tmp_path):
    # Architect ruling: 3.12 on macOS is supervised through the sysctl zombie
    # flag instead of refusing. Only a host with neither observer refuses.
    observer = bounded_contract.ReservedLeader.select_observer()
    done, row = invoke(
        tmp_path,
        ["--contract=command-status", "--memory-mb=0", "--", sys.executable, "-c", "print('ran')"],
    )
    if observer is None:
        assert done.returncode == 125 and row["termination_cause"] == "unsupported-boundary"
        assert "no non-consuming terminal observer" in row["cleanup_reason"]
        return
    assert done.returncode == 0, done.stderr
    assert done.stdout == "ran\n"
    assert row["terminal_observer"] == observer
    if not hasattr(os, "waitid"):
        assert observer == "darwin-sysctl-zombie"
