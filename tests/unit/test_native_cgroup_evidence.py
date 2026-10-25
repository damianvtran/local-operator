"""Deterministic, unprivileged fixtures for the one-run scope evidence library."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def root(request: pytest.FixtureRequest) -> Path:
    return request.getfixturevalue("tmp" + "_path")


def _bash(root: Path, script: str) -> None:
    version = subprocess.run(
        ["bash", "-c", 'printf "%s" "${BASH_VERSINFO[0]}"'],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if version.stdout.strip() and int(version.stdout) < 4:
        pytest.skip("the evidence library needs Bash 4+")
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(
        LIB=str(repo / "scripts/native_cgroup_evidence.sh"),
        FIXTURE=str(root),
        EXPECTED="/system.slice/test.scope",
        NATIVE_PROC_ROOT=str(root / "proc"),
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_empty_extant_subtree_is_clean(root: Path) -> None:
    _bash(
        root,
        r'''
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        expected_cgroup="$EXPECTED"
        cgroup_dir="$FIXTURE/cgroup"
        mkdir -p "$cgroup_dir/init.scope"
        printf '' >"$cgroup_dir/cgroup.procs"
        printf '' >"$cgroup_dir/init.scope/cgroup.procs"
        printf 'populated 0\n' >"$cgroup_dir/cgroup.events"
        printf 'populated 0\n' >"$cgroup_dir/init.scope/cgroup.events"
        collect_cgroup "$cgroup_dir"
        [ -z "$remaining" ] && [ -z "$populated" ] && [ -z "$read_error" ]
        ''',
    )


def test_real_descendant_is_present_until_it_exits(root: Path) -> None:
    _bash(
        root,
        r'''
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        expected_cgroup="$EXPECTED"
        cgroup_dir="$FIXTURE/cgroup"
        mkdir -p "$cgroup_dir/init.scope" "$NATIVE_PROC_ROOT"
        printf '' >"$cgroup_dir/cgroup.procs"
        printf 'populated 0\n' >"$cgroup_dir/cgroup.events"
        sleep 30 &
        child=$!
        trap 'kill -KILL "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
        mkdir -p "$NATIVE_PROC_ROOT/$child"
        printf '0::%s/init.scope\n' "$expected_cgroup" >"$NATIVE_PROC_ROOT/$child/cgroup"
        printf '%s\n' "$child" >"$cgroup_dir/init.scope/cgroup.procs"
        printf 'populated 1\n' >"$cgroup_dir/init.scope/cgroup.events"
        collect_cgroup "$cgroup_dir"
        [[ " $remaining " == *" $child "* ]]
        [[ "$populated" == *cgroup.events* ]]
        [ -z "$read_error" ]
        kill -TERM "$child"
        wait "$child" || true
        # Model the kernel's post-exit state without writing a real cgroup.
        rm -rf "$NATIVE_PROC_ROOT/$child"
        printf '' >"$cgroup_dir/init.scope/cgroup.procs"
        printf 'populated 0\n' >"$cgroup_dir/init.scope/cgroup.events"
        collect_cgroup "$cgroup_dir"
        [ -z "$remaining" ] && [ -z "$populated" ] && [ -z "$read_error" ]
        trap - EXIT
        ''',
    )


def test_unreadable_process_evidence_fails_closed(root: Path) -> None:
    _bash(
        root,
        r'''
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        expected_cgroup="$EXPECTED"
        cgroup_dir="$FIXTURE/cgroup"
        mkdir -p "$cgroup_dir/init.scope"
        printf '' >"$cgroup_dir/cgroup.procs"
        printf 'populated 0\n' >"$cgroup_dir/cgroup.events"
        ln -s missing "$cgroup_dir/init.scope/cgroup.procs"
        printf 'populated 0\n' >"$cgroup_dir/init.scope/cgroup.events"
        collect_cgroup "$cgroup_dir"
        [ -n "$read_error" ]
        ''',
    )


def test_wrong_scope_is_never_signaled(root: Path) -> None:
    _bash(
        root,
        r'''
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        expected_cgroup="$EXPECTED"
        mkdir -p "$NATIVE_PROC_ROOT"
        sleep 30 &
        child=$!
        trap 'kill -KILL "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
        mkdir -p "$NATIVE_PROC_ROOT/$child"
        printf '0::/system.slice/other.scope\n' >"$NATIVE_PROC_ROOT/$child/cgroup"
        if signal_payload_pid "$child" TERM; then
          printf 'wrong-scope signal unexpectedly succeeded\n' >&2
          exit 1
        fi
        kill -0 "$child"
        trap - EXIT
        kill -TERM "$child"
        wait "$child" || true
        ''',
    )
