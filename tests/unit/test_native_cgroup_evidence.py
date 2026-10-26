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
        r"""
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
        """,
    )


def test_real_descendant_is_present_until_it_exits(root: Path) -> None:
    _bash(
        root,
        r"""
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
        """,
    )


def test_unreadable_process_evidence_fails_closed(root: Path) -> None:
    _bash(
        root,
        r"""
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
        """,
    )


def test_wrong_scope_is_never_signaled(root: Path) -> None:
    _bash(
        root,
        r"""
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
        """,
    )


def test_missing_populated_record_fails_closed(root: Path) -> None:
    _bash(
        root,
        r"""
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        expected_cgroup="$EXPECTED"
        cgroup_dir="$FIXTURE/cgroup"
        mkdir -p "$cgroup_dir/init.scope"
        printf '' >"$cgroup_dir/cgroup.procs"
        printf '' >"$cgroup_dir/init.scope/cgroup.procs"
        printf 'populated 0\n' >"$cgroup_dir/cgroup.events"
        # An empty processes file plus no populated record is NOT proof of an
        # empty subtree. The old collector returned three empty values here.
        printf 'other 0\n' >"$cgroup_dir/init.scope/cgroup.events"
        collect_cgroup "$cgroup_dir"
        [ -z "$remaining" ] && [ -z "$populated" ]
        [ -n "$read_error" ]
        """,
    )


def test_already_reaped_child_is_a_noop_not_a_wrapper_failure(root: Path) -> None:
    _bash(
        root,
        r"""
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        expected_cgroup="$EXPECTED"
        NATIVE_PROC_ROOT=/proc
        sleep 0.01 &
        child=$!
        wait "$child" || true
        if kill -0 "$child" 2>/dev/null || [ -d "/proc/$child" ]; then
          printf 'child is unexpectedly still present\n' >&2
          exit 1
        fi
        # Disappearance before the signal is success: there is no descendant to
        # clean. Existing-but-unreadable/wrong-scope PIDs keep their failures.
        signal_payload_pid "$child" TERM
        """,
    )


def test_systemd_init_scope_migration_preserves_authenticated_cleanup(root: Path) -> None:
    _bash(
        root,
        r"""
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        PROBE_SCOPE=test.scope
        expected_cgroup="$EXPECTED"
        NATIVE_CGROUP_ROOT="$FIXTURE/sysfs"
        cgroup_root="$NATIVE_CGROUP_ROOT$expected_cgroup"
        mkdir -p "$NATIVE_PROC_ROOT/$$" "$cgroup_root/init.scope"
        printf '0::%s/init.scope\n' "$expected_cgroup" >"$NATIVE_PROC_ROOT/$$/cgroup"
        id() {
          case $1 in
            -u|-g) printf '1001\n' ;;
            -un|-gn) printf 'runner\n' ;;
            *) return 1 ;;
          esac
        }
        stat() {
          case $2 in
            %u|%g) printf '1001\n' ;;
            *) return 1 ;;
          esac
        }
        # systemd moves the payload from the delegated root into init.scope.
        # Authentication follows that one legitimate move but still anchors the
        # owner check and observation at the exact owned root.
        authenticate_scope
        [ "$scope_authenticated" = true ]
        [ "$cgroup_dir" = "$cgroup_root" ]

        sleep 30 &
        child=$!
        trap 'kill -KILL "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
        mkdir -p "$NATIVE_PROC_ROOT/$child"
        printf '0::%s/init.scope\n' "$expected_cgroup" >"$NATIVE_PROC_ROOT/$child/cgroup"
        printf '' >"$cgroup_root/cgroup.procs"
        printf 'populated 0\n' >"$cgroup_root/cgroup.events"
        printf '%s\n' "$$" "$child" >"$cgroup_root/init.scope/cgroup.procs"
        printf 'populated 1\n' >"$cgroup_root/init.scope/cgroup.events"
        collect_payload_cgroup
        [[ " $remaining " == *" $child "* ]]
        [[ "$populated" == *cgroup.events* ]]
        [ -z "$read_error" ]
        trap - EXIT
        kill -TERM "$child"
        wait "$child" || true
        """,
    )


def test_foreign_init_scope_is_not_authenticated(root: Path) -> None:
    _bash(
        root,
        r"""
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        PROBE_SCOPE=test.scope
        expected_cgroup="$EXPECTED"
        NATIVE_CGROUP_ROOT="$FIXTURE/sysfs"
        mkdir -p "$NATIVE_PROC_ROOT/$$" "$NATIVE_CGROUP_ROOT$expected_cgroup"
        printf '0::/system.slice/foreign.scope/init.scope\n' >"$NATIVE_PROC_ROOT/$$/cgroup"
        id() {
          case $1 in
            -u|-g) printf '1001\n' ;;
            -un|-gn) printf 'runner\n' ;;
            *) return 1 ;;
          esac
        }
        stat() {
          case $2 in
            %u|%g) printf '1001\n' ;;
            *) return 1 ;;
          esac
        }
        scope_authenticated=true
        if authenticate_scope; then
          printf 'foreign init.scope unexpectedly authenticated\n' >&2
          exit 1
        fi
        [ "$scope_authenticated" = false ]
        """,
    )


def test_own_init_scope_control_pid_is_signaled(root: Path) -> None:
    _bash(
        root,
        r"""
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        expected_cgroup="$EXPECTED"
        scope=test.scope
        mkdir -p "$NATIVE_PROC_ROOT"
        sleep 30 &
        child=$!
        trap 'kill -KILL "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
        mkdir -p "$NATIVE_PROC_ROOT/$child"
        printf '0::%s/init.scope\n' "$expected_cgroup" >"$NATIVE_PROC_ROOT/$child/cgroup"
        mock_systemctl() { printf '%s\n' "$child"; }
        mock_ps() { printf '%s\n' "$(id -u)"; }
        SYSTEMCTL=mock_systemctl
        PS_BIN=mock_ps
        signal_scope_control_pid TERM
        wait "$child" || true
        if kill -0 "$child" 2>/dev/null; then
          printf 'own init.scope ControlPID was not signaled\n' >&2
          exit 1
        fi
        trap - EXIT
        """,
    )


def test_foreign_init_scope_control_pid_is_not_signaled(root: Path) -> None:
    _bash(
        root,
        r"""
        set -euo pipefail
        shopt -s nullglob globstar
        source "$LIB"
        expected_cgroup="$EXPECTED"
        scope=test.scope
        mkdir -p "$NATIVE_PROC_ROOT"
        sleep 30 &
        child=$!
        trap 'kill -KILL "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
        mkdir -p "$NATIVE_PROC_ROOT/$child"
        printf '0::/system.slice/foreign.scope/init.scope\n' >"$NATIVE_PROC_ROOT/$child/cgroup"
        mock_systemctl() { printf '%s\n' "$child"; }
        mock_ps() { printf '%s\n' "$(id -u)"; }
        SYSTEMCTL=mock_systemctl
        PS_BIN=mock_ps
        if signal_scope_control_pid TERM; then
          printf 'foreign init.scope ControlPID unexpectedly signaled\n' >&2
          exit 1
        fi
        kill -0 "$child"
        [[ "$scope_error" == *outside* ]]
        trap - EXIT
        kill -TERM "$child"
        wait "$child" || true
        """,
    )


def test_history_allows_only_the_three_consumed_runs(root: Path) -> None:
    _bash(
        root,
        r"""
        set -euo pipefail
        source "$LIB"
        GH_BIN=mock_gh
        JQ_BIN=jq
        current=99999999999
        # The live history holds the three spent, reviewed diagnostics.
        mock_gh() {
          printf '%s\n' \
            '[{"databaseId":40223296178},{"databaseId":40238106261},{"databaseId":40257911039}]'
        }
        native_history_allows_retry owner/repo native-delegated-once.yml "$current"
        # Order must not matter, and the current run may also be listed once.
        mock_gh() {
          printf '%s\n' \
            '[{"databaseId":40257911039},{"databaseId":99999999999},{"databaseId":40223296178},{"databaseId":40238106261}]'
        }
        native_history_allows_retry owner/repo native-delegated-once.yml "$current"
        # A run that lists ITSELF beside the consumed set is the same history
        # with the current row already present, and is allowed.
        mock_gh() {
          printf '%s\n' \
            '[{"databaseId":40223296178},{"databaseId":40238106261},{"databaseId":40257911039}]'
        }
        native_history_allows_retry owner/repo native-delegated-once.yml 40238106261
        """,
    )


def test_history_refuses_a_fourth_run(root: Path) -> None:
    """MUTANT TARGET: a guard that ignores an extra row would authorize a rerun."""
    _bash(
        root,
        r"""
        set -euo pipefail
        source "$LIB"
        GH_BIN=mock_gh
        JQ_BIN=jq
        mock_gh() {
          printf '%s\n' \
            '[{"databaseId":40223296178},{"databaseId":40238106261},{"databaseId":40257911039},{"databaseId":123456789}]'
        }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'fourth run unexpectedly authorized\n' >&2
          exit 1
        fi
        # The extra row must be refused EVEN WHEN it is not the smallest possible
        # new id, so the rule cannot be a numeric range check in disguise.
        mock_gh() {
          printf '%s\n' \
            '[{"databaseId":40223296178},{"databaseId":40238106261},{"databaseId":40257911039},{"databaseId":99999999998}]'
        }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'high-id fourth run unexpectedly authorized\n' >&2
          exit 1
        fi
        """,
    )


def test_history_refuses_a_duplicate_current_row(root: Path) -> None:
    """The current run may appear once; a duplicated row is unexpected history."""
    _bash(
        root,
        r"""
        set -euo pipefail
        source "$LIB"
        GH_BIN=mock_gh
        JQ_BIN=jq
        mock_gh() {
          local rows='['
          rows+='{"databaseId":40223296178},'
          rows+='{"databaseId":40238106261},'
          rows+='{"databaseId":40257911039},'
          rows+='{"databaseId":99999999999},'
          rows+='{"databaseId":99999999999}]'
          printf '%s\n' "$rows"
        }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'duplicate current row unexpectedly authorized\n' >&2
          exit 1
        fi
        """,
    )


def test_history_refuses_a_duplicate_or_missing_consumed_run(root: Path) -> None:
    """MUTANT TARGET: `>= 1` of the three, or any single missing id, must refuse."""
    _bash(
        root,
        r"""
        set -euo pipefail
        source "$LIB"
        GH_BIN=mock_gh
        JQ_BIN=jq
        mock_gh() {
          printf '%s\n' \
            '[{"databaseId":40223296178},{"databaseId":40223296178},{"databaseId":40238106261},{"databaseId":40257911039}]'
        }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'duplicate consumed id unexpectedly authorized\n' >&2
          exit 1
        fi
        # Every one of the three is required: dropping any single id refuses.
        mock_gh() { printf '%s\n' '[{"databaseId":40223296178},{"databaseId":40238106261}]'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'missing third consumed id unexpectedly authorized\n' >&2
          exit 1
        fi
        mock_gh() { printf '%s\n' '[{"databaseId":40223296178},{"databaseId":40257911039}]'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'missing second consumed id unexpectedly authorized\n' >&2
          exit 1
        fi
        mock_gh() { printf '%s\n' '[{"databaseId":40238106261},{"databaseId":40257911039}]'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'missing first consumed id unexpectedly authorized\n' >&2
          exit 1
        fi
        # A `>= 1` rule would accept this one.
        mock_gh() { printf '%s\n' '[{"databaseId":40223296178}]'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'single consumed id unexpectedly authorized\n' >&2
          exit 1
        fi
        # A substitution: one reviewed id replaced by an unreviewed one.
        mock_gh() {
          printf '%s\n' \
            '[{"databaseId":40223296178},{"databaseId":40238106261},{"databaseId":55555555555}]'
        }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'substituted consumed id unexpectedly authorized\n' >&2
          exit 1
        fi
        """,
    )


def test_history_refuses_a_substituted_or_missing_history(root: Path) -> None:
    _bash(
        root,
        r"""
        set -euo pipefail
        source "$LIB"
        GH_BIN=mock_gh
        JQ_BIN=jq
        mock_gh() { printf '%s\n' '[{"databaseId":123456789}]'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'substituted history unexpectedly authorized\n' >&2
          exit 1
        fi
        mock_gh() { printf '%s\n' '[]'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'empty history unexpectedly authorized\n' >&2
          exit 1
        fi
        """,
    )


def test_history_refuses_api_and_malformed_response_failures(root: Path) -> None:
    _bash(
        root,
        r"""
        set -euo pipefail
        source "$LIB"
        GH_BIN=mock_gh
        JQ_BIN=jq
        mock_gh() { return 1; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'API failure unexpectedly authorized\n' >&2
          exit 1
        fi
        # Valid JSON followed by a non-zero exit must ALSO refuse.
        mock_gh() {
          printf '%s\n' \
            '[{"databaseId":40223296178},{"databaseId":40238106261},{"databaseId":40257911039}]'
          return 1
        }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'non-zero exit with allowed body unexpectedly authorized\n' >&2
          exit 1
        fi
        mock_gh() { printf '%s\n' 'not-json'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'malformed history unexpectedly authorized\n' >&2
          exit 1
        fi
        mock_gh() { printf '%s\n' '{"databaseId":40223296178}'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'non-array history unexpectedly authorized\n' >&2
          exit 1
        fi
        mock_gh() {
          printf '%s\n' '[]'
          printf '%s\n' '[{"databaseId":40223296178}]'
        }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'two JSON documents unexpectedly authorized\n' >&2
          exit 1
        fi
        mock_gh() { printf '%s\n' '[{"databaseId":null}]'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'null id unexpectedly authorized\n' >&2
          exit 1
        fi
        mock_gh() { printf '%s\n' '[{"databaseId":"40223296178"}]'; }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'string id unexpectedly authorized\n' >&2
          exit 1
        fi
        mock_gh() {
          printf '%s\n' \
            '[{"databaseId":40223296178},{"databaseId":40238106261}'
        }
        if native_history_allows_retry owner/repo native-delegated-once.yml 99999999999; then
          printf 'truncated JSON unexpectedly authorized\n' >&2
          exit 1
        fi
        """,
    )
