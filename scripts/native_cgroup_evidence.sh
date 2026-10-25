#!/usr/bin/env bash
# Shared by the one-run scope wrapper and its deterministic fixtures. All
# cgroup/proc reads use Bash builtins: an observer cat/awk must not become a new
# scoped PID and then be mistaken for a leaked descendant.
# shellcheck shell=bash

NATIVE_PROC_ROOT=${NATIVE_PROC_ROOT:-/proc}

read_pid_cgroup() {
  local pid=$1 line file="$NATIVE_PROC_ROOT/$1/cgroup"
  [ -r "$file" ] || return 1
  IFS= read -r line <"$file" || return 1
  case $line in
    0::*) printf '%s' "${line#0::}" ;;
    *) return 1 ;;
  esac
}

collect_cgroup() {
  local directory=$1 file pid key value extra fd path
  local -A seen=()
  remaining=""
  populated=""
  read_error=""
  [ -d "$directory" ] || return 0

  local process_files=("$directory"/**/cgroup.procs)
  if [ "${#process_files[@]}" -eq 0 ]; then
    read_error="no cgroup.procs beneath $directory"
  fi
  for file in "${process_files[@]}"; do
    if [ ! -r "$file" ]; then
      read_error+="${read_error:+; }unreadable $file"
      continue
    fi
    if ! exec {fd}<"$file"; then
      read_error+="${read_error:+; }could not open $file"
      continue
    fi
    while IFS= read -r -u "$fd" pid || [ -n "$pid" ]; do
      # Empty records are not PIDs and never become a one-character "remaining".
      [ -n "$pid" ] || continue
      if [[ ! "$pid" =~ ^[1-9][0-9]*$ ]]; then
        read_error+="${read_error:+; }non-PID record in $file"
        continue
      fi
      [ -z "${seen[$pid]+x}" ] || continue
      seen[$pid]=1
      if ! kill -0 "$pid" 2>/dev/null; then
        # A genuinely gone PID needs no signal. Anything still present but not
        # signalable/readable is unknown and must fail closed, not disappear.
        if [ -d "$NATIVE_PROC_ROOT/$pid" ]; then
          read_error+="${read_error:+; }live but unsignalable PID $pid"
        fi
        continue
      fi
      if ! path=$(read_pid_cgroup "$pid"); then
        read_error+="${read_error:+; }unreadable cgroup for live PID $pid"
        continue
      fi
      if [ "$path" != "$expected_cgroup" ] && [[ "$path" != "$expected_cgroup/"* ]]; then
        read_error+="${read_error:+; }PID $pid changed or mismatched scope"
        continue
      fi
      remaining+="$pid "
    done
    exec {fd}<&-
  done

  local event_files=("$directory"/**/cgroup.events)
  if [ "${#event_files[@]}" -eq 0 ]; then
    read_error+="${read_error:+; }no cgroup.events beneath $directory"
  fi
  for file in "${event_files[@]}"; do
    if [ ! -r "$file" ]; then
      read_error+="${read_error:+; }unreadable $file"
      continue
    fi
    if ! exec {fd}<"$file"; then
      read_error+="${read_error:+; }could not open $file"
      continue
    fi
    while IFS=' ' read -r -u "$fd" key value extra || [ -n "$key" ]; do
      [ "$key" = populated ] || continue
      case $value in
        0) ;;
        1) populated+="$file"$'\n' ;;
        *) read_error+="${read_error:+; }invalid populated value in $file" ;;
      esac
    done
    exec {fd}<&-
  done
}

scope_state() {
  local output rc
  scope_status=""
  scope_error=""
  if output=$("$SYSTEMCTL" is-active "$scope" 2>&1); then
    rc=0
  else
    rc=$?
  fi
  case $output in
    active | activating | inactive | failed) scope_status=$output ;;
    *"could not be found"* | *"not found"*) scope_status=not_found ;;
    *) scope_error="systemctl is-active rc=$rc: $output" ;;
  esac
}

signal_scope_control_pid() {
  local output pid owner pid_cgroup
  if ! output=$("$SYSTEMCTL" show "$scope" --property=ControlPID --value 2>/dev/null); then
    scope_error="could not read ControlPID for $scope"
    return 1
  fi
  [[ "$output" =~ ^[1-9][0-9]*$ ]] || return 0
  pid=$output
  if [ ! -r "$NATIVE_PROC_ROOT/$pid/cgroup" ]; then
    scope_error="unreadable $NATIVE_PROC_ROOT/$pid/cgroup"
    return 1
  fi
  owner=$("$PS_BIN" -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ "$owner" = "$(id -u)" ] || {
    scope_error="ControlPID $pid is not runner-owned"
    return 1
  }
  if ! pid_cgroup=$(read_pid_cgroup "$pid") || [ "$pid_cgroup" != "$expected_cgroup" ]; then
    scope_error="ControlPID $pid is outside $expected_cgroup"
    return 1
  fi
  kill -TERM "$pid" 2>/dev/null || [ ! -d "$NATIVE_PROC_ROOT/$pid" ]
}

collect_payload_cgroup() {
  remaining=""
  populated=""
  read_error=""
  [ -n "${cgroup_dir:-}" ] || {
    read_error="no cgroup path was available for exact cleanup"
    return 0
  }
  collect_cgroup "$cgroup_dir"
}

signal_payload_pid() {
  local pid=$1 signal=$2 pid_cgroup pid_owner
  [ "$pid" != "$$" ] || return 0
  pid_cgroup=$(read_pid_cgroup "$pid") || return 1
  if [ "$pid_cgroup" != "$expected_cgroup" ] && [[ "$pid_cgroup" != "$expected_cgroup/"* ]]; then
    return 1
  fi
  pid_owner=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ "$pid_owner" = "$(id -u)" ] || return 1
  kill "-$signal" "$pid" 2>/dev/null || [ ! -d "$NATIVE_PROC_ROOT/$pid" ]
}

authenticate_scope() {
  uid=$(id -u)
  gid=$(id -g)
  [ "$uid" -ne 0 ] && [ "$gid" -ne 0 ] || return 1
  [ "$(id -un)" = runner ] && [ "$(id -gn)" = runner ] || return 1
  IFS= read -r cgroup_line </proc/self/cgroup || return 1
  case $cgroup_line in
    0::*) cgroup=${cgroup_line#0::} ;;
    *) return 1 ;;
  esac
  expected_cgroup="/system.slice/$PROBE_SCOPE"
  [ "$cgroup" = "$expected_cgroup" ] || return 1
  cgroup_dir="/sys/fs/cgroup$cgroup"
  owner=$(stat -c '%u' "$cgroup_dir") || return 1
  group=$(stat -c '%g' "$cgroup_dir") || return 1
  [ "$owner" = "$uid" ] && [ "$group" = "$gid" ] || return 1
  scope_authenticated=true
}
