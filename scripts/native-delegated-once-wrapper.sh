#!/usr/bin/env bash
# One operator-authorized ephemeral CI diagnostic. No normal-PR invocation and
# no retry: the workflow's label/run-attempt/history guards call this only once.
# Exactly one privileged command is used (sudo -n systemd-run). Cleanup signals
# only authenticated runner-owned PIDs; all other scope inspection is read-only.
set -euo pipefail
shopt -s nullglob globstar

: "${GITHUB_WORKSPACE:?}" "${GITHUB_REPOSITORY:?}" "${RUNNER_TEMP:?}"
: "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}" "${EXPECTED_HEAD:?}"
: "${EXPECTED_WRAPPER_SHA256:?}" "${EXPECTED_PROBE_SHA256:?}" "${GH_TOKEN:?}"
[ "$GITHUB_RUN_ATTEMPT" = 1 ]
EXPECTED_BASE=7f4b0bab4e2ba4b7327ae36808627ba83c41d0c6
# The one-shot label carries the independently reviewed commit. The workflow
# need not contain its own commit SHA, and a later/relabelled head cannot pass.
[[ "$EXPECTED_HEAD" =~ ^native-delegated-once-([0-9a-f]{40})$ ]] || {
  printf 'label does not carry one exact reviewed head\n' >&2
  exit 2
}
EXPECTED_HEAD="${BASH_REMATCH[1]}"

SUDO=$(command -v sudo)
SYSTEMD_RUN=$(command -v systemd-run)
SYSTEMCTL=$(command -v systemctl)
BASH_BIN=$(command -v bash)
PROBE_PYTHON=$(command -v python)
PROBE_DBUS=$(command -v dbus-run-session)
GH_BIN=$(command -v gh)
JQ_BIN=$(command -v jq)
PS_BIN=$(command -v ps)
SHA256SUM=$(command -v sha256sum)
for executable in "$SUDO" "$SYSTEMD_RUN" "$SYSTEMCTL" "$BASH_BIN" \
  "$PROBE_PYTHON" "$PROBE_DBUS" "$GH_BIN" "$JQ_BIN" "$PS_BIN" "$SHA256SUM"; do
  [ -x "$executable" ] || { printf 'missing executable: %s\n' "$executable" >&2; exit 2; }
done

[ "$(git -C "$GITHUB_WORKSPACE" rev-parse HEAD)" = "$EXPECTED_HEAD" ]
[ "$(git -C "$GITHUB_WORKSPACE" merge-base HEAD "$EXPECTED_BASE")" = "$EXPECTED_BASE" ]
actual_wrapper_sha=$("$SHA256SUM" "$GITHUB_WORKSPACE/scripts/native-delegated-once-wrapper.sh")
actual_wrapper_sha=${actual_wrapper_sha%% *}
[ "$actual_wrapper_sha" = "$EXPECTED_WRAPPER_SHA256" ]
actual_probe_sha=$("$SHA256SUM" "$GITHUB_WORKSPACE/scripts/native_relay_update_probe.py")
actual_probe_sha=${actual_probe_sha%% *}
[ "$actual_probe_sha" = "$EXPECTED_PROBE_SHA256" ]

# A prior run of this one-shot workflow consumes the authorization even if a
# later label names another head. The concurrency key serializes relabels, so
# this read-only history check cannot race a second dispatch into sudo.
runs=$("$GH_BIN" run list --repo "$GITHUB_REPOSITORY" \
  --workflow native-delegated-once.yml --limit 100 --json databaseId)
consumed=$("$JQ_BIN" -r --arg current "$GITHUB_RUN_ID" \
  '[.[] | select((.databaseId | tostring) != $current)] | length' <<<"$runs")
[ "$consumed" = 0 ] || {
  printf 'one-shot diagnostic was already consumed by an earlier run\n' >&2
  exit 2
}

scope="lop-native-relay-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}.scope"
expected_cgroup="/system.slice/$scope"
scope_cgroup="/sys/fs/cgroup$expected_cgroup"
payload="$RUNNER_TEMP/native-delegated-payload.sh"
payload_rc="$RUNNER_TEMP/native-payload.rc"
preflight_receipt="$RUNNER_TEMP/native-scope-preflight.json"
cleanup_receipt="$RUNNER_TEMP/native-scope-cleanup.json"
outer_receipt="$RUNNER_TEMP/native-scope-outer.json"
rm -f "$payload_rc" "$preflight_receipt" "$cleanup_receipt" "$outer_receipt"

# Recursive process/populated evidence. Any unreadable file is an error, never
# an empty cgroup. These globals are reset on every collection.
collect_cgroup() {
  local directory=$1 file content value
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
      read_error="unreadable $file"
    elif ! content=$(cat "$file" 2>/dev/null); then
      read_error="could not read $file"
    else
      remaining+="$content"$'\n'
    fi
  done

  local event_files=("$directory"/**/cgroup.events)
  if [ "${#event_files[@]}" -eq 0 ]; then
    read_error="no cgroup.events beneath $directory"
  fi
  for file in "${event_files[@]}"; do
    if [ ! -r "$file" ]; then
      read_error="unreadable $file"
    elif ! value=$(awk '$1 == "populated" {print $2}' "$file" 2>/dev/null); then
      read_error="could not read $file"
    elif [ "$value" = 1 ]; then
      populated+="$file"$'\n'
    elif [ "$value" != 0 ]; then
      read_error="missing populated value in $file"
    fi
  done
}

scope_status=""
scope_error=""
scope_state() {
  local output rc
  scope_status=""
  scope_error=""
  if output=$("$SYSTEMCTL" is-active "$scope" 2>&1); then
    rc=0
  else
    rc=$?
  fi
  case "$output" in
    active | activating | inactive | failed)
      scope_status="$output"
      ;;
    *"could not be found"* | *"not found"* | *"No such unit"*)
      scope_status="not_found"
      ;;
    *)
      scope_status="unknown"
      scope_error="systemctl is-active rc=$rc: $output"
      ;;
  esac
}

# Signal only the scope's runner-owned ControlPID, after checking its exact
# cgroup path (or a descendant of it) and uid. No privileged stop is needed.
signal_scope_control_pid() {
  local pid owner pid_cgroup output
  if ! output=$("$SYSTEMCTL" show "$scope" --property=ControlPID --value 2>/dev/null); then
    scope_error="could not read ControlPID for $scope"
    return 1
  fi
  [[ "$output" =~ ^[0-9]+$ ]] || return 0
  [ "$output" -gt 0 ] || return 0
  pid=$output
  if [ ! -r "/proc/$pid/cgroup" ]; then
    scope_error="unreadable /proc/$pid/cgroup"
    return 1
  fi
  pid_cgroup=$(awk -F: '$1 == "0" {print $3}' "/proc/$pid/cgroup")
  case "$pid_cgroup" in
    "$expected_cgroup" | "$expected_cgroup"/*) ;;
    *)
      scope_error="ControlPID $pid is outside $expected_cgroup"
      return 1
      ;;
  esac
  owner=$("$PS_BIN" -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ "$owner" != "$(id -u)" ]; then
    scope_error="ControlPID $pid is not runner-owned"
    return 1
  fi
  kill -TERM "$pid" 2>/dev/null || [ ! -d "/proc/$pid" ]
}

attempted=false
cleanup_interrupted=false
deadline=$((SECONDS + 920))
outer_cleanup() {
  rc=$?
  # Prevent recursive EXIT while retaining signal handlers that only RECORD an
  # interruption. Verification therefore cannot be skipped by a second signal.
  trap - EXIT
  trap 'cleanup_interrupted=true' INT TERM HUP
  set +e
  if [ "$attempted" = true ]; then
    scope_state
    if [ "$scope_status" = active ] || [ "$scope_status" = activating ]; then
      signal_scope_control_pid
    fi
    while [ "$SECONDS" -lt "$deadline" ]; do
      scope_state
      [ "$scope_status" = active ] || [ "$scope_status" = activating ] || break
      sleep 2
    done
  fi

  scope_state
  collect_cgroup "$scope_cgroup"
  cleanup_ok=false
  case "$scope_status" in
    inactive | failed | not_found)
      if [ -z "$scope_error" ] && [ -z "$read_error" ] && \
        [ -z "$remaining" ] && [ -z "$populated" ]; then
        cleanup_ok=true
      fi
      ;;
  esac
  printf '{"scope":"%s","status":"%s","status_error":"%s","read_error":"%s","remaining":"%s","populated":"%s","cleanup_verified":%s}\n' \
    "$scope" "$scope_status" "$scope_error" "$read_error" \
    "$(printf '%s' "$remaining" | tr '\n' ' ')" \
    "$(printf '%s' "$populated" | tr '\n' ' ')" "$cleanup_ok" >"$outer_receipt"
  [ "$cleanup_ok" = true ] || { [ "$rc" -ne 0 ] || rc=1; }
  [ "$cleanup_interrupted" = false ] || { [ "$rc" -ne 0 ] || rc=130; }
  exit "$rc"
}
trap outer_cleanup EXIT
trap 'exit 130' INT TERM HUP

cat >"$payload" <<'PAYLOAD'
#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob globstar
expected_cgroup="/system.slice/$PROBE_SCOPE"
cgroup_dir="/sys/fs/cgroup$expected_cgroup"
uid=""
gid=""
cgroup=""
owner=""
group=""

# Authenticate independently in preflight AND cleanup. Cleanup starts with
# signals disarmed and may signal only after this exact predicate succeeds.
authenticate_scope() {
  uid=$(id -u)
  gid=$(id -g)
  cgroup=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
  [ "$uid" -ne 0 ] && [ "$gid" -ne 0 ] || return 1
  [ "$(id -un)" = runner ] && [ "$(id -gn)" = runner ] || return 1
  [ "$cgroup" = "$expected_cgroup" ] || return 1
  [ -d "$cgroup_dir" ] || return 1
  owner=$(stat -c '%u' "$cgroup_dir")
  group=$(stat -c '%g' "$cgroup_dir")
  [ "$owner" = "$uid" ] && [ "$group" = "$gid" ]
}

collect_payload_cgroup() {
  local file content value
  remaining=""
  populated=""
  read_error=""
  if [ ! -d "$cgroup_dir" ]; then
    read_error="scope cgroup is absent while payload is still running"
    return 0
  fi
  local process_files=("$cgroup_dir"/**/cgroup.procs)
  [ "${#process_files[@]}" -gt 0 ] || read_error="no cgroup.procs beneath scope"
  for file in "${process_files[@]}"; do
    if [ ! -r "$file" ]; then
      read_error="unreadable $file"
    elif ! content=$(cat "$file" 2>/dev/null); then
      read_error="could not read $file"
    else
      remaining+="$content"$'\n'
    fi
  done
  local event_files=("$cgroup_dir"/**/cgroup.events)
  [ "${#event_files[@]}" -gt 0 ] || read_error="no cgroup.events beneath scope"
  for file in "${event_files[@]}"; do
    if [ ! -r "$file" ]; then
      read_error="unreadable $file"
    elif ! value=$(awk '$1 == "populated" {print $2}' "$file" 2>/dev/null); then
      read_error="could not read $file"
    elif [ "$value" = 1 ]; then
      populated+="$file"$'\n'
    elif [ "$value" != 0 ]; then
      read_error="missing populated value in $file"
    fi
  done
}

signal_payload_pid() {
  local pid=$1 signal=$2 pid_cgroup pid_owner
  [ "$pid" != "$$" ] || return 0
  [ -r "/proc/$pid/cgroup" ] || return 0
  pid_cgroup=$(awk -F: '$1 == "0" {print $3}' "/proc/$pid/cgroup")
  case "$pid_cgroup" in
    "$expected_cgroup" | "$expected_cgroup"/*) ;;
    *) return 1 ;;
  esac
  pid_owner=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ "$pid_owner" = "$uid" ] || return 1
  kill "-$signal" "$pid" 2>/dev/null || [ ! -d "/proc/$pid" ]
}

cleanup_interrupted=false
payload_cleanup() {
  rc=$?
  trap - EXIT
  trap 'cleanup_interrupted=true' INT TERM HUP
  set +e
  scope_authenticated=false
  if authenticate_scope; then
    scope_authenticated=true
  fi

  if [ "$scope_authenticated" = true ]; then
    collect_payload_cgroup
    for pid in $remaining; do
      signal_payload_pid "$pid" TERM || rc=1
    done
    for _ in $(seq 1 20); do
      collect_payload_cgroup
      pending=""
      for pid in $remaining; do [ "$pid" = "$$" ] || pending+="$pid "; done
      [ -n "$pending" ] || break
      sleep 0.1
    done
    collect_payload_cgroup
    for pid in $remaining; do
      signal_payload_pid "$pid" KILL || rc=1
    done
  fi

  collect_payload_cgroup
  remaining_without_self=""
  for pid in $remaining; do [ "$pid" = "$$" ] || remaining_without_self+="$pid "; done
  cleanup_verified=false
  if [ "$scope_authenticated" = true ] && [ -z "$read_error" ] && \
    [ -z "$remaining_without_self" ]; then
    cleanup_verified=true
  fi
  [ "$cleanup_verified" = true ] || { [ "$rc" -ne 0 ] || rc=1; }
  [ "$cleanup_interrupted" = false ] || { [ "$rc" -ne 0 ] || rc=130; }
  printf '%s\n' "$rc" >"$PAYLOAD_RC"
  printf '{"payload_rc":%s,"scope_authenticated":%s,"read_error":"%s","remaining_except_payload":"%s","populated":"%s","cleanup_verified":%s}\n' \
    "$rc" "$scope_authenticated" "$read_error" "$remaining_without_self" \
    "$(printf '%s' "$populated" | tr '\n' ' ')" "$cleanup_verified" >"$CLEANUP_RECEIPT"
  rm -rf "$RUNNER_TEMP/driver-venv" "$RUNNER_TEMP/uv-cache"
  exit "$rc"
}
trap payload_cleanup EXIT
trap 'exit 130' INT TERM HUP

# This proof precedes venv/pip/generation installation. A wrong cgroup or owner
# fails here; the cleanup above still cannot signal an unauthenticated scope.
authenticate_scope
[ -w "$cgroup_dir" ]
printf '{"uid":%s,"gid":%s,"scope":"%s","cgroup":"%s","owner":%s,"group":%s,"writable":true,"installs_started":false}\n' \
  "$uid" "$gid" "$PROBE_SCOPE" "$cgroup" "$owner" "$group" >"$PREFLIGHT_RECEIPT"

cd "$PROBE_REPO"
unset PYTHONPATH PYTHONHOME
"$PROBE_PYTHON" -m venv "$RUNNER_TEMP/driver-venv"
driver_python="$RUNNER_TEMP/driver-venv/bin/python"
"$driver_python" -m pip install --upgrade pip
"$driver_python" -m pip install uv
"$driver_python" -m pip install -e '.[dev]'
export GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0
export UV_CACHE_DIR="$RUNNER_TEMP/uv-cache"
export PATH="$RUNNER_TEMP/driver-venv/bin:$PATH"
# Do not exec: the payload EXIT trap must perform exact cleanup on every result.
"$PROBE_DBUS" -- "$driver_python" scripts/native_relay_update_probe.py \
  --repo "$PROBE_REPO" --base "$PROBE_BASE" --json "$RUNNER_TEMP/native-relay-update.json"
PAYLOAD
chmod 700 "$payload"

# The only privileged invocation. --uid/--gid supply the credential drop;
# User/Group supply delegated cgroup ownership; RuntimeMaxSec is the hard bound.
attempted=true
set +e
"$SUDO" -n "$SYSTEMD_RUN" --scope --collect \
  --unit="$scope" \
  --property=Delegate=yes \
  --property=User=runner --property=Group=runner \
  --property=RuntimeMaxSec=900 \
  --uid=runner --gid=runner \
  --setenv=HOME=/home/runner \
  --setenv=PROBE_REPO="$GITHUB_WORKSPACE" \
  --setenv=PROBE_BASE="$EXPECTED_BASE" \
  --setenv=PROBE_SCOPE="$scope" \
  --setenv=PROBE_PYTHON="$PROBE_PYTHON" \
  --setenv=PROBE_DBUS="$PROBE_DBUS" \
  --setenv=RUNNER_TEMP="$RUNNER_TEMP" \
  --setenv=PAYLOAD_RC="$payload_rc" \
  --setenv=PREFLIGHT_RECEIPT="$preflight_receipt" \
  --setenv=CLEANUP_RECEIPT="$cleanup_receipt" \
  --setenv=GITHUB_WORKSPACE="$GITHUB_WORKSPACE" \
  --setenv=GITHUB_RUN_ID="$GITHUB_RUN_ID" \
  --setenv=GITHUB_RUN_ATTEMPT="$GITHUB_RUN_ATTEMPT" \
  "$BASH_BIN" "$payload" >"$RUNNER_TEMP/native-systemd-run.log" 2>&1
launch_rc=$?
set -e
printf '%s\n' "$launch_rc" >"$RUNNER_TEMP/native-launch.rc"
[ "$launch_rc" -eq 0 ]

while [ "$SECONDS" -lt "$deadline" ]; do
  scope_state
  [ "$scope_status" = active ] || [ "$scope_status" = activating ] || break
  sleep 2
done

# File presence alone is not evidence: validate the actual preflight and exact
# cleanup predicates. The outer EXIT trap independently verifies the empty scope.
"$JQ_BIN" -e '.uid != 0 and .gid != 0 and .owner == .uid and .group == .gid and .writable == true' \
  "$preflight_receipt" >/dev/null
"$JQ_BIN" -e '.scope_authenticated == true and .cleanup_verified == true' \
  "$cleanup_receipt" >/dev/null
[ -f "$payload_rc" ]
exit "$(cat "$payload_rc")"
