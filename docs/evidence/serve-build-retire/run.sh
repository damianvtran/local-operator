#!/usr/bin/env bash
# Evidence for the ``serve`` daemon's build watch: announce early, keep serving,
# refuse once the drain is empty, then leave.
#
# Six runs against REAL daemons on ephemeral ports, each with its own isolated
# config root and a FAKE install root (LOP_BUILD_PREFIX) whose ``.lop-source``
# marker is flipped under the running process — exactly the file ``lop-update``
# writes last, and the whole signal this feature reads.
#
#   1. unsupervised  - an unclaimed daemon: the handover is announced in the
#                      record WHILE the daemon keeps answering, the restart line
#                      is logged, the refusal follows, the record is removed.
#   2. refusal matrix - a claimed daemon: the announcement is readable while a
#                      create still answers 200, then the refusal is live and
#                      EVERY path that can admit or start work answers the typed
#                      503 (and no runtime record appears).
#   3. held relay    - THE app-attached case: the desktop app's own
#                      ``/v1/desktop/sessions/{id}/events`` relay is held across
#                      the update. The announcement must be READABLE within one
#                      check while the daemon keeps serving, across several check
#                      intervals, with NO latch and NO exit; dropping the relay is
#                      what then lets it finish. (A job SSE stream would not test
#                      this: it is per-turn, and it is the one term the desktop
#                      app does not use.)
#   4. write failure - the retirement poll's announcement cannot be written (the
#                      record directory is made read-only): the daemon keeps
#                      serving, does NOT latch, and says so loudly; when the
#                      directory is writable again the sequence completes.
#   5. --reload      - a dev-mode child must not self-retire: nothing at all
#                      happens, and /health keeps answering.
#   6. probe failure - a probe the daemon cannot READ must mean "stay" rather
#                      than "nothing in flight" (``inject.py``; a raising probe
#                      cannot be provoked in an unprivileged real daemon).
#   7. both phases    - ONE daemon, the same routes, twice: merely announced (every
#                      route answered, and the daemon log NAMES the spawn seam
#                      while they run) and then latched (every route refused with
#                      the typed 503, and the same instrument counts zero). The
#                      second half of that contrast is what makes the first a
#                      measurement rather than a list (review round 2, MINOR-1).
#   8. reverted       - the announcement is RE-READ: back to the boot build it is
#                      withdrawn and the daemon keeps serving; forward again it is
#                      announced again; on to a third build the record names that
#                      one (review round 2, MINOR-2).
#   9. unreadable     - a build marker nobody can READ is not a build to leave for:
#                      aged past the settle and ``chmod 000``-ed, the daemon stays,
#                      and restoring the mode retires it onto the same file (QA
#                      round 2, OBS-1).
#
# Nothing here touches a live daemon: every config root is under $WORK, every port
# is ephemeral (--port 0), and the operator's exported desktop token is scrubbed
# so the ungoverned run exercises the ungoverned posture it claims to.
#
# Run from anywhere; the repo root is located from this script's own path:
#     bash docs/evidence/serve-build-retire/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
LOP="$REPO/.venv/bin/lop"
WORK="${WORK:-/tmp/lo-serve-retire-evidence}"
OLD_SHA="1111111111111111111111111111111111111111"
NEW_SHA="2222222222222222222222222222222222222222"
THIRD_SHA="3333333333333333333333333333333333333333"
TOKEN="evidence-desktop-token"

rm -rf "$WORK"
mkdir -p "$WORK/prefix"
PREFIX="$WORK/prefix"
printf '%s %s\n' "$OLD_SHA" "old-build" > "$PREFIX/.lop-source"

banner() { printf '\n========== %s ==========\n' "$*"; }

# Every daemon this script starts, so a failure mid-run cannot leave listeners
# (or, worse, a half-retired daemon) behind on the operator's machine.
DAEMON_PIDS=()
cleanup() {
  for pid in "${DAEMON_PIDS[@]:-}"; do
    [ -n "${pid:-}" ] || continue
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

start_daemon() {  # <tag> [extra env assignments...]
  local tag="$1"; shift
  CFG="$WORK/cfg-$tag"
  mkdir -p "$CFG"
  # The operator's own desktop token and every CMUX_* variable are scrubbed per
  # daemon: an inherited token would make an "ungoverned" run governed, and an
  # inherited workspace id is how a headless run has renamed real workspaces here.
  env -u LOCAL_OPERATOR_DESKTOP_TOKEN -u LOCAL_OPERATOR_DESKTOP_ORIGINS \
      -u CMUX_SOCKET_PATH -u CMUX_WORKSPACE_ID -u CMUX_SURFACE_ID \
      LOCAL_OPERATOR_CONFIG_DIR="$CFG" \
      LOP_BUILD_PREFIX="$PREFIX" "$@" \
      "$LOP" serve --host 127.0.0.1 --port 0 >"$WORK/$tag.log" 2>&1 &
  DAEMON_PID=$!
  DAEMON_PIDS+=("$DAEMON_PID")
  RECORD="$CFG/run/serve/$DAEMON_PID.json"
  for _ in $(seq 1 600); do [ -f "$RECORD" ] && break; sleep 0.05; done
  if [ ! -f "$RECORD" ]; then
    echo "FAIL: the daemon published no record; log follows:"
    cat "$WORK/$tag.log"
    return 1
  fi
  echo "started daemon pid=$DAEMON_PID tag=$tag"
  echo "record at $RECORD:"
  cat "$RECORD"; echo
}

stop_daemon() {  # never leave a stray listener behind
  kill "$DAEMON_PID" 2>/dev/null || true
  wait "$DAEMON_PID" 2>/dev/null || true
}

reset_marker() { printf '%s %s\n' "$OLD_SHA" "old-build" > "$PREFIX/.lop-source"; }

banner "1. an unsupervised daemon announces its handover while still serving"
start_daemon unsupervised LOG_LEVEL=INFO
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" --cwd "$WORK" | tee "$WORK/run1.txt"
echo "--- daemon log: the announcement, the refusal and the restart line ---"
grep -E "build watch baseline|announced in the record|refusing new work|lop serve" \
  "$WORK/unsupervised.log" || tail -8 "$WORK/unsupervised.log"
echo "--- record after exit ---"
ls -l "$CFG/run/serve/" || true
stop_daemon

banner "2. a claimed daemon: announced-but-admitting, then the refusal matrix"
reset_marker
# A LONG refusal window (test-only override) so the whole matrix runs inside it,
# and a short settle so the run does not wait 10 s for the marker to age.
start_daemon governed LOG_LEVEL=INFO LOCAL_OPERATOR_DESKTOP_TOKEN="$TOKEN" \
  LOP_BUILD_SETTLE_S=1 LOP_BUILD_STAGGER_S=300
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" \
  --token "$TOKEN" --refuse-matrix --leave-latched \
  --log "$WORK/governed.log" --cwd "$WORK" | tee "$WORK/run2.txt"
echo "--- SIGTERM inside the refusal window (the harness' own stop) ---"
stop_daemon
echo "--- record after the stop (a clean exit removes it) ---"
ls -l "$CFG/run/serve/" || true
echo "--- runtime records under the isolated root (an OBSERVATION, not the instrument: ---"
echo "--- the claim is the spawn-seam count printed by the matrix above) ---"
ls -A "$CFG/run/mobile" 2>/dev/null || echo "(no run/mobile directory at all)"
echo "--- sessions under the isolated root (an observation: the driver's own PRE-latch"
echo "--- creates; the refused create is the 503 row in the matrix above) ---"
ls -1 "$CFG/sessions" 2>/dev/null | wc -l
echo "--- daemon log ---"
grep -E "build watch baseline|announced in the record|refusing new work" "$WORK/governed.log" \
  || tail -8 "$WORK/governed.log"

banner "3. the app's own relay holds the announced daemon, then lets it finish"
reset_marker
# Production constants on purpose: this run is the one whose timings the design
# quotes (settle 10 s, check 5 s, notice a jittered slice of 20 s).
start_daemon held LOG_LEVEL=INFO LOCAL_OPERATOR_DESKTOP_TOKEN="$TOKEN"
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" \
  --token "$TOKEN" --hold-desktop 24 --cwd "$WORK" | tee "$WORK/run3.txt"
echo "--- daemon log: it names the term holding it ---"
grep -E "build watch baseline|announced in the record|in flight|refusing new work" \
  "$WORK/held.log" || tail -10 "$WORK/held.log"
echo "--- record after exit ---"
ls -l "$CFG/run/serve/" || true
stop_daemon

banner "4. an unwritable record directory neither latches the daemon nor stops the poll"
reset_marker
start_daemon readonly LOG_LEVEL=INFO LOCAL_OPERATOR_DESKTOP_TOKEN="$TOKEN" \
  LOP_BUILD_SETTLE_S=1 LOP_BUILD_STAGGER_S=10
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" \
  --token "$TOKEN" --readonly-record-dir 12 --cwd "$WORK" | tee "$WORK/run4.txt"
echo "--- daemon log: the failed write is loud, and the sequence then completes ---"
grep -E "could not be written|announced in the record|refusing new work" "$WORK/readonly.log" \
  | head -6 || tail -8 "$WORK/readonly.log"
echo "--- record after exit ---"
ls -l "$CFG/run/serve/" || true
stop_daemon

banner "5. a --reload child does not self-retire (nothing happens at all)"
reset_marker
CFG="$WORK/cfg-reload"
mkdir -p "$CFG"
env -u LOCAL_OPERATOR_DESKTOP_TOKEN -u LOCAL_OPERATOR_DESKTOP_ORIGINS \
    -u CMUX_SOCKET_PATH -u CMUX_WORKSPACE_ID -u CMUX_SURFACE_ID \
    LOCAL_OPERATOR_CONFIG_DIR="$CFG" LOP_BUILD_PREFIX="$PREFIX" LOG_LEVEL=INFO \
    "$LOP" serve --reload --host 127.0.0.1 --port 0 >"$WORK/reload.log" 2>&1 &
RELOAD_PID=$!
DAEMON_PIDS+=("$RELOAD_PID")
RECORD=""
# ``|| true`` on the pipeline: the child takes a few seconds to boot, so the
# first iterations find no record at all and ``set -e -o pipefail`` would end
# the run there rather than waiting for it (which is exactly what happened the
# first time this run existed).
for _ in $(seq 1 600); do
  RECORD="$(ls "$CFG"/run/serve/*.json 2>/dev/null | head -1 || true)"
  [ -n "$RECORD" ] && break
  sleep 0.1
done
if [ -z "$RECORD" ]; then
  echo "FAIL: the --reload child published no record; log follows:"
  cat "$WORK/reload.log"
  exit 1
fi
echo "reload child record: $RECORD"; cat "$RECORD"; echo
PORT="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['port'])" "$RECORD")"
echo "reloader parent pid=$RELOAD_PID (it owns the port; the child serves through it)"
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" \
  --expect-no-retire --window 24 --cwd "$WORK" | tee "$WORK/run5.txt"
echo "--- reload log ---"
grep -E "build watch baseline" "$WORK/reload.log" || echo "(no build watch line at all)"
echo "--- stopping the reloader (it owns the child) ---"
kill "$RELOAD_PID" 2>/dev/null || true
wait "$RELOAD_PID" 2>/dev/null || true
sleep 2
echo "records left under run/serve: $(ls -A "$CFG"/run/serve 2>/dev/null | wc -l | tr -d ' ')"
echo "listeners left on port $PORT: $(lsof -nP -iTCP:"$PORT" 2>/dev/null | wc -l | tr -d ' ')"

banner "6. a probe the daemon cannot read means STAY (injected at the probe seam)"
# the worktree's own interpreter: inject.py drives the REAL poll, so it has to
# import ``local_operator`` (drive.py deliberately does not, and runs anywhere).
"$REPO/.venv/bin/python" "$HERE/inject.py" | tee "$WORK/run6.txt"

banner "7. one daemon, both phases: ANNOUNCED (every route answered) then LATCHED (every route refused)"
reset_marker
# A short settle so the announcement lands inside the relay hold, and the default
# stagger: the announced matrix (which includes one real /warm and one /watch,
# so it takes a few seconds) runs while the relay is still held, and the relay is
# dropped the moment it finishes, which is what lets the drain empty.
start_daemon phases LOG_LEVEL=INFO LOCAL_OPERATOR_DESKTOP_TOKEN="$TOKEN" LOP_BUILD_SETTLE_S=1
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" \
  --token "$TOKEN" --hold-desktop 60 --matrix-announced --refuse-matrix \
  --log "$WORK/phases.log" --cwd "$WORK" | tee "$WORK/run7.txt"
echo "--- daemon log: announce, then refuse ---"
grep -E "build watch baseline|announced in the record|in flight|refusing new work" \
  "$WORK/phases.log" || tail -8 "$WORK/phases.log"
echo "--- record after exit ---"
ls -l "$CFG/run/serve/" || true
stop_daemon

banner "8. the announcement is RE-READ: a reverted install withdraws it, a further move re-announces it"
reset_marker
start_daemon revert LOG_LEVEL=INFO LOCAL_OPERATOR_DESKTOP_TOKEN="$TOKEN" LOP_BUILD_SETTLE_S=1
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" \
  --token "$TOKEN" --hold-desktop 120 --revert --revert-to "$OLD_SHA" \
  --third-sha "$THIRD_SHA" --cwd "$WORK" | tee "$WORK/run8.txt"
echo "--- daemon log: the withdrawal and the re-announcements ---"
grep -E "build watch baseline|announced in the record|no longer holds|moved on to|refusing new work" \
  "$WORK/revert.log" || tail -10 "$WORK/revert.log"
echo "--- record after exit ---"
ls -l "$CFG/run/serve/" || true
stop_daemon

banner "9. an UNREADABLE build marker means STAY (never leave for a build you could not read)"
reset_marker
start_daemon unreadable LOG_LEVEL=INFO LOCAL_OPERATOR_DESKTOP_TOKEN="$TOKEN" LOP_BUILD_SETTLE_S=1
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" \
  --unreadable-marker --window 22 --token "$TOKEN" --cwd "$WORK" | tee "$WORK/run9.txt"
echo "--- daemon log: no announcement while it was unreadable, then the ordinary sequence ---"
grep -E "build watch baseline|announced in the record|refusing new work" "$WORK/unreadable.log" \
  || tail -8 "$WORK/unreadable.log"
echo "--- record after exit ---"
ls -l "$CFG/run/serve/" || true
stop_daemon

banner "10. the two instruments, shown FAILING (they are the reason a green run means something)"
# No daemon: the completeness walk runs over a scratch copy of a real router module,
# and the spawn-seam spy drives the ASSEMBLED app with the door's refusal removed —
# the same state review round 2 measured on a live daemon. The worktree interpreter,
# because this deliberately imports the suite's own walker and the real app.
"$REPO/.venv/bin/python" "$HERE/ungated.py" | tee "$WORK/run10.txt"

banner "daemon logs"
for tag in unsupervised governed held readonly reload phases revert unreadable; do
  echo "--- $tag.log ---"
  cat "$WORK/$tag.log"
done

banner "raw artifacts under $WORK"
ls -l "$WORK"
