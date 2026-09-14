#!/usr/bin/env bash
# Evidence for the ``serve`` daemon's build watch: announce, refuse, leave.
#
# Three runs against REAL daemons on ephemeral ports, each with its own isolated
# config root and a FAKE install root (LOP_BUILD_PREFIX) whose ``.lop-source``
# marker is flipped under the running process — exactly the file ``lop-update``
# writes last, and the whole signal this feature reads.
#
#   1. unsupervised  - an unclaimed daemon: the handover is announced in the
#                      record, the restart line is logged, the record is removed.
#   2. governed      - a claimed daemon: the same sequence, plus the new-session
#                      request refused with 503 daemon-retiring while retiring.
#   3. held stream   - an SSE job stream held open across the update: NO
#                      retirement across several check intervals, and the
#                      retirement once the stream is released.
#
# Nothing here touches a live daemon: every config root is under $WORK, every port
# is ephemeral (--port 0), and the operator's exported desktop token is scrubbed
# so run 1 and 3 exercise the ungoverned posture they claim to.
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
TOKEN="evidence-desktop-token"

rm -rf "$WORK"
mkdir -p "$WORK/prefix"
PREFIX="$WORK/prefix"
printf '%s %s\n' "$OLD_SHA" "old-build" > "$PREFIX/.lop-source"

banner() { printf '\n========== %s ==========\n' "$*"; }

start_daemon() {  # <tag> [extra env assignments...]
  local tag="$1"; shift
  CFG="$WORK/cfg-$tag"
  mkdir -p "$CFG"
  env -u LOCAL_OPERATOR_DESKTOP_TOKEN -u LOCAL_OPERATOR_DESKTOP_ORIGINS \
      LOCAL_OPERATOR_CONFIG_DIR="$CFG" \
      LOP_BUILD_PREFIX="$PREFIX" "$@" \
      "$LOP" serve --host 127.0.0.1 --port 0 >"$WORK/$tag.log" 2>&1 &
  DAEMON_PID=$!
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

banner "1. an unsupervised daemon announces its handover, then exits"
start_daemon unsupervised LOG_LEVEL=INFO
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" --cwd "$WORK" | tee "$WORK/run1.txt"
echo "--- daemon log (tail) ---"
grep -E "retire|lop serve" "$WORK/unsupervised.log" || tail -6 "$WORK/unsupervised.log"
echo "--- record after exit ---"
ls -l "$CFG/run/serve/" || true
stop_daemon

# Recreate the fake install for the next run: run 1 flipped the marker.
printf '%s %s\n' "$OLD_SHA" "old-build" > "$PREFIX/.lop-source"

banner "2. a claimed daemon refuses a new session while retiring"
start_daemon governed LOG_LEVEL=INFO LOCAL_OPERATOR_DESKTOP_TOKEN="$TOKEN" LOP_BUILD_STAGGER_S=10
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" \
  --token "$TOKEN" --cwd "$WORK" | tee "$WORK/run2.txt"
echo "--- daemon log (tail) ---"
grep -E "retire|lop serve|503" "$WORK/governed.log" || tail -6 "$WORK/governed.log"
echo "--- record after exit ---"
ls -l "$CFG/run/serve/" || true
stop_daemon

banner "3. a daemon holding a live SSE stream does not retire, then does"
printf '%s %s\n' "$OLD_SHA" "old-build" > "$PREFIX/.lop-source"
start_daemon held LOG_LEVEL=INFO LOP_BUILD_SETTLE_S=1
python3 "$HERE/drive.py" "$RECORD" "$PREFIX" "$NEW_SHA" \
  --hold-sse 14 --cwd "$WORK" | tee "$WORK/run3.txt"
echo "--- daemon log (tail) ---"
grep -E "retire|lop serve|in flight|SSE" "$WORK/held.log" || tail -8 "$WORK/held.log"
echo "--- record after exit ---"
ls -l "$CFG/run/serve/" || true
stop_daemon

banner "full daemon logs"
for tag in unsupervised governed held; do
  echo "--- $tag.log ---"
  cat "$WORK/$tag.log"
done

banner "raw artifacts under $WORK"
ls -l "$WORK"
