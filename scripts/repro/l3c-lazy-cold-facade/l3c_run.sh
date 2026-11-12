#!/bin/bash
# Real-path evidence for the L3c deferred cold facade.
#
# Two worktrees, ONE synthetic fixture (APFS clones of one seed), the same code
# path from the outside: a desktop cold open of exactly that journal.
#   * in-process: backend CPU and journal bytes per open. ONE process per side,
#     replay cache cleared between samples -- NOT a fresh process, so its hit
#     figure is a facade-open cost and must not be quoted as a restart cost
#   * real `serve`: the real /events route over HTTP, first frame captured
# Nothing reads the operator's store: every root is a temp dir, every process is
# reaped by exact pid.
set -u

OUT="${1:?usage: l3c_run.sh <output-dir>}"
# The scripts beside this one, and the two checkouts to compare. Set
# L3C_BRANCH / L3C_BASE to the branch worktree and a pristine origin/main
# worktree (each with its own `.venv`).
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="$HERE"
BRANCH="${L3C_BRANCH:?set L3C_BRANCH to the branch worktree}"
BASE="${L3C_BASE:?set L3C_BASE to a pristine origin/main worktree}"
SID=c0ffee123456
TOKEN=l3c-synthetic-token
mkdir -p "$OUT"

ISO=$(mktemp -d)
SEED=$(mktemp -d)
ROOT_BASE=$(mktemp -d)
ROOT_BRANCH=$(mktemp -d)

echo "== fixture (one seed, cloned per side)"
env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$SEED" PATH="$PATH" L3C_WORKTREE="$BRANCH" \
  "$BRANCH/.venv/bin/python" "$SCRATCH/l3c_measure.py" seed "$SEED"
cp -Rc "$SEED/." "$ROOT_BASE/"
cp -Rc "$SEED/." "$ROOT_BRANCH/"

inproc() {
  local worktree=$1 root=$2
  env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$root" PATH="$PATH" TERM=xterm-256color \
    L3C_WORKTREE="$worktree" "$worktree/.venv/bin/python" \
    "$SCRATCH/l3c_measure.py" inproc "$root" 4 > "$OUT/inproc.json" 2>"$OUT/inproc.err"
}

show() {
  env -i PATH="$PATH" python3 - "$OUT/inproc.json" "$1" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
for index, sample in enumerate(data["samples"]):
    print(f"  sample {index}: cpu={sample['cpu_ms']:>9.2f} ms  "
          f"journal passes={sample['journal_passes']}  bytes={sample['journal_bytes']}")
PY
}

echo "=== in-process cold opens: origin/main (baseline)"
inproc "$BASE" "$ROOT_BASE"; show baseline; cp "$OUT/inproc.json" "$OUT/inproc-base.json"

echo "=== in-process cold opens: feat/l3c-lazy-cold-facade"
inproc "$BRANCH" "$ROOT_BRANCH"; show branch; cp "$OUT/inproc.json" "$OUT/inproc-branch.json"

serve_once() {
  local worktree=$1 root=$2 port=$3 label=$4
  local log="$OUT/serve-$label.log"
  env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$root" PATH="$PATH" TERM=xterm-256color \
    LOCAL_OPERATOR_DESKTOP_TOKEN="$TOKEN" \
    "$worktree/.venv/bin/local-operator" serve --port "$port" >"$log" 2>&1 &
  local pid=$!
  local waited=0
  until curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    sleep 0.25; waited=$((waited + 1))
    if [ "$waited" -gt 200 ]; then
      echo "  serve ($label) never became healthy; see $log" >&2
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 1
    fi
  done
  sleep 1
  # TWENTY SNAPSHOT REQUESTS, because macOS reports a process's CPU in
  # centiseconds-to-seconds: one request is below the resolution, twenty are not.
  # TWO ROUNDS OF TWENTY. The first pays whatever the process has not done yet
  # (a parse, a preheat that is already warm in memory); the second is the
  # steady state. Reporting both is how the two costs stay separable instead of
  # being averaged into one number that answers neither question.
  for round in 1 2; do
      cpu_before=$(ps -o cputime= -p "$pid" | tr -d ' ')
    for _ in $(seq 1 20); do
      curl -sf -m 10 -H "Authorization: Bearer $TOKEN" \
        "http://127.0.0.1:$port/v1/desktop/sessions/$SID" >/dev/null 2>&1
    done
    cpu_after=$(ps -o cputime= -p "$pid" | tr -d ' ')
    python3 "$SCRATCH/l3c_cpu.py" "round $round:" "$cpu_before" "$cpu_after"
  done
  if [ -f "$root/cache/cold_replay/$SID.json" ]; then
    echo "  durable metadata record: $(stat -f %z "$root/cache/cold_replay/$SID.json") bytes (present)"
  else
    echo "  durable metadata record: absent"
  fi
  local start end
  start=$(python3 -c 'import time;print(time.time())')
  curl -sN -m 20 -H "Authorization: Bearer $TOKEN" \
    "http://127.0.0.1:$port/v1/desktop/sessions/$SID/events" \
    > "$OUT/frames-$label.txt" 2>/dev/null
  end=$(python3 -c 'import time;print(time.time())')
  python3 -c "print(f'  stream wall: {float(\"$end\") - float(\"$start\"):.3f} s (bounded by the -m 20 cut, not a latency)')"
  python3 "$SCRATCH/l3c_frames.py" extract "$OUT/frames-$label.txt" "$OUT/frame-$label.json"
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  echo "  serve ($label) reaped: pid $pid"
}

echo "=== real serve, real /events"
serve_once "$BASE" "$ROOT_BASE" 8731 base
serve_once "$BRANCH" "$ROOT_BRANCH" 8732 branch

echo "=== wire equality (canonical first frame: origin/main vs branch)"
python3 "$SCRATCH/l3c_frames.py" compare "$OUT/frame-base.json" "$OUT/frame-branch.json"
echo "exit=$?"

echo "== leftover l3c serve processes (must be none)"
pgrep -fl "local-operator serve --port 873" || echo "  none"
