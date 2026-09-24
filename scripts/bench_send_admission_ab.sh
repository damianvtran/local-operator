#!/usr/bin/env bash
# Interleaved A/B driver for scripts/bench_send_admission.py.
#
# WHY IT EXISTS. On this host, wall time under fleet load swings more between
# consecutive minutes than between two builds (AGENTS.md "Timing, flakes"), so
# the two arms must alternate run by run on the same machine, never run as two
# blocks. Review round 1 (F3) also showed how easy it is to label an arm wrongly:
# a "base" arm that imported the head tree measured head against head. So this
# driver runs ONE copy of the rig (this tree's) against each arm's source via
# BENCH_SEND_TREE, and then REFUSES a result whose recorded import is not the
# arm it claims to be.
#
# Usage (each arm is a checkout with the product code to measure; both share the
# venv given, which must hold the dependencies of both):
#   scripts/bench_send_admission_ab.sh <base-tree> <fix-tree> <out-dir> [reps]
# Env: CONDITIONS (default "idle lanes roster"), PROBES (default 6),
#      BURN_REPS (default 0: extra "roster --burn 6" reps), PYTHON (default
#      <fix-tree>/.venv/bin/python), RUN_BOUND_S (default 900).
#
# Isolation: every run gets a fresh HOME and LOCAL_OPERATOR_CONFIG_DIR under
# $TMPDIR (session-unique by mktemp), is started with `env -i`, so no CMUX_*,
# LOP_* or XPC_FLAGS is inherited, and the root is removed afterwards. The rig
# gates notifications in its own child environment and reaps its own children.
set -euo pipefail

base=${1:?base tree}
fix=${2:?fix tree}
out=${3:?output directory}
reps=${4:-2}
conditions=${CONDITIONS:-idle lanes roster}
probes=${PROBES:-6}
burn_reps=${BURN_REPS:-0}
here=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-$fix/.venv/bin/python}
bound=${RUN_BOUND_S:-900}
mkdir -p "$out"

run() { # arm tree label args...
  local arm=$1 tree=$2 label=$3
  shift 3
  local iso json rc
  iso=$(mktemp -d -t bench-send-ab)
  json="$out/$label-$arm.json"
  rc=0
  env -i HOME="$iso" LOCAL_OPERATOR_CONFIG_DIR="$iso/.local-operator" PATH="$PATH" \
    TERM=xterm-256color BENCH_SEND_TREE="$tree" \
    "$python" "$here/scripts/run_bounded.py" --timeout "$bound" -- \
    "$python" "$here/scripts/bench_send_admission.py" "$@" --json "$json" \
    >"$out/$label-$arm.log" 2>&1 || rc=$?
  rm -rf "$iso"
  if [ -f "$json" ]; then
    # The arm is what was IMPORTED, not what the label says.
    "$python" - "$json" "$tree" <<'EOF'
import json, sys
from pathlib import Path
report, tree = json.load(open(sys.argv[1])), Path(sys.argv[2]).resolve()
module = Path(report["imported"]["module"]).resolve()
if tree not in module.parents:
    sys.exit(f"MISLABELLED ARM: {sys.argv[1]} imported {module}, not {tree}")
EOF
  fi
  echo "$label $arm rc=$rc load=$(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)"
}

# Arithmetic loops, not `seq`: BSD `seq 1 0` counts DOWN ("1 0"), so a zero
# rep count would still run two reps on macOS.
for ((rep = 1; rep <= reps; rep++)); do
  for cond in $conditions; do
    run base "$base" "$cond-$rep" --condition "$cond" --probes "$probes"
    run fix "$fix" "$cond-$rep" --condition "$cond" --probes "$probes"
  done
done
for ((rep = 1; rep <= burn_reps; rep++)); do
  run base "$base" "roster+burn6-$rep" --condition roster --probes "$probes" --burn 6
  run fix "$fix" "roster+burn6-$rep" --condition roster --probes "$probes" --burn 6
done
