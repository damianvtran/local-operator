#!/bin/bash
# Restore the pristine seeded config, then start stub hub + real `lop serve` as group leaders.
set -e
S=/Users/damian/.local-operator/sessions/3d299fb1b279/scratchpad/hb
ISO=$S/iso; WT=/Users/damian/local-operator/.worktrees/bridge-inbound-body
HUBP=$(cat $S/hubport); SP=$(cat $S/serveport)
LOG=${1:-$S/hub.log}
rm -rf "$ISO/.local-operator"; cp -R "$S/pristine-cfg" "$ISO/.local-operator"; mkdir -p "$ISO/tmp"
: > "$LOG"
set -m
cd "$ISO"
( exec env -i PATH=/usr/bin:/bin /usr/bin/python3 $S/stub_hub.py $HUBP "$LOG" ) > $S/hub.out 2>&1 &
echo $! > $S/hub.pid
( exec env -i HOME=$ISO LOCAL_OPERATOR_CONFIG_DIR=$ISO/.local-operator PATH=$WT/.venv/bin:/usr/bin:/bin TMPDIR=$ISO/tmp \
  GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0 RADIENT_API_BASE_URL=http://127.0.0.1:$HUBP RADIENT_ORG_ALLOW_NONCANONICAL_BASE=1 \
  LOCAL_OPERATOR_DESKTOP_TOKEN=$(cat $S/token) $WT/.venv/bin/lop serve --port $SP ) > $S/serve.out 2>&1 &
echo $! > $S/serve.pid
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:$SP/health && break; sleep 1; done
echo "hub $(cat $S/hub.pid) serve $(cat $S/serve.pid) health $(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:$SP/health)"
