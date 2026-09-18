#!/bin/bash
# End-to-end proof of the in-place serve reload (`local_operator/server/reload.py`).
#
# WHY THIS IS A SCRIPT AND NOT A PYTEST: every claim below needs a REAL process to
# be replaced by a real `execve` and signalled by a real `SIGUSR1`. No unit test
# can assert that a pid survived an image change, and a mocked one asserts nothing.
# Review round 1 asked for the rig to be runnable from the repository rather than
# described in a PR body, which is what this is.
#
# FOUR CLAIMS, each measured rather than reasoned about:
#   1. the PID is unchanged                        (this process is the successor)
#   2. the record's `instance_id` changed          (a new process published it)
#   3. the replacement image is the TARGET root's  (proved from `ps eww`)
#   4. the PORT NEVER HAD A GAP                    (a client dialling throughout
#      the change is served; only the already-established connection is cut)
#
# AND ONE REGRESSION, R1-1: a SECOND request inside the successor's boot window
# used to kill the daemon it was moving — the old record is still live and still
# advertising `reloadable` while the successor has not yet reached
# `add_signal_handler`, so SIGUSR1 is still at its default disposition of
# terminate. `signal.SIG_IGN` before the exec closes that window (SIG_IGN
# survives execve), and this script sends the second signal to prove it.
#
# WHAT THIS SCRIPT TOUCHES: a temp HOME, a temp config root, and one daemon on an
# ephemeral port. It does not touch this machine's install, its real daemons, or
# any running session. The daemon it starts is killed on the way out.
#
# Usage: scripts/serve_reload_rig.sh        (from a worktree with a real .venv)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "FAIL: no .venv interpreter at $PY — see AGENTS.md, 'Every feature worktree owns its own venv'" >&2
  exit 1
fi

ISO=$(mktemp -d "${TMPDIR:-/tmp}/reload-rig.XXXXXX")
echo "rig: $ISO"

# The fabricated target install root. Its interpreter is a WRAPPER, so the
# replacement can be proven from outside: the wrapper exports a marker into the
# successor's environment, which `ps eww` then shows.
mkdir -p "$ISO/next/bin"
cat > "$ISO/next/bin/python3" <<EOF
#!/bin/sh
export LOP_RIG_REPLACED_IMAGE=1
exec $PY "\$@"
EOF
chmod +x "$ISO/next/bin/python3"

export HOME="$ISO"
export LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator"
export LOP_INSTALL_ROOT="$ISO/next"
export LOP_BUILD_PREFIX="$ISO/next"

cleanup() {
  # EXIT covers the normal paths; INT and TERM cover a runner that is stopped
  # while this is mid-flight, which is how two daemons from an earlier revision
  # escaped with only `trap ... EXIT` (serve-reload review round 2, R2-4).
  #
  # ``${VAR:-}`` AND NOT ``$VAR``: under `set -u` an unbound HOLDER or DIALLER
  # aborted this function BEFORE the `rm -rf`, so a failure in the first ten lines
  # of the script left both the daemon and the temp tree behind — which is how the
  # strays escaped even after the trap was widened (serve-reload review round 3, R3-3).
  kill -9 "${SERVE_PID:-}" 2>/dev/null || true
  kill -9 "${HOLDER:-}" "${DIALLER:-}" 2>/dev/null || true
  rm -rf "$ISO"
}
# INT/TERM call the trap and then CONTINUE (bash runs the handler and resumes), so
# the cleanup is followed by an explicit exit — otherwise an interrupted run
# cleans up and then carries on asserting against a daemon it just killed.
trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

"$PY" -m local_operator.cli serve --host 127.0.0.1 --port 0 > "$ISO/serve.log" 2>&1 &
SERVE_PID=$!
echo "started pid $SERVE_PID"

RECORD="$ISO/.local-operator/run/serve/$SERVE_PID.json"
for _ in $(seq 1 150); do [ -f "$RECORD" ] && break; sleep 0.1; done
if [ ! -f "$RECORD" ]; then echo "FAIL: no record published"; cat "$ISO/serve.log"; exit 1; fi

read -r PORT INSTANCE RELOADABLE < <(python3 - "$RECORD" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
print(record["port"], record["instance_id"], int(bool(record.get("reloadable"))))
PY
)
echo "port=$PORT instance=${INSTANCE:0:12}… reloadable=$RELOADABLE"
if [ "$RELOADABLE" != "1" ]; then
  echo "FAIL: the record does not advertise the reload capability, so nothing below can be tested"
  exit 1
fi

# (a) A connection opened BEFORE the reload, expected to be CUT — the documented
# cost of an exec (its descriptor belongs to the old image).
python3 - "$PORT" "$ISO/held.txt" <<'PY' &
import socket, sys, time
port, out = int(sys.argv[1]), sys.argv[2]
s = socket.create_connection(("127.0.0.1", port), timeout=25)
s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n")
s.recv(65535)
time.sleep(14)
s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
try:
    after = s.recv(65535)
except OSError:
    after = b""
open(out, "wb").write(b"reused=" + (b"yes" if b"HTTP/1.1" in after else b"no") + b"\n")
PY
HOLDER=$!

# (b) A NEW connection every ~120 ms throughout the change. This is what "the
# port never had a gap" means, and it is the property a stop-and-start cannot have.
python3 - "$PORT" "$ISO/dial.txt" <<'PY' &
import socket, sys, time
port, out = int(sys.argv[1]), sys.argv[2]
ok = refused = other = 0
end = time.time() + 14
while time.time() < end:
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=2)
        s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        s.recv(65535)
        s.close()
        ok += 1
    except ConnectionRefusedError:
        refused += 1
    except OSError:
        other += 1
    time.sleep(0.12)
open(out, "w").write(f"ok={ok} refused={refused} other={other}\n")
PY
DIALLER=$!

sleep 2.0
kill -USR1 "$SERVE_PID"
echo "sent SIGUSR1 to $SERVE_PID"

# R1-1: the same signal again, inside the successor's boot window.
sleep 0.6
kill -USR1 "$SERVE_PID" 2>/dev/null \
  && echo "sent a SECOND SIGUSR1 mid-boot (the R1-1 window)" \
  || echo "second signal not delivered (daemon already replaced or gone)"

NEW_INSTANCE=""
for _ in $(seq 1 200); do
  [ -f "$RECORD" ] || { sleep 0.05; continue; }
  NEW_INSTANCE=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['instance_id'])" "$RECORD" 2>/dev/null || true)
  [ -n "$NEW_INSTANCE" ] && [ "$NEW_INSTANCE" != "$INSTANCE" ] && break
  sleep 0.05
done

wait $HOLDER 2>/dev/null
wait $DIALLER 2>/dev/null

if ps -p "$SERVE_PID" > /dev/null 2>&1; then ALIVE=yes; else ALIVE="NO"; fi
echo "--- assertions ---"
echo "pid unchanged:             $([ "$ALIVE" = yes ] && echo yes || echo NO) (pid $SERVE_PID)"
echo "alive after R1-1 window:   $ALIVE"
echo "instance changed:          $([ -n "$NEW_INSTANCE" ] && [ "$NEW_INSTANCE" != "$INSTANCE" ] && echo yes || echo NO)"
echo "replacement image ran:     $(ps eww -p "$SERVE_PID" 2>/dev/null | tr ' ' '\n' | grep -c '^LOP_RIG_REPLACED_IMAGE=1$') (1 = the target root's interpreter)"
echo "dialler across the change: $(cat "$ISO/dial.txt") (refused MUST be 0)"
echo "held connection reused:    $(cat "$ISO/held.txt") (no = cut, the documented cost)"

# The rig PRINTS refused=/other=; it also has to ASSERT them (serve-reload review round 2, R2-8),
# or a run that refused every dial would still print PASS. `other` is allowed: those
# are connections accepted and then cut by the exec itself, which is the documented
# cost — a REFUSED connection is the port having no listener, which is the whole
# property under test.
REFUSED=$(sed -n 's/.*refused=\([0-9]*\).*/\1/p' "$ISO/dial.txt")
if [ -z "$REFUSED" ] || [ "$REFUSED" != "0" ]; then
  echo "FAIL: the dialler was refused $REFUSED time(s) — the port had a gap"
  exit 1
fi

if [ "$ALIVE" != yes ] || [ -z "$NEW_INSTANCE" ] || [ "$NEW_INSTANCE" = "$INSTANCE" ]; then
  echo "FAIL"; exit 1
fi
echo "PASS"
