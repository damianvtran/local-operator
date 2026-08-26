set -e
SOCK=lopmuxtest
tmux -L $SOCK kill-server 2>/dev/null || true
tmux -L $SOCK new-session -d -s ev -x 100 -y 30
PANE=$(tmux -L $SOCK list-panes -t ev -F '#{pane_id}' | head -1)
echo "pane=$PANE"
tmux -L $SOCK send-keys -t "$PANE" "echo ready" Enter
sleep 0.3

# Run the REAL backend inside the tmux pane's environment.
tmux -L $SOCK run-shell -t "$PANE" "true" 2>/dev/null || true
TMUXENV=$(tmux -L $SOCK show-environment -t ev 2>/dev/null | head -1 || true)

/tmp/lop-mux-resume/.venv/bin/python - "$SOCK" "$PANE" <<'PY'
import os, subprocess, sys
sys.path.insert(0,"/tmp/lop-mux-resume")
sock, pane = sys.argv[1], sys.argv[2]
from local_operator.multiplexer.markers import TmuxBackend, SESSION_OPTION, COMMAND_OPTION
from local_operator.multiplexer.broadcast import build_binding
from local_operator.multiplexer.registry import active_backend
import local_operator.multiplexer.markers as m

# tmux backend shells out to `tmux`; point it at our private socket so this
# cannot touch any real tmux server.
real_run = m._run
m._run = lambda argv: real_run([argv[0], "-L", sock] + argv[1:])

env = {"TMUX": f"/tmp/tmux-{os.getuid()}/{sock},1,0", "TMUX_PANE": pane}
print("detected backend:", type(active_backend(env)).__name__)
b = build_binding("abc123abc123", cwd="/work")
print("publish ->", TmuxBackend().publish(b, env))
for opt in (SESSION_OPTION, COMMAND_OPTION):
    out = subprocess.run(["tmux","-L",sock,"show-options","-pv","-t",pane,opt],
                         capture_output=True, text=True)
    print(f"  {opt} = {out.stdout.strip()!r}")
print("retire ->", TmuxBackend().retire(b, env))
for opt in (SESSION_OPTION, COMMAND_OPTION):
    out = subprocess.run(["tmux","-L",sock,"show-options","-pv","-t",pane,opt],
                         capture_output=True, text=True)
    print(f"  {opt} after retire = {out.stdout.strip()!r} (rc={out.returncode})")
PY
tmux -L $SOCK kill-server 2>/dev/null || true
echo "tmux server torn down"
