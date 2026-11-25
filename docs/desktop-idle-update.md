# Private desktop idle-update control (v1)

This is a **core daemon** protocol, not an update policy or install command. It
is not exposed over HTTP, mesh, registry, or the renderer. Other clients cannot
obtain it by adopting a daemon, knowing its PID, or getting the desktop bearer.

## Launch and bind

- Opt in only for a fresh, directly spawned, managed `lop serve` child on macOS
  or Linux. Do not opt in for reused, attached, tunnelled, externally managed,
  autoreload, or inherited-listener daemons. Check the pinned launch interpreter
  for `local_operator.server.desktop_idle_update` (e.g. `importlib.util.find_spec`)
  before adding `--desktop-idle-update-fd=3`; an older parser rejects it.
- Node's `spawn(..., {stdio: ["pipe", "pipe", "pipe", "pipe"]})` creates a private
  duplex pair at `child.stdio[3]`. Leave fd 3 open for the lifetime of this child;
  do not forward it to any other process. Pass the normal desktop spawn bearer
  through stdin with `LOCAL_OPERATOR_DESKTOP_TOKEN_FD=0` (newline-terminate it,
  then end stdin); omit the legacy bearer environment variable. The legacy
  environment path re-execs, and control is deliberately not handed over.
- The child verifies unnamed connected Unix stream and kernel peer = direct
  parent, then sets close-on-exec before any login-shell or runtime child. There
  is no named socket or second connection. Startup branding must be chosen by
  the launcher if needed, not by daemon re-exec.
- First frame after daemon startup is `desktop-idle-update-v1 <pid> <instance_id>\n`.
  Match both the original, still-current `ChildProcess.pid` and `/health`'s
  `instance_id` from the directly connected, owned backend. Treat missing or
  unrecognized hello, any mismatch, EOF, re-exec, and owner death as unavailable.
  Do not use a stale channel or add a fallback signal.

## Atomic stop

On that same channel, send `stop <instance_id>\n`; keep at most one outstanding
request. The daemon briefly parks HTTP, scheduler, and hub admissions, validates
its owned work on the same event-loop turn, and either rolls back or commits
ordinary uvicorn shutdown. It gracefully ends only its two replayable desktop
feeds. Detached session runtimes, passive attachments, and the separate network
relay are not stopped by this operation; it does not authorize updating or
restarting any of them.

- `accepted <instance_id>\n`: committed, not yet exited. Only continue the
  update after the same `ChildProcess` exits normally with code 0. Do not wait
  on a PID or port sampled from elsewhere. Exit can reach the app before its
  receipt parser runs: collect the two observations independently and proceed
  only after both. Never retry `accepted`; that channel is consumed. A lost or
  partial receipt, abnormal exit, or timeout is not authorization to install
  or force-stop. The installer must retain the previous generation read-only
  while an independent runtime or relay still uses it.
- `deferred <reason>\n`: no stop, gates have rolled back. Retry only after a
  fresh app policy check (idle, update still pending and exact same launch).
  No reservation, release message, or expiry timer exists. The reasons are
  diagnostic, not permission to fall back.
- `refused identity\n`, invalid reply, or channel closed before a full accepted
  receipt: fail closed. The server never turns an incomplete request into a stop.

This does not change normal app quit or manual updates. Policy (overnight window,
human inactivity, and ensuring explicit Quit stays closed) belongs in desktop
main and must not be inferred from this daemon's owned-work inventory.
