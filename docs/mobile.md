# Mobile: drive every lop session from a phone

`lop mobile` turns the machine you run agents on into a phone-facing control
plane for every lop session on it. A single supervised daemon owns the web
surface; every interactive `lop` TUI registers with it over an authenticated
loopback control socket, so a phone can watch transcripts, steer a running
turn, answer approval and ask prompts, switch models and effort, run slash
commands, drill into subagents, and start new sessions.

It descends from omp's mobile relay (`omp mobile`), with these deliberate
differences:

1. **One daemon, not two.** omp needed a content-blind relay because its
   sessions spoke an end-to-end-encrypted collab-room protocol the portal had
   to join as a guest. lop has no room protocol — the integration seam is the
   in-process `Session` object — so a relay process would add a port and a
   failure mode without buying anything. The daemon *is* the hub.
2. **TUI sessions are first-class, not invisible.** omp's honest gap was that
   a running terminal session hosted nothing the portal could see. Here every
   interactive `lop` instance registers itself automatically (publish record +
   control socket), so the phone sees and drives terminal sessions too.
3. **Model/effort/slash commands are in the control vocabulary from day one**,
   along with subagent drill-down and resume.

## Security invariants

- **Every listener binds `127.0.0.1` only.** Remote access is something you
  put in front (a tunnel plus an identity proxy, exactly as with omp mobile);
  never a wider bind.
- **The phone leg is HTTP + SSE, never WebSocket.** An identity proxy answers
  an unauthenticated request with a redirect to its login page, and a browser
  cannot follow a redirect on a WebSocket handshake. Every state push is a
  snapshot/repaint, not a delta — no delta protocol to drift.
- **The discovery record is the only key distribution.** A session's control
  key lives only in its `0600` record under a `0700` directory. Anything that
  can read the record is already the owning account; the daemon never learns
  a key it cannot read itself.
- **The password is never on a command line or in a log.** It comes from the
  Keychain (`security -i` over stdin), `LOP_MOBILE_PASSWORD` for containers,
  or the interactive `lop mobile password` prompt. Cookies are signed with a
  key derived from the password, so rotation invalidates every session for
  free.

## Components

### Discovery records — `~/.local-operator/run/mobile/<pid>.json`

Every `lop` process that hosts a mobile-reachable session (TUI, `exec`) plus
the daemon itself publishes a record at startup and removes it at exit:

```json
{
  "pid": 4242,
  "kind": "tui",                  // tui | exec | daemon
  "session_id": "…",
  "conversation_name": "…",
  "cwd": "/Users/damian/work",
  "model": "anthropic/claude-opus-5",
  "control_port": 52711,          // loopback JSON-lines socket
  "control_key": "…",             // 32 random bytes, hex
  "started_at": "2026-08-19T…",
  "heartbeat_at": "2026-08-19T…"  // rewritten every 15 s
}
```

The daemon scans this directory every 2 s and validates each record by pid
liveness — a SIGKILLed session leaves its record behind, and the heartbeat
catches a live pid whose owner wedged. Publication is staged-write + rename.

### The control socket

Each registrant hosts a length-delimited JSON-lines socket on a random
loopback port. The daemon dials it with the record's key. Auth is a single
`hello` frame carrying the key (constant-time compare); the session answers
with a `welcome` snapshot and then streams events. One connection, both
directions:

- **daemon → session**: `prompt`, `steer`, `abort`, `set_model`,
  `set_effort`, `slash` (execute a slash command), `resume_session`,
  `new_conversation`, `approval_answer`, `ask_answer`, `snapshot`
- **session → daemon**: `welcome`, `delta` (assistant streaming),
  `tool.start/end`, `turn.start/end`, `notice`, `todos`, `subagents`,
  `approval_request`, `ask_request`, `state` (model, effort, cwd, name,
  streaming flag)

The daemon folds each session's stream into a bounded projection (transcript
tail, todos, subagent roster, pending requests) and re-serves it to phones.
When a registrant's socket is unreachable but its record is fresh the daemon
shows it as *degraded*; when the pid dies the record is reaped and the
session is shown as ended (its history stays resumable).

### The daemon — `lop mobile serve`

- **Web server** (Starlette app mounted in the same process; uvicorn): serves
  the built SPA from `local_operator/mobile/web/dist/`, the REST API, and the
  SSE stream. Port from `~/.local-operator/mobile.json` (default `4098`),
  loopback only.
- **Registry watcher**: scans the record directory, dials new sessions,
  reaps dead ones.
- **Owned sessions**: sessions started from the phone are built in-process
  with `session_factory.create_session` and driven directly (no socket hop).
  The daemon registers them through the same code path so every phone-visible
  session, owned or remote, has one shape.
- **Auth**: signed cookie (`hmac-sha256(password, expiry)`) via
  `itsdangerous`-free stdlib signing; login form POST → 303, API → 401.
  `/healthz` is unauthenticated and asserts the gate.

### CLI surface

| Command | Effect |
|---|---|
| `lop mobile install` | Write the LaunchAgent, generate/keep the Keychain password, load, verify health |
| `lop mobile status` | Install state, health probe, registered sessions, log paths |
| `lop mobile start` / `stop` / `restart` | launchd control |
| `lop mobile logs` | Tail the daemon log (`--lines`, `--follow`) |
| `lop mobile password` | Set or rotate the password (interactive prompt; restarts the daemon) |
| `lop mobile uninstall` | Unload and delete the LaunchAgent (`--purge` also deletes the password) |
| `lop mobile serve` | Run the daemon in the foreground (what the LaunchAgent runs) |

Every action takes `--json`. Registration from the TUI/exec side needs no
install step: publishing the record is unconditional and free; without a
daemon listening the control socket just never gets dialed (bounded listen
backlog, no threads).

### The web UI — `local_operator/mobile/web/`

Vite + React + TypeScript + Tailwind v4 + shadcn-style primitives, built to
hashed static assets the daemon serves; nothing Node-related runs at
runtime. Theme roles come from the local-operator-ui palette contract (the
TUI's own dark/light brand themes included), mapped onto Tailwind `@theme`
roles (`bg-surface`, `text-ink-muted`, `border-control`, `text-accent`, …) —
never a raw hex in a component.

Screens, following branding.md §7's agent-output hierarchy:

- **Login** — minimal, brand mark, password field, 16 px inputs.
- **Session list** — one card per session: name, cwd, model label, streaming
  shimmer, needs-attention badge (approval/ask pending), running-subagent
  chip. New-session button with a cwd picker (home + recents).
- **Session view** — transcript with TUI-parity rendering: user rows,
  assistant markdown, one-line tool calls with state glyphs and green/red
  diff counts, tap to expand/collapse args+output+diff; todos panel;
  subagents panel with tap-to-drill into a subagent's transcript and a
  back-to-parent crumb; approval/ask cards pinned above the composer.
- **Composer** — the TUI composer, mobilized: multiline auto-growing field,
  model label + effort as tappable chips (opens the model sheet / effort
  rungs), typing `/` opens the slash-command sheet with fuzzy filtering and
  argument hints, send/steer/stop button morphing with turn state, resume
  affordance after an abort.

## Failure modes and rules

- **Daemon down, TUI up**: the TUI is unaffected; the record sits unpublished
  until the daemon returns and adopts it. Registration retries are cheap
  (one dial attempt per scan tick, backoff on refused).
- **TUI dies**: pid check + heartbeat reap it within one scan; the phone card
  flips to *ended*, offering resume.
- **Two daemons**: the LaunchAgent label owns the port; a second `serve`
  fails to bind and exits loudly. No split-brain by construction.
- **Upgrade**: `lop mobile restart` after `lop-update`; the daemon re-serves
  the new bundle, phones reload on next open, cookies survive (keyed on the
  password, not the build).

## Non-goals (v1)

- Push notifications (the `needs attention` badge and deep links anticipate
  a future service-worker push; not in this pass).
- Multi-user/ scoped tokens (single owner password, as with omp mobile).
- Editing files or browsing the filesystem from the phone beyond session cwds.
