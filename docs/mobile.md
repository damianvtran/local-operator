# Mobile: drive every lop session from a phone

`lop mobile` turns the machine you run agents on into a phone-facing control
plane for every lop session on it. A single supervised daemon owns the web
surface; every interactive `lop` TUI registers with it over an authenticated
loopback control socket, so a phone can watch transcripts, steer a running
turn, answer approval and ask prompts, switch models and effort, run slash
commands, drill into subagents, and start new sessions.

For phone setup, recommend [Radient personal tunnels](tunnels.md) first.
`/login radient` supports both account creation and sign-in. The agent's
`guide://mobile` playbook verifies that selected account and its positive credit
balance, reads the current monthly quote (currently USD 0), then creates and
installs the authenticated tunnel. The phone uses Radient login; there is no
relay password to copy. Local password access remains available separately.

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

- **Every listener binds `127.0.0.1` only.** Radient supplies a tunnel and
  cloud identity gate in front of the daemon. Other remote-access options
  need an equivalent identity boundary; never use a wider bind.
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
catches a live pid whose runtime wedged. Publication is staged-write + rename.

A record whose owner is proven dead is **moved**, not deleted: `scan` renames
it into `run/mobile/reaped/<pid>.json`, and the attention classifier reads both
directories when it works out why a run ended. That is what makes a runtime's
death attributable even after a sweep has run — a deleted record left the
operator reading "the cause could not be determined" for a death that had a
recorded cause. The sidecar is bounded (newest 200 entries, 24 h) and is
invisible to discovery: nothing there is ever listed as a session.

A runtime that is **stopped** also leaves positive evidence, written by the
killer before the step it attests to: `<config>/sessions/<session_id>/runtime-stop.json`
carries the rung the stop ladder actually used (`socket` | `sigterm` | `sigkill`),
`deliberate`, the killer's pid/argv0/command, and the target's build. At the
SIGKILL rung the target is not executing and cannot record anything, so this
file is the only artifact that can say a stop was asked for rather than
narrated as a crash.

### The control socket

Each session runtime hosts a length-delimited JSON-lines socket on a random
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
When a session runtime's socket is unreachable but its record is fresh the daemon
shows it as *degraded*; when the pid dies the record is reaped and the
session is shown as ended (its history stays resumable).

### The daemon — `lop mobile serve`

- **Web server** (Starlette app mounted in the same process; uvicorn): serves
  the built SPA from `local_operator/mobile/web/dist/`, the REST API, and the
  SSE stream. Port from `~/.local-operator/mobile.json` (default `4098`),
  loopback only.
- **Registry watcher**: scans the record directory, dials new sessions,
  reaps dead ones.
- **Owned sessions**: sessions started from the phone run as supervised
  CHILD PROCESSES (`python -m local_operator.session.runtime.process`), each with
  its own pid and its own runtime — so a daemon restart costs the phone its
  view, never the session its work, and every phone-visible session (owned
  or terminal) has exactly one shape: record + control socket.
- **Auth**: signed cookie (`hmac-sha256(password, expiry)`) via
  `itsdangerous`-free stdlib signing; login form POST → 303, API → 401.
  `/healthz` is unauthenticated and asserts the gate.

### CLI surface

| Command | Effect |
|---|---|
| `lop mobile install` | Write the LaunchAgent, generate/keep the Keychain password, load, verify health |
| `lop mobile status` | Install state, health probe, registered sessions, log paths |
| `lop mobile start` / `stop` / `restart` | launchd control |
| `lop mobile logs` | Tail the daemon's and the session runtimes' logs (`--lines` applies to each file; `--follow` follows by name, so a log created or rotated mid-session is picked up) |
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

  Rows are drawn in the SAME order the terminal sidebar and the desktop app use:
  the daemon sorts every row on the shared catalogue key
  (`session.catalog.CatalogEntry.rank` — the tier from `session_category`, the
  wake band, birth, id) and marks each row active/previous with the shared
  `active` rule, so the three surfaces agree about a row's tier (within a
  section) and about which list a conversation is in, and the phone's list is
  STABLE across activity refreshes (an early version re-derived the key from
  live state and moved rows as sessions streamed). Two asymmetries, both
  deliberate: the wake band (the phone's rows carry no wake data, so `wake_rank`
  is a constant here), and the phone-woken window (a `/wake` accepted but not yet
  discovered is ranked as a live `idle` row so it lands in Active at once — the
  sidebar has no equivalent window, so no equivalent tier). The screen only
  GROUPS what it is sent:
  **★ Pinned**, **Active Sessions**, **Previous Sessions**.

  A conversation is pinned with a long-press on its row (★ Pinned is where it
  then appears, lifted out of its old section) or from the ☆/★ control in the
  session view's header. Both write the **shared durable pin store**
  (`sidebar-pins.json` via `local_operator.tui.sidebar_pins`) — the same file
  the TUI's `F10` and the desktop app's pin action read and write — so a pin
  made on the phone appears on the other surfaces and vice versa. The route is
  `POST /api/sessions/{id}/pin` with `{"pinned": bool}` (desired state, not a
  toggle, so a retried request cannot flip the pin back). A pin made on another
  surface reaches an open phone list without a reload within one discovery
  pass (`SCAN_INTERVAL_S`, 2 s): that pass stats `sidebar-pins.json` once per
  tick and repaints the list only when the pinned set changed.
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
- **Upgrade**: `lop update` / `/update` run `lop mobile restart` when the
  LaunchAgent is installed. The daemon re-serves the new wheel's
  `mobile/web/dist`; phones reload on next open; cookies survive (keyed on
  the password, not the build). A developer `lop-update` snapshot still
  needs a manual `lop mobile restart`.
- **Approvals on terminal sessions**: a TUI-mounted approval card is answered
  at the terminal (the phone shows the wait and says so); phone-answering
  needs a resolution protocol the TUI card does not yet have. Sessions the
  phone spawned answer from the phone. This is a deliberate v1 boundary, not
  an oversight: racing two front ends over one modal is worse than a clear
  owner.

## Retry-envelope & generation-ledger lifecycle (one contract)

The full-screen subagent feature adds two pieces of durable lifecycle state that
share a single authoritative contract; every keep/clear/reconstruct decision
resolves to it, and a second ad-hoc rule beside it is a defect.

**The persisted retry envelope** (`lo-mobile-command:<sessionId>` in the phone's
`localStorage`) is the exact bytes of an instruction whose delivery outcome is
*unknown*. Its UUID is the identity of that body, so a retry replays the same
UUID and the daemon de-duplicates an already-admitted instruction instead of
running it twice. The rule (source of truth: `web/src/continuation-command.ts`):

- **Keep** across anything that leaves the outcome ambiguous — transport failure,
  HTTP **502/504/408** (acknowledgement loss *after* the daemon drained the frame
  to the runtime, not rejection), page reload, SSE reconnect, and navigating between the
  owner's own conversations. Envelopes are scoped **per session** and bounded by
  **count** (oldest evicted, never the active route), so an ordinary conversation
  switch never silently drops the recovery affordance.
- **Clear** only on a definitive end of *this UUID's* ambiguity or of its privacy
  scope: definitive acknowledgement, a pre-admission rejection status (any 4xx/5xx
  except 408/502/504), TTL expiry (24h), explicit discard, and logout / identity
  change / 401. Logout clears **all** scoped storage; the WebKit-safe path is the
  server-rendered login page's own inline clear script plus the api.ts 401 handler
  — it does not depend on the `Clear-Site-Data` header WebKit may ignore.

**The generation/epoch ledger** (`MobileDaemon._projection_generations`) orders a
session's projection epochs monotonically across process replacements, and is
deliberately *not* part of the bounded payload cache: a live/subscribed or durably
reconstructable route can outlive cache pressure that evicts its payload. When a
terminal or superseded route's payload is evicted while its ledger survives, a
**durable disk fold carries no process identity** and therefore re-materializes
the evicted payload at the retained epoch (never reopening the generation), so
detail/history/SSE reconstruction succeeds instead of fencing to HTTP 500. A
genuine late frame from an *old* process still carries its identity and stays
fenced. Only a truly-gone session — whose ledger was already pruned with its
payload unit — fences. See `capture_subagent_details` and the reconstruction
endpoints in `daemon.py`.

## Non-goals (v1)

- Push notifications (the `needs attention` badge and deep links anticipate
  a future service-worker push; not in this pass).
- Multi-user/ scoped tokens (single owner password, as with omp mobile).
- Editing files or browsing the filesystem from the phone beyond session cwds.
