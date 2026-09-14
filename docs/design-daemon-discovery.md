# Design: daemon discovery, attach, and update alignment between local-operator and local-operator-ui

Status: proposal (architect). Scope: two repos, three backend PRs and one UI PR.
No `pyproject.toml` version bump — the release owner handles that.

## 1. The problem as found in the code

`BackendServiceManager` decides whether a backend exists by asking **one fixed
URL** for `/health`:

- `local-operator-ui/src/main/backend/backend-service.ts:494-595`
  (`checkExistingBackend`) probes `backendConfig.VITE_LOCAL_OPERATOR_API_URL`
  with a 5 s timeout, then retries **one hardcoded** `http://localhost:1111`
  (:539). The only test is HTTP 200 — there is no identity check — so any
  listener answering `/health` is "the Local Operator backend".
- The port is a build-time constant (`src/main/backend/config.ts:31-37`; the URL
  is assembled at `backend-service.ts:154-177`).
- With `VITE_DISABLE_BACKEND_MANAGER=true` (the state of `local-operator-ui/.env`
  today) discovery does not run **at all**: `checkExistingBackend()` returns true
  before probing (`backend-service.ts:495-501`) and `src/main/index.ts:650-653`
  skips the whole startup block, so `start()` is never called, `startupMode`
  stays `NOT_STARTED`, and the update check bails at
  `src/main/update-service.ts:2054-2062` ("nothing to check or update"). That is
  why the app can neither find nor authenticate against a daemon.
- Liveness is a **single** failed probe: `startHealthCheck()`
  (`backend-service.ts:1164-1191`) calls `start()` on one 30 s tick's failure,
  however busy or mid-restart the daemon is.
- Ownership is by `pkill`: `backend-service.ts:987,1013,1089,1097`,
  `src/main/index.ts:934,943,1015,1062,1068` (including `pkill -9 -f python`),
  and `update-service.ts:1096-1101`. Those patterns match the operator's global
  daemon (`Local Operator [serve] port=7341`, `procname.py:172`) — the mechanism
  behind "the UI killed my daemon".

Measured on this machine (2026-09-13): three `serve` daemons live at once, all
invisible to each other — `:1111` v0.54.27 (the app's bundled venv under
`~/Library/Application Support/Local Operator/local-operator-venv`), `:7341`
v0.54.32 (the global uv tool, PPID 1, its TUI gone), `:8080` v0.44.66 (a stale
dev server from `~/local-operator/.venv`). Each answers `/health`; each satisfies
`checkExistingBackend()`. The version dialog is that divergence: `lop --version`
is 0.54.32, the bundled venv 0.54.27.

The desktop plane is the other half. `server/desktop.py:28-59`
(`require_desktop`) gates every desktop router on
`LOCAL_OPERATOR_DESKTOP_TOKEN` **in the daemon's environment**; a daemon a TUI
started has none, so `/v1/desktop/*`, `/v1/auth/*`, `/v1/settings*` and `/v1/mcp`
answer `503 "Desktop controls require a backend started by the desktop app."`
(verified against `:7341`). `managed_desktop_boundary` (`server/app.py:318-334`)
then leaves the legacy control paths *ungated* for that same daemon, because the
predicate it tests is the same environment variable.

The primitives already exist:

- `session/runtime/registry.py:36-42` `run_dir()` (created 0700), `:45-58`
  `record_path` (`<pid>.json`), `:61-79` `publish()` (staged `mkstemp` +
  `os.replace`, `chmod 0600`), `:82-88` `unpublish()`, `:91-128` `pid_alive()`
  (zombie-aware, opt-in), `:131-172` `scan()` (live / wedged / stale, reaps
  stale), `:175-210` `RecordPublisher` (publish + heartbeat + close, pins root).
- `session/runtime/types.py:206` `RUN_DIRNAME = "run/mobile"`, `:210-211`
  `HEARTBEAT_INTERVAL_S = 15` / `HEARTBEAT_TIMEOUT_S = 45`; `SessionRecord`
  already carries `version` / `source_ref` (`:308-312`).
- `server/app.py:73-152` — the daemon has a `lifespan`, i.e. a start and a stop
  hook, and nothing publishes from it today.
- `update.py:111-146` `BuildStamp`, `:149-190` `installed_version()`, `:804-820`
  `installed_build()`, `:864+` `build_marker_age_s()`, `:1023-1095`
  `perform_upgrade()` (installer first, `.lop-source` written **last** — its
  mtime is the settle signal), `:1142-1184` `refresh_mobile_after_upgrade()`;
  and the retirement machinery in `session/runtime/process.py:84,93,101`
  (`BUILD_CHECK_S = 5`, `BUILD_SETTLE_S = 10`, `BUILD_STAGGER_S = 20`) with
  `_build_changed` (`:191-215`) and `_refresh_for` (`:397-470`, `begin_retire`
  latch). The TUI's `/update` is `tui/app.py:12118-12265`.

## 2. The rendezvous record for the `serve` daemon

**A new namespace, not the session one:** `run/serve/<pid>.json` **under the
config root** (`paths.config_dir()` — `LOCAL_OPERATOR_CONFIG_DIR` when it is set,
else `~/.local-operator`), owned by a new module
`local_operator/server/registry.py` (stdlib-only, import-light). Nothing in this
design ever means the checkout at `~/local-operator`: that is a source tree, and
a reader that globs inside it finds nothing on a real install.

Do **not** add the daemon to `run/mobile`. `types.py:~190-205` calls the dirname
a wire constant precisely because an upgrade window has two binaries scanning it;
worse, every reader of `scan()` (`lop sessions`, the phone daemon,
`find_runtime_record` at `mobile/attach_client.py:684-724`) treats each record
there as a **session**, and `kind` is a `Literal` old readers pass through
unvalidated — a daemon record would surface as a phantom session with an empty
`session_id`, silently.

```python
# local_operator/server/registry.py
SERVE_RUN_DIRNAME = "run/serve"

@dataclass
class ServeRecord:
    pid: int                    # os.getpid() — the serving process
    port: int                   # the port actually bound (see §7 on --port 0)
    host: str                   # "127.0.0.1"
    instance_id: str            # secrets.token_urlsafe(32), minted at startup
    version: str                # update.installed_version()
    source_ref: str             # update.source_ref()
    prefix: str                 # sys.prefix — WHICH install this daemon runs
    install_kind: str           # update.install_kind(): uv-tool|pipx|pip|editable|unknown
    desktop: bool               # is the desktop plane live right now
    claim_key: str = ""         # §4; "" when an env token already governs
    started_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)
```

Reuse, in `session/runtime/registry.py`: `run_dir`/`record_path`/`publish`/
`unpublish`/`RecordPublisher` gain `dirname: str = RUN_DIRNAME` and their record
annotations widen from `SessionRecord` to a `Protocol` naming the three members
they actually touch (`pid`, `heartbeat_at`, `to_json()`); `publish()` is already
duck-typed (`:65-79`). `pid_alive` is reused unchanged. Every parameter gains a
default equal to today's value, so no existing call site changes behaviour.

Lifecycle:

- **Publish** from `server/app.py`'s `lifespan` (`:73-152`) just after
  `app.state.scheduler_service.start()`, with the heartbeat driven by a small
  `asyncio` task on the loop (15 s, `HEARTBEAT_INTERVAL_S`) — the same discipline
  as `RecordPublisher`, whose `heartbeat()` rewrites the whole record.
- **Unpublish** in the shutdown half of `lifespan`, beside the existing
  `desktop_sessions`/`desktop_auth` teardown, under `finally:` — an exit path
  must never raise over a missing file (`registry.py:82-88`).
- **Atomicity and permissions** are `publish()`'s, unchanged: staged tmp in the
  same directory, `chmod 0600`, `os.replace`.
- **Staleness / liveness** are the same two independent checks
  (`registry.py:131-172`): pid dead → `stale` (reaped by the reader); pid alive
  but `heartbeat_at` older than `HEARTBEAT_TIMEOUT_S` → `wedged` (report, never
  attach). Reparenting to launchd — the `:7341` case — is a non-event: liveness
  is decided by `kill(pid, 0)`, never by the parent.
- **Who cleans up**: clean exit → `lifespan`; SIGTERM/SIGINT → uvicorn's signal
  handling runs the lifespan shutdown; SIGKILL → the file stays and the next
  `scan()` (backend) or the UI's enumerator (§3 step 4) unlinks it, matching the
  convention that the reader reaps.

## 3. The UI's discovery algorithm

Replace `checkExistingBackend()` with `discoverDaemon()` in
`src/main/backend/backend-service.ts` (new method, same class), called from
`start()` and from the health-check failure path.

1. **Enumerate** the `run/serve/*.json` records from the config root —
   `<paths.config_dir()>/run/serve/*.json`: `LOCAL_OPERATOR_CONFIG_DIR` when set,
   else `~/.local-operator` (`paths.py:56-67`, `DEFAULT_CONFIG_DIRNAME =
   ".local-operator"`). NOT `~/local-operator/run/serve/*.json`, which is inside
   the source checkout on a developer machine and nonexistent on every real
   install. Unparseable → ignore; `started_at` far in the future (clock
   skew) → ignore.
2. **Liveness** — `process.kill(record.pid, 0)`: `ESRCH` → stale, do not attach;
   `EPERM` → alive. Same rule as `registry.py:118-128`.
3. **Validate by identity** — `GET http://{host}:{port}/health`, 2 s timeout,
   require `result.instance_id === record.instance_id`. Both halves are needed:
   pid liveness alone is defeated by pid reuse, the port alone by port reuse.
   This is the check today's probe lacks, and the reason `:8080` at v0.44.66 was
   acceptable to it. The record's `host` is always DIALABLE: a wildcard bind
   (`--host 0.0.0.0` / `::`) is recorded as that family's loopback, while an
   explicit address is recorded verbatim — and an IPv6 literal needs brackets
   when this URL is built.
4. **Stale records** — unlink only when `process.kill(pid, 0)` threw `ESRCH`
   *and* the record is older than `HEARTBEAT_TIMEOUT_S`. Never unlink a wedged
   (live-pid) record.
5. **Rank** among valid candidates: (1) the record whose `prefix` is the install
   the UI's own classification resolves as the user's `lop`
   (`resolveLocalOperatorPath()` / `resolveLopUpdatePath()`,
   `update-service.ts:1985-1994`) — the daemon the user's CLI runs, whose
   `version` therefore matches it; (2) otherwise newer `version`, then
   `source_ref`; (3) then newer `started_at`. Rationale: the complaint is that the
   app prefers its own bundled venv, so "same install as your CLI" outranks
   "newest process". Log all candidates with the pick, so a wrong pick is
   diagnosable.
6. **Start our own only when** no candidate validates *and* the manager is
   permitted to manage. Spawn with `--port 0` and the desktop token in the
   environment as today (`backend-service.ts:658-667`), and remember the child's
   pid as the only process this app may kill (§6).
7. **`VITE_DISABLE_BACKEND_MANAGER=true`** stops meaning "assume a backend
   exists" and starts meaning "do not spawn or kill one": `index.ts:650-653` must
   call discovery unconditionally, `checkExistingBackend()`'s early return must
   go, and `startupMode` must not stay `NOT_STARTED` when a daemon was discovered
   (add `DISCOVERED_DAEMON`, or reuse `EXISTING_SERVER`). Then
   `update-service.ts:2054` no longer bails and the version surface describes the
   daemon that is actually serving.
8. **Rotation** must reach both consumers: main's `backendUrl` (used by
   `requestDesktop` and `DesktopStreamRelay`, `backend-service.ts:108-142`, which
   already rebuilds on rotation) *and* the renderer's frozen const — add one IPC
   read (`backend:get-url`) and build the renderer client from it, or route the
   remaining direct-legacy call sites through main.

## 4. Desktop capability handoff

**Recommendation: an explicit claim handshake against a secret published in the
record.** The daemon mints `claim_key = secrets.token_urlsafe(32)` at startup
(only when no env token governs it) and publishes it in the 0600 record; main
reads it and claims the plane once.

- Desktop posture becomes one module-level state in `server/desktop.py`:
  `desktop_posture() -> (token, origins)`, true when the **env** token is set *or*
  a claim has been accepted (new state: `_CLAIMED: tuple[str, set[str]] | None`).
  Replace all five `os.environ.get("LOCAL_OPERATOR_DESKTOP_TOKEN")` predicates
  with it — `desktop.py:29`, `app.py:326`, `app.py:342`, `capabilities.py:19`,
  `dependencies.py:72`. Missing one is the shape of the bug: a claim that turns on
  the routers but not the legacy gate.
- `POST /v1/desktop/claim`, `Authorization: Bearer <claim_key>`, body `{}`:
  `compare_digest` (as `desktop.py:58`), a **single-claim latch** so a second
  claim is refused, and the caller's `Origin` (when present and not `null`) added
  to the runtime allowlist. Respond `{"claimed": true, "instance_id": …}`. The key
  is never returned, logged, or put in a tool result.
- `capabilities.py:19`'s `desktop_available` becomes `False` before a claim and
  `True` after — the signal the UI needs to choose its path without guessing.

**Threat model.** (a) *Who can read the record*: only the owning account — 0600
under 0700, the boundary `registry.py:1-8` already relies on for `control_key`.
Anything that can read it can already attach to the user's sessions, and can also
read `auth.db` and run `lop`; this publishes no new class of secret to that
principal. (b) *What a sandboxed renderer can reach*: nothing new — it never
learns the key, and privileged calls already ride `window.api.desktop.request`
(`shared/api/local-operator/desktop-api.ts:37-51`), so only the token's *source*
changes. (c) *What a local page script can forge*: it cannot read the record, so
it cannot produce the bearer, and it cannot set or remove `Sec-Fetch-Site`
(forbidden header) — the existing defence at `desktop.py:41-56`. **A page
therefore cannot claim at all, even holding the key**: a claim request carrying
`Sec-Fetch-Site` is refused unconditionally, and the intended caller (the app's
main process) sends none. Defence in depth rather than a capability boundary —
the request that installs an Origin puts its SENDER on the allowlist, so a
leaked key must not be spendable by page script. The claim's own `Origin` rule
still differs from `require_desktop`'s about the *value being installed*: an
unknown origin may be installed by a caller that proved the key, and a NATIVE
caller (no Origin header) may declare the origins its renderer needs in the
claim body (`desktop.parse_claim_origins`) — a packaged renderer loads from
`file://`, whose origin is the literal `null` this plane never admits, so
without a declaration there is no way to admit the origin the renderer is
served from. Guessing 256 bits is not a threat. Residual risk, named: the
claim flips the daemon into managed mode, which **tightens** the legacy control
surface (`app.py:326`) and therefore gates `/v1/agents`, `/v1/jobs`,
`/v1/schedules`, `/v1/config`, `/v1/credentials`, `/v1/models` for every other
local caller — open question 1. The CORS half of that tightening is scoped to
the ALLOWLIST IN FORCE, so it narrows only where the plane actually admits a
browser origin: a claim that declared an origin (or a daemon with
``LOCAL_OPERATOR_DESKTOP_ORIGINS`` set) admits exactly that origin and strips
the grant from every other. An EMPTY admitted set keeps the historical wildcard
echo, deliberately and as a measured correction (#1093 shipped the opposite and
broke the app): the shipped renderer is loaded with
``mainWindow.loadFile(...)``, so it runs at ``file://`` and every request it
makes carries the opaque origin ``"null"``, which this plane never admits to an
allowlist. The app sets the token but no origins list, and it reads ``/health``
DIRECTLY as its "server offline" signal — suppressing the echo there removed
``Access-Control-Allow-Origin`` from a 200 and made the app report a healthy
daemon as down. So the residual is named rather than closed: on an
allowlist-less daemon the wildcard echo stays. What protects that state is the
CONTROL half — ``require_desktop`` on ``/v1/credentials``, ``/v1/models``,
``/v1/config`` and the ``/v1/agents``/``/v1/jobs``/``/v1/schedules`` families,
plus the legacy boundary — which is unchanged, and the UI PR removes the last
dependence on the echo by probing health/version from the main process.
(`/v1/chat` is deliberately outside both gate families, so it is unauthenticated
either way — see `DESKTOP_API.md`.)

## 5. "Down" semantics and the state machine

`/health` (200 + `instance_id` match) is the **only** liveness signal. A `401`,
`403` or `503` from a gated route is a *capability* result and must never be
rendered as "server down" — that conflation, plus the `NOT_STARTED`
short-circuit, is most of "reports the server is down when it is really not".

Constants (UI-side, `src/main/backend/config.ts`): `PROBE_INTERVAL_MS = 10_000`,
`PROBE_TIMEOUT_MS = 2_000`, `DEGRADED_AFTER_FAILURES = 3`,
`DETACHED_AFTER_MS = 90_000`, `REATTACH_BACKOFF_MS = 30_000` (×2 to a 5 min
ceiling). The backend's `HEARTBEAT_INTERVAL_S`/`HEARTBEAT_TIMEOUT_S` (15/45) are
reused as the record's freshness budget rather than re-invented.

States; the *owner* axis (`owned` = the child we spawned, `external` =
discovered) is orthogonal:

| state | entered when | exit |
| --- | --- | --- |
| `attached` | a probe passes with an identity match | 1 failure → `degraded` |
| `degraded` | 1–2 consecutive failures, **or** heartbeat older than 45 s while the port still answers | next success → `attached`; 3rd failure → `detached` |
| `detached` | 3 consecutive failures, **or** `kill(pid,0)` → `ESRCH`, **or** re-discovery finds no valid candidate | full re-discovery each backoff tick, then (only then) consider starting |
| `replaced` | this app started a successor (owned only) | as `attached` |

Rules that make it safe: `degraded` never restarts anything; `detached` always
re-discovers before it starts; an **external** daemon is never restarted or
replaced — the UI reports "daemon stopped" and offers to start one; a `wedged`
record (live pid, stale heartbeat) is degraded-and-named, never reaped.

## 6. Ownership and shutdown

- Delete all three `pkill` families: `backend-service.ts:987,1013,1089,1097`;
  `index.ts:934,943,1015,1062,1068` (including `pkill -9 -f python`, which kills
  unrelated interpreters — this host runs five at all times);
  `update-service.ts:1096-1101`. Replacement: kill only `this.process`, by
  handle, `SIGTERM`, wait the existing timeout, then `SIGKILL` on the same pid.
  `backend-service.ts:896-1058` already tracks the pid correctly; the `pkill`
  lines are redundant escalation on top of a working kill.
- The invariant: **the UI kills only pids it spawned** — `owned ⇔
  this.process !== null`. `stop()`'s `isExternalBackend` guard (`:856-862`) stays
  and now also covers `DISCOVERED_DAEMON`.
- On quit while attached: clear the health timer, dispose the SSE relay, leave the
  daemon running, touch nothing on disk. Its heartbeat continues, so a later
  `lop sessions` and a later UI run both still find it.
- When the UI started the daemon, quit should also **leave it running** by default
  (that is what a background server means) and offer an explicit "Stop server"
  action. That removes the last reason for a quit-path sweep.

## 7. Update alignment

- **Which install**: the record names it (`prefix`, `install_kind`).
  `resolveBackendUpdatePlan()` (`update-service.ts:1917-1965`) keeps its
  classification but takes its input from the record when one exists instead of
  re-deriving from `which local-operator` — the install that is *serving* is the
  install to update.
- **How**: for `uv-tool`/`pipx`/`pip`, shell to that install's own entry point,
  `<prefix>/bin/local-operator update` — `update.py::update_command`, the same
  code path as the TUI's `/update` (`app.py:12118-12265`). Do not re-implement the
  installer: the value is that `perform_upgrade` (`update.py:1023-1095`) writes
  `.lop-source` **after** the installer exits 0, and that mtime is the settle
  signal the whole rollout reads. For `APP_BUNDLED_VENV` the existing
  pip-into-venv path stays valid.
- **Zero downtime**: stop installing in front of a stopped server (today's
  `update-service.ts:2357+` stops the backend first). Upgrade in place, then let
  the existing mechanism roll out — idle runtimes see the stamp move
  (`process.py:191-215`), wait out `BUILD_SETTLE_S`, stagger over
  `BUILD_STAGGER_S`, announce `retiring`, and their viewers re-engage a successor
  on the new build; `refresh_mobile_after_upgrade()` (`update.py:1142-1184`)
  kickstarts the mobile relay's LaunchAgent, and its failure is a notice, never a
  rollback. In-flight turns are never cancelled: the residency predicate
  (`process.py:397-470`) decides when a runtime may leave. The UI must not call
  `restart()` around an update.
- **The daemon itself announces only.** A settled install change writes
  `retiring_from`/`retiring_to` into its record, but every production daemon
  continues serving: no build-triggered drain, admission latch, or exit. The
  legacy field names describe a new-build announcement, not a handoff promise.
  The daemon DOES own legacy scheduled/async work: `SchedulerService._run_tasks`
  holds in-process runs and lifespan shutdown cancels them. Detached desktop
  runtimes are only one execution path; an empty attachment probe does not prove
  the daemon is safe to stop. Nor does the install marker prove a successor is
  ready to accept requests.
  - **Re-read announcements while serving.** Returning to the boot build or an
    unreadable stamp withdraws the fields; moving on again retargets them. Record
    identity, discovery, ownership claims, and update installation are unchanged.
  - **UI contract: do not release SSE/watch on these fields.** Keep the session
    relay and watch heartbeat alive, including during turns. The announcement
    alone does not authorize disconnecting, stopping, or rebinding the daemon.
    A future handoff must first prove successor readiness and protect all
    daemon-owned work; that protocol is deliberately not implemented here.
  - **Internal test seam only.** An explicitly injected `exit_process` callback
    retains the drain/latch tests, including typed `503 daemon-retiring` refusal
    coverage. Production lifespan supplies no callback, and there is no flag or
    environment variable to opt into that unsafe path. This is not automatic
    daemon rollout or a zero-downtime daemon upgrade guarantee.
- **Bundled venv**: keep the code path, demote its role. It exists so a fresh
  machine has *a* backend; once a global daemon is discoverable it must not be
  started, must not be updated, and must not be what the version banner describes.
  The banner already reads `/health` (`getInstalledBackendVersion`,
  `update-service.ts:1730-1790`) — the right source; what was wrong was attaching
  to the bundled daemon at all, which §3's ranking fixes.
- **`--port 0`**: `serve_command` (`cli.py:3973-4001`) resolves an ephemeral port
  before starting uvicorn (bind / `getsockname` / close when the requested port is
  0), passes it through, and hands it to the app for the record. The CLI default
  stays 1111 so `curl 127.0.0.1:1111` and existing docs keep working; only the
  UI's own child asks for 0. The residual bind race is harmless because §3.3's
  identity check — not the record's port — is what admits a candidate.

## 8. Skew tolerance and PR split

Order matters: UI work is inert without the record, and the handoff is inert
without the posture refactor.

**Backend, three PRs (each independently reviewable):**
1. `feat(server): publish a serve discovery record and identify the daemon` —
   lands this design's items 1 and 2 TOGETHER (manager decision 2026-09-13: the
   record is untestable end to end without `--port 0` resolution and the
   `/health` identity that makes a candidate verifiable):
   `session/runtime/types.py` (Protocol + `SERVE_RUN_DIRNAME`),
   `session/runtime/registry.py` (dirname parameter, widened annotations), new
   `server/registry.py`, `server/app.py` lifespan publish/heartbeat/unpublish,
   `HealthCheckResponse` gains `instance_id`, `pid`, `prefix`, `install_kind`
   (additive, defaulted; `schemas.py:721-728`), `routes/health.py`, `cli.py`
   listener bind + port resolution + announcement.
2. `feat(server): accept a desktop claim from the record` — `server/desktop.py`
   `desktop_posture()` + `POST /v1/desktop/claim`, and the five predicate sites
   switched to it. Security-sensitive: the only PR here whose QA matrix must cover
   refusal paths, not just the happy path.
3. `fix(server): announce daemon build drift without exiting` — the poll task
   and the record's informational `retiring_from`/`retiring_to` fields.

**UI, one PR:** `src/main/backend/backend-service.ts` (`discoverDaemon`, ranking,
state machine, pid-scoped shutdown), `src/main/backend/config.ts` (constants, no
fixed-port default), `src/main/index.ts` (discovery runs regardless of the manager
flag; delete the sweeps), `src/main/update-service.ts` (record-driven install
plan, no stop-before-update, delete `forceTerminateBackendProcess`),
`src/main/desktop-transport.ts` + renderer URL rotation.

**Skew both ways.** Old daemon + new UI: no `run/serve` records, so discovery
finds nothing and the UI keeps **one release** of the legacy fixed-port probe as a
*logged* fallback (`checkExistingBackend` with its identity-less 200 check, marked
deprecated) — otherwise a UI update strands users whose daemon predates it. New
daemon + old UI: unchanged, because a human-run daemon still defaults to 1111 and
only the new UI asks for `--port 0`. `claim_key`/`install_kind`/`prefix` absent ⇒
defaulted: the UI refuses desktop controls with a named reason and falls back to
`resolveGlobalInstallPlan` for the update plan. Nothing here moves
`PROTOCOL_VERSION`, and the record's fields follow the additive contract
documented at `types.py:255-315`.

## 9. Rejected alternatives

- **Extend `SessionRecord` + `run/mobile` with `kind="serve"`.** Every `scan()`
  reader treats a record there as a session, `kind` is a `Literal` old readers do
  not validate, and `types.py:~190-205` calls the dirname a wire constant — a
  phantom session row is the silent-failure class this repo keeps paying for.
- **Probe a list of candidate ports (1111, 7341, 8080, …).** Needs a hardcoded
  list, cannot find a daemon on an ephemeral port, and validates nothing — today's
  probe accepts the v0.44.66 dev server.
- **Gate the desktop plane on "the record exists" rather than a claim.** No
  authenticated act to test, no revocation short of a restart, and it widens the
  legacy surface of every daemon on the machine, test daemons included.
- **Refuse desktop controls on external daemons, in the UI.** Makes the
  operator's own workflow (TUI-started daemon, sessions already running) unusable;
  kept as the documented fallback, and it is what an un-upgraded daemon does by
  construction.
- **The UI keeps its own daemon and mirrors state with the external one.** Two
  writers for one transcript, and the UI cannot own the other daemon's runtimes —
  `server/utils/desktop_sessions.py` is the adoption mechanism, and that is the
  hardest part of the backend to duplicate.
- **Bump `PROTOCOL_VERSION` for the new fields.** `types.py:27-49`: bumps are for
  breaking socket frames, and a bump makes peers refuse records they can use.
- **Let the UI take ownership of a running daemon (kill and adopt).**
  Destructive, and the daemon may be mid-turn (`busy`, `subagents_running` are
  published per session).

## 10. Risks to watch during rollout

1. **The claim gates legacy routes for third-party local callers** (question 1):
   watch for a working `curl` script starting to 401, and for a renderer call site
   still hitting a gated path directly.
2. **`--reload`** (dev only) publishes from uvicorn's child; verify the reloader's
   own exit cannot orphan the record. **Resolved for the retirement case, and the
   first answer was wrong:** the child ran the build watch, so it announced,
   removed its record and asked its own process to stop — leaving the reloader
   parent still accepting on the port with no record to explain it, and
   `/health` timing out (QA round 1, Q3, three runs). The child now runs no
   build watch at all (`registry.is_reload_child`): a dev-mode supervisor is
   not a production daemon, cannot hand a socket to a successor, and the
   operator is watching its console.
3. **Daemon shutdown cancels owned work.** Production build drift only announces;
   it must not consult the incomplete attachment drain, latch, or exit. Future
   daemon handoff needs protection for legacy scheduled/async runs as well as a
   proven ready successor. Runtime retirement has its own residency predicate
   and is unchanged by this daemon safety correction.
4. **Deleting the `pkill` sweeps** exposes anything that silently depended on them
   to clear a wedged child. Verify the quit path against a daemon that ignores
   SIGTERM, and that an orphan becomes a reportable state rather than a leak.

## 11. Open questions for the operator

1. **Security boundary.** Should a claim put the daemon into fully managed mode
   (gating `/v1/agents`, `/v1/jobs`, `/v1/schedules`, `/v1/config`,
   `/v1/credentials`, `/v1/models` for *every* local caller), or should an attached
   daemon keep its standalone posture and expose only `/v1/desktop/*` to the
   bearer? The first matches the app-managed backend and closes the wildcard-CORS
   surface the app now actively drives; the second avoids breaking local scripts
   against a TUI-started daemon.
2. **May the UI start a daemon when none exists**, and from which install — the
   discovered global `lop`, or the bundled venv? I recommend yes, ephemeral port,
   global install preferred, and never installing without asking.
3. **Daemon restart handoff.** Is a brief "reconnecting" gap acceptable, or do you
   want the daemon supervised (a launchd agent beside the mobile relay's,
   `local_operator/mobile/install.py:110-133`) so the handoff is invisible?
4. **Bundled venv**: keep shipping it for new installs (two installs, two
   versions, indefinitely), or make a `lop`/uv install a prerequisite?
5. **Quit behaviour**: leave the daemon running by default (recommended) or stop
   it on quit?

## 12. Decisions (manager, 2026-09-13)

All five are settled so implementation has one spec. Each is reversible except
where noted, and each carries the reason it was chosen.

1. **Claim puts the daemon into fully managed mode (option A).** The alternative
   leaves the drive-by-page vector open on the very daemon the app is attached
   to: `server/app.py:359-365` registers `allow_origins=["*"]` with
   `allow_credentials=True`, the boundary at `app.py:318-334` only engages when
   `LOCAL_OPERATOR_DESKTOP_TOKEN` is set, and `app.py:368-388` documents that a
   standalone server keeps its historical wildcard CORS. So today the daemon the
   operator's TUI started on a predictable loopback port is readable by any page
   they visit. A claim TIGHTENS that surface, and it is the same posture the app
   already imposes when it starts the backend itself, so behaviour stays
   uniform. Accepted cost, named: `/v1/agents`, `/v1/jobs`, `/v1/schedules`,
   `/v1/config`, `/v1/credentials`, `/v1/models` become bearer-gated for other
   local callers of a claimed daemon. Rollout watch: a working `curl` script
   starting to 401 (design risk 1).
2. **The UI may start a daemon when no candidate validates**: yes, on an
   ephemeral port (`--port 0`), preferring the discovered global install, and
   never installing one without asking.
3. **No launchd agent for `serve` in this program.** The record plus
   pid-liveness liveness makes a reparented daemon findable, which is the
   reported failure; the UI-started daemon also outlives the app by default
   (§6). Supervision is a reversible follow-up if a brief reconnect gap proves
   unacceptable in practice.
4. **Keep the bundled venv, demoted.** It is the only backend a fresh machine
   has, so removing it is a packaging decision, not this change's. Once a global
   daemon is discoverable the bundled one must not be started, updated, or
   described by the version banner.
5. **Quit leaves the daemon running** (both owned and external); the app offers
   an explicit "Stop server" action instead of a quit-path sweep.
6. **The CORS echo is scoped to a NON-EMPTY admitlist, not to the posture**
   (corrects #1093, which scoped it to ``posture.enabled``). The claim still
   tightens the CONTROL surface unconditionally; what changed back is only the
   echo: with ``desktop_posture().origins`` empty the response is returned
   untouched. Measured reason: the SHIPPED app's renderer is loaded with
   ``mainWindow.loadFile(...)``, so it runs at ``file://`` and every request it
   makes carries the opaque origin ``"null"`` (a value this plane never admits
   to an allowlist). The app sets the token but never an origins list, and it
   reads ``/health`` DIRECTLY as its "server offline" signal, so scoping the
   suppression to the posture removed ``Access-Control-Allow-Origin`` from a
   ``200`` ``/health`` and the app reported a live daemon as down — the exact
   symptom this program exists to remove. Named residual: an allowlist-less
   daemon (the app-managed default, and a native claim that declared nothing)
   keeps the historical wildcard echo, i.e. this host's pre-program posture.
   The control half is what protects that state, it is unaffected, and the
   residual retires when the UI PR probes health/version from main instead of
   from the renderer (the renderer call sites named in open question 1).

PR split is confirmed as three backend PRs plus one UI PR (§8), with the first
merged before the other two branch, so both of them build on the record module.
