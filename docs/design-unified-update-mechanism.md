# Design: one update mechanism for every component

Status: proposal (architect). Base: `origin/main` `5766653f9` (local-operator) and
`9a39bcbf1` (local-operator-ui). Docs-only: no `pyproject.toml` or
`package.json` bump — the window's release owner handles that. Six PRs (§7).

Operator's requirement, verbatim:

> There seems to be an issue with the local-operator-ui and backend updates, I
> just checked the backend version and it says 0.56.8 but the latest is 0.56.11,
> and I clicked to check updates and it showed that I'm up to date despite that
> clearly being behind.
>
> Also make sure there's a mechanism to properly run the backend update and
> update the mobile daemon, any central session servers, etc. while keeping the
> runtimes running until they each idle and then each runtime will update to the
> latest version but the central servers, mobile relay, extension workers,
> tunnel, etc. should all properly update with a clickable update button to
> update the backend without taking down or killing any runtimes. It should be
> similar to the lop-update script mechanism. It should also account for whether
> the lop install is app-owned or if it's already installed globally and be able
> to update the global install via package manager or use its own and update its
> own install robustly.

Every `file:line` below is against those two base refs. Lines prefixed
`local-operator-ui:` are in the sibling repository; everything else is in this
one.

## 1. The problems as found in the code

### 1.1 One install, four numbers, all read off this machine

The reproduction is live, and it is worse than the report. `curl
http://127.0.0.1:1111/health` on 2026-09-17 answers:

```json
{"status":200,"message":"ok","result":{
  "version":"0.56.8","instance_id":"g9o4_jr1K_gj_b3M890BXAFHgZm5e3TFyiJFa7vryDA",
  "pid":52156,
  "prefix":"/Users/damian/Library/Application Support/Local Operator/managed-python/packaged/environments/183bbfdc…-7ad511f5…",
  "install_kind":"pip"}}
```

and the same machine carries five disagreeing readings for that one install:

| reading | value | where it comes from |
|---|---|---|
| what answers `/health` | **0.56.8** | `installed_version()` read **per request** (`local_operator/server/routes/health.py:79-81`) |
| what the serve record says | **0.56.2** | the record is minted at boot (`server/registry.py:279`), and `~/.local-operator/run/serve/52156.json` says `version: 0.56.2`, `retiring_from: 0.56.2`, `retiring_to: 0.56.9` |
| the app's selection stamp | **0.55.7** | `selected-environment.json`'s `backendVersion`, written once when the env was prepared (`local-operator-ui:src/main/backend/managed-python.ts:932-944`) |
| the global uv-tool install | **0.56.11** | `~/.local/share/uv/tools/local-operator/lib/python3.12/site-packages/local_operator-0.56.11.dist-info` |
| PyPI | **0.56.11** | `check_latest()` (`update.py:436`) |

The 0.56.2 vs 0.56.8 pair inside one process is not a curiosity, it is the
mechanism in one line: `/health` re-reads the dist-info on every request
deliberately — its docstring says so, because "an install's dist-info and
`sys.prefix` can change UNDER a running daemon (`lop-update` replaces the tree
in place)" (`health.py:91-96`) — while the record deliberately stamps the boot
build, because that is what the process actually loaded. The two are *different
questions answered honestly*, and the product reads them as one.

The env's `site-packages` mtime (2026-09-17 04:35) and the backend installer's
own log (`~/Library/Application Support/Local Operator/logs/backend-installer.log`
2026-09-15 18:29: `Successfully installed … local-operator-0.55.7`) date the
in-place rewrites: the app-managed environment has been pip-installed over in
place at least twice since it was prepared, and the only stamp that records
"which build is published here" is the one written at preparation time, which
therefore lies.

### 1.2 The check's subject is not the install that serves

`UpdateService.resolveBackendUpdatePlan` (`local-operator-ui:src/main/update-service.ts:3330-3379`)
routes **both** `EXISTING_SERVER` and `GLOBAL_INSTALL` into
`resolveGlobalInstallPlan({identity: …})` (`:3353-3354`), where the identity
comes from `resolveInstallIdentity()` → `readInstallIdentity(this.resolveLocalOperatorPath())`
(`:3417-3418`, `:2881-2919`) — that is, from the `local-operator` **console
script found on the search path**, not from the process that is answering the
app.

On this machine, verbatim from `~/Library/Application Support/Local Operator/logs/update-service.log`
(2026-09-17 13:35:37):

```
External backend install (EXISTING_SERVER): local-operator resolves to
/Users/damian/.local/bin/local-operator
(/Users/damian/.local/share/uv/tools/local-operator/bin/local-operator),
classified as uv-tool. …; remedy is `lop update`; install version 0.56.11
Backend version from health API: 0.56.8
Install on disk reports: 0.56.11, running backend reports: 0.56.8,
Latest: 0.56.11, Update needed: false, Startup mode: EXISTING_SERVER
```

`checkForBackendUpdates` compares `plan.installedInstallVersion` against PyPI
(`:3587-3591`, `:3667`), so it compares **the global install** and concludes
"nothing newer". The install that is actually serving the app — 0.56.8, three
patches behind — is never the subject of the comparison. The line at
`:3669-3676` logs both readings and then does nothing about the disagreement
except send `backend-update-not-available` with `runningVersion` (`:3751-3779`);
the *status* stays `current`, and `current` is the whole input to the
affirmation.

The root of the wrong subject is one line further up: the daemon serving this
app is one the app itself spawned. Its argv is
`…/managed-python/packaged/environments/183bbfdc…/bin/python -c "from
local_operator.cli import main; main()" serve --port 1111`, which is
byte-for-byte the argv `ownedServeLaunch` builds
(`local-operator-ui:src/main/backend/owned-serve-launch.ts:7`, `:315`), and its
record carries `desktop: true`. But it has outlived the app process that
started it (`ps -o ppid` says 1), and on the next launch discovery adopted it —
`attachTo` sets `this.isExternalBackend = true` and
`startupMode = EXISTING_SERVER` (`local-operator-ui:src/main/backend/backend-service.ts:1469-1477`,
log: `Attached to daemon http://127.0.0.1:1111 (pid 52156, v0.56.8, pip, …)`).
So the app's *own* install is classified as somebody else's, and the plan is
resolved from PATH instead.

### 1.3 The suite of numbers the user sees is internally consistent and jointly wrong

Three surfaces read three different numbers on purpose, and nothing reconciles
them:

* Settings' "Server version" row prints the **serving daemon's** version, read
  by main (`local-operator-ui:src/renderer/src/features/settings/components/app-updates-section.tsx:118-123`,
  `:292-310`, fed by `DaemonStatusSnapshot.version`) → `0.56.8`.
* The update panel prints the **verdict's affirmation**, which is earned
  whenever both channels report `current`
  (`local-operator-ui:src/main/update-check-verdict.ts:102-117`), and the server
  channel reports `current` from the **global install** → "The application and
  server are up to date".
* The update service log records both readings and treats the disagreement as
  informational (`update-service.ts:3751-3762`).

`readingsDiffer` (`:3751-3756`) is the code that already knows the two numbers
disagree. It is wired to *speak* (`backend-update-not-available` carries
`runningVersion` even on a silent check) but not to *withhold the affirmation*:
`BackendCheckReport.status` has no value for "the install is current and the
daemon serving you is not", so `updateCheckVerdict({app, server: server.status})`
(`:4740`) cannot see it.

### 1.4 What already works, and must not be rebuilt

This is the part that decides the shape of the answer. Almost everything the
operator asked for exists; what is missing is the app's *reading* of it and
three daemon classes' *application* of it.

* **`lop update` is already a complete, generation-based, fleet-safe updater.**
  `update_command` (`update.py:3943`) detects the install kind
  (`install_kind`, `:638`), refuses `editable`/`unknown` by name
  (`editable_refusal` `:3247`, `unknown_refusal` `:3275`), and for a uv tool
  install builds a whole new generation and flips a pointer
  (`perform_upgrade:3305` → `install_into_generation:2038` → `flip_pointer:1547`)
  so that no running process's tree is written. It prunes
  (`prune_generations:2602`) and it refreshes the daemons
  (`refresh_daemons_after_upgrade:3713`).
* **Runtime self-refresh on a stale build already exists and is idle-deferred.**
  `process.py:776-805` (`refresh_check`) → `_refresh_for:862` retires an idle
  runtime so the next engage loads the new build; `_begin_drain:962` and
  `_drain_for:1188` handle the runtime that is *not* idle: it announces, stops
  admitting work, and leaves at the first instant its own turn ends, viewer or
  not. Nothing in flight is aborted.
* **There is a fleet-wide verb for exactly this**: `lop refresh`
  (`cli.py:5083`) asks every live runtime to move to the build on disk without
  killing anything, with a per-runtime verdict vocabulary
  (`control.py:1551-1594`: `moved` / `busy` / `draining` / `current` /
  `unsettled` / `kept` / `unsupported` / `unreachable`) and exit codes that
  distinguish "answered" from "could not be asked".
* **The daemon knows the install that is serving it.** `/health` returns
  `pid`, `prefix`, `install_kind` and `instance_id` (`health.py:90-99`), the
  serve record publishes the same identity plus `desktop` and the
  `retiring_from`/`retiring_to` announcement (`registry.py:261-329`), and the
  desktop's own snapshot already carries `prefix` and `installKind`
  (`local-operator-ui:src/shared/backend-status.ts:73-74`).
* **The app already refuses to run an installer that rewrites a live tree.**
  `resolveGlobalInstallPlan` sets `canManageUpdate` from the *layout*, not the
  version: `generationInstallRoot(identity) !== null`
  (`local-operator-ui:src/main/update-install.ts:2946-2972`, `:3098`), and
  `updateBackend` only calls `updateGlobalInstall` when that is true
  (`update-service.ts:4369-4381`).
* **The app-managed environment already has the generation layout** — it just
  does not use it for updates. `prepareManagedPython` publishes
  `environments/<id>-<uuid>` behind `selected-environment.json` with an
  `environment-ready.json` stamp and reaps superseded generations
  (`local-operator-ui:src/main/backend/managed-python.ts:60-61`, `:926-947`),
  and refuses to overwrite a `ready` selection (`:895-896`). The update path
  bypasses all of it and pip-installs into the published tree
  (`update-service.ts:4441-4470`).

### 1.5 What is actually missing

Five things, all small relative to what exists:

1. **The check has no way to name the serving install.** §1.2. The readings are
   all present (`prefix`, `installKind` on the snapshot); the plan does not use
   them.
2. **`serve` daemons announce but never leave.** `server/retire.py`'s module
   docstring is explicit: "Production lifespan supplies no `exit_process`
   callback: marker drift only updates `retiring_from`/`retiring_to` and the
   daemon keeps serving" (`retire.py:1-20`, `:350-374`, `:267-283`). So after
   an upgrade the central session server keeps serving the old build
   indefinitely, and the only tool the operator has is a manual restart.
3. **Three of the four supervised daemons are bounced only when their plist
   *content* changes.** `refresh_service_daemons_after_upgrade`
   (`update.py:3626-3677`) runs `daemons_refresh_command` (`:3798`), which calls
   each installer's `refresh_plist_if_stale` (`mobile/install.py:157`,
   `browser_bridge/install.py`, `tunnels/install.py:80`, `wakes/install.py`);
   that calls `launchd.rewrite_if_stale` (`launchd.py:682`) and returns a
   non-`repaired` outcome when the rendered plist is byte-identical. With the
   generation layout the plists name the **stable shim**
   (`~/.local/share/lop/bin/python3`, confirmed in the four live
   `~/Library/LaunchAgents/com.local-operator.*.plist`), so the pointer flip
   changes no plist and these daemons keep running the superseded generation.
4. **The mobile daemon is bounced unconditionally** (`refresh_mobile_after_upgrade`,
   `update.py:3495`) — a fix for a much older gap, and the one place today where
   an upgrade interrupts a live daemon rather than deferring to it.
5. **The app-owned environment has no atomic update path**, and the app treats
   it as foreign whenever it adopted its own daemon (§1.2). This is the one that
   produced the operator's report.

## 2. Component inventory and ownership

| # | component | how it is installed | where its version is stamped | who may legitimately update it | how it is launched and supervised | what "idle" means | cost of killing / restarting it |
|---|---|---|---|---|---|---|---|
| 1 | **backend, uv tool (generation layout)** | `uv tool install` into `~/.local/share/lop/generations/<id>` behind `current` (`update.py:1121-1157`, `:2038`) | `lib/pythonX.Y/site-packages/local_operator-<v>.dist-info`, `.lop-source` at the install root (`update.py:774`, `:867`) | the install's own front end: `lop update` (`update.py:3943`) | not supervised. Started by `lop`, by the desktop app's `ownedServeLaunch`, or by hand | not applicable to the *install*: an install is idle when no process has it open, which the generation layout makes true by construction | none — the running build is in another tree |
| 2 | **backend, pipx** | `pipx install local-operator` | dist-info under `~/.local/pipx/venvs/local-operator` | `pipx upgrade local-operator` (`update.py:3223` argv table) | as #1 | as #1 | rewrite in place: `update.py:3329-3335` documents that pip/pipx still rewrite `site-packages` under the fleet |
| 3 | **backend, ordinary pip venv** | `pip install` into a venv or a toolchain base prefix (`update.py:615-635`) | dist-info in the prefix | the package manager that owns the prefix | as #1 | as #1 | as #2 |
| 4 | **backend, editable checkout** | `uv pip install -e` from a repo | dist-info frozen at install time; `direct_url.json` says editable | nobody but the developer (`editable_refusal` `update.py:3247`) | as #1 | as #1 | n/a — the app must not touch it |
| 5 | **backend, app-managed env** | the desktop app builds `managed-python/packaged/environments/<id>-<uuid>` and publishes `selected-environment.json` (`managed-python.ts:880-954`) | dist-info in that venv (**live**), `environment-ready.json` + the pointer's `backendVersion` (**frozen at preparation**, `managed-python.ts:932-944`) | **the app** — it owns the tree | the app spawns `ownedServeLaunch` (`owned-serve-launch.ts:315`) or adopts a live record; nothing supervises it across app restarts | its in-flight probe empty: no SSE subscriber, no websocket, and no desktop bridge holding a request, a watch lease or a warm (`retire.py:122-207`, `desktop_sessions.py:2460-2503`) | today: `pip install --upgrade` in place under ~16 live runtimes, then a restart that drops in-flight turns (`update-service.ts:4441-4470`) |
| 6 | **desktop app** | signed DMG / electron-updater; installed as `/Applications/Local Operator.app` | `CFBundleShortVersionString`; staged artifact sha512 checked before install (`update-service.ts:1832-1858`) | electron-updater, plus a watchdog that survives a quit mid-install (`update-service.ts:1358-1447`) | the OS; the app's own ShipIt job after download | n/a | app quits and relaunches; live turns on *detached* session runtimes survive (see #10) |
| 7 | **mobile daemon (and there is no separate relay)** | `lop mobile install` → LaunchAgent `com.local-operator.mobile` (`mobile/install.py:35`, `:110`) | none of its own: the plist names the **stable shim**, so its build is the install the shim resolves at start (`launchd.py:524-553`) | `lop update` (via `refresh_mobile_after_upgrade`, `update.py:3495`) or `lop mobile restart` | launchd, `KeepAlive`/`RunAtLoad`; `Program` = the stable shim, `argv[0]` = the branded label | no attached phone client holding an SSE stream, no in-flight control op | drops every phone stream for the restart window (a few seconds); a phone mid-turn sees a reconnect |
| 8 | **browser-bridge daemon** ("extension workers") | `lop browser install` → `com.local-operator.browser` (`browser_bridge/install.py:44`) | as #7 | `lop update` (only when the plist content changes, `update.py:3626`) | launchd | no extension connected, no command in flight | the extension loses its socket and reconnects; a command in flight fails |
| 9 | **tunnel (cloudflared connector)** | `lop tunnel install` → `com.local-operator.tunnel` (`tunnels/install.py:15`); the **binary is external**, resolved from PATH and required `>= 2025.4.0` (`tunnels/service.py:29-41`) | the plist names the stable shim; cloudflared's own version is whatever the user's package manager installed — the connector runs with `--no-autoupdate` (`tunnels/service.py:283`) | the harness owns the launchd unit; the *binary* is owned by whoever installed cloudflared (Homebrew) | launchd; the connector + loopback gateway are supervised as one unit (`tunnels/service.py:1`) | nothing to defer: a connector holds no session state | the public URL goes dark for the reconnect window (phone sees Cloudflare 1033 while it is down — measured on this project, `launchd.py:394-398`) |
| 10 | **session runtimes** (one process per session) | spawned on engage; detached from the launcher (live `ps` shows `ppid 1`) | the process publishes its own record: version + `source_ref`, plus a boot record and a turn journal (`process.py:652-700`) | themselves: they self-retire onto the new build (`process.py:776-930`) | **not supervised** — a successor is spawned by the next engage, a wake, or a peer message | `_should_exit` (`process.py:570-602`): no work in flight, no wake within `WARM_WINDOW_S`, no interactive viewer attached | never kill: `lop refresh` (`cli.py:5083`) asks instead, and a busy runtime retires at its own boundary |
| 11 | **serve daemons** (the "central session server") | started by `lop serve`, by the desktop app, or by a legacy fixed-port probe | the serve record: `version`, `source_ref`, `prefix`, `install_kind`, `instance_id` (`registry.py:261-287`); the record's `version` is a **boot** reading, `/health` is live | nobody, today: it announces (`retire.py:210`) and never leaves | launchd covers none of them; the desktop app re-attaches after a restart via discovery | `in_flight()` (`retire.py:122-207`): SSE subscribers, websocket connections, and the desktop plane's bridges/leases/warm tasks. Deliberately **not** a complete work-safety predicate — scheduler-owned work is not counted (`retire.py:9-14`) | a restart is invisible to session runtimes (they re-attach) but cuts the app's event stream and any phone stream for the restart window |
| 12 | **wakes supervisor** | installed on demand by the wake persist path (`ensure_supervisor_installed`, `wakes/install.py:336`) → `com.local-operator.wakes` (`wakes/install.py:67`) | as #7 | `lop update` (only when the plist changes) | launchd | no wake due within `WARM_WINDOW_S` (`buildwatch.py:171`) | a due wake is deferred; the wake store is on disk |
| 13 | **browser extension** | Chrome Web Store (staged review → publish, `AGENTS.md` "Releasing the browser extension") | `extension/manifest.json` `version` (live: **0.1.17**); `extension/package.json` tracks it | the Chrome Web Store's own update channel; the harness submits and promotes via workflow | the browser | n/a | the extension reconnects to #8 on its own |

Ownership, as a rule rather than a table: **the installer that created a tree is
the only thing that may write it.** A `uv tool`/pipx/pip tree belongs to the
package manager that made it; the app-managed environment belongs to the app;
an editable checkout belongs to nobody but its author; the launchd units belong
to the install that rendered them (`_repair_refusal`, `update.py:3745-3795`,
already enforces exactly this for the plist repair).

## 3. The version that is true

Four different questions are being asked, and today all four are answered with
whichever number was nearest.

| the question | the authoritative reading | where it is read | what it must never be |
|---|---|---|---|
| **what is published** | PyPI's `info.version` | `check_latest()` (`update.py:436`) — the only live network read; `cached_latest()` (`:470`) is the offline fallback and never fetches | a version anybody has not fetched |
| **what is installed on disk** | the dist-info name plus `.lop-source` under an install root — `installed_build(prefix)` (`update.py:867`) | for the *pointer*: `disk_build()` (`:1328`), i.e. what a fresh `lop` would load. For a specific root: `installed_build(root)` | `/health` (it is a *running* process's reading) |
| **what is running right now, and from which install** | `installed_build(process_install_root())` for the build — but the identity is `prefix` + `install_kind` + `pid` + `instance_id` from `/health` (`health.py:90-99`), cross-checked against the serve record (`registry.py:238-329`) | `DaemonStatusSnapshot.prefix` / `.installKind` already carry it to the app (`local-operator-ui:src/shared/backend-status.ts:73-74`) | a PATH lookup, a version string alone |
| **what the next launch will load** | for a generation install: the generation `current` resolves to (`current_generation()`, `update.py:1197`; `flip_pointer()` `:1547` is the only write). For the app's env: the venv the pointer names (`managed-python.ts:496-560`, `readManagedSelection`) | `lop --version` / `installed_version()` for a *supervised* process is whatever the shim resolves *at spawn* | the boot stamp of a live process |

Three consequences the design depends on:

1. **A version string is not a build.** `lop-update` builds from `main`
   while `pyproject.toml` names the last release, so two genuinely different
   builds share one version — the same-version rebuild is the dominant handover
   on this host (`control.py:1597-1619`). Every comparison that matters must
   carry the `source_ref` too: `BuildStamp.label()` is `version@ref[:7]`
   (`update.py:181-191`).
2. **The install and the running build are different questions and both belong
   on screen.** The code already says this in a comment
   (`update-service.ts:3669-3672`) and already sends both readings
   (`:3751-3779`). What it does not do is let the disagreement reach the
   affirmation.
3. **A check may affirm "up to date" only about an install it positively read
   and only when nothing older is serving.** Concretely, the affirmation must be
   withheld when `runningVersion` and `installedVersion` disagree, whatever
   either parse to — because the honest sentence in that state is the one the
   operator was owed and did not get: *the install is current; the server
   answering this app is 0.56.8 and will take 0.56.11 at its next restart.*
   The verdict vocabulary needs one more value for it (`serverStillOlder`),
   because `current` and `available` are both false statements about it.

Recommendation: **the affirmation rule is the fix and it is one module.** Extend
`local-operator-ui:src/main/update-check-verdict.ts` — the pure, Electron-free
rule with tests that already exist for exactly this class of lie — with the
running-vs-installed pair as an input, and give the server channel a fourth
status for "the install is current and the daemon is behind". The rejection that
must be recorded here is the tempting shortcut of *suppressing* the affirmation
in the renderer when the two numbers differ: the button paints the verdict, and
a second copy of the rule in a component is how two surfaces come to disagree
about one installation (`check-for-updates-button.tsx:60-65` says the same thing
about a first attempt at this).

## 4. App-owned vs global

### 4.1 The resolution rule

The update plan must be resolved from **the install that is answering**, in this
order:

1. `GET {backendUrl}/health` → `prefix`, `install_kind`, `pid`, `instance_id`.
   This is the only reading that names the serving *install*; it is already
   fetched on every check (`update-service.ts:3141`).
2. Cross-check against the serve record for that pid
   (`~/.local-operator/run/serve/<pid>.json`, `registry.py:238-329`; the app
   already has a reader, `local-operator-ui:src/main/backend/discovery.ts:322`).
   `instance_id` matching is what stops a stranger on the port being described
   as the install (`health.py:53-62`).
3. Classify the prefix: **app-owned** iff it is inside the app's own support
   root (`managedPythonRoot`, `managed-python.ts:222`, and
   `environmentsRoot`, `:314`). Ownership is a fact about the *tree*, not about
   which process started which process.
4. If the prefix is not app-owned, classify by `install_kind` for the command
   that owns it, exactly as `resolveGlobalInstallPlan` already does — but
   against the serving prefix's own markers, not against `~/.local/bin`.

`ownedServeLaunch`'s own argv is the proof that step 3 is needed and sufficient:
the app spawned `…/managed-python/packaged/environments/<id>-<uuid>/bin/python`,
so the install it must update is the one under its own support root, even
though the process has been adopted as `EXISTING_SERVER`
(`backend-service.ts:1469-1477`). The `desktop` flag on the serve record is
**not** the ownership test and must not be repurposed as one: it means "a
desktop plane governs this daemon" (`registry.py:288-301`), which is true of a
daemon the app claimed after the fact.

### 4.2 Options

**A. Keep resolving from PATH, and only fix the affirmation.** Cheapest, and it
does fix the *symptom* the operator reported — the panel would stop claiming
"up to date". Rejected as the whole answer: the check would still be comparing
the wrong install, so the app would still offer to update a global install
while the app-managed one serves, and the operator's second requirement
("update the backend … without taking down any runtimes") would still route
through `pip install --upgrade` into a live tree. It leaves the button able to
name the wrong command.

**B. Resolve from `/health`'s `prefix`, and act per kind.** Recommended. The
readings are on the wire today; no new endpoint is needed for the desktop. It
keeps the one existing gate that matters (`canManageUpdate` = generation layout
only, `update-install.ts:3098`) and it makes the app-owned case explicit rather
than accidental.

**C. Move the whole decision into the backend and have every client ask it.**
A `/v1/update/status` route returning the serving install's build, the disk
build and the published version, with one rule implemented in Python. Tempting
for TUI/phone parity, and rejected *for now* because it inverts the dependency
for no current client: the TUI already answers this correctly for itself
(`/update` runs `check_latest()` against its own install, `tui/app.py:13197`,
`:13245`), the phone has no update surface, and the desktop already holds the
bearer and the identity. Where C earns its keep is the day a *second* surface
needs to name someone else's install; the rule module (§6) is deliberately
shaped so that porting it is a translation rather than a redesign.

### 4.3 What the app must never do to an install it does not own

* Never run `pip install --upgrade` inside a tree it did not create —
  `update-install.ts:3021-3034` already states why, and the classifier already
  distinguishes `uv-tool` / `pipx` / `pip` / `editable` / `global-unknown`.
* Never write a plist, a systemd unit or a `.lop-source` marker for an install
  whose prefix is not its own. `_repair_refusal` (`update.py:3745-3795`) is the
  Python-side twin of this rule and is worth mirroring: a repair may change how
  a daemon is *named*, never which *install* it runs.
* Never touch an `editable` install. A developer's checkout is answered by
  `editable_refusal()` and by `resolveGlobalInstallPlan`'s `editable` branch
  (`update-install.ts:3061-3075`) — and the app must keep naming the remedy
  rather than running it.
* Never treat "we could not classify it" as "pip" (`update-install.ts:3031-3034`).

### 4.4 The app-owned case specifically

The managed environment should be updated **the way everything else in this
product already updates**: publish a new tree, then move a pointer. The
machinery is present and unused.

* Preparation already builds `environments/<id>-<uuid>` with an atomic pointer
  publish and a reap of the superseded generation
  (`managed-python.ts:926-947`).
* The app's current update path instead runs `pip install --upgrade
  local-operator` into the *published* venv with ~16 runtimes importing from it
  (`update-service.ts:4441-4470`), which is the exact incident
  `docs/design-install-generations.md` was written about, one level down.
* The concrete change: add an app-side "prepare an update environment" path that
  (a) creates a **new** venv generation beside the published one, (b) installs
  the target version into it (a target the app already knows how to read from
  PyPI, `update-service.ts:3276`), (c) smoke-tests it via the existing
  `smokeEnvironment` (`managed-python.ts:784`+), and (d) flips
  `selected-environment.json` — leaving the previously published generation
  intact as the rollback. Only then is the app-owned daemon bounced, at an idle
  boundary (§5).
* The per-install front end stays the fallback for **non**-app-owned
  uv-tool/pipx installs: run the install's own `lop update`
  (`update-service.ts:3973-3992` already does exactly this, and its comment
  `:3949-3971` already gives the reasons). One command, reached through the
  *serving* install's launcher.

## 5. Idle-deferred application without killing anything

The operator's sentence — "keeping the runtimes running until they each idle and
then each runtime will update" — is already the implemented behaviour for the
one class where it matters most. The design's job is to extend the *shape*, not
invent a second one.

**The shape, in order of preference:** (1) land the new build in a tree nothing
is reading, (2) tell the consumer the build moved, (3) let it leave at its own
boundary, (4) only then start anything.

### 5.1 Per component class

| class | can it be updated in place? | mechanism | what the user sees meanwhile |
|---|---|---|---|
| uv-tool install (generations) | no — and that is the feature | `lop update` builds a new generation and flips `current` (`update.py:2038`, `:1547`) | nothing on the install side; running sessions keep working |
| session runtimes | no, and never killed | idle → `_refresh_for` (`process.py:862`); busy → `_begin_drain`/`_drain_for` (`:962`, `:1188`), which stops admissions and leaves when the turn ends | the viewer gets a `retiring` frame and re-engages onto the successor; no `⊘ interrupted` is painted (`docs/design-runtime-autorefresh.md` §4.1) |
| app-managed env | no (today it is, and that is the bug) | new venv generation + pointer flip (§4.4) | the running daemon keeps serving the old tree; Settings shows both numbers once §3 lands |
| app-owned serve daemon | n/a | announce, then bounce **when its own in-flight probe is empty AND no session it hosts is mid-turn** | the app's stream reconnects; in-flight turns are never cut |
| adopted/foreign serve daemon | n/a | never bounced. Announce in the record and let the app re-attach | the skew is *stated* (§6), not acted on |
| mobile daemon | n/a | bounce at idle (no phone stream, no in-flight op); today it is bounced unconditionally | the phone reconnects |
| browser bridge | n/a | bounce at idle (no extension attached, no command in flight) | the extension reconnects |
| tunnel | n/a | bounce, but as a *stated* cost: the public URL is dark for the reconnect window | the phone shows a tunnel error for a few seconds |
| wakes supervisor | n/a | bounce at idle, i.e. no wake due within the window | a due wake is deferred, never lost (the store is on disk) |
| extension | n/a | the Web Store's own channel; nothing for this mechanism to do | browser-native |

### 5.2 A runtime that never goes idle

There is already a bound, and it is worth restating because it is the answer to
the hardest part of the requirement. The idle gate (`_should_exit`,
`process.py:570-602`) is *soft*; the belt is not:
`BUILD_MAX_STALE_GENERATIONS = 3` and `BUILD_MAX_STALENESS_S = 30 * 60.0`
(`process.py:154-155`). A runtime that keeps declining a settled newer build
trips `hard_stale` (`:378-379`), which drains: it stops admitting work, finishes
its own turn, and leaves — "nothing in flight is ever aborted by either path"
(`:760-767`). So "a runtime that never goes idle" converges in at most 30
minutes and never by killing a turn. `lop refresh` (`cli.py:5083-5103`) is the
same thing on demand, with `busy` reported as a queued move rather than a
failure (`control.py:1572-1594`).

### 5.3 Mid-turn, blocked-on-a-tool, and watched

* **Mid-turn**: nothing in this design ends a turn. The runtime's own drain
  waits for the turn; the daemon's retirement predicate exists precisely to say
  "an exit here would CUT rather than pause" (`desktop_sessions.py:2460-2503`).
* **Waiting on a tool**: indistinguishable from mid-turn to the predicate
  (`_work_in_flight` is `handle.is_busy()`, `process.py:560-567`, and a tool is
  inside the turn) — and an unreadable probe means *stay* (`retire.py:117-119`,
  `process.py:565-567`).
* **Being watched by an attached viewer**: the viewer term holds the *quiet*
  exit and deliberately does **not** hold the *build* exit
  (`process.py:1202-1205`) — holding for a viewer is exactly what kept
  five-hour-stale runtimes resident (`docs/design-runtime-autorefresh.md` §1.1).
  A viewer learns first (`retiring` frame / `retiring_from`+`retiring_to` in the
  record) and re-engages afterwards.
* **A daemon hosting sessions** is the one place a plain restart is not enough:
  sessions are separate, detached processes (live evidence: `ppid 1`), so a
  daemon restart is survivable — but the app's own event stream and any phone
  stream are not. That is why the daemon's terms are an announcement plus a
  drain, and why the design keeps the app out of a daemon a person started.

### 5.4 One gap this leaves, stated rather than smoothed over

The four launchd daemons name the **stable shim**, so neither a pointer flip
nor a plist rewrite restarts them, and the daemon refresh is a no-op when the
rendered plist is unchanged (`launchd.py:682`, `mobile/install.py:187-199`).
This is why §1.5 items 3 and 4 are on the PR plan rather than left implicit: a
correct `lop update` that leaves the mobile daemon, the browser bridge and the
tunnel running the previous generation for an unbounded time is not "properly
updated" in the operator's sense, however clean the pointer flip was.

## 6. The clickable action

### 6.1 One action, one report

The desktop already has one button (`CheckForUpdatesButton`) whose check returns
a verdict (`update-service.ts:4728-4741`) and whose "update" call is
`updateBackend` (`:4303`). The design keeps that shape and changes two things:
the plan's *subject* (§4) and the *report* (below).

The action is `updateAll()` — one call, one progress stream, one result. Its
per-component result vocabulary, which is deliberately an extension of words the
harness already uses rather than a new set:

| value | means | where the word already exists |
|---|---|---|
| `current` | this component is on the published build, positively read | `REFRESH_SETTLED_METHODS` (`control.py:1594`) |
| `updated` | the new build landed and this component is on it | — |
| `deferred-until-idle` | the build landed; this component is still on the old one and will move by itself at its own boundary | `busy` / `draining` (`control.py:1555-1560`), `LEAVING_FOR_BUILD` (`types.py:411`) |
| `needs-a-package-manager` | the app may not write this tree; a command is named | `update-install.ts:3125-3127` |
| `could-not` | the attempt ran and did not land (with the installer's own stderr tail) | `update-service.ts:4115-4135` |
| `not-checked` | **nothing was read about this component** | new, and the point of it |

`not-checked` is the load-bearing addition. The rule that a surface may not
affirm what it did not check is already written down for a whole check
(`update-check-verdict.ts:1-20`); per component it becomes concrete: a component
the app cannot see (no mobile daemon installed, no tunnel configured) reports
`not-checked`, and the summary sentence may then say nothing about it. The
forbidden sentence is not "an error occurred" — it is **"up to date"**, which
today can be printed over a component nobody asked.

### 6.2 What the button may claim when it finishes

* It may claim what it *did*, per component, with the reading behind it
  (`0.56.8 → 0.56.11`, named).
* It may claim "at the published version" **only** for components it positively
  read, and **only** when no older build is serving the app (§3).
* It may not claim anything about a component it did not check, and it may not
  claim that a deferred component is "up to date" — the honest sentence is the
  one the log already writes (`update-service.ts:3669-3672`) and no surface
  renders: *the install is current; the server answering this app is on the
  previous build and will move when it is next restarted.*
* It may not offer an action it cannot perform on that install. Today the plan
  already gates that (`canManageUpdate`, `update-install.ts:3098`); the button
  must be *rendered* from the same gate rather than from the version comparison,
  so a legacy-layout install shows the command and a reason instead of a button
  that will refuse.

### 6.3 Where the single source of truth lives

`local-operator-ui:src/main/update-check-verdict.ts`, extended — not a second
module. It is already pure (`update-check-verdict.ts:19-22`), already has node-runner tests that need
no Electron fixture, and already exists because a check was allowed to speak for
a channel it had not asked. The extension is:

* the server channel gains a fourth status (`serverStillOlder`), derived in the
  rule from `(runningVersion, installedVersion)`;
* `updateCheckAffirmation` refuses on it (`:110-118`);
* the per-component report type (§6.1) lives beside it, so the button, the
  preload contract and the tests read one vocabulary.

The *readings* stay where they are produced — `/health` and the daemon snapshot
in main, `lop`'s own output for the installer side — because a rule module that
also fetches is a rule module that cannot be tested without a network.

## 7. PR split

Six PRs, each independently shippable, in dependency order. PR 1 is the one a
separate agent is already implementing; I agree with that boundary and the
reason is in §7.1.

| # | repo | scope (one line) | why the boundary is here |
|---|---|---|---|
| 1 | local-operator-ui | **the reported defect**: the server channel reports `serverStillOlder` and the whole-check affirmation is withheld while the daemon serving the app is behind the install | it is one rule module plus its wiring; it fixes the operator's sentence without touching what the update *does*, so it can land while §4–§6 are still in design |
| 2 | local-operator-ui | resolve the backend update plan from the **serving install** (`/health.prefix` + `install_kind`, cross-checked with the serve record), and make app- vs externally-owned a fact about the tree | this is the change that makes the *plan* right; separating it from PR 1 means each can be reviewed on its own evidence and PR 1 can ship first |
| 3 | local-operator-ui | app-owned environment updates by **publishing a new generation** (prepare → install → smoke → pointer flip) instead of `pip install --upgrade` in place | it is the only PR that writes to a live tree's neighbour; it needs PR 2's ownership resolution to know when it applies |
| 4 | local-operator | supervised daemons take the new build **without a plist rewrite**: the refresh path restarts a daemon whose *install* moved, at its idle boundary, and reports it | this is the `update.py`/launchd half of §1.5 items 3–4; it is language-separate from PRs 1–3 and independently testable (fake plists, fake launchctl) |
| 5 | local-operator | `serve` daemons gain a bounded, announced handover for an **app-owned, unsupervised** daemon, and `lop refresh --all`'s vocabulary covers them | depends on nothing in PRs 1–3; the safety argument (successor readiness) is the hard part and deserves its own round |
| 6 | local-operator-ui | the one-click `updateAll()` action and its per-component report | last, because it is the surface over results PRs 2–5 produce; building it first would freeze a contract against unbuilt behaviour |

### 7.1 The split I would change

The task statement says PR 1 fixes the reported defect by itself. I agree, with
one qualification: **PR 1 must not try to fix the plan's subject.** The
temptation is to make the check compare the serving install in the same PR
("while we are here"), which would (a) put a second decision path into a PR
whose whole value is that it is one rule, and (b) produce a check that says
"0.56.8 is behind 0.56.11" and then offers `lop update` — a command that updates
the *global* install and leaves the app-managed one at 0.56.8. That is a worse
lie than the current one, because it is actionable and wrong. The subject must
move (PR 2) *before* the check can be allowed to name a remedy.

Genuinely out of scope for this work, and flagged rather than dropped:

* **The extension's store channel** — the Web Store's review queue is the
  update mechanism, and `AGENTS.md` already documents the staged-publish
  protocol. Nothing here should drive it.
* **The desktop app's own electron-updater path** — it already has a verified
  staged artifact, a ship-it watchdog and a pending-install marker
  (`update-service.ts:1358`, `:1850`). PR 6 should *report* the app's channel in
  the same per-component vocabulary, not change how it installs.
* **cloudflared's binary** — an external package-manager-owned dependency
  (`tunnels/service.py:29-41`). The mechanism may report its version and bounce
  the connector; it must not install it.
* **A cross-process resource budget for concurrent installs** — deliberately
  not built elsewhere in this harness (`AGENTS.md`, group reaper note), and not
  needed here: `globalUpdateInFlight` (`update-service.ts:4040-4053`) and the
  generation lock (`_reserve_generation`, `update.py:1392`) already serialise
  the writers.

## 8. Test plan

Unit-first, because every rule above is a pure decision:

* **PR 1** — table-driven cases over `(runningVersion, installedVersion, app,
  published)`: the operator's exact triple (0.56.8 running / 0.56.11 install /
  0.56.11 published) must **not** earn the affirmation and must set
  `serverStillOlder`; `("", "0.56.11", …)` and `isReadableVersion("Unknown")`
  stay `unavailable`; the 558-version corpus in `update-check-verdict.ts`'s
  docstring must still be accepted. Prove the test can fail: revert the rule and
  watch the operator's triple pass the affirmation.
* **PR 2** — the plan resolver against fixture identities: an app-support-root
  prefix ⇒ app-owned even when the daemon was adopted
  (`startupMode === EXISTING_SERVER`); a `~/.local/bin` shim pointing at a uv
  tool generation while `/health.prefix` names the app env ⇒ the **env** is the
  subject; a mismatched `instance_id` ⇒ refuse rather than describe.
* **PR 3** — the generation publish against a temp root: a second generation
  appears, the pointer flips atomically, the previous generation survives as a
  rollback, and a failed smoke test leaves the pointer untouched. Then a real
  run on this machine's layout, with a live daemon reading the old tree, proving
  the daemon's files are not written.
* **PR 4** — fake launchd: a plist whose rendered content is unchanged but whose
  *install* moved must be restarted; an unchanged install must not be; a
  restart failure must surface the recovery command (`launchd.reload_failure`).
* **PR 5** — the handover: an app-owned daemon with an in-flight turn must not
  leave until the turn ends; an unsupervised daemon must not exit without a
  successor; the refusal must be typed and carried on the wire.
* **PR 6** — the report: a component that was not checked renders `not-checked`
  and the summary says nothing about it; a deferred component never renders
  "up to date".

End-to-end, per the house rule that a green unit suite is not evidence: stand
the real thing up on this host's layout — a real generation install, a real
serve daemon, real session runtimes mid-turn — run the action, and capture
`/health`, the serve records, and `lop refresh --json` before and after. The
visual half needs rendered frames: the Settings panel in the "install current,
server behind" state, before and after PR 1, since that state is the one no
existing storyboard frame covers (`local-operator-ui:docs/evidence/settings-app-updates-section/`
captures `all-current` and `server-update-offered`, neither of which is this).

Risks that need a QA pass rather than a unit test: two updaters racing
(`globalUpdateInFlight` is per-process), a pointer flip observed mid-write
(the settle window, `buildwatch.py:79`), and a daemon restart while the phone
holds a stream.

## 9. Risks to watch during rollout

* **A check that names a remedy it cannot perform.** §7.1. The guard is that
  PR 1 and PR 2 ship separately and PR 1's rule cannot name a command.
* **Bouncing a daemon into a window it did not choose.** The mobile bounce is
  unconditional today (PR 4 changes it); until then an upgrade can interrupt a
  phone stream, and the tunnel's dark window is a *user-visible* cost that must
  be stated before the click, not after (`update-install.ts:3125-3127` is the
  precedent for saying it).
* **The `desktop` flag being repurposed as ownership.** It is per-daemon
  governance, refreshed by the claim (`registry.py:288-301`). Ownership is the
  tree's prefix; a reviewer should treat any use of `desktop` for the other
  question as a finding.
* **Generation growth.** A new env per backend update is a real cost (the
  prepared env is ~47 MB of venv plus a shared runtime, and the runtime is
  reused). The retention policy must be the existing structural one
  (`prune_generations`, `update.py:2602`; `reapSupersededGenerations`,
  `managed-python.ts:679`) rather than a second rule.
* **A same-version rebuild.** Version equality is not build equality
  (`control.py:1597-1619`); every comparison in the new code must carry
  `source_ref`, or `lop-update`'s dominant handover shape will read as "current".

## 10. What I would NOT do

* **Not a second updater.** `lop update` is the install's front end, and the app
  already delegates to it (`update-service.ts:3973-3992`). A desktop-side
  installer that runs `uv`/`pip` itself would bypass the generation layout, the
  `.lop-source` marker the runtimes converge on, the prune, and the daemon
  refresh — four things the operating host depends on.
* **Not a new wire protocol for "which install is serving".** `/health` already
  carries `prefix` and `install_kind`, the serve record carries the same
  identity plus `instance_id`, and the app's own snapshot already forwards
  `prefix`/`installKind`. Anything new here is a second opinion.
* **Not a kill-then-update path for the app-managed env.** The in-place pip
  rewrite under a live fleet is the incident `docs/design-install-generations.md`
  documents; repeating it inside the app-owned tree, where ~16 session runtimes
  are importing from it, is the same defect with a smaller blast radius.
* **Not a hard deadline on a session's retirement for the operator's benefit.**
  The belt exists (30 minutes, `process.py:154-155`) and it is already the right
  answer: it bounds the wait without ever cutting a turn. A "force update now"
  button that kills a runtime mid-turn would be the 2026-09-14 signal sweep
  again, wearing a UI.
* **Not updating the extension from here.** The Web Store owns that channel, and
  a local installer for a store-managed extension is a second, unsigned update
  path for code that runs in the user's browser.
* **Not making the tunnel's bounce silent.** A public URL that goes dark for a
  reconnect window is a fact the user must see before they click, because the
  cost falls on their phone, not on this machine.
