# Design: one update mechanism for every component

Status: proposal (architect), **revision 3** — round-1 review findings worked in
(R2 is now §1.6, the spine of the document; R1 reshaped §5.1/§6.1) and round 2's
(R10 makes the daemons' live reading the generation each actually *runs*, which
sharpens §1.6 rather than softening it; R11 names the owner of the convergence;
R12 scopes the client-release state and says whose bound it is). Base:
`origin/main` `5766653f9` (local-operator) and `9a39bcbf1` (local-operator-ui).
Docs-only: no `pyproject.toml` or `package.json` bump — the window's release
owner handles that. Six PRs (§7).

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

### 1.1 One install, six readings, two trees — all read off this machine

The reproduction is live, and it is worse than the report. `curl
http://127.0.0.1:1111/health` on 2026-09-17 answers:

```json
{"status":200,"message":"ok","result":{
  "version":"0.56.8","instance_id":"g9o4_jr1K_gj_b3M890BXAFHgZm5e3TFyiJFa7vryDA",
  "pid":52156,
  "prefix":"/Users/damian/Library/Application Support/Local Operator/managed-python/packaged/environments/183bbfdc…-7ad511f5…",
  "install_kind":"pip"}}
```

and the same machine carries six disagreeing readings — belonging to **two
different install trees**, which is the fact the rest of this document turns on
the most:

| reading | value | where it comes from |
|---|---|---|
| what answers `/health` | **0.56.8** | `installed_version()` read **per request** (`local_operator/server/routes/health.py:79-81`) |
| what the serve record says | **0.56.2** | the record is stamped at boot (`server/registry.py:398-405`), and `~/.local-operator/run/serve/52156.json` says `version: 0.56.2`, `retiring_from: 0.56.2`, `retiring_to: 0.56.9` |
| the app's selection stamp | **0.55.7** | `selected-environment.json`'s `backendVersion`, written once when the env was prepared (`local-operator-ui:src/main/backend/managed-python.ts:932-944`) |
| **the install the four supervised daemons are *running*** | **0.56.6 (mobile), 0.56.2 (the other three)** | each daemon's argv names the generation it started from: the mobile daemon (pid 54993) on `…/generations/20260917T055738Z-0.56.6/…`, and the browser, tunnel and wakes daemons (pids 80178/88076/57921) on `…/generations/20260916T212045Z-0.56.2/…`. Every `com.local-operator.*` LaunchAgent's `Program` names `~/.local/share/lop/bin/python3`, the shim that resolves `current` — **once, at exec** (`update.py:1161-1163`) — so a running daemon is pinned to the generation it started with; the pointer itself says **0.56.9**, which is what a daemon restarted *now* would load |
| **the install `lop` on PATH runs** | **0.56.11** | `~/.local/bin/lop` → `~/.local/share/uv/tools/local-operator/bin/lop`; dist-info at `lib/python3.14/site-packages/local_operator-0.56.11.dist-info`, `.lop-source` = `5766653f9… main` |
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

Reproduced here from the trees' own interpreters:

```
$ ~/.local/share/uv/tools/local-operator/bin/python -c '<update.installed_build / disk_build / current_generation>'
kind: uv-tool            prefix: /Users/damian/.local/share/uv/tools/local-operator
installed_build()        0.56.11@5766653f9
disk_build()             0.56.9
current_generation()     /Users/damian/.local/share/lop/generations/20260917T091055Z-0.56.9
stable_root()            /Users/damian/.local/share/lop
$ ~/.local/bin/lop --version
v0.56.11
$ ~/.local/share/lop/bin/python3 -c 'import local_operator; print(local_operator.__file__)'
.../generations/20260917T091055Z-0.56.9/tools/local-operator/lib/python3.14/site-packages/local_operator/__init__.py
```

That last reading is a **fresh child** of the shim: it resolves `current` at
exec, so it reports what a daemon started *now* would load — not what the running
daemons load, which is 0.56.6 and 0.56.2, because that resolution happens once,
when the process starts (the daemon row of the table above, and §1.6).

`installed_build()` names the tree the *reading process* runs — by design, so
`lop --version` describes the code in memory (`update.py:1328-1336`). `disk_build()`
reads the **pointer** (`current_install_root()`, `update.py:1371`), which is what
a daemon launched through `<stable>/bin/python3` loads. On this machine those two
are different trees at different versions, and §1.6 is where that is worked out.

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
(`:3587-3591`, `:3667`), so it compares **the legacy uv-tool tree** — the one
`lop` on PATH resolves, not the install serving the app — and concludes
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
  channel reports `current` from the **uv-tool tree** (not the tree the daemons
  load, §1.6) → "The application and
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
   **And this is not only the app-owned case**: for a daemon a person started
   (`lop serve` in a terminal, a dev server, a rig), nothing in this design
   *moves* it either — the app has no standing to restart another owner's
   process. §5.1 makes that class's deliverable explicit (a stated skew plus the
   exact command) and §7 says why PR 5 is scoped to the app-owned case rather
   than leaving the omission to be inferred.
3. **Three of the four supervised daemons are bounced only when their plist
   *content* changes.** `refresh_service_daemons_after_upgrade`
   (`update.py:3626-3677`) runs `daemons_refresh_command` (`:3798`), which calls
   each installer's `refresh_plist_if_stale` (`mobile/install.py:157`,
   `browser_bridge/install.py`, `tunnels/install.py:80`, `wakes/install.py`);
   that calls `launchd.rewrite_if_stale` (`launchd.py:682`), which returns
   `kind="current"` and "never restarts anything" on byte-identical content
   (`:688-700`). With the generation layout the units name the **stable shim**
   (`~/.local/share/lop/bin/python3`, confirmed in all four live
   `~/Library/LaunchAgents/com.local-operator.*.plist`), so a pointer flip
   changes no unit — **and, because the shim resolves `current` once at exec, a
   daemon that is already running is pinned to whatever generation it started
   with**, which is why only a restart ever moves one. **This is not hypothetical
   and not future: it is the state of this machine right now** — the mobile
   daemon runs 0.56.6 and the other three 0.56.2, five and nine patches behind the
   published 0.56.11, while the pointer they would load on a restart says 0.56.9
   (§1.6). The same is true on a systemd host through the sibling unit renderer
   (§2's note on supervision).

   **Landed 2026-09-24 (PR #1515), for the supervised daemons.** The refresh now asks
   the second question after the plist comparison: the pid launchd holds for the
   label (``launchd.job_pid``), then the generation that process's own argv names
   (``update.stale_generation_of_process``, which reads ``ps -o args=`` — the shim
   execs the generation's own image, so argv carries the generation). A daemon whose
   build provably moved is restarted with a ``kickstart -k``; every unreadable answer
   leaves it where it is, and a machine without the layout is not probed at all. §5.4
   carries the measurement. Items 2 and 4 above are untouched by that change, and the
   same question is still unasked on a systemd host — there the unit names
   ``procname.supervised_image()``, so there is no stale interpreter path for a
   rewrite to fix and nothing in argv to compare.
4. **The mobile daemon is bounced unconditionally** (`refresh_mobile_after_upgrade`,
   `update.py:3495`) — a fix for a much older gap, and the one place today where
   an upgrade interrupts a live daemon rather than deferring to it.
5. **The app-owned environment has no atomic update path**, and the app treats
   it as foreign whenever it adopted its own daemon (§1.2). This is the one that
   produced the operator's report.

Plus the one below, which is not a missing mechanism at all — it is a mechanism
that never runs.

### 1.6 Two update paths, one machine: the convergence problem

The layout the code intends (`update.py:1064-1071`, comment verbatim):

```
~/.local/bin/lop -\
~/.local/bin/local-operator --+--> ~/.local/share/lop/current
                                        |
                                        v
                  ~/.local/share/lop/generations/<id>/
                      bin/lop          -> tools/local-operator/bin/lop
                      tools/local-operator/         (the venv, sys.prefix)
                      tools/local-operator/.lop-source
```

What this machine actually runs, verified today:

* **Path 1 — `lop-update`**, the script at `~/.local/bin/lop-update` (line 291:
  `uv tool install --force --from "$SNAPSHOT" local-operator`). It rewrites
  `~/.local/share/uv/tools/local-operator` in place — the incident
  `docs/design-install-generations.md` was written about — and uv re-writes that
  tool's console scripts itself: the receipt at
  `~/.local/share/uv/tools/local-operator/uv-receipt.toml` records
  `install-path = "/Users/damian/.local/bin/lop"`, and the tree's `.lop-source`
  and both launcher symlinks all carry the same mtime, 13:27 today. Result:
  **the uv-tool tree is at 0.56.11, and `lop` on PATH is that tree.**
* **Path 2 — `lop update`** (the product's own updater, `update.py:3943`).
  Publishes a generation into `~/.local/share/lop/generations/<id>` and flips
  `current` (`install_into_generation:2038`, `flip_pointer:1547`), then writes
  stable launchers so that `~/.local/bin/lop` resolves the pointer
  (`write_stable_launchers:1960-1968`). The newest generation here is
  `20260917T091055Z-0.56.9`, `.lop-source` = `pypi 0.56.9` — a PyPI-sourced
  generation, so it came from this path (or from `lop-fleet-update`, the
  operator's tool that drives `lop update` around a live fleet,
  `~/tools/lop-fleet-update/docs/README.md`). **The four supervised daemons load
  Path 2, and are running generations older than its pointer**: every
  `com.local-operator.*` LaunchAgent's `Program` is
  `~/.local/share/lop/bin/python3`, the shim that resolves `current` — and it
  resolves it **once, at exec**: "It resolves the install pointer ONCE, here, and
  execs that generation's interpreter by an absolute path" (`update.py:1161-1163`).
  A running daemon therefore keeps the generation it started with, and no pointer
  flip reaches it. Live now: the mobile daemon (pid 54993) runs
  `…/generations/20260917T055738Z-0.56.6/…`, and the browser, tunnel and wakes
  daemons (pids 80178/88076/57921) run
  `…/generations/20260916T212045Z-0.56.2/…`, started between Sep 16 17:20 and Sep
  17 05:07 — **five and nine patches behind the published 0.56.11**, not the one
  patch the pointer's 0.56.9 implies. A child of that shim spawned *now* does
  report `…-0.56.9/tools/local-operator` and build `0.56.9`; that is a fresh
  launch's reading, and it was mistaken for the daemons' own.

  **This makes the section's point stronger, not weaker.** The pointer's 0.56.9
  is not the daemons' state — it is the state of a restart that has not happened
  — and the once-at-exec resolution is precisely why a restart is the only thing
  that ever moves a supervised daemon, and why the pointer flip alone changes
  nothing for them.

**Two publishers, one shared artifact.** `~/.local/bin/lop` is written by uv
(Path 1) and by `write_stable_launchers` (Path 2), with no coordination between
them: whichever ran last wins. Today Path 1 won, so `lop` resolves the uv-tool
tree while the pointer — and therefore what a supervised daemon would load on a
restart — stays on 0.56.9. What those daemons are *running* is older still
(0.56.6 and 0.56.2 in §1.1's row), because each resolved the pointer once, at
exec, and has kept that generation since.

Three consequences, and the first one *is* the operator's requirement failing:

1. **`lop update` is a no-op, so the daemon refresh never runs.** Run from `lop`
   on PATH, `check_latest()` compares `installed_version()` = 0.56.11 against
   PyPI's 0.56.11 and prints "is the latest", returning 0 before any upgrade
   (`update.py:3984-3986`). `refresh_daemons_after_upgrade()` (`:3713`) is
   therefore never reached, and the four daemons stay pinned to the generations
   they started on — 0.56.6 and 0.56.2, five and nine patches behind the published
   0.56.11 — for as long as nothing restarts them.
   Executed here rather than inferred:

   ```
   $ ~/.local/bin/lop update --check
   local-operator 0.56.11 is the latest
   exit=0
   ```

   That is the tree the four daemons are *not* running, reporting that there is
   nothing to do.
   The machinery the operator asked for exists —
   `refresh_service_daemons_after_upgrade` (`:3626`) and
   `refresh_mobile_after_upgrade` (`:3495`) are exactly "update the mobile
   daemon, the extension workers and the tunnel" — and nothing wrong with it is
   why it does not fire: it is never asked.
2. **The reading the app's check compares is Path 1's** (§1.2), while the install
   the four daemons load is Path 2's. Neither the check nor the plan knows
   Path 2 exists.
3. **The product already named this state**, before this change:
   `write_stable_launchers`'s own docstring (`:1981-1987`) describes
   "a machine split between two layouts, reported as a clean adoption" — the
   failure its return value exists to prevent. `lop install migrate`
   (`update.py:2846`) is the manual adoption, and it refuses a source checkout.

**Which path the button should drive: Path 2, the generation pointer — wherever
the serving install is a tool-install daemon.** The subject is still always the
*serving* install (§4.1); Path 2 is the answer for the daemons the operator
named, because that is the tree their units load. It is the only one that can be
moved without rewriting a tree a live process has open, and it is the layout the
product's own code intends. Concretely the action must (i) take the pointer as
the daemon-side subject (`disk_build()`, `current_install_root()`,
`update.py:1328-1374`), (ii) run the update from the install that owns the
pointer rather than from whatever `lop` resolves to on PATH, and (iii) converge
`~/.local/bin` onto `<stable>/current/bin` as part of the same action
(`write_stable_launchers`), because otherwise the next `lop-update` silently
flips `lop` back to the uv-tool tree and strands the daemons again. A button
that ran `lop update` through today's PATH `lop` would report success while
moving nothing — the same class of lie as §1.2, one level up.

Until convergence lands, the honest report says *which* install each reading
belongs to: "0.56.11 is published and is what `lop` runs" and "your four
supervised daemons are running 0.56.6 and 0.56.2, and would load 0.56.9 at their
next restart" are both true, and neither is the whole answer. That is what §6's
per-component report is for.

## 2. Component inventory and ownership

| # | component | how it is installed | where its version is stamped | who may legitimately update it | how it is launched and supervised | what "idle" means | cost of killing / restarting it |
|---|---|---|---|---|---|---|---|
| 1 | **backend, uv tool — the generation layout** | uv aims at one reserved root: `generations/<id>` under the stable root, behind `current` (`install_into_generation`, `update.py:2038-2140`; `UV_TOOL_DIR`/`UV_TOOL_BIN_DIR` are set per generation by `_generation_env:1416`) | `lib/pythonX.Y/site-packages/local_operator-<v>.dist-info`, `.lop-source` at the install root (`update.py:774`, `:867`) | the install's own front end: `lop update` (`update.py:3943`) | not supervised. Started by `lop`, by the desktop app's `ownedServeLaunch`, or by hand | not applicable to the *install*: an install is idle when no process has it open, which the generation layout makes true by construction | none — the running build is in another tree |
| 1b | **backend, uv tool — the legacy fixed tree** (`~/.local/share/uv/tools/local-operator`) | `uv tool install --force … local-operator` — what the host script at `~/.local/bin/lop-update` runs (its line 291), the in-place rewrite `docs/design-install-generations.md` was written about | dist-info plus `.lop-source` at the tool root; **live: 0.56.11 = `5766653f9`** | uv. **`uv tool upgrade` is the wrong command for it** when it was built from a snapshot — the module says so at `update.py:3206-3207` and re-installs with `--force` instead — so its legitimate updater is whatever its owner runs: `uv tool install --force …` (`installer_argv`, `:3223`), i.e. the host script `lop-update` | not supervised; **`~/.local/bin/lop` resolves THIS tree today** — uv's own receipt records `install-path = "/Users/damian/.local/bin/lop"` | n/a | **rewrites the tree in place under every live runtime** — the 36-session incident |
| 2 | **backend, pipx** | `pipx install local-operator` | dist-info under `~/.local/pipx/venvs/local-operator` | `pipx upgrade local-operator` (`update.py:3223` argv table) | as #1 | as #1 | rewrite in place: `update.py:3329-3335` documents that pip/pipx still rewrite `site-packages` under the fleet |
| 3 | **backend, ordinary pip venv** | `pip install` into a venv or a toolchain base prefix (`update.py:615-635`) | dist-info in the prefix | the package manager that owns the prefix | as #1 | as #1 | as #2 |
| 4 | **backend, editable checkout** | `uv pip install -e` from a repo | dist-info frozen at install time; `direct_url.json` says editable | nobody but the developer (`editable_refusal` `update.py:3247`) | as #1 | as #1 | n/a — the app must not touch it |
| 5 | **backend, app-managed env** | the desktop app builds `managed-python/packaged/environments/<id>-<uuid>` and publishes `selected-environment.json` (`managed-python.ts:880-954`) | dist-info in that venv (**live**), `environment-ready.json` + the pointer's `backendVersion` (**frozen at preparation**, `managed-python.ts:932-944`) | **the app** — it owns the tree | the app spawns `ownedServeLaunch` (`owned-serve-launch.ts:315`) or adopts a live record; nothing supervises it across app restarts | its in-flight probe empty (`retire.py:122-207`): no SSE subscriber, no websocket, and no desktop bridge holding a request, a watch lease or a warm. **The app's own relay stream is a STANDING term with no turn boundary and no TTL**, released only when the client lets go — see §5.1, which is where that fact has to be handled rather than assumed away | today: `pip install --upgrade` in place under ~16 live runtimes, then a restart that drops in-flight turns (`update-service.ts:4441-4470`) |
| 6 | **desktop app** | signed DMG / electron-updater; installed as `/Applications/Local Operator.app` | `CFBundleShortVersionString`; staged artifact sha512 checked before install (`update-service.ts:1832-1858`) | electron-updater, plus a watchdog that survives a quit mid-install (`update-service.ts:1358-1447`) | the OS; the app's own ShipIt job after download | n/a | app quits and relaunches; live turns on *detached* session runtimes survive (see #10) |
| 7 | **mobile daemon (and there is no separate relay)** | `lop mobile install` → LaunchAgent `com.local-operator.mobile` (`mobile/install.py:35`, `:110`) | none of its own: the plist names the **stable shim**, so its build is the install the shim resolves at start (`launchd.py:524-553`) | `lop update` (via `refresh_mobile_after_upgrade`, `update.py:3495`) or `lop mobile restart` | launchd / systemd user unit, keep-alive; the launcher is the **stable shim**, `argv[0]` is the branded label | no attached phone client holding an SSE stream, no in-flight control op | drops every phone stream for the restart window (a few seconds); a phone mid-turn sees a reconnect |
| 8 | **browser-bridge daemon** ("extension workers") | `lop browser install` → `com.local-operator.browser` (`browser_bridge/install.py:38`) | as #7 | `lop update` (only when the unit's content changes, `update.py:3626`; `unsupported` on systemd by design, `browser_bridge/install.py:185-191`) | launchd / systemd user unit | no extension connected, no command in flight | the extension loses its socket and reconnects; a command in flight fails |
| 9 | **tunnel (cloudflared connector)** | `lop tunnel install` → `com.local-operator.tunnel` (`tunnels/install.py:15`); the **binary is external**, resolved from PATH and required `>= 2025.4.0` (`tunnels/service.py:29-41`) | the plist names the stable shim; cloudflared's own version is whatever the user's package manager installed — the connector runs with `--no-autoupdate` (`tunnels/service.py:285`) | the harness owns the unit; the *binary* is owned by whoever installed cloudflared (Homebrew) | launchd / systemd user unit; the connector + loopback gateway are supervised as one unit (`tunnels/service.py:1`) | nothing to defer: a connector holds no session state | the public URL goes dark for the reconnect window (phone sees Cloudflare 1033 while it is down — measured on this project, `launchd.py:394-398`) |
| 10 | **session runtimes** (one process per session) | spawned on engage, detached from the launcher — a sample taken here: 59 live, of which 37 re-parented to pid 1, 16 children of the serve daemon, 4 of another. Re-parenting is NOT what makes them independent; the control socket and their own published record are | the process publishes its own record: version + `source_ref`, plus a boot record and a turn journal (`process.py:652-700`) | themselves: they self-retire onto the new build (`process.py:776-930`) | **not supervised** — a successor is spawned by the next engage, a wake, or a peer message | `_should_exit` (`process.py:570-602`): no work in flight, no wake within `WARM_WINDOW_S`, no interactive viewer attached | never kill: `lop refresh` (`cli.py:5083`) asks instead, and a busy runtime retires at its own boundary |
| 11 | **serve daemons** (the "central session server") | started by `lop serve`, by the desktop app, or by a legacy fixed-port probe. **Its install is the tree it was launched from** — on a converged host that is the pointer, `disk_build()` (`update.py:1328-1374`) | the serve record: `version`, `source_ref`, `prefix`, `install_kind`, `instance_id` (`registry.py:261-287`); the record's `version` is a **boot** reading, `/health` is live | nobody, today: it announces (`retire.py:210`) and never leaves | launchd covers none of them; the desktop app re-attaches after a restart via discovery | `in_flight()` (`retire.py:122-207`): SSE subscribers, websocket connections, and the desktop plane's bridges/leases/warm tasks. Deliberately **not** a complete work-safety predicate — scheduler-owned work is not counted (`retire.py:9-14`) | a restart is invisible to session runtimes (they re-attach) but cuts the app's event stream and any phone stream for the restart window |
| 12 | **wakes supervisor** | installed on demand by the wake persist path (`ensure_supervisor_installed`, `wakes/install.py:336`) → `com.local-operator.wakes` (`wakes/install.py:67`) | as #7 | `lop update` (only when the unit changes) | launchd / systemd user unit | no wake due within `WARM_WINDOW_S` (`buildwatch.py:171`) | a due wake is deferred; the wake store is on disk |
| 13 | **browser extension** | Chrome Web Store (staged review → publish, `AGENTS.md` "Releasing the browser extension") | `extension/manifest.json` `version` (live: **0.1.17**); `extension/package.json` tracks it | the Chrome Web Store's own update channel; the harness submits and promotes via workflow | the browser | n/a | the extension reconnects to #8 on its own |

**Supervision is launchd on macOS and a systemd user unit on Linux** —
`browser_bridge/install.py:38-39` (`LABEL`, `SYSTEMD_UNIT`), `:116-129`
(`systemd_unit()`, `systemd_path()`), and each sibling installer's own renderer.
The two platforms are NOT symmetric in one way that matters here: the
"repair the unit" half is macOS-only by construction, and the code says so —
`refresh_plist_if_stale` returns `kind="unsupported"` when the supervisor is not
`launchctl`, because "the systemd unit is re-read on every start and has no image
name to go stale" (`browser_bridge/install.py:185-191`). So on Linux there is
nothing to repair and still everything to *restart*: a unit is read at start, so
a flipped pointer reaches a systemd daemon on its next start exactly as a launchd
one does. Every "plist" in this document means "the unit for that platform", and
§7's PR 4 must cover the systemd restart path too or say plainly that it delivers
nothing on a Linux host.

**Rows 1 and 1b are two trees on this machine, at two versions** (0.56.9 and
0.56.11) — the divergence §1.6 is about, and the reason the document never says
"the global install" in the singular.

Ownership, as a rule rather than a table: **the installer that created a tree is
the only thing that may write it.** A `uv tool`/pipx/pip tree belongs to the
package manager that made it; the app-managed environment belongs to the app;
an editable checkout belongs to nobody but its author; the supervised units
belong to the install that rendered them (`_repair_refusal`, `update.py:3745-3795`,
already enforces exactly this for the plist repair).

## 3. The version that is true

Four different questions are being asked, and today all four are answered with
whichever number was nearest.

| the question | the authoritative reading | where it is read | what it must never be |
|---|---|---|---|
| **what is published** | PyPI's `info.version` | `check_latest()` (`update.py:436`) — the only live network read; `cached_latest()` (`:470`) is the offline fallback and never fetches | a version anybody has not fetched |
| **what is installed on disk** | the dist-info name plus `.lop-source` under an install root — `installed_build(root)` (`update.py:867`) | `installed_build(prefix)` for a named root. For **the pointer** — the tree a daemon launched through `<stable>/bin/python3` loads — `disk_build()` (`:1328-1374`), which reads `current_install_root()`. Its docstring calls that "the build a FRESH `lop` would load", and that holds **only on a converged host**: here `~/.local/bin/lop` names the legacy uv-tool tree, so `disk_build()` answers 0.56.9 while `lop --version` answers v0.56.11 (§1.6) | `/health` (it is a *running* process's reading) |
| **what is running right now, and from which install** | `installed_build(process_install_root())` for the build — but the identity is `prefix` + `install_kind` + `pid` + `instance_id` from `/health` (`health.py:90-99`), cross-checked against the serve record (`registry.py:238-329`) | `DaemonStatusSnapshot.prefix` / `.installKind` already carry it to the app (`local-operator-ui:src/shared/backend-status.ts:73-74`) | a PATH lookup, a version string alone |
| **what the next launch will load** | for a generation install: the generation `current` resolves to (`current_generation()`, `update.py:1197`; `flip_pointer()` `:1547` is the only write) — which for a *supervised* daemon is also what it loads **on its next start**, because its unit names the shim (`daemon_image_path()` `:1140-1153`) and the shim resolves once at exec; a daemon already running is still on the generation it started with (§1.6). For the app's env: the venv the pointer names (`readManagedSelection` `local-operator-ui:src/main/backend/managed-python.ts:613`). For a session runtime: whatever `local-operator` resolves at spawn | `installed_version()` / `lop --version` for the *reading* process; `disk_build()` for the pointer | the boot stamp of a live process |

Four consequences the design depends on:

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
3. **Every component's subject has to be named as one of the two readings.**
   A session runtime's install is its own `sys.prefix` (`installed_build`);
   a supervised daemon's is the **pointer** (`disk_build`), because its unit
   names the shim; the app's is `/health.prefix`. "The daemon's install moved"
   is therefore never one sentence for the fleet, and §6's report carries the
   reading it used beside each row. A design (or PR) that compares the console
   script's reading on behalf of a daemon is comparing a different tree — §1.6.
4. **A check may affirm "up to date" only about an install it positively read
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

## 4. App-owned, legacy uv tool, or somebody else's

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
the wrong install, so the app would still resolve the legacy uv-tool tree
while the app-managed one serves, and the operator's second requirement
("update the backend … without taking down any runtimes") would still route
through `pip install --upgrade` into a live tree. It leaves the button able to
name the wrong command.

**B. Resolve from `/health`'s `prefix`, and act per kind.** Recommended. The
readings are on the wire today; no new endpoint is needed for the desktop. It
keeps the one existing gate that matters (`canManageUpdate` = generation layout
only, `update-install.ts:3098`) and it makes the app-owned case explicit rather
than accidental. Note that this gate is **already false on this host** — the
resolved console script has no `generations` element above it — so the act-able
half of this option is not available until §1.6's convergence lands; §6.2 says
what the interim must render.

**C. Move the whole decision into the backend and have every client ask it.**
A `/v1/update/status` route returning the serving install's build, the disk
build and the published version, with one rule implemented in Python. Tempting
for TUI/phone parity, and rejected *for now* because it inverts the dependency
for no current client: the TUI already answers this correctly for itself
(`/update` runs `check_latest()` against its own install — `_check_for_update`
at `tui/app.py:13198` calls it at `:13203`, and `_run_update` at `:13246` calls
`check_latest(force=True)` at `:13264`), the phone has no update surface, and the desktop already holds the
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
  **This pointer is the app's own and is not `~/.local/share/lop/current`**
  (§1.6): the two layouts converge in *shape*, not in path, and they are read by
  different processes.
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

**One class takes an in-place variant of that shape, and it is the `serve`
daemon.** The shape above assumes the consumer must LEAVE and be replaced, which
is why successor readiness is its hard part: a daemon that exits hands its port,
its record and its desktop claim to a process that has to be proven ready first.
A daemon does not have to leave. It can hand its **listening socket** to the new
build's interpreter and come back as the same pid, with the same cwd and the same
environment (`local_operator/server/reload.py`), at which point (1) and (4) are
fused, there is no successor to prove, and (3) has no meaning — nothing is being
left. What that trades away is stated in §5.1's row: steps (2) and (3) do not
happen for the daemon, so the app's standing relay is CUT rather than released,
and it re-attaches through its own reconnect path. The design accepts that cost
deliberately; it is recorded as a cost rather than as a recovery that happens to
work.

### 5.1 Per component class

| class | can it be updated in place? | mechanism | what the user sees meanwhile |
|---|---|---|---|
| uv-tool install (generations) | no — and that is the feature | `lop update` builds a new generation and flips `current` (`update.py:2038`, `:1547`). **The subject is the pointer, not whatever `lop` on PATH resolves** (§1.6) | nothing on the install side; running sessions keep working |
| session runtimes | no, and never killed | idle → `_refresh_for` (`process.py:862`); busy → `_begin_drain`/`_drain_for` (`:962`, `:1188`), which stops admissions and leaves when the turn ends | the viewer gets a `retiring` frame and re-engages onto the successor; no `⊘ interrupted` is painted (`docs/design-runtime-autorefresh.md` §4.1) |
| app-managed env | no (today it is, and that is the bug) | new venv generation + pointer flip (§4.4) | the running daemon keeps serving the old tree; Settings shows both numbers once §3 lands |
| `serve` daemon, whichever build started it | its process image, not its install | **requested in-place replacement**: `lop update` (or `lop services restart`) asks, the daemon drains, and it `execve`s the pointer's interpreter with `-P`, keeping its pid, its **listening socket**, its cwd and its environment (`server/reload.py`; `services.py` owns the fleet-wide request). The request is `SIGUSR1`, so the capability is **authored by the daemon that has it** — the record's `reloadable` field — because that signal's default disposition is to terminate. Fail-closed: no pointer, no listener of its own, a drain that does not empty inside its budget, or a request to move onto the build it is already running all leave the daemon serving what it loaded. **This supersedes the earlier "never bounced by the app" rule for this class, deliberately** (operator decision, 2026-09-18): the alternative was a machine whose install was current and whose backend stayed on the build it booted, with the skew named and nobody to act on the name — which is the reported defect. The standing objection was *standing* ("the app has no more standing to restart it than it has to write its tree"), and it is answered rather than ignored: nothing external kills, owns or respawns anything, because the daemon moves ITSELF on a request from the operator's own update command | the app's stream drops and reconnects onto the same pid and port. **Accepted cost, stated:** the relay is cut rather than released — §5's steps (2) and (3) do not happen — so the app's own reconnect path does the recovery; and the drain deliberately does NOT gate on daemon-owned `SchedulerService` work, which no probe can report |
| app-owned serve daemon (the successor handover route, not taken) | n/a | **land → announce → the app releases its own relay → the daemon's probe empties → it latches and leaves → the app re-attaches to the successor.** Each step is load-bearing and the order is the code's, verbatim: "the announcement has to precede the drain, because this stream is only released when the client decides to" (`retire.py:145-159`, `desktop_sessions.py:2474-2481`). The app's `GET /v1/desktop/sessions/{id}/events` relay is a **standing** term — no turn boundary, no TTL — so the release is a step only the CLIENT can take, and §6 makes it part of the action's contract rather than assuming it | the app's stream drops and reconnects onto the successor. A turn in flight on a *session runtime* is untouched: those are separate processes with their own control sockets (§2 row 10), so they are not a term in the daemon's retirement at all |
| ~~adopted/foreign serve daemon~~ | — | **Superseded** by the `serve` daemon row above. This class used to be left to a person — "announce in the record, state the skew, name the exact command, and do not act" — and that is exactly the state the reported machine was stuck in. The in-place reload reaches it without anyone needing standing over the process | the skew is acted on, at the next request, by the daemon itself |
| mobile daemon | n/a | bounce at idle (no phone stream, no in-flight op); today it is bounced unconditionally | the phone reconnects |
| browser bridge | n/a | bounce at idle (no extension attached, no command in flight) | the extension reconnects |
| tunnel | n/a | bounce, but as a *stated* cost: the public URL is dark for the reconnect window | the phone shows a tunnel error for a few seconds |
| wakes supervisor | n/a | bounce at idle, i.e. no wake due within the window | a due wake is deferred, never lost (the store is on disk) |
| extension | n/a | the Web Store's own channel; nothing for this mechanism to do | browser-native |

### 5.2 A runtime that never goes idle

There is already a bound, and it is worth restating carefully because it is the
answer to the hardest part of the requirement. The idle gate (`_should_exit`,
`process.py:570-602`) is *soft*; the belt is not:
`BUILD_MAX_STALE_GENERATIONS = 3` and `BUILD_MAX_STALENESS_S = 30 * 60.0`
(`process.py:154-155`). A runtime that keeps declining a settled newer build
trips `hard_stale` (`:378-381`), which drains: it stops admitting work, finishes
its own turn, and leaves — "nothing in flight is ever aborted by either path"
(`:760-767`).

**The bound is on the commitment, not on the departure**, and the repo says so
in its own words: `session/runtime/types.py:406-411` — "NO BOUND IS NAMED …

`process._drain_for` waits for this runtime's work and nothing else (the build
path draws no clock), while the signal path is cut by `SIGNAL_DRAIN_S`". A
runtime can legitimately be busy for hours with nothing but a turn
(`process.py:150-153`). So the honest sentence is: **within 30 minutes of the
build settling, a stale runtime stops accepting new work, and it leaves when its
own turn ends — which may be much later. Nothing is killed either way.**
`lop refresh` (`cli.py:5083-5103`) is the same thing on demand, with `busy`
reported as a queued move rather than a failure (`control.py:1572-1594`).

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
* **A daemon hosting sessions** is the one place a plain restart is not enough,
  and the reason is the *relay*, not a turn. Session runtimes are separate
  processes with their own control sockets and their own published records — a
  sample here: 59 live, 37 re-parented to pid 1, 16 children of the serve
  daemon; re-parenting is incidental, the socket is the point — so a daemon
  restart is survivable *for them*. What a restart cuts is the app's own event
  stream and any phone stream, and the app holds its stream until it decides to
  let go (`retire.py:145-159`). That is why an *exit-and-replace* daemon's terms
  would be an announcement, then the client's release, then a drain.

  **A RELOAD does not have to ask, and that is the change of position recorded
  in §5.1.** The client-release ordering exists because a leaving daemon needs
  its successor to be ready and its clients to have let go. A daemon that keeps
  its pid and its listening socket needs neither: the app's stream is cut, and
  the app reconnects — measured on the reporting host, where it re-read the
  record, re-claimed the plane and re-attached (`Claimed the desktop plane on
  http://127.0.0.1:1111.` → `Attached to daemon … (pid 61225, v0.59.0, uv-tool)`)
  with eight live runtimes untouched. The design's earlier sentence — "the
  design keeps the app out of a daemon a person started" — is withdrawn for this
  class on the operator's own decision (2026-09-18), because the remedy it named
  (state the skew, name a command, act never) left the reported machine on
  0.56.14 with a *current* install and no route out.

### 5.4 One gap this leaves, stated rather than smoothed over

The four supervised daemons name the **stable shim**, so neither a pointer flip
nor a unit rewrite restarts them — and because the shim resolves `current` once
at exec, a running daemon is pinned to the generation it started with, so a
restart is the only thing that moves it. The daemon refresh is a no-op when the
rendered unit is unchanged (`launchd.py:682`, `:688-700`;
`mobile/install.py:187-199`; the systemd renderer behaves the same way). This is
why §1.5 items 3 and 4 are on the PR plan rather than left implicit: a correct
`lop update` that leaves the mobile daemon, the extension workers and the tunnel
serving the generation each started on for an unbounded time is not "properly
updated" in the operator's sense, however clean the pointer flip was — **and
§1.6 shows that is the state of this machine today**, not a risk.

**Landed 2026-09-24 (PR #1515): the refresh asks a second question.** After the
plist comparison comes the RUNNING daemon: the pid launchd holds for the label
(`launchd.job_pid`) and the generation that process's own argv names
(`update.stale_generation_of_process`, reading `ps -o args=` — the shim `exec`s
the generation's own image, so argv carries the generation and not a proxy for
it). A daemon whose build provably moved is restarted with a `kickstart -k`, not
a `bootout`/`bootstrap` pair: nothing about the unit changed, so the in-memory
definition is exactly what wants re-executing, and it names the shim. Measured on
the operator's machine that day — four byte-identical plists, two daemons on the
current generation and two two-and-three generations behind — this moved the
wakes supervisor and the browser bridge and left the tunnel and the mobile daemon
alone. Every unreadable answer (no `ps`, a dead pid, an argv naming no generation,
no readable pointer) leaves the daemon where it is, and a machine without the
layout is not probed at all.

What this closes, concretely: a released fix now reaches a running daemon without
the hand-run `lop tunnel restart` the v0.62.22 release note had to carry — the step
whose absence let the tunnel connector serve a pruned generation for three days.

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
| `awaiting-client-release` | the build landed; this component is held by a stream **this action cannot release**, and the detail names the holder (the app's own daemon relay when the app is that client; a phone-held SSE stream when it is not) | new, and §5.1's standing terms are why it must exist |
| `not-checked` | **nothing was read about this component** | new, and the point of it |

Two of these are load-bearing. `not-checked` is the rule that a surface may not
affirm what it did not check, already written down for a whole check
(`update-check-verdict.ts:1-22`) and made concrete per component: a component the
app cannot see (no mobile daemon installed, no tunnel configured) reports
`not-checked`, and the summary sentence may then say nothing about it. The
forbidden sentence is not "an error occurred" — it is **"up to date"**, which
today can be printed over a component nobody asked.

`awaiting-client-release` exists because §5.1's app-owned daemon cannot be
bounced by a predicate the app's own relay makes false: the action must release
that relay (`retire.py:145-159`), and until it has, the honest state is "waiting
on someone", not "deferred-until-idle" (which would blame the daemon) and not
`could-not` (which would blame the installer). The holder is named in the detail
because the app is not always the one holding it: a phone on the same daemon's
SSE stream (§2 row 11, §5.3) is a standing term this action cannot release at
all, and the state has to say so rather than report a wait on us. What the action
bounds is **its own wait, not the class**: it waits a stated window for the
release, and if the release does not come the result says so and the daemon keeps
serving. The class's bound is the client's release — which is exactly why the
window has to be stated rather than assumed.

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
* **It must release what it holds.** For an app-owned serve daemon the app is the
  client holding the standing term (§5.1); an action that announced and then
  waited for a predicate it was itself keeping false would never finish. The
  release is the app's own step in the contract, and `awaiting-client-release`
  is how a failure to take it is reported rather than hidden.

**The button is off on the reporter's own machine today, and that needs saying
plainly.** `canManageUpdate = generationInstallRoot(identity) !== null`
(`update-install.ts:3098`) is decided by walking up from the *resolved console
script*. On this host that path is
`~/.local/share/uv/tools/local-operator/bin/local-operator`, which has no
`generations` element above it, so `generationInstallRoot` returns null and the
gate is **false right now**; the app renders the legacy arm — it names
`lop update` and refuses to run it (`update-install.ts:3125-3127`). The tree that
*does* have the shape is `~/.local/share/lop/generations/…` with
`<stable>/current` present. So the clickable-update half of the operator's
requirement stays unavailable until `~/.local/bin` is relinked through `current`
— which is exactly what §1.6's convergence step does, and what
`lop install migrate` (`update.py:2846`) is for by hand. PR 6 must render this
state honestly (the command, the reason, and the convergence as the remedy)
rather than shipping a disabled button with no explanation.

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
| 2 | local-operator-ui | resolve the backend update plan from the **serving install** (`/health.prefix` + `install_kind`, cross-checked with the serve record), make app- vs externally-owned a fact about the tree, and **name which reading is the subject** — `disk_build()`/the pointer for a daemon, the serving prefix for the app — never the console script's tree (§1.6) | this is the change that makes the *plan* right; separating it from PR 1 means each can be reviewed on its own evidence and PR 1 can ship first |
| 3 | local-operator-ui | app-owned environment updates by **publishing a new generation** (prepare → install → smoke → pointer flip) instead of `pip install --upgrade` in place | it is the only PR that writes to a live tree's neighbour; it needs PR 2's ownership resolution to know when it applies |
| 4 | local-operator | supervised daemons take the new build **without a unit rewrite**: the refresh path restarts a daemon whose *install* moved, where "its install" is **the pointer the shim resolves** (`disk_build()`), not the console script's tree — at its idle boundary, on launchd **and** systemd, and reports it | this is the `update.py`/unit half of §1.5 items 3–4; it is language-separate from PRs 1–3 and independently testable (fake units, fake `launchctl`/`systemctl`). Without the two readings named apart it restarts a daemon onto the build it was already loading (§1.6) |
| 5 | local-operator | `serve` daemons take the new build **in place**: a requested reload per daemon (`SIGUSR1`, capability published in the record) that keeps pid, listening socket, cwd and environment, driven fleet-wide by `lop services restart` and finishing `lop update` — **superseding** the announced exit-and-re-attach handover this row used to specify | **Delivered differently from the original scope, by operator decision (2026-09-18).** The hard part named here — successor readiness — is dissolved rather than solved: a daemon that does not leave has no successor to prove. The cost it accepts instead is that the app's standing relay is cut rather than released (§5.1), which the app already recovers from; and `SchedulerService` work is not gated, recorded as a known limit. Scope is **not** app-owned only, which is the sentence this amendment retires |
| 6 | local-operator-ui | the one-click `updateAll()` action, its per-component report, and the consent-taking convergence it owns (the third note below) | last, because it is the surface over results PRs 2–5 produce; building it first would freeze a contract against unbuilt behaviour |

Four notes the table cannot carry:

* **The PR 2 → PR 3 gap.** After PR 2 the app-owned env is the subject and the
  plan has no command for it (§4.4 keeps the per-install front end for
  non-app-owned installs), so for one PR the app would report "an update is
  available" with nothing to press. PR 2 must therefore hold that arm at a
  stated state — report the skew, name PR 3's convergence as the remedy, offer
  no action — rather than leaving an offer with no handler.
* **PR 5 does not answer the foreign-daemon half of the requirement.** The
  operator asked that "the central servers … should all properly update"; PR 5
  moves only a daemon the app started. For a daemon a person started the deliverable
  is a stated skew plus the exact command (§5.1), and that is a deliberate choice
  — an app that restarted somebody's `lop serve` would be the same class of
  overreach as an app that pip-installed into somebody's uv tool tree.
* **PR 6 is gated on a convergence the operator has to accept — and PR 6 is
  what owns offering it.** Until `~/.local/bin` resolves through `current`,
  `canManageUpdate` is false on this host and the button is legitimately off
  (§6.2). PR 6's honest rendering of that state — command, reason, remedy — is
  what makes the interim acceptable; a disabled button with no explanation is
  not. **No other PR in this plan owns the step that closes the loop**: §1.6(iii)
  puts the convergence *inside* the action this gate blocks, and §7.1 leaves
  fleet-wide migration to a rollout, so the owner is named here rather than left
  to the reader. PR 6 must offer the convergence as a consent-taking action, not
  only print it: run `lop install migrate` (`update.py:2846`, idempotent by
  outcome, refused only for a source checkout) through the install that currently
  owns `~/.local/bin`, then re-check. Until that runs, the clickable-update half
  of the operator's requirement stays unmet on a host like this one, and the
  report says so instead of implying otherwise.

* **PR 4 depends on a generation tree that nothing but the installer writes.**
  Its restart path goes through the stable shim into a generation's own `bin/`
  (§9's last risk), and the shim prefers the branded image planted there whenever
  that path is executable (`update.py:1173-1174`). This host writes into live
  generations today — 167 orphaned plant temps, and a branded image that came and
  went inside four minutes — so PR 4 states the invariant it needs in the PR, and
  its QA runs on a root no test run has written into.

### 7.1 The split I would change

The task statement says PR 1 fixes the reported defect by itself. I agree, with
one qualification: **PR 1 must not try to fix the plan's subject.** The
temptation is to make the check compare the serving install in the same PR
("while we are here"), which would (a) put a second decision path into a PR
whose whole value is that it is one rule, and (b) produce a check that says
"0.56.8 is behind 0.56.11" and then offers `lop update` — a command that, run
from today's PATH `lop`, updates the legacy uv-tool tree (and, on this host,
does nothing at all, §1.6) while leaving the app-managed env at 0.56.8. That is a
worse lie than the current one, because it is actionable and wrong. The subject
must move (PR 2) *before* the check can be allowed to name a remedy.

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
  needed here: `globalUpdateInFlight` (`update-service.ts:4040-4053`) serialises
  the desktop's own attempts, and the tool-side writer reserves its generation
  directory exclusively (`_reserve_generation`, `update.py:1392-1413` — an
  `os.mkdir` without `exist_ok`, so two installs in the same second cannot aim
  uv at one path).
* **Windows supervision** — this document's supervised-daemon half is launchd
  (macOS) and a systemd user unit (Linux); Windows has neither of these install
  paths (`browser_bridge/install.py` and the sibling installers guard on
  platform). PR 4 lands for the two it covers; a Windows supervisor, if it is
  wanted, is its own piece of work.
* **Converging the two update paths for every existing install** — §1.6 names
  the problem and the mechanism (`write_stable_launchers`, `lop install
  migrate`), but a fleet-wide migration of machines in the wild is a rollout
  question, not a design one. The design's obligation, met in §6.2, is to render
  the unconverged state honestly rather than pretend it is converged.

## 8. Test plan

Unit-first, because every rule above is a pure decision:

* **PR 1** — table-driven cases over `(runningVersion, installedVersion, app,
  published)`: the operator's exact triple (0.56.8 running / 0.56.11 install /
  0.56.11 published) must **not** earn the affirmation and must set
  `serverStillOlder`; `("", "0.56.11", …)` and `isReadableVersion("Unknown")`
  stay `unavailable`; the 558-version corpus in `update-check-verdict.ts`'s
  docstring (`:52-63`) must still be accepted. Prove the test can fail: revert the rule and
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
* **PR 4** — fake units, both families: a unit whose rendered content is
  unchanged but whose *install* moved — where "its install" is the **pointer the
  shim resolves**, so the fixture must give the daemon a pointer tree that
  differs from the console script's tree (§1.6) — must be restarted; an
  unchanged install must not be; a restart failure must surface the recovery
  command (`launchd.reload_failure`, and the systemd equivalent). A test that
  compares the console script's reading here would pass on the broken machine,
  which makes this the discriminating case for the whole of PR 4.
* **PR 5** — the reload, asserted on the RIGHT terms. The old list assumed an
  exit and is kept only where it still bites: an unsupervised daemon must not
  exit without a successor (it does not exit at all here), and a refusal must be
  typed and carried rather than silent. What the in-place mechanism must prove
  instead, and what the tests in `tests/unit/server/test_serve_reload.py`
  assert: a daemon whose record says it CANNOT reload is **never signalled**
  (the request's default disposition is death); a same-version rebuild is not
  read as a no-op (only `source_ref` moves on this host); the successor keeps
  the fd, the interpreter and `-P`; every refusal path leaves the daemon serving
  what it loaded; an established connection is CUT and a dialling client is not
  refused — the second is the property a stop-and-start cannot have, and the
  end-to-end rig measures both (`82 dials, 0 refusals`). A *session runtime*
  mid-turn is a different process and must be shown to be unaffected, which the
  same rig does by counting the runtimes either side.
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
* **Two readings quietly becoming one again.** The convergence problem (§1.6)
  is easy to re-introduce: any code that resolves "the install" from one place
  and compares it against a version read from another will look right on a
  converged machine and lie on this one. The guard is that every comparison in
  PRs 2/4/6 carries the root it read, and the test in §8's PR 4 fixture keeps
  the two trees deliberately different.
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
* **A generation tree written by something other than the installer — the
  assumption PR 4's restart path rests on, with a live counterexample.** PR 4
  restarts a supervised daemon through this chain: the unit →
  `<stable>/bin/python3` (the shim) → `current`, resolved once at exec → `exec
  <generation>/tools/local-operator/bin/{Local Operator | python3}`
  (`update.py:1140-1176`). Two links are outside PR 4's control — the shim, and
  the generation being a tree only the installer has written. The second is an
  assumption, and it is not literally true even by design: the harness plants its
  own branded image into `<venv>/bin` on first use (`ensure_branded_interpreter`,
  `procname.py:563`, link-to-temp then `os.replace`), so PR 4 has to tolerate that
  writer. The invariant worth stating is that nothing *else* writes there. On this
  host something else does, or did today:
  `<stable>/generations/*/tools/local-operator/bin/` holds **167** mode-`700`
  `.Local Operator.<pid>.tmp` entries (85 in the 0.56.2 generation, 21 in 0.56.6,
  61 in 0.56.9 — up from the ~60 counted earlier today), which is the residue
  `_plant_hardlink` leaves when a process dies between its `os.link` and its
  `os.replace` (`procname.py:502-510`). The shipped sweeper (`:457-486`) removes
  only temps whose embedded pid is already dead, and 22 of these 167 now resolve
  to a live process (pid reuse), so the debris accumulates rather than being
  reaped; a hardlink carries the source inode's mtime, so none of it can be dated
  from the filesystem. This is not only untidiness: the shim's first branch execs
  `<generation>/tools/local-operator/bin/Local Operator` **whenever that path
  exists and is executable** (`update.py:1173-1174`), and today that image is
  absent from all three generations — so a writer into a generation's `bin/`
  decides which image the *next* restart runs, i.e. it can change what PR 4
  restarts *onto*. The same class appears in a shim spawn aborting on `dyld:
  Library not loaded: @rpath/libpython3.14.dylib` from a
  `pytest-of-damian/garbage-<uuid>/popen-gw2/…` path, and in an executable that
  existed in one of those directories at 15:31 and was gone by 15:35 today. This
  document records the dependency rather than diagnosing it (a separate agent owns
  that investigation): PR 4 should carry the invariant it needs, and its QA must
  run on a root no test run has written into.

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
  answer: it bounds the *commitment*, not the departure — a stale runtime stops
taking work within that window and still leaves only when its own turn ends
  (`types.py:406-411`). A "force update now"
  button that kills a runtime mid-turn would be the 2026-09-14 signal sweep
  again, wearing a UI.
* **Not updating the extension from here.** The Web Store owns that channel, and
  a local installer for a store-managed extension is a second, unsigned update
  path for code that runs in the user's browser.
* **Not making the tunnel's bounce silent.** A public URL that goes dark for a
  reconnect window is a fact the user must see before they click, because the
  cost falls on their phone, not on this machine.
