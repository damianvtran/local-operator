# Design: session runtimes survive the UI, and are re-adopted when it comes back

Status: **proposal (architect)**. Scope: two repos — `~/local-operator`
(Python: `serve`, session runtimes) and `~/local-operator-ui` (Electron/TS).
No version bump; the release owner handles that.

This is outage #3 of 2026-09-15. The other two outages are other streams and are
deliberately **out of scope here**: the `lop-update` install tearing
site-packages under busy runtimes (generation-directory install + pointer flip),
and RAM. Nothing below proposes a change to install layout, to `lop-update`, or
to the residency/memory policy.

Every file:line was read on this machine tonight, mid-incident, read-only. No
suite was run and no Electron app was booted (other lanes hold the box).

---

## 1. The problem as I found it — and two corrections to the timeline

The requirement, stated plainly: **closing or restarting the UI must not end the
work running under it, and reopening the UI must show that work as it is** — a
session that finished, finished; a session that was interrupted, interrupted and
named as such.

### 1.1 What the code does today (verified)

- A runtime is spawned by whoever engages the session, and is **detached but
  still a child**: `subprocess.Popen(..., start_new_session=True)`
  (`session/runtime/launch.py:342-362`). `start_new_session` gives the child its
  own session *and* its own process group; it does **not** reparent it.
  Confirmed live: every runtime on this box has `PGID == PID`, and all of them
  have `PPID` pointing at one other process.
- The runtime's lifetime, socket and discoverability are **already independent
  of the spawner**: it binds its own loopback control socket
  (`process.py:683-684` `RuntimeServer(handle, kind="daemon")` +
  `start_in_process()`), publishes its own record
  (`session/runtime/registry.py:61-79`, `<pid>.json` under `run/mobile`), holds
  its own transcript lease, and decides its own exit (`process.py:265-296`
  `_should_exit`, reaper at `:330-359`).
- A viewer leaving does **not** end a turn: `_should_exit` tests `is_busy()`
  first and alone (`process.py:289-291`), and the viewer-side contract says the
  same in prose (`session/attached.py:4574`: "Do NOT end the turn here. A
  dropped socket says nothing about the turn").
- **Re-adoption already exists** and needs no new mechanism: `engage_runtime`'s
  first step is "a live record? deliver over its socket and return. The common
  case" (`launch.py:20-31`), with the transcript lease arbitrating construction
  (`launch.py:655-670`). Anything — the app's backend, a phone, `lop send`, the
  wake supervisor — that engages a session with a live runtime *attaches to it*
  rather than spawning a second one.
- **The discovery half of re-adoption is already shipped.** `~/.local-operator/run/serve/`
  holds live records tonight (`71125.json`, 0.55.8, `install_kind: pip`, `desktop: true`)
  plus a `reaped/` subdirectory for stale ones, and the UI reads exactly those
  (`src/main/backend/discovery.ts`). `run/serve` does not appear in *this
  worktree's* Python source, which sits on an older branch — so the UI-side
  details below must be read against shipped behaviour, not against this checkout.
- The app's quit path stops **its own** backend generation:
  `index.ts:2049-2096` (`will-quit` → `backendService.stop(false)`) →
  `backend-service.ts:1891-1901` → `stopGeneration` (`:1982-2022`), which is
  SIGTERM → grace → SIGKILL → grace and **throws if exit is unconfirmed**. It
  signals only `generation.child` (`canSignal`, `:1964-1970`), i.e. the pid it
  spawned itself.
- The `pkill` family the earlier design pass documented (design-daemon-discovery
  §1: `pkill -9 -f python` and friends) is **gone from the tree** — `grep` over
  `src/main`, `src/preload`, `src/shared` finds four signal sites and every one
  is scoped: `owned-serve-launch.ts:160-161` (the interpreter probe's own child),
  `backend-installer.ts:524` (the installer's *own* detached group, spawned
  `detached: true` at `:621`), `backend-service.ts:2005/2010/2080` (the owned
  serve generation), `update-service.ts:1201` (a sentinel-verified leftover
  relaunch watchdog).
- `POST /v1/desktop/stop` is a **stop, not a drain**: `desktop_lifecycle.py:209-233`
  → `ServingSessionHandle.request_stop` (`serving.py:1181-1219`), whose
  documented ordering denies parked gates, **aborts an in-flight turn**, flushes
  the transcript, releases the lease and exits.
- The desktop bridge pool's teardown on the backend's lifespan
  (`server/app.py:137-140` → `utils/desktop_sessions.py:237-241` →
  `Bridge.close` → `_detach` → `remote.dispose()`) closes a **socket**; it does
  not stop a session. That distinction is load-bearing and currently untested
  (§8, §6).

### 1.2 Correction one: the app did not restart at 19:41, so no quit path ran

The established ground truth offered the app's quit/restart path as the class of
cleanup to remove. That is right as a *class*, and the design below removes it —
but it is **not** the 19:41 mechanism, and a design built as if it were would
fix the wrong night:

```
  PID  PPID  PGID STARTED                     COMMAND
70141     1 70141 Tue Sep 15 19:28:36 2026    /Applications/Local Operator.app/.../Local Operator
71125 70141 70141 Tue Sep 15 19:29:11 2026    .../managed-python/.../bin/python -c ... serve --port 1111
98105 71125 98105 Tue Sep 15 19:43:22 2026    Local Operator [session] id=ee3ef11e -m local_operator.session.runtime.process
```

The app (70141) and its owned backend (71125) were started at 19:28:36 and
19:29:11 and were **still the same processes** at 19:53. A `will-quit` that ran
at 19:41 would have stopped 71125 via `stopGeneration` and a restart would have
replaced 70141. Neither happened. The quit path is a real hazard (it is the only
code in either repo with a *tree* relationship to a backend, and
`owned-serve-launch.ts:332` explicitly rejects descendant sweeps as "the class of
cleanup this change exists to remove"), but it did not fire at 19:41.

### 1.3 Correction two: the fleet died hard, and it is a chain, not a star

Two facts from the live box and the logs:

- **The fleet has one parent, and that parent is itself a session runtime.** All
  26 runtimes alive now have `PPID 98105`, which is
  `Local Operator [session] id=ee3ef11e`. That is ground truth #7 seen from the
  other end (`lop send` wakes a session as the child of the *sending* runtime):
  the tree is app → backend → runtime → runtimes. A per-runtime failure is
  therefore also a *tree* failure, and today's "one agent woke thirty others"
  shape means one death is a fleet-wide event.
- **The 19:41 deaths wrote nothing.** `~/.local-operator/logs/runtime.log` is
  present and carries 19:43:48–19:45:56 lines from the successor fleet, so the
  file was writable and would have carried 19:41 exits had any been writeable.
  There are no such lines: not `exiting (SIGTERM, …)`, not
  `idle for 3.0s … exiting cleanly`. A SIGTERM would have logged
  (`process.py:695-705` handler → `:777-782` log). So the fleet died **hard**
  (SIGKILL-class), which is the only way this exit path leaves no trace.
- No memory pressure kill: a query of the unified log for
  `memorystatus`/`jetsam`/`lowswap` across 19:40–19:44 returned only ordinary
  `runningboardd` noise for Chrome/node/`osascript`, no process kills.

Leading hypothesis, stated as a hypothesis: **a machine-wide stop sweep**. The
kill switch's `stop_all` targets "Every OTHER agent on THIS machine"
(`session/runtime/control.py:867-885`, `stop_all` at `:897`, `lop stop --all` in
`cli.py:672`, described at `:3730` as able to "end a dozen agents"), and it is
the one existing code path whose *documented purpose* is to signal every record
in the run directory at once. Its ladder's hard rung is consistent with the
silence. I have not proved it, and I am not proposing to chase it: §4 makes that
class of event survivable and *observable*, which is worth more than the
attribution.

What would settle it: an exit record produced by a supervising parent (§5 — the
kernel tells the parent `SIGKILL`, which is exactly the fact the fleet could not
record about itself), plus the `lop stop` receipt / `attention.db` rows for
19:41 and any `/stop` invocation in a neighbour session's transcript.

---

## 2. The constraint that decides the design

Three properties are already true and must not be traded away:

1. **Work is authoritative over presence.** `is_busy()` is checked first and
   alone (`process.py:277-291`); an attached viewer does not hold a runtime
   against work, and a leaving viewer does not end a turn.
2. **Exit is the runtime's own decision.** `_should_exit` + the reaper
   (`process.py:265-359`) are the residency policy, and the idle-reap design
   (`docs/design-idle-reap.md`, §Decision) already settled *who may initiate*:
   the viewer, by closing its socket, with the runtime's own unchanged predicate
   deciding the consequence. Nothing here reopens that.
3. **Nobody outside may end a session.** `request_stop` is the user's kill
   switch, not a lifecycle tool (`serving.py:1204-1210`), and the operator's rule
   is that no design may require stopping a live session to work.

The gap is not in the runtime's design. It is in **who spawns it and who can see
it die**: today that is a viewer, a backend, or another runtime — every one of
which is a process that the app's restart, an install, an OOM, or a kill sweep
can take away, and none of which can report the loss afterwards.

---

## 3. Deliverable 1 — the lifetime-ownership model I recommend

**A session runtime owns itself. No app is ever its owner. The only party outside
a runtime that may spawn one or signal one is a supervised host process (§4),
and even that may not end a session — only a person may.**

Three roles, and the distinction is the whole design:

| role | who | may spawn | may signal | may end a session |
|---|---|---|---|---|
| **owner** | the runtime itself | n/a | n/a | yes — its own reaper, and SIGTERM handled cleanly |
| **spawner / observer** | the session host (§4); today: the app's backend, another runtime | yes | only the children it spawned, on its own exit | **no** |
| **viewer** | the Electron app, the TUI, the phone | no | **nothing** | no — it may *ask* (the user's `/stop`), and only a user asks |

The UI's ownership is therefore **narrow and nameable**: it owns exactly the
`serve` generation it spawned (`backend-service.ts:161` `this.process`,
`canSignal` at `:1964-1970`), and the four signal sites listed in §1.1. It owns
no runtime, holds no runtime pid, and has no vocabulary in which "clean up my
processes" can mean a runtime.

### What must NOT be reachable from the app's quit path — named

1. **`stopGeneration()` applied to a daemon the app did not spawn.** The guard
   exists (`isExternalBackend`, `backend-service.ts:163/1236`, and
   `stop()`'s `isExternalBackend` check) and the read-side predicate is written
   and **has no caller**: `mayManageDaemon()` (`backend/daemon-status.ts:229`,
   documented at `:34` as "the one predicate with no caller yet"). Give it its
   callers and make every kill site conditional on it.
2. **Any group kill.** `process.kill(-pid, …)` stays legal *only* at
   `backend-installer.ts:524`, whose target is that installer's own detached
   group (`:621`). No new `-pid` call in either repo; a lint-style test can pin
   the count (§8).
3. **Any name/pattern sweep.** The `pkill`/`pgrep -f python` family is already
   deleted (design-daemon-discovery §6) and must not return in any spelling,
   including "kill the children of the app".
4. **`POST /v1/desktop/stop` from any lifecycle path.** Quit, restart, update,
   health-recovery and build-retirement must never call it: it aborts an
   in-flight turn by contract (`serving.py:1184-1191`). It is reachable only from
   an explicit user action, with the user's own confirmation.
5. **The backend's own shutdown reaching a session.** Today the desktop pool's
   teardown only closes sockets (§1.1). Pin that with a test, because
   "`dispose()` closes a socket" and "`dispose()` ends the session" are one
   refactor apart and the second one turns every backend restart into a
   fleet-wide stop.
6. **"Stop the backend on quit" as the default.** Recommend adopting
   design-daemon-discovery §6's own conclusion and *finishing* it: a quit leaves
   the daemon it started **running**, closes the viewer's leases, and offers an
   explicit "Stop server" action. This is the smallest change that removes the
   last honest reason for quit-path escalation, and it is already written down as
   the destination in that design — the point here is that it is not optional
   polish, it is the survival mechanism (§4).

What the UI's quit path *should* do instead is short and testable: clear the
health timer, dispose the SSE relay and the desktop bridges, close the viewer
record/endpoint (`index.ts:2065-2066`), and stop only the generation it owns.

---

## 4. Deliverable 2 — the survival mechanism

### Options

- **(A) Do nothing structural; rely on POSIX reparenting.** A runtime whose
  parent dies is reparented to launchd and keeps running. *Rejected as a whole
  design* — it is true, and it is not sufficient: it protects against a parent's
  death by accident, but not against a signal (the app is one keystroke away from
  a sweep, and the 19:41 event shows signals are what actually happen), not
  against a spawn landing in a window where the loader's paths are being
  replaced (the DYLD class — a runtime must not depend on the app's environment
  being stable), and it answers nothing about *observing* a death.
- **(B) Decouple the runtime's socket and lifetime from the backend.** *Already
  true* (§1.1: own socket, own record, own lease, own exit predicate). Nothing to
  build; the value of naming it is that it is the reason re-adoption needs no new
  protocol.
- **(C) A hand-off protocol at quit.** The app hands its viewer leases to a
  successor. *Rejected*: it invents a second owner, and a hand-off can fail
  halfway. With (B) already true, the app has nothing to hand off — it only has
  to stop *taking*.
- **(D) Spawn outside the app's tree, under a supervised host that owns the
  death signal.** **Recommended.**

### Recommended: (D), in two steps, sized separately

**Step 1 (small, and it is tonight's fix) — the app stops being able to reach a
runtime, and stops stopping the daemon.**

- Every kill site is gated on `mayManageDaemon()`; `POST /v1/desktop/stop` is
  unreachable from lifecycle code; no group kills and no sweeps (§3).
- Quit leaves the owned daemon running (design-daemon-discovery §6), with the
  explicit stop action as the opt-in.
- A spawn is pinned to a **stable install**: the interpreter a runtime is born
  from must not be the app's own bundled, replaced-in-place venv. The mechanism
  already exists — `interpreter.SAFE_PATH_FLAG`, the branded argv0, and the
  `exec`-shaped launcher in `owned-serve-launch.ts:315-323` — what is missing is
  that a runtime's birth interpreter is chosen by the *spawner's* `sys.executable`
  (`launch.py:336-340`), and the spawner may be the app. The manager's DYLD
  evidence (`@rpath/libpython3.13.dylib`, "terminated at launch", clustering in
  the app's environment churn) is exactly the population this kills: a spawn that
  lands in a tear-down window dies at load and writes nothing, because
  `logging.basicConfig` has not run yet — which is the same anonymous-death shape
  as 19:41 and must not be confusable with it.

This step is **prohibition plus one resolution change**. It is small, it is
reversible, and it is what the stated problem ("closing the UI kills my
sessions") actually needs.

**Step 2 (the survival destination) — a session host that owns spawning, and
therefore owns the death signal.**

A single machine-wide process, launchd-supervised, **the only party permitted to
spawn a session runtime**, so that:
- a runtime's parent is a process designed to outlive the app, and no viewer,
  backend or sibling runtime is ever named in a runtime's ancestry;
- a hard death is **observable rather than inferred**: the parent can `wait()`,
  so `SIGKILL`/`SIGSEGV`/exit status is a fact the host writes down. This is the
  one thing today's architecture structurally cannot do, and the reason the
  reference investigation could not attribute a single one of tonight's deaths;
- rescue has somewhere to live (§5): the host is the party that notices a
  session with an open turn and no runtime.

**Why a new unit and not an existing one.** Both existing supervised units are
candidates and neither fits as-is:
- the **mobile daemon** (`com.local-operator.mobile`, `mobile/daemon.py:1983-1999`)
  already states the correct principle — "a session living inside it would die
  with every restart … a child with its own pid gets the same lifetime as a
  terminal session" — and already adopts every session. But it is a *phone*
  feature; making the desktop depend on the portal inverts the dependency, and it
  is documented as able to run in observer mode, i.e. not always able to spawn.
- the **wake supervisor** (`com.local-operator.wakes`, `wakes/install.py`) is
  launchd-supervised and already spawns runtimes on schedule — but it is
  **install-on-demand and self-retiring** (`KeepAlive: {SuccessfulExit: False}`,
  `SELF_HEAL_INTERVAL_S = 900`): a machine with no wakes runs no supervisor, so it
  cannot be the always-available spawner.

So: a new `com.local-operator.sessions` unit, **install-on-demand in the same
shape the wake supervisor already established** (installed on the first runtime
spawn, `KeepAlive` for crash recovery, exiting when no session is live and no
viewer is attached, so a fresh machine still runs nothing idle). Single-instance
election by lock in the config dir, because two hosts is the first failure mode
(§7). Non-macOS keeps today's direct spawn as the documented degradation, exactly
as `wakes/install.py` does for Linux.

**What breaks it, named honestly:**
- *A host that dies with sessions live.* Its children survive (`start_new_session`
  — verified, `PGID == PID` on every runtime today), and the successor adopts by
  record. Cost: the death signal is lost for that window, which is why the
  **exit record must also be written by the runtime at boot** (§5) as a
  belt-and-braces "I existed, and here is the turn I was carrying".
- *Two hosts.* Lock + record-election; the second refuses to spawn and serves as
  viewer only.
- *Nothing is reparented by the host.* `wait()` needs parentage, so the host must
  be the direct parent — double-fork would defeat the whole point. That is the
  one place this design deliberately differs from the usual "daemonise" advice.
- *A host on a stale build.* Reuse the settlement vocabulary that already exists
  (`BUILD_CHECK_S`/`BUILD_SETTLE_S`/`BUILD_STAGGER_S`, `process.py:84-101`, and
  the daemon-retirement shape in design-daemon-discovery §7) rather than inventing
  a second one; the host retires like the daemon does, with the successor adopted
  by record.

---

## 5. Deliverable 3 — background continuation

**Mid-turn when the UI closed: it must finish, and today it already does — the
requirement is survival, not new machinery.** A viewer's socket closing does not
abort a turn (`process.py:289-291`; `attached.py:4574`), the turn's terminal
event flushes the transcript through `handle.dispose()` at exit
(`process.py:299-327` `_clean_exit`), and the durable turn-end checkpoint
(`frontend_state.py:1716` `last_turn_outcome`) is written by the session. So a
turn that outlives every viewer completes and is durable, *provided the process
survives*. Step 1 of §4 is what makes that true by construction rather than by
luck; Step 2 is what makes its failure visible.

**Three additions I do recommend, all durable-state, none of them lifecycle:**

1. **A turn journal row, written by the runtime at turn start and closed at turn
   end.** Fields: `turn_seq`, `command_id`, `started_at`, pid, build stamp, and
   the last completed tool boundary. This is the *positive evidence* the existing
   taxonomy demands but cannot currently obtain: `attention._classify_orphaned_run`
   (`session/attention.py:368-431`) can only reach "the record's pid is dead"
   (`runtime-killed`) or nothing at all (`CUT_OFF_UNKNOWN`), because a stopped
   process leaves no statement about its own turn. An **open journal row** is a
   statement: *turn N was in flight and never ended*.
2. **A continuation errand, delivered on the next engage — never an automatic
   re-run.** The operator's own instruction to the model is already "do not
   assume the request completed" (`incidents.py:246-249`), and auto-replaying a
   turn that already executed tool calls duplicates side effects. So: the next
   engage of a session with an open journal row (the app re-attaching, the user
   sending a message, a wake firing, the host's bounded retry) delivers a
   `ContinuationErrand` alongside the user's work, in the additive `Errand`
   family at `launch.py:125-201`. It carries the interruption notice below and
   lets the agent choose to resume or restart — the decision belongs to the agent
   and the person, not to a reaper.
3. **The interruption notice, injected into the agent's own context.** This is
   the peer's point and it is the right one: a session resumed after a kill must
   be able to tell an interruption from a completion **in its own state**, not
   merely in the UI's. The notice is a transcript row the successor writes at
   boot: which turn, when, how the previous incarnation died (host-provided:
   `SIGKILL`, exit status, or "unobserved"), and which boundary it last completed
   — so an agent that "remembers" finishing step 3 is told that step 4's state is
   unknown before it acts on that memory.

**Idle when the UI closed: it exits, and that is correct.** A viewer-less runtime
quits ~3 s after going idle by design (`DEFAULT_GRACE_S = 3.0`, `process.py:63-67`;
`_should_exit` term 3, `:242-262`). I am **not** proposing to change that: it is
the deliberate memory policy, the RAM outage is another stream's, and the cost of
the alternative is ~283 MB per idle session (`process.py:75`) against a ~1.2 s
cold start (`launch.py:47-49`). What brings a session back is unchanged and
already implemented: a wake inside `WARM_WINDOW_S = 90` (`process.py:69-78`), a
peer message, a new engage from the viewer, or the app's re-adoption dial when the
user opens it — all through `engage_runtime`'s record/lease arbitration
(`launch.py:20-31`). The consequence to state in the UI: *an idle session the user
left is expected to be cold; opening it costs one cold start, and it shows its
durable history immediately* (the cold facade already exists — `attached.py:1289`
"A viewer bound to NOTHING: durable history and a spool, no runtime").

---

## 6. Deliverable 4 — UI restart: re-adoption, re-attach, resume marker

**Re-adoption needs no new protocol.** On start the UI discovers the daemon from
`run/serve/*.json` with a pid-liveness check and an `instance_id` identity match
(`src/main/backend/discovery.ts`, `HEALTH_PATH`/`CLAIM_PATH`, `readServeRecords`,
`probeIdentity`; state machine in `daemon-status.ts`), claims the desktop plane,
and lists sessions from the daemon. Attaching a session with a **live** runtime
is `engage_runtime` step 1 — the dial to the record's socket — so the UI attaches
to the runtime that survived rather than spawning a rival, and the lease makes
that arbitration safe for concurrent engagers (`launch.py:655-670`).

**Sessions the UI never had a tab for** are not a special case: the session list
is durable session directories, not tabs. Two things are worth adding, both
cheap:
- the UI should mark, per session, **live vs cold** from the record's existing
  fields (`busy`, `started`, `detached`, `waiting_for` — `runtime/types.py:266-291`,
  explicitly additive and protocol-preserving), and
- it should restore the tabs it had open last time, so "the sessions that were
  running" come back as tabs rather than as a list the user has to re-find.

**The resume marker.** The durable half exists and should be reused, not
duplicated: `last_turn_outcome` in the frontend checkpoint
(`frontend_state.py:1716`), the Stop marker
(`attention._stopped_marker`, `:317-335`), the run-record evidence
(`:272-316`), and the classifier that turns them into a verdict
(`:368-431`). What is missing is the **runtime-authored** half — §5.1's open
journal row — because every one of the existing signals is a *viewer-side
reconstruction*: today a session resumed after a kill shows a verdict derived
from a dead record, which is exactly why `runtime-killed` is the usual answer and
`CUT_OFF_UNKNOWN` the answer when nobody reaped the record first
(`attention.py:398-408` names that exact failure). With an open journal row the
verdict is a fact: *the previous incarnation died with turn N in flight, last
completed boundary B*, and the agent is told before it acts.

What the user sees after a kill-then-restart, therefore: the transcript exactly
as durable, the cut-off notice with a **named** cause (and, with §4 Step 2, a
cause that can say `signalled` rather than "disappeared"), the reason rendered by
the existing taxonomy (`incidents.CUT_OFF_CAUSES`, `:285-303`), and the session
ready to continue with the interruption already in context.

**What the user sees while the backend is gone** is already designed and should
be reused verbatim: `/health` + instance id is the only liveness signal, three
consecutive failures before `detached`, `degraded` never restarts anything,
`detached` always re-discovers before starting, an external daemon is never
restarted or replaced, a capability refusal never reads as "down", and a wedged
record (live pid, stale heartbeat) is degraded-and-named, never reaped
(`daemon-status.ts` header, `discovery.ts`). The one addition: the banner should
distinguish **"daemon is gone; your sessions keep running"** from "your server is
offline" — because after this design those are different facts, and the second
one is no longer implied by the first.

---

## 7. Deliverable 5 — files, states, and failure modes

### Backend (`~/local-operator`)

| file / function | change |
|---|---|
| `session/runtime/launch.py` `_spawn_runtime` (`:259-368`) | spawn through the host when one is live; resolve the birth interpreter to a stable install rather than the caller's `sys.executable` (`:336-340`); register the spawn with the host so the child is the host's, not the caller's |
| `session/runtime/launch.py` `engage_runtime` (`:495+`) | unchanged arbitration; gains the `ContinuationErrand` path (§5.2) |
| new: session-host module + `wakes/install.py`-shaped installer | single-instance lock, spawn admission, `wait()`-based exit records, install-on-demand plist `com.local-operator.sessions`, retirement on build change |
| `session/runtime/process.py` `amain`/`main` (`:653-845`) | write the boot "I existed" record and the open/closed turn journal rows; keep the existing `exiting (<reason>, pid …)` line (`:777-782`) as-is |
| `session/runtime/registry.py`, `types.py:230-300` | exit-record file next to `run/mobile/<pid>.json`; any record addition is additive (`types.py:255-265`), `PROTOCOL_VERSION` unchanged |
| `session/attention.py` (`:368-431`) | classify from the journal row when present, preferring it to the dead-record rung — positive evidence over inference |
| `server/app.py` lifespan (`:73-152`) | publish/deregister the daemon with the host; shutdown still only closes sockets, never sessions |
| `session/runtime/control.py` (`:867-922`) | `stop_all` stays, but its receipt must name the actor and the rung per target (the 19:41 audit trail) |

### UI (`~/local-operator-ui`)

| file / function | change |
|---|---|
| `src/main/index.ts` `will-quit` (`:2049-2096`) | stop only the generation it owns; never stop an external/discovered daemon; keep the viewer record/endpoint close (`:2065-2066`); no `desktop/stop` |
| `src/main/backend/backend-service.ts` `stop`/`stopGeneration`/`canSignal` (`:1891-2022`, `:1964-1970`) | every signal conditional on `mayManageDaemon()` |
| `src/main/backend/daemon-status.ts` `mayManageDaemon` (`:229`) | give it its callers (it is written and unused) |
| `src/main/backend/discovery.ts` | unchanged; the attach path already exists |
| new: "Stop server" action + "daemon gone, sessions still running" banner copy | the explicit replacement for quit-path escalation |
| renderer: live/cold per session from record fields; restore open tabs | re-attach the sessions that were running |

### New states and records

- **`run/mobile/<pid>.json`** (existing) — unchanged shape, additive fields only.
- **Exit record** (new, written by the host; and a boot-time "I existed" sibling
  from the runtime itself): pid, session id, build, `started_at`, `exited_at`,
  `status` (exit code) or `signal`, and `observed_by` (`host` | `runtime`).
  This is the artifact tonight's incident lacked.
- **Turn journal row** (new): `turn_seq`, `command_id`, `started_at`, `open` /
  `closed_at`, `last_boundary`, pid, build.
- **Host record** (new): `run/host/<pid>.json` — pid, lock, build, `serving_since`,
  in the shape `session/runtime/registry.py` already provides
  (`run_dir`/`record_path`/`publish`/`RecordPublisher`, `:36-210`), **in its own
  namespace** — never `run/mobile`, for the reason design-daemon-discovery §2
  already gives: every reader of that directory treats a record as a session.

### Failure modes

| failure | what happens | what stops it |
|---|---|---|
| two hosts | both spawn; a session gets two runtimes | config-dir lock + the **transcript lease** already arbitrates a second candidate into a designed loser (`launch.py:23-30`) |
| double engage (UI + phone + wake at once) | one runtime, N deliveries | unchanged: the record-then-lease arbitration; `command_id` idempotency (`launch.py:540-543`) |
| orphaned runtimes nobody owns | they finish, flush, and exit by residency policy | they own themselves; the host adopts by record on sight; → host state is *advisory*, never required |
| a turn that outlives every viewer | it completes and is durable; the session then goes cold | §5: journal row + continuation errand on the next engage |
| host dies with a live fleet | children survive; death signal lost for that window | runtimes' own boot record + the successor host's record scan |
| a stale host build | new spawns mixed across builds | reuse `BUILD_SETTLE_S`/`BUILD_STAGGER_S` + retirement-by-record |
| the UI holds a stale daemon port | wrong-runtime attach | `instance_id` identity match; already implemented |
| a sweep happens anyway | fleet dies hard, silently | the exit record makes it *visible*; making it *impossible* is a separate, harder claim — see the risk below |

---

## 8. Deliverable 6 — test plan

**Unit (backend, `.venv/bin/python`, whole tree as CI runs it):**
- the turn journal opens on turn start and closes on every terminal path
  (completed / aborted / error / deliberate stop) — one test per path, because
  the taxonomy's whole value is that "no evidence" means *error* and not
  *interrupted* (`attention.py:373-383`);
- the boot record is written before the control socket listens, so a
  die-at-load spawn and a vanished fleet are distinguishable;
- `_classify_orphaned_run` prefers an open journal row to the dead-record rung;
- a viewer disconnect mid-turn does **not** abort the turn (pin
  `process.py:289-291` from the outside, at the attach-client boundary);
- host spawn admission: two hosts cannot both spawn for one session id;
- `_should_exit` and `DEFAULT_GRACE_S`/`WARM_WINDOW_S` are **unchanged** — a
  guard test so this design cannot quietly become a residency change.

**UI (Electron-free, driven directly — `discovery.ts`/`daemon-status.ts` are
written for this):**
- `mayManageDaemon()` gates every kill site: assert *no* `kill` is issued for an
  external daemon on quit/restart/health-recovery, modelled on
  `scripts/owned-serve-lifecycle.test.mjs`, which already pins that the quit
  gate lives in the synchronous part of the listener (`index.ts:2118-2129`);
- a source-level invariant test: the number of signal sites is exactly the four
  in §1.1, and `process.kill(-` appears exactly once (the installer's own group);
- quit-path contract: `POST /v1/desktop/stop` is never issued.

**The app-level proof ("quit the UI mid-turn, the turn still completes, restart
the UI, the session is there").** This cannot be one test in one repo, and saying
so is the honest part. Three pieces, and together they are the proof:
1. **Backend e2e** (`tests/e2e -m e2e -n0`, the established assembled-app stage):
   engage a session with a real turn through a real tool, attach a viewer, kill
   the **viewer** process, assert (a) the turn reaches a terminal event, (b) the
   transcript and the turn-end checkpoint are durable, (c) a second engage
   **adopts the same pid** (no new spawn) — the assertion that proves re-adoption
   rather than re-spawn;
2. **UI-side quit test** against a fake external daemon with a fake child: quit
   the app, assert the daemon and the child are alive and unsignalled, and that
   the app's own log says "left running";
3. **Operator acceptance on the live host** (scripted, *not* run unattended by an
   agent): start a turn in one session, quit the app, watch that session finish,
   relaunch the app, confirm the session appears with the result — with a
   synthetic config dir, `env -u CMUX_WORKSPACE_ID` and every other `CMUX_*`
   unset (the documented hazard: an inherited workspace id renamed the operator's
   real cmux workspaces), and targeting **one session id** — never a fleet-wide
   pattern kill, on a box that holds ~26 live runtimes.

**Evidence that would be convincing on the live host:** `ps -o
pid,ppid,pgid,lstart` snapshots of the fleet before and after the quit (proving
the same pids, same pgids, same start times — i.e. untouched); the runtime's own
`exiting (…)` line for the session under test in `~/.local-operator/logs/runtime.log`;
the exit record for any runtime that did die, with `signal` filled in; and the UI's
daemon-state transitions in its log. This is the evidence tonight's incident could
not produce, which is the point of §5.

**Instrumentation as an acceptance criterion, not a nicety.** Run the exit-record
path for a day and count deaths by cause. That count is also the experiment that
settles §1.3's hypothesis — and it should gate any Step-2 work, on the principle
that building a supervisor against an unknown cause is how you end up with two
answers to a question nobody asked.

---

## 9. Deliverable 7 — what I am NOT proposing to change

- **The residency policy.** `_should_exit` and all three of its terms,
  `DEFAULT_GRACE_S = 3.0`, `WARM_WINDOW_S = 90`, and the reaper's cadence
  (`process.py:63-78`, `:265-359`) stay exactly as they are. `docs/design-idle-reap.md`
  already decided who may initiate a reap (the viewer, by closing its socket) and
  that decision is untouched here.
- **Build-age retirement.** `retiring`, `_should_refresh`, `BUILD_CHECK_S` /
  `BUILD_SETTLE_S` / `BUILD_STAGGER_S`, and the eager re-engage on the refresh
  frame (`docs/design-runtime-autorefresh.md`) are untouched; the "retiring" frame
  keeps its single meaning, and I am not overloading it with "your viewer left".
- **`request_stop`'s contract.** A stop is still a stop: deny gates, abort the
  turn, flush, release, exit (`serving.py:1181-1219`). Only its *reachability*
  from lifecycle code changes.
- **`lop stop` / the kill switch.** The user's ability to end a machine's
  sessions is preserved; `stop_all` is not removed, only made accountable
  (per-target actor and rung in the receipt).
- **design-daemon-discovery.md's record, claim, health-identity and update-plan
  work.** Not re-designed. Its §6 conclusion (leave the daemon running; the UI
  kills only pids it spawned; delete the sweeps) is *completed* here, not replaced.
- **The mobile daemon's adoption model.** Untouched, and explicitly **not** made a
  dependency of the desktop.
- **The other two outages.** Install tearing (`lop-update`, generation-directory
  install) and RAM are other streams; nothing here changes install layout, the
  `.lop-source` settle signal, or the memory budget.
- **The wire protocol.** No `PROTOCOL_VERSION` bump; the new records and fields
  follow the additive contract at `types.py:255-265` and the additive-op bar at
  `mobile/types.py:242`.

Deliberately rejected, so nobody re-derives them: auto-rerunning an interrupted
turn (duplicate side effects); a second supervisor beside the mobile daemon (two
adopters is the failure mode, not the fix); making the phone portal required for
desktop sessions; a viewer heartbeat to keep runtimes warm (the idle-reap design
already rejected it — a viewer that says "I am alive" is exactly what defeats an
inactivity reaper, and the viewer already knows the answer locally).

---

## 10. Risks I would watch during rollout

1. **The default disposition of the owned daemon on quit.** Leaving it running is
   correct and is also a visible change: a user who quits expecting the machine to
   be quiet now has a daemon and (maybe) live runtimes. Watch: the "Stop server"
   action must be discoverable, and the daemon's idle footprint must be stated in
   the UI, not just in a design note.
2. **Spawn-interpreter resolution.** Choosing a stable install over the app's
   bundled venv touches the one thing that must never be wrong (a runtime running
   the wrong code silently — AGENTS.md's editable-venv trap). Watch: the boot
   build stamp in the record and in the exit record, and a test that the resolved
   interpreter is the same install the daemon runs.
3. **Unattributed deaths remain unattributed until the exit record ships.** Until
   then, a repeat of 19:41 is still anonymous. Sequence Step 1's instrumentation
   first for that reason.
4. **`wait()`-based observation is a hard dependency of the host design**, so the
   host must be the direct parent (no double-fork). Watch: any "let's daemonise it
   properly" change to the host is a regression against its purpose.
5. **The kill switch remains able to end the fleet in one command.** This design
   makes its consequences visible and survivable-by-recovery; it does not make it
   impossible. If the operator wants that, it is a separate decision (confirmation
   surface, per-user scoping, or a receipt-gated `--all`) and I am not smuggling it
   in here.
6. **Compressed-timeline pressure.** The temptation after a third outage is to
   ship Step 2 tonight. The evidence in §1.2/§1.3 says the cause is not yet known;
   Step 1 is the part that is justified by what we *do* know.

---

## 11. Suggested split (for the manager's sequencing)

- **PR A (backend, small):** turn journal row + boot record + classifier
  preference + the tests that pin them. No lifecycle change. This is the
  instrumentation every later decision wants.
- **PR B (UI, small):** `mayManageDaemon()` callers, no external-daemon stop on
  quit, no `desktop/stop` from lifecycle code, the invariant test over signal
  sites, the "daemon gone, sessions still running" copy, the explicit "Stop
  server" action.
- **PR C (backend, medium):** spawn-permission + spawn-interpreter resolution
  (§4 Step 1's second half).
- **PR D (backend, large):** the session host, install-on-demand, exit records,
  retirement by record — **gated on PR A's day of exit-record data**, and its own
  design review once that data exists.

Each PR needs the full gate: agent review round on the MR, QA round on the same
head, and a design round for B (it is user-visible). No PR carries a version bump.
