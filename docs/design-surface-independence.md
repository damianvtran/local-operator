# Design: a runtime's life does not depend on any surface

Status: **proposal (architect)**. Scope: `~/local-operator` (backend); the desktop app is named only
where a contract changes. No version bump.

Supersedes the *spawn-ownership* half of `docs/design-session-survival.md` (2026-09-15). That note's
diagnosis holds and its step 1 (the app stops being able to reach a runtime) is still right; its
step 2 — a launchd-supervised session host as the only party allowed to spawn — was designed and
never built (`com.local-operator.sessions` appears nowhere in the tree but that document). This note
re-derives it against measurements taken since, decides what the first pass left open, and adds the
half now measured to be missing: **a spawner is not a defence.**

Evidence is tagged **[M]** measured (here or in the incident record this note was commissioned
with), **[C]** read in the code at the cited `file:line` on `origin/main` = d0281fa91bf, **[I]**
inference — implied by the above, unrun.

---

## 1. The problem, in one measurement

2026-10-30 21:57:52 -0400: 12 background exec workers, ≥5 session runtimes and 2 `lop serve` daemons
took SIGTERM within 1.33 s when the operator force-quit a frozen desktop app. Nothing staged a stop
for any of them, no `logs/stop-sweeps.jsonl` exists anywhere, and the victims wrote
`cause=runtime-shutdown` receipts. **[M]**

The natural experiment on the same host at the same moment decides the direction: runtimes spawned
by the launchd-owned wake supervisor (`com.local-operator.wakes`) **survived**; everything that died
descended from the app (`cli.py` ← session runtime ← `lop serve` ← `Local Operator.app`). **[M]**

Two earlier fixes are already in the tree and neither is sufficient, which is why this note is not a
third variation on them: every runtime is spawned with its own session and group
(`start_new_session`, `launch._spawn_runtime`) **[C]**; and the macOS responsibility disclaim
(`macos_disclaim.spawn_disclaimed`) severs the responsibility chain, is **active on this host**
**[M]**, and did not prevent this event — it fixed the 2026-10-09 force-quit sweep and this one went
around it.

The excluded-by-experiment list matters as much: `kill -9` of a launchd job's main pid does not
sweep detached children, `launchctl kill SIGKILL` of a runningboard-submitted app job does not
either, and the actor is *not* the OS reacting to app death. **[M]** So the sender is a party that
**knows the pids** — a tree walk or an enumeration of the discovery/job stores — and the surviving
supervisor is a *measured* discriminator rather than a theory: a spawner whose children are in
neither the app's tree nor its store.

The requirement, in the operator's words: closing or force-quitting the UI must never terminate a
runtime; runtimes must be independent, reparented away from the app, discovered through the session
store rather than owned by whichever surface launched them; a surface dying is a display event.

---

## 2. Who spawns

### Options

**(A) Extend `com.local-operator.wakes`.** One fewer unit. Rejected, three independent reasons, all
readable in the code:

1. *It self-retires.* `KeepAlive: {SuccessfulExit: false}` plus exit-0-on-empty
   (`wakes/install.py:342-346`) **[C]** means a machine with no scheduled wakes runs no supervisor
   at all: it cannot be the always-available spawner, and making it always-on deletes the property
   the unit was designed around.
2. *Its stop semantics are the inverse of what is required.* On stop it forwards SIGTERM to the
   runtimes it spawned (`wakes/owned_runtimes.py::sweep`, written for #3028) **[C]**. A routine
   supervisor restart would become a fleet stop — exactly what rule R3 forbids. Extending the job
   means surgically removing that sweep, which removes the reason the module exists.
3. *Import weight.* The job's justification is being import-light. Measured today here: importing
   the spawn path (`session.runtime.launch`) costs **417.5 ms and 190 modules** against **82.1 ms
   and 109** for `wakes.store` **[M]**. Every supervisor start would pay that.

**(B) A sibling launchd job — the session host — install-on-demand, single instance,
launchd-owned.** **Recommended.** The installer shape (`wakes/install.py::render_plist`, `:315`),
the "installed means running" repair (`supervisor_state`, `:448`; `ensure_supervisor_installed`,
`:635`) and the platform matrix (`is_supported`, `:167`) are reuse, not invention **[C]**. Cost,
stated: a fourth supervised unit to install and reason about, and a second installer-shaped module.

Ownership cap: the host may **spawn** a runtime and may never **stop** one. The only parties that
end a runtime are the runtime itself (`process._should_exit` and the reaper) and the stop ladder on
a person's instruction (`control.stop_session`, `:1350`) **[C]**.

### Cold start: the host is not running

An engage must not silently fail, and must not fall back to a surface spawn (that restores the bug).
A ladder with no local rung:

1. `ensure_session_host()` — bootstrap when the unit is absent, kickstart when it is
   loaded-but-stopped, reusing `wakes/install.py`'s state parse rather than a second copy of "is it
   really running" **[C]**.
2. Connect to the host socket with a bounded timeout inside the engage's own deadline.
3. On any failure, raise. `RuntimeStartupError` already carries an actionable sentence for exactly
   this (`launch.py:796-811`) **[C]**; the copy is a new keyed string (§10).

Fresh-machine ordering: the first engage pays the host's process start before the runtime's.
Cheapest mitigations first — pre-warm the host at the two places that already warm a standby
(`cli.py:17529` for `lop serve`, `cli.py:18443` for the TUI) **[C]**, a call that only *ensures
running* and never spawns from the surface; and, if the spike shows the boot cost does not fit the
engage deadline, launchd **socket activation** (`Sockets`/`SockPathName` in the plist: launchd holds
the listening fd and starts the job on first connect), which removes the bootstrap race and the
"exited successfully" hole outright. That last part is **[I]**; the spike that would settle it is
named in §9.

### Transport: a new socket, and why no existing surface works

The request cannot ride the wake index, `run/serve`'s HTTP plane, or `launchctl`: the operator
capability must travel as **a descriptor**, and `harness/approval.py` forbids it touching a file,
the environment or argv **[C]**. `run/serve` is worse still — it is the app's own backend, i.e. the
tree this change exists to leave. So: `<root>/run/session-host/host.sock`, mode `0600`, framed
request/reply plus `SCM_RIGHTS` for the capability. The framing is not new — `standby.try_adopt`
(`:1209` **[C]**) already sends exactly this request (`root`, `interpreter`, `env`, `cwd`,
`capture`, plus the cap fd by `SCM_RIGHTS`); only the transport's owner changes. (`run/host/` is the
boot-record directory, `run/mobile/` the discovery records — a new directory keeps the three
namespaces from being read as one.)

The request must carry **the requester's environment**. launchd hands the host a bare
`PATH=/usr/bin:/bin:/usr/sbin:/sbin`, which is why the wake supervisor repairs it for its own
children (`wakes/supervisor.py::main`, `setup_cross_platform_environment`) **[C]**; a host-spawned
runtime must get the engaging surface's environment or every such turn loses Homebrew, nvm, cargo,
kubectl. The standby request already carries `env` wholesale **[C]**.

The reply must carry **pid, capture path and the child's exit status**. The surface no longer holds
the `Popen`, and `_spawn_runtime`'s docstring is explicit that the spawned process's liveness is the
only sound death signal in the window before the lease is taken (`launch.py:398-413`) **[C]**.
Without a reply-carried death fact, a candidate that dies before `logging.basicConfig` degrades from
"reported with the child's own reason" to "timed out" — avoidable, and the reason the host must keep
`_harvest_on_exit`'s job.

---

## 3. No double spawn

The guarantee does not move, because it was never the spawner's: the transcript lease is taken by
the **child** (`session_lease.acquire_session_lease`, `:401`, `O_CREAT|O_EXCL` claim carrying pid +
birth token) **[C]**. Whoever forks a candidate, the loser exits; `engage_runtime`'s `_lease_holder`
probe (`launch.py:305`) is unchanged and still decides whether to ask for a spawn at all.

When the host is slow, the surface's engage waits inside its existing `deadline_s` (the loop at
`launch.py:1055-1200` is untouched) **[C]** and on expiry raises `TimeoutError`/`RuntimeStartupError`.
It never spawns locally, and it never asks twice from one engage — the negative control that pins
that is R4 in §6.

Two additions, neither load-bearing:

- **Host-side coalescing** — an in-memory "spawning since" per session, so two concurrent requests
  inside the window fork one child. An optimisation with a counter; the lease stays authoritative.
- **`EngageOutcome.spawned`** (`launch.py:283-294` **[C]**) keeps its meaning: true when this call
  caused a candidate to start, whoever forked it.

---

## 4. The operator capability

`--operator-fd` hands a 32-byte capability from the process the operator's keyboard is attached to
into the runtime, on an inherited descriptor whose *number* (not value) rides in argv; the value is
minted per spawn, registered against the child's pid in the spawner's memory, never written anywhere
(`harness/approval.py:854-867`, `open_operator_cap_handoff`, `:1415`; minted `launch.py:568`,
delivered `:706`, remembered `:715`) **[C]**. A supervisor spawn has no such descriptor.

**(a) Supervisor-held capability, surface proxies through the host.** The host mints and keeps the
cap and performs authority-increasing acts on request. Consequence: the host becomes an authority
broker on a machine-wide socket, and — since `AttachClient.authority_proof` reads the table in the
*presenting* process — the runtime would see the host, not the keyboard's process, as the presenter.
It also requires inventing a control-plane client inside the host. **Rejected**: the largest new
attack surface for the weakest meaning.

**(b) Surface mints; the child end travels by `SCM_RIGHTS` with the spawn request; the host never
reads the value.** On success the surface calls `remember_operator_cap(pid, cap)` with the pid from
the reply — the call it already makes on both existing routes (`launch.py:629`, `:715`) **[C]**.
**Recommended.** Semantics are byte-for-byte today's, "the process that engaged presents the proof".
The value's **lineage** is preserved — the descriptor exists only because the surface called
`socketpair()` — while it never touches argv, the environment, a file, or the host's memory.

**(c) Drop the authority for host-spawned runtimes.** The operator loses `/approvals auto` and
approved-card decisions from the surface, for exactly the spawn shape every session is moving onto.
**Rejected**: it trades a feature in use for a migration.

**The security consequence of (b), plainly.** The host's socket is machine-wide and same-uid
reachable — precisely the shape `standby.py`'s docstring records rejecting in review round 1 (R1-1),
where `<root>/run/standby/standby.sock` would have let any same-uid process bind first and receive
the capability **[C]**. Two differences keep (b) from re-opening that: the host is a *relay*, never
a *holder* (no capability outlives the request in flight — a design rule, testable against its API
surface); and an impostor that asks for a spawn receives a capability for **its own** child, which
it can already obtain by calling `engage_runtime` itself — so this is a tidier path to an existing
privilege, not a new one.

What (b) does **not** buy, and must not be claimed: the runtime cannot tell who asked. It cannot
today either (`operator_cap_for` answers "did *this* process spawn it", `:867` **[C]**). Requester
identity stays unauthenticated; provenance is what is preserved.

---

## 5. Reattach: a fresh surface reads the store and believes it

Read path — all existing, no new protocol:

- `roster.read_fleet` (`session/runtime/roster.py`) composes four independent sources and says which
  one knew a row: `run/mobile` discovery records (the only attachable one), `run/host` boot records
  (attribution, never liveness), the process census (`reclaim`) for a runtime whose records are both
  gone, and the socket table for a record-less runtime's port **[C]**.
- `registry.classify` (`:583`) is the single owner of `live` / `wedged` / `stale`: pid alive **and**
  beat fresh = live; pid alive, beat old = **wedged, "not answering"**, never "dead"; pid gone =
  stale **[C]**.
- `info.collect.collect_sessions` (`:276`) is the one implementation behind `lop sessions` and
  `/info`, and `session.catalog.CatalogEntry.status_code` (`:268`) is the wire field the desktop
  renderer branches on (`RUNNING_STATUS_CODES`, `:43`) **[C]**.
- The surface writes a **viewer** record (`session/runtime/viewers.py`) — a display fact, not
  ownership **[C]** — and attaching a session with a live runtime is `engage_runtime`'s first step,
  the dial, so a fresh app reattaches rather than spawning a rival **[C]**.

**What a surface must NOT do on start** (rules with owners, not style notes): no
`stop_session`/`stop_all` (`control.py:1350`, `:1912`) **[C]**; no `reclaim.reclaim_runtimes`
(`:1461`) — a residency sweep is never a startup action **[C]**; no write or withdrawal of a stop
marker; no engage of a session whose record is `live`; and no re-spawn to "rebuild state" for a
session it has no tab for (that session is an ordinary cold row).

**Stale versus live, operationally:** `scan(reap=True)` unlinks a record whose pid is gone, and the
run/exit records (`journal.py`: `runtime-run.json`, `runtime-exit.json`, boot record at `:290`) let
a STORED row carry a *named* departure reason instead of a borrowed one **[C]**. A `wedged` row is
degraded-and-named; nothing on start may reap it. The app-side contract this implies is named in
`docs/design-session-survival.md` §6 and not re-read here: live-vs-cold per row off the catalogue,
restore the tabs it had, and never treat "not in my memory" as "not running".

---

## 6. What must never happen — as rules with a test each

**R1. A surface's exit, quit or crash never signals a runtime.** Test: extend
`tests/e2e/test_terminal_close_survives_e2e.py` (real pty, three real death shapes — `ctrl-d`,
`pty-close` → a real `SIGHUP` to the group, `SIGKILL` to the group — with a dual-witnessed death
precondition **[C]**) to a **host-spawned** runtime, asserting the runtime's parent identity as well
as its survival. Unit control: a lint-shaped test that no kill site in the surface modules names a
runtime pid — precedent, the prior design's pin on the `process.kill(-pid)` count. Negative control
in §8.

**R2. Only the stop ladder and the runtime's own residency policy end a runtime.** Test: the
sanction gate (§7) — a direct SIGTERM with nothing staged does not end the runtime; the same signal
after `control.stop_session` stages its marker does; `test_stop_ledger.py` and
`test_signal_drain.py` stay green.

**R3. A supervisor restart does not kill runtimes it did not spawn this time — adoption, not
ownership by birth.** Test: unit — the host's exit path signals nothing (it has no
`owned_runtimes`-style sweep); dev-machine scripted test — start the host, let it spawn, SIGTERM the
host, assert the runtime survives and the successor adopts it by record. Supporting measurement:
`launchctl kill SIGKILL` of a submitted job does not sweep its `SETSID` children **[M]** — which is
why no sweep is needed and why one must not be added.

**R4. A host-absent engage never spawns locally.** Test: a refusing double behind the spawn seam →
engage raises a keyed error and the `Popen` count is zero (§3).

---

## 7. The failure this does not fix — and the invariant that does not need the sender

**We have not proven which party sent the SIGTERM, and this design cannot name it.** The victim's
own instrument records name, number, arrival time and whether anything was staged
(`signal_receipt.observe`, `:279`) plus a spawn-chain liveness snapshot, and says `sender:
unavailable` **[C]**. Measured here today: `hasattr(signal, "sigwaitinfo")` is **False** **[M]** —
macOS stdlib exposes no `siginfo`, so there is no sender pid or uid to be had; on Linux the sender's
uid would be this uid anyway, which distinguishes nothing.

So the defence cannot be "the spawner is different". It has to be enforced **at the victim,
independently of who asks**: *no terminating signal ends a runtime unless a durable sanction covers
that run.*

Where it lives: `process._route_signal` (`:4092` **[C]**) already has the order this needs —
receipt, then the turn's cause, then the branch. The change is to make step 3 conditional on the
receipt's verdict (`signal_receipt.cut_off_verdict`, `:420` **[C]**): a signal with **no covering
marker** is recorded and then **refused**, and the runtime keeps serving. The exec worker gets the
same rule at its SIGTERM handler (`exec_worker.py:118-160`, `:210-241` **[C]**), where today it
exits 130 unconditionally **[C]**.

Nearly free, and where it is not:

- The sanctioned senders **already** stage before signalling: `control.stop_session` writes
  `sigterm` (`:1717`), `sigkill` (`:1767`) and the socket rung (`:1283`); `update._attests`;
  `reclaim._attest_before_signal` (`:1377`), whose docstring names the same operator rule **[C]**.
- Exactly one in-tree sender does not: `wakes/owned_runtimes.sweep`, which `os.kill`s its marked
  children with nothing staged (`:190` **[C]**). It must stage `note_involuntary_stop` per candidate
  (`control.py:1031` **[C]**) in the same slice, or the wake supervisor's stop path breaks.
- No interactive capability is lost: a runtime is its own session leader with no controlling
  terminal (`start_new_session`) **[C]**, so no keystroke can deliver SIGINT or SIGHUP to one. The
  only person-shaped stop is the ladder, and the ladder stages.
- The escape hatch survives: SIGKILL is unrefusable and deliberately absent from every ladder
  (`reclaim.py` docstring; `owned_runtimes.py` "SIGKILL is deliberately absent") **[C]**; `lop stop
  --force` uses the SIGKILL rung.

**Stated limit, because claiming more would be false:** a same-uid process can write the marker
itself — the model's own `bash` tool runs as this uid, which is the premise `standby.py` records
**[C]** — so this is **attribution plus inertia, not authentication**. It converts an anonymous mass
sweep into a durable, attributable act, and it makes the measured class (an automated tree/store
walk, which stages nothing) inert by construction: such a sweep would have killed nothing on
2026-10-30. A determined same-uid sender still wins; the two things that would change that are an OS
that reports the sender (we do not have one) or runtimes under a different uid (a real architectural
change, out of scope). **That is the honest boundary of this design.**

---

## 8. Test plan: the two gates, and which one can be CI

**Gate A — a process-group kill of the surface; every runtime survives, stays reachable, and a fresh
process reattaches. CI-able, backend half.** It is the existing shape of
`tests/e2e/test_terminal_close_survives_e2e.py` extended: a real pty hosts a real TUI, engages a
runtime, runs a real turn, the group is killed in three shapes, then a **second** process attaches
to the surviving runtime and reads the live turn. In CI the host runs as an ordinary subprocess (its
`main` is a module entry point); the launchd wiring is not what Gate A tests, and saying so is more
honest than staging `launchctl` in CI.

**Gate B — several real sessions through `kill -9` of the real UI's process group. Rigged run only;
never CI.** It needs the Electron app bundle and a real desktop session, and one further fact closes
the argument: the force-quit **sweep itself** is not reproducible from a scripted context —
`macos_disclaim`'s docstring records that `LQForceQuit` killed the app instance and did not sweep
even an undisclaimed detached child in the rig **[C]**. So Gate B is an interactive, recorded run
(real app, 3+ sessions mid-turn, `kill -9` the group, relaunch, see them live) whose output is
evidence on the PR, not a regression test. CI cannot replace it and must not be reported as if it
did.

Between the two, on a dev machine (scripted, not CI): the launchd restart test in R3.

**Negative controls, required** (a passing test proves nothing until it can fail): Gate A must go
red when the spawn reverts to surface-parented — assert parent identity, not only survival; the
sanction-gate unit must go red with the gate removed; R4 must go red if a local fallback is added.

---

## 9. Slices, in order, with the riskiest assumption in each

**Slice 1 — the victim-side sanction gate and its attribution** (`_route_signal`, the exec worker,
`owned_runtimes.sweep` staging, keyed copy for the refusal). *Riskiest:* that every legitimate
sender stages before signalling. Verified for three, false for one, and unknown for out-of-tree
senders — an operator's `kill` or a hand-written script would stop working (SIGKILL and `lop stop
--force` remain). First because it is the only slice that makes the measured failure class inert
**without waiting for the fleet to turn over**.

**Slice 2 — the session host** (sibling unit, install-on-demand, single-instance lock, socket,
request/reply carrying env/cwd/capture/cap-fd and a death report; `engage_runtime` delegates with no
local rung). *Riskiest:* cold-start availability, and whether a host process start plus a fork fits
inside the engage deadline — settled by the spike: a scratch label, a `Sockets` plist, a connect
with the job stopped, and a measured engage-to-record time.

**Slice 3 — surfaces stop being spawners, and the fleet reattaches** (prohibition plus its lint
test, the reattach contract on the store, the app-side PR, Gate A in CI and Gate B recorded).
*Riskiest:* that the app's session-state rebuild can be expressed purely as "read the store" without
a second ownership notion appearing — it already holds `this.process` for the serve generation,
which is exactly the temptation to avoid for runtimes.

---

## 10. Migration: nothing is restarted

Live runtimes are surface-spawned and keep running. Nothing here restarts, signals, reaps or
re-engages a live runtime; convergence is the fleet's existing self-retirement — a runtime that goes
idle retires on its own, and the engage that replaces it goes through the host
(`_spawn_interpreter`'s mixed-generation convergence is the precedent: one session at a time, no
fence, no drain) **[C]**.

**One hour after deploy** the fleet is mixed by design: sessions engaged since the deploy are
host-spawned, the rest are still app-spawned. That is normal (the tree already runs mixed builds)
and **observable without new machinery** — each boot record carries `parent_pid` and `spawn_chain`
(`journal.py:230-241` **[C]**), so the census can count "spawned under the app" against "spawned
under the host", which is also the rollout's progress metric.

**What stays exposed during the window, said plainly:** while a session's runtime is still
app-spawned, an app death can still take it. The population that matters is the sessions that never
go idle — those do not converge on their own, and the only thing protecting them during the window
is Slice 1.

`i18n`: every new user-visible sentence (host not running; engage refused because the host could not
be reached; the sanction-refusal notice) is a keyed string translated into every locale in
`i18n/languages.json` with its ledger rows, per `AGENTS.md` — a feature is not done unlocalized.

---

## 11. What I am not proposing, and the risks to watch

Not proposing: one launchd job per runtime (it makes `launchctl` a fleet lever and adds per-session
plist churn); a host inside `lop serve` (that is the app's tree — the bug); double-forked
daemonisation (it loses `wait()`, the one thing a parent buys, and is why the host must be the
direct parent — `docs/design-session-survival.md` §4); any new third-party dependency; automatic
restart of a killed runtime (a killed turn is interrupted and named, not silently resumed).

Watch during rollout: (1) the host becoming a latency single point — engage time is now host-hop +
fork, and the metric is engage-to-record, not CPU; (2) two hosts, the first failure mode —
single-instance election by lock in the config root, the second serving as observer only; (3) a host
on a stale build — it must retire like the daemons do, successor adopting by record, not a drain;
(4) the sanction gate silently breaking a stop path nobody exercised in review — mitigated by
enumerating every sanctioned sender above with a test each, and by failing *loud* (a receipt naming
an unsanctioned signal) rather than silently.