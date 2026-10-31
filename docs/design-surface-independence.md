# Design: a runtime's life does not depend on any surface

Status: **proposal (architect)**. Scope: `~/local-operator` (backend); the desktop app is named only where a
contract changes. No version bump.

Supersedes the *spawn-ownership* half of `docs/design-session-survival.md` (2026-09-15): its diagnosis holds
and its step 1 (the app stops being able to reach a runtime) is still right, but its step 2 — a
launchd-supervised session host as the only party allowed to spawn — was designed and never built
(`com.local-operator.sessions` appears nowhere in the tree but that document). This note re-derives it
against measurements taken since, decides what the first pass left open, and adds the half now measured to
be missing: **a spawner is not a defence.**

Tagged **[M]** measured (here or in the record this note was commissioned with), **[C]** read in the code at
the cited `file:line` on `origin/main` = d0281fa91bf, **[I]** inference, unrun.

---

## 1. The problem, in one measurement

2026-10-30 21:57:52 -0400: 12 background exec workers, ≥5 session runtimes and 2 `lop serve` daemons took
SIGTERM within 1.33 s when the operator force-quit a frozen desktop app. Nothing staged a stop for any of
them, no `logs/stop-sweeps.jsonl` exists anywhere, and the victims wrote `cause=runtime-shutdown` receipts.
**[M]**

The natural experiment at the same moment is the measured discriminator: runtimes spawned by the
launchd-owned wake supervisor (`com.local-operator.wakes`) **survived**; everything that died descended from
the app (`cli.py` ← session runtime ← `lop serve` ← `Local Operator.app`). **[M]**

Two earlier fixes are in the tree and neither suffices, so this is not a third variation on them: every
runtime is spawned with its own session and group (`start_new_session`, `launch._spawn_runtime`) **[C]**,
and the macOS responsibility disclaim (`macos_disclaim.spawn_disclaimed`) severs the responsibility chain,
is **active on this host** **[M]**, and did not prevent this event — it fixed the 2026-10-09 force-quit
sweep and this one went around it.

Excluded by experiment, worth exactly what it is worth: `kill -9` of a launchd job's main pid does not sweep
detached children; `launchctl kill SIGKILL` of a runningboard-submitted app job does not either; the
responsibility chain is not the carrier. **[M]** Three OS mechanisms eliminated is **not an exhaustive
exclusion**, so the *actor class* — "a party that knows the pids: a tree walk or a store enumeration" — is
**[I]**; a same-uid peer surface enumerating `run/mobile` records (another `lop serve`, the phone daemon,
another session) is not excluded, nor is memory pressure. What is measured is narrower and stands alone:
**descendants of the app died, launchd-fed runtimes lived.**

The requirement, in the operator's words: closing or force-quitting the UI must never terminate a runtime;
runtimes must be independent, reparented away from the app, discovered through the session store rather than
owned by whichever surface launched them; a surface dying is a display event.

---

## 2. Who spawns

**(A) Extend `com.local-operator.wakes`.** One fewer unit. Rejected, three reasons in the code:

1. *It self-retires.* `RunAtLoad: True` (`wakes/install.py:338`) plus `KeepAlive: {SuccessfulExit: False}`
   (`:344`) means a machine with no scheduled wakes runs no supervisor at all: it cannot be the
   always-available spawner, and making it always-on deletes the property the unit was designed around.
2. *Its stop semantics are the inverse of what is required.* On stop it forwards SIGTERM to the runtimes it
   spawned (`wakes/owned_runtimes.py::sweep`, written for #3028) **[C]** — a routine supervisor restart
   would become a fleet stop, exactly what rule R3 forbids, and removing the sweep removes the reason that
   module exists.
3. *Import weight.* The job's justification is being import-light. Measured here: importing the spawn path
   (`session.runtime.launch`) costs **190 modules** against **109** for `wakes.store` **[M]** — the
   reproducible part; the wall times are cache- and load-dependent (mine ran in fresh processes, and the
   reviewer reproduced the same counts at 289.2 ms / 101.5 ms with warmer caches). Every supervisor start
   would pay that.

**(B) A sibling launchd job — the session host — install-on-demand, single instance, launchd-owned.**
**Recommended.** The installer shape (`wakes/install.py::render_plist`, `:315`), the "installed means
running" repair (`supervisor_state`, `:448`; `ensure_supervisor_installed`, `:635`) and the platform matrix
(`is_supported`, `:167`) are reuse, not invention **[C]**. Cost: a fourth supervised unit to install and
reason about, and a second installer-shaped module.

Ownership cap: the host may **spawn** a runtime and may never **stop** one. The only parties that end a
runtime are the runtime itself (`process._should_exit` and the reaper) and the stop ladder on a person's
instruction (`control.stop_session`, `:1350`) **[C]**.

### Cold start: the host is not running

An engage must not silently fail and must not fall back to a surface spawn. A ladder with no local rung:
`ensure_session_host()` (bootstrap when absent, kickstart when loaded-but-stopped, reusing
`wakes/install.py`'s state parse rather than a second copy of "is it running" **[C]**) → connect with a
bounded timeout inside the engage's own deadline → on failure raise, `RuntimeStartupError` already carrying
an actionable sentence for exactly this (`launch.py:796-811` **[C]**, new keyed string per §10).

The first engage on a fresh machine pays the host's start before the runtime's. Cheapest mitigation first:
ensure-running at the two places that already warm a standby (`cli.py:17529` for `lop serve`, `cli.py:18443`
for the TUI) **[C]**, never a spawn. If the spike shows the boot cost does not fit the engage deadline,
prefer launchd **socket activation** (`Sockets`/`SockPathName`: launchd holds the listening fd and starts
the job on first connect), which also removes the bootstrap race and the "exited successfully" hole. That
part is **[I]**; the spike is in §9.

### Transport: a new socket, and why no existing surface works

The request cannot ride the wake index, `run/serve`'s HTTP plane, or `launchctl`: the operator capability
must travel as **a descriptor**, and `harness/approval.py` forbids it touching a file, the environment or
argv **[C]**. `run/serve` is worse still — the app's own backend, the tree this change exists to leave. So
`<root>/run/session-host/host.sock`, mode `0600`, framed request/reply plus `SCM_RIGHTS` for the capability.
The framing is not new — `standby.try_adopt` (`:1209` **[C]**) already sends exactly this request (`root`,
`interpreter`, `env`, `cwd`, `capture`, plus the cap fd by `SCM_RIGHTS`); only the transport's owner
changes. (`run/host/` is the boot-record directory, `run/mobile/` the discovery records — a new directory
keeps the three namespaces apart.)

The request must carry **the requester's environment**: launchd hands the host a bare
`PATH=/usr/bin:/bin:/usr/sbin:/sbin`, which is why the wake supervisor repairs it for its own children
(`wakes/supervisor.py::main`, `setup_cross_platform_environment`) **[C]**; a host-spawned runtime must get
the engaging surface's environment or every such turn loses Homebrew, nvm, cargo, kubectl. The standby
request already carries `env` wholesale **[C]**.

The reply must carry **pid, capture path and the child's exit status**. The surface no longer holds the
`Popen`, and `_spawn_runtime`'s docstring is explicit that the spawned process's liveness is the only sound
death signal in the window before the lease is taken (`launch.py:398-413`) **[C]**. Without a reply-carried
death fact, a candidate that dies before `logging.basicConfig` degrades from "reported with the child's own
reason" to "timed out" — avoidable, and why the host keeps `_harvest_on_exit`'s job.

---

## 3. No double spawn

The guarantee does not move, because it was never the spawner's: the transcript lease is taken by the
**child** (`session_lease.acquire_session_lease`, `:401`, `O_CREAT|O_EXCL` claim carrying pid + birth token)
**[C]**. Whoever forks a candidate, the loser exits; `engage_runtime`'s `_lease_holder` probe
(`launch.py:305`) is unchanged and still decides whether to ask for a spawn at all.

When the host is slow, the surface's engage waits inside its existing `deadline_s` (the loop at
`launch.py:1055-1200` is untouched) **[C]** and on expiry raises `TimeoutError`/`RuntimeStartupError`. It
never spawns locally and never asks twice from one engage — the negative control that pins it is R4 in §6.

Two additions, neither load-bearing: **host-side coalescing** (an in-memory "spawning since" per session so
two concurrent requests fork one child — an optimisation with a counter, the lease stays authoritative), and
`EngageOutcome.spawned` (`launch.py:283-294` **[C]**), which keeps its meaning: true when this call caused a
candidate to start, whoever forked it.

---

## 4. The operator capability

`--operator-fd` hands a 32-byte capability from the process the operator's keyboard is attached to into the
runtime, on an inherited descriptor whose *number* (not value) rides in argv; the value is minted per spawn,
registered against the child's pid in the spawner's memory, never written anywhere
(`harness/approval.py:854-867`, `open_operator_cap_handoff`, `:1415`; minted `launch.py:568`, delivered
`:706`, remembered `:715`) **[C]**. A supervisor spawn has no such descriptor.

**(a) Supervisor-held capability, surface proxies through the host.** The host mints and keeps the cap and
performs authority-increasing acts on request. Consequence: the host becomes an authority broker on a
machine-wide socket, and — since `AttachClient.authority_proof` reads the table in the *presenting* process
— the runtime would see the host, not the keyboard's process, as the presenter, while a control-plane client
would have to be invented inside the host. **Rejected**: the largest new attack surface for the weakest
meaning.

**(b) Surface mints; the child end travels by `SCM_RIGHTS` with the spawn request; the host never reads the
value.** On success the surface calls `remember_operator_cap(pid, cap)` with the pid from the reply — the
call it already makes on both existing routes (`launch.py:629`, `:715`) **[C]**. **Recommended.** Semantics
are byte-for-byte today's, "the process that engaged presents the proof". The value's **lineage** is
preserved — the descriptor exists only because the surface called `socketpair()` — while it never touches
argv, the environment, a file, or the host's memory.

**(c) Drop the authority for host-spawned runtimes.** The operator loses `/approvals auto` and approved-card
decisions from the surface, for exactly the spawn shape every session is moving onto. **Rejected**: it
trades a feature in use for a migration.

**The security consequence of (b), plainly.** The host's socket is machine-wide and same-uid reachable —
precisely the shape `standby.py`'s docstring records rejecting in review round 1 (R1-1), where
`<root>/run/standby/standby.sock` would have let any same-uid process bind first and receive the capability
**[C]**. Two differences keep (b) from re-opening that: the host is a *relay*, never a *holder* (no
capability outlives the request in flight — a design rule, testable against its API surface); and an
impostor that asks for a spawn receives a capability for **its own** child, which it can already obtain by
calling `engage_runtime` itself — a tidier path to an existing privilege, not a new one. What (b) does
**not** buy: the runtime cannot tell who asked, and cannot today either (`operator_cap_for` answers "did
*this* process spawn it", `:867` **[C]**); provenance is preserved, requester identity is not authenticated.

---

## 5. Reattach: a fresh surface reads the store and believes it

Read path — all existing, no new protocol:

- `reclaim.read_fleet` (`reclaim.py:935`; `Fleet` `:913-916`) reads the store's own four sources —
  `run/mobile` discovery records (the only attachable one), `run/host` boot records (attribution, never
  liveness), **viewer** records, and the socket table for a record-less runtime's port. The **process census
  is not one of them**: it is `roster.build_roster`'s (`roster.py:288`; rows `RosterRow`, `:125`), which
  unions census ∪ records ∪ boots and says which source knew each row — the only source that sees a runtime
  whose records are both gone **[C]**.
- `registry.classify` (`:583`) is the single owner of `live` / `wedged` / `stale`: pid alive **and** beat
  fresh = live; pid alive, beat old = **wedged, "not answering"**, never "dead"; pid gone = stale **[C]**.
- `info.collect.collect_sessions` (`:276`) is the one implementation behind `lop sessions` and `/info`, and
  `session.catalog.CatalogEntry.status_code` (`:268`) is the wire field the desktop renderer branches on
  (`RUNNING_STATUS_CODES`, `:43`) **[C]**.
- The surface writes a **viewer** record (`session/runtime/viewers.py`) — a display fact, not ownership
  **[C]** — and attaching a session with a live runtime is `engage_runtime`'s first step, the dial, so a
  fresh app reattaches rather than spawning a rival **[C]**.

**What a surface must NOT do on start** (rules with owners, not style notes): no `stop_session`/`stop_all`
(`control.py:1350`, `:1912`) **[C]**; no `reclaim.reclaim_runtimes` (`:1461`) — a residency sweep is never a
startup action **[C]**; no write or withdrawal of a stop marker; no engage of a session whose record is
`live`; and no re-spawn to "rebuild state" for a session it has no tab for (that session is an ordinary cold
row).

**Stale versus live, operationally:** `scan(reap=True)` unlinks a record whose pid is gone, and the run/exit
records (`journal.py`: `runtime-run.json`, `runtime-exit.json`, boot record at `:290`) let a STORED row
carry a *named* departure reason instead of a borrowed one **[C]**. A `wedged` row is degraded-and-named;
nothing on start may reap it. The app-side contract this implies is named in
`docs/design-session-survival.md` §6 and not re-read here: live-vs-cold per row off the catalogue, restore
the tabs it had, and never treat "not in my memory" as "not running".

---

## 6. What must never happen — as rules with a test each

**R1. A surface's exit, quit or crash never signals a runtime.** Test: extend
`tests/e2e/test_terminal_close_survives_e2e.py` (real pty, three real death shapes — `ctrl-d`, `pty-close` →
a real `SIGHUP` to the group, `SIGKILL` to the group — with a dual-witnessed death precondition **[C]**) to
a **host-spawned** runtime, asserting the runtime's parent identity as well as its survival. Unit control: a
lint-shaped test that no kill site in the surface modules names a runtime pid — precedent, the prior
design's pin on the `process.kill(-pid)` count. Negative control in §8.

**R2. Only the stop ladder and the runtime's own residency policy end a runtime.** Test: the sanction gate
(§7) — a direct SIGTERM with nothing staged does not end the runtime; the same signal after
`control.stop_session` stages its marker does; `test_stop_ledger.py` and `test_signal_drain.py` stay green.

**R3. A supervisor restart does not kill runtimes it did not spawn this time — adoption, not ownership by
birth.** Test: unit — the host's exit path signals nothing (it has no `owned_runtimes`-style sweep);
dev-machine scripted test — start the host, let it spawn, SIGTERM the host, assert the runtime survives and
the successor adopts it by record. Supporting measurement: `launchctl kill SIGKILL` of a submitted job does
not sweep its `SETSID` children **[M]** — which is why no sweep is needed and why one must not be added.

**R4. A host-absent engage never spawns locally.** Test: a refusing double behind the spawn seam → engage
raises a keyed error and the `Popen` count is zero (§3).

---

## 7. The failure this does not fix — and the invariant that does not need the sender

**We have not proven which party sent the SIGTERM, and this design cannot name it.** The victim's own
instrument records name, number, arrival time and whether anything was staged (`signal_receipt.observe`,
`:279`) plus a spawn-chain liveness snapshot, and says `sender: unavailable` **[C]**. Measured here today:
`hasattr(signal, "sigwaitinfo")` is **False** **[M]** — macOS stdlib exposes no `siginfo`, so there is no
sender pid or uid to be had; on Linux the sender's uid would be this uid anyway, which distinguishes
nothing.

So the defence cannot be "the spawner is different". It has to be enforced **at the victim, independently of
who asks**: *no terminating signal ends a runtime unless a durable sanction covers that run.* **The gate, as
the contract slice 1 implements:**

1. **Arrive.** A handled signal reaches `_route_signal` (`process.py:4092`): receipt → the turn's cause →
   the branch.
2. **Classify before acting, on coverage.** The predicate is `signal_receipt.stop_class_of(entry)` (`:404`)
   ∈ {`CLASS_DELIBERATE`, `CLASS_ATTRIBUTED`} — i.e. `sanction == "marker"` on the arrival entry. **Not**
   `cut_off_verdict is None`: that answers "what does a signal-cut turn say", so a covering *involuntary*
   marker (reclaim's own sanctioned act) returns `runtime-killed`, non-None, and a gate keyed on it would
   refuse the sweep this note certifies as compliant.
3. **Refusal comes first, and does none of what a signal otherwise does.** No cause armed
   (`session.note_cut_off` not called, so `_note_signal_cut_off` at `:1453` runs only on the accepted path —
   otherwise an anonymous sweep brands a live turn `runtime-shutdown` and `Session._classify_cut_off`
   carries that sentence into any later abort); no trigger set, no drain started, `stop` not set; and **no
   pairing-eligible evidence**. Refusals get **their own file**, `runtime-signal-refused.json` beside
   `runtime-signal.json` — not a `refusals` key inside the receipt. That is the fix slice 1 implements,
   chosen over the `covers_run` guard because `runtime-signal.json`'s name asserts *a signal arrived and
   here is its pairing*, and it already has two independent coverage predicates
   (`signal_receipt.covers_run`, `:378`; `rescue._receipt_covers_run`, `rescue.py:418`), a merge writer
   (`:350`) and a rendered `count` (`describe`, `:260-271`). A refusals-only receipt asks every current *and
   future* reader to remember a field that contradicts the artifact's name — and **this finding is the proof
   that the second reader gets missed**: `_death_class_receipt` (`rescue.py:588-597`) answers
   `unclassified/receipt-no-signals`, which its own docstring calls TERMINAL, so the rescue pass stops
   firing. With a separate file the receipt keeps its meaning by construction and the rescue lane needs no
   change at all. Refusals are counted and coalesced in their **own** counter per run, never the receipt's
   `count` (which `attention.py:1042` feeds to `cut_off_verdict` and `describe` renders as "(+N more)"), so
   a SIGTERM loop can neither inflate a user-facing sentence nor grow a ledger without limit.
4. **Accept = today's behaviour, unchanged:** receipt, cause, branch, drain.

**Where the gate can ask, and where it cannot.** The rule is *refuse only where a run key exists*;
everywhere else behaves exactly as today and records the arrival.

- The runtime arms at `install_loop_signal_handlers` (`process.py:5538`, SIGTERM + SIGINT); before that, and
  during the adopted-spare window (a spare renames its argv at adoption, `standby._commit_adoption:1914`,
  and only then becomes a runtime), default disposition applies — bounded, and not where work lives.
- `local_operator/exec_worker.py` has no conversation directory during provider discovery —
  `_record_arrival` (`:144`) says so and records `sanction: none` **[C]** — so there is nothing to ask: it
  keeps today's behaviour (stop, exit 130) and the attempt is recorded; once its session exists, the gate
  applies. Closing the discovery window needs a key the worker holds before its session exists (its
  `exec-jobs.jsonl` arrival row, or the job id off argv), which makes the job ledger a second sanction store
  — a real change, deferred, not hand-waved.
- `SIGHUP` needs no gate: the runtime makes it a no-op first thing in `amain` (`_install_sighup_ignore`,
  `:4804`, called `:5254`) **[C]**; `SIGUSR1` is the evidence dump and stays installed-not-swallowed,
  because *installing* it is what makes it non-fatal. **`SIGQUIT`, `SIGUSR2` and `SIGALRM` are unhandled and
  therefore default-fatal — the one-signal bypass of this gate, and slice 1 closes it** by installing the
  same route for each at `amain`'s own arming, beside the SIGTERM/SIGINT map (`:5538`) — and **never inside
  `procstate.install_loop_signal_handlers`** (`procstate.py:245`), which also serves the portal
  (`mobile/service.py:125`) and the tunnel service (`tunnels/service.py:589`) with `{SIGTERM: stop.set,
  SIGINT: stop.set}`: arming there would install a refusal where no run key exists and take away both
  services' only graceful stop. Nothing in-tree sends the three to a runtime, so closing them costs no
  functionality **[C]**.

**The sanctioned senders, and the one that is not.** `control.stop_session` stages on all three rungs —
`sigterm` (`:1717`), `sigkill` (`:1767`), socket (`:1283`) — and `reclaim._attest_before_signal` (`:1377`)
stages too, its docstring naming the same operator rule **[C]**. `update._attests` (`update.py:4949`) is an
**attestor, not a sender**: it stages a marker for a tree removal, and `update` sends no signal to a runtime
(`:6935`, *"not SIGTERM that process: it is not ours to bounce"*). `wakes/owned_runtimes.sweep` (`os.kill`
at `:190` **[C]**) stages nothing and must stage `note_involuntary_stop` per candidate — **built from the
run's PUBLISHED record** (`reclaim.read_fleet:935`, `journal.read_boot_record:339`), not from the census
row: the run key is `(session_id, pid, started_at)`, coverage is strict to `_RUN_KEY_TOLERANCE_S = 1.0`
(`signal_receipt.py:83`), and the census age the sweep holds is 2-s-rounded — so a marker built from `age_s` can miss
coverage and the supervisor's SIGTERM is then refused, which is #3028 exactly (an approval-parked runtime
outliving its supervisor for the day-long gate). `standby.py:1479` (`warm.proc.terminate()`) is an **exempt
sender** to leave alone: pre-adoption a spare is `-m …session.runtime.standby` on its socketpair, with no
run key and no runtime handlers, so it dies by default disposition — correct, it is not a session's work.

**No interactive capability is lost.** A runtime is its own session leader with no controlling terminal
(`start_new_session`) **[C]**, so no keystroke can deliver SIGINT or SIGHUP to one; the only person-shaped
stop is the ladder, and the ladder stages. The escape hatch survives: SIGKILL is unrefusable and
deliberately absent from every ladder (`reclaim.py` docstring; `owned_runtimes.py` "SIGKILL is deliberately
absent") **[C]**; `lop stop --force` uses the SIGKILL rung.

**Stated limit, because claiming more would be false:** a same-uid process can write the marker itself — the
model's own `bash` tool runs as this uid, which is the premise `standby.py` records **[C]** — so this is
**attribution plus inertia, not authentication**. It converts an anonymous mass sweep into a durable,
attributable act, and makes the measured class (an automated tree/store walk, which stages nothing) inert
*in any runtime that has armed it*. A determined same-uid sender still wins; the two things that would
change that are an OS that reports the sender (we do not have one) or runtimes under a different uid (out of
scope). **That is the honest boundary of this design.**

---

## 8. Test plan: the two gates, and which one can be CI

**Gate A — a process-group kill of the surface; every runtime survives, stays reachable, and a fresh process
reattaches. CI-able, backend half.** The shape of `tests/e2e/test_terminal_close_survives_e2e.py` extended:
a real pty hosts a real TUI, engages a runtime, runs a real turn, the group is killed in three shapes, then
a **second** process attaches to the surviving runtime and reads the live turn. In CI the host runs as an
ordinary subprocess (its `main` is a module entry point); the launchd wiring is not what Gate A tests, and
saying so is honest.

**Gate B — several real sessions through `kill -9` of the real UI's process group. Rigged run only; never
CI.** It needs the Electron app bundle and a real desktop session, and one further fact closes the argument:
the force-quit **sweep itself** is not reproducible from a scripted context — `macos_disclaim`'s docstring
records that `LQForceQuit` killed the app instance and did not sweep even an undisclaimed detached child in
the rig **[C]**. Gate B is an interactive, recorded run (real app, 3+ sessions mid-turn, `kill -9` the
group, relaunch, see them live) whose output is evidence on the PR, not a regression test. CI cannot replace
it and must not be reported as if it did.

**Negative controls, required** (a passing test proves nothing until it can fail): Gate A must go red when
the spawn reverts to surface-parented — assert parent identity, not only survival; the sanction-gate unit
must go red with the gate removed; R4 must go red if a local fallback is added. Scripted on a dev machine
but not CI: the launchd restart test in R3. And the wake sweep's own test must drive the **real** sweep
against a **real published record** (§7), not a fabricated run key, since the tolerance rule is what #3028
depends on.

---

## 9. Slices, in order, with the riskiest assumption in each

**Slice 1 — the victim-side sanction gate and its attribution** (`_route_signal`; the separate refusal file
`runtime-signal-refused.json` **plus its one test** — a refused arrival, then a SIGKILL-shaped death, and
the rescue pass must still fire; the exec worker; `owned_runtimes.sweep` staging from the published record;
the `SIGQUIT`/`SIGUSR2`/`SIGALRM` route armed at `amain`'s own call site; keyed copy). *Riskiest:* that
every legitimate sender stages before signalling — verified for three, false for one, unknown for
out-of-tree senders (an operator's `kill` or a script would stop working; SIGKILL and `lop stop --force`
remain). It comes first because it removes the false attribution and makes an anonymous sweep inert **in any
runtime that boots with it** — and its **arrival condition is the next start of each runtime**, so a runtime
already running keeps today's behaviour and the never-idle fleet stays exposed until it retires, which is
why §10 promises no more than that.

**Slice 2 — the session host** (sibling unit, install-on-demand, single-instance lock, socket, request/reply
carrying env/cwd/capture/cap-fd and a death report; `engage_runtime` delegates with no local rung).
*Riskiest:* cold-start availability, and whether a host start plus a fork fits inside the engage deadline —
settled by the spike: a scratch label, a `Sockets` plist, a connect with the job stopped, and a measured
engage-to-record time.

**Slice 3 — surfaces stop being spawners, and the fleet reattaches** (prohibition plus its lint test, the
reattach contract on the store, the app-side PR, Gate A in CI and Gate B recorded). *Riskiest:* that the
app's session-state rebuild can be expressed purely as "read the store" without a second ownership notion
appearing — it already holds `this.process` for the serve generation, exactly the temptation to avoid for
runtimes.

---

## 10. Migration: nothing is restarted

Live runtimes are surface-spawned and keep running. Nothing here restarts, signals, reaps or re-engages a
live runtime; convergence is the fleet's existing self-retirement — a runtime that goes idle retires on its
own — and the lever that moves a live runtime onto a new build is `_refresh_for` (`process.py:2117`,
*announce and retire so the next engage runs `newer`*), which the build poll hands `poll.newer` (`:1825`),
which reaches `begin_retire("runtime-retired", …)` (`:2254`), with `_Drain.cause` defaulting to
`runtime-retired` (`:2753`). This design changes none of it **[C]**. (The nearby `_consume_update_marker`,
`:2683`, is the **successor's confirmation that an update already applied**, not the mover; round 1 of this
note named it wrongly.) The engage that replaces a retired runtime goes through the host
(`_spawn_interpreter`'s mixed-generation convergence is the precedent: one session at a time, no fence, no
drain) **[C]**.

**One hour after deploy** the fleet is mixed by design: sessions engaged since the deploy are host-spawned,
the rest still app-spawned. That is normal (the tree already runs mixed builds) and **observable without new
machinery** — each boot record carries `parent_pid` and `spawn_chain` (`journal.py:230-241` **[C]**), so the
census can count "spawned under the app" against "spawned under the host", which is also the rollout's
progress metric.

**What stays exposed during the window, said plainly:** a session whose runtime is still app-spawned can
still be taken by an app death, and **slice 1 does not change that for it** — the gate arms in a process
booted on the new build. The population that matters is the sessions that never go idle: they do not
converge on their own, so they stay exposed until they retire, and nothing here restarts them to protect
them.

`i18n`: every new user-visible sentence (host not running; engage refused because the host could not be
reached; the sanction-refusal notice) is a keyed string translated into every locale in
`i18n/languages.json` with its ledger rows, per `AGENTS.md`.

---

## 11. What I am not proposing, and the risks to watch

Not proposing: one launchd job per runtime (it makes `launchctl` a fleet lever and adds per-session plist
churn); a host inside `lop serve` (that is the app's tree — the bug); double-forked daemonisation (it loses
`wait()`, the one thing a parent buys, and is why the host must be the direct parent —
`docs/design-session-survival.md` §4); any new third-party dependency; automatic restart of a killed
runtime.

Watch during rollout: (1) the host as a latency single point — engage time is now host-hop + fork, and the
metric is engage-to-record, not CPU; (2) two hosts, the first failure mode — single-instance election by
lock in the config root, the second an observer only; (3) a host on a stale build — it retires like the
daemons do, successor adopting by record, not a drain; (4) the sanction gate silently breaking a stop path
nobody exercised — mitigated by enumerating every sanctioned sender above, one test each, and failing *loud*
(a receipt naming an unsanctioned signal); (5) a refusal loop growing a receipt — the coalescing rule in §7
is the guard and its count is worth watching.