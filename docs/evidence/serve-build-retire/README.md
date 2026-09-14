# Historical evidence: unsafe daemon retirement (superseded)

> **Not the current behavior or a rollout runbook.** This directory records
> the pre-release #1102 experiment, whose automatic daemon exit was unsafe:
> lifespan shutdown cancels scheduler-owned tasks, and no successor is proven
> ready. Its scripts/transcripts below describe that historical path, not a
> production capability. Do not run `run.sh` as current validation (it also
> predates full inherited-environment and record-secret scrubbing).
>
> Production now only announces changed builds, keeps serving, and reconciles
> withdrawal/retargeting. `retiring_from`/`retiring_to` do NOT tell the UI to drop
> SSE/watch leases. Drain/latch behavior survives only as an explicitly injected
> internal test callback. Current real-process regression evidence is generated
> by `tests/e2e/test_serve_build_announcement.py`, including actual scheduler task
> ownership and lifespan cancellation on the unsafe baseline.

## Historical experiment

Raw-process evidence for the daemon's build watch: an install replaced on disk
under a running `lop serve`, and the daemon's answer to it — **announce the
handover in its rendezvous record immediately, keep serving, and only once
nothing is attached latch against new work and leave.**

Ten runs, each on an ephemeral port (`--port 0`) under its own
`LOCAL_OPERATOR_CONFIG_DIR` and pointed at a FAKE install root with
`LOP_BUILD_PREFIX`, driven by flipping that root's `.lop-source` — the file
`lop-update` writes last, and the only signal this feature reads. No live daemon
on the machine is touched, the operator's exported desktop token is scrubbed per
daemon so the ungoverned run is genuinely ungoverned, and every `CMUX_*`
variable is unset so nothing here can reach the operator's own workspaces.

```sh
bash docs/evidence/serve-build-retire/run.sh     # ~7 minutes, 9 daemons + 2 injections
```

`transcript.txt` is a verbatim run (`run1.txt`…`run10.txt` are the per-run slices,
echoed inline). `drive.py` starts nothing — `run.sh` starts the daemons — and it
only ever reads the record file, makes ordinary HTTP requests and watches the pid.
`inject.py` drives the real poll in-process for the one fault a real daemon cannot
be given (a raising probe); `ungated.py` does the same for the two INSTRUMENTS that
are supposed to be able to fail (run 10), because a guard nobody has seen go red is
a guard nobody has tested.

## What the runs show

**1. Unsupervised — announce while serving, then leave.** The record exists with
`retiring_from: ""`/`retiring_to: ""`; after the marker flips, the record
announces `0.54.45@1111111 → 0.54.45@2222222`, the daemon answers requests the
whole time, the log names the build and the command that brings the daemon back,
and the clean exit removes the record.

| measured | value |
| --- | --- |
| flip → announce | 14.204 s (settle 10 s + one check interval 5 s) |
| announce → record gone | 6.21 s (the drain was empty, so the latch came on the next check + a jittered slice of the 20 s stagger) |
| record present after exit | False |
| pid alive after exit | False, 0.2 s after the record was removed |

**2. Claimed — announced-but-admitting, then the refusal matrix.** The same
daemon answers a create **200** while its record announces the handover, and only
answers the typed refusal once its drain has emptied:

```
t= 4.295s still admitting: create -> HTTP 200 (announced, not refusing)
--- t=9.278s: the daemon LATCHES (refuses new work) ---
create while latched: HTTP 503 {"detail":{"code":"daemon-retiring","message":
  "This backend is restarting onto a new build and is not accepting new work.
   Reconnect to the new backend and retry."}}
```

Every route that reaches the door, on that one latched daemon — **23 routes, all
23 the typed 503** (the four `…/variables` rows are `main`'s code-memory surface,
added upstream while this branch was in review: the completeness test reported
them, and the seam gate already covered them):

```
  POST /v1/desktop/sessions (create)             HTTP 503 daemon-retiring
  GET  /v1/desktop/skills?session_id=…           HTTP 503 daemon-retiring
  GET  …/{id}/mcp   (a session route)            HTTP 503 daemon-retiring
  POST …/{id}/mcp                                HTTP 503 daemon-retiring
  POST …/{id}/credentials                        HTTP 503 daemon-retiring
  POST …/{id}/fork                               HTTP 503 daemon-retiring
  POST …/{id}/asides                             HTTP 503 daemon-retiring
  POST …/{id}/asides/{aside}/adopt               HTTP 503 daemon-retiring
  POST /v1/desktop/stop (a session route)        HTTP 503 daemon-retiring
  GET  …/{id} (snapshot)                         HTTP 503 daemon-retiring
  GET  …/{id}/history                            HTTP 503 daemon-retiring
  GET  …/{id}/variables (a session route)        HTTP 503 daemon-retiring
  POST …/{id}/variables                          HTTP 503 daemon-retiring
  PATCH …/{id}/variables/{key}                   HTTP 503 daemon-retiring
  DELETE …/{id}/variables/{key}                  HTTP 503 daemon-retiring
  GET  …/{id}/failovers                          HTTP 503 daemon-retiring
  GET  …/{id}/command-entities                   HTTP 503 daemon-retiring
  GET  …/{id}/events (the app relay)             HTTP 503 daemon-retiring
  POST …/{id}/messages                           HTTP 503 daemon-retiring
  POST …/{id}/commands                           HTTP 503 daemon-retiring
  POST …/{id}/answers                            HTTP 503 daemon-retiring
  POST …/{id}/watch                              HTTP 503 daemon-retiring
  POST …/{id}/warm                               HTTP 503 daemon-retiring
  spawn-seam lines the daemon logged during the LATCHED matrix: 0
```

**Why that list is the 23 and not the five it used to be.** Round 2 measured the
five-row version (`create`, `/warm`, `/messages`, `/commands`, `/answers`)
answering 503 while `POST …/mcp`, `…/credentials`, `…/fork` (and its child prompt
admission), `…/asides` and `…/adopt` reached `bind_runtime()` — the spawn seam
itself — on the SAME latched daemon, with the daemon log naming the engage
attempt. The gate is now `DesktopSessions.session()`, the one place a desktop
route obtains a bridge and the only place in the process that constructs one, and
this list is a coverage TEST rather than the mechanism: it is walked out of the
routers by `tests/unit/server/test_serve_retire.py`, which fails when a route
reaching the door has no row. The reads are on it deliberately — a session-scoped
answer can only come from the build the daemon has already told its readers to
leave — and the record plane (`GET /v1/desktop/sessions`, `/health`, the record
file) is what stays readable, because it never takes a bridge to begin with.

**The last line is not an argument from an empty directory.** It used to be: the
cell read `run/mobile` and claimed "no runtime was started" from its absence,
which an isolated config root produces whether the request was refused or never
tried at all — the runtime cannot even be constructed without a configured
hosting platform, so neither answer leaves a record. What is counted now is the
daemon's own words for entering the seam (`engage:`, `could not start a runtime`),
and run 7 is the same instrument over the same routes on one daemon in both
states: **18 lines while merely announced, 0 while latched**. That is a
measurement that can fail, and run 10 shows it failing on demand.

The run sets `LOP_BUILD_SETTLE_S=1` and `LOP_BUILD_STAGGER_S=300` (the documented
test-only overrides) so the whole matrix fits inside the refusal window; the
daemon's own announcement and refusal ordering is unchanged by them. It is
stopped with a plain SIGTERM, which still exits cleanly and removes the record:
`record after the stop` shows an empty `run/serve`.

**3. The app's own relay — the case this feature exists for, and the one round 1
measured as impossible.** `DesktopStreamRelay`'s exact request
(`GET /v1/desktop/sessions/{id}/events`) is held open across the update, at
production constants:

```
--- t=13.620s: the record announces the handover ---   (retiring_from/retiring_to set)
t=14.646s holding: pid_alive=True getting_health=200 retiring_to='0.54.45@2222222' relay_frames=4
... ten samples, one per second, two check intervals ...
t=24.789s: dropping the relay (the view closes)
t=30.530s still admitting: create -> HTTP 200 (announced, not refusing)
--- t=33.665s: the daemon LATCHES (refuses new work) ---
--- t=45.911s: the record is removed (clean exit) ---
```

The daemon's own log names the term holding it:

```
[INFO] serve daemon: build 0.54.45@2222222 is on disk and announced; 1 in-flight
       desktop request(s) on 4a07ae0f8e1a is still in flight, so it keeps serving
       until that completes
```

Two details worth reading off it. The announcement is **readable while the
daemon keeps serving** (round 1's blocker was that it never appeared at all),
and the latch trails the relay's drop by ~4 s here (up to ~10 s in the first
capture): the server notices an abandoned stream at its next write, which is the
15 s heartbeat cadence — during which the daemon is still admitting work, exactly
as the design says it must while merely announced. Dropping the relay is the
client's act, which is why it is specified for the UI in
`docs/design-daemon-discovery.md` §7.

**4. A write that fails neither latches the daemon nor stops the poll.** The
record's directory is made `UF_IMMUTABLE` — which is the only way to do this that
stays done, because `run_dir()` re-applies `chmod 0700` on every publish, so the
first version of this run "proved" nothing while the chmod was silently undone by
the very write it meant to block. Twelve seconds of a real daemon whose handover
cannot be written:

```
t= 1.027s check 1: pid_alive=True create=200 retiring_from='' retiring_to=''
...  twelve samples, the create answering 200 throughout, no announcement ...
--- ... writable again at t=12.276s ---
--- t=14.589s: the record announces the handover ---
--- t=19.607s: the daemon LATCHES (refuses new work) ---
--- t=20.870s: the record is removed (clean exit) ---
```

with a WARNING per check naming the file it could not write, and no latch: the
old order latched first, so a failed write left a daemon answering 503 forever
and never leaving.

**5. `--reload` does nothing at all.** 24 samples after the flip: no
announcement, no exit, `/health` 200 throughout, and no build-watch line in the
reload child's log. Round 1 measured the opposite — the child retired, removed
its record and asked its own process to stop while the reloader parent kept the
port, so `/health` timed out with nothing to explain it. That is why the reload
path now runs no build watch: its port belongs to uvicorn's reloader and a
dev-mode supervisor is not a production daemon. When the harness stops the
reloader afterwards, the child exits with it — `records left under run/serve: 0`,
`listeners left on port 51810: 0` — i.e. the replacement shape leaves nothing
behind either.

**6. A probe the daemon cannot read means STAY.** `inject.py` drives the real
poll, the real predicate and a real held desktop bridge, with one thing broken:

```
t=0.222s record announced: retiring_from='0.54.39@1111111' retiring_to='0.54.39@2222222'
t=1.260s with the probe UNREADABLE: latched=False exits=0 poll_alive=True
          verdict(probe broken): 'an in-flight probe that could not be read (the desktop plane)'
t=2.266s with the probe readable again: reason='1 in-flight desktop request(s) on …' exits=0
t=2.461s after the viewer lets go: latched=True exits=1
```

Announced, still serving, not latched and not exited while the probe is broken —
and it still retires the moment the drain really is empty. The injection is at
the probe seam because a raising `stats()` inside a live unprivileged daemon is
not something this harness can produce; the poll, the predicate, the record write
and the log are the real ones.

**7. ONE daemon, both phases, the same 23 routes — and the instrument that shows
the difference.** This is the run round 2 asked for: the routes are identical in
both halves, so the difference between them is the latch and nothing else. Held
across the announcement is the app's own relay, which is what keeps the daemon
from latching out from under the observation.

```
--- refusal matrix, phase 1: merely ANNOUNCED (every route answered) ---
  POST /v1/desktop/sessions (create)             HTTP 200
  GET  /v1/desktop/skills (session in the query) HTTP 200
  GET  .../{id}/mcp (a session route)            HTTP 200
  GET  .../{id} / …/history / …/failovers / …/command-entities   HTTP 200
  GET  .../{id}/variables                       HTTP 200 ("no code memory yet")
  POST/PATCH/DELETE .../{id}/variables…         HTTP 409 {"code":"runtime_cold"}
  POST .../{id}/mcp                              HTTP 503 {"detail":"No model provider is configured yet…"}
  POST .../{id}/credentials                      HTTP 503 (same, untyped)
  POST .../{id}/fork                             HTTP 503 (same, untyped)
  POST .../{id}/asides                           HTTP 503 (same, untyped)
  POST .../{id}/asides (a REPEATED id: 409 once claimed) HTTP 409
  POST .../{id}/asides/{aside}/adopt             HTTP 404
  POST /v1/desktop/stop (a session route)        HTTP 200
  GET  .../{id}/events (the app relay)           HTTP 0 TimeoutError (the stream opened)
  POST .../{id}/messages                         HTTP 503 (same, untyped)
  POST .../{id}/commands                         HTTP 503 (same, untyped)
  POST .../{id}/answers                          HTTP 409
  POST .../{id}/watch                            HTTP 404
  POST .../{id}/warm                             HTTP 200 {"state":"warming"}
  spawn-seam lines the daemon logged during the ANNOUNCED matrix: 18

t=60.551s still admitting: create -> HTTP 200 (announced, not refusing)
--- t=64.603s: the daemon LATCHES (refuses new work) ---
  ... the SAME 23 rows, every one of them the typed 503 ...
  POST .../{id}/asides (a REPEATED id: 409 once claimed) HTTP 409 (an id claimed
    in phase 1: a duplicate is a client error whatever the daemon's state)
  spawn-seam lines the daemon logged during the LATCHED matrix: 0
```

The untyped `503 "No model provider is configured yet…"` in the first half is the
point of the contrast: while announced, these routes answer on their merits — and
one of those merits, in an isolated config root, is failing to start a runtime and
saying so. The latched half carries the ONE refusal sentence instead. And the two
counts above it are why the zero means something: the same instrument, on the same
daemon, over the same routes, named the seam **18 times** while the daemon was
still admitting. What moves between invocations is how many of the routes get far
enough into an engage attempt to log one (19 in the first capture); what does not
move is that it is non-zero while announced and exactly zero while latched, and
`drive.py` fails the run if the announced half is zero at all.

**8. The announcement is RE-READ: a reverted install withdraws it, a further move
re-announces it.** Round 2's MINOR-2: the announcement used to be written once and
acted on for the rest of the process's life, so an install that went back to the
running build still latched, exited and removed its record — telling every reader
to hand over to a build that was no longer on disk. The relay is held across the
whole sequence here, so the daemon cannot latch out from under the observation:

```
--- t=3.349s: the record announces the handover (0.54.45@1111111 → 0.54.45@2222222) ---
--- t=3.349s: putting the install BACK on the boot build (1111111) ---
record after the reversion: retiring_from='' retiring_to=''
create after the withdrawal: HTTP 200
--- moving the install ON, first to the build it announced, then further ---
re-announced: retiring_to='0.54.45@2222222'
moved on again: retiring_to='0.54.45@3333333'
--- t=18.426s: the relay is dropped and the daemon may finish ---
--- t=33.612s: the daemon LATCHES (refuses new work) ---
timings: flip -> announce 3.349s, latch at 33.612s, record gone at 34.533s
```

with the daemon's own log naming both directions of the re-read:

```
[INFO] serve daemon: the handover to 0.54.45@2222222 no longer holds — the install
       on disk is back on 0.54.45@1111111 or no longer readable; withdrawn from the
       record and still serving on this build
[INFO] serve daemon: the install on disk moved on to 0.54.45@3333333 while
       0.54.45@2222222 was announced; the record now names 0.54.45@3333333
```

Note the withdrawal is a RECORD field, not the latch: the daemon went on serving,
answered a create 200 in the withdrawn state, and only refused once its drain was
empty — after which it left for the build that was actually there.

**9. An UNREADABLE build marker means STAY** (QA round 2, OBS-1). The new build is
written, the marker is aged past the settle so "not settled yet" cannot be the
explanation, and the read permission is taken away:

```
--- the new build is on disk, aged past the settle, and UNREADABLE (chmod 000) ---
t= 1.036s check 1: pid_alive=True getting_health=200 retiring_from='' retiring_to=''
…  22 samples, one per second, no announcement and no exit at any of them  …
t=22.407s check 22: pid_alive=True getting_health=200 retiring_from='' retiring_to=''
--- t=22.408s: chmod 644 on the SAME file (still the new build) ---
--- t=24.050s: the record announces the handover ---
--- t=29.053s: the daemon LATCHES (refuses new work) ---
```

`update.source_ref` answers `""` for a marker it cannot read, so the stamp on disk
is version-only (`0.54.45`) and differs from the boot stamp (`0.54.45@1111111`) by
the ref ALONE — which is exactly the same-version-rebuild case the ref exists to
disambiguate. Before the guard this run retired onto a build it could not read
(QA measured announce → latch → exit → record removed); now the same file, made
readable again, retires the same daemon 1.64 s later (well inside one check
interval), which is what makes this a guard rather than a disabled watch.

**10. The two instruments, shown FAILING.** A guard nobody has seen go red is a
guard nobody has tested, and round 2 left two that could not fail. `ungated.py`
drives both, in-process, with the worktree's interpreter:

```
1. the completeness walk, over a scratch copy with ONE ungated route:
  scratch.desktop_lifecycle:544 reaches the pool's bridge cache directly from frobnicate()
2. the spawn-seam spy, with the door's refusal removed on a latched daemon:
  POST /v1/desktop/sessions/460c4acac193/mcp -> AssertionError: the runtime spawn seam was entered
  spawn-seam entrances recorded by the spy: 1
```

The first is the walk that decides whether the refusal matrix is complete: given a
router module with one route that reaches a runtime without the door, it names the
bypass. The second is the assertion that "no runtime was started": with the door's
refusal removed — the state round 2 measured on a live daemon — the same request
through the same route on a latched pool reaches `_ensure_bound` and the spy fires.
Neither instrument can be green for the wrong reason, because neither is green
here.

## What the numbers say about the constants

The announcement lands one settle + one check after the flip (`BUILD_SETTLE_S`
waits for the installer to have finished writing `.lop-source`, `BUILD_CHECK_S`
is the poll interval): 14.204 s in run 1 with production constants, 3.349-4.503 s
with `LOP_BUILD_SETTLE_S=1`. The latch trails the announcement by however long the
drain takes — one check with nothing attached (runs 2 and 4: 4.98 s and 5.02 s) —
and the exit then lands one jittered slice of `BUILD_STAGGER_S` later, drawn from
[0, 20 s). The announcement's *life in the record* is therefore
never shorter than the old notice window, and it is unbounded in the direction
that matters.

The re-read (round 2) costs one stamp read per check interval, on the same
cadence the DETECTION phase has always run at: `handover_build` reads
`installed_build` and compares, and it deliberately skips the settle, because a
move that has already been announced does not need to settle twice. That is why
run 9 announces 1.64 s after the marker becomes readable again and run 8's
reversion is noticed on the next tick rather than a settle later.

## Boundaries this evidence does not claim

- **The app-attached rollout completes only with the UI PR.** Run 3 drops the
  relay by hand, which is what `docs/design-daemon-discovery.md` §7 now specifies
  as the app's obligation. Until `local-operator-ui` implements that valve, a
  real app holds the relay and the watch lease and the daemon announces and keeps
  serving (which is strictly better than the silence it replaces, and is not the
  same as a completed update path).
- **No successor is spawned.** Nothing in the daemon starts a replacement
  process; the record names the build and the log names `lop serve` / `lop
  update`. For a daemon an app owns, re-discovery is the UI's job; this run shows
  the terminal line an unsupervised daemon leaves behind.
- **Once latched, session-scoped READS are refused too** (round 2: the gate is at
  the door every desktop route comes through, so the refusal cannot be partial).
  What stays readable is the record plane, which never takes a bridge: `GET
  /v1/desktop/sessions`, `GET /health` and the record file itself, which is what
  lets a reader observe the handover. Runs 2 and 7 are the measurement; run 3 is
  the same daemon while merely announced, where every read still answers.
- **One 409 is not a hole.** `/asides` keys its off-record entries by request id
  and answers 409 for a repeated one whatever the daemon's state — a duplicate is
  a client error rather than an admission question, and the route claims nothing
  on the way out. Run 7 shows both: a fresh id answering the typed 503, and the
  phase-1 id answering 409.
- **The receipt journal is shared with the successor**, which is why two routes
  ask the refusal before they claim one: `/stop` and `/adopt` claim their receipt
  before they take a bridge, and a claimed-but-unfinished receipt is indeterminate
  for the retry (`DesktopReceipts._claim`, `retry_safe=False`). The evidence does
  not measure that separately; the matrix asserts no receipt is claimed on a
  refusal, and the reasoning lives in the handlers.
- **Legacy admission surfaces are not gated.** `/v1/chat` and `/v1/jobs` still
  accept work on a latched daemon (named as out of scope in the PR body); this
  evidence only covers the desktop plane plus the two per-turn terms.
- **`lop sessions` does not list daemons.** The announcement is visible to a
  reader of `run/serve/<pid>.json` (`scan()` / the desktop app), not in the
  session list — the daemon record lives in its own namespace.
- **The mobile relay is not involved.** `update.refresh_mobile_after_upgrade()`
  already kickstarts it from the updater; this change does not touch it.
- **A turn in a detached runtime is not this process's to cancel.** Turns run in
  `session/runtime/process.py` children, which keep their own residency drain;
  what holds this daemon is the client's VIEW of that turn, which is what run 3
  demonstrates.
- **Windows.** `_request_shutdown`'s documented `TerminateProcess` limit (the
  record left for `scan()` to reap) is not testable on this host.
