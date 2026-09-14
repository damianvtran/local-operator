# Retiring the `serve` daemon onto a new build

Raw-process evidence for the daemon's build watch: an install replaced on disk
under a running `lop serve`, and the daemon's answer to it — **announce the
handover in its rendezvous record immediately, keep serving, and only once
nothing is attached latch against new work and leave.**

Six runs, each on an ephemeral port (`--port 0`) under its own
`LOCAL_OPERATOR_CONFIG_DIR` and pointed at a FAKE install root with
`LOP_BUILD_PREFIX`, driven by flipping that root's `.lop-source` — the file
`lop-update` writes last, and the only signal this feature reads. No live daemon
on the machine is touched, the operator's exported desktop token is scrubbed per
daemon so the ungoverned run is genuinely ungoverned, and every `CMUX_*`
variable is unset so nothing here can reach the operator's own workspaces.

```sh
bash docs/evidence/serve-build-retire/run.sh     # ~4 minutes, 5 daemons + 1 injection
```

`transcript.txt` is a verbatim run (741 lines; `run1.txt`…`run6.txt` are the
per-run slices, echoed inline). `drive.py` starts nothing — `run.sh` starts the
daemons — and it only ever reads the record file, makes ordinary HTTP requests
and watches the pid. `inject.py` drives the real poll in-process for the one
fault a real daemon cannot be given.

## What the runs show

**1. Unsupervised — announce while serving, then leave.** The record exists with
`retiring_from: ""`/`retiring_to: ""`; after the marker flips, the record
announces `0.54.39@1111111 → 0.54.39@2222222`, the daemon answers requests the
whole time, the log names the build and the command that brings the daemon back,
and the clean exit removes the record.

| measured | value |
| --- | --- |
| flip → announce | 14.83 s (settle 10 s + one check interval 5 s) |
| announce → record gone | 22.55 s (the drain was empty, so the latch came on the next check + a jittered slice of the 20 s stagger) |
| record present after exit | False |
| pid alive after exit | False, 0.2 s after the record was removed |

**2. Claimed — announced-but-admitting, then the refusal matrix.** The same
daemon answers a create **200** while its record announces the handover, and only
answers the typed refusal once its drain has emptied:

```
t= 4.817s still admitting: create -> HTTP 200 (announced, not refusing)
--- t=9.840s: the daemon LATCHES (refuses new work) ---
create while latched: HTTP 503 {"detail":{"code":"daemon-retiring","message":
  "This backend is restarting onto a new build and is not accepting new work.
   Reconnect to the new backend and retry."}}
```

Every path that can admit or start work, on that one latched daemon:

```
  POST /v1/desktop/sessions (create)             HTTP 503 daemon-retiring
  POST /v1/desktop/sessions/{id}/warm            HTTP 503 daemon-retiring
  POST /v1/desktop/sessions/{id}/messages        HTTP 503 daemon-retiring
  POST /v1/desktop/sessions/{id}/commands        HTTP 503 daemon-retiring
  POST /v1/desktop/sessions/{id}/answers         HTTP 503 daemon-retiring
  (run/mobile records: no run/mobile directory at all)
```

The last line is the "no runtime was started" claim: every spawn publishes a
runtime record under `run/mobile`, and after `/messages` and `/commands` — the
two that reach `_ensure_bound` — the directory does not exist at all. The run
sets `LOP_BUILD_SETTLE_S=1` and `LOP_BUILD_STAGGER_S=300` (the documented
test-only overrides) so the whole matrix fits inside the refusal window; the
daemon's own announcement and refusal ordering is unchanged by them. It is
stopped with a plain SIGTERM, which still exits cleanly and removes the record:
`record after the stop` shows an empty `run/serve`.

**3. The app's own relay — the case this feature exists for, and the one round 1
measured as impossible.** `DesktopStreamRelay`'s exact request
(`GET /v1/desktop/sessions/{id}/events`) is held open across the update, at
production constants:

```
--- t=14.526s: the record announces the handover ---   (retiring_from/retiring_to set)
t=15.570s holding: pid_alive=True getting_health=200 retiring_to='0.54.39@2222222' relay_frames=6
... ten samples, one per second, two check intervals ...
t=24.731s holding: pid_alive=True getting_health=200 retiring_to='0.54.39@2222222' relay_frames=6
--- t=24.733s: dropping the relay (the view closes) ---
t=30.650s still admitting: create -> HTTP 200 (announced, not refusing)
--- t=34.546s: the daemon LATCHES (refuses new work) ---
--- t=47.350s: the record is removed (clean exit) ---
```

The daemon's own log names the term holding it:

```
[INFO] serve daemon: build 0.54.39@2222222 is on disk and announced; 1 in-flight
       desktop request(s) on 65524-… is still in flight, so it keeps serving until
       that completes
```

Two details worth reading off it. The announcement is **readable while the
daemon keeps serving** (round 1's blocker was that it never appeared at all),
and the latch trails the relay's drop by ~10 s: the server notices an abandoned
stream at its next write, which is the 15 s heartbeat cadence — during which the
daemon is still admitting work, exactly as the design says it must while merely
announced. Dropping the relay is the client's act, which is why it is
specified for the UI in `docs/design-daemon-discovery.md` §7.

**4. A write that fails neither latches the daemon nor stops the poll.** The
record's directory is made `UF_IMMUTABLE` — which is the only way to do this that
stays done, because `run_dir()` re-applies `chmod 0700` on every publish, so the
first version of this run "proved" nothing while the chmod was silently undone by
the very write it meant to block. Twelve seconds of a real daemon whose handover
cannot be written:

```
t= 1.103s check 1: pid_alive=True create=200 retiring_from='' retiring_to=''
...  twelve samples, the create answering 200 throughout, no announcement ...
--- ... writable again at t=12.277s ---
--- t=14.763s: the record announces the handover ---
--- t=19.816s: the daemon LATCHES (refuses new work) ---
--- t=24.738s: the record is removed (clean exit) ---
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
`listeners left on port 50882: 0` — i.e. the replacement shape leaves nothing
behind either.

**6. A probe the daemon cannot read means STAY.** `inject.py` drives the real
poll, the real predicate and a real held desktop bridge, with one thing broken:

```
t=0.214s record announced: retiring_from='0.54.39@1111111' retiring_to='0.54.39@2222222'
t=1.215s with the probe UNREADABLE: latched=False exits=0 poll_alive=True
          verdict(probe broken): 'an in-flight probe that could not be read (the desktop plane)'
t=2.216s with the probe readable again: reason='1 in-flight desktop request(s) on …' exits=0
t=2.232s after the viewer lets go: latched=True exits=1
```

Announced, still serving, not latched and not exited while the probe is broken —
and it still retires the moment the drain really is empty. The injection is at
the probe seam because a raising `stats()` inside a live unprivileged daemon is
not something this harness can produce; the poll, the predicate, the record write
and the log are the real ones.

## What the numbers say about the constants

The announcement lands one settle + one check after the flip (`BUILD_SETTLE_S`
waits for the installer to have finished writing `.lop-source`, `BUILD_CHECK_S`
is the poll interval): 14.83 s, 14.53 s and 14.76 s with production constants, and
4.79 s with `LOP_BUILD_SETTLE_S=1`. The latch trails the announcement by however
long the drain takes — one check with nothing attached (run 4: 5.05 s), tens of
seconds with the app's relay held (run 3: 20.0 s) — and the exit then lands one
jittered slice of `BUILD_STAGGER_S` later. The announcement's *life in the record*
is therefore never shorter than the old notice window, and it is unbounded in the
direction that matters.

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
