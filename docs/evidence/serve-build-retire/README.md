# Retiring the `serve` daemon onto a new build

Raw-process evidence for the daemon's build watch: an install replaced on disk
under a running `lop serve`, and the daemon's answer to it — announce the
handover in its rendezvous record, refuse new session spawns, leave when nothing
is in flight.

Three daemons, each on an ephemeral port (`--port 0`) under its own
`LOCAL_OPERATOR_CONFIG_DIR`, each pointed at a FAKE install root with
`LOP_BUILD_PREFIX` and driven by flipping that root's `.lop-source` — the file
`lop-update` writes last, and the only signal this feature reads. No live daemon
on the machine is touched, and the operator's exported desktop token is scrubbed
per run so the ungoverned runs are genuinely ungoverned.

```sh
bash docs/evidence/serve-build-retire/run.sh     # ~2 minutes, 3 daemons
```

`transcript.txt` is a verbatim run. `drive.py` is the driver (it starts nothing;
`run.sh` starts the daemons), and it only ever reads the record file, posts one
HTTP request, and watches the pid.

## What the runs show

**1. Unsupervised — announce, then leave.** The record exists with
`retiring_from: ""`/`retiring_to: ""`; after the marker flips the record
announces `0.54.39@1111111 → 0.54.39@2222222` (both build labels), the log names
the build and the command that brings the daemon back, and the clean exit
removes the record.

| measured | value |
| --- | --- |
| flip → announce | 14.46 s (settle 10 s + one check interval 5 s) |
| announce → exit ("notice") | 20.21 s (jittered over the 20 s stagger) |
| record present after exit | False |
| pid alive after exit | False, 0.02 s after the record was removed |

**2. Claimed — the refusal.** Same sequence, plus a `POST /v1/desktop/sessions`
issued the instant the announcement was readable:

```
--- new-session request while retiring: HTTP 503 ---
{"detail":{"code":"daemon-retiring","message":"This backend is restarting onto a
new build and is not accepting new sessions. Reconnect to the new backend and
retry."}}
```

A typed 503 with a code, not a 500 — so the client's move is to rediscover the
successor through the record rather than retry here. This run sets
`LOP_BUILD_STAGGER_S=10` (the documented e2e-only override) to widen the notice
window for the request; the announce is still ~14.8 s after the flip.

**3. Held stream — the drain.** An SSE job stream is held open, then the marker
flips:

```
verdict: 14 samples across a held stream, no retirement
2026-09-14 00:55:26 [INFO] retire: build 0.54.39@2222222 is on disk but 1 SSE
  subscription(s) is in flight; retiring when it completes
```

14 s of samples (more than two 5 s check intervals) show the pid alive and the
record untouched. The stream is released at t=14.1 s and the daemon announces at
t=18.8 s — the next check, plus the 1 s settle this run sets — then exits and
removes the record.

## What the numbers say about the constants

The announce lands one settle + one check after the flip (`BUILD_SETTLE_S` waits
for the installer to have finished writing `.lop-source`, `BUILD_CHECK_S` is the
poll interval), and the exit lands one jittered slice of `BUILD_STAGGER_S` later
— so the record carries the handover for that whole slice. Observed notice
windows: 20.2 s, 8.6 s (override 10 s) and 11.8 s, all inside the configured
bounds.

## Boundaries this evidence does not claim

- **No successor is spawned.** Nothing in the daemon starts a replacement
  process; the record names the build and the log names `lop serve` / `lop
  update`. For a daemon an app owns, re-discovery is the UI's job; this run
  shows the terminal line an unsupervised daemon leaves behind.
- **`lop sessions` does not list daemons.** The announcement is visible to a
  reader of `run/serve/<pid>.json` (`scan()` / the desktop app), not in the
  session list — the daemon record lives in its own namespace.
- **The mobile relay is not involved.** `update.refresh_mobile_after_upgrade()`
  already kickstarts it from the updater; this change does not touch it.
- **A turn in a detached runtime is not this process's to cancel.** Turns run in
  `session/runtime/process.py` children, which keep their own residency drain;
  what holds this daemon is the client's VIEW of that turn (term 1–3 of
  `server/retire.py::in_flight`), which is what run 3 demonstrates.
