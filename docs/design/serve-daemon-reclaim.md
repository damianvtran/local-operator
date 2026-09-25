# Serve daemons: naming a daemon that is not serving, and ending it on request

Design record for the address axis in `local_operator/services.py` and the
`lop services reclaim` verb. Written after the incident of 2026-09-23, where the
operator's desktop app was down for twelve minutes and the only way out was
`kill` from a shell.

## 1. What happened, and the two states that had no name

A `lop serve` daemon (pid 1276) was the desktop app's backend on
`127.0.0.1:1111`. It went **alive, recorded and deaf**: its pid was there, its
record was beating, its heartbeat was fresh — and nothing accepted on its port.
Then a test rig from another session started `lop serve --port 1111` with its own
`LOCAL_OPERATOR_CONFIG_DIR`, so the address was held by a stranger. The app
refused the stranger by identity (correct) and refused to spawn its own (also
correct: minting a token over an occupied address overwrites that daemon's
credential) and then quit, reporting *"Failed to start the Local Operator
backend service"*. The operator could not see who held the port, and nothing in
the product could end the holder.

The shared vocabulary — `live` / `wedged` / `stale`
(`session/runtime/registry.py`) — could not describe either state. It answers
*is the owner there* from the record alone, with no network cost, which is what
makes `scan()` cheap enough to run on every `lop` invocation. It does not ask
whether anything is **serving the address the record names**. A daemon that lost
its listener reads as `live`, and every reader believed it:

| mechanism | what it did for the incident's daemon |
|---|---|
| `registry.scan` | classified it `live`; `wedged` is never reaped, never signalled |
| `services.live_serve_daemons` | returned it as live, so `lop services status` said *"serve daemons: none running"* about a record it had just decided was live and a port that was held |
| `services.reload_serve_daemons` | asked live daemons to move; a `wedged` one was **not mentioned at all** |
| `server/reload.py` | fail-closed: keeps serving, or refuses without a listener fd |
| `server/retire.py` | announces drift, never exits |
| `lop stop`, `lop sessions reclaim` | session plane only — a serve daemon is unreachable by both (`reclaim` refuses any candidate whose record exists) |

So the machine had no answer to *"who holds 1111?"* and no command that could
say *"end it"*. That is the whole of what this change adds.

## 2. The mechanism

### 2.1 One owner for the address axis

`services.probe_address(record)` — beside the identity probe that already owns
"prove the process before touching it" — asks the record's address who is
serving it, in ONE loopback round trip, and returns a structured
`AddressProbe`. The sentence every existing caller wants is still produced by
`_answers_as_record`, now a renderer over it, so the serve-reload path kept its
contract and behaviour unchanged.

The verdicts, composed in `ServeDaemonReport` with the shared classification.
**The probe outranks the beat wherever a probe ran**: a `wedged` record whose
address answers as its own instance IS `serving` (a stale beat is not a verdict on
the process — `server/registry.py` is explicit about that), and `wedged`/`stale`
keep their own word only where the address agrees nobody is serving.

| verdict | evidence | what a reader does with it |
|---|---|---|
| `serving` | the probe answered with the record's own `instance_id` | nothing; this is the only state that may be signalled a reload |
| `deaf` | the address did not answer at all (refused, timed out, reset before a response) | the owner's loop is alive and it is not serving — the state the incident was in |
| `squatted` | something answered and it was not this record's daemon: a different `instance_id`, a non-200, an unintelligible response, or no instance at all | somebody else holds this address |
| `wedged` / `stale` | the shared classification, when the address agrees nobody is serving | as before |

`deaf` and `squatted` are deliberately distinct, and the line between them is
**an answer versus a silence** rather than "is anything listening": the first says
*nobody answered*, the second says *something answered and it is not this record's
daemon*, and only the second describes a process the operator could go and look at.
A port that accepts a connection and then speaks something that is not this
product's health endpoint is therefore `squatted`, not `deaf` — the sentence for
the first version of this split ("nothing is answering there") was about a port
that had just answered.

### 2.2 Report it, in the two surfaces that used to be silent

- `lop services status` now prints a row per recorded-but-not-serving daemon
  (address, pid, the state in words, and the command that ends it). Additive:
  every existing line is unchanged, and `none running` is still printed on a
  machine with no records at all.
- `reload_serve_daemons` (the `lop update` / `lop services restart` path) reports
  the daemons it cannot ask to move, *before* its own "nothing to say" guard, and
  signals exactly what it signalled before — nothing, for these.

A `wedged` record is probed too, because its address can have been taken by
somebody else, and "your stuck daemon's port is now held by a stranger" is a
different sentence from "nothing answers there". A `stale` record is not probed:
its pid is gone and the next scan reaps it.

### 2.3 `lop services reclaim <pid>` — asked for, proven, bounded

The only destructive verb in the `services` group, and the only path in the
product that can end a stray serve daemon. It refuses, in this order:

1. **The pid must be this product's serve daemon now.** The command line is read
   and matched as a WORD SEQUENCE — the spawn contract, never a substring, the
   same rule `session/runtime/reclaim.parse_process_row` states for runtimes, so
   a person running `grep 'local_operator.cli serve'` is not mistaken for a
   daemon. THREE spellings are accepted, in one predicate:
   - the branded interpreter, `-m local_operator.cli serve` (any position);
   - the launcher by basename (`…/lop serve`, `…/local-operator serve`) **at
     `argv[0]` or immediately after the `procname` label**, because `procname`
     REPLACES `argv[0]` — a live labelled daemon reads
     `Local Operator [serve] port=18490 /Users/…/local-operator serve --host …`,
     and the label pattern is derived from `procname.LABEL_SERVE` rather than
     typed again;
   - the **desktop app's managed backend**, `<interpreter> -c "from
     local_operator.cli import main; main()" serve …`, which is deliberately never
     branded (`procname` refuses a `-c` launch on purpose: the app verifies its
     backend by asking the same string to report `sys.executable`, and a re-exec
     through the branded link would fail that check on every machine). Without
     this third spelling the predicate refuses the app's own backend — the daemon
     that held 1111 in the incident.
2. **The same uid.** A process belonging to another account is refused with the
   `sudo` remedy named.
3. **The verdict must be one that is not serving.** `serving` refuses outright —
   a working plane is never ended by this command. `deaf` and `squatted` must
   hold across `PROBE_CONFIRMATIONS` consecutive probes, because one refused
   connection is not evidence: the loop may be mid-restart or the host may be
   starved (this host ran at a load average of 130 for hours). An unreadable or
   raising probe sends nothing.
4. **Re-identification at signal time, and one last address reading.** The command
   line is re-read immediately before the signal, and so is the ADDRESS: a daemon
   that begins serving between the confirmation and the signal — the recovery this
   command must not punish — is refused, as is an address whose occupant changed
   under the verdict (`problem="changed"`). Both refusals send nothing and say to
   run it again.

Then `SIGTERM` → bounded wait (`RECLAIM_TERM_GRACE_S`) → `SIGKILL` → bounded wait,
with a receipt that names the pid, the address, the verdict and which signal
ended it. The daemon is signalled, never its children: a serve daemon may have
spawned session runtimes, and those are conversations that keep their own state.

**It is never automatic.** A serve daemon may own scheduler runs and be mid-drain
for a runtime spawn, and no reader of a record has a successor-readiness proof —
the reasons `server/retire.py` refuses to exit on a build change apply here word
for word. The operator's rule (destructive and irreversible acts need approval)
is applied to a process: the pid they name is the approval, and the verdict they
read is the evidence.

## 3. Rejected alternatives

- **A machine-wide address reservation, with `lop serve` refusing a reserved
  port.** The direct approach, and the trap. "Owned by the plan's app" is not a
  property of an address: the port is a configurable default
  (`backend-service.ts`), and the same 1111 is documented for
  `docker-compose.yml`, the `Dockerfile` and a plain `curl`. A rule keyed on it
  would refuse the default port to every rig, test and developer while any
  wedged record on the machine named it — and it could not see the incident's
  squatter anyway, whose records live in another config root by design (that
  isolation is correct and paid for). Prevention belongs on the app's side: an
  address it did not choose must never cost it its backend (see §5).
- **Auto-reaping a wedged daemon.** Covered above: no successor-readiness proof,
  and the process may be doing work. Reported and reclaimable-on-request instead.
- **Adding an ``listening`` field to the serve record.** The record would then
  claim a fact about the socket, and a record is rewritten on a timer: a daemon
  killed between the socket closing and the next beat would publish a lie, and a
  reader would have to distrust the field it just added. The address is
  probed instead — the same reason `server/registry.py` says a 200 alone is not
  identification.
- **Widening the shared classifier with the address axis.** Its two-fact
  contract (pid liveness, heartbeat freshness, no network cost) is load-bearing
  for `scan()`'s cost on every `lop` invocation, and ~15 call sites read it
  positionally. The axis lives in the services layer, where probing already
  lives.

## 4. Interaction points this must not break

- **`lop services status` / `restart`**: lines and receipts for live daemons are
  unchanged; `restart` still signals only `live`, `reloadable`,
  identity-proven daemons. The report reads the fleet from ONE source now, so a
  record gets exactly one row — the property `_fleet_action_lines` documents for
  `grep`/`awk` counting.
- **`lop update`**: a failed nudge is still a warning on a successful update.
  The new stuck-daemon warnings are additive `ServiceRefresh` warnings, printed
  by the same printer.
- **The serve-reload path**: `_answers_as_record` and `probe` are unchanged in
  contract and behaviour;
  `tests/unit/test_services.py`'s identity and IPv6 assertions pass untouched.
- **The desktop app**: reads the record and probes `/health` exactly as before.
  Nothing in the record's shape changed.
- **Isolated and CI runs**: nothing here reads another config root, writes
  outside its own root, binds an address, or signals anything without a pid the
  operator named. No test binds 1111.
- **Session runtimes**: untouched — they are a different plane and this module
  never ends one.

## 5. Follow-ups this record does not cover

1. **The app must not die from an address it did not choose.** The other half of
   the incident: two independent gates (`blocksSpawn` on a record in its own
   root; the occupancy gate on the configured origin) each refuse, and the app
   then quits instead of degrading with a diagnosis. The renderer's CSP pins
   `connect-src` to `127.0.0.1:1111` and `:8080`, so a fallback address is the
   second address the renderer already trusts, or a CSP change — a security
   decision, not a bug fix. Tracked for the UI repository.
2. **Why a daemon loses its listener and keeps beating** is still unknown. The
   record cannot express it (§3), the reload path's smoke check only guards the
   successor, and no code path read in this investigation produces the state.
   Settle it with a real reload under a listener poll, plus the negative control
   (a planted deaf process) to prove the instrument can see an absence.
3. **A machine-level address *claim*** (a uid-scoped record keyed by address,
   published outside any config root) is the mechanism to reach for if operators
   still cannot tell who holds an address when the holder's root is not theirs,
   or if two planes on one address ever produce a silent wrong-attach rather than
   a reported refusal. Not built: nothing observed so far needs it, and it must
   not become a lock — absence must never refuse a bind.
4. **The ``-c`` entry point is transcribed, not shared.** `SERVE_ENTRYPOINT_WORDS`
   mirrors ``local-operator-ui``'s ``SERVE_ENTRYPOINT``
   (``src/main/backend/owned-serve-launch.ts``); nothing on either side can detect
   drift, and a change there silently re-creates the refusal of the app's own
   backend (review round 4, R4-2, recorded only). A cross-repo constant is not
   available today: the app is TypeScript and this is Python, and the two ship
   independently.
5. **What the ``-c`` search admits, stated plainly** (review round 5, R5-1): an
   argv carrying an earlier unrelated ``-c``, and a generic shell cell wrapping the
   spawn (``bash -c 'exec "$@"' owned-serve … -c <entry> serve``), are accepted as
   well as the app's own backend. Both still carry this product's serve entry point
   followed by the verb, which is the whole of what a STRAY proof has — and the
   app's plan execs, so no nameable pid is ever that ``bash``. Recorded rather than
   narrowed: narrowing it is what refused the app's own backend in round 3.
