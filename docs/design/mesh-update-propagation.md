# Mesh update propagation — the trigger: a primary's update rolls the mesh

Status: **design, pre-implementation**. Verified against `origin/main` @ `ff97ee6996`
(v0.68.21), 2026-10-09.

Companion to `mesh-rolling-updates.md` (cited as *RU §n*). RU decided what may happen to a
peer — the receiver-side `update` capability and its trust bound (§2), the member pass (§4),
the record (§6) — and left the trigger at one sentence: "the origin's own completed update,
under a per-network policy" (§3). This note specs that trigger far enough to implement: the
hook, the scope, the pass, the switch, the audit. It adds **no wire op, no capability and no
protocol change**. It consumes `mesh-remote-onboarding.md` §2.1/§2.4/§2.5 (the approval and
authority vocabulary it must not fight) and the operator's local drain discipline
(`~/tools/lop-fleet-update/docs/README.md`).

The intent, in the operator's words (2026-10-05, quoted in RU's opening): *"if the primary
host updates, it will also trigger an update across the mesh in the same safe, rolling way
which waits for runtimes to complete/idle and then updates them."*

---

## 0. The answer in one page

**The problem.** RU is a mechanism with no starter. `network.rollout_on_update` is a name in
RU §3 and nowhere else (§1), and both update front ends finish without a thought for the
mesh: `lop update` prints a frozen report line and returns, and `/update` relaunches the TUI.

**The decisions, in order of consequence.**

1. **Authority (§2) — recorded, not re-opened.** The operator's standing instruction is the
   authority for the primary's decision to propagate its own update. It replaces a per-update
   card. It does not replace the member's own `update` grant, whose bound the operator
   ratified the same day (RU §2).
2. **Trigger (§3) — a hand-off through a durable record.** A successful move to a strictly
   newer version writes an `active` rollout record and one audit row, then the front end
   returns. The **relay is the only driver**; the front end never calls a peer.
3. **Scope (§3) — membership, not links.** Every other active member of every network, minus
   pool members. Whether a member granted `update` is learned from *its* answer, never guessed.
4. **The pass (§4) — serial apply, skip-then-retry.** One member applying at a time; a busy,
   failed or offline member is parked and the pass moves on; nothing is ever forced; every
   state re-enters from the record.
5. **The switch (§5) — `network.rollout_on_update`: `auto` (default, ON) or `off`**, plus
   `lop update --no-roll` and `lop network update --cancel`; re-read before every member.
6. **Audit (§6) — one durable row naming the authority** when the record opens, one row per
   member state change, one closing row; the member writes its own, naming its own basis.

**What this note changes in RU**, itemised so a reviewer can check each off: (a) RU §3 "minus
members that do not hold the grant" — the origin learns that from the refusal (§3); (b) RU §3
policy value `ask` — deferred (§5); (c) RU §4.2 wait-per-member — skip-then-retry (§4);
(d) RU §9.6 driver — relay only (§4); (e) RU §6.4 — `update_member_triggered` folds into
`update_member_state`, and `update_rollout_suppressed` is added (§6); (f) RU §7.3 summary —
at open it names the queued set and the skips known without a call; outcomes arrive on the
peers rows and `lop network update --status` (§3).

```
primary: lop update, /update         primary relay (mesh-rollout)           member
  ├─ install moved v_old → v_new
  ├─ services tail (relay rolled)
  ├─ open_after_update():  v_new > v_old?  policy?  --no-roll?
  │    write rollouts/ro_….json ───────► scan (≤ 15 s) picks it up
  │    audit update_rollout_started        for each member, one at a time:
  │    print "rolling to N peers"            net_update {target, rollout} ──►  probe, lock
  └─ return (report line stays last)                        ◄── busy, done, refused, failed
                                           park busy / failed / offline; re-ask on the cadence
```

---

## 1. Ground truth this adds to RU §1 (verified at the base)

- **Three exits, one hook seam.** `lop update` has two successful-install exits —
  `--from-snapshot` through `_generation_upgrade` (`update.py:6828`, via `_snapshot_command`
  `:6856`; the `lop-update` script execs it) and the PyPI branch's inline tail
  (`:7137-7158`) — and one no-install exit (`:7104-7120`, services tail only). All end
  `_run_services_tail` (`:6764`) then `_emit_update_report` (`:6781`), the frozen machine line
  the desktop parses, documented as *last*. `/update` skips that tail: `perform_upgrade`
  (`tui/app.py:16782`), `refresh_service_daemons_after_upgrade` (`:16802`), then
  `_request_relaunch(force=True)` (`:16843`) — the process exits, so nothing after it can drive
  anything. `--no-services` is the precedent for a per-run flag on this tail (`cli.py:1771`,
  mapped at `:14570`). **Base drift:** every file this note cites except `tui/app.py` is
  byte-identical between the verified base and the fleet's installed build (`00940f0cb8`);
  `app.py` grew 138 unrelated lines (the `_run_update` anchors below are unchanged), and
  `origin/main` has since moved four commits touching none of them.
- **The member's relay is rolled by the update it just ran.** The services stage restarts a
  relay left a generation behind (`update.py:6147` → `relay.py:11701`), and RU §4.5 puts the
  executor in that relay. (Derived from those two; Q4.)
- **Membership, not links.** `DefinitionsSyncer._targets` walks every other `active` member of
  every network (`definitions.py:2236`), because a walk over live links "fired NEVER" on a
  quiet mesh (`:2140-2148`, measured).
- **The requester cannot read the grant.** The peer authorises against *its own* row for the
  requester (`definitions.py:2303-2335`), and a self-decided scope is written only on the
  receiving device (`types.py:526-543`; `update` joins `SELF_DECIDED_SCOPES` per RU §2). A
  refusal asked on a timer wrote 20 refusals in 300 s until it was parked
  (`definitions.py:2137-2212`; `REFUSED_MIN_INTERVAL_S`, `:2274`).
- **Refusal path.** A missing capability is `not_authorised` (`authorizer.py:211-228`),
  audited on the member as the durable `authorisation_refused`, cause `capability_denied`
  (`authorizer.py:459-465`; `audit.py:519-541`). An op the build lacks is `unknown_op` (`:318`).
- **Audit is a closed taxonomy.** `EVENT_KINDS` (`audit.py:112`), `CAUSES` (`:259`), the
  per-event `DETAIL_KEYS` whitelist (`:318`), `DURABLE_EVENTS` (`:519`); a drift test asserts
  every emitted kind is registered (`:109-111`). Write-through is for rows whose loss "would
  hide an ATTACK or a MEMBERSHIP change", and an *authority decision* qualifies
  (`audit.py:513-519`, `:551-558`). Non-relay writers exist and are best-effort
  (`approvals._audit`, `approvals.py:1781`).
- **Two predicates for idle.** `is_busy()` (`serving.py:1627`) is "may this runtime exit",
  inclusive (turn, compaction, goal loop, subagents, parked gate, queued prompts, background
  jobs); `is_conversationally_active()` (`:1681`) is the spinner's, narrower on purpose after a
  measured 8-of-8 false `busy` (`:1697`). A runtime retires for a newer build on
  `may_refresh()` (`:2597`) = `is_busy()` plus a warm-window wake; one that keeps declining
  becomes hard-stale after 3 checks or 30 min and leaves at its first idle instant
  (`process.py:300-301`). The fleet tool drains on the registry `busy` bit
  (`SessionRecord.busy`, `session/runtime/types.py:1325`); where the runtime publishes
  that bit was not established by name (Q1).
- **Not built.** No hit in `local_operator/` or `tests/` at `00940f0cb8`, on any remote ref
  (pickaxe since 2026-10-05), or in the installed v0.68.21 (generation
  `20261010T005620Z-00940f0cb8e1`), for `network.rollout_on_update`,
  `net_update`, `mesh-update-v1`, the `update` capability, or any rollout record or thread.
  The four commits since touch `tui/app.py` only among the files cited here.

---

## 2. Authority — the standing instruction, recorded

**Decision (settled; recorded here, not re-opened): the operator's standing instruction is the
authority for a primary's decision to propagate its own update to the mesh.** It is derivable
from the record: the instruction itself (RU's opening, 2026-10-05); the operator's decisions on
PR #2003 (comment of 2026-10-05, "the note's three open questions are RESOLVED": roll
automatically after the origin's update settles, with the `--no-roll` escape; published
releases from each device's own channel only; re-engage displaced sessions); and the team
brief for this note (2026-10-09), which restates it as settled.

**Fences in the record, so the settlement is not read wider than it was given.** This
note's scope is the trigger's shape. The operator's call, not this note's to change: the
default value of the switch, whether the standing instruction may ever stand in for the
member's own `update` grant, whether `ask` returns as an intermediate, and whether the
standing path ever gains a forced arm. Where this note reports what was decided, the record
is the cited PR thread and the instruction quoted in RU; where it says *Recommend*, that is
this note's own judgement — the two are kept apart in every section.

In the vocabulary the onboarding design owns, so nothing here fights it:

- It is a **standing instruction, not an approval.** An approval is a single-use, expiring,
  signed record answering one card (onboarding §2.1, §2.4). Nothing here is signed, single-use
  or expiring; no card is filed and no `approvals` state is borrowed.
- It is **not operator authority.** The anchor and signing root (onboarding §2.5) are neither
  used nor needed: the install runs as the member's own login, with no sudo.
- It is **not a scope on a card.** `approve` and `unattended` say what a peer may do on *this*
  device; the instruction says what the primary may *decide*.
- **The safe direction is ordinary** (onboarding §2.1, "deny is ordinary"): switching
  propagation off is a plain config write or flag from any surface; being on needs no gesture.

What it authorises: on completing its own update to a newer version, the primary **decides**
to ask each member to move to that version, with no per-update confirmation. What it does not
do, so the settlement is not read wider than it was given:

- It does **not** replace the member's own `update` grant (RU §2). That grant is the member's
  consent to be updated by *this* peer; the operator ratified its bound on 2026-10-05 ("the
  trust bound holds as written"), and no wire op may write it (`types.py:526-543`). A member
  without it is skipped, by name, with the remedy. The trigger is indifferent to how the
  member-side gate is authorised: a later decision to let the instruction stand in for the
  grant on the operator's own devices would change one table row and the onboarding card, not
  this note.
- It does not widen the bound: a version published on the member's own channel, strictly
  newer, nothing from the peer but a target identity (RU §2, rules 1-4).
- It does not authorise a forced update, a downgrade, or a cascade (§3, §4, §8).

---

## 3. Trigger and scope

**Decision: the primary's update process decides, once, when the install settles; it opens a
durable record and returns; the relay drives.** Nobody else decides — not a member, not a
model, not a peer's frame. "Primary" is not a configured role: it is whichever device its
peers have granted `update` (RU §3) — in practice the device the operator updates. A device
nobody granted rolls nobody; the first answer tells it so.

*Hook.* One helper, `rollouts.open_after_update(before, after, via, no_roll)`, called from the
two successful-install exits **between** `_run_services_tail` and `_emit_update_report` (the
summary precedes the frozen report line) and from `/update` after
`refresh_service_daemons_after_upgrade()` (`app.py:16802`), before the relaunch. Not from the
no-install exit, and not from the daemon-refresh child (`update.py:6097`): that child runs
concurrent repairs behind marker lines the desktop parses, and it also runs when nothing was
installed. The helper reads the store and writes files — no network — and **never changes the
exit code**: a failure prints one sentence ("updated; the mesh roll could not be queued: …")
because the install is already in.

*Fires only when all hold.* (1) The install moved to a **strictly newer version** — `after >
before` by `update.parse_version`, the comparator `readiness.compare_builds` uses
(`readiness.py:1079`). A snapshot that changes only `source_ref` (the operator's `lop-update`
from `main` between releases keeps the last tag's version) gives members nothing to do: they
compare versions (RU §2, rules 3-4). (2) Policy is `auto` and `--no-roll` was not given (§5).
(3) This device has at least one active member (otherwise no row and no record). Editable and
unknown installs are refused earlier (`update.py:7122-7128`).

*What it writes.* `<config>/network/rollouts/ro_<id>.json`, one per (origin, network), in
RU §6.1's shape and write discipline, plus an `authority` object — `{authority:
"standing_instruction", policy: "auto", via:
"update", "snapshot" or "tui"}` (`via` names the front end) — and the durable audit row
(§6), both before the front end returns. The record keeps RU §6.1's `origin` as this
device's own id and adds `network`, which is what `rollouts/` filenames and the peers-row
lookup key on. It is a file write because the
relay may be mid-restart at that moment (the services tail just rolled it). Opening
supersedes an older `active` record for the network (RU §6.3). The summary names who is
queued and who is already skipped; `/update` prints it as a notice.

*Cascade.* An update run **on behalf of a peer's request** (the member executor's installer
child, RU §4.2 step 3) runs with `no_roll`: a propagated update never propagates. Symmetric
grants are safe anyway — the second origin's pass finds `already_on_target`.

*Scope.* Per network the origin belongs to, the walk of `DefinitionsSyncer._targets`: every
other `active` member, snapshotted at open in network order (a later admission rides the next
rollout). Before any call: self is out; `kind: "pool"` (`types.py:1236`) is
`skipped (unsupported_kind)`. After the first answer, never before: a member that does not
advertise `mesh-update-v1` in the negotiated features (`wire.py:113`, `:709`) is not sent the
op — `skipped (predates_rolling_updates)`; a member answering `not_authorised` has not granted
`update` — `skipped (no_grant)` with RU §2's remedy sentence, **parked for this rollout** and
never re-asked on the cadence, until the next rollout or an explicit `lop network update
<peer>`. Do **not** pre-filter with `unholdable_capability` (`definitions.py:2303`): for a
scope written only on the receiver it would skip every member. *Offline:* a dial failure
within the connect budget is `unreachable`; the record stays `active` and the cadence re-asks
until the window (24 h, RU §6.1) ends — "caught on next contact", with the cadence as the
contact detector (Q3).

*The first release.* No member advertises `mesh-update-v1` or holds the grant before this
ships, so the release carrying it is the last that needs a hand update on each existing
member, and the grant itself is a node-side act (RU §2): by hand once per existing member, on
the onboarding card for new ones. The first rollout after it reports the skips, visibly (§7).

---

## 4. The pass: rolling, idle-gated, never forced, resumable

*Driver.* One relay thread, `mesh-rollout`, registered through the `on_start` hook shape of
`definitions.install` (`definitions.py:2352-2366`). It scans `<config>/network/rollouts/` every 15 s
(a directory listing, so no wake-up op and no new local op) and, on relay start, resumes every
`active` record at its first non-terminal member. **Its own thread, not a tick step on the
definitions thread** (`add_tick_step`, `:2063`): an apply is a slow call bounded at 900 s plus
`SLOW_REPLY_MARGIN_S` (RU §3; `onboard.py:107-110`; `relay.py:344-355`) and would stall
definitions sync for every member meanwhile. The CLI never drives: a foreground driver dies
with its terminal (`/update` literally exits) and two drivers need a lease; `lop network update
--wait` polls the record instead.

```
for member in record.members (snapshot order), one at a time:
    skip if terminal, or member.next_ask_at > now
    if record.authority is standing_instruction and policy != auto: stop          # §5
    reply = net_update {target, rollout}                # a fresh probe on every ask
    busy                -> deferred(busy), next_ask_at = now + 300 s;          continue
    done, already_on_target, ahead_of_target, refused(terminal) -> terminal;   continue
    failed              -> failed, next_ask_at = now + backoff(n);             continue
    not_authorised      -> skipped(no_grant), parked for this rollout;         continue
    dial failed, silent -> unreachable (deferred(no_answer) if an apply began); continue
record done when no member is non-terminal, or the window lapses
```

- **One at a time** is the loop itself: an apply returns, or times out, before the next ask. A
  `busy` reply costs milliseconds (RU §3), so a pass over N members is N probes plus the
  applies that were possible.
- **Busy is skip-then-retry, not a head-of-line wait.** RU §4.2 waits up to 15 min per member
  before deferring. Here a member that answers `busy` is parked after that single probe and
  re-asked every 300 s, a *fresh* probe each time (the `MOVE_WAIT_POLL_S` shape,
  `mobility.py:90`), so one busy member cannot hold the rest for 15 minutes. Its budget is the
  record's window; `--wait` bounds only the foreground poll (RU §4.4).
- **A failed peer never stalls the rest.** `failed`, `unreachable`, `deferred` and `refused`
  are recorded and the loop continues. `failed` backs off 5, 10, 20, 40 min… capped at 2 h
  (a consecutive-failure count in the record): the retry-storm guard (RU §10). There is no
  give-up counter; the window is the only bound (RU §6.5).
- **A dropped link mid-apply is not a failure.** The member's last act of a successful pass is
  rolling its own relay (§1), so the `done` reply may never be sent. After a drop following
  `applying`, the origin reconnects (bounded, 120 s) and re-asks: `already_on_target`, or a
  handshake `peer_build` equal to the target (`readiness.compare_builds`), records `done`. (Q4.)
- **Never forced.** There is no force parameter on `net_update`, in the record, or in any flag
  this note adds. `--force` stays the operator's hand-run escalation on `lop-fleet-update`; a
  forced arm would be a new, named, human-only decision recorded in RU (RU §4.4).

*What "idle" means.* On the member, the registry is the only authority (RU §4.3): **no live
session record with `busy` set**, evaluated fresh at the ask and again under the update lock
just before the install. That is the fleet tool's own drain predicate, so the rolling drain
and the operator's measured local discipline cannot disagree. After the pointer flip and the
relay roll, each runtime is asked to retire (`refresh_if_idle`) and answers by its own
`may_refresh()`: background jobs, subagents and parked gates come back `kept: <reason>`, and
those runtimes keep their intact tree and retire at their next idle instant or on the
hard-stale bound. The drain gates the install; the runtime's own predicate gates each
retirement; neither is overridden. Re-engaging displaced sessions is RU §4.2 step 5, as the
operator decided. (Q1, Q2.)

*Resumable, by interruption.* The record is the only state. Origin relay restart: the thread
resumes every `active` record at its first non-terminal member; a `done` member is never
re-applied (the op is idempotent, RU §3). Origin off for the whole window: the record lapses
with a sentence, and `lop network update` rolls what remains. Member relay dies mid-install: a
pre-flip failure removes its own tree, the stale lock is superseded, the next ask re-runs. A
newer update on the origin: the new record supersedes. `--resume [<id>]` clears the
`next_ask_at` floors and wakes the thread; it does not drive.

---

## 5. The switch

`network.rollout_on_update`: **`auto` (default — ON) or `off`.** `auto` opens a record when
the primary's update moves to a newer version (§3). `off` opens none and writes
`update_rollout_suppressed` (§6). The manual verbs (`lop network update <peer>`, `--all`; RU
§7.4) work under either — they are an operator command, authority `operator_command`.

- *Per run:* `lop update --no-roll`, beside `--no-services` (`cli.py:1771`), threaded as
  `roll=not no_roll` the way `services` reaches `update_command` (`:14570` → `update.py:7042`).
  The `lop-update` script and `/update` carry no flag and follow the key.
- *Stop a running roll:* the thread re-reads the key before every ask, so `off` halts a
  policy-driven roll at the next member boundary with no other command; `lop network update
  --cancel [<id>]` marks a record `abandoned`. An apply already in flight finishes — an
  install cannot be recalled — and is recorded.
- *Per member:* the member's own revoke of `update` (RU §2) is the member-side off-switch.
- *Registry:* a `Setting` in the `network` section of `SETTINGS` (beside `network.audit.*`,
  `settings_io.py:4076`), a module-level default constant next to its reader, and a row in
  `_consumer_defaults()`, enforced by `test_every_default_matches_its_consumer` (AGENTS.md,
  "Adding a configuration key"). (Q5.)
- *`ask` is deferred.* RU §3 names a third value, `ask` ("one confirmation"). v1 does not
  accept it: the instruction is `auto` with an escape, and the catch-up path runs in a relay
  with no surface to ask on. The registry's ENUM refuses it until a decision adds it.

---

## 6. Audit

Both ends, append-only JSONL, the existing writer and rotation; **one row per semantic
change, never per poll** (`audit.py:1-9`). New kinds go into `EVENT_KINDS`, `DETAIL_KEYS` and,
where marked, `DURABLE_EVENTS`, with the drift test updated. RU §6.4's names are kept except as
item (e) above says.

| Kind | End | Written when | Durable | Outcome · cause | New `detail` keys |
|---|---|---|---|---|---|
| `update_rollout_started` | origin | the record opens | **yes** | ok | `rollout authority policy via target from members` |
| `update_rollout_suppressed` | origin | an update moved but `off` or `--no-roll` stopped it, and ≥1 active member exists | no | ok · `policy` | `via policy flag` |
| `update_member_state` | origin | a member's state changes | no | per state · `busy`, `capability_denied`, `peer_unreachable`, `timeout`, `internal` | `rollout state code reason method version` |
| `update_rollout_done` | origin | no member non-terminal, or window lapsed, or `--cancel` | no | ok or partial | `rollout counts` |
| `update_requested` | member | the authorizer admitted a `net_update` | **yes** | ok | `rollout target capability` |
| `update_refused` | member | executor refusal by rule (ahead, not published, editable, in progress) | **yes** | refused · `policy` | `rollout code target` |
| `update_started`, `update_completed`, `update_failed` | member | the install begins, ends | no | ok or failed · `internal` | `rollout target method from version` |

**Naming the authority.** `update_rollout_started` carries `detail.authority`, a closed token:
`standing_instruction` (the primary decided under the operator's standing instruction, policy
`auto`) or `operator_command` (a human ran `lop network update` at this device). `actor` is
the primary (`self`), `network_id` the network; the record's `authority` object holds the same
fact, written in the same step. Every later row of that rollout repeats `rollout`, so one
`lop network log` filter reconstructs who decided, under what authority, and what each peer
did. The member's rows name the *member's* basis independently — the verified requester and
`capability: update` — never the origin's claim, so a lying peer cannot write its own authority
into a member's log. A member that has not granted `update` writes nothing new: the existing
durable `authorisation_refused` already records `op: net_update`, `capability: update`.

One new `CAUSES` entry, `busy` (a deferral); the rest are reused. The installer's failure tail
stays in the record's receipt, not in `detail` (value cap 200 chars, `audit.py:108`), and no
key collides with `FORBIDDEN_DETAIL_KEYS` (`:482`). Receipts in the record coalesce identical
consecutive answers into one entry with a count and a last-seen time, capped at 20 per member.
Durability follows the module's own rule (`audit.py:516-519`, "Refusals qualify") and the
family's precedent — `pairing_refused`, `handshake_refused` and `authorisation_refused` are
all durable (`:519-558`); `update_rollout_suppressed` stays batched because it records a
decision *not* to act, remade at the next update.

---

## 7. Visibility and mixed versions

RU §7 stands. The trigger adds three rules:

1. The rollout clause on a `peers` row comes from the record alone and follows
   `build_suffix`'s discipline (`readiness.py:1109`): the suffix reports the build comparison
   and the clause reports the rollout — never merged, and the clause is omitted whole (`""`)
   whenever the suffix is `""` (unknown or absent build), so there the old line survives byte
   for byte (`compare_builds`, `:1079-1107`; `_peer_line`, `network/cli.py:6536`).
2. A skip is never silent: `skipped (no_grant, predates_rolling_updates, unsupported_kind)`
   shows on the peers row and in the update summary with its remedy, because the first rollout
   after this ships will skip most members.
3. The readiness build row stays ADMISSION-class (`readiness.py:883-887`): a `behind` peer is
   still not admitted to offloads while its rollout is pending. The rollout clears the row; it
   relaxes nothing.

Mixed versions follow RU §5: `net_update` is never sent to a member lacking the negotiated
`mesh-update-v1`, and an absent answer is never a pass — a silent member is `unreachable` or
`deferred`, never `done`.

---

## 8. Not in scope for v1

- The `ask` policy value and any per-update confirmation surface.
- Forced updates, downgrades, source-ref or dev-tree channels, Windows members (RU §8.4).
- Health-gated promotion or canary halting: verification is `build == target`, so a member
  that installs and then misbehaves is not detected. The mitigations are serial order, the
  switch, and `--cancel`.
- Relay-observed triggers (a manual `uv tool install`, a Hub path): only the product's update
  front ends open a record.
- A `--no-roll` on `lop-update` or `/update`; a rollout key in the `update_report` machine
  line (additive later); per-member ordering or priority.
- Pool members; transitive or network-wide grants; coordination between two origins beyond
  monotonicity; anything that writes another device's grant (`types.py:526-543`).

---

## 9. Open questions (verify at implementation), each with my recommendation

1. **Which predicate publishes the registry `busy` bit?** The gate reads `SessionRecord.busy`.
   *Verify* it is published from `is_conversationally_active()` (`serving.py:1681`), not
   `is_busy()` (`:1627`). *Recommend* the registry bit for the gate (the measured discipline),
   with the runtime's own `may_refresh()` gating each retirement; if a member without the
   generation layout must gate on the inclusive predicate, publish it as a second registry
   field and leave `busy` alone (the 8-of-8 incident, `serving.py:1697`).
2. **A read-only idle probe?** `refresh_if_idle` asks *and acts*; no "would you refuse?" probe
   was found. *Recommend* none in v1; add one only if the drill shows the pass retiring
   runtimes the operator would have kept.
3. **A link-open nudge?** RU §3 wants a re-ask on "the next link-open from that member". No
   link-established seam turned up by name in `relay.py`. *Recommend* shipping the 300 s
   cadence alone — next contact within ≤ 5 min — and adding the nudge only if the seam exists
   and the drill measures the cadence as too slow.
4. **Reply before the self-roll, or re-probe after?** If the member's last act is rolling its
   own relay (§1), `done` may never be sent. *Recommend* both: the executor replies first and
   rolls services as a detached child, and the origin re-probes after a drop (§4). The second
   alone is sufficient for correctness; the first makes the receipt exact.
5. **The key's `Scope`.** The update process reads the key at each update, but the `network`
   section's declared `Scope` (`settings_io.py:126`, uniform within a section) is what
   `/settings` will label it. *Recommend* the `network` section if its scope is compatible,
   else a small section of its own, per AGENTS.md. Settled by reading `SECTIONS` (`:312`).
6. **A member shared across networks.** Which network's link carries the ask decides which
   row authorises it (`definitions.py:2315-2318`). *Recommend* the record stays per (origin,
   network), the dial names the network, and the second network's pass finds
   `already_on_target`; verify the dial seam can name a network.

---

## 10. Acceptance cells for the trigger half, and risks

Cells (real processes, isolated config roots, synthetic sessions; RU §8.3 is the drill):

1. **Hook.** PyPI move → record + `update_rollout_started`; snapshot with an unchanged
   version → nothing; nothing-to-install → nothing; `/update` → record, then relaunch; a hook
   failure leaves the exit code unchanged.
2. **Switch.** `off` and `--no-roll` → `update_rollout_suppressed` and no record; `off` flipped
   mid-roll → no ask after the in-flight one; `--cancel` → `abandoned`.
3. **Never forced.** A synthetic busy session on the member → `busy`, build stamp unchanged,
   transcript intact; the member rolls after the turn ends.
4. **Non-stall.** Busy member A and idle member B → B `done` while A is `deferred`; a failed
   member leaves the rest rolling; an ungranted member writes one refusal row per rollout
   across ≥ 3 ticks.
5. **Resume.** Kill the origin relay between members → continues at the first non-terminal
   member; a `done` member is not re-applied. **Drop-after-apply:** cut the link after
   `applying` → `done` by re-probe.
6. **Skew and cascade.** A member without `mesh-update-v1` gets no op; a member ahead answers
   `ahead_of_target`; a propagated update opens no record on the member.
7. **Audit.** Both ends' rows correlate on `rollout`; no forbidden key; drift test updated.

Risks to watch:

- **A surprise auto-roll** on the first release after this ships. The summary line, the
  switch, and `--cancel` are the mitigations; watch the operator's reaction, not the tests.
- **A refusal or failure storm.** Parking and backoff should hold `update_member_state` to one
  row per change; watch the row rate on a mesh with an ungranted member over 24 h.
- **The driver blocking sync.** While a member applies, definitions sync must keep its
  cadence; watch it in the drill.
- **An agent-run `lop update` now rolls peers.** Same bound, audited; `actor_kind` is a hint,
  not proof of who typed it.

---

## Relationship to the sibling designs

This note adds no R-numbered requirement. It implements RU §8.2's S3 (rolling orchestration,
the policy key, the `lop update` hook) and touches S4's row additions only by rule (§7); S1,
S2 and S5 are unchanged. `mesh-consent-provisioning.md` §7 evaluated RU's transport for
credentials and declined it; nothing here changes that — the trigger carries no payload. The
spine's detail-design table row remains the follow-up RU already names.
