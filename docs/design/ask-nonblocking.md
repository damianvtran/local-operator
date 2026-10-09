# Design: non-blocking, queued, timeout-bounded `ask`

Status: PROPOSED — gate for the implementation PRs. Base: `origin/main` @ `302a061e5`
(v0.64.10). Author: architect, 2026-09-30. **AMENDED 2026-10-03 — see §10**
(in-flight answer revision, #1936): §10 is binding on §2.2's event table, §2.4, the §4
ops line and §5's copy contract; read it beside those sections.
**§5.0 and R7 (§7) AMENDED 2026-10-08, in place — see the "Superseded text" note in
§5.0** (the composer-routing invariant is reversed: on the queued-ask path the composer
never answers an ask; the answer is given in the ask surface, whose explicit free-text
door is its trailing `Other` row with its own input — the flag-off blocking gate is out of
scope, see the rule). Unlike §10 this amendment has no section of its own: the rule, the
placeholder copy and R7's asserts were rewritten where they stand.

**Provenance.** Every `file:line` is against `origin/main` @ `302a061e5` (verified with
`git show`/`git grep`) unless tagged **(scout)** = taken from a scout report on the stale
0.61.9 local checkout and NOT re-verified on main (re-check before editing), or **(UI)** /
**(app)** = the other repos' scouts. Paths are `local_operator/…` unless noted. Two scout
claims were wrong and are corrected here: a `monitor` tool **does** exist
(`tools/registry.py:93`), and `is_busy` is `serving.py:1459`, not 1362.

---

## 0. Problem

`ask` parks the whole agent turn on a human. Five layers each assume exactly one live ask:

| # | Assumption | Load-bearing site |
|---|---|---|
| 1 | The tool awaits the human | `tools/builtin.py:24408` `answers = await ask_user(params.questions)`; tool is `exclusive` |
| 2 | One card in the TUI | `tui/app.py:24949` single-slot guard, `:24989-90` `_ask_screen`/`_ask_pending` (one future) |
| 3 | One gate on the frontend-state wire | `session/frontend_state.py:2909` `pending_gate: PendingGateState \| None`; `serving.py:5359` `_publish_pending_gate` mirrors only the front card |
| 4 | One viewer task | `session/attached.py:6350/6362` `_apply_pending_gate`/`_maybe_start_gate` keep one `_gate_task`; `:3157` `answer_gate` matches the single gate; desktop route `server/routes/desktop_sessions.py:3380-3415` 409s on epoch mismatch |
| 5 | One projected card on the phone | `mobile/projection.py:2854-2880` `push_pending` FIFO but `_sync_pending` projects only `[0]` + `pending_count` (`mobile/types.py:875`) |

Consequences that motivate the change: nothing survives a runtime death (futures live in
`serving.py:798`, the fold and `_parked_announcement`); a parked gate pins the runtime
(`is_busy`, `serving.py:1459`); the per-question timeout multiplies (N questions × timeout,
`serving.py:989-1077`, `_gate_timeout_s` `:4610`); a timeout is indistinguishable from Esc
in the live turn (`ASK_UNANSWERED_TEXT`, `builtin.py:24152`) and the only record is a
transcript-only row (`_record_gate_timeout`, `serving.py:4818`); the desktop composer is
turned into the answer box by the blocking gate (`chat-page.tsx:1624-1698` **(UI)**; that
swallow is the flag-off path and is out of scope of the §5.0 amendment); and a resumed session can
hold a dangling `tool_use` for an ask.

## 1. What ships (one paragraph)

`ask` returns an immediate **receipt**. The ask is appended to a **durable per-session event
log** (`asks.jsonl`), owned by the session, not by any front end. (Spec's "response tool
trace/card" is read here as **one transcript receipt block per surface's own idiom**: a
transcript receipt block in the TUI, a receipt kind in the desktop UI, a card in the mobile
app/web — the same `ask_response`/`ask_timeout` data, not a new tool call; PROPOSAL for Aida
to confirm.) Any surface lists it,
answers it (whole ask, atomic), or declines it. On **answer** the runtime injects one
`ask_response` turn (questions + answers, one source of text for model and card) and every
surface paints the same response as a card. On **deadline** the runtime injects one
`ask_timeout` notice; the ask stays visible as *timed out* and a **late answer** is still
accepted and attributed (`status: late`, same `ask_id`). A cold session is engaged at the
deadline through the existing wake engine. Approvals are untouched and keep blocking.

---

## 2. Proposed semantics

### 2.1 Tool contract (`builtin.py:24138` `AskParams`, `:24276` `build_ask_tool`)

- **Params:** `questions` (unchanged shape) + `timeout`. Units: **integer SECONDS**,
  bounds **120–86400 s** (floor 2 min, cap 24 h), default `3600`. A duration string
  (`"30m"`/`"2h"`) is accepted via `harness/wake.py:199 parse_wake_duration` — **which
  returns MILLISECONDS, so divide by 1000 before comparing; a 1000× bug otherwise ships**
  (§7 asserts `"2h"` == 7200 s). Out-of-range is a *validation error that names the bounds*
  (`_validation_error`), never a silent clamp — the model learns the calibration from the
  error. No `urgent` param (PROPOSAL D1): **urgent is derived, `timeout <= 900` (15 min)**,
  so the schema that rides every request grows by one field, not two.
- **Flags:** `concurrency` `"exclusive"` → `"shared"` (it no longer holds a human);
  `interruptible` → `False` (a steer has nothing to cancel). `approval_tier="read"` unchanged.
- **Availability is unchanged:** `build_ask_tool` still returns `None` without the host hook
  (`builtin.py:24276+`), which preserves "subagents/headless hosts have no `ask`" and the
  `<interactivity>` channel logic (`prompts_api.py:588-680`). `Session.set_ask_handler`
  (`session.py:8245`) stays the capability switch; its argument changes from "await answers"
  to an optional **presenter** (`on_ask_changed(record)`), and the queue itself lives on
  `Session` so the in-process TUI host (`tui/app.py:11516,12077`) and the owning runtime
  (`serving.py:1086`) share one implementation.
- **Queue caps (spam guard — non-blocking makes asking free):** three caps, each a
  validation error whose text says "wait for responses; do not re-ask" — (1) ≤ **8 open
  asks per session**; (2) a second open secret question for a credential key already open in
  this session is refused (`AskQuestion` id *is* the key, `harness/types.py:1031`); (3) an
  ask whose question text is **byte-identical** to an already-open ask's is refused (the
  dare-I-ask-again loop), and the same text is allowed again once the first is terminal.
- **Receipt (tool result, no question repetition).** Reach is measured, not asserted — reuse
  `serving._attached_surfaces()`/`_desktop_notification_available()` (`serving.py:4639`),
  which prove "a client is connected / an OS banner is possible", **NOT** "a human was
  told". So the receipt claims *presentation*, never *notice delivered*:
  - reachable: *"Ask a-7f3 queued (2 questions); showing on <terminal|phone|desktop>.
    Continue with other work — their answer arrives as an **ask response** turn. A receipt is
    NOT consent: do not run anything the answer was meant to authorise. If nothing arrives by
    <expires_at> you will get a timeout notice."*
  - unreachable: *"…queued; nobody is attached right now, so it will be shown when the user
    next opens this session…"*.
  - `details = {ask_id, status:"queued", question_ids, timeout_s, expires_at, urgent, secret, reach}`.
- **Legacy tail-of-turn safety:** the ask no longer leaves a dangling `tool_use` on a restart
  (the tool returns at once), which retires the resume-repair question in the scout.

### 2.2 Queue model

**Truth: `<config_dir>/sessions/<sid>/asks.jsonl`** — append-only events, `O_APPEND` + bounded
`LOCK_NB` flock, exactly the `session/runtime/inbox.py` discipline (`:18-26`, `append_inbox`
`:303`), because the writers are several processes and not all hold the transcript lease.

| event | fields (all carry `v:1`, `ask_id`, `at`) |
|---|---|
| `queued` | `expires_at, timeout_s, urgent, tool_call_id, questions[{id,question,options,multi,secret,persist,recommended}]` — written only by the owning runtime |
| `answered` | `by:{surface}`, `answers{qid:[str]}` — secret answers hold **`[<key>]` only, never a value** |
| `declined` | `by` — explicit "no answer / decide yourself" (today's Esc) |
| `dismissed` | `by` — view-only removal of a timed-out ask; injects nothing |

**Status is a pure fold of (events, now)** — no `timed_out` event, which removes the
answer-vs-deadline race. The precedence is **total** (first match wins, so every
`(events, now)` has exactly one answer); `state = fold(events, now)`:

| # | condition | status |
|---|---|---|
| 1 | an `answered` event exists with `at <= expires_at` | `answered` — a real in-window answer outranks every view action |
| 2 | a `declined` event exists | `declined` — terminal-on-write |
| 3 | a `dismissed` event exists | `dismissed` — terminal-on-write; the UI only offers dismiss on a `timed_out` ask, so it can never shadow an in-window `answered` |
| 4 | an `answered` event exists, `at > expires_at`, and `now <= expires_at + 7 d` | `late` |
| 5 | an `answered` event exists, `at > expires_at`, and `now > expires_at + 7 d` | `expired` |
| 6 | no terminal event, `expires_at <= now <= expires_at + 7 d` | `timed_out` |
| 7 | no terminal event, `now > expires_at + 7 d` | `expired` |
| 8 | otherwise | `open` |

- **`timed_out` vs `expired`.** `timed_out` = unanswered, past deadline, **notice still
  owed** (delivered once). `expired` = the same ask past `expires_at + 7 d`, **or** an answer
  that arrived past the window: the notice / response is **not** injected — it is stale by the
  same 7-day bound the wake supervisor already uses
  (`wakes/supervisor.py:136 STALE_AFTER_S`), so §7 asserts an expired ask produces no row.
- **Terminal-on-write** (`dismissed`, `declined`) means a later `answered` does **not**
  reopen the ask; it is refused with the copy below. Ties are decided by first writer in
  flock order, so two surfaces racing produce one winner. **Rule 1 is deliberately above
  rule 3**: a dismissal is a view action on an ask that already timed out, and it must never
  swallow an answer that arrived inside the window. Since the UI only offers dismiss on a
  `timed_out` ask the two cannot normally co-occur; rule 1 makes the fold total if a
  hand-written log ever does.
- **Answer refusal copy** (one sentence per state, from `asks/render.py`, so every surface
  shows the same words): `open` → accepted; `timed_out` → accepted, folded to `late`;
  `late` → accepted again, same `ask_id`, still one response row; `expired` → "this ask
  expired 7 days ago — ask again if it is still needed"; `declined`/`dismissed` → "you
  already declined this"; `answered` → "already answered by <surface>" (the
  `_resolve_pending` single-winner semantics, `serving.py:2519`, moved onto the log).

**Delivery marker = the transcript row, PER (ask_id, kind) — not one boolean.** Each
terminal state has its own expected row, and `reconcile` writes exactly the rows that are
missing; `transcript.has_entry(<id>)` (`session/transcript.py:1953`) is the guard, so
idempotence is structural and a `late` ask owes **one** row — the response, which supersedes its deadline row for good (§2.3). The marker is DURABLE: the transcript row, not the
handoff; `reconcile`'s `_handed` set is scheduling only (stops a re-entrant reconcile
double-handing the handoff→append gap) — never a window term, never the wire's `delivered`.

| status | expected transcript row(s) | injects a turn? |
|---|---|---|
| `answered` | `ask-response-<ask_id>` | yes (one) |
| `late` | `ask-response-<ask_id>` — the response supersedes the deadline row for good (§2.3); both rows exist only when the deadline fired first | yes (one; two across reconciles when the deadline fired first) |
| `timed_out` | `ask-timeout-<ask_id>` | yes (one) |
| `declined` | `ask-response-<ask_id>` with `status:"declined"` — **the same type and id as an answer**, so the model learns the user declined rather than re-asking | yes (one) |
| `dismissed` | **NONE** — terminal-and-satisfied; `reconcile` must never attempt a delivery (else it loops forever) | no |
| `expired` | **NONE** — same rule | no |
| `open` | none yet | no |

**Level-triggered reconcile**, not an event handler: `AskQueue.reconcile(now)` scans the log,
computes the fold, and delivers every missing row above. No two-phase "mark delivered" write
to lose in a crash. It runs at runtime boot (beside `process._drain_inbox_into`,
`process.py:3612`), at turn start (beside `_drain_spooled_peer_inbox`, called
`session.py:11748`), on every answer/decline/dismiss op, and from the timer below.
Idempotent by construction. **Wire `delivered`** (contract §4; amended 2026-10-04) means
"the row(s) THIS status requires are DURABLE in the transcript":

| status | `delivered: true` iff |
|---|---|
| `open` | never — nothing owed |
| `answered` | response row durable |
| `late` | response row durable — **not** the deadline row |
| `timed_out` | deadline row durable |
| `declined` | response row durable |
| `dismissed` / `expired` | any response/timeout row durable |

The flag is the CONSUMPTION flag: a `late` answer whose deadline notice went out reads
`false` until its response row lands — the revision window staying open. It is **sticky** —
once the required row is durable it stays `true` for the life of the record, so an answered
ask cannot flip back to undelivered when it folds to `expired` seven days later; the one
sanctioned flip is `timed_out`→`late` (the ANSWER is what is undelivered there), and
`dismissed`/`expired` are `false` only when no delivered row exists at all. Handoff alone
never reads true, for any reader — in-process, another process, or the index.

**Derived index (non-authoritative, stdlib-only, self-healing):**
`<config_dir>/asks/<sid>.json` — open + last-7d terminal asks with deadlines and a
`delivered` hint. Same contract as `wakes/store.py` (module docstring `:1-41`; `write_entry`
`:223` staged write, corrupt/unknown-schema = absent). Lives outside the session dir so the
cross-session "all my open asks" view is O(sessions-with-asks) and works with no runtime.
Any writer recomputes it from the log, so last-write-wins is safe. **Deletion path (the
session dir's guards do not cover this file, which lives outside it):** the entry is removed
by the same cleanup that deletes a session dir (`session/cleanup.py`, beside the existing
junk-reap — the `wakes/` store is the precedent, its entry has no separate sweeper), **and**
a TTL sweep in the index reader drops an entry whose every ask is terminal and older than
7 d, so an orphaned file (a session deleted while no cleanup ran) cannot accumulate. Both are
in A1 with a unit test (delete the session dir by hand, reopen the index → entry gone). New
package
`local_operator/asks/` (`store.py` stdlib-only, pinned by `tests/unit/test_import_graph.py`
like `wakes/`; `policy.py` constants; `render.py` the one text function).

**Ownership.** Only the runtime holding the session appends `queued`, writes the transcript
rows and publishes state. Any process may append `answered/declined/dismissed` (cold answer
path, §2.4) — under flock, non-secret only. The wire (`asks` on frontend state, §4) is
published by the owner from the fold on every change.

**Timeout scheduling.** Two layers, deliberately different:
1. *In-runtime*: one earliest-deadline timer per session (WakeScheduler `_arm` style, one
   task, ≤60 s tick — `harness/wake.py:71 MAX_ARM_MS`), not a task per ask. It calls `reconcile`.
2. *Cold runtime*: a hidden wake row `kind="ask_timeout"`, `next_due_at=expires_at`, riding
   the wake engine exactly as patience rows do (`wakes/patience.py:1-53`,
   `wakes/store.py:161 is_patience_row`, `session.py:18957 _deliver_patience_wake`). It buys:
   supervisor engagement of a cold session (`wakes/supervisor.py` `WakeErrand` delivers
   nothing — `launch.py:181`), the index, dormancy under `stopped_at`
   (`store.py` `is_held`, `supervisor.py:136` staleness) and catch-up on load. The fire's
   handler is just `reconcile` (the fire is stale if the ask is already terminal — the
   patience "watermark" rule, no cross-process row deletion needed). Human wake surfaces
   must subtract the new kind: introduce one `is_internal_wake_row` predicate =
   patience ∪ ask_timeout and swap it into the *human-surface* callers only. **Measured on
   `origin/main@302a061e5`: **19 occurrences of `is_patience_row`, of which 14 are call
   sites** — `aida/proactive.py:1457`, `session.py:17331,17756,18764,19218`,
   `tools/builtin.py:12384`, `wakes/patience.py:168,179,183,706,726,727`, `wakes/store.py:161,178`.
   The other 5 are module imports, **not sweep targets**: `aida/proactive.py:76`,
   `session.py:17329/17754/18762`, `wakes/patience.py:69`. This is F's sweep list; the
   patience *cap* logic keeps the narrow predicate, the human surfaces (picker, CLI listing,
   desktop listing/feed, dormant-count receipt) take the union.
   *Degrade if a host has no wake scheduler:* layer 1 still works; a cold session then notices
   the deadline at next boot (documented; not a correctness bug).
3. **Stop:** asks survive stop — `_deny_pending_gates` (`serving.py:2492`) becomes
   approvals-only. A user-stopped session is dormant to the supervisor (no timeout firing);
   on reopen the load-time reconcile delivers overdue notices annotated "lapsed while the
   session was stopped".

**Restart re-surface.** The log survives; boot `reconcile` republishes `asks` on the wire
and re-arms the timer; open asks re-appear on every surface and in the aggregate index with
no per-front-end work. `retention._SIDECAR_NAMES` (`session/retention.py:85`) gains
`asks.jsonl`; `cleanup._has_spooled_mail`/its guard (`session/cleanup.py:1048`) gains a
"has open asks" guard so a junk-reap or cleanup never deletes a session with a live ask.

**Residency.** Open asks no longer make `is_busy()` true (`serving.py:1459`), so an idle
runtime can be reaped/retired — durability + the wake row are what make that safe.

### 2.3 Injection model

Reuse the wake/peer rail; add no new transport for the live case. New
`Session.deliver_ask_messages(msgs)` modelled on `_deliver_wake` (`session.py:18748`):

| target | path |
|---|---|
| live, mid-turn (`_is_streaming`) | `_steering_queue.put_nowait`, `_courtesy_wake_count += 1` (does **not** cancel a running tool), then `_peer_arrival.mark(type)` to wake a parked `wait`; persisted at the next boundary by `_drain_steering` (`:14031`). Never touch `_context.messages` mid-batch. |
| live, idle | `_spawn_background(self._prompt_messages(msgs))` (`:11212`) — takes `_turn_lock`, clears a sticky abort (only for non-residue types), flushes journals |
| cold / stopped | **No new errand payload.** `AskErrand` (new, beside `launch.py:181 WakeErrand`, added to the union `:231`, `_deliver` `:797`, `_fields` `:1464`) *delivers nothing*: it only makes a runtime exist; boot `reconcile` does the delivery. This is why the design needs no `CustomMessage`-carrying errand, and why it is session-agnostic (wake-, monitor-, supervisor- or user-started sessions all boot the same reconcile). |

- **Messages — TWO custom types, not three.** `CustomMessage(custom_type="ask_response",
  attribution="user", id=f"ask-response-{ask_id}", details={ask_id, status, questions, answers,
  at, text})` with **`status ∈ {answered, late, declined}`** — a decline is a response-shaped
  fact (the ask settled, a human decided) and reusing the type keeps one registration, one
  render branch and one `EntryKind`, exactly as the "any miss silently drops the row" rule
  wants; only its `text` and `status` differ. Plus `custom_type="ask_timeout"`,
  `id=f"ask-timeout-{ask_id}"`. `text` comes from **one** function (`asks/render.py`, reusing
  `_ask_report`/`_report_secret_answers`, `builtin.py:24261/24176`); the model sees it, the
  card shows it, no surface re-derives Q&A. Late answer: the same `ask-response-<id>` (so it
  can never double-deliver) with `status:"late"` and text "you already proceeded when this
  timed out at <t>; reconsider only if the answer changes your work".
- **Registration is mandatory — any miss silently drops the row** (scout, verify on main):
  `harness/message_types.py` (import-free constants), `harness/render.py` branch (beside
  the `GATE_TIMEOUT_CUSTOM_TYPE` branch, `:204`), `session._PERSISTABLE_CUSTOM_TYPES`
  (`session.py:1087`, allow-list), NOT `BOOKKEEPING_CUSTOM_TYPES` (a human answer is real
  activity), projection `mobile/projection.py` (GATE_TIMEOUT handled `:1293`), TUI
  `tui/session_presentation.py` (`:989-1011`, `:1239`), `harness/rows.py` copy (`:660`).
  Keep the legacy `GATE_TIMEOUT_CUSTOM_TYPE` ask arm rendering for old transcripts.
- **Stopped-session rule.** `ask_response` in **all three** statuses is fresh human intent
  (an answer, a late answer, or an explicit *decline* is a person deciding) → clears a sticky
  abort like a typed prompt. `ask_timeout` is **not** intent → add to
  `_STOPPED_WORK_RESIDUE_TYPES` (`session.py:556`) so a timeout never buys a paid turn on a
  session the user just stopped.
- **Batching/ordering.** One `reconcile` collects everything due, ordered by log seq
  (answers) / `expires_at` (timeouts), and delivers as ONE `_prompt_messages(msgs)` when idle
  (one paid turn, not N). `MAX_QUEUED_PROMPTS=32` (`serving.py:223`) is not involved.
  **A late answer's deadline row is suppressed FOR GOOD, not postponed** (review round 1,
  MAJOR 2 — the one-batch form only deferred the contradiction): when a late answer's
  rows would land together (a cold boot after the answer arrived past the deadline), only
  the `ask-response-` row is emitted — replaying "[Ask timed out] … you will be told"
  immediately before the answer it announces reads to the model as a contradiction — and
  the `ask-timeout-` row is never written for that ask. A deadline that genuinely fired
  first, in its own reconcile, is unaffected: that is the `timed_out` status, where no
  response exists yet (§7 asserts both orders).
- **Dedupe.** deterministic ids + `has_entry` in `_drain_steering` (`:14031` already skips a
  durable id). Two runtimes cannot both deliver: only the lease holder writes the transcript.
- **Content is re-resolved at the append.** The handed message is built once for events and
  scheduling; the row that lands is rebuilt from the fold at the transcript append (all four
  writer paths), so a revision accepted between reconcile and append is what the model reads.
  `AskResponseDeliveredEvent` stays at handoff — a paint-ahead preview, not the content of
  record.
- **Event for live paint:** `AskResponseDeliveredEvent`/`AskTimeoutDeliveredEvent` beside
  `WakeDeliveredEvent`, emitted **before** the turn spawn so surfaces paint the card ahead of
  the work it triggers.

### 2.4 Answer paths

`ask_respond {ask_id, answers{qid:[str]}}` (whole ask, atomic — never a per-question wire
race; partial drafts are client-local and secret drafts are never persisted),
`ask_decline {ask_id}`, `ask_dismiss {ask_id}`.

- **Live runtime:** op → handle → single-winner append → `reconcile` → inject.
- **No runtime, non-secret:** the surface appends `answered` under flock and fires
  `engage_runtime(AskErrand)`; boot reconcile injects. (Desktop route: the current
  `answer_gate` requires a connected client and refuses a cold daemon — the route grows a cold
  arm for `ask_id` bodies.)
  > **As shipped (2026-10-03, the cold-answer fix).** The relay's cold arm does it the other
  > way round, and one way for both secret and non-secret: it ENGAGES FIRST
  > (`engage_session_client` → `engage_runtime(AskErrand)`, the same seam the prompt path
  > uses) and then sends the op over the dial. The runtime's own `respond`/`revise`/
  > `decline`/`dismiss` then do the validating, the single-winner append, the secret hop and
  > the `reconcile` — so no surface carries a second copy of those rules, and the refusals
  > (`expired`, `already answered by <surface>`) are the queue's own sentences rather than a
  > route-local guess. The desktop route already worked this way through its viewer facade's
  > bind (`AttachedSession._ensure_bound` → `engage_runtime(WarmErrand)`); this brings the
  > relay to the same shape. A conversation with no durable transcript is refused in words
  > (`ask_session_gone`) before any engage is attempted: it can never be read, so reporting
  > success for it would be worse than a refusal.
  >
  > **The clock on the surface is the ENGAGE's, not a warm prompt's.** One cold answer is
  > engage + dial + ack in a single call, so its worst case composes the engage budget
  > (`launch.DEFAULT_DEADLINE_S`, 30 s) with the op's ack (`attach_client.ACK_TIMEOUT_S`,
  > 15 s) — measured end to end at ~**30 s** on the fleet, against the 1–3 s a warm prompt
  > pays. Any in-flight affordance on a cold answer must be sized for that window (the
  > phone card's copy and spinner are the design round's, not this note's); nothing may
  > shorten it by skipping the engage, because a refusal that arrives without a runtime is
  > the bug the arm exists to fix.
- **Secret:** the value rides the op to a **live runtime only** (it is stored in session
  memory, `variables.py:389-410` **(scout)**, and must live in the process that runs `bash`).
  Cold → engage first, then send; if that fails the ask stays open with a clear error. The
  runtime appends `answered{[key]}` only after `store_credential` succeeded. **Nothing secret
  ever reaches `asks.jsonl`, the index, the transcript, an event, a notification or a card
  (card shows key / "provided").** Restart between answer and use: the key is gone
  (`session.py` journal_credential is live-context-only, scout `9267-9270`), so delivery
  verifies the key is still in the store; if not, the response says "credential was provided
  but the session restarted before it could be used — ask again" and the card shows
  *lost*. `persist=true` promotes at answer time, so it survives.
  Secret via a relayed (`locality=="remote"`) client: **preserve today's behaviour**;
  verify against `server.py` `credential` op refusal (scout `4954`) in PR A1 and state the
  result in its evidence — do not change policy in this feature.

### 2.5 Timeout-notice model

`ask_timeout`, attribution `system`, rendered to the model as an injected user-role message
with a bracketed header (the `wake_prompt` shape), never as "the user denied".

- **Normal:** *"[Ask timed out] No reply to ask a-7f3 arrived within 1h (asked 14:02).
  Q1 <≤200 chars>… Proceed without it: use your recommended option or best judgment and state
  the assumption in your report. The ask stays open for the user; if they answer later you
  will be told."*
- **Urgent (≤15 min):** adds *"This ask was urgent. Do not wait — resolve it now: delegate the
  question to a `task` subagent with the relevant expertise (e.g. `architect`, `reviewer`) and
  decide on its answer."*
- **Secret:** never quotes the prompt text; says only that the credential was not provided
  and which key.
- Delivered even when the agent is idle (that is the point: a session waiting on an ask must
  not stall forever), except on a stopped session (residue rule above).

---

## 3. Timeout policy

| class | default | floor | cap | why |
|---|---|---|---|---|
| normal | **1 h** | **2 min** | **24 h** | the spec's "less-sensitive" figure; because late answers stay attributable (§2.2), a short default costs a *timeout notice*, not a lost answer |
| urgent (derived, ≤15 min) | model sets 5–10 min | 2 min | — | matches the calibration; floor is ≥2× the 60 s wake tick (`MIN_WAKE_INTERVAL_MS`) so a deadline is never sub-tick |
| **secret** | **1 h** (PROPOSAL D3) | 2 min | 24 h | length is *not* a security dimension — the value is never at rest in the queue. The real hazard is restart between answer and use, handled by verify-on-delivery (§2.4), which a shorter window would not fix. Notification body stays terse (no prose). |

- Cap `86400` equals `DEFAULT_UNATTENDED_GATE_TIMEOUT_H = 24` (`serving.py:214`) by
  coincidence, not coupling. **`runtime.unattended_gate_timeout`
  (`settings_io.py:2360`) now governs approvals only**; asks ignore it and the 30 s
  no-registrant cap (`PENDING_REQUEST_TIMEOUT_S`, `serving.py:208`) — a queued ask does not
  depend on anyone being attached. **No new config key in v1** (constants in
  `asks/policy.py`; test seams inject a clock, not a setting). Timeout is **per ask**, not per
  question.
- Documented in the tool description and `guides/ask/GUIDE.md` with the calibration table:
  ~1 h routine · 5–10 min urgent (someone is expected to answer now) · up to 24 h genuinely
  non-urgent.

---

## 4. Wire and compatibility (additive only; no `PROTOCOL_VERSION` bump)

The bump note (`mobile/types.py:33`) is a *breaking* lever; `peer_message`'s comment
(`:389-391`) is the precedent: purely additive ops need no bump, an old registrant answers
"unknown op". Old clients ignore unknown fields.

**Frozen contract** (downstream PRs code against this text; any change = amend this note):

```
PendingAsk (frontend_state.asks[] and SessionProjection.asks[] and index entry)
  ask_id, session_id?, created_at, expires_at, timeout_s, urgent, status
  (open|answered|declined|timed_out|late|dismissed|expired), answered_at?,
  delivered: bool,   // CONSUMPTION: the row(s) THIS status requires are DURABLE in the
                     // transcript (§2.2); false while merely handed/queued. For `late` the
                     // response row — the deadline notice does not deliver the answer (a
                     // `timed_out` ask answered late flips to false until its row lands).
                     // `dismissed`/`expired`: any row, else false. Once the required row is
                     // durable it never flips back.
  questions[{id, question, options[{label,description?,recommended?}], multi, secret,
             persist}],
  answers?  {qid:[str]}        // secret: [<key>] only
  answered_by?  {surface}
SessionProjection.asks: PendingAsk[]  (cap 20 newest; open first)   asks_open: int
SessionListRow.asks_open: int   (pending_kind unchanged = approvals)
EntryKind "ask_response": details {ask_id, status: "answered"|"late"|"declined",
                                     questions, answers, at, text}
EntryKind "ask_timeout" : details {ask_id,status:"timed_out",waited_s,urgent,text}
ops:   ask_respond{ask_id,answers} · ask_decline{ask_id} · ask_dismiss{ask_id}
desktop: POST /v1/desktop/sessions/{id}/answers  body gains optional
         {ask_id, answers, decline} (epoch check SKIPPED when ask_id is set — asks outlive
         owner epochs); GET /v1/desktop/asks (aggregate, index-backed);
relay:   GET /api/asks (aggregate) ; command op ask_respond via existing /command
```

> **Amended 2026-10-03 (§10).** The ops line gains `ask_revise{ask_id, answers, by?}` and
> the desktop answers body gains `revise` (with `ask_id`+`answers`); §10 carries the
> binding text for the revision window, the `revised` event, the refusal copy and the
> kill-switch behaviour.
>
> **Amended 2026-10-04 (§10, consumption bound).** `delivered` re-pins from row-existence at
> handoff to CONSUMPTION. Additive semantics only, no `PROTOCOL_VERSION` bump: clients gating
> the change affordance on `delivered:false` now keep it open until the answer is committed
> to the conversation — exactly §10's window. An old core still answers `unknown op` for
> `ask_revise`; a new client against an old core reads the old flag.

- **Queue is separate from approvals.** `pending_gate`/`pending`/`pending_count` keep meaning
  *blocking* things; `pending_count` stays the approval queue length (test comment,
  `session-view.pending.test.tsx` **(scout)**).
- **Legacy mirror (PROPOSAL D4).** For exactly one release **all three** publishers of the
  old single slot also project the **head open ask's first unanswered question** as today's
  per-question card *when no approval is pending*, with
  `request_id = "<ask_id>.<qidx>"`: (1) the runtime's `_publish_pending_gate`
  (`serving.py:5359`), (2) the TUI host's `_publish_pending_gate` (`tui_handle.py`),
  (3) **the mobile fold's `_sync_pending` (`mobile/projection.py:2876-2880`)**, so a new core
  with the **old mobile app / web client** still shows and can answer queued asks
  (`projection.pending` + `pending_count` unchanged). The legacy `ask_answer` op
  (`mobile/types.py:331`) maps that synthetic id onto the queue. Why: the desktop app and the native app ship on their own schedules
  (skew of days is expected); without the mirror new-core + old-UI silently loses the ask
  feature until they update. Cost ~60 lines in one publisher; removed in cleanup PR F2. Known
  old-client wart (accepted, and unchanged by the §5.0 amendment — it describes an *old*
  client): an old desktop composer still routes typed text as the answer while the mirror
  is up. **Client rule (N3): once `asks` is present, a client IGNORES
  any `pending_gate` whose `kind == "ask"`** — otherwise a new client on a new core renders
  the same ask twice (the mirror card *and* the `asks[]` row). Approvals (`kind != "ask"`)
  are unaffected and keep the single slot. The residual race (a mirror frame in flight when
  the first `asks` frame lands) is contained by the fold's single-winner rule: both views
  address the same `ask_id`, and the loser gets "already answered by &lt;surface&gt;".
  Existing single-select truncation (`attached.py`
  `_run_ask` posts `values[0]`, scout `5707`) is fixed for new clients by the `answers` map.
- **Backend-owned session status** (UI must not infer): status code stays activity-derived
  (`working`/`idle`); open asks are a separate `asks_open`. `record.pending`
  (`session/runtime/types.py:963` **(scout)**) keeps the single string `"answer"` while
  ≥1 open ask so `lop sessions`/the TUI picker "Answer needed" ranking still work.
- **Skew matrix.** old core + new UI/app: no `asks` field → clients render nothing new
  (capability = field presence, the UI contract's own rule, `desktop-session-contract.ts:53`
  **(UI)**). new core + old UI/app — **both the desktop client and the mobile app/web**: the
  legacy mirror above covers all three publishers, so asks are neither lost nor unanswerable
  during the mirror window. old runtime + new op: "unknown op" → clients say "this session's
  runtime predates queued asks; update". new runtime + old app/web (mobile): the fold's
  `_sync_pending` mirror; §7 asserts the old-shape `pending` card still answers.
- **Push notifications:** none on the relay (app ADR 0002 §7/`docs/architecture.md:299`
  **(app)**). Out of scope; §8 D7.
- **Presence of `asks` IS the flag (the client-side proxy, made true).** A2 publishes
  `asks`/`asks_open` **only while `asks.policy.NONBLOCKING_ASK` is on** — with the flag off
  the fields are absent from the frontend state, the projection and the list rows, exactly as
  an old core omits them. Without this rule the proxy would be false for the whole A2→F
  window (the field would ship while the server default was still blocking) and a new client
  would take the flag-on path against a blocking backend.

**A2 addendum (wire presence and the mirror's two extra keys).** The paragraph above is the
rule; A2's implementation sharpens three things it left open, and every client codes against
THIS wording:

- **Presence ⇒ the flag is ON *and* this frame carries at least one ask row.** Absence ⇒
  either nothing to render in this frame (the flag is on, the session has no asks, or the
  frame's byte bound could not carry them) or the feature is off. "Supported but empty" is
  deliberately **not expressible**, and no client may depend on it: an empty array still pays
  for its keys on an attach frame with ~100 B of slack, so the empty case is published as
  absence. The risky direction is unchanged — a blocking backend never publishes the fields at
  all — and the old-client mirror covers the skew window either way. A2 also publishes
  **`asks_truncated`** (true only when the wire bound dropped rows, absent when the list is
  complete) so a client can never draw a prefix beside a full count and call it complete.
- **The mirrored card's `head` is the OLDEST open ask**, not the first row of the published
  list (which is open-first *newest*-first): a card that jumped to each new arrival would move
  under a user's finger mid-tap. The divergence is named here rather than discovered; a later
  PR may make the list lead with the oldest too.
- **The legacy path answers ONE QUESTION AT A TIME.** `ask_respond` stays atomic per ask and
  refuses a partial map; the mirrored card merges each tap into a per-ask DRAFT in the runtime
  (never in the log, which is still written once, atomically, on the last question) and the
  published row carries `draft_question_ids` so the card advances to the next unanswered
  question. A draft is not a durable answer: a runtime death returns the ask to open, and the
  index and aggregate routes never carry drafts.

---

## 5. Surfaces (all: queued + timed-out states honest; no surface may say "notified" it
cannot substantiate; agent-is-working ≠ "waiting for you")

Shared copy contract: **open** "Queued — the agent is continuing; expires in 42 m"; **answered**
"Answered — delivering" (`delivered:false`) → response card; **timed out** "Timed out — the
agent moved on; you can still answer"; **late** "Answered late — the agent was told";
**declined** "Declined — the agent was told"; **dismissed** "Dismissed — no reply was sent"
(no agent turn was bought); **expired** "Expired — this ask is too old to answer; ask the
agent again", with every answer control disabled and no error register (an expiry is not a
failure). Countdown is rendered from `expires_at` on the client clock.

**INVARIANT — flag-off behaviour was exactly today's, continuously through A1–D; that is now
HISTORY.** Until PR F flipped `asks.policy.NONBLOCKING_ASK`, every surface's default behaviour
was unchanged: the blocking card, the single `_ask_screen`/`_ask_pending` slot, the single
`pending_gate`, the view bridge's single `_gate_task`, the desktop composer swallow and typed
ordinals, and the fold's single `pending`. **Every new path introduced in B/C1/C2 exists only
with the flag on** (client-side that is the presence of the `asks` wire field — true because
A2 publishes it only when the flag is on, §4). No PR in B/C1/C2 may delete or repurpose an old
path: the old paths were to be removed **once, in F** — and F did **not** remove them. It kept
them byte-unchanged, because they are what the operator's escape hatch selects. The current
spec is the D6 STATUS note below: `NONBLOCKING_ASK` defaults **`True`** and
`LOP_ASK_NONBLOCKING=0` restores the blocking arm. This is what makes §8.9's "the pinned
tests stay green through A1–D" true, and a B/C1/C2 diff that removes an old path is a bug
against this note.

### 5.0 Shared ask-surface interaction model (R7)

Every surface presents the open ask set in one of **two states**. Both are **client-local
interaction state: no wire change, §4 is untouched by this section.**

- **EXPANDED** — the answer surface is active (TUI picker/list row, desktop `QuestionDock` or
  sheet, phone/app sheet). **Entered by the user** (the f4 toggle, click the minimized bar,
  Enter on a focused row, or the explicit `/asks` / header action), **or ONCE by the
  open-by-default policy** when a conversation is opened with asks already pending (the shared
  six-clause contract `tui/ask_open_policy.py` states — landed with the TUI/relay change
  (#2067, released in v0.68.9); the desktop drawer's open-by-default arrives with UI #897
  and the app sheet with its own PR). **Never automatic on ask arrival** — §5.1's
  no-arrival-mount, no-focus-steal rule
  is unchanged for an ask that lands while the conversation is already on screen. Left by Esc,
  the collapse control, or a click on the bar/chevron.
- **MINIMIZED** — a compact, persistent single-line affordance; the answer surface is *not*
  mounted. This is the state a new ask lands in when the user is already mid-answer or
  mid-draft, and the state the user returns to when they collapse.

**Composer rule (INVARIANT, every surface) — AMENDED 2026-10-08.** On the queued-ask
path the composer **never answers an ask**, in any state (expanded, minimized, no ask
surface mounted): what is typed in it is always an ordinary conversation message, and its
draft is never converted into an answer. (The one carve-out is the flag-off path — the
`LOP_ASK_NONBLOCKING=0` escape hatch, or a client that predates the queue — where the
desktop's blocking-gate composer swallow remains; that path is out of scope here and is
deleted with the blocking gate in PR F.) An answer is given **in the ask surface**, whose
explicit free-text door is the trailing **`Other`** row on every non-secret question, with
its own input. The wire half of image attachments on that input has landed
(`ask-attachments-v1`, #2058, released in v0.68.8); each surface's attachment UI follows in
its own repo, so the door is text-only until then. A
free-text-only question has no `Other` row to open: its input is the question's only
control and is shown open (on the TUI that is the picker's single free-text row, which is
itself the input). **Status 2026-10-08:** the TUI picker has its `Other` row today and the
desktop's landed with UI #892; the relay web and the app gain the explicit `Other` field
with their own `Other` PRs, and this section states the target those PRs build to. An ask
surface never
moves, sends or discards composer text: the composer's draft stays in its own buffer
through expand, collapse, settle and a conversation switch, and the ask surface's draft is
never sent into the composer's channel. (A *user* expand — f4, the bar, a list row — DOES take
the caret for the card, which is caret movement, not text movement; the TUI's automatic
open-by-default mount does NOT take it, so the composer keeps focus until the user asks for
the surface, and the caret hand-off on such an ask is Tab from the composer (a draft there
stays put) or a click. f4 itself is a pure TOGGLE and not an expand: on an auto-opened
surface it CLOSES that surface and records the dismissal, exactly as a click on the bar
does.)
Where a surface keeps an answer draft across collapse (desktop and relay web do; the native
app deliberately persists nothing — ADR 0005 §2 — and re-expands empty), toggling the surface
preserves it. When the ask **settles** (answered, timed out, declined, dismissed) its draft
is released with it (a surface may hold the dead key until it unmounts; nothing offers it
again).

**Secret-only asks.** The invariant is that a secret-only answerable ask never lets the
CHAT door carry the credential — a credential-safety rule, not routing. It holds by
different mechanics per surface: the **desktop** refuses input (typing in the composer is
disabled while such an ask is answerable); the **TUI** refuses the *submit* (the text stays
in the composer, nothing is sent or recorded, and a notice points at the card's hidden
field; slash, shell and aside entries still work); the **relay web and native app** have no
gap to close, because their sheets are modal (the composer is unreachable behind an inert
ancestor).

> *Superseded text.* The first revision of this rule routed the composer to the ask while the
> answer surface was EXPANDED (separate ask/chat buffers, a swapped placeholder, Enter sends
> the answer). It is removed because one input with two meanings made a chat message and an
> answer indistinguishable at the moment of Enter, and because nobody could tell that
> typing in the composer was how a free-text answer was given; the `Other` row gives free
> text a labelled home instead. §12.0's "answered in chat" path (the model records the attribution
> with `ask_withdraw(answered_in_chat)`) is now the only way a chat reply settles an ask, and
> §12.0 is stronger for it.

**Minimized affordance — the minimal design, per surface.** One line, directly above the
composer, built from each app's existing composer-chip vocabulary — it must read as a chip,
not as a modal, a toast or a banner:

- **TUI:** a single bar line above the composer, `? 1 question waiting — click to answer`
  (`? 3 questions waiting` when n>1), with an expand chevron at the right edge. Click or Enter
  expands; Esc or a click on the chevron collapses. Its accent is a **persistent** colour on
  the glyph, never animated (a pulse would be the focus-steal this design exists to avoid, in
  colour instead of keys).
- **Desktop UI:** the same bar under the chat, styled as the existing composer status chip
  (`composer-status-row.tsx`) so it sits in the wake/monitor chip row's language; click →
  `QuestionDock`. Same accent, no badge ticker (D8).
- **Phone web:** a chip/bar above the composer in the `pending-card.tsx` family; tap → asks
  sheet.
- **App:** the bar above the composer plus the list-row badge; tap → sheet.
- Every surface: **no focus steal**, no modal, nothing that displaces a draft, and the
  affordance disappears at zero asks.

**Sidebar / list status.** An **outstanding-asks** state on the session's row/icon, distinct
from the approval state and honest at 0 (absent, never a zero badge): the TUI sidebar and
`tui/session_catalog.py` ranking, the UI chat list, the mobile web list row, the app list.
The agent may be *working* while asks are outstanding — this is not a "waiting for you"
state (§5 header) and it must not read as one.

**Placeholder copy** (AMENDED 2026-10-08). The composer's placeholder is **never** swapped
for an ask: each app's existing placeholder is shown in every ask state. The ask surface's
own input carries its own copy, and the strings are **per surface**, not one string
everywhere: today the desktop's free-text-only input reads `Type your answer`, the TUI's
free-text row `Other (type your own)`, the relay web's `your answer` and the native app's
`Your answer`; the desktop's `Other` field copy has landed (UI #892, `Type your answer`);
the relay web's and the app's land with their `Other` PRs.

**Multiple asks.** The minimized bar always shows the **head** ask plus the count; expansion
opens the list/picker (TUI) or the sheet (others). Answering one advances to the **next
outstanding ask** when it was picked out of a list — the TUI returns to that list at the row
that takes the answered one's place (the next outstanding row, or the last one before it when
the answered ask was the last), and the sheet surfaces stay open and advance — and collapses
when there is nothing left to answer. (Audit of the merged TUI surface: the TUI
collapsed on every answer, so a queue of N cost N re-expands. A single ask's card still
collapses — there was no list to return to.)

### 5.1 TUI (PR B)
- **Entry/badge:** a count chip in the working line/status band (`◆ 2 asks`), sidebar/`/resume`
  catalog badge (`tui/session_catalog.py` `pending` ranking, scout `188/287`), OS notification via
  the existing `tui/notify.py` map (scout `177/237`). **No auto-mount on ask ARRIVAL and no focus
  steal** — the open-by-default mount (landed with #2067, released in v0.68.9;
  `tui/ask_open_policy.py`, §5.0 amended 2026-10-08) happens ONCE when a conversation is
  opened with asks already pending, is not an arrival mount, and takes no caret. A new ask
  must not displace a card the user is mid-answer on or their composer draft
  (`docs/design/composer-focus-default.md`; #1315 machinery `app.py:7715/7810`). The
  **§5.0 minimized bar is what names the first question** — it replaces the ad-hoc one-line
  notice, so there is one affordance, not two.
- **List:** `/asks` (+ keybinding) opens a list in `#prompt-host`; per row: status glyph,
  first question, age/expiry. Enter mounts that ask in the **existing** `AskPickerScreen`
  (`ask_picker.py:537`); the single-mounted-card widget model stays — the *queue* is a store,
  not widgets. **Flag on:** `on_settle` (`:846`) calls `respond` (the queue). The
  parked-future resolution (`_ask_pending`, `app.py:24949-24990`) is **deleted in F, not in
  B** — with the flag off the TUI is byte-for-byte today's path.
  Gate identity for #1315 drafts becomes `("ask", ask_id)`; approval identity untouched.
- **Minimized state (§5.0, R7):** the single-line bar above the composer is the default
  presentation of a queued ask; the composer always sends chat
  (§5.0 amended: it never answers; the picker's `Other` row is the free-text door).
  Sidebar/`session_catalog` gains the outstanding-asks state, distinct from `pending`/approval.
- **Interaction change to confirm (D5, flag-on only):** with the flag on, Esc *closes the
  card and leaves the ask open* and decline is an explicit action; with the flag off Esc
  declines exactly as today (`builtin.py:24408+`). Esc here is the collapse in §5.0.
- **Response card:** transcript block for `ask_response`/`ask_timeout` in
  `tui/session_presentation.py`, tool-card visual family (`tool_card.py:278` registers `ask` as
  `tool.row.name_meta`), collapsed one-liner, expandable Q&A; secret shows key only.
  The original `ask` tool row shows the receipt ("queued", ask_id).
- **Capture recipes:** new `scripts/ask_queue_shot.py` (pattern of `ask_shot.py`,
  `isolate_capture()`/`save_capture()`, real `OperatorApp` + `FakeSession`, `run_test`,
  `save_screenshot` → view the SVG). Frames at 100x30/130x30/190x50: chip only, list (1/3/8
  asks), card from list, timed-out, late-answered, response card collapsed+expanded, secret,
  **minimized bar (1 and 3 asks), expanded, and the `Other` row closed/open**.
  BEFORE = `ask_shot.py`/`ask_scroll_shot.py`/`ask_long_shot.py`/`approval_shot.py` on
  `origin/main`. Geometry numbers alongside (AGENTS.md "Visual validation" step 4): prompt-host
  height, composer focus, no reflow between first and settled frame. `ask-long-descriptions`
  invariants (approval frames byte-identical) must still hold.
- **Design + UX rounds:** both (new list entry point, Esc semantics, no-steal policy,
  **and the §5.0 minimized bar + the composer-never-answers rule + the `Other` door, which
  the UX round must walk with a real typed draft**).

### 5.2 Desktop UI (repo `local-operator-ui`; PRs C1/C2)
- **Read rule:** when `asks` is present, ignore a `pending_gate` with `kind == "ask"`
  (it is the legacy mirror of an ask already in `asks[]`, §4); approvals keep `pending_gate`.
- **Wire read:** optional `asks` on `CanonicalFrontendState` (`desktop-session-contract.ts`
  ~L790-850 **(UI)**; today's single slot is `pending_gate: PendingDesktopGate | null` at
  `:844`), full-list replace (shallow spread, `use-canonical-session.ts:3556-3569`);
  `pending_gate` retained for approvals/old backends.
- **Entry/badge:** composer status chip beside wake/monitor chips
  (`composer-status-row.tsx` **(UI)**), absent at zero; "Asks" section in run-details
  (`run-details-panel.tsx:136-260`); dock shows the head open ask (`QuestionDock`
  `trace/question-dock.tsx:317-603`), not blocking. **Badge policy conflict:**
  `run-details-trigger.tsx:36-37` forbids count badges; the chip argues an exception
  (a question is actionable) or uses a non-numeric mark — design round decides (D8).
- **Must unwind blocking assumptions (flag-on only — the client reads the flag as the
  presence of the `asks` wire field, §4; old-UI/new-core keeps today's behaviour via the
  mirror):** composer stops swallowing sends when only asks are
  open (`chat-page.tsx:1624-1698`, `canonical-sessions-store.ts:1124`); typed-ordinal answer
  (`ask-answer.ts:99-111`) retired for asks; working line no longer suppressed
  (`working-line-model.ts:455-461`); sidebar must not classify a working session as
  `answer`→RUNNING purely for open asks; `answerState` keyed per ask.
- **Minimized state (§5.0, R7):** the chip/bar under the chat is the default; the composer
  never routes to the ask (§5.0 amended; landed with UI #892) — an answer is typed in the
  dock/sheet, whose `Other` row carries the free-text input. The unconditional composer
  swallow is retired (it stays for the flag-off path only). Chat-list status gains the
  outstanding-asks state, distinct from approval.
- **Answer path:** addressed by `ask_id` + whole-ask `answers`; new refusal-copy families for
  timed-out/late beside `SETTLED_ELSEWHERE`/`QUESTION_MOVED_ON` (`ask-answer.ts:428-469`);
  late press must reach the backend, not be refused client-side (`:954-974`).
- **Response card:** a receipt kind like wake/peer (`transcript-reducer.ts:458-492`,
  `receipt-row-model.ts`, dispatch `canonical-transcript.tsx`), not a decorated tool row.
- **Notifier:** dedupe key includes `ask_id` (`desktop-notifier.ts:1044`); each queued ask
  may banner once; banners are not retracted on answer/timeout (accepted).
- **Evidence:** Storybook stories next to `ask-options.stories.tsx` registered in
  `capture-evidence.mjs`; states: single, multi-question, several open, answering, answered,
  timed-out, late, secret open/timed-out, empty, error/held; both palettes; before/after of
  dock **and** composer; **minimized bar (1 and 3 asks), expanded, the `Other` row closed/open, and the
  chat-list outstanding-asks row (light+dark)**; live-app composer frames via
  `renderer-driver` (state window mode);
  `pnpm check-themes`, contrast rows; `docs/evidence/manifest.json` re-stamp after each
  fold - one command, `pnpm evidence:fold`, which resolves the manifest per field,
  re-derives the stamps and counts from the merged tree, runs the evidence guards and
  stages the result, so the re-stamp is a step ON the fold rather than a per-field
  re-lay in a second docs-only commit. Run it after the merge has committed and before
  pushing (it amends the merge tip); the driver it installs resolves a MERGE only, so a
  rebase or cherry-pick onto a moved `main` still stops on the manifest.
  **Conflict watch:** #615 (dock mount, answer path,
  composer props, `desktop-contract.ts`), #705 (composer moves to `shared/components/composer/`),
  #708 (transcript rows/working line), #689 (`desktop-notifier.ts`, other author). Hence the
  C1/C2 split (§6): C1 touches contract/store/answer lifecycle only; C2 (views) lands after
  #705/#708 or rebases over them.

### 5.3 Relay + web client (PR D; same repo, `local_operator/mobile/`)
- **Relay:** `mobile/types.py` (`PendingAsk`, `SessionProjection.asks`, ops, `EntryKind`
  `:523-542` region), `projection.py` (asks list separate from `_pending_queue` `:1612`;
  fold `ask_response`/`ask_timeout`; `_tool_row_details` `:455` keeps `ask` receipt),
  `tui_handle.py:1233-1490` and `serving.py` both project from the one queue,
  `attach_client.py:2269` gains `ask_respond`, `daemon.py` `asks_open` on list rows.
- **Minimized state (§5.0, R7):** the chip/bar above the composer is the default; tap opens
  the sheet. The web composer never routes to the ask (§5.0 amended); the sheet's
  `Other` row (landing with the web `Other` PR) is the free-text door. Separately, the
  `forceCollapsed` behaviour must not be used to "minimize" (it hides the panels, which is a
  different promise).
- **Web (`mobile/web/src`):** session-list row chip (and the outstanding-asks row state,
  distinct from approval); header entry → **asks sheet**
  (`components/ui/sheet.tsx`/`projects-sheet.tsx` pattern), aggregated across sessions via
  `GET /api/asks`; multi-question form reusing `PendingCard` fields (`pending-card.tsx:131`);
  the card must NOT set `forceCollapsed` on todos/subagents for asks; response card in
  `tool-row.tsx`/`transcript.tsx` via the new kind (unknown-kind path degrades safely).
  Same-binary bundle ⇒ no skew for the web client.
- **Evidence:** extend `scripts/mobile_overflow_capture.py` + `mobile_overflow_fixture.py`
  (headless Chrome over CDP, touch gestures) with ask states at phone size — **including the
   minimized bar, the expanded sheet, the `Other` row closed/open and the list-row outstanding state**;
   **one reused
  browser, `--use-mock-keychain`, reaped by exact pid**; before/after, light+dark if the
  client has both. Design + UX rounds.

### 5.4 Mobile app (repo `local-operator-mobile`; PRs E1/E2)
- The app has a merged Expo shell + ADRs 0001-0004, open PRs #8/#10/#11/#12; next ADR is
  **0005**. Relay wire is defined only in core (`AGENTS.md:12-15` **(app)**); the app consumes
  §4 verbatim.
- **E1 = ADR 0005 "Queued asks"** (docs PR; agent review + QA; design round only if it carries
  mock frames): wire cardinality = `asks[]` per §4; queue **authority = the relay/runtime**
  (no app-side durable ask store; nothing secret ever cached); terminal states from
  `status`; secret asks never persisted client-side; **route neutrality** (Radient tunnel and
  custom URL identical; inherits header rules `docs/relay/tunnel-edge.md:128-141`);
  notifications v1 = in-app badge + refetch on foreground, and **copy that says the app cannot
  alert while backgrounded** (push = separate RFC, D7). **tui-hosted sessions are
  answerable — the enforcing mechanism is named:** the owner is the TUI app process that
  adopted the session (`tui/app.py` + `mobile/tui_handle.py`), which runs the same
  `AskQueue.reconcile` on the Textual loop. A foreign `answered` appended by the relay under
  `flock` (the app holds no transcript lease for it) is picked up by the owner's reconcile,
  triggered by (a) the ≤60 s earliest-deadline tick (`MAX_ARM_MS`, §2.2), (b) on-focus /
  becoming the active surface, and (c) every turn boundary. **Latency bound: ≤60 s.** With
  **no live owner** (dead/nonexistent TUI process) the relay keeps the historical
  terminal-only refusal (`pending.ts:7-14` region **(app)**, PR #12-scoped) with copy naming
  the reason, so a queued ask is never silently unanswerable — it is either answered by the
  owner or explicitly refused. E1's ADR records both the override and the bound. Updates
  `docs/relay/contract.md`/`feature-map.md`
  (`:46,66,135-137`), `docs/ux/`.
- **Minimized state (§5.0, R7):** the app implements the bar-above-composer + list-badge
  pattern and the amended composer rule (the composer never answers; the sheet's `Other`
  row is the free-text door); E2's docs pass carries a **one-paragraph pointer to
  §5.0** in the ADR — **E1 is not reopened**.
- **E2 = implementation** (after #11/#12 land, since it edits `pending.ts`/`pending-card.tsx`
  — **#12-scoped (`feat/screens-session`), not yet repo fact on `main`**): badge on list/tab,
  asks list, multi-question answer form,
  response/timeout cards, timed-out/late honesty. Evidence via the repo's own harness (PR #8
  mock relay + frame harness); native builds in CI; light+dark, phone size, before/after.
- Rides that repo's own release process; no version bump in the PR.

---

## 6. PR breakdown, sequencing, dependency graph

**Rollout mechanism (PROPOSAL D6): dark-merge, then flip.** Core PRs merge to `main` behind
one module constant `asks.policy.NONBLOCKING_ASK` (default `False`; `LOP_ASK_NONBLOCKING=1`
for QA/evidence — an env seam, **not** a config key), so no release window can ship a
half-feature (TUI without a list, etc.). The blocking path is retained only until PR F, which
flips it, **deletes** the blocking path (`ask_gate` `serving.py:989`, `request_user_choice`
future machinery `app.py:24926`), rewrites the tests that pin it, and ships the prompt/tool
description edits (§9) — i.e. the prompts change in the *same change that makes the
semantics live*. Alternative rejected: a long-lived integration branch (hot files
`serving.py`, `app.py` [2.4 MB], `builtin.py` move under it daily).

> **STATUS 2026-10-03 — F landed in part, and the deletion above is DEFERRED by the
> operator's direction, not by drift.** F as executed (`feat/ask-flip-default`) flips the
> default to the queue, rewrites the tests that pinned dark-by-default, and takes the §9
> agent-facing text with it. It does **not** delete the blocking path: that path is what the
> operator-facing KILL SWITCH (`LOP_ASK_NONBLOCKING=0`) selects, so it stays, stays tested,
> and its text is byte-unchanged. A later PR may retire it once nobody needs the escape
> hatch — until then, do not read the sentence above as the current spec for a deletion.

| PR | Repo / branch | Scope | Depends on | Rounds | Release line |
|---|---|---|---|---|---|
| **A1** | core `feat/ask-queue-core` | `asks/` package (store, fold, policy, render); `AskParams.timeout`+receipt+caps; message types + all registrations (§2.3); `Session` queue, timer, `reconcile`, `deliver_ask_messages`; `ask_timeout` hidden wake row + `is_internal_wake_row`; `AskErrand`; ops `ask_respond/decline/dismiss` (server.py, handle, `attach_client`, `mobile/types.py` validation); retention/cleanup guards; flag-off default; **this design note** committed as `docs/design/ask-nonblocking.md`; new unit + real-runtime e2e | — | reviewer, QA | `Release: minor — dark: durable ask queue engine (no user-visible change until flip)` |
| **A2** | core `feat/ask-queue-wire` (stacked on A1) | `frontend_state.asks`, `PendingAsk`, projection `asks`/`asks_open`, `EntryKind` ask_response/ask_timeout + fold, legacy mirror, desktop route additive fields + `GET /v1/desktop/asks`, relay `GET /api/asks`, list-row `asks_open` | A1 | reviewer, QA | `Release: patch — dark: ask queue wire` |
| **B** | core `feat/ask-queue-tui` (stacked on A2) | §5.1 | A2 | reviewer, QA, **design, UX** | `Release: minor — TUI ask list, response cards` |
| **D** | core `feat/ask-queue-relay-web` (from A2, parallel to B; disjoint files under `mobile/web/`) | §5.3 | A2 | reviewer, QA, **design, UX** | `Release: minor — phone web ask sheet` |
| **F** | core `feat/ask-nonblocking-flip` | flip flag, delete blocking path, rewrite pinned tests, ALL §9 prompt/guide/description edits, full regression e2e across surfaces | A1, A2, B, D merged | reviewer, QA (whole-feature matrix) | `Release: minor — ask is now non-blocking (queued, timeout-bounded)` |
| **F2** | core, one release later | remove legacy mirror | C1 + E2 shipped | reviewer, QA | `Release: patch` |
| **C1** | UI `feat/ask-queue-contract` | contract types, store, answer lifecycle, composer un-swallow, notifier dedupe by ask_id, retire ordinals; no new visual surface beyond what's forced | A2 (fixtures earlier) | reviewer, QA, UX | UI repo's own window |
| **C2** | UI `feat/ask-queue-views` | chip, asks section, dock for queue, response receipt kind, timed-out/late states | C1 | reviewer, QA, **design, UX** | UI window |
| **E1** | app `docs/adr-0005-queued-asks` | ADR 0005 (+ contract/feature-map/ux edits) | — (starts now) | agent review, QA (source verification) | none (docs) |
| **E2** | app `feat/queued-asks` | §5.4 | E1, #11/#12 merged, A2 | reviewer, QA, **design, UX** | app repo's process |

**Dependency graph.**
```
E1 ─────────────────────────────────────────┐ (starts now, no wire dependency)
A1 → A2 ─┬→ B ──┐
         ├→ D ──┼→ F ─→ (F2 after C1+E2 ship)
         ├→ C1 → C2
         └→ E2 (after E1, #11, #12)
```
- **Can start immediately, in parallel:** A1, E1; and, against fixtures built from the §4
  frozen contract, C1, D, B (implementers cut from A2's head once it exists, and rebase — no
  contract change expected).
- **Must wait for the wire freeze:** everything except A1/E1. "Frozen" = this note approved by
  Aida; from then a wire change is an amendment PR to this file, reviewed like code.
- **R7 (minimized ask surface) is in scope for B, C1/C2, D and E2, and for each one's design
  and UX rounds** — the minimized bar, the amended composer rule (never answers; `Other` door) and the
  list/sidebar outstanding-asks state are user-visible, so those rounds cover them (§5.0, §7). E1 (the
  ADR) is **not reopened**: E2's docs pass adds a one-paragraph pointer to §5.0.
- **Merge discipline (each PR):** never a version bump (`pyproject.toml` stays at last
  release); `Release:` line in the body; on merge send the window owner (0.64.10 lock is PR
  #1834; measure the window from tags, not sessions) PR#, merge SHA, Release line; flag Aida at
  open and merge; assign `damianvtran`, no reviewers/pings; non-draft; disclose `--admin`.
  Reviewer round (`### Agent review — round N`, `Reviewer:`/`Scope:`), remediation reply, QA
  round on the same head, design/UX rounds batched with them into ONE remediation commit.
- **PR-size note:** A1 is the biggest; if review wants it split, the natural seam is
  *(store+tool+message types)* vs *(Session integration+timer+wake row+AskErrand)*, both dark.

---

## 7. Test / evidence strategy

Real execution, not green suites. Time-dependent tests inject a clock into `AskQueue`
(floor is 2 min; never sleep on it) plus **one** real-time e2e at the floor marked slow.
Wait on events (`ChangeSignal`), never on the clock (AGENTS.md timing section).

**Core (A1/A2), `tests/e2e -m e2e -n0` against a real runtime + control socket, isolated
`env -i` config dir, CMUX_* unset, synthetic ids:**
- accumulation: 3 asks queued while the agent keeps working; answer out of order; each yields
  exactly one `ask_response` (assert transcript row count by id).
- multi-question atomic submit; partial submit rejected; duplicate submit from two surfaces →
  one winner, loser gets "already answered by <surface>".
- timeout → `ask_timeout` in transcript AND model context (render.py branch) for normal and
  urgent; late answer → `late` with the same `ask_id`, the **two-row pair**
  (`ask-timeout-` then `ask-response-`, or response-only when both land in one batch — N8),
  never double-delivered (`has_entry`); a **declined** ask writes one
  `ask-response-` row with `status:"declined"`; dismiss of a timed-out ask injects nothing and
  writes **no** row; an ask past `expires_at + 7 d` with no answer writes **no** row; the
  `delivered` flag never flips `true`→`false` after folding to `expired`; answer-vs-deadline
  race in both orders (pure fold).
- **restart durability:** SIGKILL the runtime with open asks → engage → asks re-surface on
  `asks` wire and aggregate index; answer-then-kill-before-delivery → delivered exactly once on
  boot; timeout while no runtime → wake supervisor engages, notice delivered once.
- **session-agnostic:** session started by `wake` fire; session started by a `monitor` fire;
  `exec --control`; in-process TUI host — each asks, is answered, gets the response.
- **stop:** ask open + stop → ask survives, no timeout fires while `stopped_at`, reopen
  delivers "lapsed while stopped"; `ask_timeout` does not clear a sticky abort,
  `ask_response` does.
- **secret:** sentinel value never appears in `asks.jsonl`, the index, transcript, events,
  notification text or card details (grep the files); cold answer path refuses; restart
  between answer and use → "lost" response; `persist` survives; duplicate open key refused.
- caps and bounds: >8 open refused; timeout 119 s / 86401 s / `"2h"` / `"junk"` errors name
  the bounds.
- old-shape compat / skew (the §4 matrix, exercised for real): (1) legacy `ask_answer`
  against the mirror resolves onto the queue; (2) **new core + old *mobile* client** — the
  fold's `_sync_pending` mirror (`mobile/projection.py:2876-2880`) still projects the head
  open ask as today's `pending` + `pending_count`, the old app/web shows and answers it, and
  no ask is lost; (3) new core + old *desktop* client — same via `_publish_pending_gate`;
  (4) new core + old *TUI host* — same via `tui_handle._publish_pending_gate`; (5) unknown-op
  from an old runtime → the client's "predates queued asks" copy; (6) old core + new
  client → no `asks` field, nothing new rendered.
- Unit: fold table, index self-heal (corrupt/unknown schema = absent), flock append under
  concurrency, `is_internal_wake_row` subtracted from every human wake surface,
  import-graph pin for `asks/store.py`, prompt-text tests (`tests/unit/test_prompts_api.py:388`
  and the `tools`-side description pins in `tests/unit/tools/test_ask_tool.py:107-180`)
  rewritten in F, present unchanged in A1/A2.

**Per surface (B, D, C1/C2, E2):** the same matrix through that surface's real UI —
queue accumulation, answer flow (single + multi-question + secret), timeout→timed-out state,
late answer, restart (kill runtime, surface re-shows asks), response card expand/collapse,
no-regression on approvals and the composer. **Before/after frames** per §5 recipes with
geometry numbers; timed-out and late states must appear in frames. Frames go on the PR, not
in the repo (AGENTS.md visual step 7). Frame the first and settled state when anything
animates or ticks.

**R7 — the minimized surface (every user-visible PR: B, C1/C2, D, E2) — AMENDED 2026-10-08.**
A dedicated case, because the composer rule is a *behavioural* claim that stills cannot show:
- **Asserts (automated where the surface allows, walked by hand in the UX round otherwise):**
  (1) an ask arriving while a chat draft is in the composer leaves the draft intact and the
  ask MINIMIZED; (2) in **every** state — minimized, expanded, no surface mounted — Enter on
  the composer sends a **conversation** message and produces zero answers (the route is
  never reached from the composer; the flag-off path of the rule above is the exception and
  is not asserted here); (3) with the surface expanded, the answer is given only in the ask
  surface: the `Other` input (or a free-text-only question's input) writes the draft, and
  nothing typed or pressed there reaches the composer. Each surface states its own submit
  gesture: the TUI advances/accepts on Enter; the desktop commits on Enter inside its
  `Other`/free-text field — Enter does what `Send answer` does when the whole ask is
  complete, otherwise it moves to the next question that still needs an answer (Shift+Enter
  is a newline); the relay web and native surfaces submit through their own explicit control
  over the whole draft (one atomic form), not on Enter; (4) collapsing preserves the
  composer's draft and the ask surface's draft, and
  re-expanding restores the ask surface's draft on surfaces that keep one (not the native app, which persists
  nothing, ADR 0005 §2); (5) an ask settling while expanded leaves the composer exactly as
  the user left it (placeholder never swapped, draft kept in the buffer, text never moved); (6) the minimized bar is absent
  at zero asks and shows head + count at 3.
- **Frames:** minimized bar (1 and 3 asks), expanded with the `Other` row closed and open
  (landed for the desktop with UI #892; other surfaces once their `Other` PR lands), and the
  list/sidebar outstanding-asks state — **light and dark** where the surface has
  both, at phone size for web/app. Before/after against `origin/main`.
- **No wire assertions:** R7 is client-local interaction state and changes nothing in §4 —
  any PR that touches the wire for R7 is out of scope.

**F:** full cross-surface matrix on the flipped default plus a regression pass on approvals,
wake/monitor/patience surfaces (the `is_internal_wake_row` swap), `lop sessions`, cleanup
guards. Whole-tree unit suite only at the frozen head / CI (AGENTS.md: 40-55 min).

---

## 8. Risks, open questions, decisions to confirm (all PROPOSALs)

| ID | Decision | Recommendation |
|---|---|---|
| D1 | `urgent` derived from `timeout ≤ 15 min`, no extra param | yes |
| D2 | Late window 7 d after deadline, then `expired`; a late answer **within the window does** start a turn (past it: the answer is recorded but not injected — same 7 d bound as the wake staleness) | yes |
| D3 | Secret default = normal default (1 h) | yes; alt 30 min if Aida prefers a tighter posture — no storage difference |
| D4 | One-release legacy mirror (`pending_gate` + fold `pending`) | yes; the alternative is silent feature loss for skewed clients |
| D5 | TUI/UI: Esc closes the card, ask stays open; decline is explicit | yes — it is the semantic that makes non-blocking honest; UX round to confirm |
| D6 | Dark-merge behind a module constant then flip PR F | yes; prompts/tool description ship with F |
| D7 | Mobile push notifications | **out of scope**; app v1 = badge + foreground refetch + honest copy; separate RFC (relay APNs/FCM, F-Droid constraint `docs/adr/0004-ci-cd.md:178` **(app)**). Operator to confirm they accept "no background alert" for now |
| D8 | Desktop numeric badge vs `run-details-trigger.tsx:36-37` policy | design round decides; recommend non-numeric dot on the chip, count inside |
| D9 | No new config key; open-ask cap 8 | yes |
| D10 | Wake-engine reuse (hidden `ask_timeout` row) vs a supervisor extension | reuse; a second timer substrate is the defect class AGENTS.md warns about |

**Risks to watch in rollout**
1. **Receipt read as consent.** Guides that say "ask, then run the command" (system-tools,
   console, mobile install) become *unsafe* if the agent proceeds on the receipt. Mitigation:
   receipt + description say so in capitals; every guide site in §9 rewritten in F; QA must
   probe a model transcript for premature action.
2. **Spam / cost.** Cheap asking → many asks; cap 8; identical-text dedupe; timeouts batch
   into one paid turn; `ask_timeout` never wakes a stopped session.
3. **Secret lost across restart** (handled, but the only path where the user answered and the
   agent still can't proceed) — message must be unmistakable.
4. **Wake-row reuse blast radius.** Any human surface that lists wakes and misses the new
   predicate leaks a hidden row; F's regression pass enumerates the 19 measured sites (§2.2).
5. **Hot-file conflicts:** `serving.py`, `session.py`, `app.py`, UI #615/#705/#708. Keep
   stacked PRs small and fold often; don't let B and D edit shared test helpers.
6. **flock contention / torn last line** in `asks.jsonl`: reader must tolerate a partial
   final line (inbox precedent).
7. **Unattended session with no wake scheduler** notices deadlines only at next boot
   (degrade, documented). *Evidence that settles it:* a spike listing which ask-capable hosts
   construct `WakeScheduler` (`session.py` load ~`12930-13014` **(scout)**).
8. **Old-UI composer swallowing** while the mirror is up (accepted, time-boxed to F2).
9. **Test blast radius of F:** ~40 test files pin the blocking shape (`test_ask_tool.py`,
   `test_ask_settle.py`, `test_ask_picker.py`, `test_parked_gates.py`, `test_tui_ask.py`,
   `test_reload_gate_delivery_e2e.py`, `test_desktop_sessions.py:481`, …). They stay green
   **unchanged** through A1–D **only because every new path in A1/B/C1/C2 is reachable just
   with `NONBLOCKING_ASK` on (client-side: the `asks` field present) and no B/C1/C2 diff
   removes an old path** (the §5 invariant). F is what deletes the old paths, rewrites these
   tests, and ships §9 — that is the point of D6, and it is the one PR allowed to touch
   them.

**Unverified (say what settles it):** remote secret over `ask_answer` (read `server.py`
credential refusal in A1); whether the built web bundle is committed (check before D);
`ChatPage` remount on session switch (UI C1 spike); `tui/app.py` line numbers are from a raw
fetch of main — re-grep before editing.

---

## 9. Prompt / guide / agent-facing text — edit list (all in F, drafts reviewed earlier)

1. `tools/builtin.py` `build_ask_tool` description (`:24276+`): rewrite. Must state: returns a
   receipt at once; the answer arrives later as an *ask response* turn; **a receipt is not
   consent**; `timeout` semantics + calibration (1 h default; 5–10 min urgent; ≤24 h
   non-urgent; floor 2 min); timeout notice and late answers; caps; keep the LAST-RESORT
   framing; drop "the user answers the questions back to back" and "Esc declines". Constants
   `ASK_UNANSWERED_TEXT`/`ASK_SECRET_UNANSWERED_TEXT` (`:24152-24169`) move into
   `asks/render.py` for the `declined` status.
2. `prompts_md/system.md` **L183-236** (the "Deciding is your job; `ask` is the exception"
   section through "When you do ask"): reframe — asking is non-blocking, so *continue other
   work*, never idle on an ask; if nothing else remains, end the turn stating what is queued;
   line ~224 "If the user answers nothing, take your own recommendation" → "if a timeout
   notice arrives, take it"; add urgent→subagent-expert guidance. Line ~316 (system-tools
   password `ask`) revisited. Pin: `tests/unit/test_prompts_api.py:388` (phrase "Deciding is your
   job; `ask` is the exception") — keep the phrase or update the test in the same commit.
3. `prompts_api.py` `<interactivity>` bodies: `_INTERACTIVITY_ATTACHED_ASK` (`:611`) and
   `_INTERACTIVITY_DETACHED_ASK` (`:666`). The detached one currently says "prefer to
   PROCEED… over calling `ask`" and "the turn may block for hours" — both false now; new
   text: an ask queues durably and is shown when someone attaches; it never blocks; keep "write
   for a reader who may answer hours later". `_ATTACHED_HUB/_NONE`, `_DETACHED_HUB/_NONE`
   unchanged. Stable per session (prefix-cache rule, `:588-603`).
4. New `guides/ask/GUIDE.md` (guides are packaged, `guides/discovery.py`): lifecycle, states,
   timeout calibration table, late answers, secret asks, what an `ask_response`/`ask_timeout`
   turn looks like, urgent→subagent. Listed by name/description in `<guides>` (footprint
   ladder: no extra tool schema).
5. Existing guides that assume blocking (main line refs): `guides/browser/GUIDE.md:343`
   ("NOTIFY THE USER YOURSELF… use `ask`"), `:555`; `guides/console/GUIDE.md:130,137,150,169,205,210`;
   `guides/credentials/GUIDE.md:206`; `guides/mobile/GUIDE.md:90` ("use `ask` before install");
   `guides/system-tools/GUIDE.md:110-112,265`; `guides/teams/GUIDE.md:26`. Rule for each:
   *queue the ask, do independent work, act only on the response turn.*
6. Runtime-facing strings: `harness/rows.py:660 gate_timeout_notice` neighbours (new
   `ask_timeout`/`ask_response` row copy), `harness/render.py` branches, TUI/UI/mobile card copy
   from the shared copy contract (§5).
7. Out of scope, do not touch: `evaluation/*` `ask_user_exchange`/`AskUserAction` (a separate
   benchmark concept); `harness/comms.py` `"ask"` (subagent→parent comms kind).
8. `AGENTS.md` (core): add a short "Ask queue" pointer to this note under the relevant
   section, and the two new capture scripts; `docs/design/ask-nonblocking.md` is this file.

---

## 10. Amendment (2026-10-03): in-flight answer revision, bounded by delivery (#1936)

Status: AMENDMENT to the §4 frozen contract — a wire change made HERE and reviewed like
code, per the rule above. Drafted from the ask lane's contract answer (2026-10-03) against
`origin/main` @ `78117f97b`. It is a **queued-arm behaviour and takes no new flag**: it
rides the flip (#1941), which is what makes the queued arm the default (§6). The code
change follows this merged text; it cuts only after the flip.

**What this amends.** §2.2's event table gains `revised`; §2.4 gains the revision path and
its refusals; §4's ops line gains `ask_revise` and the desktop body gains `revise`; §5's
copy contract gains the delivered refusal. The fold's precedence table, the delivery rules
and §3's timeout policy are unchanged, and the terminality horizons (`LATE_WINDOW_S`,
defined in `asks/store.py` and re-exported by `asks/policy.py`; the late/expired bounds)
are untouched: **this amendment adds no time bound of its own — its only bound is
CONSUMPTION** (amended 2026-10-04: was DELIVERY; see the window bullet).

**The problem.** A multi-question ask answers forward-only: once a response is recorded, a
changed answer is refused — `already answered by <surface>` on the whole-ask path, and
`answer_one`'s "a repeat tap is a retry, not a change of mind" on the legacy bridge — even
while the agent has NOT been handed the answers. A mis-entered answer therefore costs an
interrupt and a re-ask (#1936). The amendment admits exactly the missing case: a revision
between the answer's recording and its delivery.

- **The intent is explicit; value equality is never the marker.** A revision is its own op:
  `ask_revise {ask_id, answers{qid:[str]}, by?}` on the command wire (registered where
  `ask_respond` lives — server dispatch, the TUI handle, `attach_client`, the relay op
  list, `mobile/types.py` validation), and `revise` on the desktop answers body
  (`POST …/answers` with `ask_id`+`answers`). `AskQueue.revise` / `Session.revise_ask`
  ride `respond`'s atomic whole-ask path — the same complete-map rule (an empty list is
  how "no answer" is said), the same secret-cell rule (`[<key>]` only), one `flock`'d
  append, one reconcile — and differ in exactly what they supersede. No layer may infer a
  revision by comparing values: equal values are not a retry marker, and different values
  are not a revision. `respond` and `answer_one` keep their refusal SENTENCES
  byte-for-byte for the retry case, and their docstrings gain the one sentence that names
  the sanctioned exception — so the rule ("a repeat tap is a retry") and the exception are
  both written where taps land. Additive only, no `PROTOCOL_VERSION` bump: an old
  registrant answers `unknown op` (the client says the runtime predates queued asks), and
  an old client never sends it.

- **The window is bounded by CONSUMPTION, not by handoff.** A revision is accepted iff,
  when it is serialised against the log, the model's conversation does not yet carry the
  answer — status `answered` or `late` whose `ask-response-<ask_id>` row is not yet DURABLE
  in the transcript. `delivered: false` means exactly that (for `late` the response row is
  what delivers it — the deadline notice does not; §4). Handing a message to a delivery
  path, scheduling a turn, or returning from the answering op closes nothing: that is our
  latency, and a window sized by it is sized by nothing the user cares about. The row's
  append is the close point — the answer entering the conversation the model reads and (the
  transcript being append-only, one row, no rewrite) the last instant a revision can still
  change what the model will read. A revision arriving while that append is in flight is
  refused, conservatively and for milliseconds, in the same delivered sentence: content
  snapshot taken and carry provable cannot both hold. Successive revisions while the window
  is open stay legal; the latest accepted one is effective, and it is what the append
  carries because every delivery path RE-RESOLVES the row from the fold at its append (the
  message built at reconcile is a preview, not the contract). `reconcile`, `_handed` and
  the awaited ACK are scheduling, never the bound. Against an ask with no recorded answer yet
  (`open`/`timed_out`) the intent degrades to the plain first answer: one `answered`
  event, no `revised` — the `revised` event exists only to supersede.

- **A revision is not a race; the winner rule governs races, not revisions.** While the
  ask is undelivered, a deliberate revision from ANY surface of the session is accepted
  and supersedes — the surface is not a permission, and no layer may "restore" a surface
  gate as a safety measure. Single-winner arbitration stays for the cases it exists for:
  two submissions actually in flight, and a plain non-revise second `respond` — the loser
  still reads `already answered by <surface>`. If the code cannot cheaply tell an
  in-flight race from a sequential revision, the honest rule is explicit: plain
  `ask_respond` keeps the winner rule; `ask_revise` supersedes while undelivered
  regardless of surface.

- **Refusals: the state table stays state-only; one sentence is op-qualified.** A
  revision in-window → accepted, from any surface. **A revision once the response row is
  **durable** (or an append of it is in flight) → `already delivered — send a new message`**
  — no silent overwrite, ever: the
  row pins what the model was told. That sentence is emitted by the revision path itself
  (`AskQueue.revise` / `Session.revise_ask` — the path that carries the op):
  `render.refusal_copy(record)` receives only the record, never the op, so it keeps its
  state table byte-for-byte — a plain repeat `respond` on a delivered answered ask still
  reads `already answered by <surface>` (as the byte-for-byte sentence above promises),
  and `declined`/`dismissed` → "you already declined this." / `expired` → "this ask
  expired 7 days ago — ask again if it is still needed." stay state-mapped. A race's
  loser → "already answered by <surface>." The repeat tap on the current gate keeps
  today's exact wording.

- **The log records an EVENT, not a replacement.** New kind `revised`:
  `{v, ask_id, at, by:{surface}, answers}` plus the `at` of the write it supersedes,
  appended by the same atomic write path as `answered` (the §2.2 cold-append rule applies
  unchanged: any process may append under `flock`, non-secret only). The fold reads
  status, `answered_at` and `answered_by` from the first `answered` (unchanged), and the
  effective `answers` from the LATEST `revised` event when one exists. Delivery, row ids
  and the `delivered` flag are untouched: exactly one `ask-response-<ask_id>` row,
  carrying the effective map — a revision can never add a second row or buy a second
  turn. **Ordering invariant (what the code PR must show):** acceptance, the append and
  the row's content can never disagree — an accepted revision is what the delivery append
  carries (each append re-resolves content from the fold), and a revision that lost the
  race to a durable row **or to an append in flight** is refused with the delivered
  sentence; never accepted-and-then-dropped.

- **Under the kill switch** (after the flip, `LOP_ASK_NONBLOCKING=0` is the kill switch;
  at head the same seam is `=1`-enables, read once at import — `asks/policy.py:52`): the
  old picker submits once, so a revision never arises for a real client — and a stray
  `ask_revise` must answer in words, not with a traceback: the op family registers
  unconditionally (`session/runtime/server.py:6995`) and a session without a queue refuses
  with a sentence (`Session.respond_ask`'s guard, `session/session.py:8876`; a revision
  mirrors it). Never a silent success, never an assumption that the queued machinery is
  live.

- **Surfaces** (the code PR after the flip carries them, with their own design/UX rounds):
  the desktop question-dock and the mobile picker gain a **change** affordance on an
  answered-but-unsettled question (`delivered: false`); a revision sends the WHOLE ask map
  atomically — the same payload as the initial response, never a per-question amend post;
  the settled-refusal sentences above render honestly, because that path is reachable and
  a silent no-op would be the worse failure.

- **Evidence.** The one-way refusal today is pinned by
  `tests/unit/asks/test_queue.py:212` and a dedicated repro was written and SAVED (two
  questions; the answer recorded; a changed map refused while `delivered: false`).
  **Running it was deferred under host pressure** (free pages fell to ≈145 MB; no-builds
  policy) — the repro file and its exact bounded run command are in the PR thread, and
  the code PR re-derives it against this text with the §7-style matrix: revision accepted
  before delivery (including from another surface); refused after delivery with the new
  copy; a race's loser keeps `already answered by <surface>`; second `respond` and
  repeated `answer_one` byte-unchanged; the log shows `answered`+`revised` with ONE
  response row; kill-switch refusal without a queue.

## 11. Amendment (2026-10-05): TUI fleet scope, filter, marks

**Scope seam — one list, two scopes.** The TUI has no session-less screen, so `AskQueueList` is the ONE
expanded surface and is told its SCOPE by the door it was opened from: the bar / `f4` opens *This
conversation*, the sidebar footer's `asks: N` note opens *All conversations*. The list paints the
subject and offers no toggle (a second scope store, a second list and a second entrance are all
rejected). The fleet rows come from `asks.store.index_asks` — read ONCE, off-thread, when that scope
opens; never on the 2 s poll, because the store's own comment puts a cap before that read "ever feeds
a frame". Leaving the surface (Esc/`f4`) leaves the scope with it.

**Total on the fleet surfaces.** The fleet total is the SUM of every session's outstanding count
(index tally), unioned per session with the current session's live wire count. It is painted on the
sidebar's footer note (`asks: N`, only while N > 0 — absence is not emptiness) and on the session
picker's chrome row, and never as a chip of its own. `tui.sidebar_visible` still defaults off, so a
user with the sidebar hidden sees the total only in the picker.

**Marks.** `session_sidebar.set_asking` now takes `{session_id: count}` and marks EVERY session whose
outstanding set (open ∪ timed_out — the same set the wire's `asks_open` publishes) is non-empty, one
glyph cell per row, under the existing `!` gate mark. The index read rides the sidebar's own poll,
throttled to 10 s.

**Rows, filter, registers.** `ask_rows` no longer drops settled rows: it carries `delivered` and the
halves are `pending` (answerable ∪ delivering) and `settled`. A settled row is a READ-ONLY one-liner —
Enter/`d`/`x` are inert and it wears its status as a WORD in place of the status glyph. The one-line
header carries the three segments `All · N` / `Waiting or moved on · N` / `Settled · N` (`1`/`2`/`3`,
`[`/`]`, and the segments are press targets), and its count is the DRAWER register: `N waiting, M moved
on` when mixed, else the chip clause (`queue_headline`), with the backend's `asks_open` and no split on
a truncated frame. The bar keeps the chip register unchanged. Each empty half states its own sentence,
so a filtered view can never read as an empty queue. At the narrow floor the drawer clause yields
first, then the scope subject; the filter control yields last, because a view that cannot be
un-filtered is worse than an unadvertised key.

**Answering a fleet row.** By the ROW's own `session_id`, never by what is on screen: a row of the
adopted session keeps the owner contract, and every other row goes through
`engage_session_client(config_dir(), session_id, AskErrand(ask_id))` → `ask_respond` / `ask_decline` /
`ask_dismiss` → `close()` in a `finally`, on a Textual worker, with an in-flight row state sized for
engage 30 s + ack 15 s and no double-fire. Refusals arrive as the ask's own sentence
(`asks/render.refusal_copy`) at the row. No HTTP, no new transport, no new timer.

**Client-read-only.** No wire change, no new config key, no new composition root: this reads the index
the desktop's aggregate route already reads, and answers through the ops that already shipped.

### 11.1 Amendment (2026-10-05, round 2): the review round's rulings

**Exact labels.** Scope subjects are `This conversation` / `All conversations`; the segments are
`All · N` / `Waiting or moved on · N` / `Settled · N`; the three empty sentences are the desktop's,
verbatim. The drawer's single-half clauses now speak the DRAWER's vocabulary — `N questions moved on`
(never the chip's `N asks timed out`), because the segments above and the rows below already say
"moved on" — and the all-settled state takes the desktop drawer's own constant, `All asks settled`. The
`· K urgent` suffix on the mixed clause stays (the desktop states urgency in an sr-only channel; the
TUI has none, so it is stated in words). The bar keeps the chip register unchanged.

**The delivering half (§10) is PENDING and never SETTLED.** An answered-but-undelivered ask sits in the
`Waiting or moved on` segment; the session's MARK cleared when it was answered (§4.8's outstanding set is
`open`+`timed_out`: it tracks the debt the USER owes, and this one was paid); its drawer clause is `N answer(s) delivering
— the agent will be told`, in §5's words and
deliberately WITHOUT the desktop's "you can still change
it" (this surface has no revise wire — recorded as a deferred follow-up, not silently promised). Its row
is a read-only one-liner until delivery (not answerable — a second answer is what §10's window exists to
prevent) and its tail says `answered, delivering` / `answered late, delivering`.

**Registers and contrast.** Every settled/delivering row carries its status as a word after the row's own
`·` separator, so a muted status cannot read as part of the question. The panel's state inks are the
derived `chip-live` / `chip-success` / `chip-warning` family (`theme._fill_chip_live`), which is the
family the palette gate now checks against `overlay` — the brand light ramp's accent and success both sat
under the state floor there, and the dark ramp is byte-identical (there the derived ink IS the hue).

**Doors, in keyboard and mouse.** The fleet scope keeps ONE entrance in the model — the app's
`action_open_fleet_asks` — and now has three gestures onto it: the sidebar's footer note (mouse), the
sidebar's own `ctrl+f` in F9 mode (taught on the same footer line, beside `ctrl+k pin`), and the session
picker's `asks: N` chrome (a press target on its own cells). Pressing any of them while a fleet list is
already up RE-READS in place: the mounted list is re-pointed, never re-mounted (a second mount raised
`DuplicateIds`, which ends the session). The narrow-floor scope cue is the ROW TAIL (`handle · deadline`)
— with no room for the subject, the rows are what say which conversation a queue belongs to.

**Truncated frames.** The clause states the backend tally (`N outstanding`) and withholds the split; the
three segments keep their live counts, because they are the filter control the operator asked for. The
two disagree on screen by design — the segments describe the rows this frame carries, the clause the
queue behind them — and that difference is the honest signal that the list is a prefix.

**Hints follow the view.** The header's hint set is derived from the VISIBLE rows: `enter`/`d` need an
answerable row, `x` needs a moved-on one (dismiss is offered on a timed-out ask alone), `esc` needs
nothing. The spend order is by irreversibility — `d decline` outlives `enter answer`, which already has
the `❯` cue — so at 100×30 the irreversible key is named again, as it was before this work.

**Client-read-only, restated.** No wire change, no new config key, no new timer, no HTTP client in the
TUI: this reads the index the desktop's aggregate route reads and answers through the ops that shipped.

## 11.2 Amendment (2026-10-05, round 2): the residuals, and four corrections

Round 2's review, QA, design and UX passes left F12–F18, D11–D15 and U12–U14. What changed, and what is
now recorded rather than claimed:

**Corrections (the note was wrong, not the code).** (1) A **delivering** row does **not** hold the
session's mark: the mark is §4.8's outstanding set (`open` + `timed_out`) unioned with the current
session's answerable rows, so it clears the moment the user answers — the debt it tracks is the USER's,
and this ask's was paid. Four sites said otherwise; all four are fixed here and in the code. (2) At
**100×30** the fleet header's two-rung ladder drops the **subject** as well as the drawer clause, so the
fleet and session lists are textually identical above the rows there; the scope tell at that width is the
**row handles** (`handle · deadline`), which `test_the_fleet_rows_carry_their_own_handles_at_the_doors_own_width`
pins. (3) The `ctrl+f asks` teacher needed 46 cells against a footer whose content saturates at 43, so it
rendered **nowhere**; the chord's rung now outranks the pin's (the pin has a second teacher on `/help`,
the chord had none) and spends two lengths — `ctrl+f asks` from a **120-column terminal** (the first
width whose 34 content cells fit it; 130 is where the footer saturates at 43), `ctrl+f` at the 29-cell
floor (a 100-column terminal, the width the door's own `f9` path produces), where the note beside it
supplies the object. (4) The `ctrl+f` shadow is **declined** while an aside is
open (`check_action` → the app's `fork_aside`), because the aside's own copy advertises that fold; and
`/help`'s `ctrl+f` line now names both meanings.

**The fleet tally is a function of the INDEX, not of the writer that ran last (F12).** The list's count
facts ride with its rows on every write — the door, the re-read after an answer, and the frontend
snapshot — so a capped header cannot revert to a row-derived split one frame later.

**The clock follows the ACTIVE SCOPE (F14).** `_sync_ask_tick` arms on the scope's rows, and the door
re-evaluates it: a fleet list whose rows belong to other sessions has a countdown that moves.

**The sidebar's mark takes the derived ink too (D12).** The panel moved to `chip-*` in round 1 and the
sidebar did not: a focused cursor row paints `tint-select-hi`, a ground in neither derivation, where raw
`accent` measures 3.96:1 on the light ramp — under the repo's own 4.0 state floor — 14 of 54 ramps under
it, two under 3:1. `chip-live` clears all four sidebar grounds on every ramp (worst 4.66, `everforest`),
and `test_the_ask_marker_reads_on_every_sidebar_ground` is the pin. **The accepted residual (round 3:
D16):** on the light ramp the derived ink IS the ramp's primary text colour, so the mark on a row whose
title paints `fg` is the same ink as that title — the mark is carried there by its glyph and its cell,
not by hue, exactly as the panel's light ramp is (Ruling 1). **D13 recorded, not fixed:** the
focused sidebar's whole footer line is under the `dim` floor on the light ramp (3.35:1 against 3.4; 11
ramps miss on `tint-select`, 27 on `tint-select-hi`) — that ground is outside the gate's set for every
`dim` consumer, and re-flooring 11 curated ramps for 0.05 is not this PR's call.

**The picker's count is a DOOR and now reads like one (D14/U12).** Under the default
`tui.sidebar_visible = False` it is the only fleet surface a user sees, and it was painted `dim` —
byte-identical to the inert `N sessions` legend beside it (2.72:1 light), with the same `default`
pointer. It wears the sidebar note's affordance (the hand, and an underline on its own cells under the
pointer) and the `muted` ink, which clears its ground on both ramps.

**Evidence, fixed rather than re-worded (D11).** `visual_capture.isolate_capture` pops `NO_COLOR` by
design, so the round-2 "NO_COLOR frame" was a byte-identical copy of the colour one. The colour-less
render is now an explicit shot MODE, re-asserted above the app import, and the frame differs (md5
`ecd382615de0…`): the product was always fine colourlessly — the artifact was the lie.

**Client-read-only, restated.** Still no wire change, no new config key, no new timer, no HTTP client in
the TUI.


## 12. Amendment (2026-10-05): agent-side settle — withdrawal, and the chat-answer attribution

> **STATUS: SYNCED — shape acked (Aida, 2026-10-05); implementation starts from this
> commit.** The operator's own words confirm both reasons are AGENT-side ("as it's working" →
> moot; "if the user sends a response in the chat" → answered_in_chat). Drafted against main
> `1f0a1b909` (v0.67.16).

### 12.0 The gap this closes

Every write to the ask log today is authored by a SURFACE on the user's behalf — `ask_respond`,
`ask_revise`, `ask_decline`, `ask_dismiss` (§2.4). **The asker has no path.** Two real shapes fall
through it:

* **Moot.** The model asked; then the answer stopped mattering (it found the answer itself, the work
  moved on, the user said "never mind" in chat). The ask then sits `open` → `timed_out` → 7-day
  expiry, besetting the bar and every list as a question nobody will answer. Nothing is factually
  wrong — every surface is *honest* — and that is the defect: an outstanding set containing questions
  nobody will answer is how the operator learns to ignore the set. `dismiss` cannot carry this: it is
  refused unless `timed_out` (queue.py), it is the USER's view action, and its copy is the user's
  voice.
* **Answered in chat.** With the composer no longer an answer door (§5.0 amended),
  `answered_in_chat` withdrawal is the only way a chat reply settles an ask. So when the
  operator simply replies in the transcript — the most natural way to answer a question they
  can see — the ask keeps reading "waiting" although the agent HAS the answer, and the
  surfaced count is now a lie with a clock on it. The only safe detector of "that message WAS
  the answer" is the model itself; what it lacks is a way to RECORD the attribution.

### 12.1 Proposal: one agent-facing op, two reasons

`ask_withdraw {ask_id, reason: "moot" | "answered_in_chat", message_id?}` — agent-authored only
(the tool naming is open: `ask_withdraw` / `ask_settle`; "withdraw" matches the operator's word for
the half).

| reason | event appended | folded status | injects |
|---|---|---|---|
| `moot` | `withdrawn` (NEW kind) | `withdrawn` (NEW, terminal) | **NOTHING** (symmetric with `dismissed`) |
| `answered_in_chat` | `answered` (existing kind), `by: {surface: "chat", message_id?}` | `answered` (existing) | the standard response row, cells = the user's words |

**Cells for `answered_in_chat`.** The model records the user's message **verbatim** — one cell per
question the message answers; a question the message does not cover is sent as an EMPTY LIST, which
is how §2.4 already says "no answer" without omitting the key. A **secret** question is REFUSED for
this reason (there is no masked-entry hop from chat text, and there must not be one: the card
remains the only secret path) — its own refusal sentence.

**Refusals** (extend `refusal_copy`, render.py — every surface reads the same words):

* answering a `withdrawn` ask → `"the agent withdrew this question — if you have an answer, send it
  as a chat message."` (answer box hides everywhere, because `withdrawn` is not outstanding)
* `withdraw(moot)` on an ask with an `answered` row → refused, no state change. On
  `declined`/`dismissed` → no-op with a truthful sentence. On `late`/`expired` → no-op.
* the existing user-voiced sentences are kept for user ops; the op-level returns to the MODEL may
  carry their own wording where "you already declined this" would read wrong.

**Fold rule and the ONE race.** `withdrawn` is terminal-on-write **except that every user-act
terminal row present (`answered`, `declined`, `dismissed`) outranks a later `withdrawn`** — the
user's own acts are never overridden by the asker's retraction; a racing withdrawal loses, and the
model may re-ask. Only a `withdrawn` with no such sibling folds to `withdrawn`. The op also refuses
a settled ask, so the fold caveat is belt-and-braces for the true cross-process race only.
(Contrast `dismissed`'s documented shadowing, prevented surface-side because only a `timed_out` ask
offers it; `withdrawn` is agent-callable at any time, so the guard must live in the fold.)
**Reader tolerance**: an old build (pre-`withdrawn`) folding a log that contains `withdrawn` rows
must keep working — confirm `fold` skips unknown kinds the way the torn-line rule skips bad rows
(plugin: the store's "corrupt or unknown ⇒ tolerated" contract).

**Who may call / plumbing.** v1 is agent-only and RUNTIME-LOCAL: the model's tool call runs inside
the session runtime, and `Session.withdraw_ask` wraps `AskQueue.withdraw` exactly as
`respond_ask` wraps `respond`. **No new remote wire op, no relay change in v1** — the mobile/desktop
ops stay respond/revise/decline/dismiss. The only wire-visible change is the new folded STATUS value
travelling out through the existing index/frame flow, which surfaces must RENDER (copy/maps).
Kill switch: rides `LOP_ASK_NONBLOCKING` (the old arm has no queue; the tool is not mounted there).

**The tool-footprint question.** Exposing this needs a tool for the model. Options: (i) a small
`ask_withdraw` tool — **recommended**, implemented as a `createIf`-gated factory (footprint ladder
rung 3: zero schema wherever the queued engine is absent) with its schema delta measured via
`/context`; (ii) a mode on `ask` — rejected, `AskParams` is
question-shaped and the op targets an EXISTING ask; (iii) auto-only, no tool — rejected, moot
detection and chat attribution both need the model. Size (i) against the AGENTS.md tool-surface
footprint ladder before writing its description; the ask receipt's prose is the budget precedent.

**Detection (the hook).** Detection is the model's judgment; the tool description carries the rule —
"if the user's latest message answers a queued ask, withdraw it with reason=answered_in_chat and
their words". Optional runtime assist (PROPOSAL): when a chat message arrives while the session has
open asks, the model-visible receipt appends one line naming the open ids ("open: a-3f9c — if this
message answers it, withdraw it"). Track with the §9 text edits.

**Surfaces (all read the same fold — passive render only):**

* TUI: `ask_queue.py`'s status map and settled chip gain `withdrawn` (the bare word — the
  `declined`/`dismissed` register; the chip has no room for a clause); `SETTLED_STATUSES` grows by
  `withdrawn`; the halves partition unchanged (withdrawn is Settled, never outstanding; the
  bar/list counts drop by themselves).
* Wire/web: the status enum addition (`mobile/types.py`, web `types.ts`, `asks.ts` copy — the
  settled row's clause register names the actor: "Withdrawn — the agent no longer needs an
  answer"); an answered-in-chat row already renders through `answered_by.surface`.
* Desktop: their lane; the settled half needs the word, no op.

### 12.2 Open questions (FOR SYNC — Aida)

1. **Whose half is "withdraw"? RESOLVED (Aida, 2026-10-05)** — the operator's own words settle it:
   both reasons above are AGENT-side. **The operator-side settle is a real but DIFFERENT gap**
   (`dismiss` being `timed_out`-only means a user cannot clear an OPEN ask — a UX gap, not the
   engine gap) and is **deferred out of this slice**: it rides the PR thread as `deferred — dismiss
   remains timed_out-only; widening it is a policy change with its own copy`, so it is not lost.
2. **`/new`, session delete, or stop with open asks**: today they persist (durable by design,
   answerable from any surface). Proposal: no auto-withdraw; the operator can still answer or
   dismiss. Confirm.
3. **Auto-supersede** (a new `ask` withdrawing the same session's prior still-open ask): NOT
   proposed — a re-ask is a legitimate second question; noise control is the model's job via the
   tool. Confirm.
4. **Copy voice** for the agent-authored by-field: `by: {surface: "agent"}` for uniformity vs
   `{actor: "agent"}`. Implementation detail; noting it.

### 12.3 Evidence plan (at code time)

Unit: fold orderings incl. the answered-vs-withdrawn race and unknown-kind tolerance; refusal copy
per state; the verbatim-cells and secret-refusal rules. E2E (real runtime): queued ask → chat reply →
tool withdraw(answered_in_chat) → folded `answered` + response row + index settles; moot withdraw →
settled word only, no row, counts drop; wire value passthrough. Frames: TUI settled chip + bar count
before/after; phone card (relay lane). PR: one core PR (this §12 + code), `Release: patch` — a
self-contained feature, not a step-function.

**The PR description states two things as CONTRACT** (Aida, sync): (a) the race rule — every
user-act terminal row present (`answered`, `declined`, `dismissed`) outranks a later `withdrawn`,
and only a `withdrawn` with no such sibling folds to `withdrawn`; and (b) old builds folding
`withdrawn` rows tolerate the unknown kind. Both are cheap to state now and expensive to discover
later.
