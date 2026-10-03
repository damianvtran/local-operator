# Design: non-blocking, queued, timeout-bounded `ask`

Status: PROPOSED — gate for the implementation PRs. Base: `origin/main` @ `302a061e5`
(v0.64.10). Author: architect, 2026-09-30.

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
turned into the answer box (`chat-page.tsx:1624-1698` **(UI)**); and a resumed session can
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
idempotence is structural and a `late` ask legitimately needs **two** rows.

| status | expected transcript row(s) | injects a turn? |
|---|---|---|
| `answered` | `ask-response-<ask_id>` | yes (one) |
| `late` | `ask-timeout-<ask_id>` **and then** `ask-response-<ask_id>` (the timeout fired first) | yes (two) |
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
Idempotent by construction. **Wire `delivered`** (contract §4) means "the row(s) *this*
status requires are all present": for `late` that is both rows; for `open` it is `false`.
It is **sticky** — once any response/timeout row exists for the ask it stays `true` for the
life of the record, so an answered ask cannot flip back to undelivered when it folds to
`expired` seven days later; `dismissed`/`expired` are `false` only when no delivered row
exists at all.

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
  **When a late answer's two rows land in the SAME batch** (a cold boot after the answer
  arrived past the deadline), the `ask-response-` row is emitted **first and the
  `ask-timeout-` row is suppressed for that ask** — replaying "[Ask timed out] … you will be
  told" immediately before the answer it announces reads to the model as a contradiction. The
  timeout row is only written when no response row for that ask is being delivered in the same
  batch (§7 asserts both orders).
- **Dedupe.** deterministic ids + `has_entry` in `_drain_steering` (`:14031` already skips a
  durable id). Two runtimes cannot both deliver: only the lease holder writes the transcript.
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
  delivered: bool,   // terminal: the rows THIS status requires exist (§2.2 table) — `late`
                     // requires BOTH the timeout and the response row; STICKY (never flips
                     // back to false); `open` is always false; `dismissed`/`expired` are
                     // false only when no delivered row exists
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
  old-client wart (accepted): an old desktop composer still routes typed text as the
  answer while the mirror is up. **Client rule (N3): once `asks` is present, a client IGNORES
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

**INVARIANT — flag-off behaviour is exactly today's, continuously through A1–D.** Until PR F
flips `asks.policy.NONBLOCKING_ASK` (default `False`; env `LOP_ASK_NONBLOCKING=1` for
QA/evidence), every surface's default behaviour is unchanged: the blocking card, the single
`_ask_screen`/`_ask_pending` slot, the single `pending_gate`, the view bridge's single
`_gate_task`, the desktop composer swallow and typed ordinals, and the fold's single
`pending`. **Every new path introduced in B/C1/C2 exists only with the flag on** (client-side
that is the presence of the `asks` wire field — true because A2 publishes it only when the
flag is on, §4). No PR in B/C1/C2 may delete or repurpose
an old path: the old paths are removed **once, in F**, in the same change that rewrites the
pinning tests and ships §9. This is what makes §8.9's "the pinned tests stay green through
A1–D" true, and a B/C1/C2 diff that removes an old path is a bug against this note.

### 5.0 Shared ask-surface interaction model (R7)

Every surface presents the open ask set in one of **two states**. Both are **client-local
interaction state: no wire change, §4 is untouched by this section.**

- **EXPANDED** — the answer surface is active (TUI picker/list row, desktop `QuestionDock` or
  sheet, phone/app sheet). **Entered only by the user** (click the minimized bar, Enter on a
  focused row, or the explicit `/asks` / header action). **Never automatic on ask arrival** —
  §5.1's no-auto-mount, no-focus-steal rule is unchanged. Left by Esc, the collapse control,
  or a click on the bar/chevron.
- **MINIMIZED** — a compact, persistent single-line affordance; the answer surface is *not*
  mounted. This is the state a new ask lands in when the user is already mid-answer or
  mid-draft, and the state the user returns to when they collapse.

**Composer routing rule (INVARIANT, every surface).** The composer routes to the ask **only
while its answer surface is EXPANDED**; while MINIMIZED the composer sends an ordinary
conversation message. Toggling preserves **both** drafts: the ask buffer and the chat buffer
are separate, and neither may ever be sent into the other's channel — a chat draft must never
become an answer, and an answer draft must never be sent as chat. When the ask **settles while
expanded** (answered, timed out, declined, dismissed), the placeholder returns to normal and
the draft is **kept**, neither discarded nor auto-sent.

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

**Placeholder copy** (one short line per surface; the strings land in that surface's PR):
expanded → "Answering the agent's question — Esc to collapse"; minimized/normal → each app's
existing placeholder, unchanged.

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
  the existing `tui/notify.py` map (scout `177/237`). **No auto-mount, no focus steal** — a
  new ask must not displace a card the user is mid-answer on or their composer draft
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
  presentation of a queued ask; the composer sends chat while minimized and the answer only
  while the picker is expanded. Sidebar/`session_catalog` gains the outstanding-asks state,
  distinct from `pending`/approval.
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
  **minimized bar (1 and 3 asks), expanded, and the two placeholder variants**.
  BEFORE = `ask_shot.py`/`ask_scroll_shot.py`/`ask_long_shot.py`/`approval_shot.py` on
  `origin/main`. Geometry numbers alongside (AGENTS.md "Visual validation" step 4): prompt-host
  height, composer focus, no reflow between first and settled frame. `ask-long-descriptions`
  invariants (approval frames byte-identical) must still hold.
- **Design + UX rounds:** both (new list entry point, Esc semantics, no-steal policy,
  **and the §5.0 minimized bar + composer routing, which the UX round must walk with a real
  typed draft**).

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
  routes to the ask only while the dock/sheet is expanded, otherwise to the conversation —
  this **replaces** the current unconditional composer swallow (which stays for the flag-off
  path only). Chat-list status gains the outstanding-asks state, distinct from approval.
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
  dock **and** composer; **minimized bar (1 and 3 asks), expanded, both placeholder variants,
  and the chat-list outstanding-asks row (light+dark)**; live-app composer frames via
  `renderer-driver` (state window mode);
  `pnpm check-themes`, contrast rows; `docs/evidence/manifest.json` re-stamp in its own
  docs-only commit after each fold. **Conflict watch:** #615 (dock mount, answer path,
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
  the sheet. The web composer routes to the ask only while the sheet is open — the
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
   minimized bar, the expanded sheet, both placeholders and the list-row outstanding state**;
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
  pattern and the composer routing rule; E2's docs pass carries a **one-paragraph pointer to
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
  and UX rounds** — the minimized bar, the composer routing rule and the list/sidebar
  outstanding-asks state are user-visible, so those rounds cover them (§5.0, §7). E1 (the
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

**R7 — the minimized surface (every user-visible PR: B, C1/C2, D, E2).** A dedicated case,
because the routing rule is a *behavioural* claim that stills cannot show:
- **Asserts (automated where the surface allows, walked by hand in the UX round otherwise):**
  (1) an ask arriving while a chat draft is in the composer leaves the draft intact and the
  ask MINIMIZED; (2) with the surface minimized, Enter on the composer sends a **conversation**
  message, never an answer; (3) with the surface expanded, Enter sends the **answer**, never
  chat; (4) collapsing preserves both drafts and re-expanding restores the ask draft;
  (5) an ask settling while expanded returns the placeholder to normal and keeps the draft.
  (6) the minimized bar is absent at zero asks and shows head + count at 3.
- **Frames:** minimized bar (1 and 3 asks), expanded, both placeholder variants, and the
  list/sidebar outstanding-asks state — **light and dark** where the surface has both, at
  phone size for web/app. Before/after against `origin/main`.
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
