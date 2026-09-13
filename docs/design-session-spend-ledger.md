# Session spend ledger: one exact, recalled per-session cost

Branch: `feat/session-spend-ledger`, cut from `origin/main` = `7fe8b1005`
(released v0.54.33). This document is a DESIGN, not an implementation — no
product code is proposed here, only the seams it must use and the measurement
that says which of them is right.

Measured on this machine, 2026-09-13, macOS arm64 / Python 3.14.3, against the
operator's real store (`~/.local-operator/sessions`, ~2,000 session directories,
2.37 GB of transcripts; `~/.local-operator/analytics.db`, 853,757 call rows).
The store is LIVE, so a head-count in one table can differ from another by under
1% depending on when it was taken; the ratios and the byte totals do not. Every
number below is reproducible with the commands quoted beside it.

## 1. Summary

Of the 2,004 transcripts in the real store, **1,904 paint a cost at all — and
1,785 of those (93.8%) paint it as a lower bound, `≥`** — for reasons that have
nothing to do with the money being unknown. The mark is set because the restore
path can price exactly ONE provider reading, not because the session's spend is
genuinely incomplete. Measured across the whole store, that one reading is a
median **64x** below the session's own turn rows, and up to **9,854x** below.

The failure has a second face that is worse than a mark: **81 sessions paint no
cost segment at all**, because the newest reading is a model the paint-grade
resolver cannot price (6.86% of all rows, §2.4) — real spend rendered as
silence, which reads as free.

Three things are true at once, and the fix is to reconcile them:

1. The session **already owns a correct in-memory accumulator** —
   `FrontendSessionState.cumulative_parent_cost`, accrued per provider call
   (`frontend_state.py:3884`) and reconciled at turn end (`:3835`) — and it is
   durable only as a **fat, conditional, TUI-only checkpoint**
   (`frontend_state.py:3991`, gated at `session.py:7300`).
2. The transcript's turn rows **already carry the serving identity and the token
   buckets**, and 30% of them already carry a provider receipt — but the
   harness's own "durable money" field (`estimated_usd_cost`) is written **zero
   times in 258,436 rows**.
3. The restore path therefore **re-prices one row at paint-grade resolution**,
   which on this store cannot price 6.86% of all rows at all — including 165
   calls worth **$58.51** in one session, i.e. 86% of its true cost.

**Recommendation.** Make the accumulator durable as a small, versioned,
replay-invisible transcript custom entry (`session_spend.v1`), written at every
provider-call boundary through the existing `Transcript.append_custom`, carrying
the **recorded** micro-USD and an explicit knowledge state; recall it in O(1) at
construction (owner) and from the journal suffix the cold viewer already reads;
and run a **one-time, off-loop, bounded rebuild** for the 92.3% of sessions that
predate it, applying **no compaction boundary** (money already spent is not
invalidated by a later rewrite of the context — the rule `search_spend_rows`
already states and `usage_seed` violates).

Why the transcript and not a new sidecar or the analytics ledger: §5. Short
version — the transcript is the one store that travels with every portability
path the repo already has (fork, archive, copy, move), it is the durability
primitive the messages themselves use, and it keeps the money and the
conversation from ever diverging into two stores that must agree.

The smallest change that solves the problem is NOT "add a ledger". It is:
**persist the accumulator that already exists, recall it, and stop inferring the
mark.** Anything larger — a new store, a second pricing path, a per-session
materialised rollup — buys nothing this does not.

## 2. The problem, as measured

### 2.1 The restored figure is a point-in-time reading, not a total

`Session.restored_usage()` returns `self._last_usage` (`session.py:5932-5953`),
seeded at construction by `seed_reported_usage(transcript.usages_since_compaction())`
(`session.py:2115`), which parses every surviving usage row and takes the
**newest** one (`usage_seed.py:96-99`, a reverse scan that stops at the first
hit).

`_restore_reported_usage` (`app.py:9311`) prices that single reading
(`app.py:9367`) and then sets `self._spend_is_floor = True` unconditionally
(`app.py:9377`), which `_spend_text` (`app.py:37558`) paints as
`RESTORED_COST_PREFIX = "≥"` (`app.py:388`). The flag is documented as sticky for
the life of the conversation (`app.py:3489-3496`), and the other four writers of
the cell inherit it through `_spend_text`.

Measured over the 25 most recently touched transcripts (probe:
`read_replay_suffix` + `usages_since_newest_shrink` + `turn_cost`, oldest-first
summed):

| session | surviving rows | newest row alone | sum of the same rows | ratio |
|---|---|---|---|---|
| `e74f2707ba9d` | 174 | $0.0017 | $0.2966 | 175x |
| `c1a63e4bab85` | 129 | $0.0031 | $0.3408 | 110x |
| `ea5e0c3ee396` | 106 | $0.0027 | $0.2231 | 83x |
| `2a43eb7e6aa6` | 107 | $0.0101 | $0.2137 | 21x |
| `510a94a90cb3` | 40 | $0.0011 | $0.0573 | 55x |

Store-wide (1,998 transcripts, 1,919 with at least one priced row): **median
64x, mean 101x, max 9,854x**. The worst five by ratio: `28a800c6a783`
($36.99 summed over surviving rows, 245 rows), `d4555b021db6` ($9.43 / 106),
`b59ecf28a70b` ($4.39 / 146), `f014c50efcb7` ($4.67 / 222), `4140ee201ce1`
($4.53 / 140).

### 2.2 The compaction boundary is NOT the cause — and it changes the answer where it exists

Of the 25 most recently touched transcripts, 2 carry a compaction marker and 6
carry prune markers; `usages_since_newest_shrink` (`transcript.py:1519`) returns
every row in the other 19. So the boundary filter is a no-op for most sessions,
and the single-row read is the whole loss.

But where the boundary *does* exist, it is not a no-op, and for **money** it is
wrong: `28a800c6a783` has 9 compaction markers and 18 prunes, 2,134 usage rows
in the file, and 262 that survive the boundary. Summing every row in the file
prices at **$324.99**; summing only the survivors prices at **$36.99** — an 8.8x
difference that the boundary silently deletes.

`Transcript.search_spend_rows` already states the rule this violates, in the
repo's own words (`transcript.py:1131-1140`): *"money already spent is not
invalidated by a later rewrite of the context… Copying the boundary here would
silently drop a resumed conversation's earlier search spend and make the
restored total a floor for no reason."* The model-money restore does exactly
what that comment forbids. **The asymmetry between the two money restores is the
bug in one line.**

### 2.3 The accumulator exists, is correct, and is almost never persisted

`FrontendSessionState.cumulative_parent_cost` (`frontend_state.py:1668`) is
accrued **per provider call** on `MessageEndEvent`
(`frontend_state.py:3868-3894`: `cumulative_parent_cost = previous + call_cost`,
`current_turn_accrued_cost += call_cost`) and reconciled once at turn end
(`frontend_state.py:3826-3850`: `remainder = max(0, total - current_turn_accrued_cost)`,
so the two writers cannot double-bill a turn). It is the number the band
actually paints (`app.py:8673`).

It becomes durable only through `FrontendStateStore.checkpoint()`
(`frontend_state.py:3991`), which writes `{checkpoint_id, state: <the WHOLE
FrontendSessionState>}` as a `custom` entry, and only when the session has a UI
or a subscriber (`session.py:7300`). Measured over the real store:

| | count | share |
|---|---|---|
| transcripts | 2,004 | — |
| **with a `frontend_state_checkpoint_v1` row** | **155** | **7.7%** |
| with no checkpoint at all | 1,849 | 92.3% |
| checkpoint rows in total | 938 | — |
| `cost_knowledge` across those rows | exact 283 · partial 460 · floor 191 · unknown 4 | — |

So 92.3% of sessions have no durable total, and even the 7.7% that do are
carried by a writer whose cost is measured below.

**What the fat checkpoint costs, measured:** for the 155 sessions that have any
checkpoint rows, those rows are **359.3 MB of 1,023.9 MB of transcript bytes
(35.1%)** — and for `bda7b76d34e0`, **176.6 MB of 255.1 MB (69%)**. The row
rewrites the whole state on every turn end (`frontend_state.py:4001-4004`
documents the same class of bloat for trajectories). This is why "just
checkpoint more often" is not an option, and why the durable carrier must be
small.

(For completeness, and because it explains the 7.7%: `cumulative_parent_cost`
*is* restored correctly when a checkpoint is present —
`FrontendStateStore._restored_state` reads it at `frontend_state.py:3336-3345`,
so the band shows an unmarked total for those sessions. The defect is the
population without one.)

### 2.4 The transcript rows are priceable in principle and mispriced in practice

Two fields exist on `Usage` for exactly this (`harness/types.py:406-410`):
`usd_cost` (a provider receipt) and `estimated_usd_cost`, whose own comment says
it is *"deliberately durable money so offline resumes need not reprice"*.

Measured over all 258,436 usage rows in the store:

| field | rows carrying it | note |
|---|---|---|
| `usd_cost` (receipt) | 77,651 (**30.1%**) | 77,650 of them from `openrouter` (which precomputes billing); 1 from `radient` |
| `estimated_usd_cost` | **0** | never written on the parent-turn path |
| unpriceable by the paint path NOW | 17,718 (6.86%) | `turn_cost` → `None` |

Store-wide, the newest surviving reading is unpriceable in **81 of 2,004
sessions** — those bands paint no cost segment at all, so a session with real
money on it reads as free rather than as a floor. That is the same defect as the
`≥`, one state further along.

`estimated_usd_cost` is stamped only on the subagent path
(`harness/subagent.py:1353-1354`) and in the jobs fold (`harness/jobs.py:245-254`);
the parent turn's usage is never stamped. So a transcript row's money is either a
provider receipt (30% of rows) or must be reconstructed later.

**And reconstruction later is not merely slow, it is wrong.** `turn_cost`
(`tui/costs.py:85`) resolves prices through `resolve_model_info_paint`
(`tui/costs.py:55-82`) — memo-or-registry only, by design, because the full
resolver's discovery legs are synchronous HTTP (measured 418 ms warm-disk, up to
13 s worst case, per that function's own docstring). The ledger prices at record
time on a background thread through the FULL `resolve_model_info`
(`analytics/model.py:239-290`). Measured on this machine, right now:

```
openai/gpt-6-astra      memo_hit False  in 0.0  out 0.0     <- unpriceable at paint
anthropic/claude-opus-5 memo_hit False  in 5.0  out 25.0
```

Session `560f212a892c` has **166 `openai/gpt-6-astra` calls the ledger priced at
$58.51** and **1,002 `openrouter/deepseek-v4.1-flash` calls priced at $8.86**.
The transcript rows for the OpenRouter calls carry a receipt; the Astra rows
carry tokens only and are unpriceable now. Reconstructing from the transcript
yields **$9.39** against the ledger's **$67.89** — 86% of the session's cost is
invisible to any transcript-based restore, and no amount of repricing the
transcript recovers it, because the price the call was billed at is only known
at record time.

**This is the argument for persisting the number rather than deriving it.**

### 2.5 The analytics ledger is the better number for long sessions, and cannot be the store of record

`analytics.db:calls` (`analytics/store.py:130`) already holds `session_id`,
`provider`, `model_id`, `cost_micro`, `cost_known`, `calls`, `purpose`,
`request_id`, `parent_session_id`, `outcome` — 853,757 rows over 4,716 session
ids, `cost_known=1` on 836,373 of them (98%).

Per-session sums agree with the transcript's own row sums within ~20% for
ordinary sessions (e.g. `e74f2707ba9d` $0.3078 vs $0.3111; `52741171ec63`
$0.1896 vs $0.1611; `084cd8fd206b` $0.4470 vs $0.3353 — the ledger sees naming
calls, asides and failed attempts the turn rows do not). For long-lived sessions
it is decisively better: `560f212a892c` $67.89 vs $9.39; `28a800c6a783` $418.30
vs $324.99 over all rows and $36.99 over survivors; `bda7b76d34e0` $3,169.55 vs
$1,925.81 in its checkpoint.

It cannot be the authoritative store, for four reasons the code states itself:

1. **Retention deletes it.** `DEFAULT_RETENTION_DAYS = 90`
   (`analytics/store.py:60`), enforced by `prune` (`analytics/store.py:1163`)
   which `DELETE`s by `ts_ms`. A session's total would go **DOWN** over time.
   The repo already calls that shape unacceptable for a spend counter: *"a spend
   counter that falls when a finished child is evicted is worse than none"*
   (`tui/costs.py:20-23`).
2. **It is best-effort by contract.** `AGENTS.md` §"Usage analytics":
   *"Recording is off the critical path and best-effort… Nothing in analytics may
   raise into a turn; every path is guarded."* The recorder drops on a full queue
   (`recorder.py:259-276`, `_QUEUE_MAXSIZE = 4096`) and drops a batch after
   bounded busy retries (`store.py:938-948`). Correctness may not hinge on a
   subsystem that is allowed to lose data.
3. **It is not portable.** It is one global file keyed by an opaque id. A copied
   session directory, an archived session, or a session moved between machines
   arrives with no money. (`~/.local-operator/session-archive` and
   `move-recents.json` both exist on this machine.)
4. **Its scope is ambiguous.** 4,716 ledger session ids against ~2,004 session
   directories; purposes `turn` 433,949 / `unknown` 419,380 / `naming` 422 /
   `aside` 5 / `compaction` 1. "This session's cost" needs a predicate that does
   not exist yet, and every consumer would have to agree on it.

Where it *is* the right tool: as the **reconciliation oracle** (§7.3) and as a
QA cross-check, never as the runtime's recall source.

### 2.6 Child and search spend already have durable restore paths

- **Children**: `subagent-roster.v1.json` is written on every roster move
  (`session.py:11438`, triggered at `:11435`/`:11517`), is **not** gated on
  `_has_ui`, and carries `accounting` — the whole manager ledger
  (`session.py:11460-11462`). The cold viewer restores it
  (`attached.py:1541-1587`), the checkpoint also carries `subagent_cost`
  (`frontend_state.py:1673-1674`).
- **Web search**: every `web_search`/`web_read` tool row keeps its
  `search_cost`/`read_cost` details through pruning, and
  `Transcript.search_spend_rows` (`transcript.py:1128`) restores them
  deliberately unbounded by compaction; `_restore_search_spend` (`app.py:9404`)
  seeds them into `SEARCH_SPEND`, idempotently (`app.py:9443-9446`).

So the broken arm is exactly **(a) the session's own model-turn spend**. That is
what this design fixes, and scoping it that narrowly is the point: the other two
arms already answer "what did this cost" durably, and folding them into a new
store would create a second home for decisions that already have one.

## 3. What the code does today — the four accounting stores

| store | where | durability | scope | restore |
|---|---|---|---|---|
| in-memory accumulator | `FrontendSessionState.cumulative_parent_cost` (`frontend_state.py:1668`) | process lifetime | parent turns + store-observed errands | `_restored_state` from the checkpoint, else one receipt priced as FLOOR |
| fat checkpoint row | `frontend_state_checkpoint_v1` (`frontend_state.py:69`) | transcript, turn-end, **UI/subscriber-gated** | whole frontend state, incl. accumulator | `frontend_state.py:3336`, `attached.py:1370-1447` |
| transcript usage rows | one per assistant message | transcript, per message append | one call each (tokens + identity; receipt on 30%) | `usages_since_compaction` → newest only (`session.py:2115`) |
| analytics ledger | `analytics.db:calls` (`analytics/store.py:130`) | separate DB, WAL, **90-day retention**, best-effort | one call each, priced at record time | `session_report` (`store.py:1424`) — `/session` only |

The TUI has two parallel accounting paths and they must not be conflated:

- **canonical** — a Session owns a `FrontendStateStore`; the app renders the
  published snapshot (`app.py:8664-8680`). `_canonical_frontend(session)` gates
  the legacy path off (`app.py:38016-38020`).
- **legacy/reduced hosts** — the app owns `_total_cost` + `_turn_accrued_cost`
  (`app.py:3481`, `:3751`), accrued per call at `app.py:38022` and reconciled at
  turn end at `app.py:36669-36690`.

Both feed the same `_spend_text`, and both call the same `_cost_for`
(`app.py:37452`) → `turn_cost`. A design that only fixes the canonical path
leaves the legacy path's `≥` in place, so both are in scope.

## 4. Decision 1 — where the authoritative accumulated number lives

Options scored on the six criteria the operator named. Measurements are from §2.

| | (a) per-session sidecar file | (b) `analytics.db:calls` by `session_id` | (c) transcript custom entry | (d) in-memory only |
|---|---|---|---|---|
| **portability** (fork / archive / copied dir / moved machine) | session dir → travels; **needs a `COPIED_SIDECARS` entry** (`fork.py:116`) or forks lose it | **fails**: global DB, opaque ids, nothing travels; a fork starts at $0 | **travels free** — the transcript is the one mandatory fork copy (`fork.py:116`) and every archive/copy/move path already carries it | none |
| **crash consistency** | own append+rename; correct, but a second writer path to maintain | WAL + one writer thread, **plus 90-day `prune`** (`store.py:1163`) that deletes money | the messages' own primitive: `_commit` shield + append + `fsync` (`transcript.py:830-861`, `:992-996`) | none |
| **depends on an optional/best-effort recorder** | no | **yes** — drops on a full queue (`recorder.py:269-276`) | no | n/a |
| **read cost on the recall path** | one small file open | one indexed `SUM` + a fresh read connection; needs caching to stay off the paint path | **≈ 0 marginal**: the owner path already has every row in `_entries`, so it is a dict lookup (`transcript.py:1104-1115`); the cold path already does the suffix read | 0, but wrong |
| **process dies between the call and the persist** | one call, if written per call | better than turn-end (the recorder enqueues right after the stream drains, `configure.py:5368-5392`) but behind an async queue | one call, if written per call | the whole session |
| **write cost** | ~0.04 ms + rename | 0 new (already written) but shared with a 853k-row DB | **0.057 ms median append+fsync on a 267 MB file** (measured, §6.1) | 0 |
| **footprint** | ~1 small file/session | 0 new | ~200 B per call, reclaimed by the fold (`transcript.py:1330-1338`) | 0 |

### Recommendation: (c), the transcript custom entry

1. **It is the only option where the money cannot be lost by a path that keeps
   the conversation.** Every portability requirement the operator listed is a
   directory operation the transcript already survives; a sidecar needs a
   fork allow-list entry (`fork.py:116`, pinned by a set-equality test — an
   addition there is a deliberate, reviewable act) and a second reader on the
   cold path.
2. **One store, one boundary, one lock.** The repo's most expensive recurring
   defect class is two surfaces answering the same question differently
   (`AGENTS.md` §"Usage analytics": `_PARENT_EDGE_SQL` — *"Two surfaces answering
   the same question differently is the defect the rollup exists to remove"*).
   Putting the money in a second file creates exactly that, for a number that
   must equal the sum of the rows in the other file.
3. **The read cost argument for a sidecar does not survive inspection.** The
   owner path parses the whole journal at construction anyway
   (`transcript.py:643-652`) and `latest_custom` is a dict lookup on the index
   that parse already built; the cold path reads a bounded suffix
   (`read_replay_suffix`, `transcript.py:441`) and can be asked for one more
   custom type in the same pass.
4. **The blow-up risk is bounded and already solved for this exact shape.**
   `_COLLAPSIBLE_CUSTOM_TYPES` (`transcript.py:190`) keeps the newest entry of a
   type and drops the rest at fold time (`transcript.py:1330-1338`). An
   accumulator row is precisely "newest wins", which is why the record carries a
   **running total** rather than a per-call delta list.

Two consequences to accept explicitly:

- The new type must join `_COLLAPSIBLE_CUSTOM_TYPES`. That is safe only because
  nothing reads it by iteration (`transcript.py:185-189` states the bar) — the
  record is read exclusively through `latest_custom`.
- The cold path's suffix reader takes `checkpoint_type: str | None`
  (`transcript.py:445`). It must be able to serve two types. **Widen it to
  `checkpoint_types: tuple[str, ...] | None`, keeping a bare `str` accepted** so
  the change is additive at every existing call site (`attached.py:3763`,
  `attached.py:3763`'s `want_checkpoint`). Do not add a second parameter: the
  stop condition `checkpoint_seen` (`transcript.py:567`) is a single predicate
  over "the newest row of each requested type is in the buffer", and two
  parameters would mean two stop conditions that could disagree.

**Rejected: (a) as the primary.** Kept as the fallback if the coder finds the
transcript append path's `_lock` contention unacceptable in practice (§11,
"Watch during rollout"). Cost of that fallback, to state it once: one
`COPIED_SIDECARS` entry, one reader in `attached.py`'s cold path, one reader in
`desktop_sessions.py`, and a permanent second store for one number.

**Rejected: (b) as the store of record.** Keep it as the reconciliation oracle
(§7.3): it sees calls no turn row does, and it is the only place a dropped
persist can be detected from.

**Rejected: (d).** `cumulative_parent_cost` minus its checkpoint is exactly
today's 92.3% bug.

## 5. Decision 2 — how the number stays exact

### 5.1 One write site, one arithmetic site

The accumulator is currently computed inside
`FrontendStateStore.observe_event`'s cost branches (`frontend_state.py:3868-3894`
per call, `:3826-3850` turn-end remainder) and `accrue_usage`
(`frontend_state.py:3576`, used by detached leaf calls via
`serving.py:2584`). Making persistence a *second* accrual site is how a turn gets
billed twice — the exact hazard `current_turn_accrued_cost` exists to prevent
(`frontend_state.py:3874-3875`).

**Design: a session-owned accumulator, with the store as its mirror.**

```
session/spend.py
  class SessionSpend:            # the one arithmetic + the one knowledge derivation
      micro: int                 # sum of RECORDED micro-USD — integer, exact
      calls: int                 # provider calls accounted for
      priced_calls: int
      unpriced_calls: int
      floor: bool                # a rebuild over a shrunk journal: rows were dropped
      accrue(micro: int | None, identity) -> None
      knowledge() -> CostKnowledge
      to_details() / from_details(details)
```

- `FrontendStateStore.observe_event` / `accrue_usage` keep their **timing** and
  their turn-remainder reconciliation, but call `session.spend.accrue(...)`
  instead of doing their own addition, and then publish
  `cumulative_parent_cost = session.spend.micro` (converted from micro-USD to
  dollars once, at the wire boundary — the wire field stays a float for
  compatibility).
- `FrontendStateStore.checkpoint` (`frontend_state.py:3991`) writes the
  accumulator's value, so the fat checkpoint and the slim record can never
  disagree.

This is the smallest version of "one writer": the arithmetic moves, the event
timing does not.

### 5.2 The pricing contract: price once, off the loop, at the call boundary

The money for one provider call comes from, in order:

1. **`usage.usd_cost`** — the provider's own receipt. Returned verbatim
   (`cost_for_usage`, `configure.py:5565-5587`; `_usage_cost`,
   `configure.py:5607`). Already present on 30.1% of rows.
2. **A table estimate at record time**, computed with the FULL
   `resolve_model_info` + `cost_for_usage` — the pair `price_snapshot` already
   uses on the analytics writer thread (`analytics/model.py:239-290`).

Requirement 2 is the load-bearing change: the paint-path resolver
(`resolve_model_info_paint`, `tui/costs.py:55-82`) must NOT be what the durable
number is built from. §2.4 measures why — 6.86% of all rows and 86% of one
session's cost are invisible to it.

**Where it runs.** Off the event loop, on a worker, at the call boundary — the
same discipline `_record_stream` already follows (*"Cost is NOT priced here…
the pricing… runs on the recorder's background thread… and never on the event
loop this turn is unwinding on"*, `configure.py:5368-5375`). Concretely, the
session's spend writer:

1. appends the per-call delta and the new running total through
   `Transcript.append_custom("session_spend.v1", details, preserve_mtime=True)`
   from `asyncio.to_thread` (the append is `_commit`-based and already runs its
   filesystem work on a worker — `transcript.py:837`);
2. reports the authoritative micro-USD back to the loop, which applies it to
   `session.spend` and republishes.

**Until it lands (one tick)**, the band shows what it shows today: the
paint-grade estimate if the warm memo can price the call, nothing otherwise
(`_resolve_for_paint` already fires the background refresh and returns `None` for
one tick, and `costs.py:97-105` documents that as the honest degradation). When
the authoritative value lands, the accumulator is **corrected by the delta**, so
the painted number and the persisted number converge instead of diverging.

What this buys: one price per call, computed where the price is knowable, used
for both the screen and the record. What it costs: the band's per-call movement
is optimistic for one tick. That trade is already the repo's stance
(`app.py:11898-11901`: *"Money is the segment where a user is least able to tell
a carried-over number from a real one, in EITHER direction"* — and the
optimistic figure is the conservative one, since the estimate is the paint
resolver's own answer).

**Rejected alternative: stamp `estimated_usd_cost` on the message's usage at
`harness/loop.py:1808`** and let the record read the row. It is the smallest
possible change and it would fix the rebuild — but the loop is on the event loop
and cannot call the blocking resolver, so the stamp would carry the paint-grade
estimate, i.e. exactly the number §2.4 shows to be wrong. The field is still
worth writing (it makes every future rebuild cheap and exact for the rows that
have it) and is listed as an optional slice (§9, T7) — but it must not be the
record's source of truth.

### 5.3 The record's shape

```json
{"custom_type": "session_spend.v1",
 "details": {"version": 1,
             "micro": 1897843,        // integer micro-USD, exact SUM semantics
             "calls": 42, "priced_calls": 41, "unpriced_calls": 1,
             "floor": false,          // true only for a rebuild over a shrunk journal
             "rebuilt": false,        // true when this row came from the one-time rebuild
             "writer": "<pid>:<boot-id>",
             "last_identity": {"provider": "openai", "model_id": "gpt-6-astra"}}}
```

`micro` as an INTEGER (USD × 1e6) is the ledger's own convention
(`analytics/model.py:186-194`: *"Integer so the aggregate SUM is exact"*) and the
reason is the same here.

It is **replacement state, not an additive delta** — the rule the checkpoint
already states for the identical reason (`session.py:7291-7295`: *"The checkpoint
is replacement state, never an additive delta, so takeover cannot double it"*).
A reader takes the newest row; it never sums the file.

### 5.4 The knowledge states, and which real situation each may represent

The operator's requirement is that the mark mean something. Four states, one
derivation, one shared predicate:

| state | the real situation | band | panels | may it be a *default*? |
|---|---|---|---|---|
| `EXACT` | every provider call in this session's history is accounted for and priced | `$1.90` | `$1.8978` | yes — this is the normal state |
| `PARTIAL` | ≥1 call ran on a model with no published price AND no provider receipt, so the known money is a true lower bound | `≥$1.90` | `$1.90+` | no — it must be earned per unpriced call, with `unpriced_calls > 0` |
| `FLOOR` | the total is missing rows that no longer exist (a rebuild over a journal that carried a compaction/prune marker: rows were dropped from the file), so the sum cannot be complete even in principle | `≥$1.90` | `$1.90+` | no — only a rebuild over a shrunk journal |
| `UNKNOWN` | not one call was priceable (a local-model-only run) | `$—` | `$—`, **no** `≥` | no |

Derivation, once, in `session/spend.py`:

```python
def knowledge(...):
    if priced_calls == 0:  return UNKNOWN   # nothing to bound — "$—", never "≥$—"
    if unpriced_calls:     return PARTIAL
    if floor:              return FLOOR
    return EXACT
```

`UNKNOWN` taking precedence over `PARTIAL` is the analytics panel's existing
rule and must not be re-derived: *"A FULLY-unpriced bucket has no dollar figure
to bound: it renders a clean `$—` with NO `≥`, because '≥ unknown' is a
contradiction"* (`analytics_panel.py:683-689`).

The equivalent predicate already exists for the ledger —
`UsageAggregate.cost_is_known` / `cost_is_partial` /
`cost_is_floor` (`analytics/model.py:1008-1032`) — so the design is to express
the states once and have both ask it, not to invent a second vocabulary.

**What `FLOOR` stops meaning.** Today `floor` means "we restored a point-in-time
reading" — i.e. it is the default for 92.3% of sessions for a reason that has
nothing to do with the money. After this change it means one thing only: *rows
that carried money were dropped from the transcript file by a compaction or a
prune, so the sum cannot be whole*. Measured on the real store, a rebuild would
classify the store as: **exact 1,367 · floor (shrunk) 475 · partial (unpriced)
151 · empty 8** — i.e. the `≥` would survive on 626 of ~2,000 sessions where it is
genuinely earned, and vanish from the ~1,367 where it was noise.

## 6. Decision 3 — the recall path

### 6.1 Recall on resume / attach / `/reload` / paint

| surface | today | after |
|---|---|---|
| owner, at construction | seed `_last_usage` from the newest row (`session.py:2115`) | unchanged — `_last_usage` keeps meaning the NEWEST reading (§10 R1) |
| owner, spend | one receipt priced as FLOOR (`frontend_state.py:3400-3406`, `app.py:9377`) | `latest_custom("session_spend.v1")` — a dict lookup on the index the constructor already built (`transcript.py:1104-1115`); set `micro`, `knowledge` |
| cold viewer | checkpoint, else one receipt as FLOOR (`attached.py:1451-1539`) | checkpoint, else the same record from the suffix read (`attached.py:3763`, widened type tuple) |
| `/reload` (`app.py:8978`) | knowledge from the frontend snapshot | unchanged mechanism; the snapshot now carries the recalled knowledge |
| `/session` | the ledger's own sum (`app.py:31750`) | unchanged, plus a reconciliation line (§7.3) |
| paint | in-memory (`app.py:37609`) | unchanged — **no new work on the paint path at all** |

The paint path is untouched in every option because it reads in-memory state;
what changes is where that state came from.

### 6.2 The one-time rebuild for pre-ledger sessions

Fires when a session has usage rows but no `session_spend.v1` row — measured at
1,849 of 2,004 transcripts (92.3%).

- **Scope**: parse the transcript and sum the money on **every** usage row, with
  **no compaction boundary** (§2.2). Price each row through the full resolver
  off-loop; use `usd_cost`/`estimated_usd_cost` verbatim where present.
- **Classification**: `floor` is earned ONLY by a positive report that a money
  row is gone — never by a compaction or prune marker, which rewrite and hide
  rows without removing one (see §12.1 item 10, revised by review R1-4);
  `unpriced_calls` = rows with tokens the resolver could not price, which is the
  bound a rebuild can earn honestly (and the reason `≥` still appears).
- **Off the loop, always.** The `refresh_model_info_background` seam named in
  the brief is for *pricing*; the rebuild's blocking work is the parse, so it
  goes through `asyncio.to_thread` from a session-owned task, and publishes
  through the same frontend mutation path the band already repaints from
  (`refresh_restored_usage`, `frontend_state.py:3555`). The band paints the
  pre-rebuild state until it lands — which is today's behaviour, so there is no
  new flicker, only a correction.
- **Bounded**: run at most once per session per process; skip when the record
  already exists; never decrease a persisted `micro`; and size-cap the work
  (measured below — one session would spend 1.65 s and parse a 255 MB file).
- **Idempotent by construction**: it writes `rebuilt: true` and is skipped
  thereafter.
- **Never on the paint path, never per frame.** This is the specific hazard the
  operator named ("which must never become a per-paint recount"), and the
  guard is structural: the rebuild is a task started once at session
  construction, not a function the renderer can reach.

### 6.3 Repair / reconciliation rules

**(i) A call billed but not persisted before a crash.** The per-call write
closes the window to one call (§6.1 measured cost: 0.057 ms append+fsync on a
267 MB journal). Beyond that, the gap is **undetectable locally and must not be
guessed at**: the record cannot know a call happened. The rule is therefore:
- the record's `calls` and `micro` are the truth it claims, and a short record is
  short — the band must not invent a mark for a loss nothing observed;
- where a doubt exists it is surfaced by **reconciliation** (§below), which is a
  diagnostic and never a runtime dependency.

**(ii) A transcript rewritten by compaction or prune.** The record is a custom
entry, ignored by replay (`transcript.py:14-16`), preserved by the fold
(`transcript.py:1330-1338` keeps the newest of a collapsible type), and
**deliberately not subject to the shrink boundary** — the same rule
`search_spend_rows` states (`transcript.py:1131-1140`). Two explicit
requirements:
- the record must be written **before** any fold that could drop an older copy,
  and a fold must always keep the newest — pinned by a test that runs
  `compact_file` and asserts the newest record survived;
- `compact_file` replaces the file with a rewritten payload
  (`transcript.py:1401-1410`) — the record rides that rewrite as a normal entry.

**(iii) A session appended to by two processes.** The record is replacement
state, so two writers is the failure mode. The protecting invariant already
exists and is not new: `resume.live_runtime_pid` makes a second front end
**attach** rather than open a second writer (`transcript.py:966-970`), and the
in-process `_lock` serializes appends within a process
(`transcript.py:830`). Beyond that:
- the record carries `writer: "<pid>:<boot-id>"` so a foreign append is
  detectable;
- a `micro` that goes **backwards** without a swap is logged, exactly as the
  existing `remainder < 0` case is logged rather than swallowed
  (`app.py:36669-36690`);
- policy: newest row wins (append order is the only ordering a JSONL file has),
  and the log is the signal that the attach invariant was broken.

**(iv) Reconciliation, as a diagnostic.** `/session` already reconciles the
band's `≥` against the ledger in prose (`session_panel.py:1056-1062`). Replace
that prose with the real thing: show the record's figure beside the ledger's sum
for the same session and name the difference. The ledger is the only observer
that can see a call the record never learned about, so this is where the
residue of (i) becomes visible without coupling the band to a best-effort
subsystem. Cheap: one indexed `SUM` (`idx_calls_session`, `analytics/store.py:171`),
already off-loop (`app.py:31727-31750`).

## 7. Decision 4 — folding in the other two money streams

The headline stays exactly what it is today: `_spend_total()` =
own + children + search (`app.py:37609-37640`), one blended number, because the
cost segment sheds at rung 8 of 12 (`_DROP_LADDER`, `status_line.py:300`) and a
split would cost the whole segment on a narrow terminal.

What changes is that each arm now **recalls** its own durable total:

| arm | durable source | knowledge |
|---|---|---|
| own model turns | `session_spend.v1` (this design) | recalled (§5.4) |
| children | `subagent-roster.v1.json` `accounting` (`session.py:11460`), not UI-gated; checkpoint `subagent_cost` as a fallback | `subagent_cost_knowledge` (already on the wire, `frontend_state.py:1674`) |
| web search | transcript tool rows `search_cost`/`read_cost`, already compaction-unbounded | `SearchSpendSnapshot.cost_is_partial` (`tui/costs.py:344`) |

The blended mark is then `min` over the arms' knowledge (any lower bound makes
the headline a lower bound), which is what `cumulative_cost_knowledge` already
computes (`frontend_state.py:1802-1806`). **`_enrich`-style composition, not a
new store**: children and search already answer their own question durably, and
folding them into the new record would put the search ledger's scope decisions
(a deliberate "searches, not subagent searches" asymmetry, `app.py:37631-37640`)
in a second place.

One consequence worth stating: the band's headline is a *blend*, so the `$—`
state of the own arm does not blank the cell when search money exists — today's
`_apply_frontend_state` already spells that case (`app.py:8708-8716`:
search-only money keeps the floor mark). Keep it, and keep it derived from the
one knowledge predicate.

## 8. Decision 5 — display

### 8.1 Is the precision ladder "truncation"?

`format_cost(float)` (`status_line.py:671-677`) is:

```python
if cost < 0.01:  return f"${cost:.4f}"
if cost < 1.0:   return f"${cost:.3f}"
return f"${cost:.2f}"
```

It **rounds** rather than truncates (f-string semantics), so `$1.8978` → `$1.90`
is a rounded reading, not a clipped one. Two real defects remain:

1. **A nonzero cost under $0.00005 renders as `$0.0000`** — which reads as
   *free*, the one reading `tui/costs.turn_cost` calls *"the more expensive lie"*
   (`tui/costs.py:91-95`). A cheap call (a handful of tokens on a cheap model,
   or a 1-token probe) can land there. This IS the operator's objection.
2. **The analytics panel abbreviates** large sums to `$1.2k`
   (`analytics_panel.py:158-165`), so `/analytics` and the band can print the
   same session's money at two different magnitudes.

### 8.2 The band's spelling, per knowledge state

| state | band cell | why |
|---|---|---|
| `EXACT`, ≥ $1 | `$1.90` | 2dp; the exact figure is one keystroke away (§8.3) |
| `EXACT`, < $1 | `$0.213` | 3dp |
| `EXACT`, < $0.01 | `$0.0042` | 4dp |
| `EXACT`, < $0.00005 and > 0 | **`<$0.0001`** (8 cells, was 7) | the only new spelling on the cell, and the only case where the ladder lies |
| `PARTIAL`/`FLOOR`, < $0.00005 and > 0 | **`≥<$0.0001`** (9 cells) | **correction (review R1-3): the mark and the sub-resolution spelling genuinely CO-OCCUR** — a sub-half-micro-cent total with an unpriced call beside it is a lower bound, so `≥` is owed, and the digits round to zero, so `<` is owed. Both are true and the cell must carry both |
| `PARTIAL` | `≥$1.90` | today's `RESTORED_COST_PREFIX`, now earned |
| `FLOOR` | `≥$1.90` | same spelling — both are real lower bounds |
| `UNKNOWN` | `$—` | no mark; `≥$—` is a contradiction (`analytics_panel.py:683-689`) |
| zero, no calls | *no segment* | `_spend_text`'s documented zero policy (`app.py:37576-37587`) |

Width budget: the cost segment is one of the last things shed
(`_DROP_LADDER`, `status_line.py:300-345`; it is not shed until the ladder's
tail), and the only widening here is one cell — to **8** for the unmarked
sub-micro-cent spelling and **9** for the marked one (`≥<$0.0001`), because the
mark cannot be avoided by construction: an unpriced call makes the total a lower
bound (`PARTIAL`) whatever the digits say, and an amount under half a
micro-cent rounds to `$0.0000` whatever the mark says (review R1-3). The
earlier claim that the two "cannot co-occur" was wrong, and a width budget
built on it would have been one cell short in exactly the state a partial
session reaches first. `RESTORED_COST_PREFIX` already costs a cell in
the marked states. **The designer must still capture the band at 80, 100 and
150 columns before and after** (§10) — and the marked sub-micro-cent state
(`≥<$0.0001`) is one of the frames to capture, not only the unmarked one — a
one-cell widening is exactly the kind
of change that a green test cannot see.

### 8.3 Where the exact micro-USD is readable

`/session` is the surface with room, it already reads the ledger off-loop
(`app.py:31727-31750`), and its `Est. cost` row is already the "what has this
session cost me" answer including children (`session_panel.py:1008-1048`). It
gains:

- the exact figure at full precision from the record's integer micro-USD —
  `$1.897843`, or `1,897,843 µ$` in the note slot, so the reader can see the
  cents are not the whole number;
- the reconciliation deltas of §6.3(iv) as a row rather than a prose apology
  (`session_panel.py:1056-1062`).

### 8.4 All money surfaces agree — the mechanical part

There are **two** `format_cost` functions today: `status_line.format_cost(float)`
(`status_line.py:671`) and `analytics_panel.format_cost(_CostLike)`
(`analytics_panel.py:136`), whose ladders already disagree (`$1.2k` above $1,000
on one side, `$1234.56` on the other). The repo's own rule is one formatter:
*"one screen showing `$1.20+` and its sibling showing a plain `$1.20` for the
same partial sum would be two honesty vocabularies"* (`analytics_panel.py:187-193`).

Smallest fix: move the *number* ladder into `tui/costs.py` as
`format_usd(micro: int) -> str` (exact integer input, so the `<` case is decided
on the true value rather than a rounded float), and have both existing
`format_cost`s call it for the digits while keeping their own mark handling
(write `≥`/`+` in dim, as `append_cost` already does — `analytics_panel.py:196-211`).
Then a change to the ladder reaches every surface by construction, and the
`$1.2k` abbreviation moves to the chart's axis labels only (where the space is
actually short), never to a table cell or a headline. *(Implemented as: the
abbreviation is DROPPED, because this repo has no money chart axis to move it
to — see §12.1 note 3.)*

## 9. Perf budget

Measured on this machine. **Every assertion in the PR must be structural, not a
wall-clock bound** (`AGENTS.md` §"Timing, flakes, and how to assert that
something is fast"): thread identity for the off-loop claims, and a
call-count/no-scan invariant for the recall path. A numeric ceiling only where
no structural fact exists, calibrated from CI logs, with the dataset and margin
written into the docstring.

| path | budget | measured |
|---|---|---|
| per turn, off-loop | one price + one append per provider call | append+fsync **0.057 ms median** on a 267 MB journal, **0.038 ms** on a 94 MB one, **0.035 ms** on a 1 MB one — size-independent (60 samples each) |
| per paint | **0 new work** | reads in-memory `session.spend` |
| at session construction (record present) | one dict lookup | `latest_custom` → `_latest_custom_entries.get` (`transcript.py:1104-1115`) — no scan, by construction |
| at session construction (record absent → rebuild) | one off-loop parse+price, once ever | whole store, serial: **14.8 s CPU / 2,017 sessions** (a later snapshot than the tables above) ≈ 7 ms median (3.7 ms suffix-read parse + 0.7 ms price); worst single session **1.65 s** (`bda7b76d34e0`, 255 MB, 8,466 rows) |
| on resume | the rebuild when absent, then a lookup | the band paints the pre-rebuild state (no segment) for the ticks it takes, then corrects — which is today's behaviour |
| `/session` reconciliation | one indexed `SUM` off-loop, on open only | already the path (`app.py:31750`) |

Two bounds the coder must add, because the rebuild is the only genuinely new
cost:

- **skip/short-circuit above a size threshold** for the rebuild, reporting
  `floor: true` and only the rows it read, rather than stalling a resume behind
  a 1.65 s parse. The threshold wants a CI-derived number, not a laptop one —
  or, better, a structural rule: rebuild off-loop and publish
  asynchronously, and never block the session's open on it.
- **one rebuild per session per process**, asserted by a call count, not by
  timing.

## 10. Risks and regressions to watch

**R1 — the compaction gate genuinely wants the NEWEST reading.** `_last_usage`
(`session.py:5927`) is handed to `should_compact` (`compaction/thresholds.py:451`)
and to the prompt-cache TTL hint; seeding it from a *sum* would be a 527x-style
lie in the other direction (`usage_seed.py:66-73` documents the measured case:
900k reported against a real 1.7k). **`_last_usage` must not change.** The new
record is additive and strictly separate. Test: a resumed session's compaction
input and `context_tokens_hint` are byte-identical before and after this change.

**R2 — `cost_components` must not be billed twice.** The accumulator is fed by
the same event that feeds `cumulative_parent_cost` today, and the turn-end
remainder logic (`frontend_state.py:3835`, `app.py:36669`) must be preserved
exactly. `turn_cost` prices `cost_components` per component
(`tui/costs.py:126-142`) while the event path prices each call separately; mixing
the two is the double bill. Test: a turn with 3 calls across 2 models asserts one
total, and a turn ending with a mid-turn compaction (the `remainder < 0` path)
still floors rather than walking backwards.

**R3 — fork / copy / archive.** A fork copies the transcript verbatim except
excluded message ids (`fork.py:116-120`, `:228-241`), so the record is inherited.
That is the **intended** semantics — the fork contains the parent's conversation
and therefore its money — and it matches today's checkpoint inheritance
(`frontend_state.py:703`: *"a same-session resume keeps"*). Two things to get
right: `_inherited_identity_fixups` (`frontend_state.py:687-717`) must not strip
it, and the fork's first turn must ADD to the inherited total, not restart it.
Test: fork a session with spend, assert the fork's band = parent's total as
`EXACT`, then one billed turn produces parent + delta.

**R4 — two processes on one session dir.** See §6.3(iii). The protection is the
existing attach protocol; the new `writer` stamp and the backwards-total log are
the observability.

**R5 — the cold viewer.** `attached.py:6654 restored_usage` currently returns
`frontend_state.last_usage`; `_seed_cold_usage` (`attached.py:1451`) prices one
receipt and sets `CostKnowledge.FLOOR` (`:1536`). Both must prefer the record.
The cold path is the one that must ALSO widen `read_replay_suffix`'s type
parameter (§4) or it will silently keep the 92.3% behaviour while the owner path
is fixed — the classic "fixed one end, left the mirror" defect.

**R6 — desktop/canonical consumers.** `server/utils/desktop_sessions.py:1221`
reads the checkpoint from the transcript for cold cwd; the desktop HTTP route
consumes `UsageAggregate` (`app.py:31750`-adjacent). Neither reads a money field
the design moves, but both must be checked rather than assumed. **Mobile has no
money surface at all** — no cost field in `mobile/projection.py` or
`mobile/web/src/` — so there is nothing to update there; that is a finding, not
an omission.

**R7 — `/session` and `/analytics` stay on the ledger.** They must, and the
design must say why: they answer "what does the ledger have on record", which is
a different question from "what did this session cost", and the reconciliation
row (§6.3(iv)) is where the two are reconciled. Do not silently switch them.

**R8 — `BOOKKEEPING_CUSTOM_TYPES` cannot match an `append_custom` row.**
`_is_bookkeeping_batch` (`transcript.py:98-114`) requires
`entry.payload.get("kind") == CUSTOM_KIND_CUSTOM`, but `append_custom`
(`transcript.py:792-795`) writes `{custom_type, details}` with **no `kind`**. The
only current member (`session_incident`) is appended as a `CustomMessage`
through `append_messages` (`session.py:7874`), which is why the exemption works
there. So the rebuild's append cannot claim to be clock-neutral as written —
and it must not move the activity clock, or an old session opened once would
escape `session.cleanup`'s age ranking (`transcript.py:79-95`). Two options; the
first is recommended:
1. extend `_is_bookkeeping_batch` to accept the `append_custom` spelling of a
   listed type, with the reason recorded (the exemption is a property of the
   TYPE, and the type is what the allow-list names);
2. append the record as a `CustomMessage` via `append_messages` — which works
   today but routes a bookkeeping row through the message encoder and the
   `_PERSISTABLE_CUSTOM_TYPES` gate (`session.py:1138`) for no benefit.

Either way, the record's type must **stay out of** `_PERSISTABLE_CUSTOM_TYPES`
(`session.py:858`): it must never enter LLM context.

**R9 — the fold.** `compact_file` must always keep the newest record
(`transcript.py:1330-1338`); the type must be in `_COLLAPSIBLE_CUSTOM_TYPES`
(`transcript.py:190`) or the rows accumulate forever, as `frontend_state_checkpoint_v1`
demonstrably does (35.1% of transcript bytes, §2.3).

**R10 — the rebuild's own footprint.** 475 sessions would persist
`floor: true`, i.e. keep a `≥`. That is honest, and it is a large improvement on
1,849, but it means the operator will still see `≥` on roughly a quarter of old
sessions — for a reason that is now real. The rollout note should say so
explicitly so it is not re-diagnosed as the bug returning.

**R11 — a second pricing path.** §5.2 adds an off-loop pricer. It must call the
SAME `resolve_model_info` + `cost_for_usage` pair `price_snapshot` calls
(`analytics/model.py:249-273`), and a test must price a fixture call both ways
and assert equality — otherwise the band and `/analytics` drift, which is the
class `tui/costs.py:1-11` exists to prevent.

## 11. Test plan and the evidence the PR must carry

**Unit**

- `session/spend.py`: the four-state derivation, including the precedence
  `UNKNOWN > PARTIAL > FLOOR > EXACT`; integer micro-USD exactness (no float
  drift over 10k accruals).
- accrual: per-call + turn-end remainder == the sum of the calls, for a
  multi-model turn and for a turn with a mid-turn compaction; and that
  `_last_usage` is unchanged by all of it (R1).
- record round trip: `to_details`/`from_details` preserves state across a
  version boundary; a malformed/older row degrades to "no record", never to a
  confident zero.
- `_is_bookkeeping_batch` accepts the `append_custom` spelling of the new type
  (R8), and the record's type is absent from `_PERSISTABLE_CUSTOM_TYPES`.
- fold: run `compact_file` with several records present and assert the newest
  survives and the rest are reclaimed (R9).
- rebuild: a fixture transcript with a compaction marker → `floor: true`; with
  an unpriceable row → `PARTIAL`; with an OpenRouter receipt → used verbatim;
  idempotent (second call writes nothing).
- formatter: the ladder in one place (`format_usd`), `$1.897843` exact from an
  integer, and a nonzero sub-$0.00005 figure rendering `<$0.0001` and never
  `$0.0000`.
- **structural, not timing** (AGENTS.md §"Prefer a structural invariant"): the
  rebuild and the per-call priced append record `threading.get_ident()` and
  assert neither equals the event-loop thread — the pattern
  `test_store_maintenance_callbacks_run_off_the_event_loop_thread` already uses.
- structural recall: assert `latest_custom` performs no scan (call-count or
  identity of the returned object off the index), and that the paint path
  performs zero rebuild work (call count == 0 after N paints).

**e2e / pilot** (the repo's assembled-application stage, `tests/e2e -m e2e -n0`,
deselected from the default run — `AGENTS.md` §Environment)

- resume across a real process restart: seed a session, run billed turns through
  the fake provider with `usd_cost` set, restart the runtime, assert the band's
  figure equals the pre-restart figure with **no `≥`** — the
  `tests/e2e/test_subagent_accounting_e2e.py` pattern (it already restarts and
  asserts `frontend_state.cumulative_cost`) is the right neighbour to extend.
- the crash window: kill between a provider call and the persist, restart, assert
  the record recovers the calls that were persisted and that the knowledge state
  does not claim `EXACT` for money it lost *observably*. (Where the loss is not
  observable, assert the reconciliation row reports the ledger delta instead.)
- the compaction case end-to-end: a session that compacts, then resumes, keeps a
  total that INCLUDES the pre-compaction spend — the assertion that pins §2.2
  against regression, using the real `28a800c6a783`-shaped fixture (9
  compactions, 18 prunes, 2,134 rows, $324.99 all-rows vs $36.99 survivors).
- fork: `/fork` a session with spend and assert the inherited total (§R3).
- cold viewer / attach: `test_viewer_attach_e2e.py` and
  `test_viewer_cold_semantics_e2e.py` neighbours — the cold band must show the
  record's total and knowledge, not FLOOR.

**Evidence the PR must carry** (AGENTS.md: real execution, not a green suite)

- a matrix over the four knowledge states × the surfaces (band, `/session`,
  `/analytics`, cold viewer), with the command and the actual output for each;
- the store-wide before/after: how many sessions show `≥` before vs after, with
  the probe script committed under `scripts/` or quoted in the PR;
- the measured append cost and rebuild cost, with the dataset (this machine, N
  samples) written down;
- **rendered before/after frames** of the band at 80/100/150 columns for each
  marked state, plus `/session`'s money rows — captured with
  `scripts.visual_capture.save_capture` through the real `OperatorApp`
  (`AGENTS.md` §Visual validation; the light test hosts do not load the
  stylesheet and cannot show a width change). Designer owns this round.

## 12. Coder task list — ordered slices

Each slice is independently implementable, reviewable and verifiable. Slices
1-3 are the minimal fix; 4-7 are what makes it exact and honest; 8-9 are the
guarantees the operator asked for explicitly. The reviewer and QA gates run on
the assembled branch per the standing rules; do not merge before a fresh
`### Agent review` round, a clean QA round on the same head, and a design round
for the band frames.

**T1 — the accumulator as a value object (no persistence yet).**
New `local_operator/session/spend.py`: `SessionSpend` with integer `micro`,
`calls`/`priced_calls`/`unpriced_calls`, `floor`, `accrue()`, the `knowledge()`
derivation (§5.4), and `to_details()`/`from_details()`. Touches:
`local_operator/session/spend.py` (new). Tests: state derivation, exactness,
round trip, malformed-row degradation. *No behaviour change.*

**T2 — one arithmetic site.**
Route `FrontendStateStore.observe_event`'s per-call and turn-end cost branches
and `accrue_usage` through `session.spend.accrue(...)`, publishing
`cumulative_parent_cost` from it. Touches:
`local_operator/session/frontend_state.py` (`:3576`, `:3826-3894`),
`local_operator/session/session.py` (construct the accumulator; expose it).
Tests: R2's multi-model turn and mid-turn compaction cases; the existing
frontend-state cost tests must pass unchanged. *Still no persistence.*

**T3 — persist and recall.**
Write the record at every provider-call boundary and on every knowledge change
(a session whose only calls are unpriced must still leave a record, or it is
indistinguishable from a new session) via
`Transcript.append_custom("session_spend.v1", ...)` from a worker; add the type
to `_COLLAPSIBLE_CUSTOM_TYPES` (`transcript.py:190`) and extend
`_is_bookkeeping_batch` (`transcript.py:98`) per R8; add
`Session.restored_spend()`; make `_restore_reported_usage` (`app.py:9311`) and
`FrontendStateStore.refresh_from_session` / `refresh_restored_usage`
(`frontend_state.py:3400-3406`, `:3555-3574`) read the record instead of pricing
one receipt as FLOOR. Touches: `local_operator/session/transcript.py`,
`local_operator/session/session.py`, `local_operator/session/frontend_state.py`,
`local_operator/tui/app.py`. Tests: resume with the record → `EXACT`, no mark;
without → today's behaviour unchanged; fold keeps the newest.

**T4 — price once, off the loop, with the full resolver.**
The spend writer prices each call with `resolve_model_info` + `cost_for_usage`
on its worker, prefers `usd_cost`, and reports the authoritative micro-USD back
to the loop for the correction delta. Touches:
`local_operator/session/spend.py`, `local_operator/session/session.py`,
`local_operator/model/configure.py` (reuse, not a new resolver). Tests: R11's
equality test against `price_snapshot`; the one-tick optimistic behaviour and
the convergence.

**T5 — the cold viewer and the suffix reader.**
Widen `read_replay_suffix`'s `checkpoint_type: str | None` to
`checkpoint_types: tuple[str, ...] | None` (accepting a bare `str`), still one
`checkpoint_seen` stop condition. Make `attached.py`'s checkpoint restore
(`:1370-1447`), `_seed_cold_usage` (`:1451-1539`) and `restored_usage`
(`:6654`) prefer the record. Touches: `local_operator/session/transcript.py`,
`local_operator/session/attached.py`. Tests: cold band shows the record's total
and knowledge; e2e cold-semantics neighbour.

**T6 — the one-time rebuild.**
Off-loop, once per session per process, no compaction boundary, `floor` from
shrink markers, `unpriced_calls` from unpriceable rows, never decreasing a
persisted total, publishing through the frontend mutation path; size-bounded
per §9. Touches: `local_operator/session/spend.py`,
`local_operator/session/session.py`, `local_operator/session/frontend_state.py`.
Tests: the four provenance fixtures; idempotence by call count; the
off-the-loop-thread structural assertion; the `28a800c6a783`-shaped
compaction fixture.

**T7 — stamp `estimated_usd_cost` on the parent turn's usage — DROPPED, with a
measurement.** See §13.1: the ordering contract holds but the price is genuinely
not known at the only site that holds the usage object before the row is
written, so the stamp could only carry the paint-grade estimate. Superseded by
the record itself, which *is* the durable money.

**T8 — one formatter, one precision ladder, exact on demand.**
Move the number ladder into `tui/costs.py::format_usd(micro: int)`, have both
`status_line.format_cost` and `analytics_panel.format_cost` call it, add the
`<$0.0001` spelling, confine `$1.2k` to chart axes, and add the exact figure
plus the reconciliation rows to `/session`. Touches:
`local_operator/tui/costs.py`, `local_operator/tui/widgets/status_line.py`,
`local_operator/tui/widgets/analytics_panel.py`,
`local_operator/tui/widgets/session_panel.py`. Tests: ladder boundary cases
including the rounding-to-zero case; the analyst panel's mark legend unchanged.

**T9 — the guarantees.**
`/session` reconciliation row (record vs ledger `SUM`, off-loop); the
`writer`/backwards-total log; a `scripts/` probe that reports the store-wide
`≥` population before/after and is quoted in the PR. Touches:
`local_operator/tui/widgets/session_panel.py`, `local_operator/analytics/`
(read-only), `scripts/`.

Slice order matters in one direction only: T2 depends on T1, T3 on T2, T4 on
T3. T5-T9 are independent given T1-T3 and can be batched into one remediation
round, which is what keeps this from becoming five cascading review rounds.

### 12.1 Implementation notes — where the code differs from this document

Recorded by the implementing agent, each with the reason, so a reviewer does not
have to reconstruct the decision.

1. **The rebuild starts from the ADOPT seam, not from `Session.__init__`.** This
document says "started once at session construction". Construction is a
synchronous function that runs in contexts with no event loop (unit tests, the
SDK facades, a fork's `__init__`), where `create_task` would either raise or
leave a task nothing awaits. The rebuild therefore starts from
`Session.rebuild_spend_if_needed`, called by `refresh_frontend_usage` — the
method the app already calls when it adopts a session for display — and is
still once per session per process, asserted by a call count. The guarantee the
document actually asks for (never on the paint path, never blocking the open)
holds by construction: no renderer names it, which `test_spend_ledger.py`
asserts against `tui/app.py`'s source.
2. **A pre-ledger session's accumulator is SEEDED from its best legacy figure**
(`Session.seed_spend_floor`), in memory only. Without it the first live call
after a resume would publish only its own cost and the restored conversation's
dollars would vanish from the cell — the regression this ledger exists to
remove, reintroduced on the one population the ledger cannot yet speak for.
The seed is deliberately NOT persisted, so the record stays absent and the
rebuild still fires. Consequence, stated plainly: a pre-ledger session that is
USED before its rebuild lands persists a truthful `floor: true` record, and the
rebuild will not fire for it on a later resume, because a record now exists.
Trusting a marked lower bound over a reconstruction is the honest side of that
trade.
3. **`$1.2k` is dropped, not moved to the chart axis.** §8.4 says to confine it
to chart axes "where the space is actually short". The repo has no money chart
axis to move it to (`analytics_panel` has no axis, sparkline or bar labeller),
so the abbreviation is simply gone: `$1200.00` wherever a table cell or a
headline renders it. Inventing an axis to park it on would be inventing a
surface.
4. **`append_custom` did not take `preserve_mtime`; it does now.** R8's option 1
assumed the flag was already plumbed to that entry point. It was not — only
`append_message(s)` had it — so the bookkeeping exemption could not be claimed
by a custom row at all. The predicate (`_is_bookkeeping_batch`) and the
parameter both changed together; the mtime itself is still restored only for a
batch the predicate admits, so a caller cannot erase a real turn from the age
ranking by asking.
5. **The rebuild prices its rows in ONE worker-thread batch** (`spend.price_rows`)
rather than one `to_thread` per row. The resolver's expensive path is per MODEL,
not per row, and a batch is what makes "every price in a rebuild happened off
the event loop" a single, assertable fact.
6. **The rebuild may never DECREASE the figure already on the band.** §6.2 says
the rebuild must not walk the total backwards; this is what that costs, and the
two paths priced different things. The seed prices ONE restored reading through
the session's own effective model, which always resolves; the reconstruction
needs every ROW to carry a serving identity of its own, and a row carrying none —
written before that field, or by a provider reporting neither — is unpriceable at
full-resolver grade (`_usage_cost` covers only the 30.1% of rows carrying the
provider's own receipt). Without the guard the accumulator took the
reconstruction's `$0.00` as authoritative and replaced a priced `$2.10`, so a
resumed conversation opened claiming NO spend — the defect the restore path
exists to prevent, reached through the repair path
(`tests/unit/tui/test_usage_continuity.py`, two tests). The richer figure now
wins and no record is written for the smaller number, so the rebuild fires again
on a later resume rather than persisting a downgrade. In the direction that
matters this changes nothing: a real reconstruction is far larger than one
restored reading (§2.2's own store: `$0.0038` restored against `$324.99` of turn
rows), so the guard only ever blocks a downgrade.
7. **Live accruals are counted separately from the seed** (`_spend_live_calls`).
The first version guarded the rebuild on `spend.calls`, which `seed_spend_floor`
also increments — so a pre-ledger session that had already been opened never
rebuilt, i.e. the rebuild was disabled for exactly the population it exists for
(92.3% of the store). Only a call that accrued LIVE in this process makes a
reconstruction redundant, because only its message may be missing from the
journal a reconstruction reads.

8. **The turn-end remainder is told about a correction** (`note_spend_correction`,
review R1-1). The remainder is `max(0, aggregate_price - accrued_this_turn)` and
`accrued_this_turn` was fed by the PAINT prices alone, so a correction that moved
the accumulator mid-turn left the remainder to re-bill the whole delta: paint
$1.00 → corrected $2.00 → aggregate $2.00 persisted **$3.00** as `EXACT`. The two
numbers stay separate — one is money, the other is "what this turn's aggregate
has already been charged" — and are moved together. The mirror is scoped by call
index to the turn that accrued the call, so a correction outliving its turn does
NOT suppress the next turn's remainder: that turn's aggregate was paint-grade
over the same calls, so the delta on top of it is the corrected total.
9. **The suffix reader separates REQUIRED from OPPORTUNISTIC types** (review
R1-2). Requiring the spend record meant a pre-ledger journal — which by
definition has none — could never satisfy the stop condition, so a cold open read
to the start of the file: 4,194,304 bytes became 16,701,655 (the whole journal)
on `28a800c6a783`, and 18,192,882 on `560f212a892c`. Opportunistic types are
collected when the backward scan passes them and never gate the stop; the
compaction boundary still does, so a row inside the replayed window is still
found.
10. **`floor` is never earned by a marker** (review R1-4). Every writer that
rewrites a row keeps its money: `_pruned_entry` replaces only
`payload["content"]` (a pruned row's `usage` survives, and a pruned tool result's
`search_cost` lives in `provider_payload.details`), `compact_file` drops prune
entries and superseded collapsible customs but never a message row, and the
compaction boundary hides rows from the CONTEXT replay without removing them —
which is why §2.2's rule exists at all. A marker is evidence of rewriting, not of
loss, so the 494 sessions the census classified `rebuild_would_be_floor` are
exact or partial in fact, and `≥` now means only what it can: a call the pricing
could not size (`PARTIAL`). The predicate that remains
(`transcript.lost_money_rows`) reads a positive report that a usage row is gone.
11. **A downward re-price is not a broken attach invariant** (review R1-6). The
record's backwards-total WARNING fired for every legitimate cheaper re-price; it
is now debug-level when a correction caused the drop (one-shot token, cleared on
every write) and stays a warning for an unexplained one.
12. **A float the ladder cannot take renders, it does not raise** (review R1-7).
`micro_from_usd` returns `None` for a non-finite value, and the two wrappers that
still hold a float print what the accounting holds (`$nan`) rather than letting
`int(round(nan))` take a frame down — `tui/costs.py`'s own contract. R1-3's
corrected width claim (9 cells, not 8) is in §8.2.

## 13. What I could not settle from the code, and what would settle it

1. **SETTLED — measured 2026-09-13, and T7 is DROPPED as a result.** The
   ordering question was: does `_record_stream`'s post-stream body run before
   the harness loop finalizes `assistant.usage`, so that `estimated_usd_cost`
   could be stamped on the object the transcript row persists?

   **The ordering holds. The premise does not.** A two-line instrumented run
   (prints in `configure._record_usage` and after `assistant.usage = usage` in
   `harness/loop.py`, plus one in `analytics.model.price_snapshot`), driving one
   real turn through a real `Session` and the real `SessionStreamFn._record_stream`
   with only the wire stream faked, printed:

   ```
   PROBE configure._record_usage 0x109eceee0 est= None in= 1200 thread= 8414060928
   PROBE loop:1808               0x109eceee0 est= None in= 1200 thread= 8414060928
   PROBE price_snapshot deepseek deepseek-chat          thread= 6164295680
   transcript usage row: estimated_usd_cost=None usd_cost=None in=1200
   ```

   So `_record_usage` DOES run before the loop's finalization (same usage
   object, same thread), **but the price is not known there**: `_record_usage`
   only ENQUEUES a snapshot (``recorder.record_call`` → `queue.put_nowait`), and
   `price_snapshot` runs later on the recorder's daemon thread — after the
   turn's row is already on disk, as the last line shows. Stamping the durable
   field at that site would therefore mean pricing synchronously **on the event
   loop**, which is exactly the paint-grade estimate this document rejects
   (§5.2) — the field would carry the number §2.4 measured to be wrong.

   T7 must not be built at that site, and the alternative (the transcript write
   path) cannot help either: by the time a row is written, no holder of that
   usage object knows a price that is better than the paint resolver's. **The
   record itself is the answer to "what did this cost"** — it is written at the
   call boundary, priced where the price is knowable, and recalled in O(1). The
   `estimated_usd_cost` stamp would only have made future REBUILDS cheaper, and
   rebuilds exist for sessions that have no record; after this change those are
   the rare cases. Dropped deliberately, with the measurement recorded here so
   nobody re-derives it.

2. **Whether the transcript's `_lock` contention is acceptable for a per-call
   append** when a `compact_file` rewrite of a 255 MB journal is in flight. The
   append itself is 0.057 ms (§9), but the lock is held across the fold's
   `to_thread` (`transcript.py:1350-1390`), so a spend append can queue behind a
   whole-file rewrite. **Settled by**: measuring the append's latency in a
   branch while a fold runs on the reference 255 MB session, on this box. If the
   queueing is material, take option (a) (the sidecar) and pay its listed costs
   (§4).
3. **The right bound for the rebuild's size threshold** (§9). A laptop-calibrated
   number is exactly what `AGENTS.md` forbids. **Settled by**: CI logs across
   several runs, or by removing the need for the number entirely — publish the
   rebuild asynchronously and never block the open on it, which is the
   recommendation.
4. **Whether `frontend_state_checkpoint_v1` should keep writing
   `cumulative_parent_cost` at all** once the slim record exists, or stop, which
   would let the fat checkpoint shrink. Both are correct; stopping is a bigger
   change (every reader of the checkpoint field must move) and is deliberately
   out of scope here. **Settled by**: a follow-up measurement of how much of the
   35.1% checkpoint footprint is accounting fields vs everything else — if it is
   small, do nothing.
5. **Whether the session or the frontend store should own the accumulator**
   long-term. This design says the session (mirrored by the store) because the
   session is what persists; a reviewer may reasonably argue the store should
   own it since it is the one that sees the events. Either satisfies the
   requirements; the deciding property is "one arithmetic site", and the slice
   in T2 keeps that whichever way the ownership is spelled.

## Appendix — reproducing every number here

The store is read-only in all four probes; nothing below writes to
`~/.local-operator`. Run them with the repo's venv and the worktree on the path
(`PYTHONPATH=<worktree> <repo>/.venv/bin/python`), which is how the figures in
this document were taken. `logging.disable(logging.CRITICAL)` is needed only to
silence the paint-resolver's expected cold-miss debug lines.

**A. The restore-vs-sum gap (§2.1, §2.2).** For each session, read the journal
suffix, apply `usages_since_newest_shrink`, price each row with
`local_operator.tui.costs.turn_cost` on the row's own `provider/model_id`, and
compare the last priced row with their sum. Store-wide this is the median-64x /
max-9,854x table; the same script reports per-session compaction and prune
marker counts (`ENTRY_COMPACTION` / `ENTRY_PRUNE` in `suffix.entries`).

**B. The compaction boundary's cost on money (§2.2).** For
`28a800c6a783`: parse the WHOLE file, price every `message` row's `usage`, and
compare that total with the total over the boundary-filtered subset —
$324.99 against $36.99, with 9 compaction markers, 18 prunes and 2,134 rows.

**C. Recorded-dollar and checkpoint coverage (§2.3, §2.4).** Walk every
`transcript.jsonl`, count `custom` rows whose
`payload.custom_type == "frontend_state_checkpoint_v1"` (and their
`details.state.cost_knowledge`), and count `message` rows whose
`payload.usage` carries `usd_cost` or `estimated_usd_cost`. Byte totals come
from `len(line)+1` over the checkpoint rows against `os.path.getsize`.

**D. The rebuild's yield and cost (§5.4, §9).** For every session: build a
`Transcript`, sum the priced money over ALL usage rows (no boundary), set
`floor` when the journal carries any compaction or prune marker, and count rows
with tokens the resolver cannot price. Serial CPU was 14.8 s over 2,017
sessions, so the rebuild is ~7 ms per session on average and 1.65 s on the
255 MB outlier.

**E. The append cost (§9).** Append one ~200 B row and `fsync` it, 60 times, to
a file pre-sized to 1 / 16 / 94 / 267 MB, in `/tmp`. Median 0.035 / 0.037 /
0.038 / 0.057 ms — the cost is a constant, not a function of journal size.
