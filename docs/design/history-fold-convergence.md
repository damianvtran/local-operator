# Converging the duplicated history folds — scope, divergences, design

**Author:** architect (lopdev team), task `arch-fold-convergence`
**Date:** 2026-09-09
**Base:** `origin/main` @ `a8f98be3b` (moved past the briefed `0cd3a9434` during
this analysis), fresh worktree `/tmp/arch-fold/wt`, own venv
**Status:** analysis and plan only. **No production code written.** Every probe
below runs outside the worktree and is re-runnable.
**Builds on:** `/tmp/arch2/wt/docs/design/session-unification-adjudication.md`
§6 item 3 and §9, which established that the folds duplicate a contract and
explicitly declined to size it.

---

## 0. Summary

The redundancy is real, it is **worse than the brief describes**, and it is
already producing user-visible wrong output on `main` today.

Four findings, each verified by execution:

1. **The brief's table conflates a pager with a fold.**
   `session/history_window.py::display_window` is **not** a row fold. It returns
   `list[AgentMessage]` (`history_window.py:46`) — it selects and signs a page
   of *messages* and never produces a row. Measured by AST statement, 25 of its
   52 statements are paging/budget and **4** touch row semantics (§2). It is not
   a third implementation of the fold; it is the transport under one of them.
   Removing it from the comparison is the single biggest correction here,
   because a plan that tried to merge it into a row fold would be merging a
   cursor-signing pager into a renderer.

2. **There are, however, still three row folds, not two** — the brief lists
   two. `mobile/projection.py` contains **two independent folds over
   `AgentMessage`**: `fold_messages_to_entries` (`:567`, history pages) and
   `ProjectionFold.fold_history` (`:746`, attach seed). They are 62 and 50
   statements, they are ~80% the same code, and their own docstrings
   (`:777-787`) assert they must not diverge.

3. **They already diverge, and the worst case is on the phone against
   itself.** A hub/parent steer renders as a clean `parent_message` card on a
   scrolled-up history page and as a **raw `<parent-message>` XML envelope** in
   a `notice` row on attach — same session, same row, one scroll gesture apart
   (§3, D1). Nine divergence classes are enumerated in §3; all nine are verified
   by execution, six against a real on-disk `Transcript`.

4. **The performance constraint the brief flags as hard is not binding on the
   fold.** Measured (§4): `fold_messages_to_entries` over 3,000 messages costs
   **5.3 ms**. The 90–900 ms in `durable.py:1-9` is the **file parse and
   replay**, not the fold — and the cache already sits *above* the fold behind a
   one-line seam (`durable.py:446-449`). Convergence of fold semantics therefore
   does **not** touch the cache at all.

**Recommendation (§6): converge the two PHONE folds into one total function
now, keep all three caches exactly where they are, and do NOT merge
`TranscriptEntry` with the TUI's block tree.** The TUI fold mounts Textual
widgets and costs 1,061 ms where the phone fold costs 5.3 ms (§4) — a 200x gap
that is entirely widget construction. They are one *contract* with two
*renderers*, and the correct shared artifact is a declarative row model plus an
exhaustiveness check, not a shared row type.

The honest framing the brief invited: this is **"converge the semantics, keep
the caches separate"** — and the evidence lands there decisively (§4, §6.4).

---

## 1. What I measured, and how to re-run it

| Probe | Establishes |
|---|---|
| `/tmp/arch-fold/probe_diverge.py` | 18 synthetic conversations through the TUI fold and the phone fold side by side |
| `/tmp/arch-fold/probe_e2e.py` | The same, but input is a **real `Transcript.build_llm_history()`** off disk |
| `/tmp/arch-fold/probe_phone_pair.py` | The two phone folds against each other |
| `/tmp/arch-fold/probe_buckets2.py` | Per-fold concern buckets over AST **statements** |
| `/tmp/arch-fold/probe_perf.py` | Fold/replay/pager costs at 120 / 600 / 3,000 messages |
| `/tmp/arch-fold/probe_rowspec.py` | **Prototype of the recommended design**: the closed union under real pyright, and a `RowSpec`-based phone fold reproducing today's output |

**Two of my own findings I corrected mid-analysis**, recorded because the
standard on this team is that a corrected finding is worth more than a confident
one:

- My first bucketing pass counted docstrings and comments as code and produced
  an `other/plumbing` bucket of ~60%. In a file where comments outnumber code by
  design, that made the table meaningless. `probe_buckets2.py` walks AST
  statement nodes instead and skips docstring `Expr` nodes; §2 uses only that.
- My first perf run measured a **128-byte journal**: `Transcript()` takes a
  *directory*, not a file path (`transcript.py:508`), so my "cold parse" numbers
  were parsing an empty file. Re-measured against real 108 KB / 543 KB / 2.7 MB
  journals; §4 uses only the corrected run. The uncorrected numbers would have
  understated cold replay by ~4x and I would have drawn the wrong conclusion
  about where the 90–900 ms lives.

---

## 2. Ground truth on the overlap

### 2.1 The four implementations, correctly classified

| # | Implementation | Location | Stmts | Input → Output | What it actually is |
|---|---|---|---:|---|---|
| 1 | `project_settled_rows` (+ `replay_tool_call`) | `tui/session_presentation.py:471`, `:801` | 119 + 20 | `AgentMessage` → Textual widgets | **Row fold** (owner/TUI) |
| 2 | `fold_messages_to_entries` | `mobile/projection.py:567` | 62 | `AgentMessage` → `TranscriptEntry` | **Row fold** (phone, history pages) |
| 3 | `ProjectionFold.fold_history` | `mobile/projection.py:746` | 50 | `AgentMessage` → `TranscriptEntry` | **Row fold** (phone, attach seed) |
| 4 | `_capture_display_window` | `session/history_window.py:223` | 52 | `AgentMessage` → `AgentMessage` | **Pager**, not a fold |

### 2.2 Concern buckets (AST statements, `probe_buckets2.py`)

```
concern             TUI.rows  TUI.tool  PH.pagefold  PH.attachfold  OWN.pager
tool pairing              13         1            8              9          4
custom rows                7         0            5              4          0
chrome suppress            3         0            1              1          0
images                     4         2            2              1          0
refusal/fail               6         4            2              2          0
payload fields             1         4            4              4          0
paging/budget              5         0            0              1         25
host mount                11         1            6              7          0
plumbing                  69         8           34             21         23
TOTAL                    119        20           62             50         52
```

Shared **row-semantics** statements (the six meaning-bearing concerns, excluding
paging, host mount and plumbing):

```
TUI.rows        34        PH.pagefold     22
TUI.tool        11        PH.attachfold   21
                          OWN.pager        4
```

Read three ways:

- **The pager is not in this problem.** 25 of 52 statements are paging/budget;
  4 touch row semantics, and all four are the tool-call/result *grouping* it
  does to avoid splitting a group across a page boundary
  (`history_window.py:282-297`). That is a genuinely paging concern that happens
  to need one fact about rows. **Leave it alone.**
- **The two phone folds are near-duplicates**: 22 vs 21 shared-semantics
  statements over the same six concerns, differing by one. They are the same
  function written twice.
- **The TUI fold is a superset, not a duplicate.** 45 shared-semantics
  statements against the phone's 22, and its extra weight is real behaviour
  (refusal/fail 10 vs 2; custom rows 7 vs 5; chrome suppression 3 vs 1) — which
  is exactly the divergence list in §3.

### 2.3 The honest overlap number

Genuinely-same-contract-implemented-more-than-once, by concern:

| Concern | Contract | Implemented in | Verdict |
|---|---|---|---|
| tool call ↔ result pairing | index results by `tool_call_id`, render on the call's row | 1, 2, 3 (+ grouping only in 4) | **Duplicated 3x** |
| custom-row dispatch | wake / peer / hub / gate-timeout / compaction-refused → their own row kinds | 1, 2, 3 | **Duplicated 3x, and disagreeing** |
| harness-chrome suppression | LOOP_PROMPT, `_CONTINUATION_PROMPT`, connectivity prompt, `$skill` payload never render verbatim | 1, 2, 3 | **Duplicated 3x, and disagreeing** |
| refusal / interrupt / failed turn | `stop_reason` → a visible notice | 1 only | **Missing from 2 and 3** |
| payload fields (`details`, `duration_s`) | diff counts + elapsed off `provider_payload` | 1, 2, 3 | **Duplicated 3x** |
| image attachment resolution | user rows carry attachments | 1, 2, 3 | Duplicated 3x, forms differ legitimately (bytes vs refs) |
| wire budget / cursor signing | — | 4 only | **Host-specific, correctly isolated** |
| cache strategy | — | 4 (`_DisplayWindowCache`), `durable.py` | **Host-specific, correctly isolated** |
| row type / mount | — | each | **Host-specific, legitimately different** |

**Five of six row-semantics concerns are implemented three times. Two host
concerns (budget, cache) are correctly isolated already.** The duplication is
entirely in row *meaning* and entirely absent from row *transport* — which is
what makes this tractable: the thing to share is the part with no host
dependencies.

---

## 3. The divergences that already exist

**This is the payload of the report.** Every row below was produced by running
both folds; the `E2E` column marks the ones reproduced against a real on-disk
`Transcript` replayed through `build_llm_history()` rather than synthetic
messages.

`probe_e2e.py` on one real 8-message transcript: **TUI renders 9 rows, phone
renders 6.** Three rows silently vanish on the phone.

| # | Conversation fact | TUI renders | Phone renders | Impact | E2E |
|---|---|---|---|---|---|
| **D1** | Hub/parent steer (`hub_message`) | *nothing* (main transcript) | **attach seed: raw `<parent-message>` XML in a notice**; history page: clean `parent_message` card | **Phone contradicts itself across one scroll.** Leaks the model-facing envelope `extract_parent_message` exists to hide | ✅ |
| **D2** | Assistant turn with `stop_reason="refusal"` | `NoticeBlock("content policy", error)` | **nothing** | User sees a truncated answer with no indication the provider cut it off | ✅ |
| **D3** | Failed turn (`stop_reason="error"`, no text/calls) | `NoticeBlock("turn failed", error)` | **nothing** | Prompt followed by silence; looks like the agent ignored them | ✅ |
| **D4** | Interrupted turn (`stop_reason="aborted"`) | `NoticeBlock("interrupted", error)` | **nothing** | Same as D3 | ✅ |
| **D5** | Gate timed out unattended | `NoticeBlock("waited 2h for approval with nobody attached, then denied it — bash · rm -rf /x", warning)` | **nothing** | The most expensive event in the detached feature is invisible on the phone | ✅ |
| **D6** | Wake delivery (`wake_prompt`) | `WakeBlock` (dedicated affordance, `catchup` aware) | generic `notice` | Wake receipts lose their identity | ✅ |
| **D7** | Compaction refused (`compaction_refused`) | `NoticeBlock`, **`error` vs `warning` ink chosen from the detail text** (`session_presentation.py:666`) | **nothing** | The optimistic "compacting context…" receipt is never corrected on the phone | ✅ |
| **D8** | `$skill` invocation | `UserBlock("$research the widget market")` via `_typed_line_of` | **the entire expanded SKILL.md payload as the user's bubble** | Phone shows a wall of skill body; also poisons any phone-side session titling | — |
| **D9** | `LOOP_PROMPT` / `_CONTINUATION_PROMPT` | suppressed | **rendered as a user bubble** | Harness chrome shown as the user's own words. Note the phone *does* suppress the third prompt, `CONNECTIVITY_CONTINUATION_PROMPT` (`projection.py:635`, `:776`) — a **partially copied list**, which is the drift signature itself | — |
| **D10** | Unanswered tool call | `ToolCard` state `interrupted` | `tool_state="done"` | A call that never returned is shown as **succeeded** | — |
| **D11** | Bang-mode (`! cmd`) | card opens expanded (`user_run=True`) | ordinary collapsed row | Cosmetic; the user's own command does not self-open | — |
| **D12** | `compaction_summary` marker | *nothing* | `notice` | Divergence, but **benign**: the phone's web renderer treats `compaction` and `notice` identically (`transcript.tsx:181-182`) | ✅ |

Notes on scoping honesty:

- **D1 is the headline.** It is the only one where the phone disagrees with
  *itself*, it leaks model-facing markup to a user surface, and both call sites
  carry comments claiming the opposite is guaranteed
  (`projection.py:777-787`: *"the two folds must not diverge or an attaching
  phone and a lazy-loaded page would disagree about the same row"*). The comment
  is wrong on `main` today. Verified end-to-end:

  ```
  PHONE ATTACH SEED:      notice | '<parent-message> focus on the parser </parent-message>'
  PHONE HISTORY PAGE:     parent_message | 'focus on the parser'
  ```

- **D2–D5, D7 are one class**: the phone fold has no `stop_reason` branch and no
  custom-row branch beyond peer/hub/generic-notice. Confirmed absent: `grep -rn
  "stop_reason" local_operator/mobile/` returns only the *live-event* aborted
  flag (`projection.py:876`) and web types — **no history fold anywhere in the
  mobile stack reads `stop_reason`.**
- **D10 is the same bug class as the three shipped duration bugs** the brief
  cites (#823 `4790838ce` and its two follow-ups): one path silently defaulting
  where another is explicit. `TranscriptEntry.tool_state` defaults to `"done"`
  (`mobile/types.py:370`), so an unpaired call is *asserted* successful rather
  than left unknown. A default that means "success" is how this class keeps
  recurring.
- **D12 I initially wrote up as a bug and downgraded** after reading the web
  renderer: `case "notice": case "compaction":` fall through to the same
  element. It stays on the list as a latent divergence — the day those two kinds
  render differently it becomes real — but it is **not** a user-visible defect
  today and I am not counting it as one.

### 3.1 Why this got worse over time, structurally

`EntryKind` (`mobile/types.py:339-351`) has a `compaction` member emitted **only
by the live event path** (`projection.py:1023`) and by neither history fold. So
the phone already has a row kind reachable live and unreachable on replay. There
is no exhaustiveness check anywhere: no test in `tests/unit/mobile/` or
`tests/unit/tui/` compares the folds' output on one input (verified by grep;
`test_projection.py` pins each fold *separately*, and its header even claims
*"these tests pin the TUI-parity contract"* while never executing the TUI). The
folds are kept in step by review attention alone, and review attention has
already failed nine times.

---

## 4. Performance: the constraint is not where the brief expects

Measured on this host (`probe_perf.py`, min of 7, real journals):

| messages / journal | cold `build_llm_history` | warm replay | **`fold_messages_to_entries`** | **`project_settled_rows`** | `display_window` (cold) |
|---|---:|---:|---:|---:|---:|
| 120 / 108 KB | 0.77 ms | 0.26 ms | **0.19 ms** | **39.8 ms** | 2.9 ms |
| 600 / 543 KB | 3.87 ms | 1.33 ms | **0.98 ms** | **205.5 ms** | 5.1 ms |
| 3000 / 2.7 MB | 19.7 ms | 6.84 ms | **5.26 ms** | **1061.3 ms** | 11.1 ms |

Three conclusions, and they largely decide §6:

1. **The fold is not the cost. The file is.** `durable.py:1-9` attributes
   90–900 ms to "construct a fresh `Transcript` per request"; my measurement
   splits that into ~20 ms parse + ~5 ms fold at 3,000 messages (the operator's
   52 MB transcripts scale the parse, not the fold). The module's own third
   design fact already says this — *"Fold cost is O(history), not O(file)…
   ~1 ms where the file parse measured hundreds"* — and my numbers confirm it
   at 5.3 ms for 3,000 messages. **A semantics change to the fold has
   approximately no performance consequence.** The cache exists to avoid the
   parse, and nothing proposed here touches the parse.

2. **The cache is already decoupled from the fold.** `durable.py:446-449` is a
   four-line seam:
   ```python
   def _fold(history: list[AgentMessage]) -> list[Any]:
       from local_operator.mobile.projection import fold_messages_to_entries
       return fold_messages_to_entries(history)
   ```
   The incremental cache calls the fold through one function. Replacing what the
   fold *does* requires changing nothing in `durable.py`. This is the single
   most important structural fact in the report: **the caches and the semantics
   are already separated, so "converge the semantics, keep the caches" is not a
   compromise — it is the shape the code is already in.**

3. **The TUI fold is 200x the phone fold and that gap is irreducible.** At 3,000
   messages: 1,061 ms vs 5.3 ms. Profiling (`cProfile`, 600 messages) attributes
   it to widget construction, not semantics — `AssistantBlock._flat_whole`
   0.330 s of 0.723 s, `rich.markdown.Markdown.__init__` 0.160 s,
   `ToolCard.__init__` 0.095 s. The TUI is not folding slowly; it is
   *rendering*, eagerly, inside the fold. **Any design that gives the TUI and
   the phone one shared row-producing function must not make the phone pay
   Textual construction, and must not make the TUI fold twice.** This is what
   kills the "single row type" option in §5.

---

## 5. Options

### Option A — one shared fold core, host-specific adapters (visitor/emit)

A pure `fold(history, emit)` where `emit` is a host-supplied sink receiving
semantic row events (`emit.user(...)`, `emit.tool(...)`, `emit.notice(kind,
text)`). The TUI's sink mounts widgets; the phone's builds `TranscriptEntry`.

- **Pro:** every §3 divergence becomes structurally impossible — one dispatch,
  one chrome-suppression list, one `stop_reason` branch. Adding a row kind is
  one `emit` method that every sink must implement.
- **Pro:** no shared row type, so the phone never constructs a widget and the
  TUI never builds a `TranscriptEntry`.
- **Con:** the TUI's fold is not purely row-emitting — it interleaves
  `_painted_tool_card` / `_settle_painted_tool_card` reconciliation
  (`session_presentation.py:550-553`), head/tail deferral into
  `_resume_pending_head`, and `batch_append`. Those are 11 "host mount"
  statements that must stay on the TUI side of the seam, and getting the seam
  wrong reintroduces the #401-class risk in the most-contended file in the repo.
- **Con (largest):** it touches `tui/app.py`'s replay path — 82 commits and
  +36,737 lines in 7 days per the adjudication's measurement. Blast radius on
  the TUI is the real cost here, not the design.

### Option B — one row type (`TranscriptEntry`) with host-specific renderers

Fold once to `TranscriptEntry`; TUI renders widgets *from* entries.

- **Pro:** conceptually cleanest; one fold, provably one semantics.
- **Con — disqualifying:** `TranscriptEntry` is a **lossy wire type** built for
  a phone. It caps args to `TOOL_ARGS_CHARS`, truncates output to a tail
  (`projection.py:684`), carries images as `{index, mime}` *references* while
  the TUI needs the actual `ImageContent` blocks (`session_presentation.py:708-712`),
  and has no representation for `user_run`, `completion_anchor_id`,
  `navigation_anchor_part`, or the painted-card reconciliation. Making it
  lossless for the TUI means growing it until it is no longer a wire type —
  and it is serialized to the phone on **every streaming token**
  (`mobile/types.py:378-385` explains why bytes must never be inlined). Widening
  it directly regresses the SSE budget.
- **Con:** forces the phone's caps into the TUI or forces two variants of the
  type, which is the divergence again with extra steps.

### Option C — converge the two PHONE folds only; share nothing with the TUI

Merge `fold_messages_to_entries` and `ProjectionFold.fold_history` into one
function. Leave the TUI alone.

- **Pro:** kills D1 outright (the self-contradiction), and it is the only
  divergence where one surface disagrees with itself.
- **Pro:** zero blast radius on `tui/app.py` — the entire change is inside
  `mobile/projection.py`, which no other in-flight PR is rewriting.
- **Con:** does **not** fix D2–D5, D7–D11, which are TUI-vs-phone. The phone
  stays silent about refusals, failed turns and gate timeouts.
- **Con:** leaves the bug class alive — two folds becomes one fold, but the
  *contract* is still asserted by comment rather than checked.

### Option D — do nothing / fix forward case by case

- **Pro:** honest baseline; each divergence is individually a small fix.
- **Con:** nine divergences accumulated under exactly this policy, three
  duration bugs shipped from the same class, and both prior architects
  independently pointed at this seam. The evidence that "fix each one" does not
  hold is the list in §3.

### Option E — shared **declarative** semantics + exhaustiveness check (recommended core)

Extract the *decisions* — not the rendering — into one pure, host-free module
that maps one `AgentMessage` (plus fold-local pairing state) to a **closed
union of semantic row descriptors**. Each host has a *total* renderer over that
union, and totality is enforced by the type checker (`assert_never`) plus a
property test that both renderers accept every variant.

This is Option A with the seam drawn at the *decision* rather than at the
*emit*, which keeps the TUI's mount mechanics entirely on the TUI side.

---

## 6. Recommendation

**Adopt E, staged, starting with C. Keep all three caches exactly where they
are. Do not converge `TranscriptEntry` with the TUI's block tree.**

Concretely:

1. **Source of truth for row semantics: the TUI fold** (`project_settled_rows`
   + `replay_tool_call`). Not because it is prettier — because §2.2 measures it
   as the strict superset (45 shared-semantics statements vs 22/21) and §3 shows
   every divergence resolves in its favour except D12 (benign) and D1 (where
   *neither* is right — the TUI renders nothing and the correct answer is the
   history page's `parent_message`). Its behaviour is also the one with review
   history attached: the comments cite review rounds and specific findings
   (MAJOR-1/U7/D1, round 2 m2, round 5 U17), so its choices are adjudicated
   rather than incidental.

2. **Source of truth for the *type*: neither.** Introduce a new closed union of
   row descriptors — `RowSpec` — in a host-free module (no Textual, no wire
   types, no session import), mirroring how `compaction/marker.py` already sits
   below hosts that "must not import the session" (`marker.py:11-14`). Both
   `TranscriptEntry` and the TUI's blocks are *projections of* `RowSpec`, not
   each other.

3. **Caches stay put.** `_DisplayWindowCache` (owner paging),
   `durable.py`'s incremental cache (daemon), and the TUI's
   `_resume_pending_head` deferral solve three different problems at three
   different layers, and §4 shows the fold is 0.5% of the cost the daemon cache
   exists to avoid. Converging them would be a large change with no measured
   benefit. **This is the part of the brief's framing I am pushing back on:
   performance is a hard constraint, but it constrains the *cache*, and the
   cache is not what is duplicated.**

4. **The pager (`history_window.py`) is out of scope.** §2.1. Do not touch it.

### 6.1 Arguing against my own recommendation

**Attack 1: "E is A with extra ceremony — you invented a type to avoid drawing a
seam."** Partly fair. The difference is load-bearing but narrow: A's `emit` sink
must be *called from inside* the shared fold, which means the shared fold owns
the loop, which means the TUI's painted-card reconciliation and head/tail
deferral must either move into the shared code (wrong — they are host mount
concerns) or be hoisted out of the loop (a real refactor of the most-contended
function in the repo). E's `RowSpec` is a *value*, so each host keeps its own
loop and its own mount mechanics and only the decisions are shared. If that
distinction turns out not to survive contact with the TUI's `_resume_pending_*`
slicing, **E degrades to C plus a checker** and I would take that outcome over
forcing it.

**Attack 2: "You are recommending a new abstraction beside two existing ones —
the thing the role brief tells you to avoid."** The strongest attack. My defence
is that the alternative is not "fix the existing thing" but "fix the existing
thing nine times and then again next quarter", and §3.1 shows there is no
mechanism that would catch the tenth. But I concede the shape of the risk, which
is why the plan (§8) makes Stage 1 and Stage 2 **independently valuable and
independently shippable**: if the team stops after Stage 2, the phone is
self-consistent and the worst divergences are fixed, with no new abstraction
landed at all.

**Attack 3: "Just delete `ProjectionFold.fold_history` and call
`fold_messages_to_entries`."** This is genuinely most of Stage 1 and I want it
on the record as the cheapest real fix. It does not fully work as stated —
`fold_history` also maintains `_tool_rows`/`_tool_args` correlation maps and
prunes them to the surviving tail (`projection.py:846-859`), which the pure
function cannot do — but the *row production* half can be replaced wholesale,
with the state maintenance layered on top. That is exactly Stage 1.

**Attack 4: "D12 shows you are inflating the list."** Fair challenge, which is
why I downgraded it in place rather than dropping it. Eight of the twelve are
user-visible today; D11 is cosmetic; D12 is latent-only.

### 6.2 What would change my mind

- If the TUI's `_resume_pending_head`/`_resume_pending_tail` slicing cannot be
  expressed over a `RowSpec` list without a second pass over messages, the
  message→row seam is in the wrong place and the answer is C + checker.
  **Evidence that settles it:** a prototype of Stage 3 that produces
  byte-identical mounted blocks for the `tests/unit/tui/test_resume_render.py`
  corpus. I did not build this — it is the first thing Stage 3 should do, and
  Stage 3 should be abandoned if it fails.
- If a `RowSpec` variant needs a Textual-only or wire-only field, the union is
  not host-free and E collapses to B's problem.

---

## 7. Proving the bug class dies

The brief's bar: a future field added to `provider_payload`, or a new row kind,
must become **impossible** to drop on one path and not the other. Four
mechanisms, in decreasing strength.

### 7.1 Exhaustiveness at the type level (kills the *new row kind* half)

`RowSpec` is a closed discriminated union. Each host renderer dispatches with a
terminal `assert_never`:

```python
def render(spec: RowSpec) -> None:
    match spec:
        case UserRow(): ...
        case AssistantRow(): ...
        case ToolRow(): ...
        case NoticeRow(): ...
        case WakeRow(): ...
        case PeerRow(): ...
        case ParentRow(): ...
        case _ as unreachable:
            assert_never(unreachable)   # pyright errors on a new variant
```

Adding a variant makes **pyright fail on every host that does not handle it** —
and pyright is already a required gate. This is the mechanism that would have
prevented D2–D7: each is a row kind one host knows about and the other silently
falls through. Today the phone's fold ends its `if/elif` chain with no `else`,
so an unhandled message is dropped in silence; that is precisely how five row
kinds went missing.

**Strength: strong — and this is EXECUTED, not argued.** I prototyped the union
and two renderers (`/tmp/arch-fold/probe_rowspec.py`) and ran the repo's own
pyright over them. A renderer that handles every variant passes; the identical
renderer with the `NoticeRow` case deleted fails:

```
error line 107 :: Argument of type "NoticeRow" cannot be assigned to
                  parameter "arg" of type "Never" in function "assert_never"
```

`NoticeRow` is precisely the variant carrying D2–D5 and D7 — the five rows the
phone silently drops today. **Under this mechanism, the phone fold as it exists
on `main` would not typecheck.** It is a compile-time failure in a gate the
project already runs, not a test someone can forget to write.

### 7.2 Totality of the payload projection (kills the *new field* half)

The `provider_payload` → row-fields projection becomes **one function** in the
shared module:

```python
def tool_payload(message: Message) -> ToolPayload:   # total, no host may re-read the dict
    ...
```

with `ToolPayload` a frozen dataclass. The rule that makes it stick: **no host
may read `message.provider_payload` directly** — enforceable by a ~15-line AST
test asserting the string `provider_payload` appears in exactly one module
outside `harness/`. Today it is read independently at
`projection.py:671-673`, `projection.py:826-833` and inside the TUI's
`replay_tool_call`, which is why the same `details`/`duration_s` keys PR #858 is
repairing on one path are pre-loaded to break on the others.

Adding a field then has exactly one edit site, and every host receives it or
fails to typecheck.

**Strength: strong for the intended class; it does not stop a host from
*ignoring* a field it receives** — which §7.4 covers.

I also prototyped the *other* half of the claim — that the phone fold can be
rebuilt as `messages → [RowSpec] → [TranscriptEntry]` — and it reproduces
today's output exactly on a non-divergent case:

```
today  : [('user','edit it',...), ('assistant','editing',...), ('tool','','edit','done',9.75,3,1)]
RowSpec: [('user','edit it',...), ('assistant','editing',...), ('tool','','edit','done',9.75,3,1)]
EQUAL: True
```

So Stage 3 is demonstrated, not merely designed. **Stage 4 (the TUI half)
remains un-prototyped and stays gated** — see §6.2 and §10.

### 7.3 Differential property test (catches semantic drift the types cannot)

The test that does not exist today and should:

> For every message shape in a generated corpus, `phone_rows(h)` and
> `tui_rows(h)` produce the **same sequence of row kinds** and the same
> per-row identity/text/state, modulo an explicitly declared and *asserted*
> allowance list.

`probe_diverge.py` and `probe_e2e.py` in this report are that test in
prototype form; `PreparedReplay` (`session_presentation.py:216-283`) already
makes the TUI fold drivable headless, so the harness cost is near zero. The
allowance list is the key discipline: a divergence must be *named* to be
legal, so D11 (bang-mode expansion) would be an explicit entry and D2–D7 could
not be.

**Strength: medium-strong.** It catches what types cannot (two hosts both
handling a kind, differently), and it fails loudly. It is a test, so it can be
skipped — but it fails on the corpus, not on a hand-written case, so it does not
rot the way per-case tests do.

### 7.4 Where the proof is honestly incomplete

**A field that every host receives and one host chooses not to render is still
representable.** No type stops a renderer from ignoring a `RowSpec` field. §7.3
catches it only if the corpus exercises that field. So the claim I will defend
is narrower than "the bug class is unrepresentable":

- **new row kind dropped on one path** → impossible (7.1, compile-time)
- **new payload field parsed on one path only** → impossible (7.2, single site)
- **field carried everywhere but rendered differently** → *caught, not
  prevented* (7.3)

Option C alone achieves none of the three; Option B achieves the first two but
at the wire-budget cost in §5. **That is the argument for E over C**, and it is
the whole argument — if the team does not value the first two, C is cheaper and
I would not fight for E.

---

## 8. Staged execution plan

Sequenced by **correctness risk and blast radius**, not effort. Each stage is
independently shippable and independently valuable; the plan can be stopped
after any stage without leaving the tree half-converged.

### Stage 0 — differential harness (no production code)

Land `tests/unit/mobile/test_fold_parity.py` (or `tests/unit/tui/`) containing
the §7.3 differential test **with today's divergences encoded as an explicit,
commented allowance list**. It goes green on `main` immediately.

- **Why first:** it is the regression net for every later stage, and it converts
  §3 from a report into an executable artifact. Every subsequent stage's
  evidence is "N allowances deleted".
- **Blast radius:** zero — tests only.
- **Parallelisable:** no (everything else depends on it).
- **Reversible:** entirely.

### Stage 1 — collapse the two phone folds (Option C)

Make `ProjectionFold.fold_history` call `fold_messages_to_entries` for row
production, keeping its correlation-map maintenance
(`projection.py:846-859`) and `_cap_tail` layered on top.

- **Fixes:** D1 (the self-contradiction, and the XML leak).
- **Why here:** the highest-severity divergence, and it is entirely inside
  `mobile/projection.py` — **no `tui/app.py` contact at all**.
- **Blast radius:** small and phone-only.
- **Parallelisable:** yes — independent of Stage 2.
- **Evidence:** the D1 reproduction in §3 must flip; QA drives a real phone
  session with a hub steer, opens it (attach seed) and scrolls up (history
  page), and captures both.

### Stage 2 — teach the phone the rows it is missing

Add the `stop_reason` branch and the missing custom-row dispatch to the (now
single) phone fold: refusal, failed turn, interrupted turn, gate timeout,
compaction-refused, wake identity, `$skill` typed line, chrome suppression,
unpaired-call `interrupted` state.

- **Fixes:** D2–D11.
- **Why here:** these are the user-visible ones, and they need **no new
  abstraction** — they are additions to one function, checked by Stage 0's
  harness with allowances deleted one per fix.
- **Blast radius:** phone renderer needs a `tool_state="interrupted"` case and
  possibly a notice severity; `mobile/web` is a real but contained surface.
- **Parallelisable:** **yes, and this is where the fan-out is** — D2/D3/D4
  (`stop_reason`), D5/D7 (custom notices), D6 (wake), D8/D9 (chrome), D10
  (`tool_state`) are five independent slices over disjoint branches of one
  `if/elif` chain. They must land as **one commit round** per the team's
  batching rule, but they can be *written* concurrently.
- **User-visible ⇒ designer round required** (new row kinds on the phone), and
  D10's `interrupted` state needs a glyph decision.

### Stage 3 — extract `RowSpec` and make the phone total (Option E, phone half)

Introduce the host-free `RowSpec` union and the single `tool_payload`
projection; rewrite the phone fold as `messages → [RowSpec] → [TranscriptEntry]`
with `assert_never`.

- **Delivers:** §7.1 and §7.2 for the phone, and the AST guard that
  `provider_payload` is read in one place.
- **Blast radius:** still phone-only. The TUI keeps its own fold, now checked
  against the phone by Stage 0's harness.
- **Irreversible-ish:** this is the first stage that adds a new module. It is
  the decision point — **stop here if Stage 3 review finds the union awkward**,
  and the tree is still strictly better than today.

### Stage 4 — move the TUI onto `RowSpec` (Option E, TUI half) — *gated*

Rewrite `project_settled_rows` as a `RowSpec` consumer, keeping
`_painted_tool_card` reconciliation, head/tail deferral and `batch_append`
on the TUI side.

- **Gate:** do not start until the §6.2 prototype shows byte-identical mounted
  blocks for the `test_resume_render.py` corpus. **If it does not, abandon
  Stage 4** — the tree keeps Stages 0–3, which is a good outcome.
- **Blast radius: the largest in the repo.** `tui/app.py` and
  `session_presentation.py` are the most-contended files; the adjudication
  measured 82 commits / +36,737 lines in 7 days on `app.py`.
- **Not parallelisable with anything.** One worktree, one PR, serialised.
- **Irreversible steps:** none technically (it is a refactor), but a regression
  here is a frozen or wrongly-rendered TUI, which is the #401 class. Requires
  the full visual-validation treatment in `AGENTS.md` — rendered before/after
  SVG frames for resume, reconnect-gap replay, and bang-mode.

### Concurrency and merge-conflict strategy

- **Stages 0, 1, 2, 3 barely touch `tui/app.py`** — deliberately. The plan
  front-loads all the value that can be delivered without entering the
  contended file, and quarantines the contended work into a single gated stage
  at the end.
- **PR #858 interaction:** Stage 3's `tool_payload` is the natural home for
  exactly what #858 makes total in `Message.tool_result`. **Stage 3 must rebase
  onto a merged #858 and adopt its factory rather than duplicating the stamp** —
  if #858 is still open when Stage 3 starts, Stage 3 waits. Stages 0–2 do not
  touch `harness/types.py` and are unaffected. I have not read #858's diff and
  am relying on the brief's description.
- **Absorbing concurrent merges:** each stage begins by rebasing on
  `origin/main` and **re-running Stage 0's differential harness before writing
  any code**. New divergences introduced by other sessions show up as harness
  failures against the allowance list, which is precisely the "periodically pull
  in-flight work into the new pattern" the operator asked for — the harness *is*
  the mechanism. A new allowance entry appearing without a comment is the review
  signal that someone added a row kind on one side only.
- **Rebase discipline:** per the team's rule, a rebase onto a moved base is
  never re-pinnable — prove content unchanged with `git range-diff '='` plus
  byte-identical ±line sets, then one convergence round scoped
  `<old-head>..<new-head>` checking only files upstream also touched.

### Roles per stage

| Stage | coder | reviewer | qa-tester | designer | ux-reviewer |
|---|---|---|---|---|---|
| 0 | ✅ | ✅ | ✅ | — | — |
| 1 | ✅ | ✅ | ✅ | — | — |
| 2 | ✅ (fan-out, one commit round) | ✅ | ✅ | ✅ (new phone rows) | — |
| 3 | ✅ | ✅ | ✅ | — | — |
| 4 | ✅ | ✅ | ✅ | ✅ (TUI frames) | ✅ (resume flow) |

---

## 9. Risks to watch during rollout

1. **Stage 2 makes the phone noisier.** Five row kinds that were invisible
   become visible. On a long agent session, "turn failed"/"interrupted" notices
   may be frequent. **Watch:** the designer round should decide severity ink and
   whether interrupted turns collapse. This is a genuine product change, not
   just a bug fix, and should be framed that way to the operator.
2. **`_cap_tail` interaction (Stage 1/2).** The attach seed caps to a render
   tail; the history page is uncapped. Adding rows changes what the cap keeps —
   `_cap_tail` deliberately preserves the opening user message
   (`projection.py:1630-1656`). **Watch:** a session whose tail is now mostly
   notices could push real content out of the seed.
3. **Stage 3 union churn.** If `RowSpec` needs a variant per custom type, it
   grows with every new custom type, and `assert_never` then makes *every* new
   custom type a multi-host change. That is the intended cost, but it is a real
   tax. **Watch:** if variants exceed ~12, prefer a generic `NoticeRow(kind,
   severity, text)` over per-type variants.
4. **Stage 4 performance.** The TUI fold is 1,061 ms at 3,000 messages *today*;
   a `RowSpec` intermediate adds an allocation per row. It should be noise
   against widget construction (§4), but it must be **measured, not assumed** —
   re-run `probe_perf.py` before and after.
5. **The allowance list rotting.** Stage 0's value depends on entries being
   deleted, not accumulated. **Watch:** any PR that *adds* an allowance without
   an explicit reviewer-acknowledged reason is the failure mode this whole plan
   exists to prevent.
6. **`durable.py` cache invalidation is untouched but adjacent.** Stages 1–3
   change fold output while the cache keys on file inode/size, not on fold
   version. **Verified:** the cache is a process-local singleton owned by the
   daemon (`daemon.py:112`, `_DURABLE_FOLD_CACHE = DurableFoldCache()`), so it
   dies with the process and cannot outlive a restart. The residual risk is
   therefore only that a **long-running daemon is not restarted** by
   `lop-update` and keeps serving old-fold rows to the phone while a new TUI
   renders new-fold rows. **Watch:** make the daemon restart part of each
   stage's rollout, and have QA confirm `mobile` service restart before
   validating phone output — otherwise a stage will appear not to have
   landed.

---

## 10. Uncertainty, stated

- **Stage 3 IS prototyped** (§7.1/§7.2, `probe_rowspec.py`): the union
  typechecks, the missing-variant renderer fails pyright, and the
  `RowSpec`-based phone fold reproduces today's rows. **Stage 4 is not.**
- **I did not prototype Stage 4.** The claim that the TUI's mount mechanics can
  sit above a `RowSpec` seam is reasoned from reading
  `session_presentation.py:471-798`, not executed. §6.2 names the experiment
  that settles it and §8 gates the stage on it.
- **I did not read PR #858's diff** (instructed not to disturb it). Stage 3's
  interaction with it is inferred from the brief.
- **D8/D9/D10/D11 are verified on synthetic messages only**, not through a real
  `Transcript` round-trip. The message shapes are taken from the code that
  builds them, but a real skill invocation and a real bang-mode record should be
  captured in Stage 0's corpus rather than trusted from here.
- **The 200x TUI/phone fold gap is measured headless** via `PreparedReplay`,
  which does not mount into a live `App`. Real mounting may be slower still;
  it will not be faster, so the direction of the §5-B argument holds.
- **`_message_text`'s hub behaviour** (§3 D1) depends on hub messages carrying a
  `text` key in `details`. I verified this is what `comms.py:2065-2074` writes.
  A hub message written by an older version without that key would degrade to a
  dropped row rather than a leaked envelope — still a divergence, different
  symptom.

---

## 11. Summary for the manager

**Overlap:** five of six row-semantics concerns are implemented three times
(TUI fold, phone history-page fold, phone attach-seed fold). The two host
concerns that would be expensive to share — wire budget and cache strategy — are
already correctly isolated. **The brief's third fold is a miscount:
`display_window` is a pager (25 of 52 statements are paging; 4 touch rows) and
should be left alone. The real third fold is `ProjectionFold.fold_history`,
inside `mobile/projection.py`.**

**Divergences:** 12 found, 11 real, 8 user-visible today. The worst is **D1**:
the phone shows a hub steer as a clean card on a scrolled-up page and as a raw
`<parent-message>` XML envelope on attach — the same row, one gesture apart,
under comments that explicitly promise this cannot happen. **D2–D5 and D7** mean
the phone silently omits refusals, failed turns, interrupted turns, gate
timeouts and compaction refusals: on one real transcript the TUI renders 9 rows
and the phone renders 6.

**Performance:** not the constraint people think. The fold is **5.3 ms at 3,000
messages**; the 90–900 ms in `durable.py` is file parsing. The cache already
calls the fold through a four-line seam (`durable.py:446-449`), so **converging
fold semantics requires no cache change at all.**

**Recommendation:** converge the semantics, keep the caches separate — which is
the option the brief invited and the evidence lands on. Source of truth for
meaning is the TUI fold (measured superset); source of truth for the *type* is
neither — a new host-free `RowSpec` union, because `TranscriptEntry` is a lossy
wire type re-sent on every streaming token and widening it regresses the SSE
budget. **Do not merge `TranscriptEntry` with the TUI's blocks. One fold, two
renderers.**

**Bug-class proof:** a closed union + `assert_never` makes a dropped row kind a
**pyright failure** (kills D2–D7's class); a single `tool_payload` projection
plus a one-site AST guard makes a dropped payload field impossible (kills the
#823/#858 class). A field carried but rendered differently is *caught, not
prevented*, by the differential test — I state that limit rather than overclaim.

**Plan:** five stages, front-loaded to deliver all value reachable **without
touching `tui/app.py`**, with the one high-blast-radius stage last and *gated*
on a prototype. Stage 0 (differential harness, green on `main` today) is the
mechanism that absorbs concurrent merges: every later stage re-runs it after
rebase, and a new allowance entry is the review signal that someone changed one
fold and not the other.

**Stop-anywhere property:** stopping after Stage 2 leaves the phone
self-consistent and the user-visible divergences fixed, with no new abstraction
landed. That is the fallback if `RowSpec` does not survive review.
