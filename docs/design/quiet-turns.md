# Design: first-class silent turns (`no_reply`) and quiet-message groups

Status: PROPOSED — §8 gates the implementation slices to this document. Base:
`origin/main` @ `01c69f40` (v0.68.21). Author: architect, 2026-10-09; revision 2,
2026-10-10.

**Provenance.** The original investigation read core `origin/main` @ `7208e28406`
as a read-only blob (git/grep/read only; nothing built or run); revision 2's
amended anchors were verified against core `origin/main` @ `01c69f40` — the base
of this document — and the S0a implementation worktree re-checked every anchor it
touches at that same cut. Paths are `local_operator/…`; `s.py` is
`session/session.py`, `loop.py` is `harness/loop.py`; the UI repo is
`~/local-operator-ui`, the native app `~/local-operator-mobile`. Numbers marked
(measured) came from a one-off scan of `~/.local-operator/sessions/*/transcript.jsonl`
made for the original note. AGENTS.md sections that apply: "The tool-surface
footprint ladder" (rung 3 `createIf`, schema tax on every request) and "Adding a
configuration key" (avoid; use an env kill switch). Nothing here needs a version
bump.

---

## 0. Problem, as found

Peer, wake and monitor deliveries open turns. The prompt tells the model "end it with no reply, and don't notify" (`prompts_md/system.md:74-76`; `guides/monitor/GUIDE.md:163-166`). No mechanism exists to do that, so models write text. The literal "(no action needed)" was taught by Aida's first seed (`git show 8b24f2b477`, 2 hits) and then copied by the model from its own history. Evidence (measured):
- Aida session `439818272d84`, 2026-10-02 to 10-09: 143 assistant replies that are exactly `(no action needed)`, longest run 9 consecutive. Triggers just before each one: 88 after a tool result, 34 `peer_message`, 11 `monitor_prompt`, 10 `session_state`. Median gap about 5 min.
- The same session holds 3,429 inbound `peer_message` rows, mean model-facing size 1,785 chars (about 450 tokens), 985 of them from one sender. **The filler's own context weight is negligible (about 6 tokens x 143). What drives context is inbound volume.** The honest wins of silence are visual noise, notifications and model drift, not tokens. Do not oversell token savings.
- Fleet scan, 1,534 sessions touched in the last 4 days: 4,184 trigger rows; the text-only replies right after them number 546 wake, 281 job_result, 57 peer, 22 monitor. Only 9 were empty and only 7 distinct replies were shorter than 70 chars. So the filler is concentrated in Aida's session; the design is still general because the failure is structural.
- Empty clean stops do happen: 19 `stop`+empty and 68 `error`+empty in that scan. They are provider glitches, not intent.

Key finding: **silence is already a recognised shape downstream.** A completed run with no persisted text-bearing assistant message is written `eligible:false` by `_publish_attention_outcome` (`s.py:12148-12158`; the text filter is `s.py:12078-12083`). That publishes no completion row, hence no unread mark and no banner. Nothing reliably produces that shape, because models never emit an empty reply. The missing pieces are (a) an affirmative way to end a turn and (b) a loop that does not re-call the model after it.

---

## 1. How turns end today, and how each check reacts to a textless turn

The loop has no "final response required" rule. A model message with no tool calls ends the inner loop: `has_more_tool_calls = bool(assistant.tool_calls)` (`loop.py:2268`). Then the outer tail runs: `on_before_yield` (`loop.py:2301`), `_collect_yield_injections` (`loop.py:2307`), the output-contract gate (`loop.py:2398`), `break` and `AgentEndEvent` (`loop.py:2436-2437`). So a textless end is legal in the loop.

| Check | Where | Reaction to a turn with no assistant text |
|---|---|---|
| Todo completion | `_todo_continuation` `s.py:16508`, wired via `get_follow_up_messages=self._guardrail_continuations` `s.py:13656` | Never reads text. Fires on open todos while the fingerprint moved. **Reacts correctly, as for any end.** |
| Project completion | `_project_continuation` `s.py:16561` | Needs `_turn_tool_calls > 0` (`s.py:16606`). The counter counts every tool start/end (`s.py:13749-13750`), so **a `no_reply` call would count as work and trigger a stale-project nudge on a quiet wake. It must be excluded** (slice S0). |
| Output contract / final-response gate | `loop.py:2398-2434`, `_last_assistant_text` `loop.py:4984` | **Misreads silence as failure**: `None` becomes "the turn produced no assistant message", and `""` is a failing answer. Only hosts that set a contract (`lop exec --output-format`, `s.py:13684`). The tool must be absent there. |
| Ask gate | `_gate_ask` `s.py:9343-9445` | Not a turn-end check. It is the `ask` tool's clearance fork and runs only when an ask is enqueued. Unaffected. |
| Supplements | `docs/design/turn-supplements.md` §2.2 (PROPOSED, PR #2116) | Eligible only for a "real user run"; wake/peer/monitor runs are ineligible and the persisted-final-message rule fails closed. Unaffected. |
| Goal judge | `serving.py:3101`, `_maybe_judge_goal` `serving.py:3632` | Reads only `error/aborted/generation`, never text. A quiet end still triggers a judge call when a goal is active. Unchanged; no regression. |
| Attention outcome | `_publish_attention_outcome` `s.py:12025`; republish skips `eligible:false` `s.py:11536`; import skips it `attention.py:1175` | Textless complete turn is `eligible:false`: no row, no unread mark, no banner on resume. **This is the existing silent path.** |
| Stop notice | `rows.assistant_stop_notice` `harness/rows.py:1379` | Only `refusal`, `length`, and `error`/`aborted` with nothing produced yield a notice. `stop`/`toolUse` with no text yields none, so it is not misread. |
| Aida banner veto | `_aida_cadence_banner_veto` `s.py:11909`; `_run_last_assistant_text` `s.py:12007` returns `""` | An empty reply already counts as quiet. Consistent with a silent end. |
| Headless / API final text | `headless_print.py:627` (`if final_text:`), `server/utils/operator.py:804` | An empty string prints or returns nothing. The tool is not built for one-shot hosts (`s.py:2995-3002`), so this is moot. |

Only two things misread silence: the output-contract gate (excluded by absence) and the project counter (fixed by exclusion). A wire-legality note: a tool-call turn followed by its result and then a new user row already occurs after an abort at a batch boundary (`loop.py` abort-after-batch, around `:2205-2232`: results are appended and not fed back). A silent-ended turn has the same shape.

## 2. Existing notification machinery (where each rule lives)

- Arm-time intent: `WakeParams.notify` `tools/builtin.py:12536` (default false, "pass true when the user asked to be told"); `WakeSchedule.notify` `harness/wake_types.py:99`; monitor `notify` `builtin.py:12784`. Carried into the delivery's `details` at `s.py:21801` (wake), `:22137`, `:22205` (monitor and notice), `:20597` (catch-up).
- Per-run trigger record: `_note_run_input` `s.py:11793` (classes `user`, `wake_prompt`, `monitor_prompt`, `internal`; ORs the deliveries' `notify`), `_has_awaiting_user` `s.py:11840`, reset per run at `s.py:13246`.
- The one rule: `_finalize_attention_notify` `s.py:11857-11907` (design `docs/design/monitor-tool.md` §14, PR #1737). User semantics win (a reply to a real user message always notifies). Any non-wake/monitor trigger notifies, **which includes a peer message: peer turns have no discretion today**. Wake/monitor-only runs notify iff `notify` was set at arm time. Errors always notify. Aida veto last (only True to False). It is computed once in `_emit` (`s.py:12806`), journaled (`s.py:12175`), stored by `AttentionStore.publish(notify=)`, and read by `desktop_sessions._maybe_publish_notification` (`:2885`), `desktop_feed._emit_notifications` (`:1321`), the TUI funnel, the push worker, and `serving._announce_completion` (`serving.py:5716`).
- §14.5 deliberately did not build a per-turn "raise" hatch. So today "at agent discretion" means arm-time for wake/monitor and nothing for peer. `no_reply` adds the negative half (suppress), and a normal reply is the positive half.

## 3. PR #2110 (merged 2026-10-09 23:46Z): what it does and how to consolidate

Scope: Aida's check-in only. `QUIET_REPLY = "Nothing needs your attention today."` plus the bounded legacy `"(no action needed)"` (`aida/proactive.py:122,129`); `reply_is_quiet` peels case, quotes and a trailing period (`:196-235`); `Session._aida_cadence_banner_veto` (`s.py:11909`) runs after the base formula and only turns True into False. It applies only to `_aida_checkin_run()` (`s.py:11971`): Aida's session, wake-only inputs, every wake id cadence-family. The reply stays persisted and visible by decision (design D1: it is the session-list preview the operator reads daily). The banner budget (`MAX_BANNERS_PER_DAY`) is stamped only for her check-in banners.

It does not touch the observed filler: peer, monitor and tool-boundary turns in her session, because the veto's guard excludes them.

Consolidation, not duplication:
1. Leave Aida's cadence path alone. Her rows declare `notify=True`, and `no_reply` is refused when `_run_notify_requested` (see §4), so the cadence keeps its visible sentence and the existing veto. No second sentinel recogniser.
2. Add one predicate beside `_run_last_assistant_text`, e.g. `Session._run_ended_quiet(event)`. It is true when the run's last tool result carries the quiet marker and no later assistant text exists. Call it from `_finalize_attention_notify` (force `notify=False`, **only when `kind != "error"`** — review R3: the error arm "always notifies, whatever the origins" must not be silenced, reachable when a quiet batch's re-entry errors afterwards) and `_publish_attention_outcome` (treat as `eligible:false`, i.e. extend the `not messages` branch at `s.py:12148`). It runs before the Aida veto, so quiet runs never reach the banner-budget stamp.
3. Do not generalise `reply_is_quiet` beyond Aida. The UI explicitly avoids text heuristics (`turn-segments.ts` header), and Hermes needed a stream filter to stop `NO_REPLY` text flashing (`gateway/stream_consumer.py:975-1021`). A sentinel in text streams to every surface before it can be recognised; a tool does not.
4. Update `aida/README.md` Notifications section with one line: "a turn that ends with `no_reply` publishes nothing". Aida's seed (`agent_seeds/aida.md:77`) keeps the sentence for the cadence; add one line for peer/monitor/job turns. Editing the seed requires regenerating `manifest.json` / `seed_revisions.json` (as #2110 did).

## 4. Recommendation: a dedicated `no_reply` tool, no turn-end text signal

Why a tool and not a sentinel or "empty reply = silent": the model must perform an affirmative act; an empty reply is indistinguishable from the provider glitches measured above (68 `error`+empty). Treating those as intentional would hide real failures. A text sentinel streams and persists.

**Tool**
- Name `no_reply`. Parameters: none (`extra="forbid"`, no fields). No `reason`: a reason can only live in the call's persisted arguments (re-billed), and the trigger row beside it already says why.
- Description (about 50 tokens): "End this turn silently when a peer message, wake, monitor or job result needs no reply and no action. Nothing is shown or notified — never answer an acknowledgement with an acknowledgement, and never write filler such as '(no action needed)'. Refused if the user asked something or the wake/monitor asked to notify." The acknowledgement ban lives HERE rather than in `system.md` (review R6): the schema is deferred, so this text is unbilled until the tool is activated, while the system prompt is paid on every request of every session.
- Tier `read`, `concurrency="exclusive"`, `interruptible=False` (like `patience`, `builtin.py:13308`).
- Result: short text "Quiet." with `useless=True` (`harness/types.py:523`, so the existing prune pass `compaction/pruning.py:671` blanks it) and `details={QUIET_TURN_KEY: True}`. `ToolResult.details` is never sent to providers, the same carrier as `OUTPUT_LIMIT_KEY` (`loop.py` limit arm).

**Loop (one change)**
At `loop.py:2268`: `has_more_tool_calls = bool(assistant.tool_calls) and not _batch_ends_quiet(tool_results)`. `_batch_ends_quiet` reads the batch's **LAST result only** (review R8): a `no_reply` earlier in a batch must never shorten the turn — its siblings' results would otherwise never be fed back — so a batch that does not END in the quiet result continues normally. Everything after runs as for a normal end: `on_before_yield`, yield injections (open todos re-nudge, a late steer or peer message re-enters the loop, so no inbound message is lost), then `AgentEndEvent`. There is no new event field: the session derives quietness from the end event's messages. Old viewers see an ordinary end.

**Availability (`createIf`, footprint rung 3)**
- Add a `ToolContext` capability `quiet_end: Callable | None` beside `gate_ask` (`harness/types.py:1600`); bind it with the other per-turn callables near `s.py:14530`. Its presence IS the fact that this session may end quiet. It is `None` for: subagent children (`_job_id` set; a parent's `wait` expects final text, `subagent.py:2014`), one-shot/headless hosts (`_one_shot_exit`), sessions with an output contract, and when `LOP_NO_REPLY=0` (env kill switch, mirroring `LOP_ASK_GATE`; no config key).
- Register `"no_reply": lambda ctx: builtin.build_no_reply_tool(ctx)` at the END of `TOOL_BUILDERS` and `DEFAULT_TOOL_NAMES` (`tools/registry.py:40-96,133`; appending keeps the cache prefix stable).
- Defer its schema: add to `DEFERRED_TOOLS` and `DEFERRED_TOOL_PURPOSES` ("end a turn silently") in `tools/deferral.py`. Reasoning: the call has no arguments, so it needs no schema to form; calls to tools absent from the array were measured accepted on four wires (`deferral.py` docstring); and the rule is named in the system prompt, so adoption does not depend on the schema. Cost avoided: roughly 60 schema tokens per request on every top-level session. **Named probe, owner S0:** 20 scripted peer-ping turns per arm (deferred vs not) on one live model. If the tool is not called in at least 80% of "nothing to say" turns, remove it from `DEFERRED_TOOLS` (one line; the deferral PR re-admitted `network` and `ask_withdraw` on exactly this adoption signal).
- The classification layer (`classification/recommend.py`) recommends skills, guides and MCP servers per user message. It has no tool kind and peer/wake/monitor turns are not user messages. No recommendation work is needed.

**Behaviour matrix (decided in `quiet_end`, which returns a refusal string or None). THE RULE IS A DENYLIST** (review R4): `_note_run_input` collapses every non-wake/monitor custom input into the single class `internal` (`s.py:11828-11838`), so peer, hub, job result, incident notice and session_state are indistinguishable in `_run_triggers`; an allowlist would need a new finer-grained per-run record for nothing. Refuse iff ANY of the three denylist conditions holds; everything else may end quietly.

| Situation | Behaviour | Why |
|---|---|---|
| REFUSE iff `"user" in _run_triggers`, or `_has_awaiting_user()`, or `_run_notify_requested` | Tool result `is_error`: "A person asked this turn; answer them in one line." / "this wake or monitor asked to tell the user; say what they need to know." The model then writes text. Never coerce silently. | R15a user semantics win; never override a user-expected notification. Costs one extra call, rare. |
| Everything else, including `internal` runs — peer, hub_message, job_result, `session_incident` notice, session_state, resume catch-up, wake/monitor with notify=false | Allowed. Stamps `notify=false`, `eligible:false`. | The denylist is the implementable rule (note above). An incident notice's operator-visible row is already published, so silence about it loses nothing; pinned by test. |
| Pending queued `ask` | Allowed. | The ask card is its own notification; the turn end is separate. |
| Open todos / stale projects | Allowed; guardrails run at the yield boundary exactly as for any end (a moved fingerprint re-nudges once). | One end semantics. |
| Same batch has other calls | They run; the turn ends quietly ONLY if the quiet call is the batch's LAST result (review R8). An earlier `no_reply` never shortens the turn — the whole batch's results are fed back and the model continues. | Pinned by test. |
| Earlier narration text in the same run | Stays visible and persisted. Quiet end still publishes nothing. | Explicit signal wins; text already streamed cannot be recalled. |
| Steer/peer message arrives after the call | Loop re-enters via yield injection and answers it. | `loop.py:2307`. |

**Brief answer recorded (requirement 3):** a peer-only run that WRITES TEXT stays notifying in v1 — the write is the agent's own choice to surface, and `no_reply` is the suppress half made newly expressible. The alternative reading — silent-by-default for peer, with an explicit notify opt-in re-aiming clause 3 of §14.2's formula (`_finalize_attention_notify`) — is recorded, not taken; it is the change to make if the operator wants peer turns quiet unless raised.

**What persists:** the assistant tool-call message and its tool result (ordinary rows; an older runtime or client replays them as a normal tool pair); the `completion_attention` marker with `eligible:false` (existing); the per-turn journal row (`serving.py:5622`) and spend record (existing). **What does not:** assistant text, a completions row, an unread mark, a banner, a push, the banner-budget stamp. "Action count" and "Worked for" derive from rows on each client; the quiet call is excluded from action counts, and the summary carrier is specified in §5's quiet-close rule (review R1). Per-turn context cost of the pair is about 40 tokens (call plus result), which is more than a 6-token filler. No elision in v1 (rewriting old history changes the cached prefix for little gain, and elided pairs would show the model a stack of unanswered messages). Measure per-turn tokens with the ledger and revisit only if the pair dominates.

## 5. Peer-message condensing: how it works today and the proposed group

**Where condensing lives (verified):** core has no server-side turn condensing. Core only has the history windows (`session/history_window.py`, signed cursors, display pages, hidden-tool subtraction at `:608,:821`), the picker preview `session/preview.py::condense_entries`, and the relay fold. Each client condenses for itself:
- UI (`local-operator-ui`): `turn-segments.ts` (cycles, answer election, V1-V4 visibility, `partitionRun`; label table at `:590-650` already names a peer-opened bar "Peer message"), `turn-collapse-model.ts` (planRun liveness: only the in-flight cycle draws in place; `settledCloseOf` near `:1370`; the end-loaded rule near `:176-190`: a bar describes exactly the rows on hand), `transcript-rows.ts:374` (`isStatementRow`). Peer/wake receipts are no longer pinned (operator report 2026-09-29, `staysVisibleWhileCollapsed`). `(no action needed)` survives as a text-bearing `stop` row, which V4 keeps visible and `cyclesOf` makes a close; that is why the operator sees a line per peer message.
- Native (`local-operator-mobile`): `src/features/session/turn-condensing.ts`: the unit is user-row to user-row; invariants "active (last) turn never condensed", "once condensed stays condensed (latch)", "bar words frozen". `STATEMENT_KINDS` (`:123`) includes `peer_message`. In a conversation like Aida's the last user turn is old but still the "active" turn, so it never condenses. **Native needs its own group item, independent of turn condensing.**
- Relay web (`mobile/web/src/components/transcript.tsx`): windowing only (`:383`), no condense. TUI: one `PeerMessageBlock` per message (`tui/widgets/transcript.py:2670`), replay at `tui/session_presentation.py:1220`.
- Rows only append in all of these, and the relay fold's `_cap_tail` slides the window (`mobile/projection.py`).

**Group model (client-derived; no new wire kind, no capability flag)**
A server-computed group row would be a growing, mutating row against append-only folds, and would need a negotiated capability (the `DisplayHistoryWindow` forbids extra fields; `history_window.py:49-60` documents the failed-attach hazard). Instead, one structural definition, implemented per client and pinned by a shared parity fixture (precedent: `mobile/web/src/lib/format.parity.json`, `spend-context.parity.json`).

Definition: a **quiet group** is a maximal run of >= 2 trigger rows (`peer`, `hub_message`, `wake`, `monitor_prompt`, `job_result`) with no visible assistant text, no `user` row and no terminal marker between them. Tool rows (including the quiet call) may sit between them. Completion/terminal markers and answers are never inside a group because they split it (same boundary vocabulary as `boundaryKindOf`).

Review round 1 confirmed the group mechanics against UI `@020c568f`: a rendered completion receipt (`notice{complete:true}`) is a terminal marker (`transcript-reducer.ts:2544-2585`) and therefore splits a group; keys are `qg:firstRowId`; latch/freeze and the head-cut "at least N" rule hold; no wire field.

Shape (derived, never sent):
```
QuietGroup { key: "qg:" + firstRowId,           // stable: rows append, the key never moves
  family: "peer"|"wake"|"monitor"|"job"|"mixed",
  count: n trigger rows, firstTs, lastTs,        // time span; absent when ts unknown (head-cut)
  senders: [{label, count}] (peer only, top 2 + "N more"),
  actions: n non-quiet tool rows, failed: n,
  open: bool,                                    // it is the tail and no later visible row exists
  rowIds: [...] }
```
Copy: `Peer messages · 12 · 2h 14m` (family word plural; `Messages` for mixed). Expansion lists each receipt with its own actions nested, exactly as the expanded bar does today. Facts freeze when the group closes (latch pattern, `LatchedTurn`); the open tail group may increase its count (single-line bar, no geometry change). A head-cut group states "at least N" and no duration, per the existing end-loaded rule.

Per surface:
- **Core:** persisted pair is ordinary. Add `QUIET_TURN_TOOL = "no_reply"` and `is_quiet_turn_call/result` to `harness/rows.py`. TUI and relay folds skip the pair's row (the `SEND_TOOL_NAME` precedent: `mobile/projection.py:1586,:3443`; the TUI `is_hidden_tool_call` precedent `tui/session_presentation.py:1614`, `tui/app.py:14649,:54207`). Do **not** add it to `HIDDEN_TOOL_NAMES` (`rows.py:352`), because `history_window` subtracts that set from display pages and the UI needs the quiet call as its structural close.
- **UI — the QUIET CLOSE rule, and who carries the summary (review R1).** A settled `no_reply` call is a quiet close: (a) `cyclesOf`/`settledCloseOf` see it as a settled close, so the tail settles and the working line retires; (b) the collapsible gate (`turn-collapse-model.ts:1464-1466`: `segments.length > 0 && (run.opensWithUserRow || run.closingAnswerId !== null)`) gains it as a closing anchor — carried as a new `quietCloseId` on the turn span, NOT folded into `closingAnswerId`, which stays null so no caption or stamp claims an answer; (c) `electAnswer` excludes quiet closes **as candidates only** — they remain closes in the cycle scan, and that distinction is load-bearing: removing them from the scan instead would leave the tail with no close, the unsettled state this rule exists to fix; for a narrated quiet turn the earlier text stays narration under the existing step rule, and election must NOT fall back to a previous cycle's close — that close was already handed over, and re-electing it would label the run's end with an earlier answer (the same shape today's textless tail already has); (d) consequence: a quiet turn has no elected answer, hence no caption, no stamp and **no foot** — the foot renders only on the row carrying `closesTurn` (`canonical-transcript.tsx:1308-1375`), and that row would be the paint-hidden quiet close, so a foot there is invisible by construction; (e) the summary carrier is instead the **extended bar facts**: the bar over the turn's work states actions (non-quiet tool rows; the `no_reply` call excluded, subtracted like `without_ask_gate_divert` subtracts a divert, `rows.py:531-559`) and duration, with the completed mark from `segmentIsCompleted` and the quiet close as the settled last row; a grouped quiet run's GROUP bar carries it. Sub-case (i), a single silent acted turn below the group minimum: same rule — its bar carries the facts; when no bar exists (live tail, or a run with nothing collapsible) the work rows draw in place and each tool row states its own duration; no aggregate line is fabricated. Sub-case (ii), textless turns generally: `no_reply` is the only sanctioned textless end; provider empty-stops stay provider glitches. Pin: a story in `turn-collapse.stories.tsx` on `[peer][tool][no_reply]` — bar present, completed mark, `actions` counting only the work tool, no caption, no foot — plus unit assertions in `scripts/turn-segments.test.mjs` and `scripts/turn-collapse-model.test.mjs` (no elected answer; the collapsible gate passes; `settledCloseOf` returns the quiet close). Old UI builds show a small `no_reply` tool row: acceptable degradation.
- **Native:** new `TranscriptItem` kind `"group"` built from consecutive `peer_message` entries, in any turn including the active one; latch on close; the relay hides the quiet pair so no hidden-row bookkeeping is needed.
- **Relay web / TUI:** committed group slices S4/S5 (§8). Until they land they show individual cards without filler, which is already the main improvement.
- **Lazy load:** group facts come only from rows on hand. Any paging (history_window, `_cap_tail`) stays as is; no server field.

## 6. Prompt guidance (exact text)

Injection points (verified): system prompt `prompts_md/system.md:71-76`; wake envelope `harness/wake.py:662-682` (`format_wake_delivery_text`, which already carries `WAKE_SCRATCH_CLAUSE`); monitor delivery `monitors/delivery.py:63-90`; peer envelope `s.py:8570-8577` (`_peer_custom_message`; `details["text"]` is model-facing and persisted, so any clause there is re-billed per message).

Replace the tail of `system.md` lines 74-76 ("A wake or monitor turn that finds nothing needing action is complete: end it with no reply, and don't notify.") with (requirement 4 recorded; review R5/R6: names job results, and the acknowledgement ban plus the filler example live in the deferred tool description and the guides instead, so this edit stays near net-neutral — the PR measures the byte delta and trims if it lands much above the replaced sentence; children that can never call the tool still pay this line):

> A peer message, wake, monitor or job result needing no reply or action: call `no_reply` and write nothing. Reply only if you acted or something changed that the user should know.

Also: `guides/monitor/GUIDE.md:163-166` ("end the turn with no reply") becomes "call `no_reply`; don't notify" (`notify:true` deliveries are exempt, see §4). `guides/peer-messaging/GUIDE.md` gets one sender-side sentence and one receiver-side line. Sender: "Don't send a message that is only an acknowledgement or a status with no ask; say 'reply needed' if you need one — the receiver may stay silent (`no_reply`)." Receiver: "End a peer turn that needs no reply with `no_reply`; assistant text is not sent back to the sender, so a bare acknowledgement reaches nobody." (Review R5: delivery mechanics stay in this guide; the receiver line closes the "write text" reading.) No envelope clause in v1; add a ≤1-line clause to the wake/monitor envelopes only if the adoption probe fails.

Filler-encouraging wording to remove: `system.md:75-76`; `monitor/GUIDE.md:163-166`. Keep Aida's `QUIET_CLAUSE`/`CADENCE_MESSAGE` (visible sentence by design D1). The tool and prompt must ship together (S0b may not merge before S0a is deployed, otherwise the prompt names an absent tool). Byte discipline (review R6): the replacement's byte delta against the replaced sentence is measured in the PR and pushed into the guides (unpaid) if it lands much above parity; only the deferred tool description and the guides carry the acknowledgement ban.

## 7. Backwards compatibility

- Runtime/client older than the change: the pair is two ordinary rows; clients render a generic tool row; `EntryKind`/schemas are unchanged (native `schemas.ts` header: unknown kinds are tolerated anyway). No new field on `AgentEndEvent`, attach frames, history windows or the relay `TranscriptEntry`, so no capability string. If a later slice must add one, it MUST be negotiated like `DISPLAY_HISTORY_*`.
- Older runtime resuming a transcript that holds the pair (mixed builds against one sessions directory are routine, `history_window.py:49-54`): replay is tool_use + tool_result for a tool absent from its registry. Wire-accepted (the deferral measurement) and legal by the abort-boundary precedent. Named probe: one request per wire.
- Persisted transcripts without the records: nothing to migrate. Historic filler rows have text, stay visible, and are not grouped (no text heuristics).
- Replay of a silent turn: `eligible:false` is skipped by republish (`s.py:11536`) and by boot import (`attention.py:1175`), so a resume never raises a banner. Risk: a runtime killed between the persisted tool result and the `eligible:false` marker is classified by `_classify_orphaned_run`, which can publish an `error` (always notifies). Same exposure as any turn, but worse for a quiet one; add a kill-between test.
- Mesh/placed sessions: the tool runs in the owning runtime; nothing crosses the wire but ordinary rows.

## 8. Work breakdown (PR-sized, ordered)

Every slice: conventional commit, no version bump, agent review round, QA round on the head. UI and native also need design rounds (rendered frames from stories/Storybook, not source). Dependencies: S0a, then S0b/S1 in parallel, S2 and S3 after S0a (and S3 prefers S1), then the two group slices S4 and S5 — **committed, not optional** (review R2: the operator asked for groups on every surface; descope only with explicit operator sign-off), on the same shared parity fixture.

**S0a — core mechanics + policy** (one PR; touches harness, tools, session)
- `ToolResult` marker constant + `build_no_reply_tool`; `loop.py:2268` end-turn; `ToolContext.quiet_end` + binding at `s.py:14530`; registry/deferral rows; `_run_ended_quiet`, notify False and `eligible:false` for quiet ends; refusal rules; exclude the call from `_turn_tool_calls` (`s.py:13749`); `LOP_NO_REPLY` kill switch; `docs/design/quiet-turns.md` carrying this design and the parity fixture.
- Targeted tests: `tests/unit/harness/test_loop.py` (quiet marker ends the turn with no extra model call; late injection still re-enters; todo reminder still fires; batch-position review R8 — a `no_reply` early in a batch does NOT end the turn and the whole batch's results are fed back); new `tests/unit/tools/test_no_reply_tool.py` (the params model declares no fields; only the injected `i` is accepted — review R7: `apply_intent_schema` injects it (`harness/intent.py:203+`) and the loop pops it pre-validation (`loop.py:3460-3461`); refusals); `tests/unit/tools/test_registry.py` (createIf absent for child/contract/kill switch; appended last); the file holding `test_every_deferred_tool_has_a_purpose`; `tests/unit/session/test_attention_notify.py` (peer-only quiet run: notify False, `eligible:false`; user/awaiting_user/notify-requested refused; `internal` runs allowed, including a `session_incident` notice — review R4; a quiet batch whose re-entry then errors still notifies — review R3); `tests/unit/session/test_runtime_completion_announce.py` (no banner after a quiet end); `tests/unit/aida/test_aida_banner_veto.py` unchanged and green (cadence refused, sentence path intact); `tests/unit/test_output_contract.py` (no tool under a contract); the project-guardrail test file (`_project_continuation` not triggered by a quiet turn alone). Real path: a scripted-provider e2e in `tests/e2e` (peer message into an idle session; assert transcript has the pair, no completions row, no notifier call), then kill -9 between result and marker.

**S0b — prompt and guides** (after S0a is deployed): `system.md`, `monitor/GUIDE.md`, `peer-messaging/GUIDE.md`, Aida seed line and regenerated seed artifacts, Aida README line. Tests: the prompt snapshot/budget tests that read `system.md` (grep `tests/unit` for `system.md`), seed manifest checks. Evidence: the adoption probe (20 turns per arm) in the PR.

**S1 — core presentation:** `harness/rows.py` helpers; hide the pair in the relay fold and the TUI fold; fold parity. Tests: `tests/unit/mobile/test_fold_parity.py`, `tests/unit/tui/test_peer_message.py`, `tests/unit/tui/test_hidden_wake_surfaces.py`, `tests/unit/session/test_history_window.py` (pair not subtracted from display pages). Visual: TUI frame via `run_test` + `save_screenshot` per AGENTS.md "Visual validation".

**S2 — UI (local-operator-ui):** the quiet-close rule of §5 (structural close, `quietCloseId` anchor, candidate-only election exclusion, bar-facts summary), hidden paint, group label and facts, parity fixture copy. Tests: `scripts/turn-segments.test.mjs` and `scripts/turn-collapse-model.test.mjs` (no elected answer for a quiet turn; the collapsible gate passes via `quietCloseId`; `settledCloseOf` stops at the quiet close; the `no_reply` call excluded from action counts), `scripts/turn-collapse-behaviour.test.mjs`, `scripts/transcript-reducer.test.mjs`, plus stories in `canonical/turn-collapse.stories.tsx` — including the R1 pin on `[peer][tool][no_reply]` (bar present, completed mark, actions=1 with only `no_reply` excluded, no caption/foot) — (collapsed, open tail, expanded, head-cut, mixed family, no-jitter frames). Designer and UX rounds.

**S3 — native (local-operator-mobile):** group item in `turn-condensing.ts`/`transcript-list.tsx`, latch and jitter pins. Tests: `src/features/session/turn-condensing.test.ts`; design round with real frames.

**S4 — relay web group** (committed, after S2/S3): group over consecutive `peer_message` entries in `mobile/web/src/components/transcript.tsx`, same parity fixture. **S5 — TUI group** (committed, after S4): one group block over consecutive receipts where the TUI today shows one `PeerMessageBlock` per message (`tui/widgets/transcript.py:2670`, replay at `tui/session_presentation.py:1220`); same parity fixture; visual evidence per AGENTS.md "Visual validation".

## 9. Risks to watch in rollout

1. Adoption: models keep writing filler. Metric: share of post-trigger text-only replies shorter than 40 chars (baseline: 143 in 7 days in the Aida session). Fallback: undefer, then add an envelope clause.
2. Over-quiet: the model goes silent where an answer was due. A sender is never told "ignored" because peer text never reached senders; `patience` exists for proactive senders. Watch for operator reports; the refusal rules cover user and notify cases.
3. Project/todo guardrails: confirm no nudge loop (counter exclusion pinned by test).
4. Orphan classification after a crash (see §7).
5. UI structural change (quiet close, answer election, summary carrier): highest regression surface. Gate on the existing jitter and one-answer tests plus the R1 pins; a quiet turn showing no foot and no aggregate line is intended — the bar facts are the summary.
6. Native active-turn assumption: confirm no overlap with the existing "latch" semantics.
7. Wire legality of tool-result-then-user on Gemini and Responses clients (probe).

## 10. Open decisions (recommendation for each; decided, not deferred)

1. Name: `no_reply`.
2. Inputs: none; no `reason` field.
3. Schema: deferred, with a one-line flip if adoption is under 80% in the deferral probe (S0b evidence).
4. Refusal rule (R4): **denylist** — refuse iff `"user" in _run_triggers`, `_has_awaiting_user()`, or `_run_notify_requested`. `internal` runs (peer, hub, job result, incident notice, session_state, catch-up) may end quietly; a refusal is never coerced into a message.
5. `notify:true` deliveries (including Aida cadence): refuse; the sentence path stays. The error arm is never silenced (R3: the notify force applies only when `kind != "error"`).
6. Open todos / pending ask: allowed; guardrails run unchanged.
7. Persist the pair with `useless=True`; no history elision in v1; measure.
8. Text sentinel: none beyond Aida's existing, bounded recogniser.
9. Groups: client-derived, structural, minimum size 2, shared parity fixture, no wire field or capability flag.
10. Pair visibility: hidden in TUI and relay; UI keeps it as a structural close and hides it at paint; not added to `HIDDEN_TOOL_NAMES`.
11. Summary carrier (R1): extended bar facts (actions excluding the quiet call, duration, completed mark); the quiet close is a structural close and the collapsible anchor (`quietCloseId`, not `closingAnswerId`); `electAnswer` excludes it as a candidate only; no foot on a quiet turn.
12. Batch position (R8): quiet end only when the batch's LAST result is the quiet one.
13. Prompt budget (R6): the system.md edit stays near net-neutral with its byte delta measured in the PR; the acknowledgement ban lives in the deferred tool description and the guides.
14. Slices S4 (relay web) and S5 (TUI) are committed, not optional (R2); descope only with explicit operator sign-off.
15. Kill switch: `LOP_NO_REPLY=0` env, no config key.
16. Children, one-shot hosts, contract sessions: tool absent.
17. Peer envelope: no extra text; system prompt carries the rule once.
18. Release: patch-class, each slice its own PR, S0b after S0a is deployed.
