# Design: the ask gate — a forked clearance check before `ask`

Status: PROPOSED — gate for the implementation PRs. Base: `origin/main` @ `2ebb831b4`
(v0.67.17). Author: architect, 2026-10-06.

**Provenance.** Every `file:line` is against this branch's base (`2ebb831b4`), verified
with `sed`/`grep` in the `feat/ask-gate` worktree on 2026-10-06. Paths are
`local_operator/…` unless noted; the desktop repo is `~/local-operator-ui`. Where a
claim rests on an earlier design note it is marked (`ask-nonblocking.md §n`).

**How to read this.** §0 is the problem and the requirements it must satisfy. §1 is the
one-paragraph shape. §2 elaborates each settled choice into a contract (signatures,
grammar, failure semantics). §3 is the hidden-mechanics seam list, per surface, with a
code-in-this-PR / verified-in-this-PR / follow-up disposition for each. §4 is the test
and evidence plan; §5 risks; §6 out of scope and follow-ups; §7 the footprint statement.

---

## 0. Problem

`ask` is the agent's one channel to a human, and since the queued engine it is cheap for
the model: `execute_ask` enqueues and returns a receipt (`tools/builtin.py:26048`;
`asks/queue.py:181`). That flips the failure mode. The model can now reach the operator
for things the operator would rather it settled itself — trivia whose recommended option
is already plainly the best choice, questions the session's own context answers — and
every such ask costs the operator a decision they never wanted to own.

The operator's requirement, in their words across the brief: **a FORKED, hidden check
runs before the ask reaches them**; if the recommended option is plainly best the agent
proceeds; if not clear, the agent resolves it with a subagent and moves forward; the
check is a decision point the agent reflects on, never an automatic block; genuine-user
cases (go/no-go, opinionated architecture, missing access/credentials, preferences/names/
rosters) still reach the operator; **robustness over volume** — never miss a decision
the operator would want, the goal is a material reduction in trivial asks; and the ask
that survives queues byte-for-byte as today.

| # | Requirement | Where this design answers it |
|---|---|---|
| 1 | forked, hidden check; stable prefix cache preserved | §2.1 (no-write side request, same stream fn), §3 (seams) |
| 2 | one question — "are the recommended options clear?"; clear ⇒ proceed | §2.3 (prompt + grammar), §2.4 (notes) |
| 3 | unclear ⇒ resolve with a subagent, then move on | §2.4 (`resolve` note; ONE `task` round), §2.7 |
| 4 | a decision point, NOT a block; re-raise is licensed | §2.2 (fail-open, nothing is ever forbidden), §2.4 |
| 5 | genuine-user cases reach the operator | §2.3 (the prompt carries the list verbatim) |
| 6 | robustness over volume — never miss a decision | §2.2, §2.5, §2.6 (every failure direction is enqueue) |
| 7 | composes with the queued ask; small prompt; no user-visible trace | §2.8 (composition), §3 (hidden mechanics), §7 |

**The name.** The feature is the **ask gate**; one run of it is a **clearance check**;
its outcome is a **verdict** (`clear | resolve | raise`). The runtime word for
"`ask` never reached the queue" is **diverted**.

---

## 1. What ships (one paragraph)

When `execute_ask` runs on the queued arm, it first awaits one forked, off-the-record
request — `Session.complete_clearance`, a sibling of `complete_aside` (no transcript
row, no context append, no event, no card) — that shows the conversation plus the ask's
own questions/options and asks one question: *are the recommended options clear?* The
fork answers with a strict two-line verdict. `clear` and `resolve`: the ask is **not
enqueued**; the tool result is a **decision-point note** telling the model what to do
(proceed with the recommendation, or resolve it with ONE `task` subagent) and licensing
an immediate re-raise. `raise`: nothing changes — the exact same enqueue call runs. Any
error, timeout, or unparseable answer enqueues (fail-open); secret/credential asks skip
the check entirely. A diverted ask's call/result rows remain in the model's context but
carry a result-level marker that every human surface honors, so nothing user-visible
ever showed a check that ran. `LOP_ASK_GATE=0` turns the gate off; the queued engine,
the blocking arm, the queue/store/wire, and the ask cards are untouched.

---

## 2. Settled choices, elaborated into contracts

### 2.1 The fork: `Session.complete_clearance` — a no-write request

A sibling of `Session.complete_aside` (`session/session.py:17793`), placed beside it.
"The fork" here means the request-scoped message copy an aside builds — **not**
`SessionStreamFn.fork` (`model/configure.py:3833`) and **not** a session `/fork`; no
new stream, no child session, no loop interception. It rides the session's own
`_stream_fn`, which is what keeps the cache lineage key (`_cache_lineage_id` →
`prompt_cache_key`, `model/configure.py:5968-5983`).

```python
async def complete_clearance(self, turns: Sequence[AgentMessage]) -> str:
    """One off-the-record request that reads the live conversation and writes nothing."""
```

- **Reads** exactly what an aside reads: `blocks, messages = await
  self._read_only_prompt(turns)` (`session/session.py:5376`) — live system blocks
  (including the frozen-vs-desired epoch check and the system-state delta message), the
  wire-legal snapshot of the live history (`_wire_legal_snapshot`, `session/session.py:
  18473`, which pairs pending `tool_use`s with placeholders so the call is legal mid-batch),
  the appended gateway message, and `bound_replay_payloads` over the result.
- **Writes** nothing: no transcript entry, no `_context.messages` append, no event
  fan-out. That is the same no-trace contract `complete_aside` documents
  (`session/session.py:17825-17836`); a gated ask must leave the conversation exactly as
  it found it, on every path including failure.
- **Request shape** (mirror of the aside, one deliberate delta):

| field | value | why |
|---|---|---|
| `purpose` | `"clearance"` | a new string, honest in the request ledger (`purpose` is open-valued; consumers compare against `"turn"`/`"compaction"` only — `model/configure.py:3701,3761,5838,5915,5990`). It also makes gate runs countable from the ledger for the later "material reduction" measurement. |
| `system_blocks` / `messages` | via `_read_only_prompt` | byte-identical prefix to the turn's last request — the cache READ this design exists to preserve. |
| `tools` | `self._side_channel_tools()` (`session/session.py:9458`) | the tools block is the FRONT of the provider cache prefix; sending `[]` changes position 0 and forces a full re-process (`complete_aside`'s docstring states the measurement). |
| `tool_choice` | `"none"` | same value the aside sends; the wire mapping (Anthropic sends the turn's own choice — cache hygiene against the documented `tool_choice` invalidation rule, measured in `scripts/measure_aside_tool_choice_cache.py`) is inherited unchanged. `complete_clearance` consumes text and usage only, so a tool call in the answer is inert. |
| `context_tokens_hint` | `self._context_tokens_hint` | the session's own TTL hint, same as the aside. |
| `isolated` | absent (`False`) | deliberate, and load-bearing: `isolated=True` strips the session's cache key and puts the call on a cold namespace (`harness/types.py:3204-3208`; the advisor's docstring measures 92.9% cache-read against ~25.6% cost when isolated). The gate must stay on the turn's warm prefix. |
| `replayable` | `True` | nothing is shown until the whole answer is parsed; a stalled read may be discarded and retried whole, exactly as `advise_compaction` argues (`session/session.py:18022-18037`). |

- **Mechanics inherited from `complete_aside`**, including the one bounded retry when the
  answer is a bare tool call (the rejected call is handed back, paired, with `tools=[]`
  for this request only). If the retry still yields no text, the aside raises
  `AsideUnanswered`; `complete_clearance` raises the same class of failure and the
  CALLABLE (below) maps it to fail-open — the gate never fails an ask.
- **No `on_delta`/`on_usage` in v1.** No surface watches the gate run; cost accounting
  is the request ledger's, as for every other request.

### 2.2 The callable: `gate_ask` — total, fail-open, bound like `enqueue_ask`

Declared in `harness/types.py` beside `enqueue_ask` (`:1429`) and `withdraw_ask`
(`:1437`), with the same two-condition binding (`asks.policy.NONBLOCKING_ASK` on AND a
host installed `ask_user` — i.e. exactly where `ask` exists and queues at all), bound in
`Session` where `enqueue_ask` binds (`session/session.py:13449`, binder beside
`_ask_enqueue_callable` at `:8927`):

```python
#: THE ASK GATE'S DOOR (design docs/design/ask-gate.md §2.2). Async — the tool
#: awaits one forked clearance check before the unchanged enqueue. Bound under
#: the same two conditions as :attr:`enqueue_ask` and ``None`` wherever it is,
#: so its presence IS the mode, one fact. TOTAL and fail-open: it returns a
#: diversion mapping or ``None``, and NEVER raises for any policy, provider,
#: parse or timeout condition — ``None`` means "enqueue, unchanged".
gate_ask: Callable[..., Awaitable[Mapping[str, Any] | None]] | None = None
```

`Session._gate_ask(questions, timeout_raw, *, tool_call_id="")` runs, in order:

| # | step | outcome on this branch |
|---|---|---|
| 1 | `asks.policy.gate_enabled()` (`LOP_ASK_GATE`, §2.7) | off → `None` (enqueue) |
| 2 | `self.ask_queue() is None` (defense; unreachable through the tool) | `None` |
| 3 | any question has `secret=True` ("Secret/credential questions skip the check") | `None` — the fork is NEVER called |
| 4 | fingerprint of the normalized ask content is recorded (§2.5) | hit → `None` — **no second check** |
| 5 | `await asyncio.timeout(GATE_TIMEOUT_S)` around `complete_clearance(turns)` | timeout/`Exception` → `None`; `CancelledError` propagates (a turn abort must not be swallowed) |
| 6 | `clearance.parse_verdict(text)` | `None` (unparseable) → `None`; `raise` → `None` (no fingerprint recorded; the queue's own caps see the re-ask) |
| 7 | `clear` / `resolve` | record fingerprint; return `{"verdict", "text" (note, §2.4), "details": {"ask_gate": {"hidden": True, "verdict": ..., "reason": ...}}}` |

`execute_ask` (`tools/builtin.py:26048`) changes in exactly one place: between the
bounds check and the **UNCHANGED** enqueue call (the block at the top of the `if
callable(enqueue):` branch):

```python
    if callable(enqueue):
        gate: Any = getattr(context, "gate_ask", None) if context is not None else None
        if callable(gate):
            try:
                verdict: Any = await gate(params.questions, params.timeout,
                                          tool_call_id=tool_call_id)
            except Exception:  # noqa: BLE001 — a broken gate must never cost the ask
                verdict = None
            if isinstance(verdict, Mapping):
                # THE DIVERSION: no enqueue, no queue entry, no events. The result
                # carries the hidden marker (findings §3) and the decision-point note.
                return _text(tool_call_id, "ask", str(verdict.get("text") or ""),
                             details=dict(verdict.get("details") or {}))
        outcome: Any = enqueue(params.questions, params.timeout, tool_call_id=tool_call_id)
        ...  # byte-for-byte today
```

Failure semantics, stated once: **every** non-divert path — flag off, secret, re-raise,
timeout, provider error, unparseable text, second guard catch — falls through to the
existing enqueue. The only way an ask is lost is a bug in the enqueue path that exists
today; the gate cannot add one. `except Exception` (not `BaseException`) is deliberate:
`asyncio.CancelledError` must keep propagating so an aborted turn aborts. The in-callable
policy and the tool-side guard are both present on purpose — the callable owns the
policy, the guard is the last line so even a contract breach cannot lose the ask.

### 2.3 The gate message and the verdict grammar

One appended user-role message, built by `session/clearance.py` (new module; the
`session/aside.py` precedent: a session-owned prompt with one home). It carries the
prompt and the ask's own questions/options, rendered from the VALIDATED `AskQuestion`s
(so the recommended option is the hoisted `options[0]`, `harness/types.py:1102-1110`).
Keep the fixed prompt small — bullets, not prose:

```
<ask-clearance>
A forked check, OFF THE RECORD: nothing here reaches the user or the conversation.
The agent is about to raise this ask. Decide, for the ask as a whole:

The ask:
Q1: <question> [multi-select]
  1. <label> (recommended) — <description>
  2. <label> — <description>
  ...

Verdicts:
- clear — the recommended option is plainly the best choice here, for ALL questions
  (with no recommendation, the context plainly settles them). The agent proceeds.
- resolve — not clear from context, but not the user's call either: the agent should
  resolve it with ONE `task` subagent and proceed.
- raise — the decision is genuinely the user's: critical go/no-go decisions;
  opinionated architectural choices; missing access or credentials; a preference,
  name, or roster only they can state. A destructive or irreversible action is
  NEVER cleared. If ANY question is the user's, raise.

Answer EXACTLY two lines, nothing else:
VERDICT: clear|resolve|raise
REASON: <one line, <=200 chars>
</ask-clearance>
```

- **Parse rules** (`clearance.parse_verdict(text) -> str | None`, pure, unit-pinned):
  split the answer into lines; strip each; a verdict line must FULL-LINE match
  `[*_>#\s]*VERDICT:\s*(clear|resolve|raise)[*_`\s]*` (case-insensitive; decoration
  tolerated); **the LAST matching verdict line wins** (reasoning models deliberate in
  the open); no match → `None` → enqueue. The reason line matches
  `[*_>#\s]*REASON:\s*(.*)` and is truncated to 200 chars; a MISSING reason does not
  force fail-open (it degrades the diagnostic, not the route — the verdict is the only
  routing input). An empty/truncated answer has no verdict line → `None`.
- **Atomicity:** one verdict per ask, never per question. Clear requires all questions
  plainly settled; any genuine-user question raises.
- The prompt carries the genuine-user list **verbatim as bullets** — that list is where
  "never miss a decision the operator would want to own" is enforced — alongside the
  destructive/irreversible rule and the secret note. Size discipline: the fixed frame
  is ~15 lines; the ask block is the model's own data.

### 2.4 The notes (tool-result copy, at least in outline)

Lives in `session/clearance.py` beside the grammar that produces it. Both notes MUST:
name that no question was put to the user (no "the user said" register anywhere — the
"receipt is not consent" discipline, `ask-nonblocking.md §9`); carry the instruction;
license the re-raise explicitly.

- **clear:** *"`[Ask clearance]` No question was put to the user. A forked check read
  the conversation and found your recommended option plainly the best path — proceed
  with it and state the assumption where it matters. The user was NOT asked and did not
  answer. If new information makes this genuinely their call, call `ask` again: a
  re-raise of this same question reaches them without another check."*
- **resolve:** *"`[Ask clearance]` No question was put to the user. The answer is not
  clear from context and it is yours to resolve: spawn ONE `task` subagent with the
  relevant expertise (scout/research, architect, designer/UX — as the question needs),
  decide on its answer, and carry on. The user was NOT asked and did not answer. If
  after that the decision is still genuinely theirs, call `ask` again: a re-raise of
  this same question reaches them without another check."*
- Both embed the check's `REASON` line (bounded 200 chars) after the instruction —
  it explains the verdict to the model and keeps the diversion auditable. Final copy is
  the implementation PR's; these drafts fix the contract (no user-attribution, the
  re-raise licence, the ONE-subagent instruction).
- A `raise` verdict produces NO note — it enqueues, and the model gets today's receipt.

### 2.5 The honor rule: per-content fingerprints

- **Normalization** (`clearance.fingerprint(questions)`): per question, in order —
  `" ".join(q.question.split())` plus the option list as
  `[" ".join(o.label.split()), " ".join(o.description.split())]` pairs. `id`,
  `recommended` and `multi` are EXCLUDED: the user-facing content is question + options,
  and a re-raise with a changed recommendation or id is the same content. `secret`
  questions never gate; they never fingerprint. Digest = sha256 over a canonical
  `json.dumps(..., separators=(",", ":"))`.
- **Record:** `Session._ask_gate_diverts: OrderedDict[str, float]` (fingerprint → epoch),
  bounded by `GATE_FINGERPRINT_CAP = 64` (LRU — a hit or a record moves the entry to the
  end; overflow drops the oldest). **Window: session lifetime, no TTL.** The skip-the-
  check outcome is ENQUEUE — the safe direction — so a generous window cannot miss a
  decision; eviction (65th distinct divert) costs one redundant check, nothing else.
  In-memory on the Session: the gate only ever runs in the session's own runtime, so the
  fingerprint does not need a durable home; a restart means one extra check for a
  re-raised ask — never a lost ask.
- **The rule:** re-raise of recorded content queues with **no second check** (terminal to
  the gate — the model's second attempt is a decision the operator can override);
  materially different content is a new ask and gets one check. Only `clear`/`resolve`
  diverts record; a `raise` does not (the queue and its caps own the re-ask path there).

### 2.6 Timeout and cancellation

`GATE_TIMEOUT_S = 30.0` (policy constant, initially uncalibrated — flagged in §5). The
bound wraps the whole check; expiry → `TimeoutError` → `None` → enqueue. Rationale: the
fork is one short request reading the turn's warm prefix; a human attention budget for a
turn step is longer than 30 s, and a fail-open budget's wrong side costs extra asks or
extra latency — never a missed decision. Cancellation of the enclosing turn propagates
(§2.2) — an aborted turn writes nothing and the queued ask never existed.

### 2.7 Kill switch

`asks/policy.py` gains, beside `NONBLOCKING_ASK` (`:75-80`) and `enabled()` (`:83`):

```python
ASK_GATE: bool = os.environ.get("LOP_ASK_GATE", "").strip().lower() not in {"0", "false", "no", "off"}
def gate_enabled() -> bool: return ASK_GATE
```

Same direction and typo discipline as the queue switch (absence/typos leave the shipped
default — ON). Read at import like the queue flag; tests monkeypatch the attribute.
`LOP_ASK_GATE=0` skips steps 5-7 in §2.2 — the fork is never called, the fingerprint is
never consulted — so gate-off is a zero-cost, zero-token path whose ask behavior is
today's.

### 2.8 Composition guarantees

- **No changes** to `asks/queue.py`, `asks/store.py`, the wire events/`frontend_state`
  fields, the ask cards (`ask_picker.py`, the bar, `asks/render.py`'s existing
  functions), the relay ops, or the blocking arm — `LOP_ASK_NONBLOCKING=0` stays
  byte-for-byte today. The queued arm's enqueue call itself is unchanged; the gate is
  composed AROUND it.
- The only new runtime surfaces: the `gate_ask` ToolContext field (in-process), the
  `clearance` module, `Session.complete_clearance` + the fingerprint store, the policy
  constants, and the hidden-marker seams in §3.
- **Footprint-ladder rung:** below rung 1 — zero new schema. No new tool, no new
  `AskParams` field, no new wire event or field, no config key. The check is threaded
  through the existing `ask` tool's execution path and is invisible to the model's
  schema; `/context`'s tool block does not move (asserted in tests, §4).

---

## 3. Hidden mechanics — the exact seam list

**The invariant:** a diverted ask leaves NO user-visible trace on ANY surface — no card,
no row, no receipt, no expandable tool trace — in live view, on reconnect, or on replay.
Its call/result rows persist and stay in the MODEL's context (the model must read the
note); only human surfaces skip them. The queued ask that survives renders exactly as
today.

**One flag first (elaborated, and why).** The verdict cannot exist when the ask call's
first paint frames fire: `tool_call_compose`/`tool_execution_start`
(`harness/types.py:1929-2010`) are emitted by the loop BEFORE `execute_ask` runs, so a
result-level marker cannot be read at `tui/app.py:53070`/`:53201` — there is no result
yet. The settled intent ("diverts create no card, no tool trace") is therefore kept by
**settle-only ask rows**: while the session's queued engine is live, an `ask` call gets
NO live row while composing/running; its one row is created at SETTLE — the receipt for
a raise, nothing for a divert (marker read there). This is a mechanism elaboration of
the briefed intent, flagged rather than silent; the alternative (mount then remove)
flashes a multi-second gate on every surface and was rejected. The two named paint
sites still receive code — they are where the suppression lives — and the marker is
honored at every settle/replay seam below. Visibility-timing consequence for a RAISED
ask: its row appears when the ask is raised, not when the model starts writing it
(under the gate the dictation window was not yet an ask; with the gate off the window is
sub-frame). Recorded here deliberately; QA checks the raised-ask frame.

**Attach seeds and frontend_state.** A re-attaching viewer is seeded from
`frontend_state`'s retained live events (the capped `tool_execution_end` rows,
`session/frontend_state.py` "Cap on retained `tool_execution_end` rows in the seed").
For a diverted ask the retained end carries the marker, so every seed arm must apply the
same skip the live stream does — the TUI/viewer seed fold (row 2) and the desktop feed
(row 6) — and the QA matrix's attach/reconnect cases exercise exactly this path.

**The marker.** The divert result carries
`details = {"ask_gate": {"hidden": True, "verdict": ..., "reason": ...}}`. It persists
to the stored row as `payload.provider_payload.details` — `Message.tool_result`
(`harness/types.py:480-527`) writes `details` into `provider_payload`, and
`encode_message_payload` (`session/transcript.py:398`) serializes it; the provider wire
builders never ship `provider_payload`, so the marker can never reach a model or a
provider. New shared predicates in `harness/rows.py`, beside the hidden-tool pair
(`:344-417`), keep ONE decision for every consumer:

| name | reads | used by |
|---|---|---|
| `is_ask_gate_divert_details(details)` | the details mapping (live event or stored payload) | every other predicate below |
| `is_ask_gate_divert_row(row)` | stored `{type, payload}` rows (`payload.provider_payload.details`) | desktop rows filter; any stored-row consumer |
| `is_ask_gate_divert_message(message)` | rendered `Message`s (`provider_payload["details"]`), replay-side twin | history window; folds' up-front indexes |
| `ask_gate_diverted_call_ids(messages)` | the set of `tool_call_id`s whose result carries the marker | fold call-chip skips (one helper, no per-fold drift) |
| `is_settle_only_ask(tool_name, *, queued_engine)` | name + mode | live paint seams (deferral switch) |

**Per-surface seam table** (C = code in the implementation PR; V = verified in it;
F = explicit follow-up; N = structurally absent, nothing to do):

| # | Surface | Seam (file:line) | Disposition |
|---|---|---|---|
| 1 | Core marker + predicates | `harness/rows.py:344-417` block; `tools/builtin.py:26048` result details | **C** |
| 2 | TUI live paint | `tui/app.py` `on_tool_composing` (`:53056`, gate at `:53070`): suppress an `ask` mount while the queued engine is live (register nothing — the source-suppression rule the patience arm documents); `on_tool_started` (`:53194`, gate at `:53201`): same belt; `on_tool_ended` (`:53304`): marker → drop; else → mount the settled receipt row (same row replay paints). In-flight restore `_mark_pending_tool_rows` (`:14159`, skip site `:14513`): skip ask calls while gating. Mode read: `self._session.ask_queue() is not None` when the app owns the session (construction is side-effect-free and already happens at turn binding); a viewer uses the presence of the `asks` wire field (`session/frontend_state.py:2963-2965`: presence ⇔ queued asks live in the owner's process). Unknown mode (an un-negotiated mixed build): keep today's mount and drop on the settle marker (flash residual, §5). | **C** |
| 3 | TUI replay fold | `tui/session_presentation.py` — per-call skip beside `is_hidden_tool_call` (`:1612`; the fold holds `results` keyed by call id); skip a call whose result carries the marker, and never settle a row for it. | **C** |
| 4 | TUI display pages (owner side) | `session/history_window.py:545,:712` (`is_hidden_tool_message` sites): add the message predicate; the wake-fire id-set helper (`_hidden_wake_entry_ids`, `:613-637`) is the shape to mirror (owner-side filtering so every viewer build receives clean rows). | **C** |
| 5 | Desktop (rows path) | `server/utils/desktop_sessions.py::visible_transcript_rows` (`:1191-1215`; called from `history` at `:3354`): extend the filter with `is_ask_gate_divert_row`. Server-side is the seam that covers every desktop build ("the client reducer has no filter of its own"). | **C** |
| 6 | Desktop (live trace) | the desktop paints tool rows from live `tool_execution_start/_end` frames (`~/local-operator-ui` `transcript-reducer.ts:3279/:3397`): needs the settle-only rule + marker drop, client-side. NOT REPO-LOCAL — see §6/§5; the history path (5) is clean meanwhile. | **F** |
| 7 | Desktop (child trajectories) | `desktop_sessions.py:5958` serves child pages VERBATIM by design ("fold them through the same reducer"). Subagents have no ask hook so cannot ask; an `exec --control` child CAN — its divert rows are filtered only by the client reducer rule in 6. | **F** (rides 6) |
| 8 | Mobile / relay projections | `mobile/projection.py`: the history/attach fold (up-front index like `settled` at `:1240-1245`, call-chip skip beside `:1509-1520`); the live fold (`ProjectionFold`, `:1623`; start arms `:1986`/`:2547`): ask rows are settle-only, marker dropped — one refusal in `_tool_row` (`:3237-3262`) the way the hidden-`send` refusal lives there ("compose, start, update and end all get it from one place"). Mode read: the serving layer that builds the fold owns the session (`session/runtime/serving.py:843`). | **C** |
| 9 | Mobile web client (in-repo) | `local_operator/mobile/web/` renders `TranscriptEntry[]` — nothing reaches it for a divert (8 hides server-side); no client change expected. | **V** |
| 10 | Mobile app (separate repo) | renders the same server-built projection; inherits 8 by construction. | **V** |
| 11 | Headless print | `local_operator/headless_print.py`: start branch (`:286`) suppressed for a settle-only ask; end branch (`:305`): emit the raised row only. `PrintRenderer.attach` holds the session (`:214-222`), so the mode read is direct. The JSON (`json_mode`) stream is a MACHINE surface: it keeps the frames; the marker rides `details` so supervisors filter (the `FAULT_KEY` precedent). | **C** |
| 12 | Search / find | tool rows are never documents ("``transcript_index``'s own rule", `session/transcript_find.py` docstring) — structurally absent. | **N** |
| 13 | Subagent panels | children have no ask hook (existing guarantee, `build_ask_tool` docstring `tools/builtin.py:25983+`); `exec --control` children ride 7. | **V** |

**What "code in this PR" means for 2/3/4/8/11:** the shared work (predicates, deferral
switch, the settle-mount path) is core-repo, so it lands here; the QA matrix (§4)
exercises each seam on a REAL runtime, and the desktop live trace (6) plus desktop child
pages (7) are the only dispositions deferred across a repo boundary.

---

## 4. Test / evidence plan

**Unit (pytest, scoped):**

- `clearance.parse_verdict`: exact two-line answers; deliberation before the verdict
  (last-match wins); decoration (`**VERDICT:** clear`); case-insensitivity; missing
  verdict → `None`; missing reason → verdict kept; truncated/empty → `None`;
  reason >200 chars truncated.
- `clearance.fingerprint`: whitespace collapse; option labels/descriptions included;
  `recommended`/`id`/`multi` excluded; order significance; same content after
  re-serialization.
- `Session._gate_ask` totality table (fake `complete_clearance`): flag off → `None` and
  the fork is never called; secret question → `None`, fork never called; fingerprint hit
  → `None`, fork not called again; timeout → `None`; exception → `None`; `raise` → `None`
  and nothing recorded; `clear`/`resolve` → mapping + record; `CancelledError`
  propagates. LRU: 65th entry evicts the 1st.
- `Session.complete_clearance`: no-write contract, mirroring the aside tests — transcript
  unchanged, `_context.messages` unchanged, no events; request shape: `purpose ==
  "clearance"`, `isolated is False`, tools = side-channel array identity,
  `tool_choice == "none"`, and `prompt_cache_key == session._cache_lineage_id` on the
  captured request (cache lineage).
- `execute_ask`: ordering — invalid params/bounds error never call the gate; a diverted
  mapping returns `_text` with the marker details and the enqueue callable is NOT
  called; a `None` verdict still enqueues; a gate that RAISES still enqueues; blocking
  arm (`context` with no `gate_ask`) byte-for-byte. Schema: `build_ask_tool`'s schema and
  `DEFAULT_TOOL_NAMES` unchanged (the `/context` tool block does not move).
- `rows` predicates: all three stored/rendered shapes; negatives (a normal ask row,
  patience rows, the wake marker) unaffected.
- TUI (`tests/unit/tui`): ask call under the queued engine mounts nothing at
  composing/started; settle mounts the receipt when unmarked; settle drops when marked;
  replay fold skips the chip and adds no row; `_mark_pending_tool_rows` skips;
  working-line/activity does not linger on a diverted ask.
- Projection: live fold start→no row / end(marked)→no row / end(unmarked)→row; history
  fold chip and row skips; `SessionProjection` byte-identical for a probe session
  without diverts.
- Desktop rows: `visible_transcript_rows` drops a marker row and keeps a patience row.
- Headless print: diverted → zero human lines; raised → today's line; JSON stream
  contains the marker.
- Policy: `LOP_ASK_GATE` parse table (absent/typo/`0`/`false`/`no`/`off`).
- Pins stay green: the ask-nonblocking suite (`tests/unit/asks/*`, `test_ask_fleet.py`,
  `test_ask_queue_surface.py`) and the `LOP_ASK_NONBLOCKING=0` blocking-arm suite.

**Real-path QA (the operator's three scenarios; QA drives the real app, isolated config
dir, synthetic CMUX ids per the team gate):**

1. **Trivial ask answered with no user.** A real session whose fake/streamed provider
   answers the clearance fork with `VERDICT: clear`. Drive the real `execute_ask`.
   Assert: no new `asks.jsonl` row; no queue entry; the transcript's residual call/result
   carry the marker; the TUI shows no card/row at any frame (SVG stills: before, during,
   after); `visible_transcript_rows` and the projection entries omit both rows; headless
   stdout is silent for the call; the model's next request contains the note (context
   keeps it).
2. **Genuine ask queued untouched.** Fork answers `raise`; assert the receipt text,
   details, queue row, ask surfaces and wire fields are byte-comparable to the same ask
   run with `LOP_ASK_GATE=0`.
3. **Nothing visible in the transcript.** As 1, on BOTH surfaces per the refinement: the
   TUI (transcript paint + a `/resume` replay) and the mobile relay/web row path
   (projection-driven — drive the relay and read the served rows), plus the desktop rows
   path via the server API. Loading/empty/error states of those surfaces show no ask
   artifact.
4. **Kill switch.** `LOP_ASK_GATE=0` + `LOP_ASK_NONBLOCKING` unset: the fork is never
   called (assert on the provider-call counter), the ask queues. `LOP_ASK_NONBLOCKING=0`:
   the blocking arm's pinned suites pass untouched.
5. **Honor rule.** Divert → re-raise the same content → enqueues with NO fork call
   (assert); rephrase → one new check.
6. **Cache evidence.** Extend `scripts/measure_aside_tool_choice_cache.py` (verified to
   exist) with a clearance arm: on a warm session, the clearance request's `cache_read`
   covers the full prefix and `cache_write` is only the appended gate block — the
   aside/advisor measurement pattern (`docs/evidence/…` envelope). The unit half:
   byte-identity of the clearance request prefix against the turn's last request up to
   the appended message.
7. **Gate-quality smoke (small, fixed).** A handful of synthetic asks through the real
   fork — trivial-with-recommendation, unclear, each genuine-user class, a destructive
   ask, a secret ask — asserting the expected verdict family (a prompt-quality guard, not
   a benchmark).

**Visual evidence:** SVG frames via the repo's capture recipes (`scripts/ask_shot.py`
family, `run_test` + `save_screenshot`) for the transcript around a divert, before/after;
the mobile/web and desktop stills ride their own lanes (§6) for the follow-up.

---

## 5. Risks and watch items

- **The desktop live trace** (§3 row 6) is the one user-visible gap this repo cannot
  close alone: until the UI change lands, a diverted ask paints a running row during the
  gate and, at settle, a row whose text is the decision note. History/reopen is clean.
  Watch: do not flip expectations of desktop correctness from this PR's evidence; the
  follow-up is recorded in the PR thread.
- **Flash on mixed-build viewers** (§3 row 2 fallback): a viewer that cannot read the
  owner's mode keeps today's mount and drops on the settle marker — a brief trace, not
  persistent. Cheap to revisit if the fleet's mixed windows are wider than assumed.
- **Mid-gate abort residual**: a turn aborted while the fork is in flight never resolves
  a verdict; the loop's synthetic pairing settles the call `interrupted` on REPLAY
  (nothing was shown live). Truthful bookkeeping ("the agent tried to ask, the turn
  died"); accepted, listed for the reviewer/QA to weigh.
- **`GATE_TIMEOUT_S = 30` is uncalibrated**: a policy constant, fail-open both ways.
  Post-rollout: ledger `purpose="clearance"` latency distribution; adjust once.
- **Verdict quality is model judgment**: fail-open caps the cost of a bad verdict at one
  extra ask; watch the honored re-raise rate (fingerprint hits) as the signal that the
  gate is being fought, and the divert:raise ratio for the "material reduction" claim.
- **Cost furniture**: one extra request per gated ask (cache read of the prefix + small
  write). If a session's asks are already rare, the gate's savings are attention, not
  tokens — the memo does not claim a token win.

---

## 6. Out of scope / follow-ups (explicit)

1. **Desktop UI (local-operator-ui) — the live trace** (§3 row 6, §5): settle-only ask
   rows + marker drop in the reducer; then the desktop child trajectory (row 7). Reads
   only the core marker + predicates shipped here; no core change needed to land it.
2. **Desktop/mobile visual evidence lanes**: the UI repo's Storybook/capture pipeline
   for the divert states rides that repo's PR (its manifest re-fold).
3. **Gate prompt calibration** (optional): if QA's smoke shows missing recommendations
   at raise time, a nudge line in the tool description or guide — deliberately NOT in v1
   (zero schema movement).
4. **Durable gate telemetry** (optional): v1 measures from the request ledger and ask
   receipts; a dedicated counter is a later ask.
5. **Per-question granularity** (rejected for v1): the queue is whole-ask atomic;
   clearing "just the clear ones" would need queue semantics this design must not touch.
6. **Blocking arm**: nothing. `LOP_ASK_NONBLOCKING=0` keeps every pinned behavior.

---

## 7. Footprint statement

**Zero new schema.** No tool added, no parameter added, no wire event/field added, no
config key added. New code is: `session/clearance.py` (prompt/grammar/parse/fingerprint/
notes, pure), two `Session` methods + one bounded in-memory dict, a `ToolContext` field
(in-process), three policy constants + the env kill switch, and the hidden-marker seams
(§3) — shared predicates plus per-surface skips. The `ask` tool's schema, the tools
array, and therefore the cached prefix are byte-identical to today; the gate runs
INSIDE a tool call the model already made, invisible to it and to the schema ladder
(AGENTS.md "The tool-surface footprint ladder" — this change exercises no rung and adds
no tax).

---

### Flagged elaborations vs the settled ack (kept the intent; stated because a reviewer
must see them)

1. **Live-paint mechanism**: the marker cannot be read at `tui/app.py:53070/:53201`
   (no result exists mid-dictation); those sites get the mode-aware suppression and the
   marker is honored at settle + every replay/fold seam. §3 "One flag first".
2. **Desktop split**: server rows filter here; the live trace is a cross-repo
   follow-up. §3 rows 5-7, §6.
3. **Fingerprint window**: session-lifetime LRU (no TTL) — §2.5 argues why (skip ⇒
   enqueue is the safe direction).
