# The Monitor tool: delta-watch over repeated read-only calls

Status: design + interface contract, 2026-09-28. Author: architect (deepseek/deepseek-flash, lopdev).
Base: `origin/main` @ `e3b117f34`. All `file:line` references are against that tree.

This file is the contract the implementation slices code against. Reviewers and
QA check the implementation against this document; where it says "decide", the
decision is made here and the rationale is recorded with it. It mirrors
`docs/design/classification-layer.md` in shape and rigour, and it **reuses**
that layer plus the wake stack rather than reimplementing either.

---

## 1. Why, and the mental model

Today, "watch X and tell me when it changes" has no primitive. The two nearest
answers are both wrong for it:

- **Poll in a loop.** A turn that re-runs the call, sleeps and re-runs it burns
  a full model turn per iteration; the harness actively refuses the long
  foreground sleep shape (`tools/sleep_guard.py`, `refuse_args` on the bash
  tool, `builtin.py:4819`).
- **Arm a wake.** A wake is a clock, not a change detector: it fires on
  schedule whether or not anything moved, and every fire is a full turn
  (`harness/wake.py`, `wake_types.py:MIN_WAKE_INTERVAL_MS = 60_000`). "Tell me
  when someone replies" on a thread that replies once a day is 1,440 turns of
  "nothing new" per day, or a daily wake that is wrong within the hour.

A **monitor** is a standing question to a read-only tool call — *"has your
answer changed?"* — answered by the harness on an interval **with no model in
the loop**. The harness re-runs the same call, normalizes the output, compares
it against the last snapshot, and only when something differs does it decide
whether the difference is material; only then does a turn happen. Ticks that
find nothing cost no model tokens, inject no context, and write no transcript
row.

The end-to-end pattern this exists for (the Slack/Telegram case in the brief):
a user says "keep an eye on this thread and reply if someone asks something" →
the agent arms a monitor on the thread-reading call at 60 s → every tick is one
read-only call; when a reply appears, the harness verifies it is material and
delivers a bounded delta into the conversation; the agent reads it and posts.
Ticks with no reply are silent and cost nothing.

## 2. Scope of this change

In scope:

1. `local_operator/monitors/` — spec model, in-session scheduler, derived
   index + per-monitor state, diff engine, read-only evaluator.
2. The agent-facing `monitor` tool (`create` / `list` / `cancel`), a createIf-
   gated builder (rung 3 of the tool-surface footprint ladder).
3. The materiality classifier gate — one typed question through the existing
   `local_operator/classification/` service (cascade, config, breaker,
   timeout), fail-open, one new public method on the service (`decide`).
4. Delivery of bounded deltas into the conversation, with the wake stack's
   idle/busy/downtime semantics.
5. Persistence: transcript `monitor_schedules` entry + derived import-light
   index + snapshot state, with the self-healing contract of `wakes/store.py`.
6. Turn-origin-aware completion notifications (`notify` control parameter),
   wired through the existing attention/notification machinery.
7. The `guide://monitor` playbook, prompt/instruction wording, config keys,
   and the awareness surfaces inventory (design only; the desktop-UI and TUI
   *implementation* lands against the insertion points enumerated in §12).

Out of scope, deliberately:

- **Cold-session engagement.** Monitors tick only while the session is hosted.
  The boundary is documented in §10.4 with numbers, per the brief's
  "document the boundary explicitly" branch.
- **`eval` / arbitrary Python as a monitor target.** Rejected in v1; §6.6.
- **A per-turn "notify me now" hatch** beyond the delivery's own parameter; §14.5.
- **Cross-session dedupe** (two sessions may watch the same URL); §20/§21.
- **A second notification system** of any kind; §14 wires the existing one.

## 3. The wake-vs-monitor split (normative)

| | `wake` | `monitor` |
|---|---|---|
| Purpose | re-engage later **on time**: a reminder or scheduled check-in | **notice when something changes** |
| Trigger | wall clock (`in` / `at` / `every`) | output of a read-only call differing from the last snapshot |
| Fire cost (no change) | a full turn per fire, always | ~zero: one tool call, no model, no injection, no transcript row |
| Time knowledge | required ("at 15:00", "every 2h") | none needed ("when the PR status flips") |
| Floor / cap | 60 s floor; 16 schedules (`wake_types.py`) | 30 s floor; 8 per session (§11) |
| Delivery | the wake message verbatim | a bounded delta diff |
| Failure mode to avoid | missed fires | notification storms from noisy diffs |

The one sentence the prompts carry: **`wake` = tell me at a time; `monitor` =
tell me when something changes.** §13 has the candidate wording.

## 4. Tool surface & schemas (R1)

### 4.1 Footprint rung: 3 — a createIf-gated tool

`AGENTS.md`, "The tool-surface footprint ladder": every core tool ships its
schema on **every** API request, so a new tool must take the highest (least
footprint) rung that solves the problem. `monitor` lands on **rung 3**: a
factory in `TOOL_BUILDERS` (`tools/registry.py:37-50`) that returns `None`
without a prerequisite — exactly the shape `build_wake_tool` uses
(`build_wake_tool` returns `None` when `context.wake_scheduler is None`,
`builtin.py:11818-11823`). A session with no monitor scheduler advertises
nothing and pays nothing.

Rungs considered and why they lose:

- **Rung 1 (extend an existing tool)** — the only fit would be new `wake` ops
  (`wake {op:"watch", …}`). Rejected: the two capabilities have different
  parameter sets (call spec + args + diff knobs vs message + schedule), and
  they must stay separately teachable — the whole point of the split in §3.
  Search also shows no existing tool whose semantics this is a variation of.
- **Rung 2 (skill + bash)** — impossible: the re-run/diff/classify/deliver
  loop is harness machinery (tool execution without a transcript row, bounded
  state, classifier access, wake-style delivery). A shell loop would be a
  second, worse scheduler.
- **Rung 5 (ungated core tool)** — no: not every session needs it; a session
  without a monitor scheduler cannot use it at all.

### 4.2 Measured footprint (method in §17)

`monitor` tool wire object (`{name, description, input_schema}` as sent on the
Anthropic leg, `providers/clients.py:3725-3733`):

| tool | chars | cl100k | billed @2.78 c/tok |
|---|---|---|---|
| `wake` (shipped, for scale) | 1,551 | 487 | ~558 |
| `monitor` (this design) | 2,133 | 620 | ~767 |
| `wake` + the new `notify` field (§14) | 1,723 | 535 | ~620 |

~767 billed tokens is the family price of a two-vocabulary gate plus eleven
params (the `secret` tool's own precedent: 828 → 556 after its rationale moved
to `guide://credentials`, `scripts/bench_context_budget.py` docstring). The
same lever is available here if a later measurement demands it: move the
`sort_lines`/`ignore` advice into `guide://monitor` (≈ 95
billed, measured by subtraction) or drop `until` (≈ 56). We do not trim now; the numbers are recorded so
a future decision is measured, not guessed.

### 4.3 `MonitorParams` (the tool schema)

```python
class MonitorParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["create", "list", "cancel"] = Field(
        description="create: arm; list: show; cancel: remove.")
    name: str | None = Field(default=None, description="Short label, e.g. 'slack-thread-42'.")
    tool: str | None = Field(
        default=None,
        description="Read-only tool to re-run: bash, web_search/web_fetch, read, grep, an mcp__ tool.")
    arguments: dict | None = Field(default=None, description="Exact arguments for that tool call.")
    every: str | None = Field(
        default=None, description="Check interval: '30s'|'60s'|'5m'|'1h' (floor 30s, default 60s).")
    until: str | None = Field(default=None, description="Stop time (ISO datetime); omit = durable.")
    description: str | None = Field(
        default=None, description="What to watch for — shown in list and on delivery.")
    notify: bool | None = Field(
        default=None, description="Notify when a delivered change completes (default false).")
    sort_lines: bool | None = Field(default=None, description="Unordered line compare (default false).")
    ignore: list[str] | None = Field(
        default=None, description="Regexes dropping lines before diffing (max 8).")
    id: str | None = Field(default=None, description="Monitor id (cancel; from list).")
```

House rules mirrored from `WakeParams` (`builtin.py:11663-11689`):

- Durations are strings (`'30s'|'60s'|'5m'|'1h'`, compound terms allowed and
  summed — `parse_wake_duration`'s grammar, `harness/wake.py:97-104`), not raw
  seconds: the interval is a wake-style duration everywhere else in the
  harness, and the model already knows that grammar from `wake`.
- `tool` + `arguments` are validated **at arm time** against the same schema
  the loop validates calls with, and against the read-only evaluator (§6).
- `id` is the cancel handle; ids are `m1…`, allocated like wake ids
  (`w{n}`): stable per session, never reused after cancel (the `wakes/arm.py`
  docstring records the CLI's reuse bug that came of `len(existing)+1`).
- No `edit` op. The wake tool deliberately has none either — "adding a fourth
  op is a change to what the model is offered rather than a change to how a
  wake is written" (`builtin.py:11794-11798`) — so monitor mirrors
  create/list/cancel. Editing a monitor is cancel + create; a reactivating
  re-arm is covered in §11.4.

### 4.4 Tool card

```python
AgentTool(
    name="monitor",
    label="Monitor",
    describe_approval=_describe_monitor_approval,   # names the call being armed, like wake's
    description=(
        "Watch a read-only call for changes (create/list/cancel): re-runs it every "
        "interval and wakes you only when the output differs."
    ),
    parameters=MonitorParams.model_json_schema(),
    approval_tier="write",          # arms unattended future calls; each material delta can open a turn
    call_approval_tier=lambda args: (  # reading the list must not prompt (the hub precedent,
        "read" if str(args.get("op") or "") == "list" else "write"  # builtin.py:20988-20996)
    ),
    concurrency="exclusive",        # create/cancel rewrite the whole list; two concurrent calls would lose one
    interruptible=False,
    execute=execute_monitor,
)
```

`build_monitor_tool(context)` returns `None` unless
`context.monitor_scheduler` is attached — the new `ToolContext` field,
mirroring `wake_scheduler` (`harness/types.py:1177`). Two plumbing points that
are easy to miss and are part of this contract:

- `SESSION_CAPABILITY_TOOLS` (`session.py:544`) gains `"monitor"`, or the
  tool reaches `create_tools`' factory context — which has no scheduler — and
  never reaches a real session (the merge at `session.py:4105-4121` exists
  precisely for session-gated tools).
- Subagents **prune** it, the same way `wake` is pruned: "a child session ends
  after one prompt, so a wake armed there would be […]" (`harness/subagent.py:
  2366-2386`, `drop = {name for name in merged_in if name == "wake"}`). The
  monitor joins that drop set.

### 4.5 Ops, results, and error style

Mirrors `_wake_list` / `_wake_create` / `_wake_cancel`
(`builtin.py:11701-11775`): actionable sentences, the id and the instant as
structured `details` as well as in the sentence, an empty list reported with
`useless=True`.

- `create` → `Armed monitor 'loom-pr-1710' (m1): bash `gh pr view 1710 --json
  state,reviews` every 60s, durable. First check in ~2s captures the baseline;
  you'll be told only what changes.` On an identical spec (§11.4):
  `Monitor 'loom-pr-1710' (m1) already watches that call every 60s.`
- `list` → one row per monitor: id, name, every, last check age, next due,
  health; `DISABLED after 5 failures (last: …) — re-arm to reactivate.` Empty:
  `No monitors.`
- `cancel` → `Cancelled monitor 'm1'.` Unknown id:
  `No monitor with id 'm9' (known: m1, m2)` — the wake cancel wording.
- Errors: `'create' requires 'tool' and 'arguments'`; `'cancel' requires the
  monitor id (see monitor list)`; read-only refusals per §6.7.

## 5. Execution semantics (R2)

### 5.1 Where the loop lives

A new `local_operator/monitors/scheduler.py::MonitorScheduler`, attached to the
`Session` at construction exactly where the wake scheduler is
(`session.py:3110-3121`) and persisted through a `persist` callback. It is
**in-process only**: no supervisor, no daemon, no cold engagement (§10.4).

The scheduler preserves the `WakeScheduler` load-bearing properties
(`harness/wake.py:661-677`) because they were paid for once already:

1. **A single armed asyncio timer**, re-armed after every pump, with
   `MAX_ARM_MS`-style re-check ticks so sleep/clock skew is absorbed by
   re-reading the wall clock rather than one long timeout.
2. **`dispose()` cancels the armed handle and every in-flight check task** —
   a pending tick must never keep the event loop alive, and a re-arm must not
   orphan a previous pump (the wake scheduler's `_tick_tasks` set exists for
   exactly that failure).
3. **`needs_rearm`** for construction without a running loop; the session's
   async init re-arms by calling `pump()` once.
4. **A write lock** around pump/update so an arm landing inside a pump cannot
   be overwritten by the pump's pre-update snapshot (the wake scheduler's
   `_write_lock` rationale, `harness/wake.py:700-704`).
5. **A delivery that throws still advances** — a broken check must not become
   a hot loop.

### 5.2 The tick algorithm

For each due monitor, in `next_due_at` order (pump pass):

1. **In-flight guard.** If the previous check is still running, the tick is
   skipped (counted `skipped_overlap`, debug log). No overlap, ever.
2. **Re-validate.** The tool still exists in the session's tool set and its
   effective tier is still `read` (§6.8). A failure here skips the tick and
   counts toward the failure ladder (§11.3) with the reason.
3. **Run** the call through the session's own tool executor — the same
   `AgentTool.execute` the loop calls, under `values.monitor.runTimeoutMs`
   (default 120 s), with the spec's captured `cwd`. Result text is the
   model-visible content.
4. **Normalize + hash + compare** (§7). Unchanged → state file write only.
5. **On change:** heuristic materiality is "the normalized output differs"
   (nothing else — §8.1); then the classifier gate (§8); then either deliver
   (§9) or suppress + count.
6. **Persist state**, advance `next_due_at` to `now + every_ms + jitter`
   (§11.1), re-arm the timer.

A global concurrency semaphore of **2** bounds simultaneous checks across a
session's monitors — the same ceiling the wake supervisor puts on concurrent
engagements (`_MAX_CONCURRENT_ENGAGES = 2`, `wakes/supervisor.py:222`). A
deferred check stays due and runs on the next pass with a free slot; it is not
counted as skipped.

### 5.3 Cost of a tick that finds nothing

**Zero model tokens, zero context injection, no transcript entry, no log line**
(debug-level logs only). One small atomic write of the per-monitor state file
(§10.3) records `last_check_at`/`next_due_at`/counters; that write is the only
per-tick I/O the harness does for a quiet monitor. The measured cost class of
such a write is the wake index's: ~400-byte JSON via staged write +
`os.replace` (`wakes/store.py:222-259`), the same shape this reuses.

### 5.4 Timeouts, jitter, first check

- **Per-run timeout** `values.monitor.runTimeoutMs` (default 120,000 ms). A
  timed-out check is a failure for the backoff ladder; the call's own abort
  signal is used so a wedged subprocess is reaped by the tool's normal paths.
- **Jitter:** `next_due = now + every_ms + uniform(0, min(5s, every_ms/10))`
  — positive-only (never early), and it exists for one reason: several
  monitors on one machine with the same interval should not fire in lockstep
  against the same upstream.
- **First check** runs `uniform(1, 3)` s after arm, so the baseline is captured
  promptly and a broken spec surfaces as a health line early rather than a
  silent week-long nothing.

## 6. Read-only enforcement (R3) — the safety core

### 6.1 The rule

> A monitor may wrap a call **iff the harness's effective approval tier for
> that call is `read`** — the same computation the loop gates on:
> `tool.call_approval_tier(args) if tool.call_approval_tier else
> tool.approval_tier` (`harness/loop.py:3220-3224`; fields `harness/types.py:
> 1369`, `1384`). There is no second tier vocabulary and no parallel gate.

The read tier's meaning here is the harness's: observing, no side effects the
operator would need to consent to. What makes a monitor different from an
interactive read is that it repeats **unattended**, so the rule is enforced at
arm time (fail loudly, §6.7) and re-checked at run time (§6.8).

### 6.2 How each target class evaluates

| target | verdict source | accept / reject |
|---|---|---|
| `read`, `grep`, `glob` (static read tier) | `approval_tier == "read"` | accept |
| `web_search`, `web_fetch` | static read tier; `web_fetch` is GET/idempotent today (`web_fetch/service.py:842-843` requires any non-idempotent method to gate itself) | accept |
| `hub` (parent), `list`/`peek` | `call_approval_tier` returns `read` for these two ops (`builtin.py:20988-20996`) | accept; reject `resume`/`steer`/`now`/… |
| `jobs` (static `read`, `builtin.py:20284`) | `approval_tier` | accept |
| `task` (static `write`, **no** dynamic tier, `builtin.py:19699-19706`) | — | **reject, even for `peek`** |
| any other dynamic-tier tool | whatever its `call_approval_tier` answers for these args | honored as-is |
| `bash` | **no tier exists at this head** — static `exec` (`builtin.py:4803`) | see §6.4 |
| MCP (`mcp__…`) | `annotations.readOnlyHint is True` on the server's `tools/list` row | see §6.5 |
| `eval` | none | reject, §6.6 |

Correction of a premise in the brief, recorded so nobody builds on it: **PR
#1696's "bash scope" is not a per-command approval tier.** At this head the
bash tool's tier is static `exec`, and #1696's bash change is the
**evaluation** confinement — macOS seatbelt profiles around episode shells
(`local_operator/tools/confinement.py`, commit `1c8735a31`/`478b0e219`, merged
as `3f848c539`). There is no existing "read-only bash command classifier" to
reuse; §6.4 specifies the one this change adds.

### 6.3 The evaluator

One module, one consumer: `local_operator/monitors/readonly.py` exposes

```python
def readonly_verdict(tool: AgentTool, args: Mapping[str, Any],
                     *, mcp_annotations: Mapping[str, Any] | None = None) -> str | None:
    """None = read-only (accept). A sentence = the refusal reason (reject)."""
```

It is consulted by the monitor tool's arm-time validation and by the
run-time re-check. It reads the same fields the loop reads; it is not a
second gating *convention* (AGENTS.md: "extend the table, do not invent a
parallel mechanism") — bash and MCP are the two classes where the existing
fields do not answer, and this module is where their answer lives.

### 6.4 bash: provably read-only, defined

A command qualifies only if it is **a single logical line** that parses to a
pipeline of simple commands where:

1. **No shell metacharacters beyond `|` and ordinary word characters**: no
   `;`, `&&`, `||`, `&`, newline, backticks, `$(`, `<`, `>`, `>>`, `2>`, no
   variable assignments before the command word, no `env`/`sudo`/`xargs`/
   `sh -c` wrappers. (These are rejected with the offending token named.)
2. **Every command word is on the v1 allow-list** (below), matched by
   basename, with no path prefix (`/bin/ls` and `./script` are rejected).
3. **No denied flag on any command** — the per-command write/execution
   switches: `find` `-delete -exec -execdir -ok -okdir -fprint* -fls`,
   `sort -o`, `grep`/`rg` `--output`, `jq` file-writing forms are impossible
   (no redirection), `git -c`, `gh` non-read subcommands, etc.

v1 allow-list, deliberately short and grown on demand (each addition is a
reviewable line plus tests):

| command | allowed form | denied |
|---|---|---|
| `cat` `head` `tail` `wc` `ls` `stat` `file` `date` `uname` `whoami` `id` `nproc` `df` `du` `ps` `tree` | as-is | — |
| `find` | read forms | `-delete -exec -execdir -ok -okdir -fprint -fprint0 -fprintf -fls` |
| `grep` `rg` | read forms | `--output` |
| `jq` | expressions only | none that write without redirection |
| `git` | `status log diff show blame rev-parse ls-files grep branch` (list form) | every other subcommand (`push`, `fetch`, `checkout`, `-c` overrides…) |
| `gh` | `pr view`, `pr list`, `pr diff`, `pr checks`, `pr status`, `issue view`, `issue list`, `run view`, `run list` | everything else, incl. bare `gh api` (its method is flag-flippable) |
| `glab` | the same read subcommands | everything else |

Rejections are sentences, e.g.:

- `monitor can't watch "rm -rf build": monitors re-run unattended, so a bash
  command must be provably read-only — and "rm" never is. Try a read-only
  command, web_fetch, or an MCP tool that declares readOnlyHint.`
- `monitor can't watch "ls > out.txt": output redirection writes to disk.`
- `monitor can't watch "find . -delete": "-delete" makes find a write.`

**Residual risk, stated rather than implied:** this is a conservative static
allow-list, not a sandbox. A missing denial on an allow-listed command is a
write vector, which is why the list stays small and every entry is tested; the
kernel-level alternative (seatbelt) exists only for evaluation-confined
sessions and macOS only (`tools/confinement.py`), and is not a general
mechanism here. The evaluator's conservatism is the current mitigation; §21
records what would replace it.

### 6.5 MCP: `readOnlyHint` is the only key

An MCP tool joins only when its `tools/list` row carries
`annotations.readOnlyHint is True` — read from the same payload the bridge
wraps (`mcp/tool_bridge.py:build_agent_tool`) or from the cached JSON. Absent
or `false` → reject with:

`monitor can't watch "mcp__slack__send_message": it does not declare
readOnlyHint, so the harness cannot rule out side effects.`

This is self-declared by the server (the MCP spec's annotation), and the
residual risk is the same trust surface the operator already accepts by
enabling the server. The bridge's current `approval_tier="exec"` for MCP tools
("unknown external side effects default to exec", `tool_bridge.py:421`) stays
exactly as it is; the monitor evaluator is the only consumer that reads the
hint.

### 6.6 eval: rejected in v1

`eval` is arbitrary Python — file writes, subprocesses, network sends are all
one `import` away, and there is no cross-platform way to prove absence
statically. v1 rejects with:

`monitor can't watch eval: arbitrary Python cannot be proven read-only, and
monitors run unattended. Watch a read-only command (bash), a URL (web_fetch),
or a tool that declares readOnlyHint instead.`

The recorded revisit criterion (§21): a sandboxed eval worker with enforced
no-write semantics, demanded by real usage.

### 6.7 Arming a write/exec call fails loudly

`monitor {op:"create"}` returns an **error result** (not a warning, not a
silent downgrade) naming the tool, the reason and a compliant alternative.
Never: "armed, but checks will be skipped".

### 6.8 Re-validate before every run

Cheap (a dict lookup + the tier computation): tool present, tier still `read`.
A call that lost its qualification (settings change moved a dynamic tier; a
`createIf` tool vanished) skips the tick and counts a failure with the reason
`call is no longer read-only`, which surfaces through the normal health path.

### 6.9 Test matrix (the R3 deliverable)

Accept: `read`/`grep`/`glob`; `web_fetch`; `web_search`; `hub {op:list}`;
`jobs`; `bash` with `gh pr view 1710 --json state`, `ls -la`,
`cat f | grep x | wc -l`, `find . -name '*.md'`.
Reject: `bash` with `rm`, `ls > f`, `cat f > out`, `find . -delete`,
`git push`, `echo $(date)`, `a && b`, `python3 -c …`, `sed -i …`; `eval`
(any args); `hub {op:"resume"}`; `task {op:"peek"}` (write tier, no dynamic tier); `mcp__x__y` without / with
`readOnlyHint: false`; a tool not in the session's set.
Each case asserts the exact refusal sentence's discriminating phrase, so a
reordered message fails a test rather than drifting.

## 7. Diff pipeline & snapshots (R4)

### 7.1 Normalization (applied in order)

1. Decode to text; normalize line endings; strip trailing whitespace per line.
2. **Timestamps → `<ts>`** when `values.monitor.normalizeTimestamps` (default
   true): ISO-8601 and `YYYY-MM-DD[ T]HH:MM[:SS]` forms, epoch-second/millis
   tokens above a plausible floor. This is the single highest-yield noise
   class (the brief calls it out; every `updated_at` rewrite would otherwise
   be a "change").
3. **Per-monitor `ignore`** regexes (≤ 8, validated at arm): matching lines
   are dropped.
4. **`sort_lines: true`** (per monitor, default false): line multiset compare
   instead of sequence compare. Off by default because ordering can *be* the
   change (a sorted-by-newest listing); on for set-like outputs.

### 7.2 Snapshot, hash, compare

- The **hash** is `sha256` over the FULL normalized text — equality detection
  is exact regardless of what is stored.
- The **stored** normalized text is capped at
  `values.monitor.snapshotMaxChars` (default 32,768); the cap keeps state
  bounded and is noted in the state file (`snapshot_truncated`).
- Compare = hash equality. If hashes differ but the stored copies are
  identical (both truncated; the change is beyond the window), the delta is
  `change beyond the stored snapshot window` plus the new checksum — honest,
  not empty.
- A **missing/corrupt state file** → silent baseline re-establishment (no
  delivery of a full dump), recorded in health as
  `baseline re-established` — the self-healing posture the store uses.

### 7.3 The delta

Line diff (`difflib.SequenceMatcher` over lines; inputs are bounded by the
snapshot cap, ~1k lines worst case, which keeps a quadratic matcher cheap —
the unit suite pins the bound), then
a bounded summary — **never the full output**:

- counts: `+N/-N` changed lines;
- up to `values.monitor.maxDeltaLines` (default 12) changed lines as short
  previews, each clipped;
- the whole delta text ≤ `values.monitor.deltaMaxChars` (default 1,200 chars);
- a truncation marker naming the remainder: `… and N more changed lines`.

### 7.4 The delta is data

Monitored output is foreign text (a web page, an MCP tool's rendering). It is
delivered inside a provenance envelope (§9.1) and treated as data the agent
may read, never as instructions; bounding (§7.3) is also the injection bound.
The guide says this explicitly.

## 8. The classifier gate (R5)

### 8.1 When it runs

Only on a **heuristic hit** — a normalized-output difference (§7). An
unchanged tick makes **zero** classifier calls. Multiple monitors' hits in one
pump pass fold into **one** call: the state is the concatenation of their
bounded deltas, each with its monitor name, total ≤ `classifyMaxChars`.

### 8.2 The service seam

One new public method on the existing `ClassificationService`
(`classification/service.py`), reusing every guard `recommend_resources`
already applies (`service.py:346-425`): `enabled`, leg resolution
(`_available_legs`, `service.py:618`), the circuit breaker, the shared HTTP
client, the timeout `values.classification.timeoutMs`, and the fail-open
posture — with a small bounded LRU of its own (32 entries, keyed by
`sha256(state)`):

```python
async def decide(self, *, state: str, question: Question) -> Answer | None:
    """One typed question under the same guards as ``recommend_resources``.

    None for every non-answer: disabled, no usable leg, breaker open, timeout,
    transport failure. Callers treat None as "no classifier" and fail OPEN.
    Never raises (cancellation excepted); never blocks a monitor tick.
    """
```

`provider_available()` (`service.py:583`) is the same credential probe the
message path uses; when it answers False, no request is placed and the
heuristic-only behavior applies. No new classifier config keys; the monitor
gate reads `values.classification.*` as-is.

### 8.3 The typed question — chosen shape and measured cost

**Decision: a `choice` question with three classes.** Measured, both ways, on
the same representative state (358 chars; method §17):

| shape | question alone | full request (state+question) |
|---|---|---|
| `choice` (3 classes) | 616 chars / 138 cl100k / ~222 billed | 1,037 chars / 253 cl100k / ~373 billed |
| `noul` | 441 chars / 102 cl100k / ~159 billed | 862 chars / 217 cl100k / ~310 billed |

The choice costs **+36 cl100k / ~63 billed tokens per call** over the noul.
It wins because the two suppressed classes feed the counters (`non-material
metadata` vs `ignorable` — the numbers that tell an operator what their
monitor is drowning in) and because a binary question invites collapsing
"metadata churn" into "noise", which is the distinction the normalization
tuning needs next. The fork is still binary; the split is telemetry, and it is
cheap. (Cost of a call at the classification layer's measured Radient-route
rate — `$0.00003` per ~1,091 input tokens, `classification-layer.md` §8 — is
≈ $0.00001 per call; the arithmetic is in §17.)

```jsonc
"monitor_materiality": {
  "type": "choice",
  "instructions":
    "Decide whether this change is MATERIAL: a human asked to be told about it. "
    "MATERIAL = new information a person would want (a reply, a status flip, a new record). "
    "NON-MATERIAL METADATA = bookkeeping that changed without meaning (timestamps, ordering, "
    "volatile ids). IGNORABLE = noise that will never matter (whitespace, boilerplate).",
  "criteria": {
    "material": "a person wanted to be told about this change",
    "non-material-metadata": "changed, but only metadata: timestamps, ordering, volatile ids",
    "ignorable": "noise that can never matter; whitespace, boilerplate, formatting"
  }
}
```

### 8.4 The fork

| answer | action |
|---|---|
| `material` | deliver (§9) |
| `non-material-metadata` | suppress; increment `suppressed.non_material_metadata` |
| `ignorable` | suppress; increment `suppressed.ignorable` |
| `noul` `true`/`false` (if ever re-pinned to that shape) | deliver / suppress+count |
| `None` — disabled, no vendor, not logged in, breaker open, timeout, failure | **fail open: deliver** (heuristic-only behavior; the operator prefers a stale-ish extra than a silent swallow) |

Nothing here blocks, mutates, or gates a capability; a wrong suppression costs
a delayed signal the counters make visible, and a wrong delivery costs one
bounded message.

## 9. Delivery & injection (R6)

### 9.1 The template (final; this is the whole injection)

A material delta arrives as a `monitor_prompt` custom message
(`custom_type="monitor_prompt"`, `attribution="user"` — the wake precedent,
`session.py:16602-16606`), exactly this shape:

```
(monitor) 'loom-pr-1710' m1: 3 changes at 09:33 — check 41 (2 skipped while the session was down).
Cancel with monitor({op:"cancel",id:"m1"}) once its goal is met.

Diff vs the previous check:
- statusCheckRollup: FAILURE -> SUCCESS
+ review by @sherman-tsui: "can you split the retry logic into its own helper?"
+ review by @bbqben: "CI is green after the rebase"
… and 2 more changed lines
```

Envelope rules, mirroring `format_wake_delivery_text` (`harness/wake.py:
629-653`): id and name ride the first line because an agent that cannot name
its own monitor cannot cancel it; the source is named when it is not obvious
from the name; the counts are absolute; the cancel hint is dropped in the
final delivery after `until` passes. The body is the bounded delta (§7.3).

**Bounds, measured:** envelope ≈ 163 chars / 53 cl100k / ~59 billed;
representative 8-line body = 668 chars / 207 cl100k / ~240 billed; worst case
(12 × ~100-char lines + marker) = **1,406 chars / 274 cl100k / ~506 billed** —
the "~≤500 tokens" budget in the brief, hit and measured. Same-pass folds
share the one 1,200-char delta budget, so the bound holds for any number of
coalesced monitors.

### 9.2 How it lands

Identical to a wake delivery (`_deliver_wake`, `session.py:16564-16638`):

- **Idle session** → a turn opens with the monitor message as its input.
- **Busy session** → courtesy delivery: the message is folded via the
  steering queue at the next tool boundary, with the same
  "handle the task, then resume the work you were doing" suffix the wake path
  appends (`session.py:16641-16663`) — a monitor that fires mid-turn
  interrupts nothing.
- A `MonitorDeltaEvent` (modeled on `WakeDeliveredEvent`,
  `harness/types.py:1855`) is emitted before the turn spawn so the TUI paints
  the receipt line ahead of the work, and `frontend_state` exposes the live
  monitor list (§12).
- **Nothing when nothing changed.** No message, no event, no note.

### 9.3 Downtime honesty and no re-surface

- Checks skipped because the process was down are counted from
  `last_check_at` and named once in the first post-resume delivery
  (`N checks skipped while the session was down`, deduplicated to a count —
  the wake catch-up pattern, `harness/wake.py:15-19`).
- **No missed-tick replay:** one consolidated delta covers the gap; the
  individual skipped ticks are never replayed, and because snapshots advance
  on every check, an unchanged state can never re-surface the same content on
  a later tick — this is mechanical, not behavioral.
- **No-action is a first-class outcome** (§13.2): a delivery whose turn
  produces no reply is normal and costs nothing extra.

### 9.4 Rate cap

Per monitor, `values.monitor.maxDeliveriesPerHour` (default **12**). Over the
cap, hits are suppressed and counted; the next allowed delivery names them
(`N earlier changes were held by the hourly cap`). This bounds a firehose
source (a counter that changes every tick) without capping legitimate
watching.

## 10. Persistence & restart (R7)

### 10.1 Source of truth: the transcript

Custom entry `monitor_schedules`, written by **one writer**
(`Session._persist_monitor_schedules`), in the wake ordering — transcript
first, index second, and only the transcript append may fail the request
(`_persist_wake_schedules`'s contract, `session.py:15861-15880`):

```jsonc
// custom_type: "monitor_schedules"   (latest entry wins, like wake_schedules)
{ "monitors": [ /* MonitorSpec.model_dump() */ ] }
```

```python
class MonitorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str                      # "m1"… stable per session, never reused
    name: str                    # short label
    tool: str                    # session tool name, e.g. "bash"
    arguments: dict[str, Any]    # validated against the tool's schema at arm time
    every_ms: int                # ge MIN_MONITOR_INTERVAL_MS
    until_at: int | None = None  # None = durable
    description: str = ""
    notify: bool = False         # §14 — the delivery's notification control parameter
    sort_lines: bool = False
    ignore: list[str] = []       # ≤ 8 validated regexes
    cwd: str = ""                # captured at arm; the check runs with it
    created_at: int = 0
```

A blank append REPLACES the list (wake semantics); an empty list removes the
index entry ("no file" and "no monitors" are the same statement).

### 10.2 The derived index — import-light, self-healing

`<config_dir>/monitors/<session_id>.json` (flat, one file per
monitor-carrying session), exactly the `wakes/store.py` contract
(`store.py:1-41`): **derived, never authoritative**; rewritten immediately
after every transcript append and on every open (after
`_load_monitor_schedules`), which is the self-healing property — a deleted,
corrupt or stale entry is repaired on the next session build, and an emptied
list removes it. Readers never patch it. stdlib-only, pinned by
`tests/unit/test_import_graph.py` (the wake siblings are pinned at
`tests/unit/test_import_graph.py:310-400`); the `wakes/deliveries/`
subdirectory precedent (`deliveries.py:52-57`) keeps scans clean — the index
scan skips anything not ending `.json`, so state lives under
`monitors/state/…`.

```jsonc
{
  "schema": 1,
  "session_id": "…", "cwd": "…", "updated_at": 1759000000000,
  "stopped_at": null,              // park marker; same meaning as in wakes (is_held)
  "monitors": [
    { /* spec fields */, "next_due_at": 1759000060000, "last_check_at": 1758999999000,
      "checks": 41, "deliveries": 3, "consecutive_failures": 0, "disabled": false }
  ]
}
```

Rewritten on: arm / cancel / disable / re-arm / delivery / park / session open.
**Not** on every quiet tick — for a quiet tick the state file alone moves, so a
cold reader's `next_due_at` is best-effort between change events (documented;
`lop monitor status` labels it as such). This is the one deliberate divergence
from `wakes/store.py`, which persists after every fire: a wake fires rarely; a
30 s monitor would rewrite its index 2,880×/day for no reader's benefit.

### 10.3 Per-monitor state (snapshots)

`<config_dir>/monitors/state/<session_id>/<monitor_id>.json` — bounded
(~≤ snapshotMaxChars + counters), atomic staged writes; the schema is §7's
snapshot + counters + health:

```jsonc
{ "schema": 1, "monitor_id": "m1",
  "content_hash": "sha256:…", "snapshot": "…", "snapshot_truncated": false,
  "last_check_at": …, "last_change_at": …, "next_due_at": …,
  "checks": 41, "deliveries": 3,
  "suppressed": {"non_material_metadata": 11, "ignorable": 2},
  "rate_window_start": 0, "rate_window_count": 0,
  "consecutive_failures": 0, "disabled_reason": "", "last_error": "" }
```

Why not the transcript: a snapshot rewrite per tick would put a durable
transcript row on disk every 30–60 s per monitor — the exact bloat the "no
transcript entry per tick" rule (R2) exists to prevent, and transcript size is
the compaction machinery's rival. Snapshots are cache-like derived state: if
lost, the next check re-establishes a baseline silently (§7.2).

### 10.4 Cold-session engagement — the boundary, documented

**Decision: monitors never engage a cold session.** They tick only while the
session is hosted (a TUI/terminal is open on it, or a serve/desktop runtime
owns it). If the session goes cold, monitors are dormant; they resume with the
session and deliver one consolidated delta (§9.3).

Why not the wake-supervisor model (`wakes/supervisor.py` engaging a runtime
for a due wake):

- A wake engagement makes a runtime **exist**; the runtime then fires its own
  overdue wake. Wakes are spaced for this (typically hours/days; the floor is
  60 s but real schedules are far apart) and engagements are bounded at 2
  concurrent with a 180 s deadline (`_MAX_CONCURRENT_ENGAGES`,
  `WAKE_DEADLINE_S = 180`, `wakes/supervisor.py:204-222`).
- A 60 s monitor under engagement policy would demand a runtime **permanently
  resident** — 1,440 engagements/day per monitor — each one a full harness
  boot. That converts the cheapest primitive in the system into its most
  expensive one, and the supervisor's own design notes call a runtime boot on
  a loaded box the thing `_MAX_CONCURRENT_ENGAGES` exists to ration.
- The watching pattern's real deployment shape already has the session
  hosted: the desktop app's runtime, or the terminal the user set the watch
  from. The dormant window is exactly the window where "tell me when it
  changes" has a human returning shortly anyway — and the resume delivery
  makes the gap honest.

Recorded as an open question (§21): a "slow monitors only" engagement policy
(e.g. interval ≥ 15 min, engaged like a wake). What would settle it: a
measured cold-boot cost on this fleet plus a real demand case; the boundary
above is the v1 contract.

### 10.5 Park, pristine, and retire parity

- **Park (stop):** `runtime/control.py::_mark_wakes_dormant`'s monitor twin
  stamps `stopped_at` on the monitors index entry; the receipt line becomes
  `— N monitor(s) dormant until you reopen it` alongside the wakes part
  (`control.py:788-810`); the open-time rewrite clears it (wakes'
  `clear=("stopped_at",)`, `wakes/store.py:203-221`).
- **Pristine probe:** `serving.py:1543-1608` refuses to call a session
  pristine while anything is scheduled; monitors join: the scheduler's
  `next_monitor_due_at()` and the on-disk index entry both answer "not
  pristine" (present-or-index, the same double check the wake arm uses).
- **Retire divert:** wakes spool to a successor inbox on a hand-off
  (`serving.py:1749-1787`); monitors have no per-fire delivery to divert —
  the successor's rehydrate runs the first check against the persisted
  snapshot, which is the same guarantee by construction.

## 11. Caps, safety, reap (R8)

### 11.1 Floor and default — decided

**Floor 30 s** (`MIN_MONITOR_INTERVAL_MS = 30_000`, field-level like
`MIN_WAKE_INTERVAL_MS`), **default 60 s** when `every` is omitted. A monitor
tick is one read-only call with no turn and no tokens; the floor exists to
bound tool-call frequency (upstream courtesy, file-churn bounds) and 30 s
matches the shortest useful web-cache window. Wakes keep their 60 s floor:
they open turns, and the brief's "propose 30s; 60s default" lands here.

### 11.2 Jitter

`+uniform(0, min(5 s, every/10))` per tick (positive-only, §5.4). Rationale:
de-synchronize monitors that share an interval; 10 % keeps a 30 s monitor's
effective cadence within 27–30 checks/min-scale bounds, and the 5 s cap keeps
long intervals from drifting visibly.

### 11.3 Failure ladder and auto-disable

- Each failed check (tool error, timeout, no-longer-read-only, executor
  exception) increments `consecutive_failures` and schedules the next attempt
  at `last_attempt + every_ms × 2^n`, capped at 15 min
  (`RETRY_CAP_S`-style ceiling, `wakes/deliveries.py:80`).
- At `values.monitor.maxConsecutiveFailures` (default **5**), the monitor is
  **disabled**, `disabled_reason` = the last error (bounded), and the reason
  is surfaced everywhere (list, TUI, `lop monitor status`, index). A disabled
  monitor does not tick, does not count against delivery caps, and cannot be
  silently forgotten: it stays listed with its reason. **Re-arm to
  reactivate:** creating the same spec again resets failures and returns
  `Reactivated monitor 'm1'.`
- **Defined as intentional:** a disabled monitor is *failed*, not abandoned —
  it is reportable state, not junk (§11.5).

### 11.4 Caps, dedupe, storms

- **Per-session cap 8** (`values.monitor.maxMonitors`, default 8) against
  wakes' 16. Rationale: every monitor holds a timer slot, a snapshot file and
  a client-side poll; 8 covers real watching patterns (a handful of threads /
  files) with headroom, and the cap is a rejection with a sentence naming it
  and suggesting `cancel` — never a silent drop.
- **Dedupe:** identity = `sha256(tool + canonical-JSON(arguments))` (NOT name
  or interval). Arming an identical spec returns the existing monitor
  (`already watches that call every 60s`). A different interval requires
  cancel + create, and the tool says so — two rows polling one source is the
  duplication this settles.
- **Storm rejection:** the 9th arm, the 3rd duplicate name from one turn's
  batch, and any arm during a disabled-index write failure all return one
  actionable error each; nothing queues.

### 11.5 Reap and cleanup

- **Orphaned index entries self-heal** on the owning session's next open
  (§10.2); a scan (TUI picker, `lop monitor status`) treats a corrupt entry as
  absent and never patches it — `wakes/store.py`'s rule, verbatim.
- **Junk-session reap cleans monitors**: when a session directory is deleted,
  the deletion path prunes `monitors/<session_id>.json` **and**
  `monitors/state/<session_id>/` — the `_forget_wake_entry` twin
  (`cleanup.py:1462-1477`, called from the deletion path at
  `cleanup.py:736-741`).
- **`sessions cleanup` integration:** a session with an armed, non-disabled,
  non-expired monitor is **refused deletion** exactly as one with an armed
  wake is (`_has_armed_wake`, `cleanup.py:939`; refusal copy pattern
  `cleanup.py:411-451`): *"That conversation has a monitor armed for it.
  Reopen it and ask it to cancel the monitor, or delete its
  `monitors/<id>.json` entry, before deleting it."* The automatic reapers
  (empty-directory / probation) must consult the monitors index before
  removing a directory — the index lives **outside** the session directory for
  this reason (`wakes/store.py:18-25`), so a session whose only visible state
  is a monitor is not "empty".
- **"Unintentional" never swallows a durable monitor.** The cleanup vocabulary
  (failed / abandoned / dangling) describes sessions; a `durable` monitor is
  first-class perpetuity — a promise — and a session carrying one is
  intentional by definition. Expired (`until` passed) or disabled monitors do
  not arm the refusal; their entries are pruned like any other dead state.
- **No unbounded accumulation:** ≤ 8 monitors/session; bounded snapshot
  (32 KiB each) and delta (1.2 k); capped deliveries/hour; classifier called
  only on hits; one state file per monitor; index files pruned with their
  sessions. Counts are visible in `monitor list`, the TUI band, and
  `lop monitor status`.

## 12. Awareness surfaces (R9) — insertion points inventory

Design-only; each row names the exact place, the change, and the parity
argument. "Parity with wakes" is the standard; divergences are stated.

| surface | where | change |
|---|---|---|
| Session frontend state | `session/frontend_state.py` — `wakes: list[WakeState]` (`:2849`), `_wake_state` (`:7447`), `_REVISED_COLLECTIONS` (`:5199-5200`) | add `monitors: list[MonitorState]` + `_monitor_state(scheduler)` + `"monitors"` in the revised collections (the revision token moves — a wire-visible change to note in its PR) |
| TUI dock band | `tui/widgets/wake_panel.py` (hides itself when the session has no wakes, `:1-24`; `MAX_WAKE_ROWS = 3`; the U7 shared-floor note) | render monitor rows **in the same band**: wake rows first, then a monitor section (`MAX_MONITOR_ROWS = 2` + overflow marker), same share-the-column arithmetic; hide when both are empty. One band, not a fourth panel: the panels' own docs record that the dock's rows are a shared column and a new sibling spends everyone's floor (U7) |
| TUI receipt | `tui/events.py::_handle_wake_delivered` (`:1060`) | a `_handle_monitor_delta` sibling posting the same expandable receipt shape |
| Session catalog | `session/catalog.py` — `decorate_rows` reads the wake index (`:972-977`), fields at `:1133-1146`, rank key at `:181-185` | read the monitors index too; add `monitors` count + dormant flag; extend the rank value (an armed monitor leads the cold group like an armed wake; a dormant one ranks with dormant wakes). The rank *tuple shape* (`(tier, wake_rank, -birth, id)`, `:691`) is unchanged — only the value function moves — so persisted cursors stay parseable |
| Runtime control | `session/runtime/control.py` — `_mark_wakes_dormant` (`:675`), `_park_wakes` (`:710`), `_stopped_line` (`:788-810`) | park monitors with `stopped_at`; receipt names them (§10.5) |
| Pristine probe | `session/runtime/serving.py:1543-1608` | monitors join the "not pristine" predicate (§10.5) |
| CLI | `cli.py` — the `wake` group's shape (`:1136-1260`) is the template | `lop monitor status [--json]` (per-session monitors: name, interval, last check, next due, health, disabled reason; dormancy) and `lop monitor cancel <session> <id>` (external-writer path mirroring `wakes/arm.py`'s transcript-first + lock + post-write verification). Deliberately **no** install/uninstall subcommands: there is no supervisor to install (§10.4) — its absence is a design statement, not an omission |
| Desktop UI (local-operator-ui) | where wakes render today: session sidebar/status from `frontend_state`, and the arm/edit command routes modeled on the wake routes | render armed monitors with health; arm/edit/cancel routes. Implementation lives in the UI repo; the backend contract is the `monitors` field + command routes |
| Agent surface | the `monitor` tool (§4) | `list` for the agent; `cancel` by id |

**Cancel affordance, stated precisely:** v1 ships cancel via (a) the `monitor`
tool (agent-mediated — "ask it to cancel"), (b) `lop monitor cancel` (CLI),
(c) the desktop command route. The TUI band does **not** get a cancel binding
in v1, matching the wake band, which also has none (recorded in
`cleanup.py:427-451`: "the wake band lists schedules with no cancel action or
binding"); adding a selection model to the band is a UI design task with its
own review, tracked as a follow-up, not smuggled into this contract.

## 13. Prompts, instructions, and the conversational loops (R10, R14)

### 13.1 Where the teaching lives (targets)

- `local_operator/prompts_md/system.md` — the "## Tools" paragraph currently
  carries the wake sentence (`system.md:110-111`). The monitor sentence lands
  beside it (candidate below). Unconditional prose, like the wake sentence;
  no new template flag pair (the browser/console pairs exist because their
  absence needs a *different* paragraph; a missing tool just costs a line).
- The tool description itself (§4.4) — the primary discovery text.
- `guide://monitor` (§15) — the full playbook, read on demand.
- The tool inventory block (`prompts_api.py:386-400`) names the tool
  automatically once it exists — names only, no description duplication.
- `ecosystem_instructions` — **not** a target: it reads *foreign* instruction
  files (`~/.agents/AGENTS.md` etc.); harness-owned teaching does not go there.
- Subagents: no monitor tool (§4.4), so no teaching to prune.

### 13.2 Candidate wording (measured, §17)

Replace the wake sentence with (or append after it):

> `wake` schedules follow-ups when the user asks to be reminded or something
> should happen later; `monitor` watches a read-only call for changes — arm one
> when the user asks you to watch, poll, or be told when something changes, and
> it reports only deltas.

(254 chars / 55 cl100k / ~91 billed; the existing sentence is 95 / 19 / ~34, so
the delta is +159 chars / +36 cl100k / ~+57 billed — one cached-prefix line.)

And one short no-action sentence (the R14 core, also echoed in the guide):

> A wake or monitor turn that finds nothing needing action is complete: end it
> with no reply, keep the same unchanged content out of later turns, and don't
> notify.

(161 / 34 / ~58.)

### 13.3 The conversational-loop rule (R14, normative)

Both the base prompt sentence and `guide://monitor` carry it:

> When the user asks you to **respond, post, or keep people updated** in a
> thread or channel, arm a monitor on that thread/channel's read call so
> replies reach you when they land (60 s is the sensible default), and reply
> only when someone messaged/tagged you or the delta genuinely needs
> attention. Being woken with nothing to do is a fine outcome — it is why
> monitors are quiet.

Mechanically backed by §9: no change ⇒ no message; unchanged content can never
re-surface (snapshot advance); no-action deliveries are bounded (§9.1) and
cost one turn the agent can end silently.

### 13.4 Notify guidance

The base prompt guidance and the guide both say: *if the user asked to be
told, arm with `notify: true`; otherwise leave it quiet* — this is the agent's
side of §14.4.

## 14. Turn-completion notifications — origin-aware and agent-controlled (R15, R15a)

### 14.1 Today's machinery (verified, not assumed)

A turn's completion is published **once**, inside the session:
`_publish_attention_outcome` (`session.py:9011-9127`) computes `kind ∈
{complete, error, interrupted}`, refuses "complete" when there is no durable
assistant message or a `task` child is still running (`:9036`, `:9092-9101`),
journals the outcome in the transcript, and inserts a `completions` row in the
attention store (schema `attention.py:1206-1216`). Consumers read rows or
state: the desktop bridge, the machine-wide feed
(`desktop_feed.py::_emit_notifications`, `:1202-1307`), the TUI completion
funnel (`tui/app.py::_notify`, `:26093`, completion branch `:47298-47325`),
the TUI's background observer, and the OS notifier with its own gates
(`tui/notify.py:47-99`). **One row per turn** is the existing invariant; this
change adds a field to it and never a second publication.

### 14.2 The rule

Computed **once**, by the session, at turn end; every surface reads the
computed value and none re-derives it (the notifications design's "the bridge
observes the publication, not a rebuilt decision", `descriptive-
notifications.md` §5.1, is exactly the posture):

```
triggers  = custom-typed inputs that opened or folded into the run
            ("wake_prompt" | "monitor_prompt")  ∪  {"user"} when any plain
            user message is in the run's inputs
notify_requested = OR over the wake/monitor deliveries in the run of their
            own ``notify`` parameter (copied from the schedule/spec)
awaiting_user    = a plain user message is queued but not yet consumed

notify = awaiting_user
         or ("user" in triggers)                       # R15a: user semantics win
         or (triggers ⊆ {"wake_prompt", "monitor_prompt"} and notify_requested)
         or (no wake/monitor delivery in the run)      # every other origin: unchanged
```

Notes that make it implementable without ambiguity:

- The **discriminator is `custom_type`, never `attribution`**: wake deliveries
  ride `attribution="user"` by design (`session.py:16602-16606`), so only the
  custom type separates a delivery from a person.
- The rule is **gate-free for user turns** — "today's behaviour unchanged":
  all existing gates (children running, focus, test hosting, loop suppression)
  keep applying on top of `notify`.
- **Errors and interruptions are out of the parameter's reach**: `kind="error"`
  keeps notifying on wake/monitor turns (an unattended turn that *failed* is
  the case a user who walked away most needs pulled in; silencing failures is
  how silent breakage starts). `interrupted` stays suppressed as today.
- **Peer messages / child re-entry / other internal triggers are unchanged**
  (not in this contract's scope; they fall into the "no wake/monitor delivery"
  line).
- **Mixed triggers → user semantics, exactly one notification.** The single
  row already guarantees "once"; the rule guarantees "not silenced".

### 14.3 Where the value rides (mechanics)

1. `_run_turn_pipeline` records the run's trigger set and
   `notify_requested` at admission (from `initial`) and as wake/monitor
   messages fold in via the busy-path steering queue; both are consumed and
   reset per run beside `_attention_run_token` (`session.py:9912-9916`).
2. At turn end, before `_flush_held_end()`, the value is finalized and
   stamped on the emitted `AgentEndEvent` as `notify: bool = True` (additive;
   default preserves every existing reader) — this is what the live TUI reads.
3. `_publish_attention_outcome` journals it (`"notify": bool`) and passes it
   to `AttentionStore.publish`, adding `notify INTEGER NOT NULL DEFAULT 1` to
   `completions` via the existing additive-column migration pattern
   (`attention.py:1296-1303` records the `reason`/`cause` precedent); the
   state projection and `published_since` carry it. Rows written by older
   builds mean `notify=1` — their behavior verbatim.
4. Every consumer gates on it: bridge `_maybe_publish_notification`, feed
   `_emit_notifications`, TUI observer, and the TUI completion funnel (via
   the event field). One value, four readers, zero re-derivations.

### 14.4 The delivery's control parameter

`WakeSchedule` (`harness/wake_types.py`) and `MonitorSpec` (§10.1) both gain
`notify: bool = False`; the names ride the deliveries' `details`
(`wake_prompt` / `monitor_prompt`) so the session's trigger record can read
them. Default **false** for both — "discretionary" made concrete: a wake or
monitor turn notifies only when its delivery asked to. This is an intentional
behavior change for wakes (previously their completions notified like any
turn), and the counterweights are:

- the wake tool description and prompt guidance tell the agent to pass
  `notify: true` when the user asked to be told (`wake create … notify:true`),
  so reminders keep their point;
- errors always notify (§14.2);
- the wake tool's schema grows one field: measured +172 chars / +48 cl100k /
  ~+62 billed over the shipped tool (§17).

CLI note: `lop wake create` gains `--notify` (opt-in; default stays quiet, so
scripted arms match the contract).

### 14.5 Explicitly not built in v1: a per-turn raise hatch

Alternatives considered — a marker in the final message, a tool op
(`monitor {op:"notify"}`), a settings flag — were rejected for v1: each is a
new agent surface (prompt cost, abuse vector) to serve the case "a quiet
monitor found something urgent". The delivery parameter (arm-time discretion,
re-armable) covers the stated requirement; the in-conversation reply is always
available and reaches the user on return. Recorded as an open question with
its trigger: real use showing a quiet monitor's urgent-but-discretionary
outcome being missed.

### 14.6 Test set

1. user-triggered turn → row `notify=1`, notification emitted (today's
   behavior byte-for-byte when no wake/monitor is involved).
2. wake-only / monitor-only turn, delivery `notify=false` → row `notify=0`;
   TUI, bridge, and feed each emit nothing.
3. wake/monitor turn with `notify=true` → emits on all three.
4. mixed (user message + wake delivery, delivery quiet) → emits, **exactly
   one** frame/toast across surfaces (duplication is the failing assertion,
   not just presence).
5. user message queued-but-unconsumed at outcome time + wake delivery →
   emits (§14.2 `awaiting_user`).
6. wake/monitor turn ending in `error` → emits regardless of the parameter.
7. a pre-existing row without the column → treated as `notify=1` (migration
   test).

## 15. The recommendations guide (R11)

Add `local_operator/guides/monitor/GUIDE.md`, frontmatter:

```yaml
---
name: monitor
description: Watch X / monitor Y / tell me when Z changes — arm a delta-watching monitor over any read-only call (bash, web, MCP) and tune what counts as material.
---
```

(150 chars / 37 cl100k / ~54 billed as a roster line; zero until suggested.)
Plumbing verified: `discover_guides()` scans `guides/*/GUIDE.md` and requires a
non-empty `description` (`guides/discovery.py:23-71`); `session_factory.py:
1741-1762` builds the guide roster the classification layer shortlists and
suggests — so "watch X", "monitor Y", "tell me when Z changes", "poll W" reach
the guide by description, with **no change to classifier semantics**. The guide
body is the playbook: arm/list/cancel examples, the wake-vs-monitor split, the
Slack/Telegram thread pattern (§13.3), what counts as material, how to tune
noise (`sort_lines`, `ignore`, `normalizeTimestamps`), the down-time honesty
line, and the no-action norm.

## 16. Configuration keys (`values.monitor.*`)

New `Section("monitor", "Monitors", Scope.NEW_SESSIONS, …)` — the scope is the
classification section's, for the classification section's reason
(`settings_io.py:447-470`): the scheduler is built per session and handed a
snapshot of the section, so an edit lands on the next session start; claiming
LIVE would be a painted lie. Every key gets its module-level default constant
beside its reader, a `Setting` in `SETTINGS`, a `Section` entry, and a
`_consumer_defaults()` row in `tests/unit/test_settings_io.py` — the whole
requirement per AGENTS.md, "Adding a configuration key":

| key | kind | default | meaning |
|---|---|---|---|
| `defaultIntervalS` | int | `60` | interval when `every` is omitted (floor 30 s is a code constant, not a setting) |
| `maxMonitors` | int | `8` | per-session cap |
| `runTimeoutMs` | int | `120000` | per-check deadline |
| `snapshotMaxChars` | int | `32768` | stored normalized-snapshot cap |
| `maxDeltaLines` | int | `12` | changed lines summarised per delivery |
| `deltaMaxChars` | int | `1200` | total delta text (all coalesced monitors share it) |
| `classifyMaxChars` | int | `1200` | classifier state bound |
| `maxConsecutiveFailures` | int | `5` | auto-disable threshold |
| `maxDeliveriesPerHour` | int | `12` | per-monitor delivery cap |
| `normalizeTimestamps` | bool | `true` | strip timestamp churn before diffing |

No master switch: the tool's `createIf` gate IS the on/off (no scheduler, no
tool); recorded in §21 as a deliberate absence.

## 17. Costs — measured accounting (R12)

**Method.** All numbers from three probe scripts run under the installed
`lop` generation interpreter (`~/.local/share/lop/generations/
20260928T132436Z-11fc505e409d/tools/local-operator/bin/python`, Python 3.14.3,
has `tiktoken`), reported three ways: **chars** (ground truth), **cl100k_base
tokens** (the repo's local ruler, `compaction/tokens.py`), and **billed
tokens = chars / 2.78** (the repo's measured prompt-surface rate,
`scripts/bench_context_budget.py:CHARS_PER_BILLED_TOKEN`). The scripts are
`probe_wake_tool.py`, `probe_question_costs.py`, `probe_injection_costs.py`,
`probe_final3.py` (this session's scratchpad). A number from this table may be
compared against another in this table, never against a provider invoice (the
token module's own rule). `WakeParams` is byte-identical between the released
0.64.1 and this head (diffed), so the wake tool measured here is the shipped
one.

| subject | chars | cl100k | billed |
|---|---|---|---|
| `wake` tool wire object (shipped) | 1,551 | 487 | ~558 |
| `monitor` tool wire object (new) | 2,133 | 620 | ~767 |
| `wake` tool + `notify` field | 1,723 | 535 | ~620 |
| classifier request — choice (representative) | 1,037 | 253 | ~373 |
| classifier request — noul (representative) | 862 | 217 | ~310 |
| classifier question alone: choice / noul | 616 / 441 | 138 / 102 | ~222 / ~159 |
| injection: envelope only | 163 | 53 | ~59 |
| injection: representative (8 lines) | 668 | 207 | ~240 |
| injection: worst case (12 × ~100 chars) | 1,406 | 274 | ~506 |
| system.md sentence change (net) | +159 | +36 | ~+57 |
| no-action sentence (new) | 161 | 34 | ~58 |
| guide description line | 150 | 37 | ~54 |

**Per-event accounting.**

- *Armed, quiet tick:* 0 model tokens, 0 injected context; one ≤ ~1 KB atomic
  state write; 0 classifier calls.
- *Heuristic hit:* one classifier call ≈ 373 billed tokens ≈ **$0.000010**
  at the classification layer's measured Radient-route rate ($0.00003 per
  ~1,091 input tokens, `classification-layer.md` §8) — derived from that
  measured rate, not a new run.
- *Material delivery:* one injection ≤ 506 billed tokens + the turn the agent
  then chooses to spend (the turn is the product working, not overhead).
- *Arming:* one `monitor` tool call; the schema tax (~767 billed) is paid only
  in sessions that have a scheduler (rung 3), and sits in the cacheable tool
  prefix.

## 18. Test & evidence plan (R13)

All pytest runs use the isolated recipe (own `ISO=$(mktemp -d)`, `HOME`,
`LOCAL_OPERATOR_CONFIG_DIR`, `env -i` — AGENTS.md "Environment"/"Isolating a
run"); TUI tests run `env -u NO_COLOR TERM=xterm-256color`. No test ever
touches the operator's live config or sessions.

**Unit.** Schedule math (next-due, jitter bounds, skip-while-in-flight,
`2^n` backoff sequence and its 15-min cap, auto-disable at N, reactivation,
caps, dedupe identity, id allocation/never-reuse). Diff heuristic (each
normalization rung incl. timestamps and per-monitor `sort_lines`/`ignore`;
bounds incl. the truncation marker; hash semantics incl. the
beyond-stored-window case; silent baseline on missing/short/corrupt state).
Read-only rejection matrix (§6.9, exact phrases). Classifier fork (material →
deliver; both suppress classes → suppress+count; `None` → deliver; bounded
state; one call per folded pass; service guards reused — enabled, breaker,
cache, timeout — with a fake vendor). Reaper/caps (index self-heal on open;
orphan prune; cleanup refusal + `_forget`; park/unpark round-trip).
Persistence sync (transcript ↔ index ↔ state coherence; a failed index write
never fails the append). Import-light index pinned in
`tests/unit/test_import_graph.py` beside the wake siblings. Notifications
(§14.6, all seven). Prompts (system.md renders the new sentences; guide
discovered with its description; inventory names the tool when present, not
when absent).

**Integration (fakes, real session, isolated roots).** Arm → kill the process
→ restart → prove rehydrate: first check diffs against the persisted snapshot
and delivers one consolidated material delta with the skipped-count line (app
restart), then the same with a kill standing in for a machine restart. A slow
fake call proves no overlap; a firehose fake proves the hourly cap and the
suppression counters; a failing fake proves backoff → disable → reactivate.

**One live end-to-end** (the brief's requirement): a monitor on something real
that changes — e.g. `bash gh pr view <this PR> --json state,reviewDecision`
against this session's own PR, with a real push flipping the state, delivering
a delta — and a suppressed non-material case (a file whose stdout changes only
in a normalized timestamp → suppressed, counter moved). Commands and their
actual outputs become PR evidence.

**TUI/UI.** Rendered **before/after** frames per AGENTS.md "Visual
validation" (`save_screenshot` → SVG → view; the numbers behind the frame —
band row budget vs content box — captured too), for the band with and without
monitors; desktop-UI screenshots land with the UI slice.

## 19. Implementation plan — PR slices and order

Each slice lands as its own PR with the team's review + QA rounds on the same
head (per the operator's standing rules); this document is the review
contract for all of them.

1. **Core + safety** — `monitors/` (spec, scheduler, store, state, diff,
   readonly evaluator), session wiring (attach, persist, load, rehydrate,
   park/pristine/cleanup guards), the `monitor` tool (createIf + capability
   merge + subagent prune), config keys, `guide://monitor`, the system.md
   sentence, unit + integration tests. Deliveries happen on every heuristic
   hit (no classifier yet); notifications behave as today.
2. **Classifier gate** — `ClassificationService.decide`, the typed question,
   the fork, counters, tests, cost logging.
3. **Notifications** — the `notify` parameter (wake + monitor), trigger
   recording, event/row field + migration, consumer gates, §14.6 tests.
4. **Surfaces** — frontend state + TUI band + frames, catalog, `lop monitor
   status|cancel`, desktop-UI contract.
5. **Release notes** — behavior-change callouts (wake completion notifications
   become opt-in; monitors dormant when cold).

## 20. Risks

- **Foreign-text injection** (§7.4): a monitored page's content reaches the
  conversation as data inside a provenance envelope, bounded. The residual is
  the same class as reading any page; the guide states the posture.
- **bash allow-list gaps** (§6.4): a missed denial on an allowed command is a
  write vector. Mitigation: small list, every entry tested, arm-time and
  run-time checks; open question records the sandbox replacement.
- **MCP server honesty** (§6.5): `readOnlyHint` is self-declared; a lying
  server can mutate. The operator already trusts the server's tools; recorded.
- **Wake notification default change** (§14.4): reminders armed by older
  builds go quiet on completion until re-armed with `notify: true` — called
  out in release notes and the wake tool description.
- **Classifier cost on noisy sources**: suppressed calls only happen on hits;
  a badly-tuned monitor costs ~$0.00001/hit — the tuning loop is the answer,
  and the counters make it visible.
- **Index/state churn on shared disks**: bounded file sizes, atomic writes,
  and the no-quiet-tick-index-write rule (§10.2); the store's measured read
  is ~0.2 ms for the wake sibling.
- **Two sessions watching one URL**: no cross-session dedupe (§2); the guide
  advises naming a watcher per source. Recorded, not solved.

## 21. Open questions

1. **Slow-monitor cold engagement** (§10.4): engage runtimes for monitors with
   interval ≥ 15 min, wake-style? Evidence that would settle it: measured
   cold-boot cost on this fleet + a real demand case; until then, dormant.
2. **Per-turn notify hatch** (§14.5): needs a real case of a quiet monitor's
   urgent outcome being missed.
3. **eval support** (§6.6): a sandboxed eval worker with enforced no-write
   semantics, demanded by usage.
4. **Ordering normalization** (§7.1): `sort_lines` covers it per monitor; a
   global smart ordering heuristic is deferred (it can hide real reorderings).
5. **A `monitor.auto` master switch** (§16): deliberately absent; the tool
   gate suffices. Revisit if an operator asks to disable without a scheduler
   change.
6. **CLI default for `wake create --notify`**: quiet matches the contract;
   revisit if scripted reminder flows want loud-by-default.

## Appendix A — corrections to the brief's map

Recorded per the architect's duty to ground every recommendation in what the
code actually does:

1. **"bash's dynamic tier scopes read-only commands (PR #1696 'bash-scope')"
   is not true at this head.** bash's approval tier is static `exec`
   (`builtin.py:4803`); #1696's bash work is the evaluation seatbelt
   confinement (`tools/confinement.py`). Consequence: R3 needs a new,
   conservative bash evaluator (§6.4) instead of reusing an existing
   classifier; the tier *machinery* is still reused (§6.1), which keeps the
   contract's "no second convention" property.
2. **MCP has no `readOnlyHint` plumbing at all today** (no occurrence in
   `local_operator/`). The annotation is read from the server's `tools/list`
   payload, which the bridge and cache already carry; the plumbing is part of
   the work (§6.5), not an existing reader.
3. **"an edit path welcome if it mirrors wake's"** — wake's *tool* deliberately
   has no edit op (`builtin.py:11794-11798`); monitor mirrors that (§4.3).
4. **The wake-supervisor precedent for cold engagement would pin a runtime
   alive at monitor cadences** (§10.4); the brief's alternative branch
   (document the boundary) is taken, with numbers.
5. **Two wake surfaces do not exist as the brief assumed**: there is no
   notification on `agent_end` (the publication is the source of truth), and
   the TUI wake band has no cancel affordance to mirror (§12).

## Appendix B — what "read-only" means here

One sentence, so reviewers and tests share one definition: a call is
**read-only** when it cannot change state the operator would need to consent
to — no filesystem writes, no process/system mutation beyond the call's own
execution, no messages sent, no remote mutations. That is the meaning of the
harness `read` tier, and it is what §6 enforces. `web_fetch` qualifies because
it GETs; `web_search` because it queries; an MCP tool because its server said
so; a bash command only when §6.4 proves it.
