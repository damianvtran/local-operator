# omp Core Agent Harness — Recon Report

Repo root: ~/oss/oh-my-pi (Bun + TypeScript monorepo).
Two packages matter for the harness:

- **packages/agent** — published as **@oh-my-pi/pi-agent-core**. The provider-agnostic
  agent loop. ~4 files, zero UI, zero session persistence.
- **packages/coding-agent** — the product. Session lifecycle, tools, subagents,
  wakes, jobs, TUI, headless modes.

The clean split is the single most important structural fact: **the loop knows
nothing about sessions, subagents, jobs, or persistence.** Everything else is
injected through callbacks on AgentLoopConfig.

---

## 1. Agent main loop

### Files

| Path | Role |
|---|---|
| packages/agent/src/agent-loop.ts | The loop (2926 lines). Entry: agentLoop / agentLoopContinue; body: runLoop -> runLoopBody |
| packages/agent/src/agent.ts | Agent class (1656 lines). Owns AgentState, builds AgentLoopConfig, exposes prompt/steer/followUp |
| packages/agent/src/types.ts | AgentMessage, AgentTool, AgentContext, AgentEvent, AgentLoopConfig (876 lines) |
| packages/agent/src/append-only-context.ts | StablePrefix + AppendOnlyLog + AppendOnlyContextManager |
| packages/agent/src/run-collector.ts, telemetry.ts, pause.ts | Run summary, OTEL spans, global pause gate |
| packages/coding-agent/src/session/agent-session.ts | AgentSession (351 KB) — wires all loop callbacks to session state |

### Flow of one user turn

    user text
      -> AgentSession.prompt(text, PromptOptions)          agent-session.ts:5026
         - expands prompt templates / slash commands / @file
         - if already streaming: routes to steer() or followUp() per
           options.streamingBehavior ("steer" | "followUp")
         - else: builds AgentMessage[] and calls
      -> Agent.prompt(messages)                             agent-session.ts:1175
         - Agent assembles AgentLoopConfig from AgentOptions + live state
      -> agentLoop(prompts, context, config, signal, streamFn)   agent-loop.ts:516
         - returns EventStream<AgentEvent, AgentMessage[]> immediately;
           the async IIFE inside pushes events
         - pushes { type: "agent_start" }
      -> runLoop -> runLoopBody                             agent-loop.ts:886 / 977

**runLoopBody** is two nested while loops:

    outer while (true)                  // re-entered when follow-ups/asides arrive
      inner while (hasMoreToolCalls || pendingMessages.length > 0)
        1. yieldIfDue()                             // cooperative yield, avoids busy-wait
        2. agentPauseGate.waitUntilResumed(signal)  // process-wide /pause
        3. drain pendingMessages into currentContext.messages
        4. resolve tool-choice directive once per turn (getToolChoice)
        5. syncContextBeforeModelCall?()
        6. prepareProviderCall(currentContext, config, signal)
             -> transformContext(AgentMessage[])       // host-level pruning
             -> convertToLlm(AgentMessage[]) -> Message[]
             -> normalizeMessagesForProvider()
             -> normalizeTools()                       // injects intent field "i"
             -> AppendOnlyContextManager.build/syncMessages (if enabled)
             -> transformProviderContext(Context, Model)
        7. beforeModelCall(context, signal) gate -> may stop the run
        8. stream.push({ type: "turn_start" }) + message_start for inputs
        9. streamAssistantResponse(...)  -> AssistantMessage
             emits message_update per delta, message_end at completion
       10. stopReason branch:
             "error" | "aborted"  -> synthesize aborted ToolResultMessage for every
                                     dangling toolCall (keeps tool_use/tool_result
                                     pairing legal), emit turn_end, agent_end, RETURN
             "toolUse" | "stop"   -> runnable; tool calls execute
             "length"             -> truncated: pair placeholders, do NOT execute
       11. soft-tool-requirement gate (see below)
       12. executeToolCalls(...)                     agent-loop.ts:2220
       13. push results into currentContext.messages + newMessages
       14. emitTurnEnd(...) -> config.onTurnEnd(...)
       15. pendingMessages = getSteeringMessages() (+ getAsideMessages() if mid-work)
      // inner loop exits when no tool calls and no pending messages
      onBeforeYield()
      lateSteering = getSteeringMessages()
      asides       = getAsideMessages()
      followUps    = getFollowUpMessages()
      if any -> pendingMessages = [...] and continue outer loop
      else break
    endAgentStream -> { type: "agent_end", messages, telemetry?, coverage? }

### Tool execution details (agent-loop.ts)

- **prepareToolCallDispatch** (line 2135) resolves each toolCall to an AgentTool,
  validates arguments against the schema (validateToolArguments), applies
  beforeToolCall hooks, and computes concurrency mode. Failures are recorded per
  call and surfaced at the record's scheduled slot — never reordered.
- **executeToolCalls** (line 2220) schedules the batch: tools with
  concurrency "shared" run in parallel, "exclusive" run alone. concurrency may
  be a function of the (raw, partial) args.
- Between tool calls the loop polls **hasSteeringMessages()** (non-consuming). If
  steering is queued and interruptMode is "immediate", remaining calls in the
  batch are skipped with synthetic skipped results.
- A tool declaring **interruptible: true** (e.g. hub wait) can be aborted
  mid-execution to deliver steering; polled every STEERING_INTERRUPT_POLL_MS = 250ms.
- **createSyntheticToolResultMessage(toolCall, reason, msg)** (line 2826) builds
  the placeholder results; SyntheticToolResultDetails carries __synthetic: true
  so renderers can distinguish them.

### Notable loop-level guards

- MAX_PAUSED_TURN_CONTINUATIONS = 8 — caps re-sampling on non-terminal
  stopDetails.type === "pause_turn".
- MAX_SOFT_TOOL_ESCALATIONS = 3 — caps forced toolChoice escalations.
- config.deadline (epoch ms) wires a TimeoutError AbortSignal via AbortSignal.any.
- HarmonyLeakInterruption — GPT-5 Harmony protocol leak detection with
  abort-retry (2x) then truncate-and-resume (2x) then escalate.

### Key exported types

    // packages/agent/src/types.ts
    type AgentMessage = Message | CustomAgentMessages[keyof CustomAgentMessages]
    // CustomAgentMessages is an empty interface extended by hosts via
    // declaration merging -> this is how coding-agent adds CustomMessage.

    interface AgentContext {
      systemPrompt: string[]      // ordered blocks, provider keeps them distinct
      messages: AgentMessage[]
      tools?: AgentTool<any>[]
    }

    interface AgentState {
      systemPrompt: string[]; model: Model; thinkingLevel?: Effort
      disableReasoning?: boolean; tools: AgentTool<any>[]
      messages: AgentMessage[]; isStreaming: boolean
      streamMessage: AgentMessage | null
      pendingToolCalls: Set<string>; error?: string
    }

    type AgentEvent =
      | { type: "agent_start" }
      | { type: "agent_end"; messages: AgentMessage[]; telemetry?; coverage? }
      | { type: "turn_start" }
      | { type: "turn_end"; message: AgentMessage; toolResults: ToolResultMessage[] }
      | { type: "message_start"; message }
      | { type: "message_update"; message; assistantMessageEvent }
      | { type: "message_end"; message }
      | { type: "tool_execution_start"; toolCallId; toolName; args; intent? }
      | { type: "tool_execution_update"; toolCallId; toolName; args; partialResult }
      | { type: "tool_execution_end"; toolCallId; toolName; result; isError? }

**AgentLoopConfig** (the whole extension surface — every field is a host callback):

    interface AgentLoopConfig extends SimpleStreamOptions {
      model: Model
      interruptMode?: "immediate" | "wait"
      sessionId?: string
      deadline?: number                       // epoch ms
      metadataResolver?: (provider) => Record<string, unknown> | undefined
      convertToLlm: (AgentMessage[]) => Message[] | Promise<Message[]>       // REQUIRED
      transformContext?: (AgentMessage[], signal?) => Promise<AgentMessage[]>
      transformProviderContext?: (Context, Model) => Context | Promise<Context>
      getApiKey?: (Model) => ApiKey | undefined | Promise<...>
      getSteeringMessages?: () => Promise<AgentMessage[]>      // CONSUMING
      hasSteeringMessages?: () => boolean | SteeringQueueState | Promise<...>  // PEEK
      waitForSteeringMessages?: (signal?) => Promise<void>
      hasIrcInterrupts?: () => boolean | Promise<boolean>
      getFollowUpMessages?: () => Promise<AgentMessage[]>
      getAsideMessages?: () => Promise<AsideMessage[]>          // non-interrupting
      getToolChoice?: () => ToolChoiceDirective | undefined
      beforeModelCall?: AgentBeforeModelCall
      syncContextBeforeModelCall?: (AgentContext) => Promise<void>
      onTurnEnd?: (messages, signal?, ctx?: AgentTurnEndContext) => Promise<void>
      onBeforeYield?: () => Promise<void>
      resolveFallbackTool?: (name) => AgentTool<any> | undefined
      softToolRequirementState?: SoftToolRequirementState
      telemetry?: AgentTelemetryConfig
      ...
    }

### Steering vs. asides vs. follow-ups — a design worth copying

Three distinct injection channels with different urgency:

| Channel | Interrupts running tools? | Drained where |
|---|---|---|
| **steering** | Yes (peek via hasSteeringMessages, abort batch) | loop start + after each tool batch + at yield |
| **asides** | Never | after each tool batch (mid-work) and at yield |
| **follow-ups** | Never | only at yield boundary |

Asides are lazy: **AsideMessage = CommittableAsideMessage | (() => CommittableAsideMessage | null)**.
A thunk decides at injection time whether the message is still worth sending
(e.g. late LSP diagnostics superseded by a newer edit -> return null).
Commit/discard is signalled through symbols ASIDE_MESSAGE_COMMIT /
ASIDE_MESSAGE_DISCARD attached to the message object.

---

## 2. Session / agent lifecycle

### Files

| Path | Role |
|---|---|
| packages/coding-agent/src/sdk.ts | createAgentSession(CreateAgentSessionOptions) -> CreateAgentSessionResult (line 1227) |
| .../session/agent-session.ts | AgentSession class (line 434) |
| .../session/agent-session-types.ts | AgentSessionConfig, PromptOptions, FollowUpOptions, AsyncJobSnapshot |
| .../session/session-manager.ts | Transcript journal (JSONL), buildSessionContext, appendCustomEntry |
| .../session/session-entries.ts | SessionEntry union incl. CustomEntry |
| .../registry/agent-registry.ts | AgentRegistry — process-global id -> AgentRef |
| .../registry/agent-lifecycle.ts | AgentLifecycleManager — idle -> parked -> revived |
| .../registry/persisted-agents.ts | Cold revival of parked refs from disk |
| .../task/index.ts | TaskTool (line 509) — the spawn surface |
| .../task/executor.ts | runSubprocess (line 2528), ExecutorOptions (305), lifecycle helpers |
| .../task/types.ts | AgentDefinition, SingleResult, AgentProgress, TaskParams, TaskItem |
| .../task/agents.ts | Bundled agents (scout, designer, reviewer, security-reviewer, librarian, task, sonic) |
| .../irc/bus.ts | IrcBus — process-global mailbox |
| .../tools/hub/index.ts | HubTool — unified messaging + jobs + processes |

### Registry model

    type AgentStatus = "running" | "idle" | "parked" | "aborted"
    type AgentKind   = "main" | "sub" | "advisor"

    interface AgentRef {
      id: string                 // "Main", "AuthLoader", ... — flat namespace
      displayName: string
      kind: AgentKind
      parentId?: string
      status: AgentStatus
      session: AgentSession | null    // null exactly when parked/aborted
      sessionFile: string | null
      createdAt: number; lastActivity: number
      activity?: string          // display-only work gist, normalized via oneLineLabel
    }

Key semantics:
- Finished agents become **idle, not removed**. They stay addressable.
- MAIN_AGENT_ID = "Main". listVisibleTo(id) returns every other running/idle
  non-advisor ref — **flat namespace, every agent sees every other agent**.
- "aborted" is a terminal tombstone: setStatus refuses to move off it.
- registerIfAvailable(input, expected) is a CAS so a delayed reviver cannot
  claim an id whose generation disappeared.
- setActivity() deliberately does NOT emit an event (per-tool-call rate would
  hammer listeners); it is read on demand by rosters.

### Park / revive

**AgentLifecycleManager** (registry/agent-lifecycle.ts) is the only thing that
flips parked <-> idle.

    adopt(id, { idleTtlMs, revive?: AgentReviver }, expected?)
      -> arms a TTL timer whenever the ref goes idle
    park(id)
      -> detach session + set status "parked" BEFORE session.dispose(),
         so concurrent ensureLive/hub-send never see a disposing session.
         Cancelable up to the detach point (a same-tick ensureLive wins).
    ensureLive(id) -> Promise<AgentSession>
      -> cancel an in-flight park (keep the live session), or await it and revive
      -> concurrent calls coalesce on one in-flight revive
    setPersistedSubagentReviverFactory(factory, idleTtlMs)
      -> cold-revive parked refs restored from disk (Agent Hub scan, resumed process)

Every adoption, park, and revival is bound to the exact AgentRef it started from,
so stale async work can never clobber a newer same-id ref. This bookkeeping is the
bulk of the file's complexity and it is genuinely load-bearing.

### Subagent spawning (task tool)

TaskTool.parameters is **dynamically generated** by getTaskSchema (task/types.ts)
from live settings — batch on/off, isolation on/off, effort on/off, and the
default agent name. Two wire shapes:

    // flat form (task.batch off)
    { name?: string, agent?: string, task: string, effort?: "lo"|"med"|"hi",
      outputSchema?: unknown, schemaMode?: "permissive"|"strict", isolated?: boolean }

    // batch form (task.batch on)
    { context: string, tasks: TaskItem[] }

Runtime TaskParams stays permissive over both so stale transcripts keep working.

Spawn path (task/index.ts):
1. Resolve spawn policy (task/spawn-policy.ts) — which agent types this session
   may spawn, recursion depth cap via canSpawnAtDepth(maxRecursionDepth, taskDepth).
2. Acquire **#spawnSemaphore** (task/parallel.ts Semaphore), sized from
   settings task.maxConcurrency, resized live on every access.
3. Detached vs. blocking: an AgentDefinition with **blocking: true** makes the
   parent wait inline; otherwise the spawn is registered as an async job
   (asyncJobManager.register("task", label, run, { ownerId, agentId, queued: true })).
   The run context calls **markRunning()** after the semaphore is acquired, so a
   large parked batch does not consume the running-job budget.
4. runSubprocess(ExecutorOptions) (task/executor.ts:2528) builds a whole child
   AgentSession via createAgentSession, inheriting from the parent:
   authStorage, modelRegistry, mcpManager, settings, contextFiles, skills,
   workspaceTree, rules, extension paths, custom-tool paths, artifact manager,
   eval session id, telemetry config, local:// protocol root, service tier.
   That inheritance list is the performance story: the child skips all discovery.
5. Progress streams back over the EventBus on three channels:
   TASK_SUBAGENT_EVENT_CHANNEL, TASK_SUBAGENT_PROGRESS_CHANNEL,
   TASK_SUBAGENT_LIFECYCLE_CHANNEL.

Result delivery: the subagent must call the hidden **yield** tool
(tools/yield.ts, HIDDEN_TOOLS). YieldItem:

    interface YieldItem {
      data?: unknown
      status?: "success" | "aborted"
      error?: string
      type?: string | string[]     // string = terminal, string[] = incremental section
      useLastTurn?: boolean
      schemaOverridden?: boolean
    }

finalizeSubprocessOutput (executor.ts:564) reconciles yield items, exit code,
stderr, and optional output-schema validation into a SingleResult. Missing yield
produces SUBAGENT_WARNING_MISSING_YIELD after 3 reminders.

Budget guard: SOFT_REQUEST_BUDGET { scout: 100, sonic: 100, default: 200 }.
Crossing it steers in buildBudgetNotice(...) telling the agent to wrap up;
at 1.5x the run is force-stopped with BUDGET_STOP_GRACE_REQUESTS = 5 extra
requests allowed for the forced yield to land.

### Peer messaging (IRC / hub)

**IrcBus** (irc/bus.ts) is a process-global mailbox, NOT an auto-reply RPC.

    interface IrcMessage { id, from, to, body, ts, replyTo? }
    interface IrcDeliveryReceipt {
      to: string
      outcome: "injected" | "woken" | "revived" | "failed"
      error?: string
    }

send() resolution order:
1. Look up recipient in AgentRegistry. Unknown / aborted / advisor -> failed.
2. If parked (or lifecycle has an in-flight park) -> await lifecycle.ensureLive().
3. If the recipient has a pending **wait()** waiter -> hand the message straight
   to it, never touching the mailbox.
4. Otherwise -> session.deliverIrcMessage(msg, opts), which injects it as a
   **non-interrupting aside** at the next step boundary (busy agent) or wakes an
   idle agent with a real turn.
5. Only a failed live hand-off buffers into the mailbox (MAILBOX_CAP = 100,
   oldest dropped). Successful delivery never lingers — otherwise a later
   wait/inbox would double-deliver.

opts.expectsReply (from hub send await:true) lets a recipient that cannot reach a
step boundary generate an ephemeral side-channel auto-reply instead of stranding
the sender.

wait(agentId, {from?}, timeoutMs, signal?, {drainPending?, liveness?}) parks a
waiter; the **liveness** option aborts the wait when no matching running peer
remains — this is what stops an agent blocking forever on a dead sender.

### HubTool — one tool, three op families (tools/hub/index.ts)

    op: "send" | "wait" | "inbox" | "list"          // messaging
      | "jobs" | "cancel"                           // async jobs
      | "start" | "ps" | "logs" | "stop" | "restart" | "describe"  // processes

Disambiguation is by field, not by op name: send with **to** is a peer DM,
send with **name** is process stdin; wait with **from** waits on a message,
with **ids** on jobs, with **name** on a process, bare waits on the first of
anything. Approval tier is computed per-call by hubApproval(params) — read for
messaging/inspection, exec for process mutation.

---

## 3. Scheduled wakes

Three files, cleanly split pure / live / persistence. This is the tidiest
subsystem in the repo and the best 1:1 port candidate.

| Path | Role |
|---|---|
| .../wake/schedule.ts | Pure: shape, parsing, recurrence math. Zero timers. |
| .../wake/scheduler.ts | Live: WakeScheduler — owns schedules + one armed timer |
| .../wake/store.ts | Persistence contract + delivered-message formatting |
| .../tools/wake.ts | The wake tool (op omitted = create, "list", "cancel") |

### Shape

    interface WakeSchedule {
      id: string            // "w1", "w2" — stable per-session handle
      message: string       // the self-prompt delivered on fire
      nextDueAt: number     // epoch ms
      everyMs?: number      // absent => one-shot
      untilAt?: number      // hard stop
      limit?: number        // retire after N deliveries
      firedCount: number
      createdAt: number
    }

    interface DueWake {
      schedule: WakeSchedule
      occurrence: number    // 1-based = firedCount + 1 at fire time
      plannedTotal?: number
      final: boolean
    }

    type WakeRetireReason = "limit" | "until" | "one-shot" | "cancelled"

### Constants (wake/schedule.ts)

    MIN_WAKE_INTERVAL_MS  = 60_000   // a wake starts a full turn; sub-minute starves the user
    MAX_WAKE_SCHEDULES    = 16
    MAX_WAKE_MESSAGE_CHARS = 2_000
    PAST_AT_GRACE_MS      = 5_000

### Parsing

- parseWakeDuration("45s"|"30m"|"2h"|"7d"|"1w") -> ms. **A bare number is
  rejected on purpose** (60 reads as both seconds and ms; guessing wrong is a
  runaway loop).
- parseWakeAt(text, nowMs) tries, in order: "+duration", "HH:MM" (next
  occurrence of that local wall-clock time; uses setDate(+1) not +24h so DST
  keeps the requested clock time), then Date.parse for ISO-8601.
- buildWakeSchedule(request, existing, nowMs) returns
  { schedule } | { error: string } — **it returns the error text rather than
  throwing**, so the tool's failure path is a sentence the model can act on.
- advanceWakeSchedule(schedule, nowMs) -> { next } | { retired: reason }.
  Missed occurrences are **skipped, not replayed**: a laptop asleep six hours
  owes one hourly check, not six.

### Firing (wake/scheduler.ts)

    class WakeScheduler {
      constructor(options: WakeSchedulerOptions)  // now, deliver, persist, onRetire, setTimer, clearTimer
      get schedules(): readonly WakeSchedule[]
      load(schedules)    // adopt persisted; NO persist (would duplicate per resume)
      update(schedules)  // caller-driven change; persists + re-arms
      pump(nowMs?): number | undefined
      dispose()
    }

Three load-bearing properties:

1. **MAX_ARM_MS = 60_000** — a wake a week out arms a one-minute re-check tick
   rather than a 604,800,000 ms timeout. Laptop sleep, clock adjustment and
   timezone changes are absorbed by re-reading the wall clock.
2. **timer.unref()** — a pending wake never keeps the process alive. A wake is a
   promise to the *session*, not the machine.
3. **LOAD_GRACE_MS = 2_000** — an overdue wake adopted at resume fires shortly
   *after* load, not inside it, so the TUI has attached and the wake appears
   live in the conversation instead of replaying from history.

Also: MIN_ARM_MS = 25 (no zero-delay re-entry loop); a delivery that throws still
advances the schedule (otherwise one broken wake becomes a hot loop); the kept
list is re-sorted by createdAt so ids and listings stay stable across fires.

### Persistence (wake/store.ts)

Schedules live **in the session transcript** as a CustomEntry with
customType = WAKE_SCHEDULES_CUSTOM_TYPE ("wake_schedules") — not a side file.
Rationale: /resume, --resume, fork and tree navigation already restore the
transcript, so a wake follows its conversation with no second source of truth.
CustomEntry is ignored by buildSessionContext, so the schedule list never enters
LLM context — only the fired message does.

getLatestWakeSchedulesFromEntries(entries) scans **backward and stops at the
first hit**: each change appends a full snapshot, so the newest entry wins and
malformed rows are dropped individually via isWakeSchedule().

Delivered text (formatWakeDeliveryText) is one envelope line then the verbatim
message:

    (alarm) Scheduled wake w1 (3/8, every 1h) — cancel with wake({op:"cancel",id:"w1"}) once its goal is met.

    <the agent's own message>

The envelope always carries the handle, because an agent that has to guess its
own wake id cannot honour "stop when the goal is met". A final delivery drops
the cancel hint (the schedule is already gone).

Session wiring: agent-session.ts:1010 constructs the WakeScheduler with
deliver -> #deliverWake (injects the self-prompt), persist -> appendCustomEntry.
Message type WAKE_PROMPT_MESSAGE_TYPE = "wake-prompt" with WakePromptDetails
for TUI dispatch.

---

## 4. Parallelism — jobs and waiting

### Files

| Path | Role |
|---|---|
| .../async/job-manager.ts | AsyncJobManager (822 lines) |
| .../session/yield-queue.ts | YieldQueue — batched aside delivery |
| .../task/parallel.ts | Semaphore, mapWithConcurrencyLimit, mapWithConcurrencyLimitAllSettled |
| .../tools/hub/jobs.ts | hub job ops: wait / jobs / cancel |

### AsyncJob

    interface AsyncJob {
      id: string
      type: "bash" | "task"
      status: "running" | "completed" | "failed" | "cancelled"
      startTime: number
      label: string
      abortController: AbortController
      promise: Promise<void>
      resultText?: string; errorText?: string
      latestDetails?: Record<string, unknown>
      ownerId?: string      // registry id of the agent that registered it
      agentId?: string      // registry id of the subagent this job RUNS
      queued?: boolean      // registered but parked behind a caller-managed gate
    }

    register(type, label, run, options?) -> jobId
      run receives { jobId, signal, reportProgress, markRunning }

Defaults: DEFAULT_MAX_RUNNING_JOBS = 15, DEFAULT_RETENTION_MS = 5 min.
**queued jobs hold no execution slot** — atCapacity and register() both count
only (status === "running" && !queued), so a large parked task batch cannot
starve registration.

### Delivery — the part most systems get wrong

    type AsyncJobDeliverySink = (jobId, text, job?) => void | Promise<void>
    registerDeliverySink(ownerId, sink)

Owned completions route **exclusively** through the owner's registered sink.
If the owner has no live sink the delivery is **dead-lettered** (dropped with a
warning; the job row keeps resultText until retention eviction) — it is never
routed to the AsyncJobManagerOptions.onJobComplete fallback, because that would
leak one agent's result into another agent's session. Only genuinely unowned
jobs use onJobComplete.

Retry: exponential backoff DELIVERY_RETRY_BASE_MS 500 -> DELIVERY_RETRY_MAX_MS
30_000, jitter 200ms, tracked in AsyncJobDelivery { jobId, text, attempt,
nextAttemptAt, lastError?, ownerId?, promise? }, exposed as:

    interface AsyncJobDeliveryState {
      queued: number; delivering: boolean
      nextRetryAt?: number; pendingJobIds: string[]
    }

Scoping: AsyncJobFilter { ownerId } gates cancel/list. cancel(id, {ownerId})
treats an owner mismatch as not-found, so a subagent teardown cannot cancel its
parent's jobs.

### Adaptive poll backoff (a genuinely clever bit)

    POLL_WAIT_LADDER_MS = [5_000, 10_000, 30_000, 60_000, 300_000]
    POLL_ESCALATION_RESET_MS = 60_000

When async.pollWaitDuration is "smart", each immediate re-poll climbs a rung, so
a tight poll loop stops burning turns on "still running" frames. Going longer
than 60s between polls means the agent stepped out to do real work, and the
next poll drops back to the ladder floor. State per owner:
PollEscalationState { level, lastPollEndAt }.

### YieldQueue — batching asides into the loop

    interface YieldDispatcher<P> {
      isStale?(entry: P): boolean         // evaluated at flush time, per entry
      build(survivors: P[]): AgentMessage | null   // one batched message per kind
      skipIdleFlush?: boolean
    }

    class YieldQueue {
      register<P>(kind, dispatcher): () => void
      enqueue<P>(kind, entry)
      enqueueWithReceipt<P>(kind, entry): Promise<void>
      flush(mode: "streaming" | "idle"): Promise<void>
      drainLazy(): Array<() => AgentMessage | null>
      clear(kind?)
    }

**drainLazy()** is the bridge to the loop: it snapshots and removes all entries
and returns one **thunk per kind**. The thunk applies staleness filtering and
builds the batched message only when the loop actually injects it — and may
return null to skip. agent-session.ts:1204 feeds these thunks into
getAsideMessages alongside drained IRC messages and a mid-run todo nudge:

    const thunks: AsideMessage[] = this.#irc.drainPending().map(record => () => record);
    thunks.push(...this.yieldQueue.drainLazy());
    thunks.push(() => this.#todo.takeMidRunNudge());

Receipts settle through ASIDE_MESSAGE_COMMIT / ASIDE_MESSAGE_DISCARD attached
via Object.defineProperties on the built message — so a producer awaiting
enqueueWithReceipt learns whether its entry actually reached the model.

### Concurrency primitives (task/parallel.ts)

    class Semaphore { constructor(max); acquire(signal?); release(); resize(max) }
    mapWithConcurrencyLimit<T,R>(items, concurrency, fn, signal?) -> ParallelResult<R>
    mapWithConcurrencyLimitAllSettled<T,R>(...)                   -> ParallelSettledResult<R>
    normalizeConcurrencyLimit(max)   // <= 0 means unlimited, returns 0

---

## 5. Headless / exec mode

Two unrelated things share the word "exec" here — be careful:

### 5a. Headless *agent* execution = print mode

| Path | Role |
|---|---|
| .../modes/print-mode.ts | runPrintMode(session, PrintModeOptions) |
| .../main.ts:1769-1783 | Lazy import + dispatch (keeps print code out of TUI startup) |
| .../cli/args.ts:260 | --print / -p, --print-thoughts |

    interface PrintModeOptions {
      mode: "text" | "json"
      messages?: string[]
      initialMessage?: string
      initialImages?: ImageContent[]
      printThoughts?: boolean
    }

Mode selection (main.ts:1271-1277):

    pipedInput  = isProtocolMode ? undefined : await readPipedInput()
    autoPrint   = pipedInput !== undefined && !args.print && args.mode === undefined
    isInteractive = !args.print && !autoPrint && args.mode === undefined
    setInteractiveHost(isInteractive)

So piping stdin auto-selects print mode. setInteractiveHost() lets headless
subagent code paths skip focusable-UI work (e.g. replan title refresh).

runPrintMode flow:
1. json mode: emit the session header line first.
2. initializeExtensions(session, { reportSendError, reportRuntimeError }).
3. Optional plan-mode default arming (plan.defaultOnStartup) — including a
   plan-proposal handler that aborts the run once the plan is proposed.
4. session.subscribe(event => ...) — always subscribed, because that is what
   drives session persistence via _handleAgentEvent. In json mode each event
   is written as one JSON line through **printableEvent(event)**.
5. await session.prompt(initialMessage, { images }) then each of messages[]
   sequentially.
6. session.prepareForHeadlessAdvisorDrain().
7. text mode: read session.getLastAssistantMessage() (NOT the raw state tail —
   a classifier refusal is pruned at settle and an aborted turn can trail
   synthetic tool results). error/aborted -> stderr + exit(1) after flushing
   telemetry and disposing. Otherwise write text blocks (and thinking blocks
   when printThoughts).
8. waitForAdvisorCatchup(PRINT_MODE_ADVISOR_DRAIN_TIMEOUT_MS = 10 min;
   30s on the error path), flush stdout, session.dispose().

**printableEvent** is the token/IO-efficiency trick for --mode json: it drops
message_update snapshots (message, assistantMessageEvent.partial, done/error
payloads) keeping only the incremental delta, and strips providerPayload
everywhere. Without it a single long turn re-serialized its whole in-progress
message on every delta, producing multi-GB logs — quadratic growth turned linear.

### 5b. exec/ — subprocess helpers, not a mode

| Path | Role |
|---|---|
| .../exec/exec.ts | execCommand(command, args, cwd, ExecOptions) -> ExecResult |
| .../exec/non-interactive-env.ts | NON_INTERACTIVE_ENV + buildNonInteractiveEnv() |
| .../exec/bash-executor.ts | The bash tool's executor (23 KB) |
| .../exec/direnv.ts | direnv integration |

    interface ExecOptions { signal?; timeout?; cwd? }
    interface ExecResult { stdout: string; stderr: string; code: number; killed: boolean }

**NON_INTERACTIVE_ENV** is a ~45-entry table worth stealing verbatim: PAGER=cat
and every tool-specific pager variant, TERM=dumb, NO_COLOR=1, PYTHONUNBUFFERED=1,
GIT_EDITOR/VISUAL/EDITOR=true, GIT_TERMINAL_PROMPT=0, SSH_ASKPASS=/usr/bin/false,
CI=1, npm/pnpm/yarn quiet flags, CARGO_TERM_PROGRESS_WHEN=never,
DEBIAN_FRONTEND=noninteractive, PIP_NO_INPUT=1, TF_INPUT=0, TF_IN_AUTOMATION=1,
COMPOSER_NO_INTERACTION=1, CLOUDSDK_CORE_DISABLE_PROMPTS=1.
buildNonInteractiveEnv() additionally forces UTF-8 groups on win32 unless the
caller already set them.

### 5c. Other non-TUI modes

- **modes/acp/** — Agent Client Protocol (editor hosts).
- **modes/rpc/** — JSON-RPC mode.
- **jsonrpc/message-framing.ts** — shared Content-Length framing for LSP and DAP
  stdio clients. MessageFramer buffers a **Buffer[] chunk list** and only
  concatenates when a full message is framed; naive per-read concatenation is
  O(n^2) for messages spanning many reads. push(chunk), *drain(onResync),
  remainder(). Bogus header without Content-Length triggers onResync and skips
  past the terminator instead of stalling forever.
- **subprocess/worker-client.ts** — shared scaffolding for ONNX inference
  subprocesses (embeddings, STT, TTS, tiny models). Each runs onnxruntime-node
  in a dedicated Bun child process because the NAPI destructors segfault Bun on
  shutdown. WorkerHandle { send, onMessage, onError, terminate };
  RefCountedWorkerHandle adds ref()/unref() so a pending request keeps the loop
  alive but an idle worker never blocks exit. SpawnedSubprocess carries
  intentionalExit (distinguishes deliberate SIGKILL from SIGSEGV/OOM) and
  stderrDrained (so a crash report carries the whole tail).

---

## 6. Token / cache efficiency

This is where the design is most opinionated and most worth porting.

### 6a. Append-only context (packages/agent/src/append-only-context.ts)

Two cooperating mechanisms so provider prefix caches (Anthropic, DeepSeek,
llama.cpp KV) hit maximally:

    class StablePrefix {
      build(context: AgentContext, options: BuildOptions): boolean  // true = changed
      invalidate()
      toContext(): { systemPrompt: string[]; tools: Tool[] }
      get fingerprint(): string; get version(): number; get built(): boolean
    }
    interface StablePrefixSnapshot { systemPrompt: string[]; tools: Tool[]; fingerprint: string }
    interface BuildOptions { intentTracing: boolean; pruneToolDescriptions?: boolean }

    class AppendOnlyLog {
      append(m); extend(ms); replaceTail(m)   // replaceTail legal ONLY for compaction
      toMessages(): Message[]; entries(): readonly Message[]
      truncate(count); clear()
    }

    class AppendOnlyContextManager {
      readonly prefix: StablePrefix
      readonly log: AppendOnlyLog
      build(context, options): Context
      syncMessages(normalizedMessages): void
      invalidateForModelChange(); resetSyncCursor(); reset(context, options)
    }

**syncMessages is the interesting method.** Three cases:

1. **Append** — same prefix, new tail: push the delta.
2. **Compaction** — array shrank: clear and replay.
3. **In-place rewrite** — per-turn pruning, transformContext re-render, image
   strip, steering re-wrap: compute the **longest byte-stable prefix** between
   the previously-synced digests and the new messages, truncate the log to that
   point, append the diverged tail.

Case 3 is a real bug fix (issue #3406): earlier revisions cleared the whole log
on any digest change, which on llama.cpp / local backends forced a full ~40k-token
re-prefill every turn an extension rewrote a single message.

The digest (#messageDigest) covers **every field the provider may serialize**:
role, content, providerPayload, toolCalls AND OpenAI-wire tool_calls,
toolCallId/tool_call_id, toolName/name, isError, assistant id. Missing any of
these makes an in-place rewrite invisible and silently corrupts the prefix.

Session wiring (agent-session.ts:7070): #syncAppendOnlyContext(model) reads
setting provider.appendOnlyContext ("auto" default), calls
shouldEnableAppendOnlyContext(setting, model), and installs/invalidates/removes
the manager. Model switches, session resets, and stale-replay recovery all call
agent.appendOnlyContext?.invalidateForModelChange().

### 6b. System prompt construction (coding-agent/src/system-prompt.ts, 922 lines)

    buildSystemPrompt(options: BuildSystemPromptOptions): Promise<BuildSystemPromptResult>
    interface BuildSystemPromptResult {
      systemPrompt: string[]           // ordered blocks kept distinct by providers
      xdevCatalogNames?: readonly string[]
    }

Returns an **array of blocks**, not one string. Ordering:

    [0] rendered system-prompt.md (or custom-system-prompt.md when a custom prompt is set)
    [1] computer-safety.md            (only when the "computer" tool is active)
    [2] project-prompt.md             (environment, cwd, workspace tree, context files)
    [3] active-repo-context.md        (only for a nested active repo)

Blocks are separated because provider cache breakpoints are per-block: volatile
project context is isolated from the stable instruction block.

**Templates are compiled into the binary** via Bun's
`import x from "./x.md" with { type: "text" }` and rendered by
prompt.render(template, data) (a handlebars-ish renderer in @oh-my-pi/pi-utils
supporting {{#has tools "lsp"}} style gates). Prompt files live in
src/prompts/{system,tools,agents,session,security,goals,steering,skills}/.

Cache-stability techniques in this file:

- **SYSTEM_PROMPT_PREP_TIMEOUT_MS = 5000** with a per-step withDeadline()
  wrapper. Every discovery step (custom prompt, SYSTEM.md walk-up, context
  files, skills, workspace tree, active repo, CPU model, GPU probe) has an
  explicit **fallback value**. A slow step degrades to the fallback and warns;
  it never delays or destabilizes the prompt.
- **GPU probe is disk-cached** (getGpuCachePath) so the byte content is stable
  across runs; the probe itself is SIGKILL-bounded at 4500ms with a 250ms
  stdout drain for descendants holding the pipe.
- **formatLocalCalendarDate()** — the prompt carries a *date*, not a timestamp,
  so the prefix is byte-stable for a whole local day. AgentSessionConfig exposes
  getLocalCalendarDate so cache invalidation happens exactly once at midnight.
- **Deduplication** — dedupeAlwaysApplyRules / dedupePromptSource /
  dedupeExactContextFiles compare *normalized paragraph blocks*
  (splitComparablePromptBlocks) so the same rule text arriving via SYSTEM.md,
  a custom prompt, and AGENTS.md is emitted once.
- **Tool inventory has two modes.** toolListMode = !inlineToolDescriptors &&
  nativeTools. Native tool calling -> a compact **name list** only, because the
  schemas already ride in the provider tools array. In-band/owned dialects or
  inlineToolDescriptors -> renderToolInventory() emits the full Harmony-style
  `namespace functions { ... }` catalog. Never both.
- The mirror-image switch lives on the wire side: AgentOptions.pruneToolDescriptions
  strips descriptions (top-level *and* nested schema annotations) from
  provider-bound tool specs when the catalog is in the system prompt.

### 6c. Tool-schema level

- **normalizeTools(tools, { injectIntent, pruneDescriptions })** (agent-loop.ts:842)
  injects the INTENT_FIELD ("i") into every schema unless PI_NO_INTENT=1.
  Per-tool AgentTool.intent controls it: "require" (default) | "optional" |
  "omit" | a function deriving intent from partial args. Tools where intent is
  obvious (yield, resolve, todo) set "omit" and save the tokens.
- **ToolLoadMode = "essential" | "discoverable"**. Discoverable tools are removed
  from the top-level schema entirely and reached via xd:// device mounts or BM25
  tool search — keeping their schemas off every request. This is the single
  biggest per-request saving in a large tool catalog.
- **AgentToolResult.useless?: boolean** marks a contextually useless result
  (zero matches, wait timeout) as safe for compaction to elide once consumed.
- **SoftToolRequirement** exists purely to protect the message cache: the host
  wants a tool called before the loop yields, but changing tool_choice
  invalidates the provider message cache. So the loop injects a reminder once
  per requirement id, runs with toolChoice **unchanged**, and escalates to a
  one-turn forced choice only if the model actually fails to comply. A compliant
  model pays zero cache invalidation.

        interface SoftToolRequirement {
          soft: true
          id: string                 // reminder re-injects only when this changes
          toolName: string
          satisfies?(toolCall): boolean
          reminder: AgentMessage[]
        }
        interface SoftToolRequirementState { id?; forcedToolChoice?; escalations: number }

- **promptCacheKey** is separate from sessionId. Forked sessions inherit the
  parent's key (providerPromptCacheKeySource: "explicit" | "fork",
  #adoptInheritedProviderPromptCacheKey at agent-session.ts:3526) so a fork
  keeps hitting the parent's warm cache. Side requests get
  `sessionId + ":side:" + Snowflake.next()` with the same promptCacheKey.

---

## 7. Tool registration and dispatch

### Declaration — tools are classes implementing an interface

    // packages/agent/src/types.ts
    interface AgentTool<TParameters extends TSchema = TSchema, TDetails = any, TTheme = unknown>
      extends Tool<TParameters> {
      // from Tool: name, description, parameters (TSchema), strict?, examples?, native?
      label: string
      summary?: string                 // one-line, used by discovery indexes
      hidden?: boolean                 // excluded unless explicitly requested
      deferrable?: boolean
      loadMode?: ToolLoadMode          // "essential" | "discoverable"
      concurrency?: "shared" | "exclusive" | ((args) => "shared" | "exclusive")
      lenientArgValidation?: boolean   // validation errors -> raw args to execute()
      interruptible?: boolean | ((args) => boolean)
      intent?: "omit" | "optional" | "require" | ((args) => string | undefined)
      approval?: ToolApproval
      formatApprovalDetails?: (args) => string | string[] | undefined
      matcherDigest?: (args) => string | undefined         // TTSR stream matching
      matcherPaths?: (args) => readonly string[] | undefined
      matcherEntries?: (args) => readonly { path, digest }[] | undefined
      execute: AgentToolExecFn<TParameters, TDetails, TTheme>
      renderCall?: (args, RenderResultOptions, theme) => unknown
      renderResult?: (result, RenderResultOptions, theme) => unknown
    }

    type AgentToolExecFn<TP, TD, TT> = (
      this: AgentTool<TP, TD, TT>,
      toolCallId: string,
      params: Static<TP>,
      signal?: AbortSignal,
      onUpdate?: AgentToolUpdateCallback<TD, TP>,
      context?: AgentToolContext,
    ) => Promise<AgentToolResult<TD, TP>>

    interface AgentToolResult<T = any, _TInput = unknown> {
      content: (TextContent | ImageContent)[]
      details?: T                  // structured payload for renderers/logs
      isError?: boolean            // non-throwing failure
      providerMetadata?: ToolResultProviderMetadata
      useless?: boolean            // safe for compaction to elide
    }

Approval model:

    type ToolTier = "read" | "write" | "exec"
    type ToolApprovalDecision =
      | ToolTier
      | { tier: ToolTier; reason?: string; override?: boolean; policy?: "allow"|"deny"|"prompt" }
    type ToolApproval = ToolApprovalDecision | ((args: unknown) => ToolApprovalDecision)

Omitted approval is treated as "exec". The function form is what lets HubTool
charge read tier for a peer DM and exec tier for writing to a process stdin.

**AgentToolContext is an empty interface** extended by hosts via declaration
merging — same trick as CustomAgentMessages. That is how coding-agent injects
approvalMode, per-tool policies, and UI handles without pi-agent-core knowing.

### Schemas

Parameters use **@oh-my-pi/omptype** (an ArkType wrapper):
type({ "op?": type("'a'|'b'").describe("...") }), inferred via typeof schema.infer.
Schemas can be **built dynamically per session** (getTaskSchema in task/types.ts
caches per "iso|flat:batch|single:effort|default:<agent>" key) so the model only
ever sees the shape the current settings actually support.

### Registration (coding-agent/src/tools/index.ts, 748 lines)

    type ToolFactory = (session: ToolSession) => Tool | null | Promise<Tool | null>

    const BUILTIN_TOOLS: Record<BuiltinToolName, ToolFactory> = {
      read: s => new ReadTool(s),
      bash: s => new BashTool(s),
      edit: s => new EditTool(s),
      write: s => new WriteTool(s),
      grep, glob, ast_grep, ast_edit, eval, task, hub, todo, wake, web_search,
      browser, computer, inspect_image, lsp, github, debug, ask, checkpoint,
      rewind, security_scan, memory_edit, retain, recall, reflect, learn,
      manage_skill,
      // factories returning null are conditionally unavailable: XTool.createIf
    }
    const HIDDEN_TOOLS: Record<HiddenToolName, ToolFactory> = {
      yield: s => new YieldTool(s),
      goal:  s => new GoalTool(s),
    }

    createTools(session: ToolSession, toolNames?: string[]): Promise<Tool[]>

The **createIf** convention is how availability is expressed: a factory returns
null when the tool cannot exist in this session (missing binary, disabled
setting, wrong provider). No separate capability table.

**ToolSession** (tools/index.ts:155) is the dependency-injection bundle passed
to every factory — roughly 100 optional fields. It is effectively the harness's
service locator. Groups:

- Filesystem/workspace: cwd, additionalDirectories, contextFiles, workspaceTree,
  skills, rules, promptTemplates
- Capability flags: hasUI, enableLsp, enableIrc, enableMCP, hasEditTool,
  restrictToolNames, requireYieldTool, taskDepth
- Identity/accessors: getSessionId, getSessionFile, getAgentId, getActiveModel,
  getArtifactsDir, getEvalSessionId
- Coordination: agentRegistry, agentLifecycle, asyncJobManager, mcpManager,
  eventBus, toolRegistry, xdev
- Session mutation hooks: steer(), queueDeferredMessage(), queueDeferredDiagnostics(),
  setWakeSchedules(), setTodoPhases(), setCheckpointState(), setPlanProposalHandler()
- Lazily-initialized per-session stores: fileSnapshotStore, editClipboard,
  conflictHistory, diagnosticsLedger, noopLoopGuard
- Lifecycle: registerDisposeCallback(), registerSessionChangeCallback(), isDisposed()

### Dispatch

1. resolveToolForCall(context.tools, toolCall, resolveFallbackTool) — the
   fallback resolver is what routes calls to tools exposed only through xd://
   device mounts instead of failing with "Tool not found".
2. validateToolArguments against the schema; lenientArgValidation passes raw
   args through instead of erroring back to the model.
3. Scheduling by concurrency mode; exclusive tools run alone.
4. execute(toolCallId, params, signal, onUpdate, context) — onUpdate streams
   partial results, surfaced as tool_execution_update events.
5. **coerceToolResult(raw)** (agent-loop.ts, before line 516) normalizes whatever
   came back: invalid content blocks are counted and reported, isError is set,
   and — importantly — an error result with empty content is backfilled with
   EMPTY_ERROR_TOOL_RESULT_TEXT because **Anthropic rejects tool_result blocks
   with is_error: true and empty content**.

---

## 8. Type/interface index (quick reference)

| Type | File | Note |
|---|---|---|
| AgentLoopConfig | packages/agent/src/types.ts | The entire host extension surface |
| AgentContext, AgentState, AgentEvent | same | |
| AgentTool, AgentToolResult, AgentToolContext, ToolApproval, ToolLoadMode | same | |
| AsideMessage, SoftToolRequirement, SteeringQueueState | same | |
| StablePrefix, AppendOnlyLog, AppendOnlyContextManager | packages/agent/src/append-only-context.ts | |
| AgentSessionConfig, PromptOptions, FollowUpOptions, AsyncJobSnapshot | session/agent-session-types.ts | |
| YieldDispatcher, YieldQueue | session/yield-queue.ts | |
| AgentRef, AgentStatus, AgentKind, RegistryEvent | registry/agent-registry.ts | |
| AgentReviver, AdoptOptions, PersistedSubagentReviverFactory | registry/agent-lifecycle.ts | |
| IrcMessage, IrcDeliveryReceipt | irc/bus.ts | |
| AsyncJob, AsyncJobDeliverySink, AsyncJobDeliveryState, AsyncJobFilter | async/job-manager.ts | |
| WakeSchedule, DueWake, WakeRetireReason, WakeCreateRequest | wake/schedule.ts | |
| WakeSchedulerOptions | wake/scheduler.ts | |
| WakePromptDetails | wake/store.ts | |
| AgentDefinition, SingleResult, AgentProgress, TaskParams, TaskItem, YieldItem, TaskToolDetails | task/types.ts | |
| ExecutorOptions | task/executor.ts:305 | ~60 fields of parent inheritance |
| ToolSession, ToolFactory | tools/index.ts | |
| BuildSystemPromptOptions, BuildSystemPromptResult, SystemPromptToolMetadata | system-prompt.ts | |
| PrintModeOptions | modes/print-mode.ts | |
| ExecOptions, ExecResult | exec/exec.ts | |
| WorkerHandle, RefCountedWorkerHandle, SpawnedSubprocess | subprocess/worker-client.ts | |
| MessageFramer | jsonrpc/message-framing.ts | |

---

## Porting notes to Python

### Maps 1:1 — port the design, not the code

| omp construct | Python equivalent |
|---|---|
| AgentLoopConfig callbacks | A Protocol / dataclass of callables, or an ABC with default no-op methods |
| The two-nested-while runLoopBody | Same structure verbatim; it is language-neutral |
| AbortSignal / AbortController | asyncio.Event or a CancelScope (anyio); AbortSignal.any -> wait on multiple events |
| yieldIfDue() | await asyncio.sleep(0) with a time-budget check |
| wake/schedule.ts (pure) | Pure module, clock injected as now: Callable[[], float]. Direct port. |
| WakeScheduler arm/pump | asyncio.TimerHandle via loop.call_later; keep MAX_ARM_MS re-check |
| AgentRegistry | Plain dict[str, AgentRef] + listener set; dataclass AgentRef |
| AgentLifecycleManager park/revive | Same state machine; asyncio.Lock per id instead of coalescing dicts |
| IrcBus | dict[str, deque] mailboxes + asyncio.Future waiters |
| AsyncJobManager | asyncio.Task registry; abortController -> task.cancel() |
| Semaphore / mapWithConcurrencyLimit | asyncio.Semaphore + asyncio.gather / TaskGroup |
| YieldQueue drainLazy thunks | list[Callable[[], Message | None]] — identical |
| AppendOnlyContextManager | Direct port; digest via hash of a canonical json.dumps(sort_keys=True) |
| NON_INTERACTIVE_ENV | Copy the dict verbatim |
| MessageFramer | Direct port over list[bytes]; keep the no-concat-until-framed property |
| printableEvent stripping | Direct port; same quadratic-log fix applies |
| Tool approval tiers + createIf factories | Same; factory returning None = unavailable |

### Needs adaptation

**1. EventStream -> async generator.**
EventStream<AgentEvent, AgentMessage[]> is push-based with a terminal predicate
and a result extractor:

    new EventStream(
      e => e.type === "agent_end",
      e => e.type === "agent_end" ? e.messages : [],
    )

In Python this is naturally AsyncIterator[AgentEvent] plus a separate
await run.result(). Two options: (a) an async generator whose final yielded
event carries the messages, or (b) an object holding an asyncio.Queue plus a
Future for the result. **(b) matches omp more closely** — the loop pushes into
the queue while callers may also await the result, and agentLoop returns the
stream *synchronously* before any work happens. An async generator cannot do
that without an explicit start.

**2. Declaration merging has no Python equivalent.**
CustomAgentMessages and AgentToolContext are empty interfaces that hosts extend
so the core stays generic while hosts get full typing. In Python:
- AgentMessage -> a Union plus a CustomMessage dataclass with
  custom_type: str and details: Any. Hosts subclass CustomMessage.
- AgentToolContext -> a Protocol the host implements, passed as Any /
  TypeVar bound in the core. Do not try to replicate open extension.

**3. TSchema / ArkType -> pydantic.**
omptype gives runtime validation + JSON Schema + static inference from one
declaration. Pydantic v2 BaseModel gives the same three. The dynamic schema
construction (getTaskSchema building a different shape per settings) becomes
pydantic.create_model(...) with a cache dict keyed the same way. Keep the
cache — rebuilding schemas per turn would churn the prompt cache.

**4. Symbols for aside commit/discard.**
ASIDE_MESSAGE_COMMIT / ASIDE_MESSAGE_DISCARD are Symbols attached to message
objects. In Python use explicit optional fields on the message dataclass
(on_commit: Callable | None, on_discard: Callable | None) — cleaner than
the symbol trick, which only exists in TS to avoid colliding with wire fields.
Since these fields must not serialize, mark them
field(default=None, compare=False, repr=False) and exclude in the encoder.

**5. Bun-specific machinery.**
- import x from "./x.md" with { type: "text" } (compile-time embedding) ->
  importlib.resources.files(pkg).joinpath("prompts/system.md").read_text(),
  or embed at build time. Keep prompts as **files**, not string literals.
- Bun.spawn / Subprocess -> asyncio.create_subprocess_exec.
- timer.unref() **has no Python equivalent.** asyncio has no unref concept. Track
  wake/job timers in a set and cancel them explicitly at shutdown, or use daemon
  semantics: the wake scheduler must not be the reason the loop keeps running.
  This is a real behavioural difference — plan for it in shutdown, not by hoping.
- The ONNX-in-a-subprocess pattern (subprocess/worker-client.ts) is a Bun/NAPI
  workaround. Python can usually run onnxruntime in-process; port the
  ref/unref-shaped *handle* only if you actually need out-of-process inference.

**6. Process-global singletons.**
AgentRegistry.global(), IrcBus.global(), AgentLifecycleManager.global(),
AsyncJobManager.instance() are module-level singletons with
resetGlobalForTests(). In Python prefer **explicit dependency injection with a
module-level default** (a contextvars.ContextVar holding the current registry)
so pytest does not need a reset hook and concurrent embedded sessions work.
Note omp already hit this: the SDK explicitly documents passing a private
AgentRegistry per createAgentSession when embedding several top-level sessions,
because the global registry admits only one "Main" per process generation.

**7. Structured concurrency changes the abort story.**
omp threads an AbortSignal manually through every layer and composes with
AbortSignal.any. Python should use **anyio/asyncio TaskGroup + CancelScope**,
which gives cancellation propagation for free but also enforces that a spawned
task cannot outlive its scope. That directly conflicts with omp's *detached*
subagent jobs, which deliberately outlive the turn that spawned them. Resolve
this explicitly: run detached jobs under a **session-scoped** TaskGroup owned by
AgentSession (not the turn), and let the turn's scope only await blocking spawns.
Getting this wrong is the most likely source of "job silently cancelled" bugs.

**8. AgentSession is 351 KB / ~9000 lines.**
Do not port it as one class. Its responsibilities are already separable and the
file names hint at the seams: messages, model-controls, retry-fallback-chains,
turn-recovery, session-entries, session-context, yield-queue, session-listing.
Suggested Python decomposition:
Session (facade) delegating to TranscriptStore, ContextBuilder,
ModelController, SteeringQueue, AsideQueue, IrcInbox,
WakeScheduler, ToolRegistry. The loop only ever sees the callback bundle,
so the decomposition is invisible to it — which is exactly the point of the
packages/agent vs packages/coding-agent split. **Reproduce that boundary first**;
everything else is easier once the loop cannot reach into session state.

### Ordered porting recommendation

1. **packages/agent** equivalent first: loop + types + append-only context. It
   is ~4 files and has no dependencies on anything else. Everything below plugs
   into it.
2. wake/ (pure + scheduler + store) — smallest complete subsystem, high value,
   near-verbatim port.
3. AsyncJobManager + YieldQueue — unlocks background work and aside delivery.
4. AgentRegistry + AgentLifecycleManager + IrcBus — multi-agent.
5. Tool registration (ToolSession-equivalent DI bundle) — but trim it. 100
   optional fields is what a 9000-line session class produces; start with the
   ~20 fields tools actually need and add by demand.
6. system-prompt block builder + print mode.

