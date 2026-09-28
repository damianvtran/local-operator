# Design — `local_operator.sdk`: one session-engagement path

**Status: accepted design (architect), reviewed and accepted by the manager
2026-09-27; carried into the tree with the implementation PR.** This is the
design of record for `local_operator/sdk.py` + `local_operator/session/spec.py`
and their consumers. File:line references were read against `origin/main @
9f4e9d8b2` (pyproject `0.63.12`) at authoring time; where a reference has since
moved, the symbol name is the stable handle.

## 0. Executive summary

The unified machinery the operator wants **already exists and is already
carrying four of the five consumers**. `session_factory.create_session` is
called "THE factory shared by `cli.py` (interactive TUI / headless REPL),
`exec_mode.run_exec` (foreground exec) and `exec_worker` (background exec)";
inside it `_prepare` is "the boundary where every open path in the product
meets". The runtime-hosted path wraps the same root, and `spawn_owned_session`
builds a session for the phone with the CLI's composition root. Delivery to a
session that may live in another process already has one router:
`engage_runtime(session_id, work: Errand)` with typed errands, used today by
the phone stack, `lop send`, and the mesh.

So the gap is **not** "wire five consumers into one loop". It is:

1. **A published, typed entry point** — today the only way in is an
   `argparse.Namespace` (exec builds exactly eight fields), undocumented as an
   API and unstable as one.
2. **Root-explicit construction by default** — `paths.py` resolves
   `config_dir()`/`agent_home_dir()` from the environment on every call, so a
   programmatic launch can silently resolve the operator's real root (the
   incident class).
3. **An ergonomic stream/approval surface** — `subscribe()` + `AgentEvent`
   exist, but there is no async-iterator adapter and no policy-preset for
   approvals.
4. **The benchmark actually using it** — today it bypasses all of the above:
   a bespoke episode runner and a reply-channel tool surface, zero references
   to `task/team/hub/todo/...`.

Recommendation: `local_operator/session/spec.py` (frozen `SessionSpec` +
`SessionRoots` + `ApprovalPolicy`) and `local_operator/sdk.py` (a thin facade
over `create_session` / `engage_runtime` / `spawn_owned_session` + an event
iterator + approval presets), as a **pure addition** with zero call-site
changes in PR 1, then the benchmark as a first-class consumer in PR 2. No new
loop, no new tool dispatch, no forked subagent runner.

## 1. The seam as found

- **One composition root, two host shapes.** In-process owner:
  `session_factory.create_session(...)` → `Session`, implementing
  `SessionProtocol` ("the one object every front end talks to"). Viewers:
  `AttachedSession` implements `ViewerSessionProtocol` on top of it — the TUI
  is viewer-only on this release.
- **One engagement router.** `engage_runtime`'s first step is "a live record?
  deliver over its socket and return. The common case"; the transcript lease
  arbitrates single-serving-runtime. Work arrives as typed errands:
  `PromptErrand`, `SteerErrand`, `PeerMessageErrand`, `WakeErrand`,
  `WarmErrand`, returning an `EngageOutcome`.
- **One event model.** `AgentEvent` (Pydantic, discriminated by `type`) with
  ~30 concrete events; `agent_start/end`, `provider_turn_start` ("THE
  acceptance boundary for an external supervisor"), tool/subagent/compaction/
  retry/wake events. Subscribers register via `session.subscribe(handler)`;
  `headless_print.printable_event` is the existing JSON projection
  (`lop exec --json`).
- **One approval seam.** `ApprovalGate` + `ask_approval`; the default headless
  gate is `_make_request_approval(yolo)` — non-TTY denies with a **typed**
  `ApprovalUnavailableError` rather than a lying "user denied"; full front ends
  replace the gate via `SessionProtocol.set_approval_handler`. "`exec --tools`
  stands as the APPROVAL for the names it lists where nobody can be asked."
- **The gap, precisely.** The benchmark apparatus drives episodes through its
  own provider client and its own decision loop (a reply-channel tool), never
  touching sessions, teams, subagents, compaction or the standard tool
  registry — while the repo's own benchmark already drives the real product
  (`lop exec --json`). One arm uses the product; the other re-implements a
  narrower one.

## 2. The seam — where the SDK sits

### 2.1 Files and symbols

| Piece | New/existing | What it is |
|---|---|---|
| `local_operator/session/spec.py` | **new** | `SessionSpec`, `SessionRoots`, `ApprovalPolicy` — plain frozen dataclasses; stdlib + `paths` only (import-cheap, guarded by a subprocess import test). |
| `local_operator/sdk.py` | **new** | Public facade: `open_session`, `spawn_session`, `deliver`, `events`, plus re-exports. All heavy imports (`session_factory`, `launch`, `serving`) are **function-local**; re-exports are served by PEP 562 `__getattr__` (same pattern as `local_operator/providers/__init__.py`). |
| `session_factory.create_session` | unchanged | The construction path `open_session` drives. |
| `session/runtime/launch.engage_runtime` | unchanged | The delivery router `spawn_session`/`deliver` drive. |
| `session/runtime/serving.spawn_owned_session` | unchanged | The runtime child's own composition path (phone parity); not called by the SDK directly. |
| `SessionProtocol` | unchanged | The handle type `open_session` returns. The SDK is a facade, **not a third implementation**. |
| `harness/types.AgentEvent` / `EventHandler` | unchanged | The event stream, re-exported. `sdk.events()` is an async-iterator adapter, nothing more. |

The test the design must pass — "a change to that path reaches all consumers":
**yes, by construction.** The rule for implementers: **any behavioural change
must be made in `Session`/`harness`/`session_factory`, never in `sdk.py`** —
`sdk.py` may adapt shape, never semantics.

### 2.2 Public shape (what a caller writes)

```python
from local_operator.sdk import (
    ApprovalPolicy, PromptErrand, SessionRoots, SessionSpec,
    events, open_session, spawn_session,
)

roots = SessionRoots(                      # REQUIRED. No ambient default.
    config_dir="/scratch/ep/home/.local-operator",
    agent_home="/scratch/ep/home",
    cwd="/scratch/ep/work",
)

spec = SessionSpec(
    hosting="openrouter", model="deepseek/deepseek-v4.1-flash",
    team="release",                        # or profile="reviewer"; shipped roles/teams
    tools=None,                            # None = full default surface; a list = exec's allow-list
    approvals=ApprovalPolicy.declared(["bash", "read", "write", "eval"]),
    name="episode task_001", goal=None,
)

# Owned, in-process (exec foreground parity):
async with open_session(spec, roots=roots) as session:        # -> SessionProtocol
    stream = events(session)                                   # async-iterator of AgentEvent
    await session.prompt(task_prompt)                          # awaited to the terminal outcome
    async for ev in stream:
        match ev.type: ...

# Runtime-hosted, survives the caller (phone/TUI spawn parity):
outcome = await spawn_session(spec, roots=roots, errand=PromptErrand(text=task_prompt))
# outcome.session_id / outcome.detail; attach later once a runtime is live:
async with open_session(spec.with_resume(outcome.session_id), roots=roots, mode="attach") as viewer: ...
```

Pinned semantics:

- **`open_session(spec, roots, mode="own"|"attach")`** — `own`: builds via
  `session_factory.create_session` (needs `has_ui=False`); `attach`: the viewer
  path (`create_session(has_ui=True)`'s `AttachedSession` when one is live; a
  cold id is refused with a remedy — auto-spawn deferred, see §5).
- **`send`/`prompt`**: the SDK branches on `session.outcome_is_synchronous` —
  owner `prompt()`; viewer `prompt_and_wait()`.
- **`events(session)`**: async iterator over `subscribe()`; synchronous
  handlers accepted; a handler failure must not kill the stream.
  `printable_event` remains the JSON mapping; no second serialization.
- **`ApprovalPolicy`**: `refuse()` (default; typed refusal), `auto()` (exec
  `--yolo`), `declared(tools)` (= a non-TTY `--tools` reach-bound whose
  members stand as approval), `callback(fn)` (installs via
  `set_approval_handler`, exactly what a full front end does). `ask` answers
  via `set_ask_handler` — not installed by any preset (same as exec).
- **Lifecycle**: `abort()` ends a turn; `dispose()` ends the owner; viewer
  `interrupt()` is turn-scoped. `request_stop` (ends the session + process)
  stays **not exposed** — only a person ends a session.
- **`resume`**: `SessionSpec.resume = "<session_id>" | "@latest"` flows into
  `args.resume` exactly as `exec --resume` does. (Implementation note: the
  design sketch spelled the fluent helper `spec.resume(id)`; the field keeps
  that name for exec parity, so the helper is `SessionSpec.with_resume(id)`.)
- **Team/profile attachment**: applied post-open through the same
  `Session` methods `exec_startup.apply_startup` calls, in the same order
  (team → profile → tool inventory → goal → name) — with ONE deliberate
  difference, recorded in §4.

**Spec → factory mapping (pinned to exec):** `hosting`, `model`,
`agent_name`, `agent_id`, `yolo`, `train`, `resume`, `workstream` are exactly
the eight fields exec passes; plus `birth_effort` (`spawn_owned_session`'s
documented extra), plus `cwd` (from `SessionRoots`). A **parity test** asserts
the SDK-built namespace equals the exec-built one field-for-field for the same
inputs (`tests/unit/session/test_spec.py`).

### 2.3 Deliberately NOT exposed

| Not exposed | Why |
|---|---|
| Tool schemas, `TOOL_BUILDERS`, dispatch | The session builds its tools via `create_tools(context, enabled)`. Consumers pass **names** only. |
| Loop internals (`LoopContext`, batches) | Inner engine; the events are the contract. |
| Provider clients / model layer | Requests already flow through `SessionStreamFn` with failover, caching and analytics attached. |
| Transcript/JSONL layout, checkpoint formats | Storage is the session's; resume goes through the factory. |
| Runtime control-plane crypto (operator caps, nonces, proofs) | Security boundary of the desktop console; adapters must not re-implement it. |
| Slash commands | exec's precedent: "deliberately a bounded startup interface". |
| TUI widgets/rendering | Consumer-side; unchanged. |

## 3. Capability inventory — the checklist to measure against

"Exposed" = reachable by a caller who writes only the SDK surface.

| # | Capability | SDK v1 exposure | Notes |
|---|---|---|---|
| 1 | Context compaction | **Exposed**: automatic every turn; `compact_now()`; events | Consumer does nothing; a compaction change reaches all surfaces alike. |
| 2 | Guides (`guide://`) | **Exposed**: automatic | Wired in `_prepare`. |
| 3 | Skills (`skill://`) | **Exposed**: automatic; roots under the session root | `LOCAL_OPERATOR_SKILL_EXTRA_ROOTS` remains the escape. |
| 4 | MCP servers | **Exposed via config — cwd-scoped, not root-scoped** | Discovery reads the session's **cwd** first (`<cwd>/.local-operator/mcp.json`, `<cwd>/.mcp.json`), then the scoped-HOME user config (`$LOCAL_OPERATOR_CONFIG_DIR/mcp.json`, `~/.claude.json`, `~/.cursor/`, `~/.codex/`), plus `<cwd>`-relative foreign imports (`.claude/`, `.vscode/`). The boundary is the session CWD: an episode held at the operator's home reads and dials their REAL `mcp.json` — root-scoping alone overstates the isolation (found in QA round 1). Sanctioned extension point (footprint ladder rung 4). **PR 2 checklist: the episode cwd stays inside the episode scratch** (see §5). |
| 5 | `task`/`team`/`hub` delegation | **Exposed, unchanged**: full subagent engine incl. roles/teams | The point of the exercise. Children are composed by the harness; the SDK must not touch that. |
| 6 | Jobs | **Exposed via tools + events** | No duplicate ledger API. |
| 7 | Todos | **Exposed via tool + events** | |
| 8 | Approvals / `ask` | **Exposed with presets** (§2.2); typed refusals surface as tool results | Footgun documented: auto-approve is session-wide incl. children. |
| 9 | Spend / usage accounting | **Exposed**: `session.spend` + per-turn usage events | Recording is off the critical path. |
| 10 | Model/provider layer | **Exposed**: hosting/model/effort in spec; `set_model()`; events | Failover stays internal. |
| 11 | Prompt caching | **Not exposed (deliberately)**: internal to the model layer | A consumer keying its own cache prefix would fork behaviour. |
| 12 | Wake / scheduling | **Exposed**: `wake` tool; `wake_delivered` event; `WakeErrand` | |
| 13 | Secrets | **Exposed as the tools are**: receipts, never values; store resolves under the session root | Never point at the live root (see §4). |
| + | Notifications | **Host-level, silenced by default for SPAWNED runtimes** | "A harness's child is a session nobody is watching"; opt in via `SessionSpec.notifications=True`. |
| + | Attachments/images | **Exposed** as parameters | |
| + | Projects / network / agent / team tools | **Exposed via tools** exactly as a user session has them | |
| + | Browser / console | **Unchanged gating** — absent where no host; SDK must not fake it | The benchmark's action surface enters as MCP (§6). |
| + | Config watching | Automatic | Live `config.yml` changes reach SDK sessions because they reach all sessions. |

## 4. Isolation & lifecycle — the safe thing is the default

### 4.1 Roots

- **`SessionRoots` is required.** `config_dir` + `agent_home` + `cwd`; no
  default, no ambient fallback unless the caller passes an explicit
  `allow_ambient=True` (a deliberate, greppable opt-out for single-machine
  scripts). `paths.py`'s contracts: `config_dir()` =
  `LOCAL_OPERATOR_CONFIG_DIR` else `~/.local-operator`; `agent_home_dir()` =
  `LOCAL_OPERATOR_HOME` else `~/local-operator-home` — *independent* overrides,
  which is why the SDK takes both.
- **Construct managers with the root, not the environment.**
  `ConfigManager(config_dir=...)` / `AgentRegistry(config_dir=...)` take an
  explicit path — the SDK uses **its** root. Where a component still resolves
  via `paths.*` at call time (skill roots, caches, team resolution), the SDK
  scopes `LOCAL_OPERATOR_CONFIG_DIR` / `LOCAL_OPERATOR_HOME` around
  construction (and the `async with` body in own mode), and **asserts** the
  resolved roots. Fail loudly; never "looks isolated".
- **The cache gets its own check** because it derives from `$HOME`
  independently of the two overrides: if it resolves outside the declared
  roots and `HOME` is the uid-default home, the run is refused with the remedy
  (redirect `HOME`).
- **The incident, and why this is a requirement not a nicety.** The benchmark
  already demonstrates the right practice (per-episode LaunchAgents carrying
  `HOME` + `LOCAL_OPERATOR_CONFIG_DIR` for a scratch home). The failure mode
  this prevents: a launch whose plist carried the **real** `HOME`, so episodes
  resolved the operator's real store — and the secret broker is machine-wide
  enough that it "broke `lop secret` machine-wide twice". The SDK's job is to
  make that misconfiguration *unrepresentable*: the root is an argument, not
  weather; child-env propagation is done by the SDK from `SessionRoots` (strip
  inherited `CMUX_*` and `LOP_*`), never remembered per rig.
- **Durability**: roots under macOS `/private/tmp` / `$TMPDIR` are refused by
  default (`allow_volatile=True` to opt out) — the same rule as the evaluation
  runner's `durable_root.py`, so scratch stores cannot be purged mid-run.
- **One root per process** (v1 implementation): a component's call-time
  resolution means two live roots in one process disagree about what they are
  reading. `open_session` refuses a second, different root while another is
  live (`allow_multi_root=True` to override); `spawn_session`/`deliver` check
  against live sessions too.
- **Danger flags / notifications**: spawned runtimes are silenced by default
  (`LOCAL_OPERATOR_NO_NOTIFICATIONS` in the child env) unless
  `notifications=True`; an agent-shell caller gets the same policy the CLI
  applies (`exec_session_refusal()` — allowed only with the delegation
  allowance or the QA escape), and any session it opens anyway is stamped by
  `_prepare` exactly as exec's is (the SDK cannot drift from the CLI here
  because the stamp site is shared).

### 4.2 Lifecycle

- **Two modes, honestly named**: `mode="own"` (caller's process runs the loop;
  if the caller dies, the turn dies — exec foreground parity) and
  `spawn_session`/`deliver` (a detached runtime
  `python -m local_operator.session.runtime.process` via `engage_runtime`;
  survives the caller, attachable later by any viewer).
- **Survival requires a supervised spawner**: long-lived programmatic work
  runs under launchd (the benchmark's tranche and the phone daemon both do);
  the SDK spawns via the sanctioned functions, never a raw `Popen`.
- **Stop semantics stay separate**: `abort()` / `interrupt()` / `dispose()`;
  `request_stop` **not exported** (user-only).

### 4.3 The one invariant to state

**Single root per process** — made explicit (see §4.1), refused rather than
pretended.

## 5. Consumers and migration (inert first)

**Order:**

1. **PR 1 — the SDK, inert.** `session/spec.py` + `sdk.py` + `docs/SDK.md` +
   `docs/design/sdk-engagement.md` + tests. Zero call-site changes. Guards:
   function-local imports (`tests/unit/test_sdk.py`,
   `tests/unit/session/test_spec.py` run subprocess import probes); parity
   tests: (a) spec→Namespace equals exec's literal namespace; (b) SDK-built
   session's tool names == `create_tools` output for the same allow-list;
   (c) approval-policy mapping table pinned; (d) roots assertion canary +
   uid-default/cache tripwires + agent-shell refusal/stamp; plus the
   additive-stable pinned-surface test (`__all__` + entry-point signatures).
2. **PR 2 — the benchmark pilot arm.** A new arm entrypoint runs episodes
   through `sdk.open_session` with the shipped roles/teams and the full tool
   surface; the existing arm stays untouched. First milestone: 10 tasks,
   compared against the current arm. **Environment checklist (QA round 1,
   non-negotiable): the episode's `cwd` stays inside the episode scratch.**
   MCP discovery is cwd-scoped (§3 row 4), so an episode held at the
   operator's home enumerates and dials their real servers; assert the MCP
   configuration sources (`load_all_mcp_configs`'s `sources`) resolve inside
   the scratch root.
3. **PR 3 — optional, after the pilot**: collapse the triplicated namespace
   literals into `SessionSpec`-based builders. Mechanical, parity-pinned.
   **The TUI never changes.**

**What must NOT move (load-bearing, named):** the TUI viewer mechanics and
approval wiring; the lease/claim boundary in `_prepare` (the single-writer
guarantee); the mobile control-socket vocabulary; `engage_runtime`'s
arbitration and `request_stop`'s user-only status; reply-channel semantics for
the existing arm; compaction/subagent/MCP behaviour inside children.

**Inertness/perf**: PR 1/2 add no imports to `cli.py`, `tui/`, `mobile/`,
`server/`; no env-var changes; no new dependencies.

## 6. What the benchmark looks like as a consumer (PR 2)

```python
# one episode, in the arm's existing launchd job (HOME + config dir already scratch)
roots = SessionRoots(config_dir=hcfg, agent_home=home, cwd=episode_ws)
spec  = SessionSpec(hosting="openrouter", model="deepseek/deepseek-v4.1-flash",
                    team="release")
async with open_session(spec, roots=roots) as session:
    session.set_approval_handler(ApprovalPolicy.declared(LISTED_TOOLS))
    stream = events(session)
    await session.prompt(TASK_PROMPT)
    async for ev in stream:
        record(ev)          # decisions in the record, not hand-built envelopes
```

- **The action surface** is exported by the episode adapter as an **MCP
  server** (the sanctioned extension point), so nothing OSWorld-specific ever
  appears in the SDK, the harness, or the session factory; the model gets
  lop's real tools *plus* the action server.
- **What it buys immediately**: shipped `task`/teams delegation, compaction,
  caching, retries/failover, spend accounting and the single event stream —
  any future harness improvement reaches the benchmark through the same
  object as the TUI, for free.

## 7. Risks / watch items

1. **Namespace coupling** — mitigated by the adapter + parity test; until
   PR 3, the adapter is the only sanctioned Namespace producer besides exec.
2. **Owner vs viewer semantics** — `outcome_is_synchronous` branch; a
   consumer misusing `prompt()` on an attached session reads as "turn finished
   early". Pinned by design; the attach path's cold-refusal is the v1 guard.
3. **Approvals footgun** — auto-approve is session-wide, inherited by
   children. Presets are explicit; default is `refuse()`.
4. **Multi-root** — refused rather than pretended (§4.3).
5. **Spawn parentage** — a `spawn_session` from an un-supervised process
   recreates the app-quit hazard; documented "for callers that are themselves
   supervised or accept foreground lifetime".
6. **Skew** — new arm vs old arm measured on identical tasks; the two must not
   share config roots.
7. **The parity check that matters for acceptance**: after PR 2, `exec --json`
   and an SDK session built from the same spec must produce the same tool
   surface, the same approval semantics, and event streams of the same shape
   (modulo transport). PR 1 satisfies the construction-time half; the
   end-to-end comparison is PR 2's acceptance test.

## 8. Decisions — resolved with the operator (2026-09-27)

- **"Stable/published" means**: in-repo module, documented as
  additive-stable, with the pinned surface test. No PyPI split, no semver
  promise — starting narrow keeps it reversible.
- **The arm runs in-process-owner per episode** (`open_session(mode="own")`),
  matching `exec` foreground parity; the existing launchd wrapper stays for
  durability. `spawn_session` still ships in v1 for callers that need
  runtime-hosted work; the OSWorld arm is simply not one of them, for now.

## 9. Implementation status (kept current with the tree)

**PR 1 as implemented (this tree):**

- `local_operator/session/spec.py` — `SessionRoots` (required roots, durable
  check, `to_env`, uid-default comparisons), `SessionSpec` (exec-parity
  `to_namespace`, wider `to_runner_args`, `with_resume`), `ApprovalPolicy`
  (four presets; the mapping table is in its docstring). Stdlib + `paths` only.
- `local_operator/sdk.py` — `open_session` (own/attach), `spawn_session`,
  `deliver`, `events` (async iterator over `subscribe()`), `SessionOpenRefused`,
  PEP 562 re-exports. Scopes the environment to the roots around construction
  and the own-mode body; asserts the resolved roots; tripwires the uid-default
  roots and an un-redirected cache; enforces the single-root invariant; applies
  the agent-shell policy and relies on `_prepare` for stamping; builds spawned
  children's environments from the roots.
- Refused loudly in v1 (with named remedies): attach auto-spawn; post-open
  state (`team`/`profile`/`tools`/`name`/`goal`/non-default `approvals`/
  `yolo`) on `spawn_session`; `birth_effort`-without-pair and half model pairs
  on `spawn_session`.
- One recorded divergence from calling `apply_startup` verbatim: the SDK calls
  the same `Session` methods it calls, but derives the declared inventory's
  stand-as-approval from the **approval policy** rather than
  `not control and not stdin.isatty()` — an SDK process has no terminal whose
  state may speak for its caller (`ApprovalPolicy.declared()` is the opt-in;
  the SDK default is strictly more conservative than a piped exec run).
- Tests: `tests/unit/session/test_spec.py` (parity, value objects, import
  guard), `tests/unit/test_sdk.py` (same-machinery surface, isolation
  tripwires, shell guard + stamp, spawn/deliver routing, single-root, attach,
  events, pinned surface).
- Corrections carried from authoring: the target ref is `origin/main @
  9f4e9d8b2` (the authoring note abbreviated it as `9f4e9d8b8`, which is not
  an object), and exec's namespace literal now sits in
  `exec_mode._make_default_session_factory` (~L672), not L123-137.

**PR 2 as implemented (this tree) — the benchmark pilot arm:**

- `local_operator/evaluation/session_arm.py` — `declare_action_server` (writes
  the per-session MCP declaration into the episode's scoped config dir, cwd
  asserted inside the scratch by [redacted] path), `ObservationRenderer`
  (frames published through the adapter's `verify_artifact`, returned as
  image content), `ActionBridge` (UNIX socket inside the scratch; one rendered
  observation per batch; `finish` terminal and non-executing; step budget;
  typed refusals; the `ask` route), `run_session_episode` (one
  `sdk.open_session(..., mode="own")` turn, `ApprovalPolicy.auto()`, bounded
  wall abort, score → `aggregate_cleanup` → close, PILOT record incl.
  `score.json`/`outcome.json`), and `_await_action_tool` — the arm holds the
  prompt until the action tool is in the session's **live** inventory and
  refuses when it never arrives.
- `local_operator/evaluation/action_server.py` — the MCP server
  (`python -m local_operator.evaluation.action_server`), stdio JSON-RPC, one
  `apply_actions` tool from the negotiated surface, forwarding to the bridge
  as a one-shot line protocol.
- `scripts/run_episode.py` — `--engagement {reply,session}` (default reply),
  `--session-route`, `--session-wall-s`; the session branch refuses an ambient
  home and requires the launchd scratch env.
- The reply-channel arm is UNCHANGED and remains the default (design §5's
  not-to-be-moved list); the session arm writes a PILOT record and is not
  comparable to reply-arm results.
- Measured behaviors worth keeping (from the PR's evidence): MCP discovery is
  asynchronous and the first request publishes its array once per turn — a
  prompt sent immediately after `open` gets `Tool not found` for the action
  tool until settle, which is why the arm gates on it; `EpisodeSession.tool_names`
  reads the live inventory, never a snapshot; the bridge endpoint is a short
  session-unique name because `sun_path` (~104 B on macOS) is close under
  `scratch-homes/<run-name>`.
- Tests: `tests/unit/evaluation/test_session_arm.py`,
  `tests/unit/evaluation/test_action_server.py`,
  `tests/unit/mcp/test_tool_bridge.py` (image forwarding, transport helpers).
- Parity (§7) as measured on this tree: an SDK session and
  `exec --json` on the same spec published IDENTICAL tool arrays (29 names,
  incl. the minted action tool) and identical 10-event sequences.

**Not yet implemented (planned):** attach auto-spawn; post-open state on
`spawn_session` via a resumed sidecar; the namespace-literal collapse (PR 3,
optional).
