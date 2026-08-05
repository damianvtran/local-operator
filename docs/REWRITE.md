# Local Operator harness rewrite — architecture contract

Branch: `feat/harness-rewrite`. This document is the binding contract between
parallel work streams. When this doc and memory disagree, this doc wins.

## Goals

Rewrite the agent experience of local-operator around a modern,
provider-agnostic Python architecture:

1. **Harness** — a provider-agnostic agent loop with native tool calling,
   streaming, steering/asides, scheduled wakes, background jobs. Replaces the
   classify→plan→act triple-LLM-round-trip loop in `operator.py`.
2. **Skills** — `<dir>/SKILL.md` skills with semantic vector search selecting
   only relevant descriptions per turn; `skill://<name>` progressive reads.
3. **Compaction** — append-only compaction entries, cache-aware pruning,
   auto/mid-turn/idle triggers.
4. **Providers** — OAuth login (OpenAI/ChatGPT, Anthropic, Kimi, xAI),
   API-key support, SQLite credential store, credential rotation + model
   fallback chains. Backward compatible with existing `credentials.env`,
   `config.yml`, and `--hosting` names.
5. **TUI** — full-screen Textual app themed with the local-operator brand kit
   (island-dark palette, one green accent, minimal one-line-per-action).
6. **MCP** — official Python SDK + manager semantics (fast-startup gate,
   deferred tools, reconnect breaker, multi-source config discovery).
7. **Headless** — `local-operator exec` runs a task and exits; `--background`
   detaches it with a log file.

## Non-goals (this pass)

- snapcompact (bitmap-image compaction), provider-native compaction endpoints.
- Rewriting the FastAPI endpoint surface (it stays byte-compatible; a facade
  keeps it running on the new engine).
- Subagent spawning / multi-agent registry (the harness is structured so a
  `task` tool can be added later; not built now).

## Module map and ownership

| Path | Stream | Role |
|---|---|---|
| `local_operator/harness/types.py` | foundation (already written) | Messages, events, tools, loop config — THE contract |
| `local_operator/harness/{loop,context,wake,jobs}.py` | A harness | Loop, append-only context, wakes, job manager |
| `local_operator/session/{session,transcript}.py` | A | Session facade, JSONL transcript entries |
| `local_operator/tools/{registry,builtin}.py` | A | Tool protocol impls: bash, read, write, edit, glob, grep, web-search stub |
| `local_operator/prompts_md/` | A | System prompt as `.md` templates (not Python strings) |
| `local_operator/providers/` | B providers | Registry, OAuth flows, auth store, failover, wire clients |
| `local_operator/model/configure.py` | B | Rewritten on top of providers (keeps `ModelConfiguration` name) |
| `local_operator/skills/` | C | Discovery, semantic index, `skill://` protocol |
| `local_operator/compaction/` | C | Compaction engine + pruning |
| `local_operator/tui/` | D | Textual app, brand theme, widgets |
| `local_operator/mcp/` | E | MCP manager, config discovery, tool bridge |
| `local_operator/exec_mode.py`, `local_operator/cli.py` | E | Headless exec, CLI surface (backward compatible) |
| `local_operator/server/utils/operator.py` | integration | Session facade: server API shape over `session.Session` (the legacy `local_operator/operator.py` is DELETED) |
| `local_operator/bootstrap.py` | integration | Composition root rewired to the new engine |

Streams MUST NOT edit files owned by another stream. Shared contract changes
go through `harness/types.py` only by flagging it in the task result.

## Core contracts (implemented in `harness/types.py`)

- `Message` / `CustomMessage` — the LLM-visible message. Roles `user`,
  `assistant`, `tool`, plus custom entries (compaction summary, skill prompt,
  wake delivery) rendered to LLM messages by the session's `convert_to_llm`.
- `ToolCall`, `ToolResult`, `AgentTool` — pydantic parameter models;
  `execute(tool_call_id, args, signal, on_update, context)`.
- `AgentEvent` — the ONLY boundary between engine and UI. TUI, print mode,
  and server all subscribe to events. Kinds: `agent_start`, `agent_end`,
  `turn_start`, `turn_end`, `message_start`, `message_update`, `message_end`,
  `tool_execution_start`, `tool_execution_update`, `tool_execution_end`,
  `notice`, `compaction_start`, `compaction_end`, `retry_start`, `retry_end`.
- `LoopConfig` — callback bundle injected into the loop. The loop never
  imports session/providers code.
- `ModelSpec` — provider/model descriptor consumed by wire clients.
- `stream_chat(request) -> AsyncIterator[StreamEvent]` — provider client
  protocol implemented by `providers/clients.py`.

## Stream specs

### A — harness core

Harness core contract:

- `loop.py`: two nested while loops (outer re-entry for asides/follow-ups,
  inner tool loop). Steering interrupts tool batches
  (`has_steering_messages` peek between calls); asides never interrupt.
  Synthetic tool results for dangling/aborted tool calls (pairing must stay
  legal). Guards: max paused continuations 8, tool-error retry is the model's
  job (return `is_error` results, don't throw).
- Prefix stability (the §A cache contract) is provided by STRUCTURE, not a
  manager class: `build_system_blocks` emits a FIXED-ARITY block list
  (instructions, tool inventory, skills-or-placeholder, env) so the wire
  clients' breakpoint derivation never shifts mid-conversation; the session
  appends to `context.messages` and never rewrites the head, and compaction
  replaces the head with a single marker whose rendered form is byte-stable.
  An earlier `AppendOnlyContextManager`/`StablePrefix` prototype was deleted
  (2026-08-05) because nothing constructed it — the doc must not claim a
  mechanism that is not in the request path.
- `wake.py`: scheduled wakes — `WakeSchedule`
  dataclass, `parse_wake_duration` (bare numbers REJECTED), `parse_wake_at`
  (+duration / HH:MM / ISO), `build_wake_schedule` returning errors as
  strings, `advance_wake_schedule` skipping missed occurrences,
  `WakeScheduler` with 60s max arm, load grace 2s, delivery that still
  advances on throw. Persistence: a `wake_schedules` custom entry in the
  transcript. asyncio timers replace `timer.unref()` — the scheduler MUST be
  disposed explicitly and never keep the loop alive (track handles, cancel on
  dispose).
- `jobs.py`: `AsyncJobManager` — register/running/completed/failed/cancelled,
  max 15 running, 5 min retention, owner-scoped delivery sink, cancel scoping.
- `session/session.py`: facade composing loop + tools + transcript + wake +
  jobs + skills + compaction hooks; `prompt(text)`, `steer()`,
  `subscribe(handler)`, `dispose()`. Session-scoped `asyncio.TaskGroup` for
  detached jobs.
- `session/transcript.py`: append-only JSONL entries
  (`message`, `compaction`, `custom`), `build_context(replay start at latest
  compaction)`.
- `tools/`: builtin tools — `bash` (asyncio subprocess with
  NON_INTERACTIVE_ENV table),
  `read`, `write`, `edit` (simple line-replace), `glob`, `grep`, `todo`,
  `wake`. Tool approval tiers read/write/exec; CLI auto-approves read,
  prompts for write/exec unless `--yolo`.
- Prompts: `prompts_md/system.md` rendered with a tiny `{{var}}`/`{{#if}}`
  renderer. System prompt blocks returned as a LIST (stable instruction block
  first, volatile env block last) so providers can set cache breakpoints.
  Date not timestamp in prompt (byte-stable per day).

### B — providers & auth

- `providers/registry.py`: `ProviderDefinition` dataclass — id, name,
  env_keys, login (async callable or None), refresh_token, get_api_key,
  store_credentials_as, callback_port, paste_code_flow. Field presence is the
  feature flag. Registry entries for: openai (+chatgpt oauth), anthropic,
  kimi, xai (+oauth), deepseek, google, mistral, ollama, openrouter,
  radient, alibaba — the existing 11 hosting names MUST resolve.
- OAuth flows (endpoints/client ids/logic):
  - `anthropic`: code+PKCE loopback, port 54545 `/callback`, paste fallback,
    refresh with `anthropic-beta: oauth-2025-04-20`, 30-day grant note,
    expiry skew 5 min, never rewrite org fields on refresh.
  - `openai` (ChatGPT): code+PKCE, pinned port 1455 `/auth/callback` (NO
    port fallback), device-code variant via OpenAI-private endpoints; JWT
    claims for identity (decode WITHOUT signature verification, no PyJWT).
  - `kimi`: RFC 8628 device code against `auth.kimi.com` with device
    fingerprint headers + persisted device-id file (0600).
  - `xai`: RFC 8628 via OIDC discovery (`auth.x.ai`, endpoint host-pinned
    validation).
  - Shared machinery: `OAuthCallbackFlow` base (server starts BEFORE auth URL
    generation; `/launch` 302 route; port-0 fallback only when allowed; 300s
    timeout), `poll_device_code_flow` (min 1s, slow_down +5s), PKCE
    (96-byte verifier, S256).
- `providers/auth_store.py`: SQLite (`~/.local-operator/auth.db`, 0600, WAL)
  — `auth_credentials` table (provider, credential_type, data JSON,
  disabled_cause, identity_key), blocks table, 7-step resolution cascade:
  runtime override > config override > OAuth credential > login-pasted key >
  env var > stored key > fallback. Round-robin + session stickiness.
  Backward compat: legacy `credentials.env` keys are read as the env-var tier.
- `providers/failover.py`: a/b/c credential rotation (initial / force-refresh
  same / rotate sibling; skip refresh on 403+usage-limit; attempted-keys set,
  64 cap), model fallback chains config (`retry.fallbackChains` in
  config.yml `values`, keys exact `provider/model` or wildcard `provider/*`,
  `default` chain), backoff `min(500*2^(n-1), 8000)` with 25% downward jitter.
- `providers/clients.py`: httpx wire clients — OpenAI-compatible
  `/chat/completions` (covers openai/openrouter/deepseek/kimi/alibaba/
  mistral/xai/ollama/radient via base_url), Anthropic `/v1/messages`, Google
  `generateContent`. Native tool calling; SSE streaming into
  `StreamEvent`s (text delta, tool-call deltas, usage, stop reason, cache
  stats). Anthropic cache-control breakpoints on system blocks.
- CLI: `local-operator login [provider]`, `logout`, `status` (new
  subcommands; additive, don't break existing).

### C — skills & compaction

Skills (deliberate design choices, user-requested):

- On-disk format identical to the ecosystem: `<skills-root>/<name>/SKILL.md`
  + YAML frontmatter (`name`, `description`, `enabled`, `hide` /
  `disable-model-invocation`). Roots: walk-up `<cwd>/.local-operator/skills`,
  `~/.local-operator/skills`. Non-recursive scan, deterministic sort
  (name.lower, name, path).
- **Semantic selection**: an `EmbeddingBackend` protocol with two impls —
  `ApiEmbedder` (OpenAI-compatible `/v1/embeddings` via configured provider)
  and `LocalEmbedder` (dependency-free hashed char-n-gram TF vectors, L2
  normalized — deterministic, offline, always available). Default: API when
  an embeddings-capable key is configured, else local.
- Index: faiss-cpu `IndexFlatIP` over description+name embeddings, cached at
  `~/.local-operator/cache/<identity>.skills.npz` + `<identity>.meta.json`
  (identity = digest of skill roots + backend), keyed by a content hash of
  (path, mtime, name, description) in discovery order; rebuilt lazily when
  stale. Skill count is small — rebuild is cheap.
- Per-turn: embed the last user message (+ latest compaction summary if
  any); select top-k (k=8) above cosine threshold 0.18 (local embedder needs
  a low bar; API backends can be stricter — make threshold configurable per
  backend). Inject ONLY the matched skills' `- name: description` lines into
  a VOLATILE system block appended AFTER the stable blocks (prompt-cache
  prefix stays intact). `hide` skills excluded from listing but readable.
- `skill://` protocol: `resolve_skill_url(url, skills)` — exact-name lookup;
  miss error lists ALL available names; no path → SKILL.md text; path →
  containment-validated join (reject absolute + `..`, re-check with
  `Path.is_relative_to` after resolve); directory → listing.
- The builtin `read` tool (stream A) accepts `skill://` URLs via a resolver
  hook injected by the session — C provides the resolver, A provides the
  hook. Contract: `SkillResolver = Callable[[str], str | None]` returning
  content or None if not a skill URL.
- Compaction (`compaction/`): strategies `context-full` and `snapcompact`,
  auto-selected: snapcompact when the active model supports image input,
  else context-full.
  - **Default threshold**: the lesser of 80% of the model context window and
    600,000 tokens (user-configurable via `values.compaction.threshold_tokens`
    / `threshold_percent`).
  - Settings in config.yml `values.compaction.*`: enabled (true),
    strategy (auto), reserve_tokens (16384), keep_recent_tokens (20000),
    threshold_percent (-1), threshold_tokens (-1), auto_continue (true),
    mid_turn_enabled (true).
  - `should_compact`/`resolve_threshold_tokens` math;
    `COMPACTION_RECOVERY_BAND = 0.8`.
  - `find_cut_point`: walk backwards accumulating estimated tokens, never cut
    at a tool result (assert), snap forward to valid cut roles.
  - Summary prompt with fixed structure (Goal / Constraints / Progress /
    Key Decisions / Next Steps / Critical Context), preserve unanswered
    questions verbatim, keep exact paths/names/errors; `<files>` appendix.
  - `CompactionEntry` appended to transcript (`summary`,
    `first_kept_entry_id`, `tokens_before`, `preserve_data`); replay =
    summary message + entries from first_kept onward.
  - **snapcompact** (vision-model archival):
    discarded history serialized, whitespace-collapsed, rendered onto PNG
    bitmap frames of pixel-font text (Pillow `ImageDraw.text`, monospace
    bitmap rendering) that vision models read back. `Archive { frames, text,
    text_head, text_tail }` stored in the CompactionEntry's `preserve_data`
    so later compactions re-render from text rather than carrying old PNGs.
    Tool results truncated to 2000 chars, useless pairs dropped, head/tail
    plain-text edges with the imaged middle (foveation: 3 HQ edge frames).
    Per-provider shapes (Anthropic ~1568-1932px wide bw, OpenAI 1568, Google
    flat-billed any size). Budgets: max 80 frames, ~5024 tokens/frame
    estimate. Requires image input support; falls back to context-full with
    a warning otherwise.
  - Pruning: superseded reads + useless-flag blanking in place (never
    delete), `MIN_PRUNE_TOKENS = 50`, warm-suffix guard
    `PRUNE_CACHE_WARM_SUFFIX_TOKENS = 8000` (precompute suffix token sums),
    idle flush `PRUNE_IDLE_FLUSH_MS = 90 min`, skill reads protected.
  - Token estimation: tiktoken cl100k_base, 1200/image, memoized per message
    id with explicit invalidation on mutation.

### D — TUI (Textual)

- Textual app, alt-screen, dark-first island theme from the brand kit:
  bg `#14110c`, fg `#e9e5db`, muted `#b5afa2`, dim `#837c6d`, edge
  `#3b3527`, accent `#38c96a`, string/success `#57c785`, amber
  `#e0b04b`, danger `#ef8078`. Light theme: paper `#f7f4ee`, ink `#211e18`,
  accent `#177b45`, hairline `#e5e0d5`. CSS variables generated from a JSON
  token dict (`tui/theme.py`) — one source of truth.

**Style direction (user mandate, supersedes earlier border plans):**
borderless chrome everywhere — NO bordered boxes. Structure comes from
symbols, text treatment, color, and spacing (status segments, tree glyphs,
icon-prefixed one-liners). One
space of padding at the screen edges (left/right/bottom) but NOT along the
top while scrolling — content meets the top edge. Sophisticated, slick,
minimal: prefer unicode symbols + dim/accent text over any rule lines;
lines only where truly necessary (e.g. the single input top border carrying
the status line). Add animated shimmer/strobe effects: animated spinner
frames on running indicators and a subtle shimmer repaint on the active
streaming block (`activity` spinner ~30fps, status spinner ~12.5fps).

**Character refinement (user feedback, supersedes "too simplistic"):**
borderless does NOT mean bare. Tool calls render as subtle BACKGROUND-FILLED cards
(one step brighter than the ground — the kit's `surface` #1e1a14; elevation
is a background step, never a border), with a per-tool icon, the tool name
in a tinted label color, the command/summary dim, and a right-aligned dim
`⟨expand⟩` hint (future expansion surface) plus duration. Structured
sections use tree glyphs (├─ └─) and tinted labels. The status line is a
full-width band on the kit's `sunken` ground with icon-led segments
(model · effort · cwd left; tokens/cost/jobs right) separated by `·`, not a
thin rule. The input sits on a `surface` panel with the `❯` chevron, no
border. Shimmer rides the working text (green crest over dim). Keep the
island palette — violet accents become our green/string tint; the STRUCTURE
is what we emphasize, not the hue.

- Layout (top→bottom): transcript scroll area; status line rendered as the
  TOP BORDER of the input box (zero extra rows); multiline input.
- **Minimalism rules**: one line per action. Tool executions render as a
  single row `icon name summary … status/duration` (collapsible expand later,
  default collapsed). Assistant markdown rendered with rich; streaming uses
  the frozen-prefix trick (re-render only the tail after the last settled
  block boundary; `markdown-it-py`), 30 Hz coalesced updates via timer,
  equality guard on identical text.
- Event-driven: `EventController` subscribes to `AgentEvent`s and mutates
  widgets; the agent never imports the TUI. Ordering hazards: superseded
  `agent_end` after next `agent_start`; orphaned tool ends buffered.
- Input: TextArea with history (up/down), slash commands `/help /model
  /login /logout /skills /mcp /compact /exit`, sync autocomplete for slash
  commands, keybindings: Enter submit, Shift+Enter newline, Ctrl+C interrupt
  turn (not exit), Ctrl+D/`/exit` quit, Ctrl+L clear.
- Interrupts: Ctrl+C during streaming sets the loop abort event → engine
  emits aborted `agent_end`.
- Headless-safe: TUI imports are lazy (cli.py imports `tui` only in
  interactive mode).

### E — MCP, exec mode, CLI

- MCP on the official `mcp` Python SDK (`ClientSession`, `stdio_client`,
  `streamablehttp_client`, `sse_client`; auth via `OAuthClientProvider` with
  a `TokenStorage` over stream B's auth store, credential ids
  `mcp_oauth:<server_url>`).
- `mcp/manager.py`: 250 ms startup gate (`asyncio.wait(timeout=0.25)`),
  deferred tools awaiting connection inside `execute`, SQLite tool cache
  (`mcp_cache.db`, separate from `auth.db` by design: the cache is disposable,
  `auth.db` is credential-grade), reconnect backoff 0.5/1/2/4s, circuit
  breaker 5 in 30s, epoch counter against post-disconnect resurrection.
- `mcp/config.py`: configs at `~/.local-operator/mcp.json` and
  `<cwd>/.local-operator/mcp.json`; also import `~/.claude.json` /
  `.claude/.mcp.json` / `~/.cursor/mcp.json` / `.vscode/mcp.json` (best
  effort). `mcpServers` shape; `disabledServers` wins over
  `enabledServers`.
- `mcp/tool_bridge.py`: `mcp__<server>_<tool>` naming (sanitize lowercase
  `[a-z_]`, strip redundant server prefix), outbound arg hygiene (drop

## Performance contract (user requirements, binding)

- **Start-of-session context budget**: a fresh conversation with the full
  installed skill set (the user's skills at `~/.omp/agent/skills` as the
  benchmark corpus) MUST start at ≤ 30,000 context tokens. Semantic skill
  selection is what makes this possible — only matched skill descriptions
  are injected. A benchmark script (`scripts/bench_context_budget.py`)
  measures it and fails CI if the budget is blown; optimize the system
  prompt if so.
- **Cache rate target**: ≥ 90% prompt-cache hit rate on multi-turn E2E
  tasks (targeting a ~95% hit rate). Requirements that serve this: byte-stable
  system blocks (date not timestamp; deterministic skill ordering; volatile
  content isolated in the last block), append-only history, pruning gated on
  the warm-suffix guard, one compaction boundary instead of drip edits.
  E2E runs record per-turn cache_read/cache_write tokens from provider
  usage and report the rate.
- **Speed**: tool batches with `concurrency="shared"` run via
  `asyncio.gather` (never serial); exclusive tools run alone. Provider
  streaming starts as early as possible (no pre-turn classification or
  planning round trips — the old triple-LLM-call per turn is gone).
- **Streaming/UI contract preserved**: the server websocket contract
  (`/v1/ws/messages/{message_id}` push shape, `CRUDResponse` envelope) is
  unchanged for the existing UI. The new engine pushes incremental
  `message_update` deltas rather than re-sending whole messages — same
  endpoint, lower latency. The AgentEvent stream is additive.
  harness-injected fields the server doesn't declare, drop empty optionals),
  retriable-error classification → one reconnect + one retry.
- stdio hardening: `start_new_session=True` on POSIX except macOS; Windows
  `.cmd`/`.bat` via `cmd.exe /d /e:ON /v:OFF /c` with percent/quote escaping.
- `exec_mode.py`: `run_exec(command, background, json_mode, agent)`.
  Foreground: run one prompt through a Session with a rich print renderer,
  exit 0 on success / non-zero on error (existing contract).
  `--background`: spawn `sys.executable -m local_operator.exec_worker`
  detached (`start_new_session=True`), log to
  `~/.local-operator/logs/exec-<ts>-<slug>.log`, register in JobManager,
  print the log path and job id, exit immediately.
- `cli.py`: keep EVERY existing flag/subcommand/dest/default byte-compatible
  (including `--agent-name` alias, default serve port 1111). Additive:
  `login`, `logout`, `login status`, `exec --background`, `exec --json`,
  `mcp list|add|remove`, root `--yolo`, `--tui/--no-tui` (default: TUI when
  stdout is a tty, plain REPL otherwise). Interactive mode routes to the TUI.

## Backward-compatibility contracts (hard)

- CLI: all existing subcommands/flags/dests/defaults/exit codes.
- `~/.local-operator/config.yml`: existing `values` keys keep meaning; new
  keys are additive (`compaction.*`, `retry.*`, `skills.*`, `tui.*`). Fix
  the known `conversation_length` vs `max_conversation_history` mismatch by
  honoring BOTH (read either, write canonical `conversation_length`).
- Deprecation: `conversation_length`, `detail_length`, and
  `max_learnings_history` are deprecated — the compaction engine supersedes
  them. Context retention is governed by `values.compaction.*`
  (`CompactionSettings`); the three legacy keys remain readable in
  `config.yml` and listed by `config list` (marked `[DEPRECATED]` there) so
  existing files and scripts keep working, but they are inert.

- `credentials.env`: still read (env tier of the credential cascade); still
  written by `credential update`. OAuth credentials live in auth.db, never in
  credentials.env.
- Agents on disk: `agent.yml` + `*.jsonl` layout readable. Drop `context.pkl`
  (dill) — agents that have one ignore it with a warning.
- Server: all 44 endpoints keep paths/verbs/`CRUDResponse` envelope. The
  chat routes run on the new engine through an `Operator` facade exposing
  `handle_user_input(user_input, user_message_id, attachments,
  additional_instructions)` with the same return shape.
- Scheduling: the legacy executor exposed agent tools (`schedule_task` /
  `stop_schedule`) that created durable schedules from inside a turn. The
  rewritten tool table has none; `/v1/schedules` (and the CLI) is the only
  creation path. This is a deliberate user-visible capability drop, not an
  oversight — autonomous schedule creation from model output is a larger
  trust surface than the frozen HTTP contract requires.

## Verification plan

- Unit: each stream ships pytest tests for its modules (no network;
  httpx.MockTransport / fake streams / tmp skills dirs).
- Live E2E: `local-operator exec` against OpenRouter with the configured key
  (file-creation task, web-question task, multi-turn agent reuse); OAuth
  flows exercised up to URL generation + callback-server round-trip with a
  local stub (no real logins in CI).
- TUI E2E: Textual pilot tests (headless app driving) + a real terminal
  session capture.
- Gates: `ruff check` + `ruff format` + `pyright` + `pytest`.
