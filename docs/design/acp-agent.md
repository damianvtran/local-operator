# Design: `lop acp` — Local Operator as an ACP stdio agent

Status: proof of concept on `feat/acp-poc`. The first client is Qwen Audio
Agent's Gateway (`AGENT_PROTOCOL=acp`, `ACP_COMMAND=<lop>`, `ACP_ARGS=["acp"]`),
which delegates spoken tasks to an agent it spawns. Any ACP client that spawns
its agent over stdio (an editor, for instance) uses the same path.

## What it is

`lop acp` speaks the [Agent Client Protocol](https://agentclientprotocol.com)
— newline-delimited JSON-RPC — on its stdin and stdout. The client sends
`initialize`, `session/new`, `session/prompt` and `session/cancel`; the agent
streams `session/update` notifications and asks `session/request_permission`
before a write- or exec-tier tool runs.

Each ACP session is one in-process lop session opened through the embedding
SDK (`local_operator/sdk.py`, `open_session`). There is no second agent loop:
the turn, the tools, the transcript and the approval tiers are the ones every
other lop surface uses.

Code: `local_operator/acp_agent/` (`stdio.py`, `mapping.py`, `agent.py`,
`main.py`) and the `acp` branch at the top of `cli.main`. The protocol library
is the optional `acp` extra (`agent-client-protocol>=0.12,<0.13`).

## Decisions

- **D1 Route.** In-process: `lop acp` → `sdk.open_session(SessionSpec,
  roots=SessionRoots(cwd=<session/new cwd>))`. The SDK is already the embedding
  surface exec, the TUI and the desktop server share, and its rule is "may adapt
  shape, never semantics". Rejected: spawning `lop exec --json --control
  --supervisor-fd` per prompt (two protocol translations, a cold start and a
  `--resume` per prompt, and it re-opens the discovery surface D3 avoids), and
  driving the `lop serve` desktop API (every approve is gated on an OS presence
  prompt on the Mac, so voice approvals cannot work, and it needs a daemon the
  client does not own).

- **D2 Who approves.** The ACP client answers each gated tool call. The
  session's approval gate is `ApprovalPolicy.callback(fn)`; `fn` sends
  `session/request_permission` with two options, `allow_once` and
  `reject_once`, and returns the answer.
  - **Why the spawning client qualifies as the approver.**
    `docs/design/approval-authority.md` §0 states the invariant: *a constrained
    subject must not be able to mint the authority that removes its own approval
    requirement*. Its §3 table already admits two surfaces of exactly this
    shape: the *pane / CLI / desktop that SPAWNED the runtime* (authority: the
    spawn capability, no prompt) and the *`lop exec --control` supervisor* (a
    run-scoped credential written up an inherited descriptor). The ACP client is
    both at once: it spawned `lop acp` and holds the only copy of its protocol
    pipes. The answer arrives in-process on that private pipe, the way the
    owning pane's keystroke does — nothing crosses a socket another process can
    dial. A model's `bash` call cannot reach that pipe (D4), so it cannot answer
    its own request.
  - `allow_always` is never offered. It would flip the session to
    auto-approve, which is the authority-increasing transition §1 of that
    document classifies.
  - A failed or closed client connection raises `ApprovalUnavailableError`
    ("nobody could answer"), never a silent `False` that the transcript would
    render as "User denied". A request left unanswered for serving's
    `PENDING_REQUEST_TIMEOUT_S` (30 s) is denied.
  - The callback replaces the headless gate, so a saved
    `tool_approval_mode: auto` is not honoured here. That is stricter than
    exec, not weaker.
  - **Caveat — the Gateway becomes part of the trust boundary.** lop now trusts
    the client exactly as it trusts an `exec --control` supervisor: whatever the
    client answers is the operator's answer. If the Gateway exposes a local API
    that can approve a pending request **without authentication**, a lop `bash`
    call could dial that API and approve its own permission request through the
    Gateway — the §0 attack, one hop over. **Check how the Gateway authenticates
    its own approval path before trusting voice approvals.** If it does not
    authenticate, D2 still stands for lop's side, but the deployment needs a
    warning (or the Gateway's API needs auth). This was not verified in the POC.

- **D3 Why D2 does not weaken the model.** Own-mode SDK sessions publish no
  discovery record and no control socket (`sdk.py` builds no `exec_control` or
  `RuntimeServer`). The attack in approval-authority.md §0 — a model's tool call
  reads the record, dials loopback and approves its own card — therefore has
  nothing to dial. The only answer path is this process's stdin, and D4 makes
  that private.

- **D4 Protect stdin and stdout.** First thing after the argument parse,
  `claim_stdio()` duplicates fd 0 and fd 1 into private descriptors (`os.dup`
  returns non-inheritable descriptors), then points fd 0 at `/dev/null` and
  fd 1 at stderr. Without it, `bash` tool children inherit stdin (the bash tool
  sets only stdout/stderr on its child), so a `cat` would swallow protocol
  frames; and the library's stray `print()`s would corrupt the stream the client
  reads. The CLI dispatches `acp` before its login-shell PATH probe and its
  startup migrations, both of which can print.

- **D5 Transport.** `agent-client-protocol>=0.12,<0.13` as the optional extra
  `acp` (also pulled in by `dev`), imported lazily; a missing extra prints the
  standard `pip install "local-operator[acp]"` line and exits 1. Apache-2.0; its
  only dependency is `pydantic`, which lop already pins. Its connection runs
  each incoming request as its own task, so `session/cancel` and the outgoing
  `request_permission` work while a prompt is in flight. The pin is narrow
  because 1.0 release candidates are published and the schema is still moving.
  It is kept out of `all` for the same reason.

- **D6 Session ids.** The ACP `sessionId` is the lop `session_id`, so a later
  `session/load` can become `SessionSpec(resume=id)`. The agent keeps one entry
  per id (the session, its event mapper, one-prompt-at-a-time lock, pending
  permission requests, a cancel flag). A second concurrent `session/prompt` on
  the same session is refused with JSON-RPC -32600; an unknown id with -32602.
  Each session lives in its own host task, because the MCP transports it builds
  hold anyio cancel scopes that must be exited in the task that entered them.

- **D7 Cancel.** `session/cancel` calls `session.abort(...)` **and**
  `cancel_subagents(...)` — ACP asks to stop all work, and `abort` alone leaves
  children running. It also settles any pending permission request as denied.
  The prompt then answers `stopReason: "cancelled"`.

- **D8 (revised): superseded the original accept-and-ignore after the live
  Gateway refused the backend.** The original D8 accepted `mcpServers` and
  ignored them. Against the real Gateway (qwen-audio-agent 2.0.1) that fails:
  in `session/new` the Gateway passes an HTTP MCP server carrying its
  coordination tools, and it refuses a backend whose `initialize` does not
  return `agentCapabilities.mcpCapabilities.http = true` (health shows
  `START_FAILED`, "ACP does not declare HTTP MCP support, so it cannot provide
  coordination tools"). So:
  - `initialize` advertises `mcpCapabilities.http = true`; `sse` stays false.
  - HTTP `mcpServers` from `session/new` are wired into **that session only**,
    through a new optional `SessionSpec.mcp_servers` field (raw
    `{name: {type, url, headers}}` entries). The SDK forwards it to
    `create_session`, which hands it to the session's own `McpManager`
    (`extra_configs=`). The manager merges them **after** the file-loaded
    configs on every load (startup, connect-by-name, reload). lop's MCP client
    connects them with its existing streamable-HTTP transport.
  - **A file-configured server wins a name collision.** A session-only server
    can add a name, never replace one the operator configured; the shadowed
    entry is logged.
  - The client's headers are passed through, with one change: `${` is escaped
    to `$${`. lop resolves `${NAME}` in MCP headers against the operator's
    secret store, and a header the client supplied must not be able to name the
    operator's secrets and send them to the client's URL.
  - Nothing is written to disk: no `.mcp.json`, no `mcp.json`. The servers
    live only in that session's manager.
  - stdio (and SSE / ACP-transport) servers are still ignored in the POC, and
    their names are logged to stderr.
  - `SessionSpec.mcp_servers` is refused for a spawned SDK session
    (`spawn_session`) and for attach, because neither has a channel to carry it.

- **D9 Stop reasons.** From the turn's `agent_end`: `aborted` → `cancelled`;
  an `error` whose final assistant message the loop marked `stop_reason ==
  "refusal"` → `refusal`; any other `error` → a JSON-RPC -32603 error carrying
  the text; otherwise `end_turn`.

- **D10 Which config.** The session roots are `paths.config_dir()` and
  `paths.agent_home_dir()`, with `allow_ambient=True` only when those resolve to
  the uid defaults (a client launching `lop acp` means "my Local Operator"). So
  `LOCAL_OPERATOR_CONFIG_DIR` / `LOCAL_OPERATOR_HOME` and a redirected `HOME`
  still isolate a test. Notifications are suppressed, because the client is the
  user interface. `--hosting` / `--model` after `acp` select the model, as for
  every subcommand.

## Event mapping (lop event → ACP `session/update`)

| lop event (`harness/types.py`) | ACP update |
|---|---|
| `message_update.delta` | `agent_message_chunk`, text |
| `message_end` | the full text only if no deltas streamed for that `message_id`; otherwise nothing |
| `reasoning_delta` | `agent_thought_chunk` |
| `tool_execution_start` | `tool_call` {id = `tool_call_id`, title = intent or tool name, kind (read → read; grep/glob → search; edit/write → edit; bash/eval → execute; web_fetch → fetch; else other), status `in_progress`, `rawInput` = args} |
| approval gate | `session/request_permission` with toolCall {same id, status `pending`} |
| `tool_execution_update` | `tool_call_update` with the partial-result text |
| `tool_execution_end` | `tool_call_update`, status `completed` / `failed` (from `is_error`), result text limited to 8,000 characters |
| `subagent_start` / `subagent_end` | `tool_call` kind `other` (id `subagent-<job>`) / `tool_call_update` `completed` or `failed` |
| `agent_end` | not an update; it settles the prompt's `stopReason` (D9) |
| notice, compaction, retry, model change, turn/provider frames | not sent to the client |

The approval gate is not given the tool-call id (its arguments are tool name,
summary and job id), and it runs after `tool_execution_start`. So a permission
request is matched to the oldest started, unfinished, not-yet-claimed call with
the same tool name. With no match — a subagent's call (job id set) or an
unexpected name — it uses a synthetic `perm-<hex>` id and announces that call
with a `tool_call` first, so the client has something to render.

## Out of scope for the POC

- `session/load` and history replay.
- `configOptions` and model switching from the client.
- stdio, SSE and ACP-transport MCP servers from `session/new` (logged and
  ignored).
- **Subagents do not inherit session-only MCP servers.** The servers live in
  the parent session's manager; a delegated child builds its own and does not
  see them.
- `allow_always`.
- The `ask` tool (SDK presets do not install it).
- Image, audio and embedded-resource prompts (`resource_link` is passed as its
  URI).
- Client file-system and terminal capabilities.
- Showing ACP sessions in the TUI or attaching to them.
- A PR, a release, and Windows.

## What this commits us to

- A new package `local_operator/acp_agent/` and a dependency behind the `acp`
  extra.
- A new public `SessionSpec` field, `mcp_servers`, and the matching
  `McpManager(extra_configs=)` seam.
- The ACP session id is the lop session id. Changing that later breaks
  clients' saved ids.
- ACP sessions write their history to the operator's normal store, so they
  appear in the operator's session list.

## Risks and what would change the answer

- **Gateway authentication** — see the D2 caveat.
- **Several sessions in one process:** if sessions with different working
  directories interfere, move to one `lop acp` process per session. That keeps
  D1.
- **Session build cost:** paid once per `session/new`. A client that opens one
  session per utterance would change this.
- **SDK churn:** if the ACP SDK's 1.0 schema breaks the pin, replace it with
  JSON-RPC written by hand (D5).
