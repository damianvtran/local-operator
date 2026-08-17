# MCP servers

local-operator speaks MCP over the official `mcp` Python SDK. Server configs
are discovered from project files (`<cwd>/.local-operator/mcp.json`,
`<cwd>/.mcp.json`), the user file (`~/.local-operator/mcp.json`), and
best-effort imports of other tools' configs (`~/.claude.json` — top-level
`mcpServers` plus project-scoped `projects.<cwd>.mcpServers`,
`<cwd>/.claude/.mcp.json`, `~/.cursor/mcp.json`, `<cwd>/.vscode/mcp.json`).
Project configs win over user configs, which win over imports;
`disabledServers` beats `enabledServers`, which beats a server's own
`enabled: false`.

## Trust model — read this before enabling project MCP configs

A committed `mcp.json` is **trusted input**: a `stdio` entry runs an arbitrary
command with arbitrary `env`, and an `http`/`sse` entry points the client at
an arbitrary remote server, all the moment a session connects to that
project. A malicious repository can therefore ship a config that executes
attacker-controlled code on the first connect, or a server whose tools
exfiltrate credentials the agent can reach.

Because of that surface, the manager logs a `WARNING` the first time each
project-sourced stdio server connects, naming both the contributing config
file and the exact command/args being spawned. **Review a repo's `.mcp.json` /
`.local-operator/mcp.json` (and any imported configs it pulls in) before
opening that repo under a credentialed profile.** Treat an unexpected MCP
server in a project config the same way you would an unexpected shell command
in a build script.

## Runtime behavior

- **Startup gate:** discovery races all connects against a 250 ms gate.
  Servers that finish contribute live tools; servers still pending with a
  tool-cache hit contribute *deferred* tools that await the connection inside
  `execute`; pending servers without a cache hit contribute nothing until a
  background continuation swaps them in.
- **Reconnect:** backoff ladder 0.5/1/2/4 s plus a sliding-window circuit
  breaker (>5 events in 30 s suspends auto-reconnect; a manual reconnect
  resets the breaker window and the ladder). A successful reconnect resets the
  backoff ladder only — the breaker window stays intact, so a flapping server
  still trips. An epoch counter prevents a late reconnect from resurrecting a
  connection after `disconnect_all`. When the breaker trips or auto-reconnect
  is abandoned, parked deferred executes fail promptly with
  `McpConnectionError` instead of hanging.
- **Tool names:** `mcp__<server>_<tool>`, both parts sanitized. Collisions
  between distinct origins (e.g. `my-server` + `a_b` vs `my` + `server_a_b`,
  which both mint `mcp__my_server_a_b`) are resolved deterministically by the
  stable origin key (server name + original tool name): the origin that sorts
  first keeps the base name, later colliding origins get `_2`, `_3`, … and a
  warning is logged. Reconnect ordering can never flip ownership.
- **Outbound args:** the harness-injected `i` (intent) field is dropped
  unless the server's schema declares it; other undeclared keys are dropped
  only when the schema sets `additionalProperties: false` explicitly (an
  absent or `{}` `additionalProperties` is open per JSON Schema and keeps all
  args); empty optional placeholders (`None`, `""`, `{}`) are dropped.
- **Tool cache:** `~/.local-operator/mcp_cache.db` — deliberately separate
  from `auth.db` because the cache is disposable while `auth.db` is
  credential-grade.

## OAuth

HTTP/SSE servers with `auth.type: oauth` use the SDK's
`OAuthClientProvider` (PKCE + RFC 7591 dynamic registration built in).
Tokens and client registrations persist in the shared `auth.db` under
provider `mcp-oauth`, one row per server URL. The headless flow prints the
authorization URL and then asks you to paste the **full redirect URL** (or a
`code state` pair); `state` (and `iss` when present) are parsed back out so
the SDK's state validation passes. Supplying a `client_id` in config pins the
client: the registration is pre-seeded and dynamic registration is skipped —
required for providers whose redirect URI was registered against a fixed
loopback port.

## Per-tool filters

A server can expose hundreds of tools while a session needs only a handful.
Filter server-owned tool names in that server's entry:

```json
{
  "mcpServers": {
    "crm": {
      "type": "http",
      "url": "https://example.test/mcp",
      "enabledTools": ["contacts_*", "get_company"],
      "disabledTools": ["contacts_delete"]
    }
  }
}
```

`enabledTools` is an allowlist when non-empty; `disabledTools` always wins.
Both accept exact names or glob patterns. Filtering happens before cached
(deferred) and live tool schemas are mounted, so a reconnect or
`tools/list_changed` cannot restore a denied tool. Run `/context` in the TUI
to see the current token cost of system blocks, wire tool schemas and
messages — this is the fastest way to spot an unexpectedly heavy MCP server.

A hosted server can challenge OAuth again during `tools/call`, after the
initial connection succeeded. HTTP 401 / `WWW-Authenticate` /
`mcp/www_authenticate` errors enter the same bounded policy as a stale
connection: one reconnect (which refreshes OAuth through the SDK provider),
one replay of the tool call, never a loop.
