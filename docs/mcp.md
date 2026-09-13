# MCP servers

local-operator speaks MCP over the official `mcp` Python SDK. Server configs
are discovered from project files (`<cwd>/.local-operator/mcp.json`,
`<cwd>/.mcp.json`), the user file (`~/.local-operator/mcp.json`), and
best-effort imports of other tools' configs (`~/.claude.json` — top-level
`mcpServers` plus project-scoped `projects.<cwd>.mcpServers`,
`<cwd>/.claude/.mcp.json`, `~/.cursor/mcp.json`, `<cwd>/.vscode/mcp.json`,
`~/.codex/config.toml` — TOML, `[mcp_servers.<name>]` tables, user scope
only, meaning literally that path: Codex's `CODEX_HOME` override is not
consulted). Project configs win over user configs, which win over imports;
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

HTTP/SSE servers use the SDK's `OAuthClientProvider` (PKCE + RFC 7591 dynamic
registration built in) when any of three things is true: the config declares
`auth.type: oauth`, a grant for that URL is already stored, or the server has
been observed answering a request with a `401`/`403` whose metadata discovery
found an authorization server.

The last two matter for configs imported from another tool. A Codex remote
entry carries only a `url` — Codex holds its OAuth grants elsewhere, so its
format has no auth block to copy — and a purely static gate would connect such
a server unauthenticated forever. The rule is deliberately transport-level and
names no config source, so every format benefits. A server that needs no auth
never challenges, so it keeps connecting unauthenticated with no discovery and
no added startup latency.

Two shapes can never take a grant and are refused outright, without a network
call: a stdio server (no transport that can carry a bearer token) and one whose
config declares a non-OAuth `auth.type` such as `apikey` — that is the user
stating how the server authenticates, and its `401` is not an invitation to an
OAuth flow.

An explicit `/mcp login` on a server whose config says nothing either way — the
imported case — resolves the question with a single metadata discovery before
starting any grant. A server that advertises no authorization server is refused
with the same message a stdio server gets, rather than being sent into a flow
that cannot complete.
Tokens and client registrations persist in the shared `auth.db` under
provider `mcp-oauth`, one row per server URL. Supplying a `client_id` in
config pins the client: the registration is pre-seeded and dynamic client
registration is skipped — required for providers whose redirect URI was
registered against a fixed loopback port.

### Token refresh and the login popup

An expired access token is refreshed **proactively before connecting**, not
after a 401. The refresh targets the token endpoint resolved from the server's
OAuth metadata (protected-resource + authorization-server discovery), which
matters for providers whose token endpoint lives on a different host than the
MCP endpoint (e.g. Datadog). The refresh is serialized across processes with a
file lock and the stored token is re-read under it, so several sessions
starting at once cannot spend a rotating refresh token twice and invalidate
each other.

Ordinary startup and auto-reconnect are **non-interactive**: they never open a
browser. If the stored grant cannot be refreshed, the connect fails with an
actionable message instead of popping a login tab.

That message names the command that fixes it, and picks the verb from what is
actually stored: a server holding a token payload that could not be refreshed
is told to `reauth` (a plain `login` would leave the dead credential in place),
while one that has never been authorized is told to `login`. A server that
refuses us with a `401`/`403` but advertises **no** discoverable OAuth endpoint
is not promised a login it cannot complete — it is told to set its API key or
headers.

Failures that are not authorization failures (a down server, DNS, TLS) are
reported as **network failures** rather than as whatever the SDK happened to
say. The copy is `network: <what the exchange did>` (`cannot reach <host>`,
`cannot resolve <host>`, `TLS handshake with <host> failed`, `no response from
<host> (timed out)`, `the connection to <host> closed`), and when **every**
failed server in a round is one of these the startup toast says
`network: a, b, c` once instead of naming the servers as unrelated faults — the
same `network: ` spelling the durable notice and the `/mcp` row print, so one
marker reads the same on all three surfaces. At a card width where the names
cannot ride beside that marker the toast reports the count (`network: 3
servers`) rather than a truncated name: the marker is the signal, and the names
are carried whole by the notice and by `/mcp`. The single-failure card carries
the phrase alone (`network: cannot reach <host>`), because the phrase already
names the server by host and a `failed: <name> —` head would say it twice; a
**hostless** stdio failure keeps that head, since nothing else in its line
identifies the server. The durable notice's closing signpost is budgeted the
same way, in four rungs: the full `— /mcp for details`, then the short `— /mcp`
shed whole, then — for the transport family only — its own `(timed out)` detail
token dropped so the short form still fits, and finally **no signpost at all**
when even that cannot fit beside the sentence (around 70 columns and below for
a long failure). It is never split across a fold, and the sentence is never
edited to make room for it **except for its own detail token**: an application
error's own parenthetical, and the host in a transport phrase, both survive
whole, and the notice wraps instead. Why: the transport is the one layer a single local fault takes out for
**every remote server at once**, and the operator's own boot screen — 10 of 11
servers "failed" with `Request 'initialize' timed out` and nothing saying the
machine's connection was the problem — was chased through MCP config and OAuth
state before a phone hotspot found it. Naming the layer costs nothing and
points at the one thing the user can act on. The host comes from the server's
**config**, because the exceptions themselves (`httpx.ConnectError`,
`socket.gaierror`, `ssl.SSLError`, anyio's resource errors) carry no URL; a
stdio server has no host to name, so its transport failure is still reported
but is never counted as the user's connectivity being down. The startup round
also **settles** on such a failure: anyio delivers a transport death as a bare
`CancelledError`, which used to be read as a teardown and dropped, leaving
`startup_settling()` True for the life of the process so the one fault class
that takes out several servers reported nothing at all.

Re-authorize deliberately with:

```
/mcp login <name>          # inside the TUI
local-operator mcp login <name>   # from a shell
```

Both run the full interactive grant (browser + loopback redirect capture, with
a paste fallback when the browser cannot reach this machine). The headless
variant prints the authorization URL and accepts the **full redirect URL** (or
a `code state` pair) on stdin; `state` (and `iss` when present) are parsed
back out so the SDK's state validation passes.

Plain `login` reuses whatever the store still holds — a refreshable token or a
stored client registration short-circuits the grant. When that is wrong (an
account switch, a scope change, a consent screen that must come back up), use
the two companions:

```
/mcp logout <name>   # forget the stored credential and disconnect
/mcp reauth <name>   # forget, then run a fresh interactive grant
local-operator mcp logout <name>
local-operator mcp reauth <name>
```

`logout` removes the whole credential row (token *and* client registration),
so the next login re-registers via DCR or re-seeds a pinned `client_id` from
config. `reauth` refuses to start the browser flow if the old row could not
be removed, because a login on top of a surviving row would not be a re-auth.

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
