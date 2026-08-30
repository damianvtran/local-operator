# Codex-imported MCP servers: authenticable + actionable failures

Evidence for the follow-up to issue #367 / PR #439 (Codex MCP import, shipped in
v0.43.14). The import worked; the imported remote servers could not be *used*,
and the failure text was a dead end.

All probes ran against a worktree off `origin/main`, with an assertion that the
imported modules resolve to that worktree rather than to another checkout.

## Root cause (confirmed live, not assumed)

Codex's remote entries carry only a `url` — that tool holds its OAuth grants
elsewhere, so its config has no auth block. Local-operator gated every
auth-capable path on an explicit `auth.type == "oauth"`, so those servers
connected with **no credentials**, the server 401'd, and the SDK reported it as
raw transport text.

`probe-live-401s.txt` — one real `initialize` POST per server:

| server | status | `WWW-Authenticate` |
|---|---|---|
| gitlab | 401 | `Bearer realm="GitLab", resource_metadata=...` |
| launchdarkly | 401 | `Bearer resource_metadata=...` |
| minerva-qa | 401 | `Bearer resource_metadata=..., scope="mcp:read"` |
| datadog | 401 | **none** |
| openaiDeveloperDocs | 200 | — (needs no auth) |

`probe-discovery.txt` — RFC 9728/8414 discovery, which decides the wording.
Note Datadog: **no challenge header, but discovery still succeeds**, so the
header alone is not a usable signal.

A second fact drove the design: the SDK destroys the status code. A 401 it
cannot resolve arrives from `session.initialize()` as
`MCPError(-32603, 'Server returned an error response')` with no status, no
headers. That is why detection is a transport-level response hook.

## Before / after

`before-connect.txt` (released behaviour) vs `after-messages.txt`:

```
BEFORE                                          AFTER
× MCP gitlab failed: Server returned an       → × MCP gitlab failed: run /mcp reauth gitlab
      error response                                — authorization expired
× MCP launchdarkly failed: unauthorized       → × MCP launchdarkly failed: run /mcp login
      access                                        launchdarkly to authorize
× MCP minerva-qa failed: Server returned an   → × MCP minerva-qa failed: run /mcp login
      error response                                minerva-qa to authorize
× MCP datadog failed: Server returned an      → × MCP datadog failed: run /mcp login
      error response                                datadog to authorize
```

`gitlab` says **reauth** and the others say **login** because gitlab is the one
holding a stored token payload that could not be refreshed. That distinction is
read from the credential store, not guessed.

## All four required classes

1. **OAuth-challenging 401** — gitlab / launchdarkly / minerva-qa, above.
2. **401 with no challenge header** — datadog, above. Discovery still finds an
   authorization server, so the login command is honest.
3. **401 with no discoverable OAuth at all** — no public server in the matrix
   has this shape, so `after-no-discovery.txt` records it against a local stub
   that 404s every `.well-known` probe. It must not promise a login:
   `apikey-only rejected our credentials (401) — set its API key or headers`.
4. **No auth needed** — `openaiDeveloperDocs` still connects with 5 tools, no
   OAuth provider attached and no discovery (`noauth-server-latency.txt`).
   Connect time is unchanged within network noise (2.79s after vs 2.37s before,
   and 1.87s vs 2.45s on an earlier run — the ordering flips between runs).

## Non-auth failures are NOT relabelled

`nonauth-parity.txt` compared this worktree against `origin/main`. Connection
refused, DNS NXDOMAIN, HTTP 500 and HTTP 404 produce **byte-identical** output
before and after. A network outage never becomes "run /mcp login".

## The `/mcp login` dead end

`after-login-gate.txt`. Before, `_resolve_mcp_server` refused every Codex import
with *"does not use OAuth login."* — the command that fixes the 401 told the
user it did not apply.

| config | before | after |
|---|---|---|
| gitlab (url only) | refused | accepted |
| launchdarkly (url only) | refused | accepted |
| openaiDeveloperDocs | refused | accepted (no-op: provider only acts on a real 401) |
| explicit `auth: oauth` | accepted | accepted |
| stdio | refused | refused |

## Rendered frames

`frames/before.svg` / `frames/after.svg`, captured from the real `OperatorApp`
(which loads `local_operator.tcss`) at 100x30 and viewed as PNG. The before
frame reproduces the user's screenshot. Geometry identical in both:
`virtual_size == size == (98, 28)`, no vertical scrollbar.

The first after-capture exposed a defect the numbers did not: including
`(401)` pushed two lines past the width and orphaned the bare `(401)` onto its
own centred row. The status code told the user nothing the verb did not, so the
OAuth wording now delegates to the existing `_auth_required_text` and every line
fits on one row.

## Not verified locally

No real OAuth grant was completed — that needs the operator's browser and
credentials. What is proven is that `/mcp login <name>` now reaches the grant
flow for these servers instead of being refused, and that a non-interactive
startup still raises rather than opening a browser.

## Reproducing

The probe scripts were scratch and are deliberately not committed — the outputs
above are the evidence. Each is reproducible from a worktree off `origin/main`:
drive `McpManager._connect_server` / `_connect_round` against the server URLs in
the table, and render the frames with `OperatorApp.save_screenshot` per
AGENTS.md "Visual validation" (the real app, so the stylesheet is applied).
