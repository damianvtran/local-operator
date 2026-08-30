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

## Round 1 remediation (F1–F4)

Four majors from the agent review, each fixed at source with a regression test
verified to fail against the pre-fix code.

**F1 — a challenge after a redirect was missed.** The hook compared the
response's request URL to the configured string, so a server canonicalizing
`/mcp` → `307 /mcp/` and challenging on the redirect target did not match, and
the user kept getting the opaque error. Endpoint identity is now semantic
(scheme/host lowercased, query dropped, trailing slash stripped).

Reproduced end to end through the real SDK transport against a local server:

```
BEFORE  F1-redirect-401   MCPError               -> "Server returned an error response"
AFTER   F1-redirect-401   McpAuthChallengeError  -> names the fix
        hook saw 307 http://127.0.0.1:8902/redir
        hook saw 401 http://127.0.0.1:8902/redir/
```

**F2 — an earlier 401 could relabel a later non-auth failure.** `status_code`
latched on the first challenge and was never cleared, so a 401 followed by a
500 on the same endpoint still reported as authorization — the exact
misrouting this feature promises not to do. The **last** response on the
endpoint now wins; a 3xx hop is not a verdict and never clears a pending one.

```
after 401 -> status_code=401
after 500 -> status_code=None
classified as: None      (stays a network/server error)
```

**F3 — `/mcp login` was enabled for every remote config.** The `deliberate`
shortcut accepted API-key and known-public servers. It is removed. A stdio
server or an explicit non-OAuth `auth.type` is now a hard refusal that costs no
network call, and the remaining unknown case is settled by one discovery probe
(`probe_oauth_capability`) before any grant starts — so the first login on a
fresh import still works, without claiming every URL is authenticable.

**F4 — the classifying store's verdict was discarded.** `_challenge_error`
resolved `has_stored_grant` against the manager's own (possibly injected)
store, then `_auth_required_text` threw it away and re-read the default machine
store, rendering `login` for a server demonstrably holding a grant. The carried
fact is now used; only the legacy `McpAuthRequiredError`, which has no such
field, falls back to a lookup.

### Parity preserved

The live matrix is byte-identical to the pre-remediation run — same verbs, same
wording — so the design round's frames remain current:

```
✓ openaiDeveloperDocs: connected
× datadog:      run /mcp login datadog to authorize
× gitlab:       run /mcp reauth gitlab — authorization expired
× launchdarkly: run /mcp login launchdarkly to authorize
× minerva-qa:   run /mcp login minerva-qa to authorize
```

Direct 401, 403, public 200, and 404/500 non-auth parity all unchanged, and
startup still opens zero browsers.

## Round 2 remediation (F5–F6)

**F5 — a challenge could still relabel a later NETWORK failure.** Round 1
cleared the latch only when a later *response* arrived on the endpoint. A retry
that dies at DNS/connect/TLS/read time produces no response at all, so no hook
ran and the earlier 401 stayed latched — the terminal network failure was
reported as `/mcp login`.

The verdict is now bound to the **request**, not the last response: a
`request` event hook (`_AuthChallengeWatcher.begin`) clears the slot as each
same-endpoint request starts, and `observe` sets it only if that request comes
back 401/403. Whatever happens next — a response, or a socket error that never
produces one — no stale verdict survives. A request to any other URL (the
provider's discovery and token traffic) neither sets nor clears.

Proven end to end with a real `httpx` client and a real server:

```
after the 401           : watcher verdict = 401
retry failed with       : ReadTimeout (no response)
after the dead retry    : watcher verdict = None
classified as           : None      (NOT auth advice)
```

**F6 — the login gates dropped the manager's effective store.** Both
`_mcp_login_worker` and `_run_mcp_grant_on_owner` called
`probe_oauth_capability(cfg)` with no store, so a server whose grant exists
only in an injected store was refused as "does not use OAuth login" whenever
discovery was unavailable — even though the same manager had just classified
its failure as `reauth` from that store.

Eligibility is now a manager-owned operation,
`McpManager.server_supports_oauth_login`, which passes its own effective store
through. Both entry points call it via one TUI helper; a reduced follower
facade that does not implement it falls back to the store-less probe, which is
correct because such a facade owns no store to consult.

With discovery offline and a grant in a custom store:

```
probe_oauth_capability(cfg)                  -> False   (the old call)
manager.server_supports_oauth_login(cfg)     -> True    (consults its store)
```

### Parity preserved

The live matrix and the whole transport matrix are byte-identical to round 1 —
same verbs, same wording, so the terminal design round remains current:

```
✓ openaiDeveloperDocs: connected
× datadog:      run /mcp login datadog to authorize
× gitlab:       run /mcp reauth gitlab — authorization expired
× launchdarkly: run /mcp login launchdarkly to authorize
× minerva-qa:   run /mcp login minerva-qa to authorize

F1-redirect-401    McpAuthChallengeError    parity-500  MCPError (unchanged)
parity-direct-401  McpAuthChallengeError    parity-404  MCPError (unchanged)
parity-403         McpAuthChallengeError
```
