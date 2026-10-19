# MCP credentials over the mesh: every shape, provisioned on approval

Status: **implementation.** Extends `mesh-credentials.md` §1 (classes 2 and 5) and
`mesh-consent-provisioning.md` §§1–5. Source baseline: `origin/main` @ `f2a215aaec`
(`v0.82.3`).

## The gap this closes

`mcpdefs` already mirrors MCP server *definitions* and carries env/header key names
as `ref:<NAME>` states; values never travel in that bundle. That is correct. What was
missing is the other half: the credential material those references name, plus a
status/repair surface that says which half is missing.

Measured on `cloud-node-1` @ 0.82.3 against a head node holding all nine servers:

| server | remote before | cause |
|---|---|---|
| datadog, hubspot, launchdarkly, linear, notion | `connected` (30–170 tools) | OAuth borrow worked |
| cloudflare, minerva-qa | `auth-required` | owner grant dead (`interactive_required`); correctly refused, but the remote said only `auth-required` |
| google-workspace | `disconnected` | env values are **literals** on the head; the mirror has `${GOOGLE_OAUTH_CLIENT_ID}` with no source |
| slack | `disconnected` | `Authorization` header is a **literal** on the head; the mirror has `${Authorization}` with no source |

Two structural facts fell out of that measurement:

1. **A literal in `mcp.json` is invisible to the needs-list.** `public_secret_refs`
   publishes references, not literals, so `needs_names()` never included
   `GOOGLE_OAUTH_CLIENT_ID` or `Authorization`; no placement grant, no copy.
2. **Placement convergence is paid only on an explicit act.** `pull_placement()` is
   called by `lop network credentials`; a session started after a share already sees
   it only because the relay boot pulls. There is no repair loop.

## Decisions

### 1. Per-shape inventory, resolved explicitly

| shape | source on the owner | delivery | never |
|---|---|---|---|
| OAuth (MCP) | `auth.db` `provider='mcp-oauth'` grant | **brokered access token**, owner refreshes; `mcp:<url>` placement holder | copy the grant/refresh token; a second refresh owner |
| env from the credential store | `${NAME}` in `env` | **class-2 copy** of `secret:<NAME>` into the member's encrypted store | put the value in the definitions bundle |
| header credential | `${NAME}` in `headers` | **class-2 copy**, same path | treat it as OAuth |
| literal env/header on the owner | the value in the user-scope `mcp.json` | **adopt** into the encrypted store, rewrite the config to a reference, then class-2 copy | send the literal over the definitions wire |
| static key/token | ordinary provider `api-key-static` row or a reference | existing class-4 copy / class-2 copy | — |

`mcp:<url>` keys stay broker-only. `secret:<NAME>` keys stay copy-only. The two
paths must not share a function that decides what to do with a value.

### 2. Adoption of literals (the new operation)

`lop network mcp credentials --adopt` (also run by `step_provision` and immediately
before a pairing offer is built) walks only the **user-scope**
`<config>/mcp.json` — never a project `.mcp.json`, never a foreign source — and for
each non-empty literal `env`/`headers` value:

1. chooses a secret name: the env key itself when free, else
   `MCP_<SERVER>_<KEY>` (uppercased, non-alphanumerics collapsed); headers always
   `MCP_<SERVER>_<HEADER>`;
2. reuses an existing store entry when its value is byte-identical; refuses a
   name collision with a different value (no silent overwrite);
3. writes the value to the encrypted store (`store.set`), then atomically rewrites
   that one entry's value to exactly `${NAME}` — the WHOLE value, including any
   scheme (`Authorization: Bearer …` becomes `${MCP_SLACK_AUTHORIZATION}` holding
   the complete `Bearer …` string). `mcpdefs.environment_state` carries only
   values that are exactly one reference; a partial `Bearer ${NAME}` would be
   withheld as a literal and would land on the mirror with the scheme missing.
   Every other field is preserved;
4. records an audit row `mcp.credential_adopted` naming server, key and secret name
   — never a value — and reports the same in the CLI.

Idempotent: a second run is a no-op. A write that fails leaves the config untouched
(the store write is the first half; the config rewrite is the second and only on
success). The operation never logs, prints or receipts a value.

### 3. Provision on approval

* **Onboarding (`step_provision`)**: adopt, then push definitions + MCP rows, then
  build the offer's default key set (which now includes `secret:<NAME>`), grant
  holders, set `broker_credential`, pull the document onto the node.
* **Pairing admission**: adopt immediately before the offer is built on the owner,
  so `needs_names()` sees the new references and the default set includes them.
* **Existing approved nodes**: `lop network mcp credentials --repair` on the owner
  adopts, grants the referenced `secret:<NAME>` keys to every active,
  credential-eligible member that already holds the server, and announces the
  copies on the existing sync tick.

### 4. Convergence, refresh and repair

* The **owner** keeps refreshing MCP access tokens (`owner._resolve_mcp`). A
  refused refresh is already typed (`interactive_required`, `refresh_failed`) and
  written to `credential.report`; the owner's doctor/`/network` repair row derives
  from it. No token material is copied.
* The **member** pulls placement on its own cadence (the definitions tick, bounded
  and in-flight-guarded), not only on a CLI invocation. A member that has lost a
  holder row drops every cached grant for that key and records `revoked`.
* `lop network mcp credentials --repair` on the member: pull placement, pull each
  missing `secret:<NAME>` copy directly from its owner, drop refusal caches, and
  borrow each OAuth key once to validate. Per-server outcome is the status row.
* A **removed member** is pruned from placement at removal (owner) and fails the
  handshake on its next contact (member); its cached grants are dropped when the
  next reconcile observes it is no longer a holder. Access ends at reconcile, not
  at token expiry.

### 5. Status: say which problem it is

One function, used by `/mcp`, `lop network mcp credentials`, the desktop catalog and
`lop network mcp state --json`:

| state | meaning | reason names |
|---|---|---|
| `synced` | local grant exists, or every reference resolves, or a holder row is live | the owner when borrowing |
| `expired` | a source exists but its token is past expiry and could not be refreshed | owner + `/mcp login` there |
| `needs_operator_login` | **never had it here or on the owner** | the owner + the command |
| `not_yet_synced` | the server came from a peer and the share/copy has not arrived | owner + `--repair` |
| `missing_source` | a reference names a key no store here holds, and no share is known | the key name |
| `revoked` | this device was a holder and no longer is | — |
| `unknown` | the store could not be read | — |

`/mcp` keeps its existing status column (`connected`, `auth-required`,
`disconnected`) and uses the **detail** column for the reason, so no client breaks
on a new status word. The catalog row gains `credential_state` and
`credential_reason` beside the existing `status`/`reason`.

### 6. Security constraints (unchanged where they already hold)

* Wire records are the existing AES-256-GCM/X25519 link codec; no plaintext path is
  added.
* Copies go only to current holders that are **active members**; the broker
  re-checks holder and membership per request; removal prunes placement.
* Values never enter logs, receipts, definitions bundles, transcript notices or
  `--json` output. The audit rows carry names and states only.
* `local-only` marks still win: a key marked local-only is never offered, never
  granted, and a copy already held is wiped on the next pass.
* No network record → the plain `AuthStore` path, byte-identical; adoption and
  repair are explicit owner operations.

## Test matrix (the deliverable)

Real paired node `cloud-node-1`, from this Mac:

1. **Five shapes reach `connected` with tool counts**: OAuth (`hubspot`/`notion`),
   env-from-store (`google-workspace`), header (`slack`), static key (`datadog`),
   static token (`launchdarkly`).
2. **Expiry**: expire an OAuth access token on the head (metadata only), force the
   member to borrow, show the owner refresh and the member connect with tools.
3. **Revoke**: remove a test-network member; its next reconcile fails closed and the
   status says `revoked`.
4. **Offline refresh**: take the member's relay down across a head-side refresh,
   bring it back, reconcile, connect.

Unit and two-relay link tests cover the same branches deterministically; the
real-node matrix is the independent QA evidence.
