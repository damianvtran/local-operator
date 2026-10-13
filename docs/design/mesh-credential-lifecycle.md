# Mesh credential lifecycle: approved inheritance and hot repair

Status: **proposal; C5 authority/data-model approval hold before merge or policy
implementation/writes.** This document creates no grants or consent. Storage
migration has a separate explicit human-approval gate before writes.

Source baseline: `origin/main` at `60f951d073ed40e7ce8200d8300c5077264b57e3`
(2026-10-13); credential/lifecycle sources are unchanged from `bc3d68bf` (#2342).
File:line references below are against that baseline, except the explicitly
identified installed 0.70.12 proof.

## Authority, scope and decision

This extends [mesh-network.md](mesh-network.md) R13–R16,
[mesh-credentials.md](mesh-credentials.md) owner-only refresh and A5, and
[mesh-consent-provisioning.md](mesh-consent-provisioning.md) §§2, 5, 6, 8.
It specifies the missing continuous lifecycle, not a second broker, syncer or
login-status system. Existing selected static/secret copy classes remain valid;
rotating provider/MCP refresh secrets remain on their sole owner.

The operator's current authorization covers **current and future eligible
provider/MCP logins on their own approved mesh**, not foreign/no-consent networks,
local-only credentials, arbitrary secrets or a global default. A reviewed repair
path may persist that actual authorization for an existing approved recipient.
Future pairing approvals can disclose and persist an inheritance subset. Neither
path may invent approval from membership, role or `broker_credential`.

Recommend two separate source slices: approval-scoped lifecycle reconciliation,
then reader-first storage preparation with separately gated migration. Reject
native OAuth replication (second refresh owner), warning-only fixes (permission
is not effective auth), and manual share/pull/restart as normal post-approval UX.

## Current facts and remaining gaps

- `network/credentials/store.py:111–140` now constructs a dormant hot-capable
  facade on bridged devices; the no-peer path retains plain `AuthStore`.
- `credentials/client.py:506–552` re-resolves provider access on each request;
  only the GitHub installation-token helper retains its distinct TTL cache.
- `credentials/offers.py:393–439` includes encrypted provider-key descriptors
  through the trusted reserved namespace. Generic secret export stays closed.
- `credentials/effective.py:63` records keyless successful-resolve evidence,
  not permission-as-bearer; receipts are process-local and expire.
- `network/types.py:1509–1516` stores a finite, reduce-only `PairDecision.shares`.
  `relay._grant_pair_shares:12555–12578` intersects it with the actual offer;
  `MemberRecord:1360–1406` contains no future-login consent policy.
- `PlacementDocument.revoke`, `credentials/placement.py:500–530`, removes a
  holder but retains no explicit negative authorization. A naive reconciler
  would recreate a deliberately revoked grant.
- `McpManager._build_oauth_auth:3126` selects local/broker auth at client
  construction. `_grant_marker:4415` reads a local MCP grant marker;
  `revalidate_auth_blocked:4560` does not discover a new owner's login by itself.

Thus #2342 is necessary native hot-auth repair, **not completion of automatic
signin inheritance**. A holder row proves permission, not delivery, native
provider acceptance, model entitlement or a working MCP connection.

## Owner-local permission receipt and sticky denies

Proposed storage: `credentials/<network-id>/consent.json`, beside existing
placement/sync metadata, written only by the local owner under a bounded lock.
Do not embed authority in peer-merged placement or in `sync.json`: an older sync
writer must not erase future consent/denies. This is local metadata using the
existing credential layout, not another distributed authorization service.

Concrete schema (all references/IDs are synthetic; no credential material):

```json
{
  "schema": 1,
  "network_id": "n_<id>",
  "owner_device": "d_<owner>",
  "revision": 1,
  "recipients": {
    "d_<recipient>": {
      "public_key_fingerprint": "<identity fingerprint>",
      "enabled": true,
      "approval": {
        "kind": "operator-request",
        "ref": "<durable approval receipt/event>",
        "approved_at": "<timestamp>",
        "approved_by": "<principal reference>"
      },
      "inherit": {
        "providers": {"selector": "eligible-owner-native", "future_logins": true},
        "mcp": {"selector": "approved-owner-servers", "future_logins": true},
        "origin": "owner-native-login",
        "scope_ref": "<explicit approved principal/account/tenant scope>"
      },
      "denies": [
        {
          "selector": {"credential_key": "openai", "account_ref": "<account>"},
          "revision": 1,
          "reason": "withdrawn",
          "approval_ref": "<withdrawal receipt>"
        }
      ]
    }
  }
}
```

Selectors are typed: an approval may instead enumerate provider IDs and canonical
MCP URLs. `scope_ref` must resolve to recorded device/session routing and explicit
principal/account/tenant constraints, not a guessed identity. Provider-confirmed
not-applicable scope is different from unknown. Unknown required scope refuses
inheritance and reports why; no synthetic person/tenant verification.

Missing, corrupt or unsupported receipts authorize **no future keys**. Legacy
finite grants keep their existing semantics; they do not become blanket consent.
The current request can be represented only with its actual provenance and exact
approved owner/network/recipient subset. Re-pair/identity rotation requires the
existing authenticated continuity rules; a matching device name is insufficient.

Deny precedence is unconditional: recipient disabled/removed, local-only, explicit
pairing reduction, revoke, foreign/copied-origin material, then positive consent.
Imported/borrowed/copied credentials are not automatically re-exported as native
owner logins. This scope does not add unmarked/local-only generic secrets to the
copy set or override existing device-bound/host-local refusals.

**Crash/concurrency invariant:** persist denial before changing placement; the
owner broker checks current denial before serving, even if placement is stale.
A reconcile started under an older policy revision must recheck before granting.
Serialize policy mutations; preserve denial tombstones across logout, re-login,
restart and sync-state loss. Only an explicit authorized superseding action clears
a deny. Logout changes availability; it does not silently erase sticky withdrawal.
No cross-file atomicity is assumed: policy is authority, placement is its projection.

## Generations and metadata-only transport

Keep two independent monotonic revisions per owner/network/credential/account:

| Revision | Advances on | Effect |
|---|---|---|
| Material/state generation | rotation, expiry repair, logout, absence, withdrawal | Invalidate stale borrowed material/status evidence |
| Login-chain/authorization revision | new native login, explicit scope change, authorized re-enable | Evidence for one bounded MCP retry/reconnect |

Reuse existing `SyncState` **locking and bounds only**, not copy-material reset
semantics. Lifecycle material/state generations and login-chain/authorization
revisions both remain forward-only. In particular, do not reuse `fingerprint`
(`credentials/sync.py:240–254`), equal-material-digest comparison, downward adoption
(`sync.py:1637–1660`, `record_ack:718–741`), or counter clamping. Those belong to
existing copy classes and are incompatible with metadata-only lifecycle events.

Proposed authenticated recovery uses owner-authored persisted checkpoints and
recipient high-water observations bound to that owner/network/epoch/context. For
each revision independently, choose `max(proven_owner, proven_recipient) + 1`
only after authenticating the baseline and within the strict bounds below; never
adopt a member's bare claimed number as authority. Lower/reset or unprovable state
reports `needs_authoritative_baseline`, blocks positive lifecycle application and
grants for the affected context, and never upgrades consent or erases a deny.
A recovery jump above 2**20, overflow or exhausted headroom fails closed and
requires an authoritative checkpoint repair, not wraparound/downward adoption.
This is a **proposed new recovery validator**, not an existing implemented guarantee.

Negotiate proposed `credential-lifecycle-v1` on both ends of the existing peer
capability advertisement. Reuse `net_broker` and its membership/epoch/locality/
`broker_credential` authorization; add `lifecycle_announce`, `lifecycle_read` and
`lifecycle_ack` kinds at `MeshCredentialBroker.on_broker:329`. Each frame carries
`req`, network, authenticated owner/recipient context and bounded key batches.

The keyless payload allowlist is `{credential_key, account_scope_ref, state,
policy_revision, material_generation, login_chain_revision, expires_at,
owner_device, recipient_device}`. States distinguish available, absent, withdrawn
and unsupported/non-portable; they never claim successful native authentication.
Recipient applies only owner-authenticated metadata for its own identity/network;
acks prove metadata application, not possession of a token or a working turn.

No access/refresh tokens, client secrets, material-derived fingerprints, provider
responses or owner master keys in lifecycle events. Native material is obtained
only through existing grant/copy paths. **Never route rotating OAuth through
`copy`.** Existing `COPY_KINDS` in `credentials/sync.py:199` remains static keys
and selected encrypted secrets; class-2/class-4 source permissions remain intact.

Old peers receive no new kinds. Preserve their existing finite grants, but report
`unsupported` for automatic lifecycle behavior. Capability absence is not no-login
and must not cause repeated unknown-kind requests or wider legacy fallback grants.
New optional grant/status context must not reinterpret old expiry or scope fields.

### Proposed lifecycle validator and limits (not implemented)

Add `credentials.types.validate_lifecycle_frame` for the new kinds; the current
`on_broker` dispatch does not implement this validator. Validate the entire frame
and every entry **before any policy, placement, generation or acknowledgement
mutation**. The negotiated v1 schema has exact envelope/payload allowlists: reject
unknown keys (including token/material/hash fields), missing required keys, wrong
types, duplicate JSON/object/context keys, invalid IDs, nonfinite values and
oversized frames. Reject, never truncate, coerce or clamp. No material-derived hash
is needed to compare metadata or recover revisions. This needs a **proposed
lifecycle-aware strict-decode hook** at `network/wire.py:694–737` (`LinkCrypto.open`)
to retain authenticated plaintext size/depth/duplicate-key evidence before it is
lost by the current `json.loads`; typed validation then runs at `on_broker`.
A validator receiving only the existing decoded dict cannot prove those checks.

| Field/resource | Proposed bound/rejection contract |
|---|---|
| Whole decoded lifecycle JSON frame | At most 16,384 UTF-8 bytes, measured before parsing; maximum nesting depth 4 |
| Batch/cardinality | 1–8 unique credential/account-context entries per frame; maximum 8 displayed inventory rows per approval page; pagination preserves the explicitly approved subset, never silently approves omitted rows |
| Network/device identity | Exact `n_` + 24 hex / `d_` + 32 hex characters, also matched to the authenticated link and receipt |
| `credential_key` | 1–200 UTF-8 bytes, no NUL/control characters; must match the typed provider/MCP key parser, not an arbitrary URL/secret selector |
| `account_scope_ref`, consent `scope_ref`, approval/provenance opaque references | 1–256 ASCII bytes from `[A-Za-z0-9_.:/-]`; resolved locally to exact approved principal/account/tenant context; an unknown required reference refuses |
| State | Exact declared enum (`available`, `absent`, `withdrawn`, `unsupported`, `non_portable`); not free-form provider text |
| `req`, `policy_revision`, material generation, login-chain revision | Positive JSON integers, never bool/string/float; at most `2**53 - 2**20 - 1`, retaining headroom and rejecting overflow before mutation |
| `expires_at` | Null for no asserted expiry, otherwise integer epoch milliseconds in `[0, 2**53 - 1]`; never an effective-access claim by itself |
| Recovery advance | Positive forward difference at most 2**20 for each counter, with a verified authoritative baseline; no bare peer-number adoption |

`ANNOUNCE_CAP=8` and headroom concepts are grounded in `sync.py:116–138`; the
specific string/frame/depth/schema/recovery checks above are **new proposed
requirements**, not claims about current validation. Invalid context, oversized
input, unknown fields and excessive forward revisions must leave all authority,
placement, generation and ack state byte-identical. Logging names the refusal
code/context reference only, never the rejected raw frame or secret fields.

## Reconciliation and runtime ownership

- Approval: extend `_grant_pair_shares:12543` and onboarding provisioning
  (`network/onboard.py:2207`) to persist disclosed consent/reductions and grant
  the existing broker capability only when authorized, including first future
  login with no initial credential. Do not change old y/N approval semantics.
- Inventory: reuse `offers.enumerate_candidates:446`, including provider-native
  metadata and registered eligible MCP endpoints; filter by origin and receipt.
  Reconcile into `placement.grant_key_set:781`/holder mutations without replacing
  another owner's rows or widening account scope.
- Change detection: login/delete hooks in `AuthStore._after_credential_write:1516`,
  `delete_credential:1668`, `delete_credentials_for_provider:1748` schedule bounded
  work. `_ensure_oauth_fresh:2398` writes directly, so persisted tick reconciliation
  remains necessary; a hook alone misses rotation and crashes.
- Carrier: reuse `credentials_sync_step:1970`, `SyncEngine._pending_announces`,
  `_exchange_blocking:1024`, and `definitions.add_tick_step:2063`. Preserve the
  existing eight-item bound and 15-second tick/60-second reachable-member floor.
  Startup/reconnect schedules the same reconciliation; missed work is recomputed
  after offline/reboot, not an unbounded queue or second sync daemon.
- Provider runtime: retain per-request owner re-resolution and local-first
  native `OAuthAccess` mode/account/org metadata. Generation changes invalidate
  keyless effective receipts; successful resolution establishes effective state.
  Publish via existing serving/TUI hooks at safe boundaries, not during an
  in-flight request. Reconsider only missing-auth fallback pins, not deliberate
  model choices or unrelated quota policy.
- MCP runtime: bridge remote login-chain evidence into `_grant_marker` and
  `revalidate_auth_blocked`; rebuild/reconnect only the affected client's auth,
  non-interactively, once per genuine change. Preserve native `McpTokenStorage`
  chain semantics (`mcp/auth.py:1232–1418`). Rotation is not a new login: treating
  it as one recreates the refresh storm documented at `mcp/manager.py:4421–4455`.
- Owner: retain refresh leases/cooldowns and grant/report authorization in
  `credentials/owner.py`. Coalesce by authenticated recipient, network, key,
  account/scope/model context and force flag; current `_coalesced_resolve:524`
  keys only `(key, session_id, force)`. Effective receipts/status must not let
  one session/account's successful borrow establish another's readiness.

**Reverse NAT/VPN:** reuse `RelayServer._ensure_link_with_reason:9274–9276`,
which selects an established authenticated link before checking endpoints/dialing.
`client._ask_owner_directly:675` and sync `_exchange_blocking:1028` already use
that relay seam. Missing fresh dial reachability is not missing auth while a
valid reverse link exists. Do not add a dialer or trust an unauthenticated socket.

## Storage: reader-first preparation, separately approved migration

This lifecycle proposal does not claim universal encryption at rest today.
`AuthStore.upsert_credential:1389` seals Muse specially; ordinary payloads are
JSON-serialized at `auth_store.py:1463`. Static sync `_apply_value:1775–1805` writes
the ordinary **0600 SQLite** row; selected store-secret copies are re-sealed under
the recipient's own key. Permissions/WAL permissions are not encryption.

Storage source slice A is **reader-only**: versioned sealed-reference readers in
AuthStore/registry and MCP token storage, native refresh/usage/account/key readers,
and compatibility guards. It performs no migration or encrypted-only activation;
legacy plaintext remains explicitly disclosed. Do not globally hydrate secrets
into metadata/event objects. Borrowed OAuth remains memory-only.

Storage source slice B is **C5 human-approved writes/migration**, only after every
live reader/writer is proven compatible. Retain provider refresh leases and MCP
rotation CAS semantics; seal material, atomically publish metadata references,
and compensate/recover either-store failures. Static copies extract through the
trusted provider seam and re-seal locally; never copy opaque owner-local refs.
No silent dual plaintext `access`/`refresh` plus encrypted reference. No activation
while old busy processes can overwrite sealed rows or spend the refresh family.
Installing a new generation does not prove live sessions adopted it.

Approval must explicitly cover migration writes/plaintext-field removal and any
destructive cleanup. Test auth.db/WAL/logs with synthetic markers and report
historical backup/APFS-snapshot residuals separately; do not erase backups or
claim secure erasure implicitly. Rollback must use a sealed-reference-capable
reader, never quietly recreate plaintext. No migration/restart/update is authorized
by this document; operational install freezes remain binding.

## Derived evidence and acceptance matrix

Two bounded isolated spikes were executed during design, with no live secret,
relay, node restart or provider request:

1. Source-extracted current `_ensure_link_with_reason` with an existing link and
   zero fresh-dial budget returned that link with no endpoint probe. This is a
   method-level check, **not** a real NAT/VPN mesh E2E.
2. Actual installed 0.70.12 `AuthStore` stayed open across a second writer. Legacy
   synthetic OpenAI material resolved; replacing it with reference-only metadata
   made the old reader's next resolve raise **`KeyError('access')`**. Its previous
   in-memory access artifact remained present. Producer: installed
   `providers/registry.py:377–378`, `_oauth_api_key`, reads `creds['access']`.
   Successful bounded reproduction: 0.7 seconds. This is open-reader compatibility
   evidence, **not** a full busy-session migration proof.

Implementation review/independent QA must execute these fail-first cells:

| Cell | Required observable evidence |
|---|---|
| New provider/MCP login after session start | Same session ID/client heals without manual share/pull/remote OAuth/restart |
| Rotation on same login chain | Fresh native access, no refresh/reconnect storm; one refresh owner |
| Logout/re-login vs revoke/local-only | Availability resumes only with consent; sticky denies never auto-regrant |
| Crash between deny and placement | Owner refuses while stale holder still exists; retry converges |
| Both nodes reboot; missed offline updates | Consent/generations survive, fresh native resolve re-establishes effective state |
| Withdrawal while recipient is offline | Recipient misses revoke/local-only; on reconnect owner denies future borrows before stale placement converges. Replay a pre-withdrawal `available` announcement/grant and stale ack; reject stale authority, keep sticky deny, never auto-regrant even on a later login |
| Borrowed native OAuth versus API-key HTTP wire | Capture actual outbound endpoint and redacted header names/presence plus account/subscription identity. OpenAI OAuth uses the native Codex responses route and subscription scope headers; API-key mode uses its configured API route without OAuth-only headers. Back with actual response/side effect, not only `OAuthAccess` metadata |
| Wrong-mode/endpoint negative control | Force borrowed OAuth onto the API-key route/header shape and API key onto OAuth-only route/scope in an isolated controlled HTTP fixture; assertions must fail or wire reject. Restore native selection and prove the discriminating request passes; no credential/header values enter evidence |
| Lifecycle validator rejection and recovery | Unknown/material/hash keys, missing/invalid account/scope references, wrong ID/types, duplicate entries/keys, >8 entries, >16,384 bytes, depth >4, integer overflow and >2**20/unproven recovery jump all reject before mutation. Lower/reset revision needs authoritative baseline; no digest-based/downward recovery |
| Reverse-link-only owner reachability | Real authenticated owner-established link carries metadata and grants despite failed fresh dial |
| Old peer/missing capability | Existing finite grant works; automatic inheritance is explicitly unsupported |
| Wrong recipient/epoch, MITM/tamper/replay | No material or authority mutation; encrypted transport refuses |
| Same session ID across recipients/accounts | No coalescing/status leakage or scope expansion |
| Foreign/no-consent/local-only/unknown scope | No future grant; auditable keyless refusal |
| OpenAI/Muse/OpenRouter/Radient + MCP | Inventory source, native mode/account/model metadata and actual effective auth |
| OpenAI `gpt-6.1-sol` + RemoteUI | Real mesh turn, then rotation on same running session; selected/effective state parity and rendered artifacts |
| Old 0.70.12 busy reader/writer | Compatibility fence blocks migration; no silent plaintext resurrection |
| Approved storage migration fault matrix | Crash-safe references, encrypted ciphertext, no synthetic plaintext in active DB/WAL/logs, compatible rollback |

Native-wire acceptance is grounded in `providers/clients.py:2671–2708`
(`_request_base_url`, `_headers`) and `CODEX_RESPONSES_URL:2462`, not a new wire
mode. Capture header names/presence and assert synthetic account/subscription
binding; redact Authorization and other values. The isolated negative fixture
must discriminate mode/endpoint errors before later real-mesh acceptance. Offline
withdrawal tests assert denial of **future borrows**; they do not claim to retract
a leaked upstream bearer or make the historical grant cryptographically unusable.

No paid calls/install/restart are required to review this document. Real E2E and
rendered validation are later implementation gates when operational holds lift.
Independent reviewer/QA first; human C5 security approval afterwards, before
merge/new authority policy writes. No unsolicited human review requests.

## Residual limits and rollout watchpoints

A generic upstream bearer is not provider-enforced device/session-scoped merely
because its mesh grant has that label. Same-user device access is the existing
trust unit; session scope is not a new isolation claim. Preserve approval/person/
tenant constraints and native account identity, never invent provider narrowing.
Owner outage is a distinct unavailable state, not permission to refresh remotely.
Grant expiry is bounded by upstream expiry and configured TTL (`owner._mint_grant:812`).
Withdrawal stops future authorized borrows but cannot retract a leaked OAuth/static
bearer instantly; static-copy compromise still needs rotation at the source.

Watch grant resurrection, identity/scope confusion, MCP retry storms, generation
reset/replay, loss of sticky denies, mixed-version writers, false effective-state
publication and storage rollback. UI consumes the core effective-state contract;
it must not recreate a local-auth-only readiness predicate in another repository.
