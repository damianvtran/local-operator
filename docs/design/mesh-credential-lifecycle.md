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

Persist owner generations using existing `SyncState` locking/reset handling;
recipient observations carry owner identity, epoch and acknowledged generations.
Lost state cannot reset authority or make a lower generation win: use the existing
safe generation adoption protocol, and fail closed on unverifiable recovery.

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
| Reverse-link-only owner reachability | Real authenticated owner-established link carries metadata and grants despite failed fresh dial |
| Old peer/missing capability | Existing finite grant works; automatic inheritance is explicitly unsupported |
| Wrong recipient/epoch, MITM/tamper/replay | No material or authority mutation; encrypted transport refuses |
| Same session ID across recipients/accounts | No coalescing/status leakage or scope expansion |
| Foreign/no-consent/local-only/unknown scope | No future grant; auditable keyless refusal |
| OpenAI/Muse/OpenRouter/Radient + MCP | Inventory source, native mode/account/model metadata and actual effective auth |
| OpenAI `gpt-6.1-sol` + RemoteUI | Real mesh turn, then rotation on same running session; selected/effective state parity and rendered artifacts |
| Old 0.70.12 busy reader/writer | Compatibility fence blocks migration; no silent plaintext resurrection |
| Approved storage migration fault matrix | Crash-safe references, encrypted ciphertext, no synthetic plaintext in active DB/WAL/logs, compatible rollback |

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
