# Owner-fetched catalogue metadata (slice A)

## Boundary and rationale

A native catalogue read may use a login exhausted or blocked for inference. A
catalogue-purpose bearer would still authenticate inference at the provider and
would bypass those operational limits. This protocol therefore returns **model
metadata only**: owner authentication, account identity, refresh material and HTTP
headers never leave the owner. It neither changes sharing nor delegates refresh
authority. Ordinary inference grant selection/quotas remain independent.

This first slice supports native OpenAI **OAuth** catalogues only. It does not
substitute Radient billing, use a public API-key endpoint for a subscription, or
return an API-key login as if it were an OAuth account. Unsupported sources fail
closed. Controller/cache/picker consumption is deliberately slice B, not a stub
in this transport implementation.

## Existing transport, additive feature

- Local runtime -> own relay: `credential_catalogue`.
- Relay -> approved owner: `net_broker`, `kind=catalogue`.
- Bilaterally negotiated feature: `credential-catalogue-metadata-v1`.
- Request: registered provider/key `openai`, TTL integer 1..300 seconds, optional
  64-character lowercase hexadecimal metadata validator. The transport supplies
  requester/network identity; no caller account ID, endpoint URL, model,
  force-refresh, affinity or exclusions are accepted.
- An older peer receives no new request from the public client. It returns typed
  `catalogue_unsupported`; there is **no inference-grant fallback**.

The validator is an observation of the returned model metadata, not an authority
or instruction to skip refresh/HTTP. Slice A always asks the owner for fresh
metadata. Native provider parsing and its fixed Codex endpoint are reused;
redirects are not followed and upstream JSON is bounded at 1 MiB before decode.

## Public seam for slice B

`MeshCredentialClient.fetch_catalogue_metadata("openai", ttl=60, validator="")`
is a blocking worker-thread API. A runtime client reaches its own relay/control
socket; a relay client uses the established authenticated peer dialer. It returns
one strictly validated document or raises a typed unavailable/unsupported/refused
outcome (malformed metadata raises `ValueError`). It never returns a `Grant`,
`OAuthAccess`, token, account/email string or raw provider response.

Document fields:

- `kind=catalogue_metadata`, `network_id`, `owner_device`, `credential_key`,
  `provider`: the namespace asserted against the requesting client's current
  placement, not provider-supplied overrides.
- `models`: at most 256 allowlisted model dictionaries, each containing only
  `id`, `provider`, `context_window`, `default_context_window`,
  `max_context_window`, `max_tokens`, `supports_images`. Nullable native context
  bounds and vision claims remain unknown, not inferred false/default values.
  Model IDs are bounded identifiers, not free prose/URLs or secret-shaped keys.
- `account_scope`: opaque namespace hash over owner/network/native account scope;
  never a reversible email/account encoding or material-derived fingerprint.
- `incarnation`: opaque hash of the native login incarnation, not credential
  material. `material_generation` and `login_chain_revision` are the existing
  owner lifecycle counters; unavailable/withdrawn lifecycle state fails closed.
- `ttl` and `validator`: bounded lifetime and a metadata-only digest. Neither
  proves inference eligibility. The complete document is at most 12 KiB, leaving
  room for the broker/control envelope in legacy bounded readers. Overlarge
  catalogues fail as a whole; they are not truncated into a false complete list.

Slice B must keep cache identity `(network, owner, provider/key, account_scope,
incarnation, lifecycle revisions, purpose=catalogue)` and receipt time/TTL
intact. It must recheck current share/deny/owner/epoch and expiry before offering
cached models. Never write these rows to an unscoped API-key cache. A failed
metadata read is not permission to recover a bearer or import another device's
cache. Local authenticated discovery remains first, explicit phone provider
allowlists remain narrowed, and plain write/login stores are not globally
replaced with mesh-aware stores. No consumer/factory/cache changes ship in A.

## Authorization and ordering

1. Existing mesh authorizer checks authenticated membership, epoch and
   `broker_credential`; owner rechecks that current authority around async work.
2. Validate the closed request and positive feature negotiation.
3. Owner placement, holder, provider namespace, local-only mark, durable
   deny/journal fence and derived account scope must permit the read.
4. Read native OAuth metadata without decrypting tokens to build approved row
   IDs. Exclude denied, copied, unstamped, disabled or out-of-scope rows **before
   refresh**. Restricted native catalogue access disables ambient/unscoped
   credential fallback; ordinary local catalogue access is unchanged.
5. Resolve/refresh through the owner's existing native authority, then recheck
   permission and login identity. Fetch the owner's authenticated account-scoped
   Codex catalogue. No consent/auth lock is held across provider HTTP.
6. Recheck after fetch and serialize final consent/fence, membership/epoch,
   placement, derived identity and lifecycle observation at the existing serving
   guard. Do not recursively acquire pre-resolution admission under that guard;
   use the same bounded final-check pattern as ordinary grant minting. Lock order
   stays consent -> journal -> lifecycle; no HTTP is performed inside it.
7. Allowlist and bound the positive response before returning it; the public
   client validates again and rechecks its current placement.

A revoke during refresh or fetch produces no positive metadata. A catalogue can
list for an inference-blocked account while its ordinary inference grant stays
blocked. No usage, affinity, demotion or cooldown decision is changed by this
API; successful native token rotation and lifecycle observation still persist.

## Validation and limits

Disposable paired relays, rotating loopback IdP and fake account-authenticated
Codex endpoint exercise both RPC legs. External DNS/connect and gh discovery are
blocked before fixtures. Tests cover owner-only refresh/account headers with no
secret response, quota separation, account-denied prefiltering, sharing/derived
scope/local-only/member/epoch refusal, revoke during refresh/fetch, old-peer
unsupported, malicious request fields and malformed/oversized/foreign/secret-
shaped metadata. Actual remote-node deployment, live provider discovery, slice B,
credential-copy/refresh handoff and UI changes are outside this slice.

## Normal consumers (slice B)

`ProviderController.shared_catalogue` is a metadata-only alternative to
`listing_credential`, not a replacement credential store. Ordinary catalogue
reads try it only after their actual local side and explicit key overrides.
Mesh-aware virtual rows are not mistaken for local native logins. Explicit
provider collections (including the phone's persisted admission list) never
acquire this alternate source, even through discovery's lower cache reader.

Only an explicit device-scoped holder can populate a machine-wide catalogue;
session/derived shares require context this interface does not carry and are
reported unsupported. The owner remains the authority for approved accounts,
refresh and native Codex authentication. The borrower receives no new bearer,
account header, refresh token or inference entitlement.

Receipts live separately from native API-key listings under the profile's
`network/catalogues` directory. One current document per network/owner/provider/
purpose route prevents orphan or older-account documents from being unioned.
Its account/incarnation/revisions/validator also bind model-info memo keys.
Readers recheck current local membership, trust, placement, epoch, lifetime and
known lifecycle observations. A fresh owner refusal fences a formerly positive
receipt rather than letting its remaining TTL override that refusal.

Cold model mutation/birth paths prepare metadata off the owner event loop with
bounded waits; ordinary inference does not refresh this catalogue. The existing
broker's account selection, quota and grant checks remain authoritative and
unchanged. Native local credentials/overrides retain precedence; opaque receipt
scopes are never passed as OAuth account IDs or secret-selection inputs.

An unavailable owner is a model-listing failure, not an instruction to sign in
locally. Existing generic listing-failure copy is reused. These consumer states
still need their own independent real-path and rendered UX/design validation;
transport-only slice A evidence does not certify the picker or a live-node
rollout. No delegated refresh handoff is introduced here.
