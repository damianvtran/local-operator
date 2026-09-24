# Design: mesh credentials — ownership, brokering, and non-regression

Status: **proposal for implementation.** Author: architect.
Base: `origin/main` @ `a7e6b9bd`, on branch `feat/mesh-network`.
Parent: `docs/design/mesh-network.md` (all `R<n>` / `A<n>` references are to it).

**Closes R13, R14, R15, R16 and decision A5.** Depends on
`mesh-transport-identity.md` for device ids, the network secret, the epoch and
the peer link, and on `mesh-session-mobility.md` for `locality`, placement and
the move/fork path. All file:line references are against the base tree above.

**Interface alignment, checked against the peer documents as they stand.**
`mesh-transport-identity.md` already reserves the peer op **`net_broker`** with
capability **`broker_credential`** (§6.4, §7.1), the capability string
**`credential-broker-v1`** (§6.4), and the link envelope this document's frames
ride in (§6.4: `{"op", "req", "locality": "remote"}` request /
`{"op":"ack"|"error","req","detail"|"message"}` reply). This document adopts all
four verbatim rather than inventing a parallel surface, and its local relay-side
op follows the transport's local vocabulary naming (`net_*`, §2.5). Device and
network ids are the transport's (`d_<hex32>`, `n_<hex24>`, §3.1, §4.2), and
whole-file documents use an integer `schema` (`mesh-transport-identity.md` §4.2,
§4.3), while standalone log records carry a namespaced string `schema`.

| Requirement | Where this document closes it |
|---|---|
| R13 — credentials follow the session, not the device | §2.4, §3 |
| R14 — an owner device; never spent off it | §2, §3.3, §4 |
| R15 — refresh on behalf of a peer | §3.4 |
| R16 — no regression on provider or MCP logins | §5 |
| A5 — brokered, never mirrored | §1.3, §1.4, §2.5, §7 |
| A seam for R20/R21 (owned by `mesh-compute-pool.md`), not a claim on them | §6 |

---

## 0. The answer in one page

**Token material never crosses a device boundary except as a short-lived,
scope-limited bearer handed back over an already-authenticated link, and never
touches the receiving device's disk.** A peer that needs a credential for a
provider it has no local login for asks the owning device's relay; the owning
device performs the refresh **through the local code path that already exists**,
under the local lease that already exists, and returns only the usable bearer.

Three properties follow, and they are the whole design:

1. **The refresh never moves.** `_ensure_oauth_fresh`
   (`local_operator/providers/auth_store.py:903`) and its cross-process lease
   (`AUTH_REFRESH_LEASE_MS`, `auth_store.py:67`; `_try_refresh_lease`
   `auth_store.py:847`) run on the owner. This is why the PR-24 class of bug —
   two processes POSTing one rotating refresh token and invalidating each
   other's new token — *cannot* occur across the mesh: there is exactly one
   process on exactly one host that ever POSTs, and it is the process that
   already coordinates with its host's siblings.
2. **The request side is a decorator, not a second cascade.** A new
   `MeshAwareAuthStore` wraps the existing `AuthStore` and implements the
   structural protocols the failover driver already consumes
   (`FailoverAuthStore`, `local_operator/providers/failover.py:2719`;
   `OAuthAccessSource`, `failover.py:2751`; `CredentialLister`,
   `failover.py:2733`). It consults the local store **first** and only then the
   mesh. With no network configured the wrapper is never constructed at all
   (§5.1), so the 0-peer topology runs byte-identical code.
3. **Ownership is explicit metadata, authored only by the owner.** A synced
   `placement.json` names, per credential key, which device owns it and which
   devices may use it. It contains no credential material and no volatile
   observations (§2.1). A device-local `placement.state.json` holds the
   *observations* (owner offline, grant invalid) and is never synced — that
   split is what stops last-writer-wins from clobbering a fact with a rumour.

Two things this design deliberately does **not** do, so nobody assumes them:

- **The `lop secret` value store is not brokered** (§4.8). A remote session
  that needs a secret stores it where the session runs, via the existing
  session-scoped `/credential` verb
  (`local_operator/session/credential_ops.py:1-26`). Brokering it would put a
  secret value on the wire to buy nothing: the `bash` command that wants it
  runs on the peer, so the value must exist on the peer regardless.
- **An MCP OAuth grant is never created remotely** (§4.7). The loopback
  callback (`DEFAULT_CALLBACK_PORT = 33441`, `local_operator/mcp/auth.py:94`)
  cannot work from another device. What the mesh adds is a way to *ask the
  owner's human* to run `/mcp login` there; the resulting access token is then
  brokerable like any other.

---

## 1. Inventory: every credential class in this repository

Six classes exist. They are not interchangeable and the design treats each
according to what happens when a second device touches it.

| # | Class | Stored where | Rotating? | What refreshes it | What breaks if used from a 2nd device | May it ever be copied? |
|---|---|---|---|---|---|---|
| 1 | Legacy env credentials | `<config>/credentials.env`, plaintext `key=value`, file `0600` / dir `0700` (`local_operator/credentials.py:24-25,68`) | No (static) | Nothing | Nothing breaks — a static key works everywhere at once | **Only** by explicit per-key opt-in (§2.2) |
| 2 | Encrypted secrets | `~/.local-operator/secrets/` — SQLite AES-GCM + `master.key`, served by the broker daemon over `broker.sock` (`local_operator/secrets/store.py:91-110`, `keys.py:383`, `broker.py:137`) | No | Nothing | The retrieval consent model is *process-ancestry* on one host (`secrets/peer.py:421 authorize_by`, socket `0600` inside a `0700` dir, `broker.py:190-200`) — a second host has no ancestry to prove | **Never.** Not brokered at all (§4.8) |
| 3 | Provider OAuth grants (rotating) | `auth.db` → `auth_credentials`, `credential_type='oauth'` (`providers/auth_store.py:118-124`) | **Yes** — the IdP rotates the refresh token on use | `_ensure_oauth_fresh` (`auth_store.py:903`) → the provider's `refresh_token` fn (`registry.py:32,104`) | **Every device logs out.** Two hosts POSTing the same rotating token → the loser's token is dead; measured as PR-24. The lease that prevents it is a *local SQLite row* (`auth_store.py:142-146`) and cannot span hosts | **Never** |
| 4 | Provider API-key logins | Same table, `credential_type='api_key'`, `source="login"` | No | Nothing | Works from anywhere — same exposure as class 1, and subject to the same opt-in rule | Opt-in only (§2.2) |
| 5 | MCP OAuth grants (rotating) | Same table, `provider='mcp-oauth'`, `identity_key=<server_url>` (`mcp/auth.py:24-41,86`), plus a per-server refresh lock file beside `auth.db` (`_oauth_refresh_lock`, `mcp/auth.py:3031,3066`) | **Yes** for most servers | `ensure_mcp_oauth_fresh` (`mcp/auth.py:3996`) under that file lock | Same rotation race as class 3, adjudicated by a *file lock* instead of a SQLite lease — also host-local. And a new grant can only be created by an interactive loopback callback | The **access token** may be brokered; the **grant** may not |
| 6 | Mobile portal password | macOS Keychain, service `lop-mobile`, account = local user (`local_operator/mobile/auth.py:36-37`) — the repository's ONLY keychain use | No | Nothing; rotated deliberately by `lop mobile password` | The password authorises *that host's* phone portal. A peer has nothing to do with it | **Never** |
| 6b | Device-bound grant (Kimi) | Derivable from class 3's rows plus `<config>/kimi/device-id` (`providers/oauth/kimi.py:66-88`), sent as `X-Msh-Device-Id` (`kimi.py:89-108`) | The access token rotates as usual | The owner-side refresh, which re-sends the **owner's** device id | A borrower replaying the token with its own device id is presenting a fingerprint the provider did not issue it. Broker the **access token** (which then rides the owner's fingerprint), never the device id | **Never** (the id, not the token, is the device-bound half) |

### 1.1 Which classes the mesh touches, and how

- **Classes 3 and 5 (rotating):** broker-only. The owner serves a bearer; the
  requester holds it in memory for the turn and never writes it.
- **Classes 1, 4 (static):** broker by default; **replication** is a separate,
  explicit, per-key opt-in. The default posture is broker, because a broker
  grant is revocable by un-sharing while a copied key is not revocable at all
  without rotating it at the provider.
- **Classes 2, 6:** untouched. Local-only, and every mesh code path must prove
  it does not touch them (§5.4).

### 1.2 One inventory fact that shapes the wire

`get_api_key` (`auth_store.py:1652`) and `get_oauth_access`
(`auth_store.py:1683`) already accept `session_id` and `model_id`. The first
drives stickiness (`_set_sticky`, `auth_store.py:1474`; key
`(storage_provider_id, session_id)`, read at `auth_store.py:1495`), the second
drives model-scoped quota blocks. **Both arguments already cross the boundary
the mesh needs**, so the broker request needs no new resolution semantics — it
replays the same call with the same arguments on the owner.

### 1.3 The rejected alternative, stated once

**The standards citation for this section's rule is RFC 9700 (BCP 240), not
"OAuth 2.1".** Refresh-token protection — rotation on use, sender-constraining,
and binding the refresh token to the client it was issued to — is the BCP's
subject, and the failure this design was built around (two processes racing a
rotating refresh token, PR-24, §0) is the exact failure the BCP exists to
prevent. OAuth 2.1 is still an Internet-Draft (`draft-ietf-oauth-v2-1-16`,
`mesh-prior-art.md` §7) and must not be cited as though it were an RFC.

*Copy `auth.db` to the peer* is the obvious cheap answer and it is exactly the
measured failure: the lease table (`auth_credentials_refresh_leases`) is a
local row, so two hosts would hold live copies of a rotating token with no
shared serialisation. It also collides on `identity_key` dedupe and on
stickiness (`auth_store.py:325`). A5 rejects it; §7 lists the rest.

---

### 1.4 The pattern has standard names, and the names are load-bearing

This document describes, in its own words, a **delegation**: a device that holds a
credential lets another device use it, without the credential moving. That pattern
is standardised and named, and using the names is not decoration — each one
carries a rule that this design would otherwise have to re-derive, and a reviewer
who knows the pattern can check our version against it.

| Name | Source | Rule it brings |
|---|---|---|
| **OAuth 2.0 Token Exchange**, RFC 8693 | `rfc-editor.org/rfc/rfc8693.txt` | The exchange is a *delegation* (§1.1 is literally *Delegation vs. Impersonation Semantics*), and the issued token names the acting party in an **`act`** claim, with `may_act` naming who may act for the subject |
| **on-behalf-of (OBO)** | the enterprise name for the same flow | the requester acts on behalf of the credential's owner, and the owner's consent is the authorisation |
| **credential broker** | SPIFFE's Broker API (Apache-2.0), `spiffe.io` | "Brokers are trusted infrastructure components that can act on-behalf-of workloads … retrieve the SVIDs and trust bundles of workloads they represent" — our relay *is* this component, and the model's rule is that the requester never holds the signing key while identities stay short-lived and rotate automatically |
| **assume-role session policy narrowing** | AWS STS | a delegated grant may only ever be *narrower* than the credential it is derived from |
| **"one holder, one use"** | RFC 9700 / BCP 240 (Jan 2025, updates 6749/6750/6819) | refresh-token rotation on use and binding to the client it was issued to; cite **RFC 9700**, never "OAuth 2.1", which is still an Internet-Draft (`draft-ietf-oauth-v2-1-16`) |

Three consequences, and they are changes rather than vocabulary:

1. **The audit record gains the delegation markers.** `credential_grant` and
   `credential_refresh` carry `act` (the device whose relay performed the
   exchange — the broker) and, on a grant, `sub` (the device it was issued to).
   The field lands in `mesh-incident-response.md` §4.3, which owns the schema; the
   reason it exists is this section. Without it, a forensic reader cannot answer
   "was this credential spent by the device that owns it, or by a peer that asked
   it to" — which is the question a member-removal review actually asks.
2. **A grant is narrowing, by construction.** The requester can never be granted
   more than the owner's own credential holds: same provider, same scope, and the
   returned bearer's expiry is the *earlier* of the token's and `grant_ttl_s`
   (§3.3, §3.5). This is STS's session-policy rule and it is also what makes a
   pool member's borrowed access strictly a subset of the operator's (§6.2).
3. **`act`/`sub` are device ids, never credential material**, so the redaction
   rule (§4.8's list, and A7) is unaffected — the audit log still never holds a
   token, and now it holds *who delegated* as well.

*A note on the alternative framing:* GNAP (RFC 9635) is the successor statement of
this pattern and names the roles; it does not name our topology (an *owner device*
holding the credential for a peer), so this document cites RFC 8693 for the
delegation semantics and SPIFFE for the broker role, and cites GNAP only as the
modern successor framing (`mesh-prior-art.md` §7).

---

## 2. The ownership model

### 2.1 The synced placement document

Written by each device for the credentials **it owns**, merged on receipt, and
persisted at `<config>/network/<network_id>/placement.json`. It carries facts
and only facts: a volatile observation never enters this file.

```json
{
  "schema": 1,
  "network_id": "n_5f3c1a2b4d5e6f708192a3b4",
  "epoch": 7,
  "doc_rev": 41,
  "written_by": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
  "written_at": 1789012345.678,
  "credentials": [
    {
      "key": "openai",
      "provider": "openai",
      "kind": "oauth-rotating",
      "owner_device": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
      "owner_device_name": "damian-mbp",
      "identity_label": "you@example.com",
      "holders": [
        { "device": "d_4b2a91c4e0b87f3a", "scope": "session", "granted_at": 1789000000.0, "granted_by": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b" },
        { "device": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b", "scope": "device",  "granted_at": 1789000000.0, "granted_by": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b" }
      ],
      "replicate": { "allowed": false, "devices": [] },
      "declared_at": 1789000000.0
    },
    {
      "key": "deepseek",
      "provider": "deepseek",
      "kind": "api-key-static",
      "owner_device": "d_4b2a91c4e0b87f3a",
      "owner_device_name": "gpu-pod-3",
      "identity_label": "",
      "holders": [
        { "device": "d_4b2a91c4e0b87f3a", "scope": "device", "granted_at": 1789000100.0, "granted_by": "d_4b2a91c4e0b87f3a" }
      ],
      "replicate": { "allowed": false, "devices": [] },
      "declared_at": 1789000100.0
    }
  ]
}
```

Rules that make this file safe to sync:

- **Single-writer rows.** Only the entry's `owner_device` may write that entry.
  A receiver merges per `key`: an entry whose `owner_device` differs from the
  local entry's `owner_device` is a *change of owner* and requires the previous
  owner's entry to have been explicitly withdrawn (`lop network credential
  own <provider>` on the new owner, which refuses while the old owner is still
  reachable and still holds the credential). Conflict on `doc_rev` resolves to
  the higher `doc_rev` per entry, not per document.
- **No material.** `identity_label` is the operator-facing account string
  (already visible in `lop login-status`). No token, no refresh token, no
  prefix, no hash of either, no `credential_id` — `credential_id` is a row id
  in *one* database and is meaningless off the owner.
- **`holders` is the authorisation.** A device not in `holders` is refused at
  the owner's broker (§4.6); absence is a refusal, not a default-allow.
- **`pool` members start with `holders: []`.** Their own entry is not written
  at all — a pool declares no credentials. It appears in other devices'
  `holders` lists as an ordinary device id when the operator grants it (§6.2).
- **`identity_label` is omitted for `pool` holders.** Stated in the same
  breath as the field: a provider-owned ephemeral member must not learn the
  operator's account email. The broker does not include `identity` in a grant
  served to a member whose record has `kind: "pool"` (§3.3).

### 2.2 The device-local state document (never synced)

`<config>/network/<network_id>/placement.state.json`, mode `0600`:

```json
{
  "schema": 1,
  "observations": [
    {
      "key": "openai",
      "owner_device": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
      "status": "active",
      "reason": "",
      "observed_at": 1789012400.0,
      "retry_after_ms": 0,
      "last_grant_id": "g_2f9c1188a0",
      "last_grant_at": 1789012345.0
    }
  ]
}
```

`status` ∈ `active | owner_offline | not_a_holder | revoked | grant_invalid |
quota_blocked`. It is a local observation with a TTL (`retry_after_ms`), and it
is the *only* place a refusal is cached. This file is why a refused peer does
not produce a retry storm (§4.6) and why the TUI can say "openai is owned by
damian-mbp, last seen 4 min ago" without asking anyone.

### 2.3 Establishing ownership at pairing time

Ownership is **never inferred**; it is confirmed by the human who is already
confirming the SAS (R3, `mesh-network.md:228-231`). The join flow gains one
screen, shown on both ends, between the SAS display and the member-row write:

```
Joining network "lab"
  device:      dev_44ef1a6c05 (gpu-pod-3)
  inviter:     dev_b91c4e77a2 (damian-mbp)
  roles:       drive
  SAS:         6F2Q-8HD4

Credentials damian-mbp will serve to this device:
  openai     (OAuth, you@example.com)   share: yes
  anthropic  (OAuth, you@example.com)   share: yes
  deepseek   (API key)                       share: no   [t] to change

Confirm SAS and share list on BOTH devices. [c]onfirm  [a]bort
```

Defaults, chosen so the common case needs no keystroke:

| Credential kind | Default `holders` grant on join | Why |
|---|---|---|
| `oauth-rotating` (classes 3, 5) | **yes**, `scope: "session"` | This is R13's core case: the peer has no login and must work without the operator re-authenticating. `scope: "session"` means the grant is bound to a named session id, which is the smallest useful authority. |
| `api-key-static` (classes 1, 4) | **no** | A static key is a bearer with no expiry and no rotation; granting it is a permanent capability increase. One keystroke when the operator wants it. |
| `secret` (class 2) | not offered | Not brokered (§4.8). |
| Anything, when the joining member is `kind: "pool"` | **yes** for `oauth-rotating`, `scope: "session"`, and the operator sees an extra line: `ephemeral member — the grant dies with the member lifecycle` | §6. |

`lop network credential share <provider> --with <device> [--scope session|device]`
and `... --revoke <device>` change the list afterwards; both are
owner-device-only verbs and both write an audit record (§4, DOC2 §4.3).

### 2.4 How a session resolves which device serves its credential

The cascade is the existing one with **one new rung inserted last**, and the
rung is consulted only when the local tiers produced nothing:

```
MeshAwareAuthStore.get_api_key(provider, session_id, ...)
  1. delegate AuthStore.get_api_key(...)            # auth_store.py:1652, unchanged cascade
  2. if it returned a key            -> return it   # local-first, always
  3. read the session's credential binding (a transcript row, below)
  4. if binding.owner_device != this device -> broker (§3)  -> bearer
  5. else -> ask this device's relay whether a peer owns `provider` (§3.6a)
  6. else -> return None                             # existing "no credential" path
```

**A session must not silently change accounts**, so a session records the
resolution that first served it, as a **transcript custom row** — the
established pattern for session bookkeeping that has to travel with the
transcript (`session_spend.v1`, `local_operator/session/spend.py:47-56`;
membership in `session._PERSISTABLE_CUSTOM_TYPES`,
`local_operator/session/session.py:772`). A transcript row rather than a
sidecar file means the binding rides the existing fork and mobility copy
(`fork.fork_session`, `local_operator/fork.py:167`) with **no change to either
path**, and never enters LLM context.

```json
{"type": "custom", "custom_type": "mesh_credential_binding.v1",
 "details": {"schema": "lop.mesh.credential_binding.v1", "version": 1,
             "provider": "openai", "owner_device": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
             "identity_label": "you@example.com",
             "policy": "local-first", "bound_at": 1789012345.0,
             "writer": "47201:1789012000"}}
```

- **Replacement state, never a delta** — newest row per provider wins, exactly
  as the spend accumulator states its own rule (`spend.py:14-27`). Two writers
  cannot corrupt it.
- `policy` is `local-first` (default) or `owner` (set when the creating human
  says "run this on my account"). `local-first` means a device that *does* have
  its own login keeps using it; `owner` means it brokers even so. Both are
  legitimate; the point is that the choice is recorded once and honoured
  afterwards rather than re-decided per call.
- **A move preserves the binding.** So does a fork. This is the mechanism
  behind R13: the credential follows the *session*, and what actually follows is
  a pointer to the device that owns it.

---

## 3. The broker protocol

### 3.1 Two legs, two transports, one vocabulary

```
session runtime (on borrower B)          relay B                  relay A (owner)         AuthStore (A)
  MeshAwareAuthStore                      forward                  MeshCredentialBroker    auth.db
        │  leg 1: loopback control socket,     │  leg 2: peer link,        │  in-process
        │  JSON-lines, relay's control_key     │  A1 vocabulary, MTLS     │  get_api_key(...)
        └──────────────────────────────────────┴──────────────────────────┘
```

**Leg 1 — runtime ↔ local relay.** The runtime dials the relay's own control
socket. The relay publishes a record in the `run/peers` namespace (A2) which
**must carry `control_port`, `control_key`, `protocol`, `capabilities` and
`install_root`** — the same fields and the same `0600`-under-`0700` trust
boundary a session record already has (`registry.py` `run_dir`; `SessionRecord`
docstring, `session/runtime/types.py:712-724`). A runtime finds its relay by
matching `install_root` against its own config root; it never guesses a port.
No change to the session runtime's `ControlOp` table
(`local_operator/mobile/types.py:279`) is needed — the runtime is a *client*
here, not a dispatchee.

**Leg 2 — relay B → relay A.** The peer link defined by
`mesh-transport-identity.md`. The frame declares `"locality": "remote"`, which
is precisely the case `ClientLocality` was added for
(`session/runtime/types.py:172` and its docstring at `:160-172`), so the owner
can make authorisation decisions on it (`mesh-network.md` §5.8).

Both legs use one JSON-lines op vocabulary. The ops are **relay-scoped**, so
they live in a new `Literal` in `local_operator/network/types.py` rather than
being appended to `ControlOp` (which is the *session* runtime's table):

```python
# local_operator/network/types.py

#: LOCAL vocabulary (viewer/runtime -> this device's relay), named `net_*` per
#: `mesh-transport-identity.md` §2.5 so a reader can tell from the frame alone
#: which boundary it crossed.
LocalCredentialOp = Literal[
    "net_credential_grant",     # give me a usable token for (provider, session)
    "net_credential_report",    # the borrowed bearer failed; attribute it at home
    "net_credential_placement", # the placement document changed
    "net_credential_repair",    # ask the owner's human to re-run an interactive login
]

#: PEER vocabulary. `net_broker` is RESERVED by `mesh-transport-identity.md`
#: §6.4 with capability `broker_credential` (§7.1); this document fills in its
#: body. One peer op with a `kind` discriminator, not four peer ops: the
#: transport's capability model authorises per op, so four ops would mean four
#: capability rows for one authority.
BrokerFrameKind = Literal["grant", "report", "placement", "repair"]
PEER_BROKER_OP = "net_broker"
BROKER_CAPABILITY = "broker_credential"
BROKER_CAP_STRING = "credential-broker-v1"   # advertised in hello/welcome `caps`
```

### 3.2 Literal frames

**Request (leg 1, runtime → relay):**

```json
{"key": "<relay control_key>", "client": "runtime", "locality": "local"}
{"op": "net_credential_grant", "req": 1, "kind": "grant",
 "provider": "openai", "session_id": "2026-09-19T18-04-11_ab12",
 "want": "bearer", "model_id": "gpt-5.4", "force_refresh": false}
```

The auth frame is the relay's control-socket convention verbatim
(`mesh-transport-identity.md` §2.5, modelled on
`session/runtime/server.py:1885-1912`): the first frame carries the `control_key`
read from the `run/peers` record (`0600` inside `0700`), compared with
`hmac.compare_digest`; a wrong key closes without a reply.

**Request (leg 2, relay B → relay A), after B has decided to forward:**

```json
{"op": "net_broker", "req": 41, "kind": "grant", "locality": "remote",
 "network_id": "n_5f3c1a2b4d5e6f708192a3b4", "epoch": 7,
 "from_device": "d_4b2a91c4e0b87f3a", "from_device_name": "gpu-pod-3",
 "for_session": "2026-09-19T18-04-11_ab12",
 "for_session_origin": "d_4b2a91c4e0b87f3a",
 "provider": "openai", "want": "bearer", "model_id": "gpt-5.4",
 "force_refresh": false}
```

`req` is the transport's integer request id (`mesh-transport-identity.md` §6.4);
`locality` is always `"remote"` on this leg, which is what
`ClientLocality` (`session/runtime/types.py:172`) was added for and what
`mesh-transport-identity.md` §7.4 requires. The op is used only when both sides
advertised `credential-broker-v1` in `caps`; an older peer answers
`error: unknown op`, which the client maps to `unsupported` below.

**Success reply (leg 2), and the same `detail` object on leg 1's `ack`:**

```json
{"op": "ack", "req": 41, "detail": {
  "kind": "grant", "grant_id": "g_2f9c1188a0",
  "access_token": "<opaque bearer; redacted in this document>",
  "token_kind": "bearer",
  "token_expires_at_ms": 1789015945678,
  "grant_expires_at_ms": 1789013245678,
  "credential_ref": {"owner_device": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b", "owner_device_name": "damian-mbp",
                     "provider": "openai", "kind": "oauth", "credential_id": 12},
  "identity": {"account_id": "org_9x2", "email": "you@example.com", "org_id": "org_9x2"},
  "scope": {"kind": "session", "session_id": "2026-09-19T18-04-11_ab12"},
  "served_by": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
  "refreshed": true,
  "latency_ms": 214
}}
```

**Failure reply:**

The transport's envelope has no `code` field, so the machine-readable code rides
**inside the `detail` object of an `ack`** rather than on the envelope — a refused
grant is a routed request that was answered, not a transport failure, and the
distinction matters because the transport's own error codes (`not_authorised`,
`not_a_member`, `unknown_op`, …, `mesh-transport-identity.md` §7.2) must not be
confused with the broker's:

```json
{"op": "ack", "req": 41, "detail": {"kind": "error", "code": "not_a_holder",
 "message": "damian-mbp does not share 'openai' with gpu-pod-3",
 "retry_after_ms": 60000, "key": "openai",
 "owner_device": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b"}}
```

`code` is a closed set — the requester's behaviour is keyed on it, never on the
message text:

| `code` | Meaning | Requester behaviour |
|---|---|---|
| `no_local_credential` | Owner holds nothing for this provider either | Cache 300 s, surface incident, do not retry |
| `owner_offline` | No reachable owner | Cache 15 s, retry once on the next provider call |
| `not_a_holder` | This device is not in `holders` | Cache 60 s, surface incident, never retry this session |
| `revoked` | Was a holder, no longer | Cache 300 s, surface incident naming the revoke time |
| `epoch_stale` | Epoch mismatch — a missed rotation (§DOC2 §2.4) | Re-handshake; do not cache |
| `grant_invalid` | The owner's grant is dead at the IdP | Cache 300 s, surface the re-login remedy |
| `refresh_failed` | Transient refresh failure on the owner | Cache `retry_after_ms`, retry |
| `quota_blocked` | The owner's row is blocked for this model | Cache `retry_after_ms`; the existing failover rotation runs |
| `interactive_required` | Only an interactive login can help (MCP) | Surface the repair verb (§4.7) |
| `rate_limited` | Broker-side backpressure | Honor `retry_after_ms` |
| `unsupported` | Older peer: no `credential-broker-v1` cap, or a transport `unknown_op` | Cache for the session, degrade to the old "no credential" path |
| `not_authorised` / `not_a_member` | The transport's own refusal (`mesh-transport-identity.md` §7.2) | Map to `revoked` (§4.2) |

### 3.3 What the owner returns, and what it must never return

**The grant is a narrowing delegation, and this section is where that is
enforced** (RFC 8693 delegation; assume-role session-policy narrowing —
§1.4). Three rules, each checkable in the code rather than in prose: the requested
provider must be one the owner actually holds a usable credential for; the granted
scope may only ever be a subset of what the owner's own credential carries; and
the expiry returned to the requester is `min(token_expiry, now + grant_ttl_s)`
(§3.5's table). A requester that asks for something the owner does not have is
refused `credential_grant_refused`, never served a credential borrowed from
somewhere else — "narrower" means the delegated authority is bounded by the
delegator's, which is also why a `pool` member's borrowed access is a subset of
the operator's by construction (§6.2).

**Returns:** one bearer, its real expiry, a grant expiry, a `credential_ref`
that names the owner and the owner's local row id, and (for non-`pool`
requesters) the identity fields the requester needs to label the account.

**Never returns, under any circumstance:** the refresh token; the row's raw
`data` dict (`auth_store.py:171-183`); any other credential of the provider or
of any other provider; the master key or any `secrets/` material; the network
secret; a `control_key` of anything (`mesh-network.md` §5.4); a token for a
provider or session the request did not name (there is no wildcard request);
and, to a `pool`-kind requester, `identity.account_id`, `identity.email` and
`identity.org_id` — those are omitted from the reply entirely rather than
blanked, so a pool cannot learn the operator's account.

The grant is **in-memory only on the requester**. `MeshAwareAuthStore` holds it
in a `dict[(provider, session_id)] -> Grant` for the process lifetime of the
turn, and nothing in the mesh path calls `upsert_credential`. This is not a
convention: §5.2 asserts the requester's `auth.db` is logically unchanged by a
brokered run.

### 3.4 Refresh discipline: the local lease, honoured across hosts

The owner serves the request by calling **the existing call**:

```python
key, row = await auth_store._resolve(provider, session_id, model_id=model_id,
                                     force_refresh=force_refresh)
```

`_resolve` → `get_api_key` (`auth_store.py:1652`) →
`_ensure_oauth_fresh` (`auth_store.py:903`) → `_try_refresh_lease`
(`auth_store.py:847`) → the provider refresh fn. Consequently:

- **`AUTH_REFRESH_LEASE_MS` never needs to span hosts.** It is a row in the
  *owner's* `auth_credential_refresh_leases` (`auth_store.py:142-146`) and the
  only process that contends for it is on the owner. Two runtime processes on
  the owner (a TUI session and a detached runtime both brokering to the same
  credential) serialise exactly as they do today — this is the mechanism that
  already fixed PR-24, unreformed.
- **`OAUTH_REFRESH_SKEW_MS` (`auth_store.py:61`) governs the owner's own
  decisions**, unchanged: the owner refreshes pre-emptively at 60 s of
  remaining life whether it is serving itself or a peer.
- **The rotation-race guards keep working.** `_ensure_oauth_fresh`'s two
  "refresh race … keeping its token" guards (`auth_store.py:984-1000`,
  `auth_store.py:1014-1026`) stay load-bearing for the owner's *own* sibling
  processes; the mesh adds no new contender.
- **The claim to verify (§8 Q1):** a cross-host race is impossible because a
  contender must be inside `_ensure_oauth_fresh` on the owner. The design's
  invariant is therefore **"no code path outside the owner's process ever calls
  a provider refresh function"**, and §5.3 tests it directly.

#### Cross-host serialisation, the part that is new

`AUTH_REFRESH_LEASE_MS` serialises *processes*. The broker must additionally
serialise *requests*, or N peers asking at once each get a serialised-but-
sequential `_resolve` — correct, wasteful, and (with `force_refresh=true`) able
to spend a second POST after the first one already succeeded. So the owner's
broker adds one coalescing map, which is a *request* lock and nothing more:

```python
class MeshCredentialBroker:
    _inflight: dict[tuple[str, str | None, bool], asyncio.Future[dict[str, Any]]]
    _join_window_s: float = 0.25          # requests for the same key within the window join
```

- The first request for `(provider, session_id, force)` creates the future and
  performs the resolve; requests arriving within `_join_window_s` join it and
  receive the same grant (and therefore the same `grant_id`).
- A `force_refresh=true` request is a *different* key: it is an intentional
  new POST and must not be deduplicated against a non-forced resolve.
- `force_refresh` is refused for a rotating credential from a **non-`admin`
  member** (§4) — gratuitous forced refreshes are the cheapest way to make an
  IdP start invalidating tokens, and R14's whole point is to keep the number of
  POSTs down to the number of legitimate refreshes.

### 3.5 TTL, caching, and refresh scheduling

| Concern | Value | Where enforced | Why |
|---|---|---|---|
| Owner's answer for a token with remaining life ≥ 120 s | Served from the existing row, `refreshed: false` | `_needs_refresh`, `auth_store.py:68-80` | No gratuitous POSTs. Never refresh what is alive. |
| Requester's copy of the bearer | `expires_at = min(token_expires_at_ms, now + grant_ttl_s)`; re-ask at `expires_at - 120 s` | `MeshAwareAuthStore` | 120 s > the owner's own 60 s skew, deliberately: the requester must not be the thing that discovers expiry. |
| `grant_ttl_s` default | **900 s** (15 min) | `network.credentials.grant_ttl_s` in `config.yml` | Bounds the window in which a revoked peer still holds a live bearer (§3.7). |
| Requester's local coalescing | one `asyncio.Future` per `(provider, session_id)` | `MeshAwareAuthStore._grant_futures` | A single turn issues several provider calls; they must share one grant. |
| Owner's refresh scheduling with several askers | one POST per `(credential_id, force)` per `_join_window_s`; the existing lease for the rest | §3.4 | N peers ⇒ 1 POST, not N. |
| Proactive top-up (`/usage` style) | Not added. A borrow is demand-driven. | — | A background refresher on the owner would be a second scheduler for something the lease already bounds; the demand path is sufficient. |

### 3.6 The code seams (exact names)

**(a) `MeshAwareAuthStore`** — new class,
`local_operator/network/credentials/store.py`:

```python
class MeshAwareAuthStore:
    """AuthStore, with brokering as the LAST rung of the cascade."""

    def __init__(self, local: AuthStore, *, mesh: MeshCredentialClient | None,
                 config_dir: Path, session_id: str) -> None: ...

    # FailoverAuthStore (providers/failover.py:2719)
    async def get_api_key(self, provider: str, session_id: str | None = None, *,
                          force_refresh: bool = False, read_only: bool = False,
                          model_id: str = "", exclude_keys: Collection[str] | None = None,
                          exclude_credential_ids: Collection[int] | None = None) -> str | None: ...

    # FailoverAuthStore — the failure must be accounted where the row lives
    def rotate_sibling(self, provider: str, session_id: str | None, error: BaseException,
                       api_key: str | None = None, block_ms: int = 60_000, *,
                       model_id: str = "") -> bool: ...

    # OAuthAccessSource (failover.py:2751)
    async def get_oauth_access(self, provider: str, session_id: str | None = None, *,
                               force_refresh: bool = False, read_only: bool = False,
                               model_id: str = "",
                               exclude_keys: Collection[str] | None = None,
                               exclude_credential_ids: Collection[int] | None = None
                               ) -> OAuthAccess | None: ...

    # CredentialLister (failover.py:2733)
    def list_credentials(self, provider: str) -> list[StoredCredential]: ...
```

- `exclude_keys` / `exclude_credential_ids` name rows **in the owner's**
  database. A brokered grant that the requester then rejects (a 401, a
  rotation error) cannot be expressed to the owner as an exclusion until it is
  reported — which is what `credential_report` is for (§3.6c). Until the report
  lands, the requester passes the flags through unchanged and the owner applies
  them to its own rows; a `credential_id` from another device is meaningless,
  so the requester translates them to a `credential_ref`-scoped exclusion and
  the owner maps it back through `credential_ref.credential_id`. Stated plainly
  because it is the one place the two id spaces meet.
- `list_credentials` answers from the **local** store, plus the placement
  entries this device is a holder for, with a synthetic `StoredCredential` that
  carries `identity_key=None` and no `data`. Its only consumer is the
  "why is there no bearer" sentence (`CredentialLister`'s docstring), so a
  synthetic row must never be able to reach a provider.

**(b) `MeshCredentialClient`** — new, `local_operator/network/credentials/client.py`:

```python
class MeshCredentialClient:
    async def grant(self, *, provider: str, session_id: str, want: Literal["bearer"],
                    model_id: str = "", force_refresh: bool = False,
                    deadline_s: float = 8.0) -> Grant | BrokerError: ...
    async def report(self, *, grant: Grant, error: BaseException) -> None: ...
    async def repair(self, *, provider: str, server_url: str = "") -> None: ...
```

`deadline_s = 8.0` mirrors the broker's existing `REQUEST_TIMEOUT_S = 10.0`
(`secrets/broker.py:78`) minus a margin: a brokered grant must never be able to
hang a turn longer than the local secret broker would.

The dial mechanics are `peer_client`'s: `asyncio.open_connection` to the
relay's loopback port, auth frame first, then match the `ack`/`error` by `req`
while tolerating intervening pushes — the exact shape
`send_peer_message` uses (`mobile/peer_client.py:104-140`, incl. its
`_FrameReader` tolerance for an oversized first frame).

**(c) `MeshCredentialBroker`** — owner side, `local_operator/network/credentials/owner.py`:

```python
class MeshCredentialBroker:
    def __init__(self, *, auth_store: AuthStore, placement: PlacementDocument,
                 audit: AuditWriter, config: MeshCredentialConfig) -> None: ...

    async def on_request(self, frame: dict[str, Any], peer: PeerIdentity
                         ) -> dict[str, Any]: ...
    async def on_report(self, frame: dict[str, Any], peer: PeerIdentity) -> None: ...
```

`on_request` order, and every step is a refusal by default:

1. **Authorise, in two halves.** The transport already performs the network
   half — membership at the current epoch, the `broker_credential` capability,
   and `locality == "remote"` — at its single chokepoint
   (`local_operator/network/authorize.py`, `mesh-transport-identity.md` §7.2,
   §7.4); this document must not re-implement it. The broker adds the one check
   the transport does not know about: **the peer's device id ∈ `holders` for
   `key`** in `placement.json` (§2.1).
2. **Refuse forced refreshes** from non-`admin` holders (§3.4).
3. **Coalesce** on `(provider, session_id, force)` (§3.4).
4. **Resolve** via `auth_store._resolve(provider, session_id, model_id=...)`.
5. **Audit** one `credential.grant` record (DOC2 §4.3) — never the token.
6. **Reply** with the `detail` in §3.2, or the `code` in §3.2's table.

**(d) MCP** — `ensure_mcp_oauth_fresh` (`mcp/auth.py:3996`) gains one branch
*before* it acquires the file lock: if the row for `identity_key=<server_url>`
is owned by another device, ask the broker for the access token and return it
**without writing anything** to the local `auth.db`. `McpTokenStorage`
(`mcp/auth.py:855`) is untouched: it continues to serve rows this device owns,
so nothing about the local MCP path changes when the row is local.

**(e) No change to `auth_store.py`.** The broker uses `_resolve` / `get_api_key`
(`auth_store.py:1652`), `get_oauth_access` (`auth_store.py:1683`) and
`rotate_sibling` (`auth_store.py:2112`) — all existing public or
stable-private entry points. If implementation finds it needs a
`credential_id`-carrying accessor for *static* rows that `get_oauth_access`
does not surface, add it as a purely additive method there (§8 Q2).

### 3.7 Threat notes specific to the broker

- **Revocation latency is bounded, not zero.** A peer removed from `holders`
  may still hold a live bearer until `grant_ttl_s` (≤ 900 s) elapses. We do
  not attempt bearer revocation because no provider offers a per-bearer revoke;
  the honest mitigations are the short TTL, the immediate link-level refusal
  (`mesh-network.md` §5.6) and the operator's ability to rotate the credential
  at the provider. This must be stated in the guide, not glossed.
- **A malicious holder can spend the operator's quota.** `holders` is a real
  capability increase and is granted explicitly. Least authority is the
  defence: `scope: "session"` by default, `api-key-static` default-off.
- **The owner is a single point of failure for the borrower.** When the owner
  is offline, a borrower with no local login has no model access. That is
  R14's intended trade (spending a rotating token off-device is worse), and
  §4.1 gives the operator a one-command remedy.
- **A pool must not persist a bearer.** §6.3 makes the pool's grant in-memory
  and expiry-bounded, and forbids the pool from declaring credentials at all.

---

## 4. The failure matrix

Every row states the observable behaviour and the exact operator-facing
sentence. Sentences come from one place — a new `render_broker_error()` in
`local_operator/network/credentials/messages.py` — mirroring the existing
session-incident formatter (`incidents.format_credential_message`,
`local_operator/incidents.py:765`) so credential copy has one home per surface,
which is the rule `incidents.py` already states for its own four records.

| # | Failure | Observable behaviour | Operator-facing message |
|---|---|---|---|
| 4.1 | **Owner offline** | `credential_grant` cannot reach a holder. `grant()` returns `owner_offline`; the incident is journalled as a `session_incident` custom row (live context + persisted, `incidents.py:1-20`) with category `credentials`; failover may still try another provider/model and its own notices are unchanged. No retry loop. | `No credential for 'openai' is reachable: damian-mbp owns it and was last seen 4 min ago. Reconnect that device, or run 'lop login openai' here to use your own account.` |
| 4.2 | **Owner revoked this device** (holder removed / member removed) | The request is refused at the owner's broker with `revoked`/`not_a_holder`; the requester writes `placement.state.json` with that status and a 300 s TTL; the TUI's credential view shows the credential struck through with the revoke time. | `Credential 'openai' is owned by damian-mbp, which no longer shares it with this device (revoked 12:04). Ask again on damian-mbp ('lop network credential share openai --with gpu-pod-3'), or log in here.` |
| 4.3 | **Provider refresh rejected** (`invalid_grant`) | The owner raises `CredentialInvalidError` (`auth_store.py:154`) — the store's existing permanent-failure type — and the broker maps it to `grant_invalid`. The owner's row is **not** disabled (that class is explicitly a statement about the grant, `auth_store.py:154-170`), so the operator's `/usage` panel keeps showing the login. The requester disables nothing — it has no row. | `The openai login on damian-mbp can no longer be refreshed (invalid_grant). Re-run 'lop login openai' on damian-mbp.` |
| 4.4 | **Token expiry mid-turn** | The runtime's next resolve re-asks (`expires_at - 120 s`, §3.5). The failover driver's existing re-resolve on a credential-invalidated error (`is_invalidated_credential_error`, `failover.py:1541`; `is_direct_credential_rotation_error`, `failover.py:1564`) reaches the mesh rung on its second attempt. | On success: **nothing**. A gratuitous "refreshed your credential" toast on a normal mid-turn re-grant is noise; the grant is logged, not announced. On failure it becomes 4.1/4.2/4.3. |
| 4.5 | **Two peers ask at once** | The owner coalesces (§3.4): **one** provider POST, one `credential.refresh` audit record, two `credential.grant` records sharing one `grant_id` when they arrived inside the join window, or two `grant_id`s from one refresh when they arrived outside it. Both requesters get a working bearer. | Nothing user-facing. |
| 4.6 | **A peer asks for a credential it is not a holder for** | Refused with `not_a_holder`; `credential.grant_refused` audit record on the owner with the requesting device, the key and the reason; the requester caches the refusal for 60 s (`placement.state.json`) and does **not** re-ask this session. | Requester: `damian-mbp does not share 'openai' with this device.` Owner (in `lop network log` / the TUI's network view): `refused a credential request for openai from gpu-pod-3 (not a holder)`. |
| 4.7 | **MCP interactive login needed remotely** | `ensure_mcp_oauth_fresh` cannot refresh, so the mesh rung returns `interactive_required`; nothing opens a browser on either device. The remote verb is refused with the existing notice (`REMOTE_GRANT_NOTICE`, `mcp/grants.py:65`, used at `grants.py:257`) — **unchanged**. The mesh adds the *repair* path: `credential_repair` reaches the owner's relay, which raises a durable notice on the owner (TUI toast + `inbox.jsonl` entry + audit record). The owner's human runs `/mcp login <server>`; from then on the access token is brokerable and the remote session's next attempt succeeds. | Remote session: `/mcp login opens a browser and stores credentials on the machine running the session — run it from a terminal on that machine.` Owner, on receipt: `gpu-pod-3 needs you to re-run '/mcp login datadog' here. [r]un it now  [l]ater` |
| 4.8 | **A remote session needs a `lop secret` value** | Not brokered, by design. The `bash` command runs on the peer, so the value must exist there; the session-scoped `/credential <key> <value>` verb (routed to the runtime, `session/credential_ops.py:1-26`) writes it there, on the peer, in the peer's store. A `$(lop secret get X)` in a peer session retrieves from the peer's broker, whose consent model is peer *process ancestry* on that host (`secrets/peer.py:421`) and therefore works exactly as it does locally. | If absent: `The secret 'X' is not stored on gpu-pod-3, where this session runs. Set it there: /credential X <value> (or 'lop secret set X' on gpu-pod-3).` |
| 4.9 | **Owner's row is quota-blocked** | The owner's `_resolve` excludes blocked rows (`is_blocked_for_model`, `auth_store.py:79-80` region) and either picks a sibling or returns nothing. A *sibling pick* is a real behaviour change for a remote session and must be visible: the grant's `credential_ref.credential_id` changes, and the requester writes a `mesh_credential_binding.v1` row with a new `credential_id` only when the *owner* changed — never silently re-binding to a different account for the same provider. | If a sibling exists: `Openai quota is exhausted on damian-mbp; this turn is running on its other openai login.` If not: the existing no-credential sentence, plus `(openai is rate-limited on damian-mbp until 14:20).` |

**Failure attribution goes home.** When a *borrowed* bearer is rejected by the
provider (401, 429, a rotation error), `MeshAwareAuthStore.rotate_sibling`
cannot act: the row is not here. It sends `credential_report`
(`{provider, credential_ref, error_class, retry_after_ms, http_status}`) and
the **owner** calls the existing `AuthStore.rotate_sibling` (`auth_store.py:2112`)
with the error, so the block table, the soft-delete and the stickiness decision
all land on the device where they are true. `rotate_sibling`'s `bool` return is
forwarded so the requester's failover driver still learns whether a sibling
remains. Report delivery is best-effort with **one** retry (a lost report costs
a re-spend, not correctness); it is audited either way.

---

## 5. Non-regression guarantees for the existing local paths

R16 is a hard requirement and the failure mode it names (a later local run
trips over a store the mesh left in a foreign state) is exactly what §5.2 and
§5.3 assert against. Each guarantee below is a named test, not a hope.

### 5.1 Structural: the wrapper does not exist without a network

`session_factory.py:3398` constructs the AuthStore for a runtime —
`auth_store = AuthStore(credential_manager=credential_manager)` — and passes it
onward at `:3408`. That is the single construction site for a session's store
(`AuthStore(` appears nowhere else in `session_factory.py`). The rule:

```python
# session_factory.py, replacing :3398
auth_store = build_auth_store(credential_manager)   # new helper, network/credentials/store.py
```

`build_auth_store` returns the plain `AuthStore` when
`config_dir()/network/` has no active network for this install, and
`MeshAwareAuthStore(AuthStore(...), mesh=...)` otherwise. **Assertion:**
`tests/unit/network/test_no_network_keeps_the_plain_store.py` pins that a
config dir with no `network/` directory gets an object whose type is exactly
`AuthStore`, and that `MeshAwareAuthStore.__init__` is never entered (a sentinel
attribute on the class proves it). A 0-peer install therefore cannot execute one
line of mesh credential code — the strongest available form of "no regression".

### 5.2 The requester's `auth.db` is untouched by a brokered run

**Assertion:** in the one-peer topology (§9), take a *logical* dump of the
borrower's `auth.db` (`sqlite3 .dump`, not the file bytes — WAL means bytes
differ for unrelated reasons) before and after a brokered turn, and require the
two dumps to be **identical**. Additionally assert `auth_credential_refresh_leases`
is empty on the borrower afterwards.

The owner's `auth.db` is allowed to change, and only in the way a local run
would: `updated_at` moves on the one refreshed row, no new row appears, no row
is disabled, and the lease row is released
(`_release_refresh_lease`, `auth_store.py:879`).

### 5.3 Credential rotation stays single-host

**Assertion** (this is the PR-24 regression test, run *across* hosts):
`tests/e2e/test_mesh_credential_rotation.py` stands up two devices against a
stub IdP that (a) rotates the refresh token on every exchange and (b) invalidates
the previous one. It runs one turn on the owner's own session and one on the
borrower's session concurrently, and asserts: **exactly one** POST reached the
token endpoint; both turns succeeded; neither device's stored refresh token was
invalidated.

A second assertion at the same time, because it is the failure that would be
invisible otherwise: with the owner's relay stopped, the borrower's turn must
**fail cleanly** rather than fall back to POSTing its own (stale) copy — the
borrower has no copy, and the test asserts the token endpoint saw zero POSTs
from the borrower's process tree (identified by the stub's per-connection
recorded `peer_pid` via `SO_PEERCRED`/`LOCAL_PEERPID`).

### 5.4 The fragile stores are provably untouched

- **Keychain (class 6).** `tests/unit/network/test_no_keychain_from_mesh.py`
  patches `subprocess.run` and fails the test if any mesh code path invokes
  `security find-generic-password` (`mobile/auth.py:36-47` is the only
  legitimate caller in the repo).
- **MCP callback port (class 5).** The same test patches `socket.socket.bind`
  and fails if any mesh path binds `33441` (`mcp/auth.py:94`).
- **`secrets/` (class 2).** `tests/unit/network/test_mesh_never_opens_secrets.py`
  asserts `local_operator/network/` imports nothing from `local_operator.secrets`
  except the (unused-here) audit-chain helper, and that no test-double broker is
  started. This is what makes §4.8's "not brokered" a fact rather than a claim.
- **`credentials.env` (class 1).** The mesh reads it only through the existing
  cascade tier 5 (`auth_store.py:1-20`, "env var — including the legacy
  `credentials.env` file read through `CredentialManager`"). Assertion: with a
  network configured and a local key present, the broker is never dialled
  (`test_a_local_login_never_dials_the_broker`).

### 5.5 A moved session carries its binding and re-resolves identically

**Assertion:** move a session owner→borrower with `--keep` off and on
(`mesh-network.md` §6), then assert (a) the `mesh_credential_binding.v1` row is
present in the destination transcript and byte-identical to the source's, (b)
the destination's resolve picks the same `owner_device`, and (c) the source
device's `AuthStore` never sees a `force_refresh` as a result of the move — a
move is not a credential event.

### 5.6 The full local matrix still passes

The 0-peer row of `mesh-network.md` §10's topology table is the regression
gate: an install with no network must behave exactly as today. That row is the
whole-tree unit suite plus the existing auth/MCP suites:

```sh
ISO=$(mktemp -d)
env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \
  TERM=xterm-256color .venv/bin/python -m pytest \
  tests/unit/providers tests/unit/mcp tests/unit/secrets tests/unit/mobile -q
```

(Isolation rules per AGENTS.md §Environment and §Isolating a run: `env -i`
strips `CMUX_*`/`LOP_*`, `HOME` moves the cache and agent home as well as the
config dir. The whole-tree gate is AGENTS.md §Quality gates and is not replaced
by this targeted run.)

---

## 6. Forward compatibility: metered capacity

R20/R21 ask that a provider-owned ephemeral member can borrow credentials
"without ever holding one", and that per-session credential attribution be
meterable. Both fall out of the design above; nothing new is needed beyond two
additive fields.

### 6.1 The pool member

A `pool` member (`mesh-network.md` §9.1) is an ordinary member with
`kind: "pool"` and `lifecycle: active|provisioning|draining|expired`. It
declares no credentials of its own, ever, and its join flow skips the "what
this device owns" half of the pairing screen (§2.3) because there is nothing to
declare.

### 6.2 It borrows; it never holds

- The operator grants it `holders` entries with `scope: "session"` at invite
  time; the pairing screen prints `ephemeral member — the grant dies with the
  member lifecycle`.
- Every grant to a pool member is `grant_ttl_s`-bounded (§3.5) and
  **in-memory only**. A pool that restarts must ask again, so a pool that
  outlives its grant window with no reachable owner simply stops being able to
  call a provider — which is the correct failure for a device that is about to
  be billed for its compute anyway.
- `identity` is omitted from pool grants (§3.3), so a provider-owned device
  never learns the operator's account.
- `expired` lifecycle ⇒ the grant path returns `not_a_holder`, because the
  member's `holders` entry is withdrawn with the lifecycle change. This is the
  same refusal 4.6 exercises, so the pool's shutdown needs no new code.

### 6.3 Attribution, and where metering reads it

Attribution already has an owner: the `session_id` the requester sends. The
design adds exactly two things:

1. **The audit stream carries `session_id` on `credential.grant`** (DOC2 §4.3).
   A billing consumer therefore gets `(at, session_id, provider,
   credential_ref.owner_device, grant_id, refreshed)` without reading session
   state — which is precisely what `mesh-network.md` §9.4 asks for
   ("Metering is an event stream, not a field").
2. **The session spend row gains a credential-owner field.** `serving_identity`
   (`local_operator/session/spend.py:107`) already stamps the model that
   actually served the call. Add `credential_owner` and `credential_ref` to the
   returned dict, sourced from the grant the request used, so a call served by
   a borrowed credential is attributable to the owning device.

   **Version rule:** `SESSION_SPEND_VERSION` (`spend.py:58`) stays **1**. The
   fields are additive inside the record's `details` map, and a reader that
   does not know them degrades to provider/model attribution — which is exactly
   what today's readers do (`_identity_of`, `spend.py:86-105`, already tolerates
   absent fields). Bump to 2 only if the *meaning* of an existing key changes,
   which this does not.

Per-session credential attribution is therefore **in place before the pool
exists**, which is what makes R21 true: the metering implementation is a new
consumer of an existing event stream, not a change to what a session is.

---

## 7. Rejected alternatives

| # | Alternative | Why rejected |
|---|---|---|
| 7.1 | Replicate `auth.db` to every member | Reproduces PR-24 exactly: `AUTH_REFRESH_LEASE_MS` is a local row (`auth_store.py:142-146`) so two hosts hold live copies of one rotating token with no shared serialisation. Also collides on `identity_key` dedupe and stickiness (`auth_store.py:325,1474`). This is the measurement A5 exists for. |
| 7.2 | Replicate `credentials.env` wholesale | Static keys are the one class where copying *works*, which is exactly why it needs per-key opt-in: a blanket copy turns a read-only member into a bearer-key holder with no revocation path. |
| 7.3 | Proxy the provider **API call** through the owner instead of brokering the bearer | Moves all model traffic through one device: the owner becomes a latency bottleneck and a single point of failure, streaming and token accounting have to be re-plumbed, and the security gain over brokering a bearer is ~zero (the borrowed bearer is already scoped and short-lived). Much larger change, no benefit. |
| 7.4 | Mint a long-lived bearer at pairing and let each peer keep it | A second stored holder is precisely what this design avoids; revocation becomes impossible without a provider-side rotation, and the TTL bound (§3.7) disappears. |
| 7.5 | Share one `auth.db` on a synced folder / network filesystem | SQLite over a network FS plus a lease row that assumes local atomic upsert (`_try_refresh_lease`'s single-statement upsert, `auth_store.py:855-877`) is a correctness hazard, and WAL over a network share is a data-loss hazard. |
| 7.6 | Broker the `lop secret` store | §4.8: the value must exist on the consuming host anyway, the retrieval consent model is process-ancestry-local (`secrets/peer.py:421`), and it would put arbitrary secret values on the peer link for nothing. |
| 7.7 | Let a borrower refresh "on behalf of" the owner by holding the grant | Explicitly forbidden by R14/R15. Only the owner POSTs. |
| 7.8 | Make `locality` redundant by keying everything on device id | `locality` is already declared once at the auth frame (`session/runtime/types.py:172`) and `mesh-network.md` §5.8 makes it an authorisation input. Re-deriving "is this remote" per call site is the exact bug that field was added to fix. |
| 7.9 | Put the credential ops in the session runtime's `ControlOp` | The runtime is the *client* on leg 1 and never dispatches these. Adding them to `ControlOp` (`mobile/types.py:279`) would make every session runtime claim a relay's capability. |
| 7.10 | Four separate peer ops (`credential_grant`, `_report`, `_placement`, `_repair`) | The transport authorises **per op** with a capability row (`mesh-transport-identity.md` §7.1), so four ops would mean four rows for one authority and four places to forget a check. One op (`net_broker`) with a `kind` discriminator, gated once on `broker_credential`. |
| 7.11 | Send the bearer back through the *session* stream (`net_forward`) | It would ride the `ControlOp` carrier as a session frame, which means the token enters a frame path that is logged and projected. The broker op exists precisely to keep credential material off the session plane. |

---

## 8. Open questions, each with my recommendation

**Q1 — Is "no non-owner process ever calls a provider refresh" actually
provable, or only testable?** *Recommendation:* treat it as testable and add
the assertion in §5.3 rather than attempting a static guarantee. *Evidence that
would settle it:* if review wants a structural proof, the cheapest is a
`FailoverAuthStore` wrapper that records the calling device id on every
`_resolve`, asserted against an allow-list in one test; I would not accept a
"grep for `refresh_token`" check, which cannot see a lazily-resolved callable
(`registry.py:226 _lazy_refresh`).

**Q2 — Does `get_oauth_access` surface `credential_id` for static (api_key)
rows?** *Recommendation:* assume yes (its non-OAuth branch builds an
`OAuthAccess` with `kind="api_key"`, `auth_store.py:1683-1740`) and, if
implementation finds a path where it does not, add an additive
`AuthStore.credential_ref_for(provider, session_id)` rather than reaching into
`_resolve`. *Evidence:* one REPL call against a fixture store that holds only a
static key.

**Q3 — `grant_ttl_s = 900 s`: too long, or too short?** *Recommendation:* ship
900 s, because the cost of a shorter TTL is a broker round trip on a link that
may be slow, and the cost of a longer one is revocation latency. *Evidence that
would settle a change:* the measured p95 `latency_ms` of a `credential_grant`
across the two real topologies in §9 — if it exceeds ~150 ms, shorten to 300 s
and accept the chatter; if it is under 20 ms, there is no reason to reduce it.

**Q4 — Should a device be allowed to **use** (not copy) a static key it does
not own, when the owner is a `pool`?** *Recommendation:* no — pool members
never own credentials (§6.1), so the case cannot arise; if a future capacity
provider offers its own model access, that arrives as a *provider definition*,
not as a credential in this document's sense. Note it so the next reader does
not add a special case.

**Q5 — Do we need `credential_placement` broadcast on every change, or only on
handshake?** *Recommendation:* broadcast on change (the op already exists in
`CredentialOp`), because `holders` is an authorisation input and must be fresh;
§4.6's 60 s refusal cache is what keeps a stale local view from costing
anything. *Evidence:* the audit `credential.placement_declared` rate on a
two-week realistic run — if it is more than a handful per day, the broadcast is
cheap; if a bug floods it, the 60 s cache bounds the damage.

**Q6 — Does the owner's `identity_label` belong in the synced document at
all?** *Recommendation:* yes, but it is the single most sensitive non-material
field there, so it is (a) omitted for pool holders in the *document* too, (b)
shown masked in the TUI, and (c) never written to the audit log. If review
judges even that too much, drop it: the only cost is that the pairing screen
and the credential view say "openai (OAuth)" instead of naming the account,
which is a real usability loss for
"[which of my two openai logins is this session on](§4.9)".

---

## 9. Test plan — the exact commands a QA agent runs

Isolation is per AGENTS.md §Isolating a run: `env -i` (strips `CMUX_*`, which
can rename the operator's real cmux workspaces, and `LOP_*`, which changes what
a child runtime thinks it is), a fresh `HOME` per cell, and never the operator's
live config. Every block makes its own `ISO`.

### 9.1 Unit (`tests/unit/network/`)

```sh
cd ~/local-operator-worktrees/mesh-network
ISO=$(mktemp -d)
env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \
  TERM=xterm-256color .venv/bin/python -m pytest tests/unit/network -q
```

Named tests the design requires:

- `test_no_network_keeps_the_plain_store.py::test_zero_peer_topology_uses_a_bare_AuthStore` (§5.1)
- `test_a_local_login_never_dials_the_broker` (§5.4)
- `test_no_keychain_from_mesh.py`, `test_no_mcp_callback_bind_from_mesh.py`,
  `test_mesh_never_opens_secrets.py` (§5.4)
- `test_placement_merge.py::test_only_the_owner_writes_its_own_entry` (§2.1)
- `test_state_never_syncs.py::test_observations_do_not_enter_the_placement_doc` (§2.2)
- `test_binding_row.py::test_newest_binding_per_provider_wins` (§2.4)
- `test_broker_frames.py` — one test per `code` in §3.2's table, asserting the
  requester's cache TTL and retry behaviour from the code, not the message
- `test_coalescing.py::test_two_requests_inside_the_join_window_share_one_refresh` (§3.4)
- `test_forced_refresh_requires_admin` (§3.4)

### 9.2 Two devices, one host (the cheap hermetic harness)

The incident/credential machinery is keyed on **device id**, not host identity,
so two config dirs on one host model two devices exactly. Run this first: it is
hermetic, fast, and does not consume a cloud peer.

```sh
cd ~/local-operator-worktrees/mesh-network
ISO=${TMPDIR:-/tmp}/mesh-cred-$$; mkdir -p "$ISO/A" "$ISO/B"
runA() { env -i HOME="$ISO/A" LOCAL_OPERATOR_CONFIG_DIR="$ISO/A/.local-operator" \
         PATH="$PATH" TERM=xterm-256color "$@"; }
runB() { env -i HOME="$ISO/B" LOCAL_OPERATOR_CONFIG_DIR="$ISO/B/.local-operator" \
         PATH="$PATH" TERM=xterm-256color "$@"; }

# A owns the credential; B borrows it.
runA .venv/bin/python -m local_operator.cli network init credlab --json
TOKEN=$(runA .venv/bin/python -m local_operator.cli network invite --role drive --json | jq -r .token)
runB .venv/bin/python -m local_operator.cli network join "$TOKEN" --confirm-sas <SAS> --json
runA .venv/bin/python -m local_operator.cli network credentials --json      # expect: openai owner=dev_A holders=[dev_A,dev_B]
runB .venv/bin/python -m local_operator.cli network credentials --json      # expect: openai borrowed-from dev_A

# The load-bearing evidence: a session ON B, whose provider login lives only on A.
runB .venv/bin/python -m local_operator.cli login openai --list             # B holds NO openai login
runA .venv/bin/python -m local_operator.cli exec --peer <dev_B> --model openai/gpt-5.4 "say hi"
```

Assertions after that turn:

| Side | Assertion | How |
|---|---|---|
| B | Turn succeeded; no `openai` row in B's `auth.db` | `sqlite3 "$ISO/B/.local-operator/auth.db" "select provider,credential_type from auth_credentials"` → no `openai`; and a `.dump` diff against a pre-run dump is empty (§5.2) |
| B | The binding row exists in B's transcript | `grep -c mesh_credential_binding.v1 "$ISO/B/.local-operator/sessions/<id>/transcript.jsonl"` → 1, citing `owner_device=dev_A` |
| A | Exactly one refresh POST, lease released, no row disabled/added | stub IdP's counter; `select count(*) from auth_credential_refresh_leases` → 0; `sqlite3 ... "select distinct disabled_cause from auth_credentials"` → `NULL` |
| A | One `credential.grant` audit record naming `for_session` | `runA ... network log --json \| jq 'select(.event=="credential.grant")'` |

### 9.3 The refusal and failure rows

For each of 4.1–4.9, the same topology, one command, and the exact sentence
asserted on the requester's surface:

```sh
# 4.1 owner offline
runA ... network serve stop
runA ... exec --peer <dev_B> --model openai/gpt-5.4 "hi"       # assert the 4.1 sentence, exit != 0

# 4.2 revoked
runA ... network credential revoke openai --from <dev_B>
runB ... exec --peer <dev_A> --model openai/gpt-5.4 "hi"       # assert the 4.2 sentence
runA ... network log --json | jq 'select(.event=="credential.grant_refused")'

# 4.5 two askers at once
runA ... exec --peer <dev_B> --model openai/gpt-5.4 "hi" & \
runA ... exec --peer <dev_C> --model openai/gpt-5.4 "hi" & wait
# assert stub POSTs == 1 and both transcripts have a complete reply

# 4.7 MCP repair
runB ... mcp login datadog                                     # assert REMOTE_GRANT_NOTICE, refusal
runA ... network credential repair datadog --on <dev_A>        # owner gets the durable notice
# owner runs /mcp login datadog; then the remote session's next attempt succeeds
```

The IdP/POST counter is not optional: **"one POST" is the requirement**, and
without a counter the test asserts nothing about $3.4. The stub must therefore
be a real loopback HTTP server that logs each grant request, not a monkeypatched
refresh function.

### 9.4 Two peers on two hosts (confirmation)

Per `mesh-network.md` §10's two-peer row: the same matrix against the EC2 peer
provisioned with the `minerva_nprod` profile. This run exists to catch what one
host cannot — real latency (feeding §8 Q3), clock skew, and a link that drops
mid-grant. Evidence: the commands, their actual output, and on each side the
`auth.db` dumps, the audit tail, and the transcript rows.

---

## 10. File-by-file change list (for the coder)

| File | Change |
|---|---|
| `local_operator/network/credentials/__init__.py` | new package |
| `local_operator/network/credentials/types.py` | `CredentialOp`, `Grant`, `BrokerError`, `CredentialPlacementEntry`, `Holder`, `CredentialBinding` |
| `local_operator/network/credentials/placement.py` | placement document read/merge/write/persist; `PlacementDocument`; the owner-only write rule |
| `local_operator/network/credentials/state.py` | `placement.state.json` read/write; the refusal cache with TTLs |
| `local_operator/network/credentials/store.py` | `MeshAwareAuthStore` + `build_auth_store()` (§3.6a, §5.1) |
| `local_operator/network/credentials/client.py` | `MeshCredentialClient` — leg-1 dial, leg-2 forward, `report`, `repair` (§3.6b) |
| `local_operator/network/credentials/owner.py` | `MeshCredentialBroker` — authorise, coalesce, resolve, audit, reply (§3.6c) |
| `local_operator/network/credentials/messages.py` | `render_broker_error()` — the one home for §4's sentences |
| `local_operator/session/spend.py` | `serving_identity` gains `credential_owner` / `credential_ref` (§6.3) |
| `local_operator/session/session.py` | new custom type `mesh_credential_binding.v1`, added to `_PERSISTABLE_CUSTOM_TYPES` (`:772`) |
| `local_operator/session_factory.py` | `:3398` → `build_auth_store(credential_manager)` |
| `local_operator/mcp/auth.py` | `ensure_mcp_oauth_fresh` (`:3996`) mesh branch before the lock; no write on the borrower |
| `local_operator/mcp/grants.py` | nothing changes for the verb (`REMOTE_GRANT_NOTICE` stays); the repair notice is raised by the relay, not here |
| `local_operator/network/types.py` (transport doc's file) | the `run/peers` record must carry `control_port`, `control_key`, `protocol`, `capabilities`, `install_root` (§3.1) |
| `local_operator/network/cli.py` | `network credentials`, `credential share/revoke/own/replicate/repair` subcommands, all with `--json` |
| `local_operator/slash_commands.py` | `/network credentials …` sub-view (surface owned by `mesh-ui.md`) |
| `local_operator/providers/auth_store.py` | **no change**, unless §8 Q2 comes back "no" (then one additive accessor) |
| `tests/unit/network/*` | §9.1 |
| `tests/e2e/test_mesh_credential_rotation.py` | §5.3 |

Nothing in this list writes to `credentials.env`, `secrets/`, `auth.db`'s schema,
or the keychain.

---

## 13b. Convergence round 1 — what changed here

1. **The pattern is named** (§1.4): OAuth 2.0 Token Exchange (RFC 8693) as a
   *delegation*, on-behalf-of as the flow's name, *credential broker* in SPIFFE's
   sense, and RFC 9700 / BCP 240 for "one holder, one use" — explicitly **not**
   "OAuth 2.1", which is still a draft. The naming brought three rules with it and
   all three are now stated: the `act`/`sub` audit markers, grant narrowing, and
   the RFC 9700 citation.
2. **The delegation markers land in the audit schema** — described here, fielded
   in `mesh-incident-response.md` §4.3, which owns the log. `act` is the broker
   device, `sub` is the grant's recipient; both are device ids.
3. **Grant narrowing is an explicit invariant of §3.3** rather than an implied
   consequence of the cascade. A requester is served only what the owner holds,
   the returned expiry is the earlier bound, and the refusal path is named.
4. **No change to the ownership model, the broker protocol or the non-regression
   matrix.** The token never crosses a boundary except as a short-lived scoped
   bearer, which is what SPIFFE's broker model and RFC 8693's delegation both
   describe; this round gave that design its vocabulary and its two missing
   invariants rather than altering it.
5. **The device-bound invite (§8 Q3) and the transport's rotation-withholding
   (§2.3.4)** are implemented in `mesh-transport-identity.md`, not here. They do
   not touch credential handling: the holder-set check is independent of the
   transport secret (§3.6c), which is why a removed device's borrowed tokens stop
   working on their own `grant_ttl_s` bound even before the rotation.
