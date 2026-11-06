# Design: delegated offline refresh — a node refreshes a login its owner granted while the owner is away

Status: **design, pre-implementation. Nothing here is built.** This is the gate
before code, and it is C5-scoped: no slice of it may land before a security
reviewer has signed the four surfaces named in §3.9.

Read against `origin/main` @ `98f447d1b88` (2026-11-05). Every `file:line` below
was read at that revision. Symbol names are the stable citation; numbers drift.

Supersedes/extends nothing: this document **contradicts** three settled positions
in the mesh design set and says so in §2, quoting them rather than writing a
parallel scheme beside them.

Served model for this design pass: `radient/auto` (declared fallback; the pinned
`anthropic/claude-opus-5-5` was quota-unavailable). Provenance is disclosed per
team policy; it does not soften any claim below.

---

## 0. The answer in one page

**What the operator asked for.** A remote node keeps running and keeps its
provider logins alive while the owner's laptop is closed — no pause, no banner.
That settles the *shape*. It does not settle the *mechanism*, and the mechanism is
a real security decision, so it is stated here in one sentence that a reviewer can
reject:

> **For the duration of an owner-signed, single-holder, time-bounded delegation,
> the rotating refresh material lives on the node instead of on the owner. What
> does not move is control: who may spend it (the owner's signature), when it
> stops (revocation, expiry, or the node's own first doubt), and where it goes
> back (an ordered handback in which the token moves only after the sender has
> already stopped).**

There is no way to have the second half without the first. Refreshing a provider
login means POSTing the refresh token; a device that POSTs a token must possess
it. Every alternative that keeps the material on the owner when the owner is
unreachable is, on inspection, either "ask the unreachable owner" or "keep a
bearer that never expires" — and both are already rejected in the docs (§2).

So the design is a **delegation**, not a brokering extension: today's broker lends
a *bearer* for minutes (§3.3 of `mesh-credentials.md`); this lends *the refresh
authority itself* for a bounded window, to exactly one named node, under a
signature, with a fence that makes "two refreshers" structurally impossible
rather than merely unlikely.

**What it costs, stated up front.** Moving the family to a node means the owner
cannot safely POST that login again until the family comes back. If the node is
destroyed mid-delegation, the recovery is an interactive re-login, not an
automatic one. That cost is deliberate: the alternative is presenting a token
that may already be spent, which is the reuse-detection POST that revokes the
whole family — the failure this repository has already measured twice (§1.1).

**What it deliberately is not.** It is not a background refresher, not a new
peer op, not a new capability row, and not a widening of any existing signed
payload (§3.8). It is not MCP OAuth in the first slice, and it is not a claim
that a compromised node can be stopped — that limit is named in §7.

---

## 1. The seam today, exactly

### 1.1 One module owns refresh for a locally-held login

`local_operator/providers/auth_store.py` owns it, end to end:

| Fact | Where |
|---|---|
| The refresh decision and the POST | `_ensure_oauth_fresh` (`auth_store.py:3459`) |
| "does this token need refreshing" | `_needs_refresh` (`auth_store.py:3448`) against `OAUTH_REFRESH_SKEW_MS = 60_000` (`auth_store.py:105`) |
| The provider's own refresh function | `_refresh_fn` → `providers/registry.py`'s `refresh_token=` entries (`registry.py:369` `_lazy_refresh`; concrete rows at `:473`, `:490`, `:508`, `:565`, `:622`, `:817`), implemented in `providers/oauth/*.py` |
| Per-process single-flight | `_refresh_lock_for` (`auth_store.py:2991`) |
| Cross-process serialisation | `auth_credential_refresh_leases` (`auth_store.py:377`), `_try_refresh_lease` (`auth_store.py:2998`), released in a `finally` (`auth_store.py:3773`/`:3030`) |
| Wall-clock cap on one exchange | `PROVIDER_REFRESH_TOTAL_BUDGET_S = 60.0` (`auth_store.py:137`), applied with `asyncio.timeout` (`auth_store.py:3607`) |
| Lease length | `AUTH_REFRESH_LEASE_MS = 90_000` (`auth_store.py:162`), derived from the budget above |
| Write-ahead "a token may already be spent" marker | `REFRESH_SEND_UNCONFIRMED_KEY` (`auth_store.py:191`), `UNCONFIRMED_SEND_TTL_S` (`auth_store.py:274`), armed at `auth_store.py:3596` |
| Permanent-refusal tombstone | `GRANT_DEAD_AT_KEY` (`auth_store.py:220`), `_mark_grant_dead` (`auth_store.py:3365`) |
| Public entry points | `get_api_key` (`auth_store.py:4900`) → `_resolve` (`auth_store.py:5306`); `ensure_oauth_fresh` (`auth_store.py:3790`) for a named row |

The concurrency primitive that matters for this design is **the presented token
as a generation marker**:

```python
# auth_store.py:3058
def _presented_refresh_token(creds) -> str | None: ...   # the token that WOULD be sent
```

Every "did someone else move the family" test in the store is a comparison of
that value (`auth_store.py:3648` in the `invalid_grant` arm, `:3739` before conveying
its own rotation into the row). It works because a rotating refresh token
*is* the family's state: whoever holds a different token is ahead. That observation
is what §3.3 generalises across hosts, but as a **counter**, never a token —
a token must not cross a link except to its single next holder.

The store's own docstring (`auth_store.py:1-59`) records the two measurements the
whole mechanism exists for: two processes racing one rotating token invalidated
each other (**PR-24**), and a Radient grant died twice in ~17 hours from
refresh-token **reuse detection**. Every constraint below is downstream of that.

### 1.2 MCP is a second, differently-locked refresher

`local_operator/mcp/auth.py:5217 ensure_mcp_oauth_fresh` serialises on a **lock
file beside `auth.db`** (`_oauth_refresh_lock`, `mcp/auth.py:4089`, path at
`:4054`), not on a SQLite row. Same class of hazard, different mechanism. This
design does not touch it in the first slice (§7).

### 1.2a Why the lease cannot simply be made cross-host

`mesh-credentials.md` §3.4 states the argument and this document does not weaken
it: the lease is "a row in the *owner's* `auth_credential_refresh_leases`" and
"**`AUTH_REFRESH_LEASE_MS` never needs to span hosts**" because "the only process
that contends for it is on the owner". A lease that spanned hosts would need a
distributed mutex over a rotating secret, which is exactly the thing RFC 9700
rotation is designed to make unnecessary — and §7.5 of that document already
rejects SQLite-over-network-filesystem for good reason.

**Therefore this design does not widen the lease.** It removes the contention
instead: while a delegation is in force, exactly one device is *authorised* to
take its local lease and POST. Exclusivity is a state, not a race.

### 1.3 Why a peer-rooted login cannot refresh at all today

Three independent reasons, all structural:

1. **No material.** A borrowed bearer is a `Grant` (`network/credentials/types.py:697`)
   whose only secret field is `access_token`; `to_detail` (`types.py:739`) names
   its fields explicitly and there is no refresh field. `mesh-credentials.md` §3.3
   is categorical: "**Never returns, under any circumstance:** the refresh token".
2. **No row.** `MeshAwareAuthStore.get_api_key` (`credentials/store.py:373`) and
   `get_oauth_access` (`:429`) consult the local store first and then `_borrow`
   (`:474`); nothing in the mesh path calls `upsert_credential`, and
   `GrantCache` deliberately has no serializer
   (`credentials/client.py:103-125`: "there is no `to_json`, no path and no
   pickle hook").
3. **No authority.** When the owner is unreachable, `_held_or`
   (`credentials/client.py:919`) may serve the grant this process already holds,
   bounded by the owner's original `grant_expires_at_ms` — the change #3439
   landed. Past that, the answer is a refusal, and the owner's own broker is the
   only thing that refreshes.

So "the node refreshes" is not a switch to flip; it needs a row, material, and an
authority record. That is the shape of the whole change, and it is why the
smallest honest version reuses the *existing* per-device store rather than
inventing a second one (§3.7).

### 1.4 Deny-wins already exists, as a recipient-local watermark

`network/credentials/held_policy.py` is a recipient-local, **deny-only**
document: per-`(owner, key)` generation plus a network generation and a
membership flag (`Stamp` at `held_policy.py:46`; schema validated by `_validate` at `:107-152`;
`transaction()` at `:159`; `deny` at `:256`; `deny_network` at `:264`). Three
properties are already built and reviewed:

- **Anything not explicitly non-restrictive is a deny** (`NONRESTRICTIVE`,
  `held_policy.py:27-39`) — so a new refusal code needs no schema change.
- **Unprovable evidence is never permission** (`PolicyUnknown`, `held_policy.py:41`;
  raised for a missing/corrupt/uninspectable file).
- **It survives a connectivity win**: `_held_or` (`client.py:919`) commits the
  deny before publishing an answer, drops the grant cache, and the fence is
  compared as an opaque generation (`client.py:959`, `GrantCache.invalidate_policy`
  at `client.py:164`), never as a clock.

That is the machinery constraint 3 needs. This design does not add a second deny
store; it makes the *refresher* consult the one that exists (§3.4).

### 1.5 A signed-authority substrate already exists, and it is root-anchored

- **The anchor.** `operator/trust.py` pins the operator's *public* key in a
  root/admin-owned file whose path "must not be redirectable by anything the uid
  can write" (`trust.py:1-40`). Verification on a machine that can read that file
  needs no secret.
- **Domain separation.** `operator/verify.py` owns one tag per signed protocol:
  `DOMAIN` (`:42`), `DEVICE_CERT_DOMAIN` (`:47`), `APPROVAL_DOMAIN` (`:57`),
  `RELEASE_DOMAIN` (`:67`) — "so a signature from any other protocol can never be
  replayed here".
- **A signed, time-bounded statement about a device** already exists:
  `operator/devices.py:14-18` — a certificate "over `(device_id, spki, label,
  issued_at, not_after)`" verified against the anchor by
  `verify.verify_device_cert`, with the substitution argument stated: "THE
  SUBSTITUTION IS DEFEATED BY THE SIGNATURE, NOT BY THE PATH".
- **The signed-gate discipline** is `approvals.py` F4 (`approvals.py:44-50`): the
  payload is domain-separated and length-prefixed (`signed_payload`,
  `approvals.py:893-918`), and `verify_for_run` re-derives and re-verifies
  "before every step of an execution".

The new artifact in §3.7 is modelled on the **device certificate** (a multi-use,
time-bounded capability), not on an approval record (a single-use transition) —
and §3.8 says exactly why that choice also keeps the approval payload untouchable.

---

## 2. Where the docs already specify this — and what this document changes

This proposal does not get to be silent about contradicting settled text. Four
places:

**(a) `mesh-credentials.md` §0, property 1.** "**The refresh never moves.**
`_ensure_oauth_fresh` … run on the owner. This is why the PR-24 class of bug …
*cannot* occur across the mesh: there is exactly one process on exactly one host
that ever POSTs."
*This document moves the refresh, for a bounded window and one named device. It
answers the PR-24 argument rather than dismissing it: there is still exactly one
device authorised to POST at any instant (§3.2), and the window is closed by
revocation, expiry, or the node's first doubt (§3.4, §3.5). The property becomes
"one authorised refresher", not "one host".*

**(b) `mesh-credentials.md` §7.7.** "Let a borrower refresh 'on behalf of' the
owner by holding the grant — Explicitly forbidden by R14/R15. Only the owner
POSTs."
*R15 ("refresh on behalf of a peer") is re-read here as R15′: the owner may
delegate its refresh authority to one node, explicitly, revocably, and
auditably — what R14 forbids (a peer spending a bearer off the owner's device
without a grant) still holds. If the reviewer reads R15 as forbidding the
delegation even under a signed grant, the answer is "then the requirement cannot
be met", and §0 states why.*

**(c) `mesh-credentials.md` §5.3 ("Credential rotation stays single-host") and
§5.4 ("the fragile stores are provably untouched", incl. "`secrets/` (class 2) …
asserts `local_operator/network/` imports nothing from `local_operator.secrets`").**
*§5.3's e2e assertion — exactly one POST, neither device's token invalidated — is
kept and strengthened (§5, rows T3/T4/T9). §5.4's literal claim no longer
describes the tree: `network/credentials/github.py:376-378` and
`offers.py:349-351` import `local_operator.secrets` for the copy classes, so "the
mesh does not touch the secret store" is already scoped to *brokering the
operator's store*, which this design still does not do (§3.6).*

**(d) `mesh-credential-lifecycle.md` §13 ("Authority, scope and decision") and
§"Residual limits".** It "Reject[s] native OAuth replication (second refresh
owner)" and states "**Owner outage is a distinct unavailable state, not permission
to refresh remotely**"; the storage section states "**Borrowed OAuth remains
memory-only**".
*This document proposes precisely the rejected option, under three conditions the
rejection did not consider: (i) exclusivity by construction, not replication
(there is never a second live holder); (ii) an operator-signed, single-node,
time-bounded grant rather than replication; (iii) an ordered handback that
returns the family to the owner. "Owner outage is not permission" is preserved —
an outage still grants nothing; only a signature does. "Borrowed OAuth remains
memory-only" is preserved for *borrowed* material and contradicted only for
*delegated* material, which is a different object with a different lifetime
(§3.6).*

**Where the docs already specify the machinery, this document reuses it and says
so**: the deny watermark (§1.4), the `announce`/`copy`/`ack` pull pattern and
owner-side generations (`credentials/sync.py:1-40`, `MESH_ORIGIN_KEY` at
`sync.py:212`), `mesh_origin` provenance on copied rows
(`session/credential_binding.py:386-400`), the sealed-reference shape
(`providers/meta_muse_credentials.py:25 seal_credential`), the audit allowlist
discipline (`network/audit.py:505`, `FORBIDDEN_DETAIL_KEYS` at `:692`, which
already lists `refresh_token` and `access_token`), and the one-op-many-kinds wire
rule (`credentials/types.py:73-86`; `mesh-credentials.md` §7.10).

---

## 3. The design

### 3.1 Vocabulary (fixed here so later slices cannot drift)

| Term | Means |
|---|---|
| **owner** | the device holding the login (its `auth_credentials` row), named as `owner_device` in the placement entry (`placement.py:588-660`) |
| **node** | a paired member device that will refresh. Not a `pool` member (`mesh-credentials.md` §6.1: a pool member "declares no credentials of its own, ever") |
| **key** | a placement key: `<provider>` or `mcp:<url>` (`types.py:110-118`) |
| **family** | one provider login's refresh-token lineage. Its state is "which token is current"; its *identity* across refreshes is `(provider, account, login_incarnation)` (`mesh-credential-lifecycle.md` L1) |
| **family_epoch** | monotonic integer, incremented on every successful refresh of the family, recorded on whichever device performed it. This is the cross-host replacement for "compare the tokens" (§1.1) |
| **delegation** | an operator-signed statement naming `(network, owner, node, key, grant_id, issued_at, not_after, epoch_at_issue, login_incarnation, max_refreshes)` |
| **handback** | the ordered return of the family to the owner (§3.3) |

### 3.2 Fence 1 — exclusivity by construction

When the owner mints a delegation it writes, in its **own** row's `data`:

```jsonc
"refresh_delegate": {
  "device": "d_4b2a…", "grant_id": "g_…", "epoch": 7,
  "issued_at": 1793000000.0, "not_after": 1789.0, "state": "active"
}
```

and `_ensure_oauth_fresh` (`auth_store.py:3459`) consults it **immediately after
the tombstone check and before `_needs_refresh`**, i.e. before any path that can
reach the POST:

```python
if self._delegated(data, now_ms=self._now_ms()) is not None:
    raise RefreshDelegatedError(...)   # new, beside RefreshUnconfirmedError
```

Consequences, each of which is a property rather than an intention:

- **A serving path is unaffected**: a live access token is still returned
  (`_holds_live_bearer`, `auth_store.py:3071`), and `_served_while_contended`
  (`auth_store.py:3411`) still serves an honest row. Only *presenting the refresh
  token* is refused. This mirrors the existing marker's rule ("it only decides
  against presenting one — never against serving a bearer that still works",
  `auth_store.py:3565`).
- **Every process on the owner obeys**, because the record is in the row, not in
  a process (the same argument as F2 in `owner.py:36-40`: the sharing list is read
  from disk on every request because the CLI is another process).
- **A crashed delegation cannot wedge the owner forever**: `not_after` is the
  ceiling, and after it the record's `state` folds to `expired` (a pure fold, the
  asks-store discipline `approvals.py:56-59` uses) — but see §4.3: an expired
  delegation does **not** re-authorise a POST by itself.
- **At most one live delegation per `(owner, key)`**, enforced by the mint
  refusing when a live record exists. Two nodes cannot both be named.

### 3.3 Fence 2 — the epoch handoff, and the last-frame rule

The family's state must move as one indivisible thing or not at all. Two ordering
rules do that, and nothing else needs to:

**Rule A — the token moves in the last frame, after the sender has already
stopped.** Handback is three messages, and each is safe to lose:

1. node → owner `refresh_return {grant_id, epoch}` — an intent, no material.
2. owner → node `refresh_return_ack {grant_id, epoch}`; the owner writes
   `refresh_delegate.state = "recalled"` and **still does not POST**.
3. node → owner `refresh_return_done {grant_id, epoch, refresh, access, expires}`,
   sent **only after** the node has wiped its own copy (or atomically with the
   wipe: wipe-then-send, never send-then-wipe).

Owner adopts the material only on step 3 and only then clears the record.
*Lose 3 after the node wiped*: the owner holds `recalled` and no token → it must
require an interactive sign-in (§4.3). Safe by construction; costly by design.
*Lose 3 in flight*: the node still holds, so nothing is lost; retry. *Lose 2*: the
node keeps refreshing and the owner still refuses — the correct state.

**Rule B — epoch comparison is total, and a lower epoch never overwrites.** The
owner accepts a handback only if `msg.epoch > refresh_delegate.epoch_at_issue`
and `msg.login_incarnation == refresh_delegate.login_incarnation`. A re-login on
the owner mints a new `login_incarnation` (the additive column
`mesh-credential-lifecycle.md` L1 describes: "a native provider upsert stamps a
random incarnation even when the same identity reuses its row id"), so a handback
from a superseded family is refused by a value the owner already has, not by a
clock.

**The epoch is written before the POST is trusted.** On the node, the sequence
inside one refresh is: take the local lease (`_try_refresh_lease`, unchanged) →
re-check the delegation and the deny watermarks (§3.4) → arm the send marker →
POST → persist `{token, epoch+1}` **atomically** (staged write + `fsync` +
`os.replace`, `store._write_private_json`'s pattern) → release the lease → return
the bearer. A crash between POST and persist leaves the marker armed, which is
exactly what the marker is for (`auth_store.py:191-218`) — the next attempt
defers instead of re-presenting. **No new crash-recovery mechanism is invented;
the node runs the owner's own.**

### 3.4 Fence 3 — deny wins, including offline

Before **every** POST — not merely at grant time — the node evaluates, in this
order, and refuses at the first failure:

1. `self._self_membership_active()` (existing, `client.py:919-940`).
2. the signed delegation: verifies against the operator anchor, names this device
   and this key, `now < not_after`, `login_incarnation` unchanged,
   `refreshes_used < max_refreshes`.
2b. the network generation: a `deny_network` stamp (`held_policy.py:264`) refuses.
3. the per-key deny watermark: `HeldPolicyStore.transaction()` →
   `policy.code(owner, key)` (`held_policy.py:243`); any code that is not in
   `NONRESTRICTIVE` refuses. `PolicyUnknown` refuses (fail closed, §1.4).
4. the local refresh lease (`_try_refresh_lease`, unchanged).

**How the node learns of a revocation without ever being reachable by the
owner.** Three channels, in increasing latency:

- **Channel 1 — the deny relay (new, §3.7).** Revocation mints a second
  operator-signed statement of the same shape with `not_after = now` (or a
  `revoked_at` field). Because the *signature* is the authority and not the
  sender, any member may relay it: it is accepted wherever `placement.merge`
  (`placement.py:692`) currently refuses third-party relay, and it can only ever
  *reduce*. That is the one place this design touches an established wire rule,
  and it is named as such.
- **Channel 2 — the next contact with the owner**, in either direction: the
  existing placement revoke (`placement.py:660`) plus the record transition to
  `recalled`, and the refusal code the broker already returns (`revoked` /
  `not_a_holder`, `types.py:457-465`).
- **Channel 3 — expiry.** At `not_after` the grant dies with no message at all.
  This is the backstop that makes the others best-effort.

**What the interval means, plainly.** Between the operator's revoke and the node
learning, the node may keep refreshing, and the bearer it already holds keeps
working until the provider expires it — a bearer cannot be recalled. The window
is bounded by `not_after` (recommend ≤ 72 h, configurable) and made small in
practice by Channel 1. That residual is stated rather than papered over, and it
is the same *class* of bound the docs already accept for a lent bearer
("Withdrawal stops future authorized borrows but cannot retract a leaked
OAuth/static bearer instantly", `mesh-credential-lifecycle.md` §"Residual
limits").

**Fail closed means three things, and it must mean all three**: no POST on
unknown evidence; no *retry* of a POST whose outcome is unknown (the marker
governs); and no widening of what is served while the delegation is unprovable.

### 3.5 Fence 4 — "it cannot tell" resolves to stop, never to guess

The external-tool case (Claude Code / Codex holding the same login) is
**unobservable from inside lop**: another process refreshing leaves our copy
stale but alive, and our first evidence is the provider's `invalid_grant`. The
design therefore does two things and claims no more:

- **Mint-time gate.** A delegation may be minted only for a login the operator's
  card asserts is *exclusively held* (`exclusive_holder: true` — a declaration,
  not a detection). Unknown → refuse to mint, with copy that names the reason.
  This is the only reliable answer; refusing to mint is cheap, and the alternative
  is a coin-flip on a family the operator may be using elsewhere.
- **Use-time gate.** If a POST earns `invalid_grant` while the token presented was
  *this node's own current token* (i.e. no sibling on the node moved it — the
  race guard at `auth_store.py:3648` does not apply), then the family moved
  outside this device. The node:
  1. tombstones the grant (`_mark_grant_dead`, unchanged) so nothing re-presents it;
  2. wipes the delegated material and marks the delegation `failed: family_moved`;
  3. reports to the owner (`kind: report`, failure `grant_dead`) so the owner does
     not serve or refresh a dead family and its surfaces say "sign in again";
  4. **does not retry** — not on a schedule, not on a later turn. Only a fresh
     signed delegation at a higher epoch re-enables it.
  A node that cannot tell does not get to try again.

### 3.6 Where the refresh material lives, what each party can read, what is logged

| Item | Owner | Node | Never |
|---|---|---|---|
| Rotating refresh token | its own `auth.db` row (unchanged), **frozen while delegated** | a row in **its own** `auth.db`, provenance-marked, from mint to wipe | on the wire except inside the encrypted link (`network/wire.py:1-10`: AES-256-GCM records after the handshake), and only in the `refresh_take` answer and the handback's last frame |
| Access token | unchanged | its own row, exactly as any local login | — |
| The signed delegation | kept `requested`-style in the owner's own store | a copy in `credentials/<network_id>/` (already deny-listed for confined shells, `tools/confinement.py:787-795`) | — |
| The operator private key | anchor-backed signer, never leaves `operator/` | **not present, not needed** | — |
| Deny watermarks | reads them | reads them; denies only | — |

**What the node can read**: its own config root, for this one key. **What it
cannot read**: any other key's material, any other member's material, the
owner's `auth.db`, the owner's placement/consent documents, the operator private
key. The graph is unchanged from today except for one new edge: this key, on this
node, for this window.

**Logs: nothing.** Two structural reasons, not a promise: the audit writer drops
any detail key outside an event's allowlist (`network/audit.py:15-18`) **and**
drops the never-list even if a future allowlist names one (`:692-717`, which
already contains `refresh_token`, `access_token`, `token`). The new events
(§3.7f) allowlist only `credential_key`, `act`, `sub`, `grant_id`, `epoch`,
`cause`, `approval_id`. Not verified in this pass, and must be before landing:
that no new code path writes the material through `logger.*` — the existing
refresh path logs ids and provider names only, and the node's path must be
reviewed line by line for the same property.

**The honest at-rest claim, and its limit.** On the node the material sits in
`auth.db` at 0600, which is *not* a vault: the secrets store's own docstring says
its default keyfile tier is "equivalent to today's plaintext `.env` against an
attacker who specifically reads the key file" (`secrets/keys.py:1-20`). Wrapping
the token under a key that lives on the same host would move the bar from "read
one file" to "read two files" and would be worth exactly that much — so this
design does **not** claim sealing as its protection, and recommends the
storage-migration slice (`mesh-credential-lifecycle.md` §"Storage") for it. Its
protection is: bounded window, single holder, one key, wipe on handback/failure,
and confinement.

**One prerequisite this exposes, measured not assumed.** `auth.db` is **not** in
`_CONFIG_SECRET_SUBPATHS` (`tools/confinement.py:787-795`), and `*.db` is
deliberately excluded from the kernel deny (`:810-812`), so a write-confined
agent shell can read `auth.db` today. That is pre-existing for the operator's own
logins, but the delegation would *add* a rotating secret to it. Slice 1 therefore
either (a) adds the exact path `auth.db` to `_CONFIG_SECRET_SUBPATHS` (exact
paths are how that list works), or (b) keeps the material under
`network/credentials/<net>/`, which is already denied — and (b) needs the loader
seam §3.7e describes. Recommendation: **(a) plus the row**, because it keeps one
refresh path, and it closes a residual that exists regardless of this feature.
Whichever is chosen, prove it by attempting the read from a confined shell before
and after; do not assert it from the list.

### 3.7 Wire and state additions (complete list)

**(a) One new broker `kind` pair on the existing op.** `net_broker`
(`types.py:73`) with `BROKER_KINDS` (`types.py:86`) growing `"refresh_take"` and
`"refresh_return"`. No new peer op, no new `OP_CAPABILITY` row — §7.10's rule,
unchanged. The `refresh_take` answer carries the material **inside the encrypted
record**; `refresh_return` carries the three messages of §3.3 in `sub`.

**(b) One advertised capability string**, `credential-refresh-v1`, beside
`BROKER_CAP_STRING` (`types.py:82`), in hello/welcome `caps`. This is an
advertisement, not an authorisation: it exists so a node never sends a kind an
older owner turns into `internal` (`owner.py:547-550` — an unknown kind is
`internal`, which `client.py:77-80` classifies as *transient*; a node must gate on
the cap string, never parse that message).

**(c) Owner row state** (in `auth_credentials.data`, no schema change):
`refresh_delegate` (§3.2) and `REFRESH_FAMILY_MOVED_KEY` — a marker with the
tombstone's semantics but its own name, so surfaces can say "being refreshed by
<node>; sign in here to take it back" instead of "the provider refused".

**(d) Node state**: a delegated `auth_credentials` row with
`mesh_origin = {owner_device, key, grant_id, epoch}` (the provenance shape
`sync.py:212` / `session/credential_binding.py:386` already read), plus the signed
delegation document under `credentials/<network_id>/refresh.json` (0600, the
`_document_lock`/`flock` discipline of `placement.py:111-170`, size-capped like
`held_policy.LIMIT`).

**(e) The refresh seam**: nothing in `_ensure_oauth_fresh` changes except the
delegation gate (§3.2). The node's row is a normal row; the node runs the normal
path. If placement option (b) of §3.6 is chosen instead, add one loader that
materialises the token into the in-flight `creds` dict and a matching write-back
— and that seal is the only place a second copy of the store's discipline could
appear, which is why (a) is recommended.

**(f) Audit events** (all allowlists non-material): `credential.refresh_delegated`,
`credential.refresh_taken`, `credential.refresh_returned`,
`credential.refresh_refused`, `credential.refresh_revoked`,
`credential.refresh_family_moved`. Every one carries `act` (the device that acted)
and `sub` (the other device), matching `credential.grant`'s existing shape
(`audit.py:505-523`).

**(g) The signed artifact — a new domain, not a widened one.**
`operator/verify.py` gains `REFRESH_GRANT_DOMAIN = b"lop-refresh-grant-v1\x00"`
beside the four tags at `:42/47/57/67`, a builder beside `signed_payload`
(`approvals.py:893`) using the same length-prefixed discipline, and
`verify_refresh_grant` beside `verify_device_cert`. Fields, all bound:

```
network_id | owner_device | owner_spki_fp | node_device | node_spki_fp |
key | provider | credential_kind | login_incarnation | family_epoch_at_issue |
grant_id | issued_at | not_after | max_refreshes | exclusive_holder | relay_deny
```

Binding `node_spki_fp` (not just `node_device`) means a node whose identity key
was rotated must be re-granted — the device-certificate lesson
(`operator/devices.py:14-18`). Binding `login_incarnation` means a re-login on the
owner invalidates the grant by a value, not by a clock.

### 3.8 Does any of this widen an existing signed payload? No — and here is the check

**Checked, not assumed.** `approvals.signed_payload` (`approvals.py:893-918`)
composes exactly `domain || _lp(kind) || _lp(request_id) || _lp(request_digest) ||
_lp(decision) || _lp(decided_at_text)`. The message **shape is kind-independent**,
and the domain tag stays `APPROVAL_DOMAIN` (`operator/verify.py:57`). So:

- Adding a *kind* to `KINDS` (`approvals.py:96-101`) would change no existing
  message byte and no existing signature's validity; it would only let an
  operator signature convey one new class of decision, and `_SUBJECT_BLOCKS`
  (`:103-110`) would gain a row.
- This design does not even do that: the delegation is its **own** domain and its
  own builder, so no approval record, digest, tag or verification changes at all.
  An older build reading a new artifact refuses it (an unknown domain is not a
  domain anyone verifies), which is the safe direction.

**The C5 surfaces are therefore these five, and not the signed payload's format:**

1. **A new operator-signed authority type** (`lop-refresh-grant-v1\0`) — the
   operator key can now authorise a *device* to spend a login's refresh family.
   New powers for an existing signer is the strongest of the five.
2. **Authority to suppress the owner's own POST** (`refresh_delegate`,
   `REFRESH_FAMILY_MOVED_KEY`) — a *reduction* in behaviour, but a new state that
   can strand a working login if it is ever wrong.
3. **New at-rest residency for a class-3 rotating secret on a second device**
   (`mesh-credentials.md` §1.1's class table marks these "**Never**" copyable;
   this is the first exception, and §3.6's confinement prerequisite is part of it).
4. **A peer-relayed owner-signed deny accepted outside `placement.merge`'s
   owner-write rule** (§3.4, Channel 1) — the one wire-trust rule this changes.
5. **The confinement deny-list change** the new residency needs (§3.6).

### 3.9 The code seams, file by file

| File | Change | Why it must be there |
|---|---|---|
| `providers/auth_store.py` | `RefreshDelegatedError`; `_delegated()` + the gate at `:3459` before `_needs_refresh`; `REFRESH_FAMILY_MOVED_KEY`; `ensure_oauth_fresh` unchanged otherwise | the only place a POST can happen |
| `network/credentials/owner.py` | `mint` arm (signs via `operator.sign`, refuses a second live delegation), `refresh_take`/`refresh_return` handlers, revoke mints the deny statement | owner-side authority, `on_broker` (`:479`) already dispatches by kind |
| `network/credentials/types.py` | the two kinds; `credential-refresh-v1` | wire vocabulary, one place |
| `network/credentials/client.py` | the node's `refresh_take` call, delegation document read, the pre-POST fence (§3.4), `mesh_origin` row write | the node's half of leg 2 |
| `network/credentials/held_policy.py` | nothing (new codes are restrictive by default) | — |
| `network/credentials/refresh.py` (new) | delegation document: load/verify/save, `flock`, size cap, epoch arithmetic | keeps `placement.py` from growing a second concern |
| `network/credentials/placement.py` | one narrow exception: accept a relayed *deny* statement after anchor verification | §3.4 Channel 1 |
| `network/credentials/store.py` | `_owner_policy_skips_local` (`:283`): a live delegation for this `(owner, key)` counts as the local answer **for that owner binding**, so a moved session (`policy: owner`) does not broker to an offline owner | otherwise a session moved onto the node breaks the moment the laptop closes — the exact scenario the requirement is about |
| `operator/verify.py` | the new domain + verifier | tag discipline |
| `network/audit.py` | six allowlists | non-material by construction |

No change to: `mcp/auth.py` (slice 2), `approvals.py` (§3.8), `provisioning`,
`secrets/`, the capability model, `wire.py`.

---

## 4. Failure matrix

| # | Trigger | Detected where | Action | Operator sees | Residual |
|---|---|---|---|---|---|
| 4.1 | Node refreshes, then finds the owner's token was already rotated (an outside refresh happened earlier) | node's POST earns `invalid_grant` with its own current token | tombstone + wipe + report + no retry (§3.5) | "sign in again" on both devices | a family may already be revoked for bystander reasons; the node is not the cause and does not claim to be |
| 4.2 | Owner revokes while offline | deny relay; else next contact; else expiry | refuse every POST from the next one; wipe on receipt; owner refuses handback | revoke is immediate on the owner's surfaces; the node's next use refuses | refreshes between revoke and receipt (bounded, §3.4) |
| 4.3 | Node holds no delegation / it expired / `refresh_delegate` absent on the owner | the pre-POST fence (§3.4 step 2) | behave exactly as today (#3439's held grant until expiry, then a refusal) | unchanged copy: "owner unreachable" | none new |
| 4.4 | Refresh minted a new family; owner later returns with the old one | owner's `refresh_delegate` gate blocks its POST; handback restores order | adopt on the last frame, else `family_moved` → re-login | "being refreshed by <node>" / "sign in again" | an owner that never gets the handback needs an interactive login — the stated cost |
| 4.5 | An external tool (Claude Code, Codex) holds the same login | mint-time declaration; use-time `invalid_grant` | refuse to mint when unknown; stop on first evidence | mint card explains the exclusivity requirement | cannot be detected while both are healthy — accepted and disclosed |
| 4.6 | Two nodes named for one key | owner's mint (single live record) | second mint refuses `already_delegated` | copy names the current node and its expiry | a hand-edited owner row could claim two; the node verifies one grant, and the owner has one record — the state is singular by construction, not by CLI politeness |
| 4.7 | Node crashes with the send marker armed | node's own store, next boot | defer, never re-present (`auth_store.py:191-218`), then `invalid_grant` → 4.1 | "sign-in expired" at worst | ~2 min deferral, as the owner's own path already accepts |
| 4.8 | Node loses its disk | — | family is gone with it; owner is `recalled` and must re-login | "sign in again" | deliberate; the alternative is a possibly-spent POST |
| 4.9 | Delegation live, owner's clock skew | `not_after` comparisons | the node treats `not_after` as a **hard** bound on its own clock; the owner never extends it | clock-skew diagnostics already exist (`lop network doctor`) | a badly skewed node stops early — the safe direction |
| 4.10 | Node is `pool`-kind | mint | refused (pool members declare no credentials of their own, `mesh-credentials.md` §6.1) | refusal names the member kind | — |

---

## 5. Test plan — denial and convergence, not the happy path

Harness that already exists and must be reused, not rebuilt: the two-device rig
`tests/unit/network/test_credentials_real_link.py:75` (`mesh` fixture: owner A
admin with an expired login, borrower B `drive`, both relays running), the
rotating stub IdP `tests/unit/network/test_credentials_owner.py:65`
(`RotatingIdP`: `posts` list, `current_refresh`, `invalid`), and the deny-fence
suite `tests/unit/network/test_held_policy_fences.py`.

| ID | Proves | Command / assertion |
|---|---|---|
| T1 | **Negative control** — no grant, no refresh | Node holds the material file and a row but **no** signed delegation: delete `refresh.json`, expire the contact, run a provider call. Assert `idp.posts == []` and the refusal is the shipped `owner_offline` path. |
| T2 | Forged/expired delegation | Tamper one byte of the signed statement; re-point `not_after` past. Assert `verify_refresh_grant` fails, zero POSTs, and the node writes `credential.refresh_refused` with `cause: "grant_invalid"`. |
| T3 | **Exclusivity** — the owner never POSTs while delegated | With the delegation live, force a resolve on the owner (`ensure_oauth_fresh`) and on the node concurrently. Assert `idp.posts` has exactly one entry **and** it came from the node's process, and the owner raised `RefreshDelegatedError`. |
| T4 | Exactly one refresher across a whole window | 5 consecutive refreshes on the node + 5 forced resolves on the owner. Assert `len(idp.posts) == 5` and every entry is the node's next expected token; owner POSTs `== 0`. |
| T5 | **Deny wins offline** | Revoke on the owner, deliver only the deny statement to the node over a third member's link, then force a node refresh. Assert zero POSTs after the deny, `held_policy` carries the code, and the grant cache is empty (`GrantCache.invalidate_policy`, `client.py:164`). |
| T6 | Deny survives a reconnection | Same as T5, then bring the owner back: assert the node's next refresh is still refused until a *new* signed delegation lands. (The `held_authorized_fence` rule, `state.py:145`.) |
| T7 | **Convergence** | Node holds epoch 9; run the three-message handback. Assert: owner's row is the node's token, `refresh_delegate` cleared, node's copy gone, and the owner's next refresh POSTs **once** and succeeds. |
| T8 | Handback with the last frame dropped | Drop message 3 after the node wiped. Assert the owner is `family_moved`, serves no bearer, and says "sign in again"; assert no POST ever presents the old token. |
| T9 | Family moved outside the device | Set `idp.current_refresh = "refresh-99"` behind the node (an external tool's refresh), then force a node refresh. Assert: one POST, `invalid_grant`, tombstone written, material wiped, `credential.refresh_family_moved` audited, and **no further POST in the next 10 minutes of simulated turns**. |
| T10 | Node without a lease | No delegation, no row: assert the shipped held-grant behaviour is byte-identical to `test_credentials_held_offline.py` (a retained control, not a new behaviour). |
| T11 | Nothing material in logs | Run T3/T5/T9 with the audit log open; grep for the token strings. Assert zero hits, and that the allowlist would have dropped them anyway (a direct call to the writer with `refresh_token` in the detail). |
| T12 | Two-process race on the node | Two node processes refresh the same delegated key at once: assert one POST (the local lease, unchanged) and both callers served. |
| T13 | **Confinement** (§3.6) | From a write-confined shell, attempt to read `auth.db` / the material path. Record before/after; the assertion is the *measured* refusal, not the list. |
| T14 | Mixed fleet | Owner on this build, node on the pre-delegation build (and the reverse). Assert `credential-refresh-v1` is absent from `caps`, no new kind is sent, and behaviour is exactly today's. |

Runs: T1–T6 and T9–T12 are `tests/unit/network/` (isolated `LOCAL_OPERATOR_CONFIG_DIR`,
`CMUX_*` stripped, synthetic device ids — never a live peer, per `guide://delivery`
"Never use a live peer as the revoke test"). T7/T8 ride the same two-device rig in
one host; a two-host confirmation is one manual pass, not a CI matrix. T13 is a
one-off measurement recorded on the PR.

---

## 6. Unproven assumptions — no implementation may rely on any of these

Each has the exact experiment that settles it, and each is a **blocking**
prerequisite for the slice that depends on it. Stated plainly: **nothing below
may be assumed in code until its experiment is recorded on the PR.**

1. **Does a provider's refresh actually rotate, and does reuse revoke the family,
   per provider?** The docs assume Radient and Kimi rotate (§1.1's two
   measurements; `mesh-credentials.md` §1.1 row 3 "**Yes** — the IdP rotates the
   refresh token on use"); Anthropic/OpenAI behaviour is not measured anywhere in
   this tree.
   *Experiment:* for each of `anthropic`, `openai` (both flows), `xai`, `zai`,
   `qwencloud`, `radient`, `kimi`: refresh once with a real grant, then present
   the **pre-refresh** token. Record: rotated / not rotated / second POST accepted
   / family revoked. **A provider that does not rotate is a different (cheaper)
   case and may get its own, simpler grant class; a provider that revokes on
   reuse gets the strictest fence.** Until measured, assume the strictest.
2. **Does signing in again invalidate an earlier refresh token?** Unknown here,
   and it decides whether a re-login on the owner is *sufficient* to recover a
   stranded family (a code path in §4.4/4.8).
   *Experiment:* sign in fresh on device A, keep the old refresh token from before
   the sign-in, present it. Record accepted/refused per provider. If accepted, the
   recovery copy in the surfaces must not claim that re-login "revokes" anything.
3. **Does a paired mesh member carry an operator anchor?** The whole signed gate
   rests on `operator/trust.py`'s root-owned public key being readable *on the
   node*. Nothing in this pass proves a paired node ever ran `lop operator setup`.
   *Experiment:* on a freshly paired node, read
   `/Library/Application Support/local-operator/operators/<uid>.json` (or the
   Linux/Windows equivalent) and verify a known-good signature with
   `operator.verify`. **If the anchor is absent on nodes, the gate degrades to the
   owner's device key** — which every member already verifies from its member row
   (`network/types.py:1417`) — and the design must be re-reviewed as a
   *machine*-authorised gate, not a human one. Do not ship assuming either.
4. **Can a headless node unseal an encrypted store with no prompt?** Only relevant
   if placement option (b) of §3.6 is chosen. `secrets/keys.py`'s default tier is
   a 0600 keyfile read "with no prompt", and `secrets/access.py:278-300`'s broker
   seam falls back to it — but that is a reading of two files, not a measurement
   on a node.
   *Experiment:* on a node with no unlocked login keychain, run a sealed
   read/write through `store_provider_key`/`provider_secret_value` and record the
   exit and the timing. Any prompt → the capability is unavailable there, fail
   closed.
5. **Does the provider's own expiry allow a 72 h window at all?** Unverified for
   long-lived access tokens; the grant's `not_after` must not exceed what the
   provider will actually honour, or the node's last hours produce refusals it
   could have surfaced earlier.
   *Experiment:* read each provider's `expires_in` from a real refresh
   (`providers/oauth/*.py` mints with a skew subtracted), and pin `not_after`'s
   default from the **shortest** measured bearer, not from a round number.

---

## 7. What this design deliberately does NOT do

- **No MCP OAuth grants.** `ensure_mcp_oauth_fresh` (`mcp/auth.py:5217`) is a
  second refresher with a file lock, and an MCP grant cannot be created remotely
  anyway (loopback callback, `mesh-credentials.md` §4.7). Slice 2 may revisit it
  once §6 item 1 is measured for MCP servers.
- **No device-bound providers.** Kimi's device fingerprint (`types.py:159-170`,
   `refuse_device_bound`, `placement.py:450`) makes a copy meaningless and
   dangerous; the mint refuses those keys by name, as the placement document
   already does.
- **No background refresher, no new scheduler.** Refresh stays demand-driven,
  exactly as `mesh-credentials.md` §3.5 declines proactive top-up. A session on
  the node triggers it; an idle node with no sessions refreshes nothing.
- **No automatic renewal of the delegation itself.** A node may never extend its
  own authority; past `not_after` it stops, full stop. Only the owner can mint,
  and only with a human present (`operator.sign`'s prompt is the gate).
- **No long-lived bearer, no copied bearer, no `auth.db` replication** (§7.1/7.4
  of `mesh-credentials.md` stand unchanged).
- **No new peer op or capability row**; no change to `approvals.py`'s payload,
  kinds or records.
- **No protection against a node compromised at the OS level while the delegation
  is live.** The attacker holds a working refresh family for the window; the
  provider knows nothing of our fence. Named here so nobody implies otherwise;
  the mitigations are the window, the wipe, the audit trail, and the re-login.
- **No re-login automation, no browser, no provider-side revocation call.**
- **No "warning-only" mode.** A delegation is either verified or the node does
  not POST — `mesh-credential-lifecycle.md` already rejected warning-only fixes
  ("permission is not effective auth") and this design keeps that.
- **No behaviour change for the 0-peer topology** (`mesh-credentials.md` §5.1):
  no network configured → none of this code is constructed.

---

## 8. Risks to watch during rollout

1. **Stranded logins.** The single most likely operator-visible harm: a node lost
   mid-delegation costs an interactive re-login. Watch the ratio of
   `credential.refresh_family_moved` to `credential.refresh_returned`; a ratio
   above ~1:10 says the window or the handback path is wrong.
2. **Delegation latency mistaken for an outage.** Surfaces must say "refreshed by
   <node>" rather than "no login" — the `held_offline` precedent
   (`types.py:730`: it exists precisely so a held grant is not reported as an
   owner resolve) applies unchanged. Any surface that renders `effective` state
   must learn the new cause, in the same slice, or the operator sees a false
   outage (the failure PR #3439 was written to stop).
3. **A second delegate by hand-edited state.** The owner's record is a JSON field
   in a 0600 row; a hand edit could claim two nodes. The singular mint and the
   node-side grant verification bound the damage, but `lop network doctor` should
   report `delegates > 1` as a fault.
4. **Provider rate pressure.** `max_refreshes` and no-forced-refresh
   (`force_refresh` refused for a delegated key — §7.7 already refuses it for
   non-admin) are the levers; §3.4's existing note "gratuitous forced refreshes
   are the cheapest way to make an IdP start invalidating tokens" is the reason.
5. **Mixed fleet.** The cap-string gate is the whole mitigation; a node that
   ignores it sends `internal` to old owners, which is classified *transient* and
   will read as a flaky broker.
6. **The audit trail must be enough to reconstruct a delegation** — who, what,
   when, under which `grant_id` — from the log alone, since the material is
   nowhere in it. That is a review item on the events, not an aspiration.

---

## 9. Open questions, with my recommendation

**Q1 — Should the delegation be an operator-signed statement or an approvals
*kind*?** *Recommendation:* the standalone signed statement (§3.7g), because a
capability grant is multi-use and time-bounded while an approval record is
single-use with a transition matrix and tombstones (`approvals.py:142-190`);
forcing one into the other would either weaken the fence or widen the approval
payload. *Evidence that would change it:* if the desktop's signing surface cannot
sign a fifth domain without bespoke code (it currently signs four), then a kind is
cheaper — and §3.8 says exactly which byte-level change that is, so the reviewer
can price it.

**Q2 — Where should the node's material live: `auth.db` or
`network/credentials/`?** *Recommendation:* `auth.db` **plus** the confinement
deny-list entry (§3.6), because it keeps exactly one refresh implementation.
*Evidence that would change it:* T13 measured showing the confinement list cannot
be narrowed to one file without breaking a legitimate reader — then the denied
directory plus the loader seam (§3.7e), and the second implementation cost is
paid deliberately.

**Q3 — `not_after` default.** *Recommendation:* 72 h, derived from the shortest
provider bearer measured in §6 item 5, not chosen as a round number; and
`max_refreshes` defaulting to that window's worst case (a 55-minute bearer ≈ 79).
*Evidence:* §6 items 1 and 5, plus a two-week audit count of how often the operator's
laptop is closed beyond each candidate.

**Q4 — Should the owner's broker still serve borrows while a delegation is
live?** *Recommendation:* yes for access tokens it still holds a live bearer for,
**no** for a forced refresh (it has no authority to POST). *Evidence:* none
needed; this is the same split the send-marker already makes.

**Q5 — Do we need the deny relay in slice 1, or is expiry enough?**
*Recommendation:* ship the relay in slice 1 if — and only if — §6 item 3 shows a node
can verify the anchor; otherwise slice 1 is expiry-bounded only and its copy must
say so, because a revoke that takes 72 h to bite is a different promise than one
that takes a minute. Say which one is shipping, on the surfaces, in words.

---

## Appendix A — constraint → discharge map (for the reviewer)

| Constraint | Discharged by | Proved by |
|---|---|---|
| 1. scope by grant, not by device | mint names one `(key, node)`; the node's material is one key; `refuse_device_bound` unchanged | T1, T2, T3, T10 |
| 2. never two refreshers | Fence 1 (owner refuses to POST) + Fence 2 (epoch, last-frame rule) + Fence 4 (stop on doubt) | T3, T4, T7, T8, T9, T12 |
| 3. deny wins, offline, fail closed | Fence 3's four pre-POST gates; deny relay; expiry backstop; `PolicyUnknown` → refuse | T5, T6, T2 |
| 4. no new authority by implication | opt-in per login per peer; `_delegated` gate; existing grants untouched; six audit events; no new op/capability row | T1, T14, T11, and §8.6's audit-reconstruction review |
| 5. material stays under the owner's control | single-holder window, wipe on return/failure, owner's signature as the only authority, nothing in logs, confinement | T7, T8, T11, T13 |
