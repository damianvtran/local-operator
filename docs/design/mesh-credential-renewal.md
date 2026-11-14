# Design: a node keeps its borrowed logins alive for the whole 24 h hold

Status: **design, pre-implementation.** Read against `origin/main` @ `a46ef6af5` plus
PR #3922 (`feat/node-credential-offline-hold` @ `c651774d97b4`, the 24 h offline
hold, reviewed and QA'd). Every `file:line` below is **at `c651774d`**; the symbol
name is the stable citation. `feat/mesh-renew-on-reconnect` (the owner pushes fresh
grants when a link comes back) was **not pushed** when this was written; §B.3 names
the one hook this design needs from it, not its internals.

Operator decisions this implements (Damian, 2026-11-13, verbatim):

- Model logins (Anthropic ~7.9 h access token, Radient ~55 min): **"Lend the node the
  renewal token too"**, chosen over a separate node login, after being told the risk:
  the node gains the power to renew his login, and if the provider rotates renewal
  tokens, the node renewing could sign his Mac out.
- GitHub and MCP logins (today not held at all): **"Yes, GitHub and MCP"** get the
  same 24 h hold.
- Priorities: simple, automatic UX; security locked down without friction.

---

## 0. The answer in one page

**A — hold GitHub and MCP (small).** Both are already *stamped* with the owner's
hold by the owner (`owner.py:1430-1444` `_mint_grant`, `owner.py:1713-1753`
`_serve_github_app`). The borrower refuses to serve them from the hold because of
one line, `client.py:1422`, and GitHub additionally kills its own token at the
15-minute window end in **two** places (owner lender `owner.py:1746-1753`,
borrower belt `github.py:1912-1913`). Remove MCP and GitHub from that exclusion,
move both GitHub revokes from the window end to the hold end, revoke at once on
panic, share revoke or membership loss, and give the MCP and GitHub status rows the
same "held until HH:MM UTC" sentence model rows got. The token's own expiry
still wins: an MCP server's access token (often ~1 h) and a GitHub App installation
token (1 h, server-capped) end the hold early; a fine-grained PAT or the `gh` CLI
token rides the full 24 h. About 150–250 product lines.

**B — lend renewal tokens for Anthropic and Radient (the real change).** Most of the
machinery is **already on `main`**, unused: slice 1 of
`docs/design/mesh-refresh-delegation.md` (commits `b9a03aebc`, `a480c5931`) built
the node row, the pre-POST fence, the epoch ledger, the handback and the owner's
atomic adoption. Nothing calls it in production: `mint_refresh_delegation`,
`refresh_take` and `refresh_return` have no caller outside tests, because slice 1
needs an **operator signature per delegation**, and slice 1 **freezes the owner's
own refresh for the whole window**. Neither fits the operator's answer. B keeps the
slice-1 parts and changes three things:

1. **Authority.** Automatic, under a standing setting the operator has now approved,
   carried by the owner's device key on the encrypted link. No per-use operator
   signature and no new prompt. The signed statement stays available as the stricter
   mode.
2. **One renewer at a time, decided by a lease rather than a freeze.** The owner
   renews while the node can hear it. The node may renew only after it has heard
   nothing from the owner for `L` (default `grant_ttl_s`, 15 min). Before the owner
   renews a lent login it asks the node first (a *claim*), so the node never later
   presents a token the owner has already rotated.
3. **Write-back on reconnect.** The node returns the newest token pair and its renewal
   receipts. The owner adopts it with a compare-and-set, through the existing
   `AuthStore.adopt_refresh_handback`. On a conflict the owner's own successful
   rotation wins (§B.4).

**What the operator gets.** With the Mac closed overnight, node sessions keep Anthropic and
Radient for 24 h from the last time the Mac was reachable. In the morning the Mac
reconnects, takes back the newest login without a prompt, and shows "node2 renewed
your Anthropic login 3 times while this Mac was away (last 06:41 UTC)".

**What it costs (plain words, for B1's PR body, §D D7; full delta in §B.8).** Today a stolen
node holds an Anthropic login for at most ~8 h (a Radient one for ~1 h). With
renewal it holds one until custody ends: 24 h after the node last heard from the
Mac, or sooner on panic, share revoke or sign-out, whichever comes first. A node
compromised *and* cut off keeps renewing for up to 24 h. A thief who copies the
renewal token off the disk can use it until the provider ends the login family
(Anthropic: 30 days from sign-in, `oauth/anthropic.py:14-15,66`) or until a renewal
by someone else rotates it out from under them. The remedy is `lop network panic`
plus a provider-side sign-out.

**Decisions of 2026-11-14 (§D) are folded in:**
- the renewer is chosen automatically;
- both providers are treated as family-revoking, with no live probe;
- the Mac's wait is visible and names the node;
- logout revokes at the provider;
- a sealed return-only copy is kept for up to 7 days;
- only lop-native sign-ins are lent;
- MCP refresh tokens stay out of scope.

---

## A. Hold GitHub and MCP

### A.1 What is lent today

| Key | What crosses the link | Lifetime of what crosses | Where minted |
|---|---|---|---|
| `github`, App arm | A fresh **installation token**, repository-narrowed server-side (`github.py:990-1031` `GithubMinter.mint`) | GitHub caps it at **1 h** (`github.py:969-980`); lop revokes it at the **grant end** (≤ 15 min) | `owner.py:1653-1754` `_serve_github_app` |
| `github`, PAT arm (`GITHUB_TOKEN` secret) | The owner's fine-grained PAT itself (`owner.py:1611-1620`) | The PAT's own expiry; there is no revoke handle (`github.py:20-24`) | `owner.py:1611-1650` |
| `github`, `gh` arm | The owner's `gh auth token` (an OAuth user token, normally non-expiring) (`github.py:25-31`, `read_gh_token` `:757`) | No expiry; no revoke handle | same |
| `mcp:<url>` | The **access token only** (`kind="mcp-oauth"`), never the refresh token (`owner.py:1270-1345`) | The server's `expires_in`; varies per server and is not measured in this tree | `owner.py` MCP arm → `_mint_grant` |

The owner already stamps `hold_until_ms = min(token expiry, now + offline_hold_s)` on
**all** of them, because both paths go through `_offline_hold_until_ms`
(`owner.py:3629-3645`): `_mint_grant` at `:1444` for MCP and the PAT/`gh` arms,
`_serve_github_app` at `owner.py:1722-1724` for the App arm. The comment there says
so: "Stated for the github arm too, though PR-A's borrower does not hold".

### A.2 Why they were excluded

- **The original gate.** The exclusion was added in `e70f7643c` ("keep serving a
  borrowed grant while its owner is unreachable", 2026-11-04), whose message says
  the hold applies only when "the request is not a forced refresh, MCP, GitHub or
  device-bound". The inline reason was conservatism, not a hazard: "Not holdable,
  and their existing cache behaviour is left exactly as it was". The comment is in
  that commit's hunk of `_held_or`.
- **GitHub's concrete hazard**, stated in the hold PR (`client.py:1408-1411`): "the
  borrower's own window-end self-revoke (`github._schedule_self_revoke`) DELETEs
  the token at `grant_exp`, so holding it past that would hand out a token this
  device has already killed. Moving that DELETE to the hold end is design §4.3."
  The owner's lender has the same window-end DELETE (`owner.py:1746-1753` registers
  `grant_exp_ms=grant_exp`; `GithubLender._schedule` fires at
  `record.grant_exp_ms`, `github.py:1234-1237`).
- **MCP has no such hazard.** Its access token has no revoke endpoint in our code.
  The only no-dial read is `BrokeredBearerAuth._live_bearer` → `live_grant`
  (`mcp_bearer.py:74-86`, `client.py:1136`), which is already fenced for panic and
  membership. On a miss, `_blocking_bearer` (`mcp_bearer.py:88-112`) calls
  `request_grant_sync`, which dials, gets `owner_offline`, and reaches `_held_or`
  (`client.py:956`, `:1006`). That call is refused only by `client.py:1422`.

### A.3 The change (same rules as model logins)

The rules are unchanged and already implemented in `_held_or`
(`client.py:1382-1518`) and `Grant.hold_end_ms` (`types.py:813-830`). The hold lasts 24 h
from the last grant received, on the borrower's own wall *and* sleep-inclusive
clocks. Panic ends it at once (`_hold_admission` → `_NETWORK_UNTRUSTED`,
`client.py:1447-1449`). A share revoke ends it (`should_borrow` false →
`request_grant_sync` drops, `client.py:896-907`). So does membership loss
(`client.py:1437-1439`). A non-transport owner answer drops it
(`client.py:1425-1428`). The token's own expiry wins (`types.py:828-829`).

| # | Site (`c651774d`) | Change |
|---|---|---|
| A1 | `client.py:1422` | `if is_mcp_key(key) or is_device_bound_provider(key) or is_github_key(key):` → `if is_device_bound_provider(key):`. Rewrite the docstring paragraph at `:1408-1411` to say GitHub and MCP are held, and why the GitHub revokes moved. Device-bound providers (Kimi) stay excluded: the hold never applied to a login that cannot be brokered. |
| A2 | `github.py:1910-1913` (`_git_env` tail) | Schedule the belt at the **hold end**, not `grant_expires_at_ms`. Use `outcome.held_until_ms` when the serve was a held one; otherwise use `min(token_expires_at_ms, hold_until_ms)` capped by `received_wall + offline_hold_s`, which is `Grant.hold_end_ms`. Schedule it **only for App tokens**: today it also fires `DELETE /installation/token` at a PAT or a `gh` token, a no-op 401 that the docstring at `github.py:86-97` should stop implying is a revoke. Record the token's arm on the grant: `CredentialRef.kind` already carries `github-app` / token arms (`owner.py` `_serve_github_app` vs `:1611`). |
| A3 | `github.py:1383-1405` `_schedule_self_revoke` | Also expose `revoke_now(token)` for the drop paths. When `_held_or` drops a held GitHub grant for panic, revoke or membership loss (`client.py:1444-1455`, `:896-907`), fire the DELETE for an App token **immediately**. This is stronger than today: currently a panic leaves the App token alive until its window end. |
| A4 | `owner.py:1746-1753` | `lender.register(..., grant_exp_ms=grant.hold_until_ms or grant_exp, ...)`, so the owner's scheduled DELETE fires at the hold end, which for the App arm is the token's 1 h ceiling. `credential revoke` stays immediate (`GithubLender.revoke_holder`, `github.py:1176`; local op `credentials/__init__.py:333-353`). Rename the field `revoke_at_ms` in `_TrackedToken` (`github.py:1069-1079`) so the name stops saying "grant". |
| A5 | `store.py:794-797` `owner_wait_hint` | **No change.** It is the model-failover waiter; MCP and GitHub never use it. Recorded so a reviewer does not "complete" A1 here. |
| A6 | Copy: `readiness.py:1598-1620` built the held-model sentence from `last_grant_expires_at` | Give the MCP row (`readiness.py` `mcp_*` helpers `:886-1090`; evidence vocabulary `mcp_state.py:94` maps `owner_offline` → `unavailable`) and the GitHub row (`github_fact`, `readiness.py:814`) the same reading when `effective` says `held_offline` (`effective.py:29,121-129`): "**node2** is using the **Linear** sign-in **Damian's MacBook** lent before it went offline, until 10:40 UTC; nothing is renewed until it is back". At lapse use the existing "sign in again there / reconnect" remedy (`mcp_login_remedy`, `readiness.py:1038`). The end time is the **token's** end when that comes first, so the copy never promises 24 h for a 1 h token. New strings go through the i18n catalogue as #3922 did. |

**Receipts.** `credential.held_served` / `credential.held_lapsed` (`audit.py:212-218`,
`:580-588`) are key-agnostic, so GitHub and MCP holds are counted with no audit
change.

**MCP 401 while held.** `BrokeredBearerAuth` drops the grant on 401 and re-asks with
`force_refresh` (`mcp_bearer.py:180-212`). A forced refresh is never answered from
the hold (`client.py:1463-1466`), so a held MCP bearer that the server refuses ends
honestly and is not re-served.

### A.4 Security delta of A (plain words)

- **PAT and `gh` arms:** a node cut off from the Mac can push and open PRs with the lent
  GitHub token for up to 24 h after its last grant (today: 15 min). The honest
  node's helper still enforces the repository allow-list (`github.py:80-86`). A
  compromised node is not bound by it: a `gh` token is as broad as the operator's
  `gh` login. This is the same class of exposure #3922 accepted for model API keys.
- **App arm:** the installation token lives to its 1 h GitHub ceiling instead of being
  revoked at 15 min. Panic, share revoke and membership loss now revoke it **at once** on the
  node (A3), and `credential revoke` revokes it at once on the owner.
- **MCP:** the access token is used until its own expiry instead of at most 15 min.
  The refresh token never leaves the owner.

---

## B. Lend renewal tokens for Anthropic and Radient

### B.1 Which providers rotate (code and evidence)

| Provider | Rotates? | On reuse of an old token | Evidence | Access lifetime |
|---|---|---|---|---|
| `radient` | **Yes**, rolling (`oauth/radient.py:333` "rolling refresh token"; `:387-388` keeps the new one) | **Revokes the whole family** (reuse detection): measured, "a Radient grant died twice in ~17 hours on this machine" (`providers/auth_store.py:30-34`) | measured in this tree | ~55 min (`refresh.py` `WORST_CASE_BEARER_S`, from the store's own measurement) |
| `anthropic` | **Yes**: the refresh merges the new `refresh_token` (`oauth/anthropic.py:289-290`); externally observed "issues a new refresh token on every refresh and rejects the old one with `invalid_grant`" (nosleeplab.com notes; anthropics/claude-code issues #80585, #108871) | Old token **rejected**. Whether the *family* is also revoked is **not measured** | external reports only | ~8 h (operator figure 7.9 h); family dies 30 days after sign-in regardless (`oauth/anthropic.py:14-15,66`) |
| `openai` | merges a new one if sent (`oauth/openai.py:422-423`) | not measured | — | ~10 days. **Not needed:** the 24 h hold already covers it (`credentials/__init__.py:64-65`) |
| `kimi` | rotates (delegation design §6) | — | — | **Excluded:** device-bound, never brokered (`types.py:196`) |

**Rule for B (decided 2026-11-14, §D D2/D3):** treat **both** in-scope providers as
the strictest class, "reuse revokes the family", **permanently in this design**.
There is **no live measurement** with the operator's real tokens: a reuse probe is
the very POST that could sign him out. The evidence is what already exists: the
auth store's two Radient measurements and the public Anthropic reports above.
Nothing in B relies on either provider being lenient, and Radient is one family
with no reuse grace (Radient server changes are out of scope).

### B.2 What already exists on `main` and is reused

Slice 1 of `mesh-refresh-delegation.md` landed (`b9a03aebc`, `a480c5931`):

| Piece | Where | Reused as |
|---|---|---|
| Node delegation document + epoch ledger (`active` / `refused` / `returned` / `failed`), `flock`, size cap, `handback_holds` | `network/credentials/refresh.py` (`install` `:317`, `active` `:389`, `record_post` `:428`, `stop` `:449`, `note_deny` `:500`, `live_for` `:550`, `handback_holds` `:581`) | the node's custody record. Its schema gains the lease fields in §B.3 |
| Owner's in-row delegation record that blocks the owner's own POST | `auth_store.py:3597` `_delegated`, gate at `:3979-3983` and `:4026-4030`; `set_refresh_delegate` `:3757`, `clear_refresh_delegate` `:3771`, `mark_family_moved` `:3780` | the owner's **lend** record; its blocking rule becomes the lease rule (§B.3 R3) |
| Pre-POST fence + outcome watch seam | `auth_store.py:2054-2088` `set_refresh_fence`, called at `:4160-4171`; node fence `client.py:2697` `refresh_fence`, installer `:2923` | the node's renewal gate (§B.3 R2) and, newly, an owner-side fence (R3) |
| Node row with mesh provenance, written from the taken material | `client.py:2648` `_write_delegated_row`; `refresh_take` `client.py:2538` | node custody storage (§B.5) |
| Owner handlers `refresh_take`, `refresh_return` (intent/ack/done) | `owner.py:2472`, `:2560`; kinds `types.py:98-110` | take = lend, return = write-back |
| Atomic adoption with incarnation and tombstone CAS | `auth_store.py:3818` `adopt_refresh_handback` | write-back adoption (§B.4) |
| Family-moved stop (no retry after the node's own `invalid_grant`) | `owner.py:1873`, `client.py:2909` `report_family_moved`, `auth_store.py:3780` | unchanged |
| Real-path rig with a rotating IdP | `tests/unit/network/test_refresh_delegation_realpath.py`; `RotatingIdP` `tests/unit/network/test_credentials_owner.py:65`; two real relays on loopback TCP: `tests/unit/network/test_credentials_reconcile_link.py` (relay at `:436`) | §C tests |

**What B does not take from slice 1:** the operator-signed statement as the
*required* authority (`owner.py:2343-2380` `mint_refresh_delegation` prompts
`operator.sign`; `refresh.active` re-verifies against the operator anchor on every
read). The delegation design's §6.3 recorded that a paired node may not carry an
operator anchor at all. The operator has now made the standing decision, so the
authority for a *lend* becomes the owner's device identity on the authenticated
link, plus the setting in §B.6. That is what `net_broker` already trusts for every grant.
The signed mode stays as an opt-in (`network.credentials.renewal_authority:
signed`) for operators who want a human gesture per delegation. This is the one
deliberate step down from slice 1, and the PR must name it (§B.8).

### B.3 The protocol: one renewer at a time

Vocabulary: **lend** = the owner gives one node the current token pair plus the
family epoch. **Custody** = the node's sealed copy and its record. **Claim** = the
owner asking the node "are you renewing? give me the newest". **Write-back** =
the node returning a newer pair.

**R1 — one renewing node per login, chosen automatically (§D D1).** Renewal is lent
to at most **one** node per `(owner, credential_id)`: **the node that currently
holds the lent login**, meaning the one whose grant requests for it carry live
sessions running turns. Nobody picks. Other holders of the same login keep today's
access-only hold. Enforced owner-side by the existing "one live record"
refusal in `mint_refresh_delegation` (`owner.py:2343-2370`), now in the automatic
path.

**The renewer moves only through the owner.** If another node becomes the active
holder while the owner is reachable, the owner first claims back from the current
renewer (R3; a write-back if that node renewed), and only then lends to the new one.
While the owner is away the renewer cannot change: a second node keeps the
access-only hold. So two renewers can never coexist.

Eligible means all of these:

- the provider is in `RENEWABLE_PROVIDERS = {"anthropic", "radient"}`;
- the login is a **lop-native sign-in** (§D D5). The owner row carries a native
  login incarnation (`auth_store.py:2589` `native_login_incarnation`; only explicit
  AuthStore login writes stamp one, `lifecycle.py:990`). An imported or
  externally shared row has none and is never lent. This replaces slice 1's
  operator-asserted `exclusivity_confirmed` (`owner.py:2349`);
- the placement row is device-scoped (sessions have no rail identity, the argument
  at `github.py:57-63`);
- the node's hello advertised `credential-renewal-v1`;
- the network is trusted;
- the setting is on (§B.6).

A mismatch on any of these is not an error: the node simply gets no renewal token.

**R2 — the node renews only while the owner has gone quiet.** The node's renewal
gate (extend `client.py:2697` `refresh_fence`; every check runs before **each**
POST) passes only if all of these hold:

1. The membership is active, the network is trusted, the node is still a holder, and no
   deny watermark is set (the existing fences 1, 2b and 3 of `refresh_fence`).
2. Custody is live: `now < last_owner_contact + offline_hold_s`, on **both** wall
   and sleep-inclusive clocks (the same two-clock rule as `Grant.hold_end_ms`,
   `types.py:813-830`).
3. **The lease has lapsed.** `now_boot ≥ takeover_not_before`. Every frame the
   node receives from the owner for this login (grant reply, push, claim) sets
   `takeover_not_before = receive_boot + L + margin`. `L` defaults to
   `grant_ttl_s` (900 s). That is the operator's existing "owner unreachable"
   unit, and the 2026-11-13 laptop-sleep incident (24 min) exceeded it, which is
   the case this exists for. `margin` = 60 s plus 0.1 % of `L` for clock-rate
   difference.
4. Its copy is **current**: it is not in `awaiting` (set by a claim, R3), and the
   owner's last stated epoch ≤ custody epoch.
5. The node's own access token is within the store's skew of expiry
   (`_needs_refresh`, `auth_store.py:3912`). The node never renews early, so an
   idle node never renews at all.
6. The slice-1 refresh budget is not spent (`refresh.max_refreshes_for(24 h)` =
   28 at the 55-minute bearer).

The node's single-flight is the store's own cross-process lease and send marker
(`auth_store.py:3181` `_try_refresh_lease`; `REFRESH_SEND_UNCONFIRMED_KEY` `:191`).
Because the node row lives in the node's own `auth.db`, every node process
contends on the same row. No new lock is invented.

**R3 — the owner renews while the node can hear it, and claims first.** Install
the same `set_refresh_fence` seam on the **owner** whenever it has a live lend:
`MeshAwareAuthStore` already installs fences (`store.py:229-243`), so this adds
an owner-side variant. The owner's `_delegated` gate (`auth_store.py:3979`) stops
blocking for the new record state `lent` (it keeps blocking `active` / `recalled`
for the signed mode). Before presenting a lent login's refresh token, the
owner's fence asks its own relay, over a new local op `credential_claim` (bounded
by `W`, below), to send `refresh_return{act:"claim", epoch e}` to the node. The node
answers one of two ways:

- **`standby`**: "I have not renewed; I am at epoch e". The node sets `awaiting =
  e+1` and **will not take over until it receives epoch ≥ e+1** or a release. The
  owner POSTs, persists, then pushes the new pair
  (`refresh_take`-shaped answer, owner→node, on the existing link; retried on every
  lifecycle tick and on reconnect, which is the hook from `feat/mesh-renew-on-reconnect`).
- **`renewed`**: "I am at epoch e+k, here is the pair and my receipts". This is a
  write-back (§B.4). The owner adopts it and **usually no longer needs to POST at
  all**: the node's fresh access token is now the owner's too.

The claim closes the only ordinary race: the owner rotating while the node holds
the old token and later takes over with it. A lost push now leaves the node
*waiting* (its sessions fall back to the access-only hold and then to
`owner_wait`) instead of presenting a spent token.

**R4 — the claim gets no answer within `W`.** `W` = 60 s, the redial backoff
cap that `owner_wait_s` already reasons from (`credentials/__init__.py:135-151`).
The wake-to-link-up distribution must be measured in QA before this ships (§C).

- If the owner heard from the node less than `L` ago, the node cannot have taken
  over yet. But a POST now would leave the node a stale copy that it could present
  after `L`. The owner therefore **does not present**. Its turn waits like
  `owner_wait` does.
- **R4 is visible (§D D2).** The Mac's status row and the turn's notice name the
  node and the state: **"Waiting for node2 to hand back your Anthropic login."**
  The remedy line reads: "It comes back as soon as node2 reconnects. Or sign in again
  here." Wire it as a new `effective_state` value, `awaiting_handback`, beside
  `held_offline`. It is published by the owner fence when it refuses, and read by
  the model row in `readiness.py` (`:1598-1620`, the pattern #3922 added). Desktop and
  mobile read it through the same effective-access fact.
- **When the wait clears on its own.** It clears on write-back (adopt, then no prompt),
  or on a claim answer that shows no renewal. Without a contact it clears only when
  the node has certainly deleted everything that could be newer than the Mac's
  copy. With D4's return-only copy that is the **7-day return cap**, not 24 h.
  Clearing at 24 h would present a token the node may have rotated, which is the
  sign-out D4 exists to prevent (see the reconciliation note under §D). Past the cap
  the owner resumes renewing with no prompt. If the node renewed and never
  wrote back, the owner's copy is spent and its POST signs the Mac out (both
  providers are treated as family-revoking). That is the one unavoidable sign-out:
  the node renewed and was never heard from again for 7 days.
- **"Sign in again here"** is always available and starts a new family
  (`login_incarnation` moves). The node's custody of the old family is dropped at
  the next contact (§B.5), and `adopt_refresh_handback` refuses its write-back by
  incarnation (`auth_store.py:3818-3850`).

This is deliberately **safety-first for the owner's own presentation**, for both
providers (§D D2). The alternative, owner-first (POST anyway), lets an owner
refresh during a link outage be followed by a node takeover with the spent token.
That is the reuse-detection POST that revokes the family on both machines.

**Why a lease and not "the node renews whenever the owner is unreachable".**
"Unreachable" is not a shared fact: during a link outage each side sees the other
as gone. The lease turns it into one: the owner treats "has the node heard from me
in the last `L`" as its own clock fact (`claim` acks are the evidence), and the
node waits a strictly later deadline (`receive + L + margin > send + L`).

### B.4 Write-back, atomic adoption, and the conflict rule

**When.** (a) Whenever the node's link to the owner comes back. The node sends
first, before any grant ask (hook: renew-on-reconnect's link-up event on the node
side). (b) As the answer to any claim (R3).

**What.** `refresh_return{act:"done", epoch, parent_epoch, parent_fp, incarnation,
refresh, access, expires, receipts[]}`. `parent_fp` is a keyed fingerprint (HMAC
under the link key) of the refresh token the node *started from*, so the owner can
tell which pair it forked from without either side logging or comparing raw
tokens. `receipts` lists `{at_ms, epoch}` per renewal (no material).

**Adoption (owner).** It runs in one `adopt_refresh_handback` call
(`auth_store.py:3818`), which already compare-and-sets incarnation and tombstone.
Add one more compare-and-set: **the row's current refresh token must still be the
lent one** (`fp(row.refresh) == parent_fp`). It runs under the store's refresh
lease (`_try_refresh_lease`), so no owner process POSTs in between.

**Conflict rule (both renewed).** For a strictly rotating provider, two
*successful* rotations from the same parent cannot both be valid: the second POST
of the parent was refused. So:

1. Only the node renewed (`parent_fp` matches): **adopt the node's pair.** Bump the
   owner epoch to the node's and clear any `awaiting_writeback` mark.
2. Only the owner renewed (`parent_fp` ≠ row, node epoch = lent epoch): **keep
   the owner's pair.** Answer `stale`; the node replaces its custody with the owner's
   current pair.
3. Both claim a success (possible only on a provider with a reuse grace window):
   **the owner's pair wins.** The node's is discarded and the node is re-lent the
   owner's pair. Write `credential.renewal_conflict` so it is visible.
4. A write-back with `epoch ≤` the owner's adopted epoch, a different incarnation,
   or from a node whose custody the owner has ended: **refused (`renewal_stale`)**,
   nothing written. This is a negative control in §C.

**The owner never marks a lent family dead on its own `invalid_grant` while a
write-back is possible.** Today an `invalid_grant` tombstones the row
(`GRANT_DEAD_AT_KEY`, `auth_store.py:220`; `_mark_grant_dead` `:3548`) and only an
interactive login clears it. For a lent login the owner records
`awaiting_writeback` instead, until the node writes back or custody provably ended
(R4). Otherwise a node that legitimately renewed overnight would make the Mac say
"sign in again" for a login the node is about to hand back.

### B.5 Storage, confinement, logging and end of life on the node

**Where.** The node row in the node's own `auth.db`, written by slice 1's
`_write_delegated_row` (`client.py:2648`) with `mesh_origin` provenance. That keeps
**one** refresh path, with all of the PR-24 and reuse-detection defences, instead
of a second refresher. It is never selected as a *local* login. Slice 1 already
keeps delegated rows out of local resolution (`store.py:374` `_delegated_for`,
`auth_store.py:3638` `delegated_credential_ids`).

**Encrypted at rest.** The refresh token field of a delegated row is stored
**sealed** under the node's secret-store master key, using a dedicated derived
record key (`secrets/crypto.py:117` `derive_record_key`, the same primitive class-2
copies are re-sealed with, `sync.py:2292-2330`). It is unsealed only inside the
fence-gated POST, through the fence seam, so `auth_store` stays mesh-unaware.
**Honest limit:** on a headless Linux node the master key is the 0600 keyfile tier
(`secrets/keys.py:1-20`: "equivalent to plaintext `.env` against an attacker who
specifically reads the key file"). Sealing raises the bar from reading one file to
reading two files. The real protection is bounded custody plus the
confinement below.

**Confinement (prerequisite, not yet built).** `auth.db` is **not** in
`_CONFIG_SECRET_SUBPATHS` (`tools/confinement.py:787-795`), and `*.db` is
deliberately excluded from the kernel deny (`:810-812`). The delegation design
named this as slice 1's prerequisite (its §3.6), but it did not land. B adds the
exact path `auth.db`. The proof is a read attempted from a confined agent shell
before and after, not a reading of the list.

**Never in transcripts or logs.** The material crosses only inside the encrypted
link record (`network/wire.py`), in `refresh_take` / push answers and the
write-back's last frame. The audit writer drops `refresh_token` / `access_token` /
`token` by name (`audit.py:744` `FORBIDDEN_DETAIL_KEYS`), and every new event below
allowlists ids and counts only. Every new `logger.*` call must be reviewed line by
line for the same property; a test greps captured logs for the fake IdP's tokens.

**End of life: what wipes custody** (`refresh.stop` `:449` → row wipe
`auth_store.py:3798` `wipe_delegated_material`; written *before* any report):

| Trigger | Mechanism | Latency |
|---|---|---|
| Panic (raised or received) | the `relay.py:3635` `set_trust` / `:3747` `panic` latch → a wipe hook in the credentials slice; the fence also refuses on `_NETWORK_UNTRUSTED` | immediate on the node that hears it |
| Share revoke | `should_borrow` false on the next pull or the owner's answer (`client.py:896-907`); the owner also ends its lend record | next contact or pull (the node keeps renewing at most until then, ≤ custody end) |
| Membership loss | `_self_membership_active` false (`client.py:1437`) | next check (every POST) |
| 24 h without owner contact | R2.2 (two clocks). **Authority ends**: nothing is served or renewed past this point. If the node has **not** renewed since the last contact (`epoch == lent epoch`), the row is wiped. If it **has**, the newest pair moves to the **return-only copy** below. The check runs on every gate and at relay start | ≤ 24 h |
| Return-only copy written back, or 7 days old | see "Return-only copy" below: deleted the moment the owner acknowledges adoption (or refuses it as stale), hard cap 7 days from the last owner contact on both clocks | write-back, or ≤ 7 days |
| Owner signs out or signs in again | the owner's row tombstone or incarnation change rides the lifecycle generation (`network/credentials/lifecycle.py`, `credential-lifecycle-v1`); the node drops custody **and** any return-only copy on a mismatch. On `lop logout` the owner **also revokes at the provider** (§D D3, below) | provider revoke: immediate where an endpoint exists; node: next contact |
| The node's own `invalid_grant` | slice 1's family-moved stop: wipe, report, never retry (`client.py:2909`) | immediate |

**Panic, share revoke and membership loss drop the return-only copy too** (§D D4),
at once. The cost, stated: if the node had renewed, the Mac's copy is spent and the
Mac needs "sign in again". Revocation wins over convenience.

**Return-only copy (§D D4).** The copy exists only because dropping a rotated pair
before write-back would sign the Mac out.

- **Shape.** The same sealed row, with `state = "return_only"` in the slice-1
  delegation document (`refresh.py` `STATES` gains it). Inside the fence it is
  **terminal for serving and renewing**: `refresh_fence` refuses every POST for it,
  and the node's grant path never serves its access token. The only reader is the
  write-back sender (§B.4), which sends it on the first owner contact.
- **When it is deleted.**
  - after the owner's adoption ack, or after a `renewal_stale` refusal (the owner
    has something newer);
  - at once on panic, share revoke or membership loss;
  - at **7 days** after the last owner contact, on both clocks. This is a hard cap.
    It is enforced at every relay start and on the lifecycle tick, so it does not
    depend on a later serve.

**Provider revoke on logout (§D D3).**

- `lop logout` (`cli.py:16094` `logout_command` → `auth_store.py:2894`
  `delete_credential`) of a login that is or was lent calls a new optional registry
  hook, `revoke_token=`, beside `refresh_token=` (`providers/registry.py:473`,
  `:490`), *before* deleting the row. Its outcome is recorded as
  `credential.provider_revoked {credential_key, provider, outcome}`.
- **Neither `oauth/anthropic.py` nor `oauth/radient.py` has a revoke call today**
  (checked: no `revoke` in either module). The hook is implemented for a provider
  only against a documented endpoint (RFC 7009 or the provider's own). It is
  verified against a throwaway test account or the fake IdP, **never** the
  operator's real login (§D B0).
- Where no endpoint exists, logout still recalls custody at the next contact and
  says so in plain words: "node2 can keep using this login until it reconnects or
  for 24 h; to end it now, sign out of Anthropic on claude.ai".

### B.6 Settings, cap and UX

- `network.credentials.lend_renewal` (bool, default **on**: the operator's
  standing C5 decision of 2026-11-13; the settings row goes beside
  `offline_hold_s`, `settings_io.py:5604-5625`). Off means no new lends, and live
  custody is recalled at the next contact.
- `network.credentials.renewal_authority`: `device` (default) or `signed` (slice
  1's per-delegation operator signature).
- Cap string `credential-renewal-v1` beside `BROKER_CAP_STRING` (`types.py:82`). An
  older node never receives material, and an older owner never sees a claim.
- **No new prompt and no new command.** The lend rides the next grant to an eligible
  node. Status rows gain one line each way:
  - node: "Renewing your Anthropic login for **Damian's MacBook** while it is away, until 06:12
    UTC tomorrow".
  - owner: "Lent to **node2** for renewal". After a write-back: "**node2** renewed your
    Anthropic login 3 times while this Mac was away (last 06:41 UTC)".

### B.7 Audit receipts the owner can see

New `EVENT_KINDS` with `DETAIL_KEYS` (`audit.py:208-218`, `:576-590`), ids and counts only:

| Event | Written on | Detail |
|---|---|---|
| `credential.renewal_lent` | owner | `credential_key, act, sub, epoch` |
| `credential.renewed` | node, per POST | `credential_key, act, sub, epoch, cause` (`lease_lapsed`) |
| `credential.renewal_returned` | owner, on adoption | `credential_key, act, sub, epoch, renewals, last_at` |
| `credential.renewed_by_peer` | owner, one per node receipt replayed from the write-back | `credential_key, act, sub, epoch, at` |
| `credential.renewal_conflict` / `renewal_stale` | owner | `credential_key, act, sub, epoch, code` |
| `credential.renewal_dropped` | node | `credential_key, act, sub, epoch, cause` (`panic`, `revoked`, `membership`, `expired`, `signed_out`, `family_moved`) |

On adoption the owner also posts **one** Notifications item, keyed per
`(node, key)` so a repeat updates it: "node2 renewed your Anthropic login at 03:12 and
06:41 UTC while this Mac was away". The owner is asleep while the renewals happen,
so receipts reach it at write-back. The node's own log has them immediately.

### B.8 Security delta versus today (for the PR body)

Today, after #3922, a node holds only a **bearer**: the access token the Mac sent,
good until that token expires (Anthropic ~8 h, Radient ~55 min) and never longer
than 24 h. It cannot renew anything. With this change, one chosen node per login
also holds the **renewal token** for that login:

- **A node cut off from the Mac** keeps Anthropic and Radient working for up to 24 h
  after it last heard from the Mac, instead of until the current token expires.
- **A stolen or compromised node.** Today it exposes about 8 h of Anthropic (one
  access token) and about 55 min of Radient. After this change it exposes **a
  renewable login for up to the family's life unless revoked at the provider**
  (Anthropic: 30 days from sign-in). Within lop, custody still ends on panic, share
  revoke, sign-out reaching the node, or 24 h of silence (7 days for a return-only
  copy). But a renewal token copied off the disk keeps working until the provider
  ends the family. Panic wipes the node's copy; it cannot un-copy a stolen one.
  **Remedy: run `lop network panic`, then sign out at the provider (`lop logout
  <provider>` on the Mac revokes there where an endpoint exists; otherwise sign out
  on claude.ai or radient).** This paragraph and the remedy go verbatim in **B1's PR
  body** (§D D7).
- **Signing his Mac out** is possible in two cases:
  - the node renews and is then never heard from for 7 days (the return-only cap);
  - a panic, share revoke or membership loss drops a renewed pair before write-back.
  
  The Mac never presents a token the node may have rotated (R3/R4). The node never
  presents one the Mac rotated (claim plus `awaiting`). Both providers are treated as
  ending the family on a token presented twice.
- **The Mac may wait** for a lent login when the node holds a lapsed lease and is
  unreachable at the moment the Mac needs to renew. Its status says "Waiting for
  node2 to hand back your Anthropic login". There is a one-click "sign in again here"
  way out, and the wait clears on its own on reconnect or at the 7-day return cap.
  Before this change, the Mac never depended on a node.
- **Authority is automatic.** A lend needs the owner device's key on the link and
  the operator's standing setting, not a per-delegation operator signature as
  slice 1 required. That signature remains available as `renewal_authority: signed`.

---

## C. PR split, order and tests

| PR | Scope | Product lines (est.) | Depends on |
|---|---|---|---|
| **A** `feat(mesh): hold borrowed GitHub and MCP logins for the offline hold` | A1–A6 | ~150–250 | stacked on #3922 (coder hosted on node2 by the manager, 2026-11-14) |
| **B0** evidence only, no code, **no live token probe** (§D B0) | cite the existing evidence (§B.1: the auth store's Radient measurements, public Anthropic reports) on B1's PR. Measure the wake-to-link-up distribution (`n`, median, max) that sizes `W`; this needs no tokens | 0 | — |
| **B1** `feat(mesh): lend the renewal token to one node; owner claims before it renews` | owner side: R1 eligibility and automatic lend over `refresh_take`; `lent` state in `_delegated`; owner fence plus local op `credential_claim`; write-back adoption with the `parent_fp` CAS and the conflict rule; `awaiting_writeback` instead of a tombstone; settings and cap; owner audit events and notification. The node only *stores* (no renewal yet), so B1 is safe alone Also in B1: the lop-native eligibility check (D5); the `awaiting_handback` status and copy (D2); `revoke_token=` on logout (D3). **The PR body carries §B.8's stolen-node paragraph and remedy verbatim (D7).** | ~350–400 | A (copy patterns), B0 for `W` |
| **B2** `feat(mesh): a node renews a lent login while its owner is away` | node side: the R2 gate in `refresh_fence`; the lease and `awaiting` fields in `refresh.py`; sealed refresh field; `auth.db` confinement; the wipe hooks table (§B.5); node audit events; status copy Also in B2: the `return_only` state with its write-back-only reader, the 7-day cap and the panic, revoke and membership drops (D4); moving the renewer through the owner (D1). | ~350–400 | B1, renew-on-reconnect's link-up hook |

Each is C5 (credentials and the security boundary). The design doc for the gate is this
file, read with `mesh-refresh-delegation.md`.

**Tests** (each PR runs `make test-affected` plus the files named; the merge gate runs the full suite once):

- **A:** extend `tests/unit/network/test_credentials_offline_hold.py`:
  - an MCP grant and a GitHub grant are served `held_offline` after `owner_offline`, until `min(token end, 24 h)`;
  - the App token's belt and the owner lender fire at the hold end (fake clock) and **not** at the grant end;
  - panic, revoke and membership loss drop the held GitHub grant and fire an immediate DELETE against the loopback GitHub (`github.py:1407-1420`'s `base_url` seam);
  - an MCP 401 while held ends the hold.

  **Negative controls:** a device-bound key is still not held; a forced refresh is never served from the hold.
- **B1/B2 real path:** two real relays on loopback TCP (`test_credentials_reconcile_link.py` pattern) with a rotating fake IdP that **revokes the family on reuse** (extend `RotatingIdP`, `test_credentials_owner.py:65`). Assert the IdP's `posts` list, not stub tolerance:
  1. Owner connected → owner renews, node renews **0** times (lease held), and the node's copy is updated (push).
  2. Owner relay stopped (laptop closed) → after `L` the node renews when the access token expires. `posts` shows one POST per expiry, all from the node, and `credential.renewed` is written.
  3. Owner relay restarted → write-back → owner adopts (row fingerprint = node's last), exactly one `renewal_returned`, receipts replayed, one notification; the owner's next renewal works with no `invalid_grant`.
  4. **Conflict:** owner renews while the node is partitioned and claims fail. The owner refuses to present (R4), and `posts` shows the owner made none. After the partition heals the node writes back and the owner adopts.
  5. **Claim race:** owner claim → node `standby/awaiting` → push dropped (fault injection) → node does **not** renew past `L` (no POST) until the push arrives.
- **Negative controls (B):**
  - **panic** on the node mid-custody → the row is wiped, the next fence refuses, and no POST follows;
  - **share revoke** → wipe at next contact;
  - **custody past 24 h** (fake sleep-inclusive clock), not renewed → wipe and no POST; renewed → `return_only`, with no serve and no POST. The next contact writes it back, the owner adopts, and the copy is deleted;
  - **return-only copy past 7 days** (each clock stepped alone) → deleted at relay start and on the tick, with no contact needed;
  - **panic, revoke or membership loss with a return-only copy** → deleted at once, nothing is sent, and the owner shows "sign in again";
  - **R4 visibility** → with the node unreachable on a lapsed lease, the owner's status row reads "Waiting for node2 to hand back your Anthropic login". The owner makes no POST (`posts` empty), and the state clears on reconnect;
  - **imported (non-native) login** → never lent (D5);
  - **logout** → the fake IdP's revoke endpoint is called once, and the node drops custody at the next contact (D3);
  - **owner sign-in again** → write-back refused by incarnation;
  - **stale write-back after adoption** (replay the earlier `done` frame) → `renewal_stale`, row unchanged, and the IdP sees no POST of the old token;
  - **second node** asking for a lend → access-only;
  - **logs and transcripts:** captured logs and the audit file contain none of the fake IdP's tokens.

---

## D. Decisions 2026-11-14

Decided by the lopdev manager (session `79bddc262663`) within the operator's 2026-11-13
approval; folded into §B above. Each item names where it landed.

- **D1, was Q1: automatic, no picking.** The renewer is the node that currently holds
  the lent login (the one running turns on it), with one renewer per credential at a
  time. Any move goes through the owner (claim back, then lend). → §B.3 R1.
- **D2, was Q2: no owner-first.** Both providers are assumed to revoke the whole
  family until proven otherwise. R4 is accepted: the Mac waits on a node that holds a
  lapsed lease, with the "sign in again here" escape. It is made visible: the Mac's
  status names the node, "Waiting for node2 to hand back your Anthropic login"
  (`awaiting_handback`). → §B.1, §B.3 R4.
- **B0: no live measurement with the operator's real tokens.** A reuse probe could sign
  him out. The evidence is limited to the auth store's existing Radient measurements
  and the public Anthropic reports; the design is defensive. → §B.1, §C.
- **D3, was Q3: revoke at the provider on logout.** It goes through a new
  `revoke_token=` registry hook, implemented per provider only against a documented
  endpoint and verified on a throwaway account or the fake IdP. Radient is one
  family with no reuse grace; Radient server changes are out of scope. → §B.5.
- **D4, was Q4: keep a sealed return-only copy.** It is never used to serve or
  renew. It exists until it is written back, then it is deleted, with a hard cap of 7
  days. It is dropped at once on panic, revoke or membership loss. Why: dropping a
  rotated pair before write-back would sign the Mac out. → §B.5.
- **D5, was Q5: lop-native sign-ins only** (`native_login_incarnation`). → §B.3 R1.
- **D6, was Q6: MCP refresh tokens are out of scope.** A holds MCP to its access
  token's end.
- **D7, §B.8: the stolen-node exposure goes in B1's PR body in plain words.** That
  is up to the family's life unless revoked at the provider, versus about 8 h
  today, plus the remedy: panic, then provider sign-out. → §B.8, §C.

**Reconciliation note (architect, for the manager to confirm).** D2's "auto-clear
on reconnect or at 24 h" and D4's return-only copy conflict at the 24 h point.
Suppose the node renewed and holds a return-only pair at hour 24. If the Mac then
resumes on its own, it presents its spent token. Under D2's family-revocation
assumption that signs the Mac out **and** kills the node's newer pair, which is
exactly the loss D4 exists to prevent. The Mac cannot tell from its side whether the
node renewed. So this doc makes the wait clear automatically on reconnect (write-back
or a claim answer) or at the **7-day** return cap, with "sign in again here" always
available (§B.3 R4). If 24 h is preferred anyway, the cost is a possible Mac sign-out
whenever the node renewed and stays unreachable past 24 h. It is a one-line
constant either way.

## E. Risks to watch during rollout

- An owner waiting on an unreachable node (R4) reads as "Anthropic broken on my
  Mac" (mitigated by the D2 status naming the node and offering the escape; it clears
  on its own at the latest at the 7-day return cap). Watch for `credential_claim` timeouts in the owner audit, and make sure the
  copy names the node and the way out.
- `L` too short means early node takeovers during short link outages, which only
  costs claims later. `L` too long means node sessions stall for up to `L` when the
  lent Radient token was nearly expired at sleep. Size it from the measured
  link-outage distribution once B2 ships, not by feel.
- Lost write-backs: count `renewal_dropped{cause=expired}` with renewals > 0.
  Each one is a guaranteed Mac sign-in.
- Clock steps: every custody and lease bound is on two clocks (wall and
  sleep-inclusive), as in #3922. Tests must step each one alone.
