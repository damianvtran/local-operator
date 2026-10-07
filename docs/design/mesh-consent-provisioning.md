# Design: consent-then-propagate — provisioning, sync and repair under one approval

Status: **proposal for implementation.** Author: architect.
Base: `origin/main` @ `7d612a2d` (`v0.68.0`).
Siblings this builds on, in their order of authority: `mesh-credentials.md` (classes,
ownership, broker), `mesh-transport-identity.md` (pairing, links, epochs, capabilities),
`mesh-remote-onboarding.md` (the approval record and the runner), `mesh-rolling-updates.md`
(evaluated in §7, not adopted).
Folded in: the operator's addendum of 2026-10-06 — the model is **three lifecycles**
(provision at approval, sync on change, repair without a human), and the carrier question
is answered in §7.

The operator's directive, verbatim:

> "we shouldn't require our users to create Github apps, any user of Local Operator should
> be able to propagate their credentials to other devices. As long as they approve the
> remote request, all the necessary data should be cloned to any remote nodes that are
> approved on the network."

Direction read from it, and the design below takes it literally where it can:

1. **Approval of a device on the network IS the authorisation.** No per-credential
   `share` commands in the default path; the approved node is provisioned automatically.
2. **The necessary data is cloned** to approved nodes — bounded per class (§2), because
   some classes cannot be cloned without breaking (rotating OAuth) and some must not
   (device-bound, host-local).
3. **No GitHub-App creation as the taught route.** The App survives as an optional
   stronger implementation (§3).

All file:line references are against the base above, read via a worktree of
`origin/main` (not the shared checkout's working tree).

---

## 0. The answer in one page

The approval that admits a device (a signed `device_onboard` record, or the pairing
admission) becomes a **provisioning transaction**: the runner reconciles the node onto
the things it needs to be useful — agent/team definitions, MCP server definitions, git
identity, credential holders, and per-class credential material — with the receipts
discipline the runner already has. The default path needs no per-credential commands;
`lop network credential share/revoke` remain as adjustment surfaces (narrow, unshare,
own), never as prerequisites.

Three lifecycles, one machine:

- **Provision (once, at approval).** New runner step after `grants`; owner-side
  placement writes + capability enablement + copies; node-side applications; receipts
  (§1, §2, §4).
- **Sync (continuous, on change).** Per-key generation counters on the owner; a tiny
  *announce* push rides the **existing definitions syncer tick**
  (`definitions.add_tick_step`, `definitions.py:1999`); the member *pulls* the value on
  announce or on first use after a generation mismatch. Reachable-member latency budget:
  ≈ 60 s floor + 15 s tick + one round trip — "within about a minute", the same number
  the definitions syncer already documents for itself (`definitions.py:2032-2043`).
  In-flight work never blocks on the sync path (§5).
- **Repair (on failure, no human where the source allows it).** Re-acquire through the
  owner; the owner's own refresh path runs where it exists (provider OAuth, PAT re-read);
  an interactive-only source (an expired `gh` login) raises the existing
  `credential_repair` row — one product action, not a terminal command. Retry-with-
  backoff for transient classes, fail-fast for revoked ones (§6).

The one invariant this design **alters**, named plainly in §8: *"token material … never
touches the receiving device's disk"* (`mesh-credentials.md` §0). For the copy classes
only, material lands on the receiving device — only ever inside that device's own
credential storage, per class (§2.1): a class-2 copy is re-sealed into the node's
encrypted secret store, and a class-4 copy lands as the node's own credential record —
the same 0600 `auth.db` row a locally-entered static key gets (the tree's existing
no-keychain posture). Never a world-readable file, never the owner's master key. The threat delta
is stated there, with the ceiling: **a leaked copy cannot be selectively revoked; the
final remedy is rotation at the source**, and the design says so on every surface that
can end a share.

The model in five lines (for the report and the release notes):

1. One approval provisions the node: definitions, MCP defs, git identity, placement
   grants, and the per-class credential set — no per-credential commands.
2. Rotating logins stay brokered (owner-only refresh); static keys and secrets become
   copies into the node's own credential storage (class 4: the node's own 0600 rows,
   the local-login posture; class 2: the node's encrypted store); device-bound/
   host-local items refuse.
3. Copies stay fresh via generation counters + announce-over-the-existing-tick + pull,
   ~1 minute to a reachable peer, and in-flight work never stalls on the sync path.
4. Repair re-acquires automatically where the owner's source allows; the one interactive
   case raises the existing repair card.
5. Un-approve = stop serving + wipe reachable copies + rotate what leaked; the ceiling
   ("a copied secret ends at the source") is stated, and the secret-copying slices hold
   for a C5-class review before they land.

---

## 1. What moves on approval — the provisioning transaction

### 1.1 The approval event, precisely, and where the transaction binds

Two surfaces already implement "a device is approved":

- **Pairing admission.** The both-ends screen ships as one sealed `net_pair_offer` record
  (owner → joiner, right after `welcome`), rendered through one module
  (`local_operator/network/credentials/offers.py`), with a reduce-only `[t]` step on the
  owner; admission grants exactly the intersection of the decision and the offer — one
  `credential.placement` per key plus the `broker_credential` capability, "the same pair
  the share verb makes" (`mesh-credentials.md` §2.3, as-built 2026-09-29).
- **Remote onboarding approval.** The `device_onboard` record
  (`local_operator/network/approvals.py`; schema `mesh-remote-onboarding.md` §2.2) is
  answered with the operator gesture (`approve`, `approvals.py:1123`), and executed by
  the runner (`local_operator/network/onboard.py`, `execute_approval:2282`) as the step
  sequence `invite → pre_read → install → join → anchor → relay → grants → verify`
  (`step_invite:1092` … `step_verify:2143`), gated before **every** credentialed step by
  state + expiry + signature (`_gate:1065`, `approvals.verify_for_run:1593`).

The transaction binds to **both**, because both are the operator's approval of the same
thing — a device — and the directive does not distinguish them. Concretely:

- **Onboarding path (primary):** a new runner step, `step_provision`, runs after
  `step_relay` (`:2068`) and `step_grants` (`:1965`), before `verify`. The order is
  load-bearing: `relay` precedes `grants` ON PURPOSE (F7b, `onboard.py:75` — a relay
  answering on the node executes the grant write from its own loaded build, and a
  pre-run relay's refusal does not fall back; pinned by
  `test_the_relay_step_runs_before_the_grants_step`), and the transaction sits after
  both so the trust decisions exist and any write it makes through the node's relay
  meets the run's relay rather than a pre-run one. Receipts append per action
  (`approvals.append_receipt`, `approvals.py:1467`); the step is resumable like every
  other (`mark_failed` → `begin_run` retry, `approvals.py:1509`, `:1369`).
- **Pairing path:** the offer's per-kind defaults are the one thing that changes (§1.4);
  the admission grant code path stays as built.

Neither surface mints new authority: the operator's signed decision is the authority
(`SELF_DECIDED_SCOPES` precedent, `network/types.py:475`), and every write the
transaction makes is one the owner's own device is entitled to make today — it just
stops requiring that the operator type a per-credential command first.

### 1.2 The primitives: what exists, what needs a small extension, what is new

| # | Primitive | Status in tree | What the transaction does with it |
|---|---|---|---|
| 1 | **Placement grants** (owner's document: who may borrow what) | **Exists.** `PlacementDocument.grant` (`credentials/placement.py:438`), owner-only write rule (`:438-498`), merge/persist (`:532`, `:596`); owner serves it on `net_broker` `kind: placement` (`credentials/owner.py:1177`); borrower pulls (`credentials/client.py:758`). | Write the per-class default holder rows for the approved device at approval time (instead of the operator running `credential share` per key), then push the document to the node within the run (a push, not a wait for the next pull) so the node's first `lop network credentials` is already true. |
| 2 | **Capability enablement** (`broker_credential` on the member row) | **Exists.** Grantable vocabulary `GRANTABLE_CAPABILITIES` (`network/types.py:444-458`), `member grant` CLI, and the admission grant already writes "placement + `broker_credential`" as one pair. | Include the pair write in the transaction (owner-side), so a run-only approval reaches the same state a pairing admission reaches. |
| 3 | **Definitions push** (agents, teams) | **Exists and continuous.** `definitions.push_to_peer` (`definitions.py:1423`), `DefinitionsSyncer` thread (`:2019`), `net_definitions` handler (`:1870`), create-path reconciliation (`local_operator/server/utils/desktop_mesh.py:520`, `create_on_peer`, forwards a target "because the relay reconciles the definition on the way"). Payloads are versioned and **non-credential** ("one versioned, non-credential payload", `definitions.local_bundle:708`; credential-shaped rows are withheld, `_withheld:571`). | Reuse as-is. The transaction forces one push for the approved node (create-path semantics) so the node is usable the moment onboarding returns; the tick keeps it current afterwards. **Keep the non-credential invariant** — values never ride the definitions bundle. |
| 4 | **MCP server definitions push** | **Exists and continuous.** `mcpdefs.push_to_peer` (`mcpdefs.py:1104`), tick step `mesh_tick_step:1350` riding the definitions thread, state with per-ref resolution ("the keys to set", `state_rows:782`; `_reference_present:844`). | Reuse as-is; then **use its refs as the transaction's needs-list**: the refs the bundles declare (`ref:<NAME>` states) are exactly the secret names the node will need, and §4 makes them the default copy set. Order: definitions + MCP defs push first, then credential provisioning, then one verification read (`network mcp state`) whose "keys still needed" must be empty for the covered set. |
| 5 | **Git identity** | **Check exists, seed is new (small).** `readiness.git_identity_fact` (`readiness.py:256`) reads `user.name`/`user.email`; the readiness row and its remedy (`:1100-1140`) currently only *suggest* fixing the node. | Seed the node's global git identity from the owner's (`readiness.py:1125` shows the intended pairing) via the transport's `run` — small, visible on the card, and it makes the first commit from the node carry the operator's identity instead of dead-ending at the next `ready`. |
| 6 | **Credential material (copies)** | **New.** The broker never writes material to a borrower (`mesh-credentials.md` §10: nothing in the implemented list writes `secrets/`, `auth.db` or the keychain), and the design-text `replicate` field of `placement.json` (`mesh-credentials.md` §2.1, `:231`) was never built. | §4 and §5 define it: per-class copy decision, generation counters, node-side encrypted storage, wipe/revoke. |
| 7 | **Repo / workspace seeding** | **Not needed for the default path — say so rather than build it.** Forge work needs no local clone to push (remote URLs; the helper serves `https://github.com` for the allow-listed repos, `github.py:922`); session state carry-over for moves already exists as the sync copy set (`sync.py:13-22`, "the copy set is the spec of what a session directory may hold"). | Nothing. If a future workstream wants offline repo priming, it is a separate slice with its own bounding; the transaction does not clone repositories. |
| 8 | **Invite / join / anchor / relay** | **Exists** (`onboard.py` steps above). | Untouched; the transaction slots after them. |

The transaction's own contract, stated so slices can be reviewed against it: **every
action is a write the owner could already make by hand; the transaction only removes the
hand.** No new authority, no new trust root, no wire op that a build without this design
would have to compose a refusal it did not ask for (the additive-rows rule,
`mesh-transport-identity.md` §12.3).

### 1.3 "The necessary data", per device role

The directive's "necessary" is bounded by the member's kind and role, which the network
already models (`MemberRecord.kind` ∈ `device | pool`; roles `read | drive | admin`,
`mesh-transport-identity.md` §7.1; `ROLE_CAPABILITIES`):

| Device | Gets |
|---|---|
| `device`, role `drive` (the operator's own paired machines — the directive's case) | Everything: definitions + MCP defs + git identity + placement grants + the per-class credential set (§2) for keys it does not own locally. |
| `device`, role `read` | Definitions + MCP defs + git identity. No credential grants: `read` holds no `prompt`/`borrow` path, and lending to a viewer-only device would widen authority for no work. |
| `device`, role `admin` | As `drive` (admin's role set is a superset, `mesh-transport-identity.md` §7.1). |
| `pool` member | **Nothing.** Pool members declare no credentials, start with `holders: []`, and are excluded from every credential path by class (`mesh-credentials.md` §2.1, §6; `mesh-rolling-updates.md` skips pool members the same way). The exclusions in §1.4 are structural, not policy. |

Two existing scopes stay exactly as they are, decided **on the node** in `step_grants`:
`approve` ("answer approval prompts for sessions here") and `unattended` ("start
sessions here without approval prompts") — the node's row, the node's say.

### 1.4 The defaults this flips, and the two exclusions that do not move

Today's join-time defaults are a closed table (`credentials/offers.py:83-92`):

| kind | today | under the directive |
|---|---|---|
| `oauth-rotating` | `True` | `True` (unchanged) |
| `api-key-static` | `False` | **`True`** — a static key is the class that *works* from a second device; the reason it was off ("a permanent capability increase", `offers.py:24-26`; `mesh-credentials.md:328`) is answered by the approval gate + wipe/rotate, not by making the operator type `share` per key. |
| `mcp-rotating` | `False` (v1 scope-surface caution, `mesh-credentials.md` §2.3 as-built bullet 2) | **`True`** — the access token is already brokerable (`owner._resolve_mcp:652`), the grant never moves (`_oauth_refresh_lock`, `mcp/auth.py:3716`, stays host-local), and the MCP defs push already reports the server set the node will run. |
| `github-app` | `False` ("closed hardest", `offers.py:87-91`) | **`True`** when the adapter resolves *any* source (§3); the "one App covers every designated repository for every command" concern is bounded by the repository allow-list (`network.credentials.github.repositories`, `github.py:92`) and by the helper's path check (`github.py:906`). |
| `radient` | never auto-offered (`NEVER_AUTO_OFFERED_PROVIDERS:98`) | **Offered by default, reduce-only, for `device` members of the operator's own networks; pool exclusion unchanged.** This is the one default I flag for review (§10 Q4): the bearer carries organization-write authority (`mesh-credentials.md` §1.1), but the operator's directive covers exactly this kind of login, and a node that cannot publish an agent cannot do the work it was onboarded for. |

The reduce-only step stays on every surface: the operator sees the list and can drop any
row before signing; a post-approval narrowing keeps the existing `credential revoke`
semantics. The card copy gains one line when a copy-class row is on it: what is copied
lands in *this node's own* credential storage — the same 0600 rows a local login
writes (class 4), or the node's encrypted store (class 2) — and the ceiling sentence
from §8.

Two exclusions do not move, because they are facts rather than postures:

- **device-bound credentials** — the Kimi device id is never served and never copied
  (`github.py`'s sibling rule in `mesh-credentials.md` §1 table row 6b; `types.py`
  `device_bound_refusal:128`).
- **host-local credentials** — the mobile portal password authorises *that host's*
  phone portal from the macOS Keychain (`mesh-credentials.md` §1 row 6); a peer has
  nothing to do with it.

---

## 2. The credential classes under "approval = authorisation"

### 2.1 Re-stated against the current tree

`mesh-credentials.md` §1 is the authority for the classes; re-derived against
`origin/main` @ `7d612a2d`, with the one retirement since:

| # | Class | Stored where (current) | Rotating? | Decision under this design | Why |
|---|---|---|---|---|---|
| 1 | Legacy env credentials (`credentials.env`) | **Retired.** `local_operator/credentials.py` is deleted; nothing reads the file as a credential source; the only survivor is `secrets/legacy_env.py`'s migrate path (`AGENTS.md` §"Credentials and the encrypted secret store", `AGENTS.md:3392-3404`). | — | **Refuse (nothing to propagate).** | The class no longer exists; copying one would resurrect the file the tree deleted. |
| 2 | Encrypted secrets (`~/.local-operator/secrets/`: SQLite AES-GCM + `master.key`, broker daemon over `broker.sock`) | `secrets/store.py:445+`; per-record `updated_at`/fingerprint (`SecretRecord:206-223`, `describe_identity:887`) | No | **Copy, selected (§4).** | The value must exist on the consuming host for `bash` to use it (`mesh-credentials.md` §4.8's own argument); consent-as-authorisation removes the last real blocker and bounds the rest. |
| 3 | Provider OAuth grants (rotating) | `auth.db` → `auth_credentials`, `credential_type='oauth'`; refresh under a local lease (`auth_store.py`, PR-24 history) | **Yes** | **Broker only, unchanged mechanics — new default: the holder grant lands at approval.** | Two hosts racing one rotating refresh token is the measured PR-24 failure; the lease is a local row and cannot span hosts (`mesh-credentials.md` §0, §4 `auth_store.py:147`, `:1777`). |
| 3b | Radient org login (a class-3 login with org-write authority) | As class 3 | Yes | **Broker; default-flipped per §1.4 with the review flag (Q4).** | Person-scoped org bearer; least authority was `scope: session`, kept. |
| 4 | Provider API-key logins | `auth.db`, `credential_type='api_key'`, `source="login"` | No | **Copy, selected — the same policy machine as class 2 (§4.2: needs-list ∪ `sync` marks).** | Static keys work from anywhere; copying them is what makes remote work survive the owner going offline. Broker remains available per key for operators who prefer TTL-bounded lending. |
| 5 | MCP OAuth grants (rotating) | `auth.db`, `provider='mcp-oauth'`; refresh lock beside `auth.db` (`mcp/auth.py`) | **Yes** | **Broker the access token; the grant never moves and never copies.** | The grant's refresh lock is host-local and a new grant needs a loopback callback (`DEFAULT_CALLBACK_PORT = 33441`, `local_operator/mcp/auth.py:111`); the access token is already served by `owner._resolve_mcp:652`. |
| 6 | Mobile portal password | macOS Keychain, service `lop-mobile` — the repository's only keychain use | No | **Refuse.** | Host-scoped by construction; a peer has nothing to do with it. |
| 6b | Kimi device-bound grant | Class-3 rows plus `<config>/kimi/device-id` | Yes | **Broker the access token; never the device id, never a copy.** | A borrower replaying the owner's device fingerprint is presenting an id the provider did not issue it (`mesh-credentials.md` §1 row 6b). |

**Mechanism mapping, one line each:** classes 3/5(+6b) = broker (owner-side refresh, TTL
bounds, revoke by unshare); classes 2/4 = copy (node-side store, generation sync, wipe +
rotate); classes 1/6 = refuse; Radient = broker with its own default story.

### 2.2 The honest constraint, stated once here and once in the guide

A borrowing device **cannot bound a secret to one process or one session**. There is no
per-session secret far side: the guide already accepts it — "the DEVICE is the trust
unit … any process or session on the borrowing device (same user) can use it, because
there is no per-session secret on a node for the design to bound"
(`local_operator/guides/network/GUIDE.md:606-610`), and the T7 disclosure ships the same acceptance for the
App (`mesh-credentials.md` §14, `:1350-1356`; `github.py:28` points at the receipt/
guide copies). Everything below inherits that: a copy is readable
by whatever the node's own consent machinery admits, and the *node's* honest limits are
the copy's limits.

What follows from it, for copies specifically:

- The far-side storage discipline is not decoration, it is the whole bounding story:
  values land **only** in the node's own credential storage. A class-2 copy is
  re-sealed under the node's own master key (`secrets/store.py:755-825` is the re-seal
  path; `keys.py:39` the `0600` file mode; the broker socket `0600` inside `0700`,
  `secrets/broker.py:197,238`); a class-4 copy lands as the node's own credential
  record — the 0600 `auth.db` row a local login writes, the same store and posture
  every locally-entered static key already has (review round 1, Q-6: this paragraph
  said "encrypted store" for both classes, which only the class-2 half has).
  Never the owner's `master.key`, never a plaintext env file, never another host's git
  credential helper (the F1 close already guarantees that for the forge path,
  `github.py:30-41`).
- Retrieval on the node is the node's own problem and stays exactly as it is locally:
  the node's broker gates reads by the node's process-ancestry consent
  (`secrets/peer.py:476` `authorize_by`, `:536` `authorize` — registered sessions and
  their live descendants), which now *works*, because the processes asking are on the
  node. This is §4's load-bearing observation.

### 2.3 Revocation semantics, per mechanism

| Mechanism | un-share / un-approve does | Ceiling (what cannot be undone) |
|---|---|---|
| Broker (classes 3/5/6b, forge default) | Stops new grants immediately; drops the borrower from `holders`; the broker refuses on the next frame. **Measured (PR #1513 QA): new borrows refused 2.3 s after `credential revoke`; a lent grant stops at the borrower's next re-ask.** | A bearer already picked up lives until its own expiry at the provider (grant TTL ≤ 900 s bounds only a well-behaved borrower; a copy of the bearer stops at the token's expiry — three latency statements, `mesh-credentials.md` §3.7). |
| Copy (classes 2/4, forge opt-in) | Sends a **wipe notice** for every copied key with this owner's provenance; the node deletes and acks. Reachable: immediate. Unreachable: queued to the next contact (§5.5). | A copy already exfiltrated from the node cannot be recalled. **Rotate at the source** — that is the only ending, and every surface that ends a copy says so (the `member rm` receipt, the network guide, `credential revoke` output). |
| Refuse (1/6) | Nothing to do. | — |

The one sentence the design commits to, on every surface that ends a copy: *"This
removed the copies it could reach. A copy that already left that device can only be
ended by rotating the secret at its source."*

---

## 3. Forge access without an App

### 3.1 What exists today, and the chore it imposes

The `github` credential is App-only (`credentials/github.py:1-56`): the owner stores
`GITHUB_APP` (`{app_id, installation_id, private_key}`) in its secret store, mints
1-hour installation tokens narrowed to `network.credentials.github.repositories` and
`{contents: write, pull_requests: write}`, delivers `GH_TOKEN`/`GITHUB_TOKEN` plus a
github.com-scoped credential-helper close (F1), and revokes at the window end
(`GithubLender`, `:500-722`; `revoke_installation_token:829`). Device-scoped by
construction; session scope refused by name. And **step 0 of that design is unresolved**:
"the App does not exist yet … push and PR-write are unavailable until a GitHub App
exists — a short one-time setup" (`mesh-credentials.md` §14; `GUIDE.md:604-649`).

That one-time setup is the chore the operator rejected. A user who already runs `gh`
has done the real work long ago; the product just never looked there.

### 3.2 The replacement: broker the owner's existing forge auth

A **source ladder** on the owner, resolved at serve time, strongest first:

1. **`GITHUB_APP`** — if configured, today's mint path runs unchanged (keep it: it is
   the optional stronger implementation, §3.5).
2. **A stored PAT-class secret** — a `GITHUB_TOKEN`-class entry in the owner's encrypted
   store. A **fine-grained PAT** scoped to the designated repositories with an expiry is
   the shape the guide should teach when the user wants a durable, revocable source.
3. **The owner's `gh` CLI login** — the zero-setup route: resolve the token the owner's
   own `gh` already holds, by asking `gh` itself on the owner's machine, at serve time
   (the same ambient user the broker runs as). Nothing new is stored; the operator's gh
   tooling stays the source of truth, which is precisely "refresh through the owner's
   existing local code path".

Everything downstream of the token is **reused, not rebuilt**: the wire grant keeps the
`net_broker` shape and the `github` key; delivery keeps `git_env_for_token`
(`github.py:746`), the F1 helper close, `borrowed_git_env`'s fetch-on-use (`:842`), the
helper core with its protocol/host/path checks (`:887-954`), and the repositories
allow-list (`:169-204`). What changes is one function: `owner._resolve_github`
(`owner.py:883-987`) gains the ladder; the App arm keeps its mint+revoke pair, the
token arms return the token with `refreshed: false` and no server-side revoke handle.

### 3.3 Lease, refresh, revocation, and the owner offline — decided

- **Lease.** Unchanged: device-scoped grants, `grant_ttl_s` (900 s, `credentials/__init__.py:47`),
  delivery per command. The grant is attribution + window; the device is the trust unit.
- **Refresh.** The owner's path is the only path (R14/R15 preserved): the broker
  re-resolves at every serve, so a re-login or PAT rotation on the owner is picked up by
  the node's next command with no gesture on the node. `gh`'s own refresh is interactive
  (`gh auth refresh`) and therefore owner-side by nature; when the owner's token dies,
  the owner-side repair row asks *the operator's human* — where only an interactive
  login helps, that is the one human act, and it is a product action (a card), not a
  terminal instruction (§2.9 copy rule).
- **Revocation.** App: `DELETE /installation/token` (exists, idempotent). PAT: revoke at
  the forge (GitHub settings / `glab` equivalent) — the revoke receipt names it. `gh`
  OAuth token: revoke at the forge; there is no per-grant handle on a user token. **No
  new mint-revoke machinery is invented for tokens we did not mint** — the lender's
  registry simply has nothing to track for arms 2/3, and the receipt says why.
- **Owner offline.** New fetches fail with the existing `owner_offline` code (cache
  15 s, retry once on the next call — `mesh-credentials.md` §3.2). Forge work is
  episodic (a push, a PR), so the decided default is **broker-only: offline owner = no
  *new* forge commands**, with the offline sentence. Operators who need pushes while the
  owner sleeps get the **opt-in copy** (per device, `github.copy`): the token lives in
  the node's store under the §4 discipline, syncs on change (§5), and inherits the
  ceiling — GitHub-side revocation or rotation is the only true end. Default **off**,
  because a full user token is the widest-blast-radius thing this design can copy.
  Flagged for review (Q3); the directive says "necessary", and this is the one place I
  read "necessary" as "when reachable".

### 3.4 Scope and narrowing, honestly

An App token can be narrowed server-side at mint. A user token/PAT cannot: the server
will hand the node whatever the owner's token is. The **enforced** bound is therefore
the helper's allow-list (exact `owner/repo`, `_path_allowed:906`) plus the grant
metadata; the guide must say that plainly for arms 2/3 — "the borrowing device can use
this token against any repository the helper is configured to serve, and nothing else
*through lop's own paths*; the token itself is your full login, so treat a copy like a
copy of your login" (T7-style disclosure, `mesh-credentials.md` §14 T7).

### 3.5 The App stays — optional, stronger, never taught

The App keeps its seat: narrower (server-side repository + permission narrowing),
revocable (per-token DELETE), and independently auditable at GitHub. A user who wants
that asks for it; the checklist moves to an appendix-shaped "stronger option" section
of the guide, and the taught route becomes the ladder above. "Never the taught route"
is a docs statement, and the docs are the deliverable: the guide's GitHub section is
rewritten in the slice that lands the ladder.

### 3.6 GitLab: same structure, stated as a position

`glab` gets the sibling adapter, same shape, tracked as its own slice because nothing
exists today (no `glab` credential path anywhere in `local_operator/**`; `glab` appears
only in read-only monitor classification and redaction shapes). The design position:

- **Sources:** owner-side `GITLAB_TOKEN` (env or store secret) or the owner's
  `glab`-stored login; a scoped, expiring PAT is the recommended durable source.
- **Delivery:** `GITLAB_TOKEN` in the child env plus a **gitlab.com-scoped** helper —
  the same F1 close pattern (reset the helper list for gitlab.com, add one brokered
  helper), a sibling of `github.py`'s delivery, not a fork of the broker.
- **Strong mode:** GitLab's revocable-per-token PAT model is closer to the App than
  GitHub's OAuth token is (a scoped token can be minted for the mesh and revoked alone);
  the slice should mint-or-use a dedicated token rather than share the operator's
  primary login where the operator chooses the strong path.
- **Refresh/lease/revocation:** identical semantics to §3.3; `glab` refresh is likewise
  owner-side.

---

## 4. Class 2 restated — why the store was unbrokered, and what becomes copyable

### 4.1 Re-deriving the §4.8 refusal

`mesh-credentials.md` §4.8 refuses to broker the `lop secret` store. The directive
forces the refusal to be re-derived rather than inherited. The three candidate blockers:

**(a) Authorization — solved, but it was never this class's blocker.** The store's
consent is *process-ancestry on one host*: a caller must be a registered session or a
live descendant of one, kernel-attributed (`secrets/peer.py:476`, `:536`; `AGENTS.md`
§Credentials). Approval answers a different question — "may this *device* hold this" —
and it answers it fully. But (a) was never the reason; the doc itself says the
authorization angle was fine because the existing session-scoped `/credential` verb
already writes values on the peer (`mesh-credentials.md` §4.8;
`session/credential_ops.py:1-26`).

**(b) Far-side ancestry — a blocker to *brokering*, not to *copying*.** A live remote
retrieval would have to satisfy the owner's ancestry check for a process it cannot
attribute — impossible across hosts, so the old design refused the wire path. A **copy**
inverts the question: the value sits in the node's store, and the node's own broker
enforces the node's own ancestry model for processes on the node. The ancestry machinery
does not need to reach across hosts; it needs to run where the process is. That is the
fact §4.8 lacked a use for, because copying wasn't on the table.

**(c) Blank-store blast radius — the real, remaining blocker.** A blind copy of the
whole store puts *every* secret on every node; the exposure grows with the store, not
with the work. This is what the bounding story must answer, and §4.2 does.

### 4.2 What becomes copyable, and the bound

**Decision: the store becomes copyable, selected per key, and the default copy-set is
the node's declared need.** Stated once here — §5, S4 and the C5 brief all read their
defaults off this paragraph:

- **The default is the needs-list:** the union of (i) the `ref:<NAME>` refs the node's
  pushed bundles declare — "the keys to set" is already computed
  (`mcpdefs.state_rows:782-825`) — and (ii) keys the operator has marked `sync`. A node
  whose work needs three secrets gets three, not the store. This is the working reading
  of the directive's "necessary": minimisation where the work itself has not named a
  need.
- `sync` — the operator's standing "send this to approved nodes" mark. A key so marked
  joins every approved device's copy-set by default, for keys the operator knows a node
  will need before the node's bundles can say so.
- `local-only` — never crosses. The operator's per-key kill switch (and the class rule
  for anything device-bound).
- `refuse`-by-class — the structural exclusions of §1.4 apply; nothing here can widen
  them.

Everything outside the default set is *offered* — visible on the approval card, one
step to add — and not copied unless selected. One keystroke up per key when wanted, not
an opt-out per key to keep it out.

The **bounding story**, stated as five concrete bounds:

1. **The gate.** Nothing copies without a signed, single-use, expiring approval record
   naming this device (`approvals.py`; the card lists the set — names, never values).
2. **The reduce step.** The card's list is visible and reducible before the gesture;
   post-hoc narrowing and unshare stay available (`credential revoke`).
3. **The needs-list by default.** The card's default selection is the union of
   (i) the `ref:<NAME>` refs the node's pushed bundles declare — "the keys to set" is
   already computed (`mcpdefs.state_rows:782-825`) — and (ii) the operator's standing
   `sync` keys. A node whose work needs three secrets gets three, not the store.
4. **The destination.** Only the node's own credential storage — per class: the node's
   encrypted store re-sealed under its own master key (class 2), or the node's own 0600
   credential rows (class 4, the local-login posture) — with provenance marking (new: an
   `origin`/`owner_device` field or sidecar index on the receiving side, so `local-only`
   marks and wipe notices are computable). **As built (S4):** the marker is `origin` — a
   small bounded object sealed INSIDE each record's payload (`secrets/store.py`'s
   `_payload`/`_validate_origin`; never plaintext on disk, surviving `update` and
   `rotate`), and `MESH_ORIGIN_KEY` inside each class-4 row's `data`, where the existing
   row shape already carries it. The `applied` sidecar keeps gen/digest/row-ids as the
   fast path, but the WIPE scans by the marker, never by the sidecar: a lost sidecar
   must not orphan a copy.
5. **The ending.** Wipe + rotate (§2.3, §5.5).

**Explicitly rejected in the other direction:** copying the owner's whole `secrets/`
directory or its `master.key` (the key never moves; each node's store keeps its own
key, `secrets/keys.py`), and shipping values through the definitions bundle (that
bundle stays non-credential by its own rule, `definitions._withheld:571`).

### 4.3 The wipe/rotate story, concretely

- **Wipe.** Un-approve or unshare → the owner sends one wipe notice per covered key
  (§5.5); the node deletes rows by provenance (`origin = mesh:<owner_device>`) and acks.
  Deletion is the store's ordinary delete path (`secrets/store.py:1186`; class-4 rows:
  the credential row delete), so no new deletion semantics are invented. **As built
  (S4), the notice and its confirmation:** the notice is an announce carrying
  `value_state: absent`, recomputed on each contact — never a queued frame — and the
  confirmation rides the notice's own REPLY, not a fresh request: a deactivated
  member's `broker_credential` capability left with its grants, and the transport
  refuses `net_broker` from its rows (measured), so a fresh-request ack would be
  refused at the owner's door for exactly the members a wipe matters most for. The
  member therefore deletes inline on its slow-op worker (local, bounded — no dial, no
  transfer) and answers `wiped`; the owner records the ledger row from that reply. The
  member-side delete is bound by the marker it scans, not by the grant it outlives.
  **As built (S4, review round 1 Q1), the NO-NEXT-CONTACT case:** the definitions tick
  only visits ACTIVE members, so `member rm` — the un-approve path — is the last
  moment a contact is possible at all. The removal handler therefore runs the ending
  exchange ITSELF, bounded (one probe-capped dial, at most a cap of frames), BEFORE the
  tombstone is written — after it the dial is refused by design, and measured the
  copies survived silently. A member unreachable at that instant keeps its copy with
  the ledger row left OPEN and the receipt naming the count plus the ceiling sentence:
  deleted-and-confirmed / open-and-named is the whole receipt vocabulary, so a removal
  with live copies never reads as a clean sweep.
- **Rotate.** The only ending for a copy that may have left the node. The design's
  guidance: rotate at the provider (API keys), `gh auth logout`/token revocation (forge),
  `lop secret` update on the owner then sync (store secrets — though for a suspected
  exfiltration the provider side is the real rotation).
- **The receipt says which happened**, and both sentences are shipped in
  `credentials/messages.py`'s house style — one home per surface
  (`mesh-credentials.md` §4 intro). **As built (S4):** `render_copy_revoke_notice`
  renders the per-state line (a queued notice / an already-wiped copy / no confirmed
  copy) from the same ledger the listing reads, and `COPY_CEILING_SENTENCE` is the
  §2.3 ceiling, printed by `credential revoke` whenever a copy existed at all.

---

## 5. Lifecycle 2 — SYNC on change

### 5.1 The carrier: the definitions syncer's tick (the rolling-updates answer is §7)

The built, running machine for "a change on this device reaches every member soon" is
the **definitions syncer**: `DefinitionsSyncer` (`definitions.py:2019`) walks durable
member records every `network.sync.tick_s` (**15 s shipped**), with a **60 s** floor
between pushes to one reachable member (`STATE_MIN_INTERVAL_S:2199`), a separate refusal
floor (`REFUSED_MIN_INTERVAL_S:2210`), and a failure retried on the next tick — and it
publishes exactly one extension seam for sibling cadences:
**`definitions.add_tick_step` (`:1999`)**, whose own seam comment argues against a
second cadence over the same members (the `#:` comment above `_TICK_STEPS`,
`definitions.py:1982-1987`: "a second thread, a second set of floors and a second retry
policy for one shared question"), with `mcpdefs.mesh_tick_step` as the first tenant
(`mcpdefs.py:1350`).

**Decision: credential sync rides that seam.** A new tick step (working name
`credentials_sync`) runs after each member's definitions push and does one bounded unit
of work: exchange generation state, deliver/refresh copies, collect acks. No new thread,
no new floors, no new retry policy — the seam's contract, kept.

**Why not `net_update` (the addendum's question, answered in full in §7):** its
transport delivers no payload (the member installs a build it could have obtained
itself), its per-member idle gate actively harms credential freshness, and one-at-a-time
serialisation has no rationale for a per-key state exchange. What it does carry — the
catch-up discipline for unreachable members — the definitions syncer already implements
in the form this design needs.

**The wire.** One op, `net_broker` (already `both`-direction, capability
`broker_credential`; `mesh-transport-identity.md` §6.4; `types.OP_CAPABILITY:524`), with
two new kinds alongside `grant|report|placement|repair`:

- `announce` (owner → member): `{kind: "announce", key, gen, digest, value_state}` for
  every key whose generation moved since this member's last ack. Tiny; idempotent;
  recomputable — a missed announce is not replayed, it is recomputed on the next
  contact.
- `copy` (member → owner → member): the member asks for `key@gen`; the owner answers
  with the value (inside the link's already-authenticated AES-256-GCM records,
  `mesh-transport-identity.md` §6.3) plus `{gen, digest, provenance}`; the member writes
  it into its own store, re-seals, acks.

Kinds on one op rather than a new op, for the reason `mesh-credentials.md` §7.10 already
records: the transport authorises per op, and four ops for one authority would be four
capability rows to keep identical.

### 5.2 Generation counters, acks, and what the owner can see

- **Owner-side generation, per (network, key).** `gen` is monotonic, bumped whenever the
  owner's stored value changes. Detection of "changed" is diff-based on reads the tree
  already does cheaply: the store's `list()` returns metadata only (`store.py:1050`) with
  `updated_at` per record (`:221`); the auth store has the same shape and a live
  precedent (`owner._row_stamps:1338` reads `{credential_id: updated_at}`). The first
  format is an owner-side `credentials/<network_id>/sync.json` (new; beside the
  placement documents); a write-hook in the store's one writer
  (`secrets/brokerd.py`) can replace the diff later without a wire change — recommend
  diff now, hook when measured to matter (Q6).
- **Ack ledger, owner-side, per member:** `{device: {key: acked_gen}}`. This is what
  makes staleness *visible to the owner*: `lop network credentials` and `doctor` grow one
  segment per member row — `synced (gen 7)` / `stale (gen 6 of 7, last acked 14:02)` —
  built on the same row shape as the rollout segment `mesh-rolling-updates.md` §7.1
  proposes (do not build two segment vocabularies; if that doc lands first, share its
  row).
- **Monotonicity rule:** a member applies a value only when `gen` is greater than the
  one it holds; an older announce is dropped, out-of-order delivery cannot roll a copy
  back.

### 5.3 Push from head vs pull by peers, cadence, and the latency budget — decided

**Both, split by weight: push the announcement, pull the payload.** The announce is a
few dozen bytes and rides the already-scheduled tick; the payload is fetched by the
member (the direction borrows already take — the borrower dials, `client._ask_owner_directly:600`),
so a sleeping member cannot make the owner do transfer work, and the pull keeps the
payload path byte-identical in shape to an existing borrow reply.

- **Cadence.** The syncer's own: 15 s tick, 60 s floor to one reachable member, failure
  retried next tick. No new tunables.
- **Latency budget (the number the addendum asks for):** a changed credential is usable
  on a reachable member **within ≈ 75 s p95** — the 60 s floor + one 15 s tick + one
  pull round trip — and **immediately before next use** when the use path notices a
  generation mismatch: the load path compares its held `gen` against the last announced
  one and, when stale and the owner is reachable, pulls first. That first command pays
  one round trip, bounded by the existing deadlines (`CLIENT_DEADLINE_S = 8.0` client-side,
  `client.py:70`; `BROKER_OP_DEADLINE_S = 75.0` owner-side, `credentials/__init__.py:54`).
  The p95 figure is a design budget to be *measured* on the two-device rig (§11, Q3 of
  `mesh-credentials.md`'s evidence set); if measurement disagrees, the budget changes,
  not the mechanism.
- **The budget's other half — the in-flight question, in the operator's words: *how
  long can a running job on the node keep going once the owner rotates a credential?*
  Per mechanism, and the mesh never cuts the job (property 1 of §5.4):**
  - a **borrowed** bearer keeps working until its own provider-side expiry or the
    borrower's next re-ask — the three statements of `mesh-credentials.md` §3.7 ("new
    grants stop now; a lent grant stops within `grant_ttl_s`; a copied bearer stops
    only when the token expires or is revoked at the provider");
  - a **held copy** outlives the owner entirely and ends only at (i) a wipe, or
    (ii) provider-side rotation killing the old value — in which case the job's next
    *use* of that value fails into the §6 repair path while the job itself keeps
    running;
  - the **replacement** reaches the node inside the same ≈75 s budget above, so the
    work-interruption window for a rotate-at-source is "until the next use after the
    old value dies", bounded by that number.
  So the budget answers both halves: ≤75 s to the replacement, and *no job is cut by
  the sync — only a credential step fails*. That last sentence is what S5's drill must
  prove (Q7).
- **Interactive freshness beats cadence:** a member that has been told a newer gen
  exists never waits for the next tick to use it — it pulls on the spot. The cadence is
  the floor for idle members, not the ceiling for busy ones.

### 5.4 In-flight work does not stall on the sync path

Three properties, each checkable:

1. **Use never awaits sync.** The load path consults the copy it holds; a stale copy is
   still served (static classes have no freshness requirement; a dead value fails at its
   own use and takes the repair path). The only place a pull can precede a use is the
   explicit freshness check above, and it is bounded by the same deadlines as any
   borrow.
2. **Sync never awaits a busy member, and a busy member never defers its sync.** This is
   the deliberate inversion of the rolling-updates idle gate (§7): a member under load
   is exactly the member whose credentials must stay fresh; the tick step is non-blocking
   per member (enqueue; the exchange runs on the broker's loop), and it holds no lock
   across network I/O.
3. **Atomic per key.** Applying a copy is one store write under a fresh nonce
   (`store.py:755` update path; upsert for auth rows), so a reader sees the old value or
   the new one, never half of either.

### 5.5 Failure semantics — the addendum's (a), (b), (c), decided

**(a) Peer offline when the owner changes a value → catch-up, never invalidate.**
There is no queue of frames to replay: the *state* (key@gen) is the durable thing, and
the next contact recomputes the work (the definitions syncer's own model — "a failure is
retried on the next tick", member records are durable, `definitions.py:2073-2093`).
Until contact, the member holds its previous copy; for static classes it keeps working,
for anything rotated at the source it fails into the §6 repair path. If the operator
needs immediate effect on an offline node, the honest remedy is the one that always
works: rotate at the source — the stale copy is then worthless, regardless of whether
the node ever comes back.

**(b) Can the owner SEE a peer's copy is stale → yes, via the ack ledger (§5.2), and
staleness NEVER blocks a session from starting there.** Decided deliberately: session
creation is a different authority surface (prompt/move), and gating it on sync state
would make an unrelated failure mode (a slow link) block work the operator explicitly
wants to run remotely — the operator's own "in-flight work must not stall". The gate is
at *use* of a credential, not at session start; the surface shows the stale chip so the
operator can act before the failure, and the repair path catches it after.

**(c) Revocation → stop, wipe, rotate; in that order, and in-flight sessions survive.**
Un-approve / unshare runs: (1) broker refusal is immediate for new grants (measured
2.3 s) and the link-level membership rules are unchanged; (2) one wipe notice per copied
key with this owner's provenance — reachable members delete and ack, unreachable ones
get it on next contact (same catch-up as (a)) — UNLESS the member is being REMOVED,
where no next contact exists: the removal path delivers the ending while the member is
still contactable and, unreachable at that instant, records the OPEN ending on the
ledger and the receipt (as built, S4 review round 1 Q1); **as built (S4), the ack for a wipe is
the reply to the notice itself — an un-approved member can no longer open a frame —
see §4.3's as-built note**; (3) the receipt states the ceiling and
the rotate guidance (§2.3). In-flight sessions on that node are **not** killed: they are
the node's sessions, and the removal is a credential-and-link event; running turns keep
what they hold (its borrowed grants die at their TTL; its copies die at the wipe or the
rotation). This matches the mesh's standing availability direction ("a session in flight
is not disrupted", `mesh-transport-identity.md` §8.4) and the operator's non-stall rule.

---

## 6. Lifecycle 3 — REPAIR, and the in-flight policy while a copy is briefly stale

### 6.1 How the node learns, with no human in the loop

Three detection channels, cheapest first:

1. **Use-time failure classes the mesh already classifies.** A borrowed bearer's
   failure runs `MeshAwareAuthStore.rotate_sibling` → `credential_report` (owner-side
   handling, `owner.report:991`); for copied values the session's own command reports
   its error, and the failing tool's class is available through the same classifier
   (`store.report_kind_for:902`). New: one added code, `copy_stale`, for "the copy you
   hold is older than the owner's and upstream rejects it" so a sentence exists for it.
2. **The repair row that already exists.** `credential_repair` is derived from report
   rows and rendered by `network doctor` / the panel (`mesh-credentials.md` §4.7,
   `GUIDE.md:590-594`).
3. **The generation mismatch itself** — the §5.3 freshness check notices the copy is
   older than announced and pulls before use; this is repair-before-failure, the best
   case, and needs no human at all.

### 6.2 Re-acquire, automatically where the source allows it

- **Owner-side refresh runs first** where the owner has a path: provider OAuth refresh
  (owner-only, R14/R15), PAT re-read, App re-mint. The node then pulls the new value on
  the next announce/use — no node-side human, no node-side login. This is the lifecycle
  the addendum asks for: "a locally-refreshed secret keeps remote work uninterrupted".
- **Interactive-only sources** (an expired `gh` login; an MCP grant that needs a new
  loopback callback) raise the owner's `credential_repair` card — the one human act, on
  the side that can perform it, as a product action. The member's sentence names the
  repair state, not a terminal command (§2.9 copy rule, `mesh-remote-onboarding.md`).

### 6.3 In-flight policy while a far-side copy is briefly stale — decided

- **Transient classes → retry-with-backoff.** Network errors, 5xx, rate limits, owner
  briefly offline: the existing TTL table drives it (`BROKER_ERROR_TTL_MS`,
  `credentials/types.py:252`: 15 s for `owner_offline`, retry-once; `retry_after_ms`
  honored), and the sync catch-up retries on its own floors. A turn does not hang: one
  bounded attempt per call, TTL-cached, and the next call retries.
- **Permanent classes → fail-fast.** 401/revoked/wrong-account: the command fails with
  the incident sentence and the repair row is raised; no silent retry loop, because a
  retried dead credential is a slower failure, not a repair. The session keeps running;
  only the credential-using step fails.

The boundary between the two is the existing failure classification (the driver's
classes, `store.report_kind_for:902`), not a new taxonomy.

---

## 7. The carrier question: rolling updates, evaluated and declined

The addendum asks whether the rolling-updates machinery carries credential propagation.
Answer, precisely:

1. **It is design-stage, not built.** `docs/design/mesh-rolling-updates.md:3` says
   "Status: design, pre-implementation" (verified against `origin/main` @ `1f0a1b909`);
   the tree at our base contains **no** `net_update`, no `mesh-update-v1`, no rollout
   record (grepped: zero matches in `local_operator/network/**`). "Already propagates"
   is true of the *definitions* machinery (`DefinitionsSyncer`), not of `net_update`.
2. **Its transport deliberately carries no payload.** `net_update` is apply-or-answer
   for a *build target* the member installs from its own channel — "The peer never
   delivers code, never names a ref, never runs a command"
   (`mesh-rolling-updates.md` §0, `:50-51`; §2 adds the fuller bound, "a nudge to
   fetch and install, not a delivery", `:258-262`). Credential sync is the opposite direction: the owner
   delivers the value. Reusing the op would mean turning a no-payload transport into a
   secret-delivery one — the exact re-plumbing this design avoids by riding
   `net_broker`, whose payload path already exists and is already redaction-audited.
3. **Its ordering semantics are wrong for freshness.** One member at a time, idle-gated
   on the member, deferral on busy — correct for a build swap (bounded installer churn,
   never cut a turn) and actively harmful for credentials: a busy member would defer the
   very refresh it needs, and serialisation around a tiny per-key exchange buys nothing.
   §5.4's non-stall property is the requirement; the idle gate is its negation.
4. **What it gets right, and where that lives already.** Catch-up on next contact,
   bounded retry, a durable record, "never force" — the definitions syncer implements
   the same discipline in the form this design needs (durable member list; failure
   retried next tick; refusal floor). Its `add_tick_step` seam was *built* for a second
   cadence to join without a second thread; the credential sync uses it.

So: **do not route credential propagation through the rolling-updates lane; ride the
definitions syncer.** If a future artifact genuinely needs a payload delivered under a
rolling, idle-gated discipline (a repo seed, a large resource), `net_update`'s
slow-pool apply-or-answer shape is the right precedent to copy — for that thing, not
for credentials.

---

## 8. Security invariants: preserved, altered, and the threat-model delta

### 8.1 Preserved (each checkable in code or tests)

| Invariant | Source | Status here |
|---|---|---|
| The refresh never moves; only the owner POSTs a refresh | `mesh-credentials.md` §0.1, §3.4 (R14/R15) | Preserved unchanged for every rotating class; copies exist only where there is no refresh. |
| Ownership is explicit metadata, authored only by the owner; single-writer rows | §2.1 (`placement.py` owner-only `grant/revoke`) | Unchanged; the transaction writes through the same functions. |
| The placement document carries **no material** (`_assert_no_material`, `placement.py:279`) | §2.1 | Unchanged — copies live in the stores, never in the placement document; the assert stays as the guard. |
| `holders` is the authorisation; absence refuses (not default-allow) | §2.1 | Unchanged; it is *the* authorisation under this design too. |
| Grant narrowing: never more than the owner holds; expiry = `min(token_expiry, now + ttl)` | §3.3 (RFC 8693/STS) | Unchanged for brokers; the copy path's bound is the selection + wipe, stated. |
| "Never returns" list (refresh token, master key, other providers, control keys, wildcard scopes) | §3.3 | Unchanged; the copy kinds serve only keys the member is a holder for, one key per request, digest-pinned. |
| A brokered run leaves the requester's `auth.db` logically unchanged | §5.2 | Unchanged for the broker; a **copy** is a deliberate, provenance-marked write — new, and named as such (below). |
| Zero-peer topology byte-identical | §5.1 | Unchanged: no placement → no copy machinery constructed; the tick step is a no-op without members. |
| A refusal is not a transport failure | §3.2 | Unchanged; new codes join the same closed set (`types.BROKER_ERROR_TTL_MS`). |
| Epoch secret withheld from removed members; no outbox entry holds a secret for one | `mesh-transport-identity.md` §8.1 invariants | Preserved and **extended**: the sync path must equally withhold copies from a non-active/removed member — pin it as a test (§8.3). |
| Unknown op fails closed; `locality: "local"` over a link is a protocol error; per-op capability at one chokepoint | §7.2 | Unchanged; new kinds ride the existing `net_broker` authority. |
| "Nothing in the implemented list writes `secrets/`, `auth.db`'s schema, or the keychain" | `mesh-credentials.md` §10 | **Altered** — see below; the alteration is scoped to the copy classes and to the *node's own* store. |

### 8.2 The invariant this design alters, and why it is acceptable under the operator's model

**Altered: "Token material never crosses a device boundary except as a short-lived,
scope-limited bearer … and never touches the receiving device's disk"**
(`mesh-credentials.md` §0; the §10 sentence is its implementation pin).

Under consent-as-authorisation, the operator's model says the approved device IS the
delegation target, and "the necessary data should be cloned". The old invariant was the
right answer to *unattended, default-off* sharing; it is the wrong answer to *approved
devices that must keep working when the owner is asleep*. The design therefore narrows
the invariant instead of deleting it:

> **Copy invariant (new):** credential material crosses to a device only inside the
> authenticated link's records; on arrival it lives only in that device's own encrypted
> store, sealed under that device's own key, marked with its origin; it is never
> written as plaintext, never into another host's helper, never with the owner's key;
> and it exists only for keys the device is an active holder for.

Threat-model delta, stated honestly: an approved node that is later compromised
escalates from "can spend borrowed bearers for ≤ TTL" to "holds copies that outlive any
revocation short of rotation". The mitigations are the §4.2 bounds and the §5.5 ending;
the residual is the ceiling sentence (§2.3). For rotating classes the delta is **zero** —
they are not copied. For the forge, the copy is opt-in (§3.3) precisely because its
delta is the largest.

A second, smaller alteration, named so a reviewer sees it: **`broker_credential` moves
from "admin-only + explicit per-credential opt-in" to "present on approved rows by
default"** (`credentials/__init__.py:10-13` states today's rule). The real authorisation
is and stays the per-key `holders` row; the capability merely stops being a second
gate the operator must open by hand. If review wants the capability retained as an
explicit switch, that is a one-line default change at the transaction (Q5).

### 8.3 New invariants the implementer should pin with tests

- `copy_requires_active_holder`: no copy to a non-member, a removed member, or a pool
  member — same check as the broker's, not a parallel one.
- `copies_are_node_local`: no file of its own beyond the node's OWN credential
  storage (class-4 copies: the same 0600 `auth.db` rows a local login writes;
  class-2: the node's encrypted store, re-sealed under the node's own key); no
  owner key; never a world-readable file. (Renamed from
  `copies_are_node_local_encrypted` in review round 1, Q-6, because the class-4
  half was never encrypted-store.)
- `sync_never_blocks_use`: a use with a stale copy completes or fails on its own terms;
  no code path awaits the syncer.
- `generation_monotonic`: an older gen never overwrites a newer copy.
- `wipe_is_acknowledged_or_queued`: every wipe is acked or re-attempted on contact;
  the receipt distinguishes the two.

---

## 9. Migration, interim, and slices

### 9.1 Interim (what a user experiences until the slices land)

- The **App stays the only working push path today**, labelled **interim** in the guide
  until the ladder ships; the guide's taught route flips in the S2 PR (docs are part of
  that PR, not a follow-up — the "taught route" claim is a docs claim).
- The pairing offer defaults stay as they are until S1's flip; nothing in the interim
  misrepresents: current copy still says push is unavailable without an App
  (`GUIDE.md:612-616`), and that sentence stays true until S2 changes the adapter.
- No version bump in any PR (release window owns that).

### 9.2 Slices (ordered; each a PR with the standard rounds)

| # | Slice | Contents | Size | Gates |
|---|---|---|---|---|
| S1 | **Approval → provisioning transaction (owner side)** | `onboard.py` `step_provision`; placement writes + push; capability enablement; git-identity seed; card/receipt copy; `offers.py` defaults flip (§1.4) — excluding `radient` if Q4 decides to hold it. | M | unit (transaction receipts, idempotent re-run, refusal paths), two-config-roots e2e; reviewer + QA. No secret material moves. |
| S2 | **Forge without an App** | `GITHUB_APP` → ladder in `owner._resolve_github`; PAT-class secret name; `gh`-CLI arm; per-source revoke receipts; guide rewrite (taught route = ladder; App = stronger option). GitLab: position doc + adapter slice stub (or the adapter itself if capacity allows). | M | unit (source ladder, refusal arms, helper unchanged), real-git loopback cell extended per source; reviewer + QA; guide copy round. |
| S3 | **Sync engine** | Generations + ack ledger; `announce`/`copy` kinds; `credentials_sync` tick step; member-side apply + ack; staleness surface (`credentials`/`doctor` segment). | L | unit (monotonicity, non-block, floors), two-device e2e (change → usable within budget); **C5 review before merge — see below.** |
| S4 | **Class-2 copies** | Policy marks (`sync` / `local-only`) with the **needs-list default** (§4.2); card selection; node-side re-seal + provenance; wipe/revoke; `copy_stale` repair code. | L | unit (encryption-at-rest, provenance, wipe ack/queue, default-set intersection), e2e (copy → use → revoke → wipe); **C5 review before merge.** |
| S5 | **Repair + visibility polish + drill** | Repair codes/sentences end-to-end; freshness check at use; the two-device drill runbook + evidence; sibling-doc reconciliation (`mesh-credentials.md`'s stale `replicate` text — its §2.1 field was never built and this design supersedes it; the §14 App section gains the "optional stronger" label). | M | drill evidence on the real topology (Mac + one cloud node), frames for the surfaces touched; reviewer + QA. |

**The C5 class, named:** under the operator's merge disposition, a foundational
security/data change is C5 — it holds for human approval before merge. S3 and S4 are
exactly that class (the first secret-copying code), and they additionally carry the
standing agent-review + QA rounds. The one default the C5 reviewer should read first is
§4.2's: **the copy-set defaults to the node's declared need (needs-list ∪ `sync`
marks), not the store.** S1/S2 are ordinary code changes (no new material crossing),
gated as usual.

**Doc reconciliation is part of the work, not an afterthought:** `mesh-credentials.md`
§2's join-default table and §14's App narrative need the two-line pointers to this
document's §1.4 and §3; the guide's GitHub section is rewritten in S2; the
`/network credential share` help text gains "not needed for onboarding anymore" wording
in S1.

---

## 10. Open questions, each with my recommendation

**Q1 — Class-2 default: the needs-list, or every key?** *Decided: the needs-list*
(§4.2 — declared refs ∪ `sync` marks; everything else offered, not copied). "Necessary"
is the directive's word, and minimisation is its reading where the work has not named a
need; the copy-everything reading was rejected as unbounded blast radius (the §4.1(c)
concern), and it is one constant away if the C5 review ever judges otherwise. This is
the biggest exposure decision in the document, so it is recorded as a decision, not
left to the implementer.

**Q2 — Do copies carry a TTL?** *Recommend no.* The point of a copy is to outlive the
owner's reachability; a TTL would reintroduce the dependence the copy removes. The
ending is wipe + rotate (§2.3), and the receipt says so.

**Q3 — Forge copy default.** *Recommend off* (broker-only default; opt-in per device).
The token is a full login; the directive's "necessary" reads as "when reachable" for
this class until the operator asks for more. If the operator wants offline pushes as a
default, this is the one constant to flip, and the disclosure copy already exists.

**Q4 — Radient default at admission.** *Recommend offering it by default (reduce-only)
for the operator's own `device` members, keeping the pool exclusion and the session
scope.* It is the class-3 login the operator most plausibly needs on a node (publish/pull
work); the org-write concern is real and this is the row most likely to draw a review
objection, hence the flag. Evidence that would settle a hold: whether the drill's node
work actually needs org-write out of the box (S1's e2e answers this cheaply).

**Q5 — Keep `broker_credential` as an explicit per-device switch?** *Recommend tying it
to the approval (auto), because it is not the authorisation (`holders` is).* If review
prefers, the switch returns as a card row with default on — same shape as the other
scopes.

**Q6 — Where the owner-side sync state lives.** *Recommend an owner-side
`credentials/<network_id>/sync.json` (generations + acks) with diff-based change
detection now; move the generation bump into `secrets/brokerd.py` (the store's one
writer) when measured to matter.* Evidence: store-write→announce latency on the drill;
if the diff tick is late by more than one tick, the hook moves.

**Q7 — Latency budget verification — two cells, both in the drill.** *Recommend the
drill pins both numbers:*
1. **The delivery cell:** change a value on the Mac, run a command on the node, assert
   usable within the §5.3 budget with the on-demand path disabled (pure cadence), then
   with it enabled (first-use pull).
2. **The in-flight cell** (§5.3's other half): start a long-running job on the node (a
   bounded sleep-and-report prompt), rotate/deliver a value mid-job, and record per
   class what the running job saw (continues on the old value / fails at the next use),
   when the replacement became usable, and that new work resumed. This is the evidence
   S5 cites to the C5 review for "work continues uninterrupted".

The numbers in §5.3 are budgets, and budgets get measured.

---

## 11. Test plan and file-by-file change list (for the coder)

### 11.1 Isolation, per `AGENTS.md`

Every harness cell: fresh `HOME` + `LOCAL_OPERATOR_CONFIG_DIR` via `env -i` (strips
`CMUX_*`/`LOP_*`), one `ISO` per block; never the operator's live config. The
two-config-roots topology from `mesh-credentials.md` §9.2 is the cheap primary rig; the
real topology (Mac + cloud node) is S5's drill.

### 11.2 Named tests the design requires

- S1: `test_provision_step.py` — receipts order, idempotent re-run, refusal arms
  (device missing, network unknown, placement write refuses), reduce-step intersection;
  `test_offer_defaults.py` — the §1.4 table, one test per row (the existing pair-offer
  tests extended; `tests/unit/network/test_pair_offer.py` is the base).
- S2: `test_credentials_github.py` (existing) gains one cell per ladder arm; a negative
  cell per arm proving the identifier is absent (App absent → PAT arm; PAT absent → gh
  arm; all absent → `no_local_credential`); the helper cells stay byte-identical.
- S3: `test_credentials_sync.py` — announce/copy exchange, monotonicity, floors,
  non-blocking (a busy member still syncs), zero-member no-op; `test_wipe_notice.py` —
  ack, queue-on-offline, removed-member never receives.
- S4: `test_secret_copies.py` — at-rest encryption (no plaintext bytes on disk outside
  the sealed store), provenance set, `local-only` never crosses, needs-list union.
- S5: the drill runbook file (commands, expected receipts), following
  `mesh-onboarding-drill.md`'s shape.

### 11.3 File-by-file (the surface a reviewer can diff against)

| File | Change |
|---|---|
| `local_operator/network/onboard.py` | `step_provision` (+ wiring in `execute_approval`'s step order). |
| `local_operator/network/credentials/placement.py` | bulk-grant helper for the transaction (per-key, through `grant`). |
| `local_operator/network/credentials/offers.py` | §1.4 defaults; card rows gain the copy disclosure line. |
| `local_operator/network/credentials/owner.py` | source ladder in `_resolve_github`; `announce`/`copy` kinds; ack handling. |
| `local_operator/network/credentials/client.py` | pull-copy path; freshness check at load; `copy_stale` code. |
| `local_operator/network/credentials/github.py` | token-source arms (gh/PAT) alongside the App; receipts per source. |
| `local_operator/network/credentials/sync.py` (new) | generations, acks, tick step, wire kinds' payloads. |
| `local_operator/network/definitions.py` | (no change expected; the seam exists — the tick step registers through `add_tick_step`). |
| `local_operator/network/readiness.py` | git-identity seed helper (S1); staleness segment (S3). |
| `local_operator/secrets/store.py` / `brokerd.py` | provenance + (when it moves) generation hook; no schema break. |
| `local_operator/guides/network/GUIDE.md` | taught-route rewrite (S2), copy/wipe/ceiling sentences (S3/S4). |
| `docs/design/mesh-credentials.md` | two pointers (§1.4/§3), `replicate` supersede note (S5). |

Nothing in this list touches `credentials.env` (retired), the keychain, or another
host's helpers — and that sentence is a test (the existing
`tests/unit/network` "never opens secrets / no keychain" class extends to the copy
path with one negative cell each).

---

## Relationship to the sibling designs (for the reader who starts here)

- **mesh-credentials.md** — this document does not replace its broker protocol; it
  widens *what is shared by default* (approval-driven) and *how static material reaches
  a node* (copies + sync), and narrows its one hard sentence ("never touches the
  receiving device's disk") to the copy classes only.
- **mesh-remote-onboarding.md** — the approval record and step machinery is consumed
  as-is; the transaction is one more step, with the same gate and receipts.
- **mesh-transport-identity.md** — no protocol version bump; new kinds on an existing
  op; the `net_sync` reservation is untouched (that is session sync's op; this design
  rides the definitions tick and `net_broker`, deliberately).
- **mesh-rolling-updates.md** — evaluated in §7: catch-up discipline borrowed in
  spirit, transport declined; the visibility segment should share its row shape if it
  lands first.
- **mesh-incident-response.md** — untouched; the wipe/rotate receipts join the audit
  kinds that document owns, and this design adds no new always-on state to seize.
