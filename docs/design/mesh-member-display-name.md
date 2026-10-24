# Mesh member display name — naming a device that has already joined

Status: **design, accepted.** P1 and P2 of §7 ship in the change that adds this note
(as one commit, not the two the note suggested); P3-P5 remain unimplemented. It
answers one question: how does a device that is already a member of a network get a
human name, without remove-and-rejoin? It owns no requirement in `mesh-network.md`;
it is a defect note against `mesh-transport-identity.md` §3.1 ("Names … are cosmetic and
operator-set. Nothing authenticates on them") and `mesh-ui.md` §2.8.1.

**Verdict in three sentences.** The name a Mesh tab shows is *the reader's own
member row* (`MemberRecord.name`), seeded once at join from the joiner's signed
`joiner_name` and never rewritten afterwards by any code path. A rename is
therefore **not** a network-distributed fact that needs agreement — it is a
per-device label that each device stores and that we want the copies to agree
about, so it needs no signature, no epoch bump and no new wire op; it needs one
acceptance rule in the existing member-list merge. The minimal *supported*
command is a **self-rename** (`lop network identity set-name <name>`), plus one
prerequisite fix (`_resolve_device` must refuse an ambiguous name) that closes a
pre-existing silent mistarget rather than creating one.

Change class: **C2 (standard iteration)** — see §8.

---

## 1. Where the name lives, and which way it flows

### 1.1 The records

There are **three** fields called a name, and only two are about a member.

| Record | Field | Scope | Serialised as | Written by |
|---|---|---|---|---|
| `MemberRecord` (`types.py:1375`) | `name: str = ""` (`:1389`) | one device's membership **in one network**, as this device holds it | `to_json()` = `asdict` (`types.py:1440`), carried on `net_member_list` (`relay.py:_op_member_list`), `net_epoch` (`relay.py:1580-1600`) and `net_show` (`relay.py:12641`) | `admit()` (`relay.py:1672`, `:1705`), at join only |
| `DeviceIdentity` (`identity.py:152`) | `name: str = ""` (`:168`) | this **install** | `identity/device.json`, 0600 (`identity.py:303`) | `mint()` (`:271`, default `socket.gethostname()` via `_default_name`, `:470-473`); `rotate(name=…)` (`:357`) keeps it |
| `PeerRecord` (`types.py:1894`) | `device_name: str = ""` (`:1915`) | this device's **relay process**, for local readers | `run/peers/<pid>.json` | `relay.py:5517`, from `self.identity.name` |

`PeerRecord.device_name` is documented as "operator-set, cosmetic, never
authority" (`docs/design/mesh-transport-identity.md:463`). It never reaches a
peer and is not the membership name; it only matters here because it reads
`identity.name`, so a stale identity shows up in local `lop network status`
output too.

### 1.2 What each surface actually reads

Every human-facing surface reads the **local member row**:

- `lop network peers` → `net_peer_ls` → `"name": member.name` (`relay.py:12460`).
- `lop network show` → `net_show` → `members_detail[]` with `"name": member.name`
  (`relay.py:12641-12652`).
- The Mesh tab → `GET /v1/desktop/networks` → `network_topology`, which takes
  `members_detail` from `net_show` when the relay answers and otherwise builds the
  same shape from `member.name` (`server/utils/desktop_mesh.py:370`, `:430`). The
  renderer's label is `name || device_id tail`
  (`local-operator-ui/src/renderer/src/features/mesh/mesh-types.ts`, `deviceLabel`).
- The `/new` peer picker and the receipt vocabulary → `peers.KnownPeer.name`
  (`peers.py:91`), fallback `device_id[:8]` (`:59-61`).
- Receipts composed by the relay → `_member_name(device_id)` (`relay.py:9414`),
  `_own_label()` (`:9447`), `_peer_label()` (`:9477`).

The device's **own** notion of itself also reads `identity.name`, in seven places:
`status.device_name` (`relay.py:12801`), the published peer record (`:5517`), the
`unattended` remedy sentence ("approve setup for {name} in the Mesh tab",
`relay.py:7880-7890`), the pair-phase welcome's `inviter.name`
(`handshake.py:883`), the join block's fallback (`cli.py:4111`,
`relay.py:10904`), `init`'s own member row (`cli.py:1340`) and `credential`
ledgers' `owner_name` (`cli.py:1676`, `network/credentials/client.py:1251`).

**These two disagree by construction**, and the code says so:
`_member_name`'s docstring (`relay.py:9414-9441`) records that a peer's label is
resolved from the member table while this device's own label was resolved from
its identity, "and the two disagree whenever the identity was minted before the
join".

### 1.3 The direction of flow — the answer

**Peer → us, once, and then never again.**

1. The joiner puts its name in the join block, `build_join_block(... joiner_name)`
   (`handshake.py:273-278`). The join block rides the `hello`
   (`handshake.py:build_hello`, `mode == "join"`), and the hello is one of the
   three canonical fields hashed into the transcript (`handshake.py:133-153`)
   that the joiner **signs with its device key** (`:548-552`). So the initial
   name is bound to the joiner's key and to the two humans' SAS comparison.
2. The inviter stores it: `admit(..., name=joiner_name, ...)` in the pair
   listener (`relay.py:10501-10505`), merged into the row by
   `existing.name = name or existing.name` (`relay.py:1705`).
3. The joiner writes *our* name into its copy of our row from the admission
   frame's rows, or — when the frame carried none — from the welcome's
   `inviter.name` (`handshake.py:883`, consumed at `cli.py:5132`).
4. After that, **nothing**. No op carries a name; `net_member_list` and
   `net_epoch` carry whatever each table already holds; `adopt_members`
   (`relay.py:1376`) deliberately does not take a name for a row it already has.

So the honest answer to "does a peer's name come from that peer's own identity,
or from a name our side stores for it?" is: **both, in that order.** It is the
peer's own assertion at the moment of joining, and from then on it is a name our
side stores. It is never re-read from the peer's identity, and there is no
mechanism by which the peer could update it.

### 1.4 Why the Linux node ends up as `ip-172-31-18-0…`

`load_or_mint` mints with `name or _default_name()` and returns an existing
identity **verbatim** (`identity.py:316-321`). Every caller that can run first on
a fresh install passes no name:

- the relay daemon: `self.identity = identity or load_or_mint(self.root)`
  (`relay.py:4450`);
- `lop network init`: `load_or_mint()` (`cli.py:1305`);
- only `lop network join` passes one: `load_or_mint(name=args.name)`
  (`cli.py:3360`) — and it is a no-op when the identity already exists, which it
  always does if the relay was installed before the join.

`git grep "identity\.name *="` over `origin/main` returns **nothing**: no code
path writes `identity.name` after minting. And because the join block is
`args.name or identity.name` (`cli.py:4111`), an approval-card name (threaded as
`lop network join --name=<approved> --automated` by `network/onboard.py`, commit
`4ceda79a3d`) reaches the **member row** but not the **identity**. That is the
whole defect: the only writable copy of a human name is the membership row, and
the only way to write one is to be admitted again.

And re-admitting is genuinely destructive: `remove_member` (`relay.py:1723`)
tombstones the row, appends the id to `removed_ids` — burned forever, per
`membership_conflict` (`relay.py:1637-1660`) — and rotates the network secret and
bumps the epoch. The burned id cannot be re-admitted, so a "re-join" first
requires `lop network identity rotate` (new key ⇒ new id, `identity.py:325`),
which discards the row's admission-time role/capabilities and every grant made on
every peer. The brief's "rotates the device secret and breaks its grants" is
precise.

---

## 2. Existing seams

### 2.1 Everything that writes `MemberRecord.name` today

| Writer | Site | Name comes from | Authenticated? |
|---|---|---|---|
| Network creation | `cli.py:1336-1340` (`admit` from `_cmd_init`) | `identity.name` (the hostname) | n/a (local) |
| Join admission | `relay.py:10501-10505` | `joiner_name` from the signed join block | **yes** — transcript signature (`handshake.py:548`) |
| Joiner's own copy | `cli.py:5084-5140` (`_persist_join`) | the admission frame's rows, else `inviter.name` from `welcome` | frame is AEAD-protected; `welcome` is plaintext but the inviter is already key-verified |
| Rotation successor | `relay.py:1352` (`_take_over_standing`) | `successor.name or local.name` | the *key hop* is signed; **the name is not** |
| Local label | — | **none exists** | — |

There is no rename path. `lop network rename` (`cli.py:1523`) writes
`NetworkRecord.name` — the *network's* name, local by design ("Rename a network
locally", `cli.py:285`), which is the precedent this note reuses in §3.4.

### 2.2 What is covered by a signature or a digest

- **The initial name**: covered. `join_block` is inside `hello`, which is inside
  `sha256(T)` signed by the joiner (`handshake.py:133-153`, `:548`). A middlebox
  cannot alter it, and the inviter's human compares a SAS derived from the same
  transcript.
- **The approval-card name**: covered. The card's `device` block — exactly
  `{device_id, name, fingerprint, host, user, transport, host_key_fp}` for the
  restricted kind (`enrollment.py:208`) — is part of `_immutable_request`
  (`approvals.py:359-378`) ⇒ `request_digest` (`:380-392`) ⇒ `signed_payload`
  (`:831-860`) ⇒ the operator's decision signature. So "the name the operator
  approved" is cryptographically bound to the approval, and then threaded to the
  join as `--name`.
- **After pairing**: **not covered, deliberately.** `members_digest_of`
  (`relay.py:1613-1634`) hashes `device_id`, `public_key`, `role`,
  `capabilities`, `lifecycle`, `removed_at` — and its docstring says why it is
  "computed from the MEMBERSHIP fields only": a digest that moved on cosmetic
  changes would make "did your list match mine" unanswerable. **`name` is not in
  the digest**, so a name divergence can never produce
  `members_digest_mismatch` (`relay.py:2095-2098`).
- **There is no `doc_rev`.** The membership document's revision counters are
  `NetworkRecord.sequence` (bumped on every `save`, `store.py:370-390`) and
  `epoch` (rotations only). Neither is compared per-name anywhere; `sequence` is
  only "do I already have this frame".
- **The handshake carries no name at all** after the join: `build_hello`
  (`handshake.py:~200-240`) has no name field; `challenge`/`auth`/`welcome` carry
  ids and instance ids. So a rename cannot be smuggled through and cannot break a
  handshake.

### 2.3 What breaks if the name changes after pairing

Each of these is a fact about the current tree, not a hypothetical:

1. **The next rotation from a peer silently reverts it.** `apply_epoch` does
   `record.members = rows` wholesale (`relay.py:2103`, `:2123`) after
   `_keep_local_authority`, and `_keep_local_authority` (`relay.py:1910-1922`)
   preserves **role and capabilities only — not the name**. So any peer's
   `member rm`, `panic` or `identity rotate` republishes that peer's copy of the
   table and overwrites every name this device holds. A local-only rename
   therefore has a lifetime measured in "until somebody rotates", which is worse
   than not having the command.
2. **Nothing converges a name today.** `adopt_members` (`relay.py:1376-1450`)
   merges endpoints and rotation retirements for rows it already holds and copies
   the whole row (including the name) only for rows it has never seen
   (`:1423-1432`). So a self-rename would not reach anyone until a *new* device
   joined and received it in an admission frame.
3. **`_superseded` uses name equality as a display heuristic**
   (`server/utils/desktop_mesh.py:328-355`): an inactive row is hidden when an
   active member of the same network carries the same name or shares a routable
   endpoint. Its own docstring says "a rename alone leaves the endpoints". A
   rename can therefore make a previously hidden tombstone reappear on the Mesh
   tab as an extra revoked node. Display-only, and arguably more honest, but it is
   a visible change and must not be a surprise.
4. **Authority-bearing verbs resolve a device by name, first match wins.**
   `_resolve_device` (`cli.py:2838-2858`) walks `record.active_members()` and
   returns the first row whose `name` equals the argument. Its callers are
   `member grant|revoke` (`cli.py:1570`), `credential share --with` (`:2201`),
   `credential revoke|mark` (`:2296`) and `member rm` (`:2929`). See §4 — this is
   the one real security consequence, and it predates this note.
5. **Typeable tokens change.** `KnownPeer.token` (`peers.py:64-71`) is the name,
   else the id; `known_peer_names` (`:101-121`) feeds the `/network` vocabulary.
   An in-flight `/new remote oldname` after a rename answers "No peer named
   oldname" (`tui/app.py`, the `resolve_peer` refusal). Acceptable, but it is a
   user-visible consequence worth a sentence in the receipt.
6. **Historical text keeps the old name, correctly.** A move tombstone records
   `device_name` at move time (`projection.py` tombstone rows, and the sentence
   "This conversation was moved to {device_name}"). A rename must not rewrite
   history; the tombstone keeps saying where it went under the name it had.
7. **Not affected**: the authoriser, the handshake, `net_sync`, session rows'
   `owner_device_id`, the credential placement document (keyed by device id), and
   `members_digest`/`sequence`/`epoch` (§2.2).

---

## 3. The minimal supported design

### 3.1 The command

```
lop network identity set-name <name> [--json]
```

**Self only.** There is no `member set-name`, and §4 explains why that is a
security decision rather than an omission. This is the smallest shape that gives
a working node a human name, because the name in the row *is* the device's own
assertion (§1.3): the device that owns the name is the only party that can
honestly change it.

What it writes, in one primitive (`rename_self(name, *, root)`), with two callers
exactly like `_apply_capability_change` (`cli.py:1596-1650`):

1. `DeviceIdentity.name` → `network/identity/device.json` (0600, staged;
   `identity.save`, `identity.py:303`).
2. `record.self_member().name` in **every** network record where
   `record.self_device_id` matches — inside `store.mutate(network_id)`
   (`store.py:396`), never a load-then-save from a stale listing.
3. Announce nothing. Peers converge by pull (§3.3).

Two callers, because of an ownership fact the tree already documents:
`store.py:36-52` names the two CLI verbs that write a record a live relay is also
writing, and says the class needs a filesystem lock it does not have. So:

- **Relay answering** (the normal case): one new **local** op — `net_identity_name`
  in `LOCAL_OPS` (`types.py:433-505`, beside `net_member_caps`), which must
  *never* appear in `OP_CAPABILITY` (the totality rule the docstring states). The
  handler runs the same primitive **and** updates the relay's in-memory
  `self.identity.name`, because `RelayServer` reads that attribute at startup
  (`relay.py:4450`) and every `identity.name` consumer (§1.2) reads the object,
  not the file. Then the next heartbeat republishes `run/peers` with the new
  `device_name` (`relay.py:5517`).
- **No relay answering**: the CLI runs the same primitive directly. This is the
  `_cmd_member_caps` fallback shape (`cli.py:1634-1650`) and, unlike
  `_cmd_identity_rotate`'s "build a RelayServer to announce" (`cli.py:8380-8390`),
  it needs no peer traffic — there is nothing to push.

The relay path is not optional polish: without it, `lop network status` (served by
the relay from `self.identity.name`, `relay.py:12801`) would keep showing the old
name while `identity show` (which reads the file, `cli.py:8337`) shows the new
one — the exact class of inconsistency this note exists to remove.

### 3.2 Validation rules

One module constant each, validated in the primitive so every caller inherits it:

- Trim surrounding whitespace; **refuse empty**. (An empty name is the state the
  codebase already calls "unnamed device"; a command that can create it would be
  creating the bug.)
- Length 1..64 characters. 256 is this codebase's ceiling for display text
  (`projection._binding_name`, `:226-233` — 256 chars; `handshake.MAX_ENDPOINT_CHARS`, `:244` — 255); 64 is chosen for
  row width — the TUI's `PEER` column is 16 with ellipsis
  (`mesh-ui.md:244`) and the Mesh node labels are short. The number is a
  judgement, not a constraint; it should be one constant so it can move.
- Refuse control characters (`U+0000-U+001F`, `U+007F`) and line/paragraph
  separators: they break the one-line receipt copy and the TUI row budget.
- Refuse **`#`** — `PEER_ID_SEPARATOR` (`peers.py:38`) and `split_peer_token`
  (`:155`) split on the last `#`, so a name containing one cannot round-trip
  through the picker token it is printed in.
- Refuse a string that is exactly a device id (`d_` + 32 lowercase hex): it would
  make `_resolve_device`'s id-first branch indistinguishable from a name match.
- Allow Unicode, store it unfolded. Case-fold only for the collision **warning**
  below.
- **No uniqueness rule.** `mesh-transport-identity.md` §16 Q8 already decided
  that names are not unique and that resolution is by id; `mesh-ui.md:938-939`
  states that two devices may knowingly present the same human name and that the
  device id is the key. Adding a refusal here cannot be enforced (a hostile build
  ignores it), would block a documented legitimate case, and would create a rule
  a peer can bypass — so it is a **warning**, printed by the renaming device when
  another active member of the same network already carries the name (it holds
  the table to check), and nothing more.

### 3.3 What makes peers agree — one acceptance rule, no new op

The wire already carries the name twice (§2.2): `net_member_list` returns
`member.to_json()` for every row (`relay.py:_op_member_list`) and `net_epoch`
carries the member list. So no protocol change is needed — only a rule for when
an incoming name is *believed*.

**Rule N1 (acceptance).** A member row's `name` is changed on this device only by
(a) admission (`admit`, `relay.py:1672`), or (b) a member-table read whose
**sender is the device that row names**, or (c) this device renaming itself
(§3.1). Nothing else. Concretely:

- `adopt_members` gains one keyword, `sender_device_id`, and applies an incoming
  name only when `incoming.device_id == sender_device_id` and the incoming name
  is non-empty and differs. Its call site is `_learn_members_from`
  (`relay.py:5187`), where `link.device_id` is the authenticated peer.
- `_keep_local_authority` (`relay.py:1910`) preserves `name` alongside role and
  capabilities: `row.name = local.name or row.name`. A rotation is a statement
  about *authority*, and its frame is a third party's table — it may not rename
  anybody.
- `_take_over_standing` (`relay.py:1352`) flips to `local.name or successor.name`
  for the same reason: the rotation statement (`identity.rotation_statement`,
  `identity.py:371-393`) signs **ids and the new public key only — not a name**,
  so the incoming name is a third party's word.

Rule N1 is what makes the change safe *and* cheap: the sender's identity is the
link's authenticated `device_id` (established by the handshake and carried on
every frame), so no new signature, no signed statement, no new capability string,
no `MESH_PROTOCOL_VERSION` bump (`types.py:42`, which must not move for an op that
is not a handshake/transcript/framing change), and no new `OP_CAPABILITY` row.

**Convergence cost.** `refresh_membership` pulls a due table from every live link
(`relay.py:5210`), gated at `MEMBERSHIP_PULL_MIN_INTERVAL_S = 15.0` (`:252`) and
ticked at `MEMBERSHIP_PULL_PASS_S = 5.0` (`:262`), with `contact_peers`
(`:5293`) forcing a pass before `show`/`ls`/`peers` answer. So a rename reaches a
peer within ~20 s, or before the next listing, with no surface and no restart —
the same convergence story the membership table already has.

**Direction is irrelevant.** The pull runs over any live link regardless of who
dialled (`refresh_membership` walks `_link_for(member.device_id)`), so a
dial-only device behind a NAT still tells its peers its name through the link it
opened.

### 3.4 Online, offline, and whether the other side must accept

- **No peer has to be online.** The rename writes this device's own identity
  and its own rows; there is no queue, no retry and nothing to lose. The next
  contact carries it. A device the mesh can never dial keeps the name it was
  admitted under — the honest outcome, and the same one the table's staleness
  rules already accept.
- **The other side merely observes.** There is no approval, and that is a
  deliberate consequence of §2.2: making a peer *accept* a rename would make the
  name consensus state, requiring a signed statement, a doc revision and a
  quorum rule — a real protocol change for a cosmetic label. §4 states exactly
  what that buys and what it costs.
- **The renaming device does not need its relay running** (§3.1, second caller),
  only a filesystem it can write.
- **A rename never touches** the key, the device id, the epoch, the secret, the
  member's role or capabilities, or the placement document.

---

## 4. Security — answered plainly

### 4.1 What binds the name to the peer identity

- **At join**: the device key. `joiner_name` is inside the hello, which is inside
  the signed transcript (`handshake.py:133-153`, `:548-552`), and two humans
  compare a SAS over the same bytes.
- **After the join, today**: nothing. The name sits in each device's row,
  unauthenticated, and the membership digest deliberately excludes it
  (`relay.py:1613-1634`).
- **After this change**: the *link's authenticated device id*. A name update is
  only accepted from the device the row names, over a link whose keys were
  derived from that device's handshake. A third device — including a peer
  relaying somebody else's table — cannot rename a device on your view. That is
  the whole binding, and it is as strong as the handshake and no stronger: it
  proves *who said it*, never *whether the name is honest*.

### 4.2 What a hostile rename can actually achieve

Plainly: **it can make a device call itself anything, including the operator's own
machine name. It cannot gain authority — but it can misdirect a human and, today,
it can silently mis-target three verbs that resolve a device by name.**

1. **Label spoofing on every surface.** One row can read `operator-mbp` while
   another device already does. Mitigation that already exists: the device id is
   the key everywhere (`mesh-ui.md:938-939`), `lop network show` prints
   `device_id` beside every row (`cli.py:1508-1514`), the Mesh tab keys its graph
   nodes by `device_id` so two same-named devices are two nodes
   (`local-operator-ui`, `features/mesh/mesh-graph.ts`), and `KnownPeer.label`
   falls back to the id tail when unnamed (`peers.py:59`). I did **not** verify
   that a Mesh row surfaces the full id: by label alone a renamed impostor is not
   visually distinguishable, which is the residual risk and the reason §4.3's
   audit trail matters.
2. **Silent mistarget in `_resolve_device` — the real finding.** That function
   (`cli.py:2838-2858`) returns the **first** active row whose `name` matches.
   Its four callers are the authority-bearing ones: `member grant|revoke`
   (`:1570`), `credential share --with` (`:2201`), `credential revoke|mark`
   (`:2296`), `member rm` (`:2929`). With two rows sharing a name, an operator who
   copies the name a listing printed can grant capabilities to, revoke, or
   **share a credential with** the wrong device — and the share is sealed to that
   device's key, so it is a usable credential handed to the impostor. **This hole
   exists today** for join-time names (§16 Q8 permits duplicates and `join`
   never refuses one); a post-join rename does not create it, but it makes it
   arrangeable *after* the human's pairing ceremony instead of during it. **Fix
   it first, and ship the two together** (§7, P0).
3. **Audit-trail confusion.** Events carry device ids as `subject`/`actor`, so the
   log is not fooled; but any human reading a transcript that quotes a name can be
   misled, and the relay's refusal copy does quote names
   (`_peer_label`, `relay.py:9477`).
4. **No authority escalation.** Roles, capabilities, the authoriser, the
   handshake and the burn list all key on the device id and the public key
   (`types.py:1385-1391`; `mesh-transport-identity.md` §3.1: "Nothing
   authenticates on them"). A rename cannot add a capability, survive a
   revocation (a removed row cannot rename itself — it has no link), or change
   who a session belongs to.

### 4.3 The controls this design ships

- **No name-based authority resolution.** `_resolve_device` refuses an ambiguous
  name and names the candidate ids (the same answer `resolve_peer`
  (`peers.py:124-143`) already gives, and that `/new remote` already turns into a
  refusal at `tui/app.py:19542-19545`). This is the only change here that is
  really about security, and it is independent of the feature — it should land
  even if the feature does not.
- **Sender-scoped names** (Rule N1): no transitive rename, so a compromised peer
  cannot rename the rest of the mesh in your view.
- **A receipt.** The renaming device prints the old and the new name;
  the receiving device records `member_renamed` in its audit with the device id,
  old and new. Two devices agree about what changed and when.
- **Names stay cosmetic in the design record.** Nothing in this change makes a
  name evidence. If a future requirement needs a *provable* name (e.g. "this row
  was named by the operator who approved it"), that is a different change: a
  signed name statement carried with the rotation statement, accepted only when
  it verifies — §10.

**What would settle the residual uncertainty:** whether the operator wants a
rename to be *visible as a change* on the receiving Mesh tab (e.g. "was
`cloud-node-1`"). The audit record gives the back end an answer; whether the tab
renders it is a design decision for `mesh-ui.md`'s owner, not a fact I can derive.

---

## 5. UI and forward compatibility

**Yes — a display-name change is visible with no protocol change.** The name is
already in two payloads the peers receive:

- `net_member_list` → `{"members": [member.to_json() …]}` (`relay.py:_op_member_list`),
- `net_epoch` → `{"members": [...]}` (`relay.py:1580-1600`),
- and locally `net_show` → `members_detail[]` (`relay.py:12641`), which is what
  `GET /v1/desktop/networks` returns.

The Mesh tab needs **no change to display** a new name: `network_topology`
already emits `name` per member (`desktop_mesh.py:430` for the relay-down path and
`net_show`'s rows for the live one), and `deviceLabel` is `name || id tail`
(`mesh-types.ts`). A rename control in the tab would need a new route (a `POST`
that posts to *this device's* relay), which is P5 and optional — and note it can
only rename *this* device, because that is the only rename that is not a lie.

**Does an older peer ignore it safely?** Yes, in both directions:

- An older peer reads `name` (the field has existed since the record's first
  version) and stores it for rows it has never seen. It simply will not apply a
  name to a row it already holds, because it lacks Rule N1 — so it keeps the
  admission-time name. Degraded convergence, not a break.
- An older peer still applies `net_epoch` wholesale (`relay.py:2103`, `:2123`),
  so **it can still revert a rename on its own copy** from a stale table. That is
  a cosmetic divergence, not a mismatch: `members_digest_of` excludes `name`
  (§2.2), so no frame is refused and no link drops.
- A newer peer reading an older peer's table sees a row whose name is whatever
  that peer admitted them under — a legitimate claim, just old.
- **No version gate is needed and none should be added.** Adding one (`wire.py`'s
  capability intersection) would refuse the rename path to un-upgraded peers for a
  label; the existing convergence rules already degrade honestly.

One UI consequence to accept deliberately: §2.3(3)'s `_superseded` heuristic can
surface an extra revoked node after a rename. It is display-only; it should be
checked with a screenshot in QA, not "fixed" by weakening the heuristic.

---

## 6. Acceptance — metadata-only checks for independent QA

Preconditions: three devices on one scratch network, isolated `HOME` and
`LOCAL_OPERATOR_CONFIG_DIR` per `AGENTS.md`; A is the operator's device, B is the
node to rename, C is a third member used for the rotation check. No secret is
read or printed by any step.

| # | Surface under test | Command | Expected ACTUAL output | PASS criterion |
|---|---|---|---|---|
| 1 | Baseline | `B$ lop network identity show --json` | `.name` = the hostname (e.g. `ip-172-31-18-0…`) | proves the defect exists before the fix |
| 2 | Baseline (peer view) | `A$ lop network peers --json` | B's row `.name` = the same hostname | negative control: the reader reflects the record, not the command |
| 3 | The new verb | `B$ lop network identity set-name build-box --json` | `.name = "build-box"`, `.previous` = the old name | rc 0, both keys present |
| 4 | Identity durability | `B$ lop network identity show --json` | `.name = "build-box"` | the identity file, not just memory |
| 5 | Own member row | `B$ lop network show <net> --json` then read `.members_detail[]` for `B` | `.name = "build-box"` | the row moved, not just the identity |
| 6 | Peer convergence | `A$ lop network peers --json` (repeat once after ≤30 s if the last pull predates the rename) | B's row `.name = "build-box"` | the pull rule landed; `contact_peers` forces a pass before the answer |
| 7 | Peer member table | `A$ lop network show <net> --json` | `.members_detail[]` row for B = `build-box` | second read path agrees |
| 8 | Mesh tab payload | A's desktop route `GET /v1/desktop/networks` (or the tab itself) | the member entry for B carries `"name": "build-box"` | the display source, not a second table |
| 9 | Third peer | `C$ lop network peers --json` | B = `build-box` | not a two-device special case |
| 9b | Restart, renaming device | restart B's relay (`SIGTERM` the `lop network serve` pid, or `launchctl kickstart -k` / `systemctl --user restart`) then re-run #4 and #5 | unchanged | the write is durable |
| 9c | Restart, reading device | restart A's relay, then `A$ lop network peers --json` | B = `build-box` before any re-pull | it is in A's own record, not a cache |
| 10 | **Rotation does not revert it** | `A$ lop network member rm <net> <C>` (a rotation: `net_epoch` with A's table), then on B `lop network show <net> --json` and on A `lop network peers --json` | B is still `build-box` on **both** | Rule N1's `_keep_local_authority` half; this is the check that pins §2.3(1) |
| 11 | Offline rename | stop B's relay, run #3 on B with no relay answering, start it, then `A$ lop network peers --json` after ≤30 s | B = `build-box` | the no-relay path and the catch-up path |
| 12 | Digest invariance | `A$` and `B$ lop network show <net> --json`, compare `.members_digest` before and after the rename | identical on both devices, unchanged across the rename | proves a rename is not membership state (and why no epoch is owed) |
| 13 | Ambiguity refusal (P0) | rename C to A's name, then `A$ lop network member grant <net> <A-name> view` and `A$ lop network credential share … --with <A-name>` | **refused**, naming both device ids | no silent first-match; this is the §4.2(2) hole closed |
| 14 | Validation | `set-name ""`, a 65-char name, `a#b`, `$'\n'` | refused, sentence names the rule | no panic, no partial write |
| 15 | Collision warning | `B$ lop network identity set-name <C's name>` | rc 0, plus a warning naming C's device id | a warning, not a refusal (§3.2) |

Checks 10 and 12 are the two that fail if the design is implemented as a naive
local label; they are the ones to run first when something looks wrong.

---

## 7. Ranked implementation order

Each item is independently reviewable; P1 and P2 ship in one PR (P1 alone is
invisible to peers and §2.3(1) would revert it), but as separate commits so the
reviewer can see the write half and the acceptance half apart.

| Rank | Change | Why this rank | Size |
|---|---|---|---|
| **P0** | `_resolve_device` (`cli.py:2838`) refuses an ambiguous name, listing ids | It is the only security-relevant defect here and it exists **today**; the feature makes it arrangeable post-ceremony. Independent of the rename; ship it even if the feature is dropped. | S |
| **P1** | `rename_self` primitive + `lop network identity set-name` + `net_identity_name` local op (relay path updates `self.identity.name` and republishes `run/peers`) + validation + the `_guard_identity_subcommand` dispatch (`cli.py:9077-9088` — **without touching it, `set-name` silently prints `identity show` and exits 0**) | The command the operator asked for; the identity/row write is the feature. | M |
| **P2** | Rule N1 in `adopt_members` (+ `sender_device_id` at `relay.py:5187`), `_keep_local_authority` keeps `name`, `_take_over_standing` prefers the local name | Without it the peer never sees the rename (P1 is a no-op on the Mesh tab) and any rotation reverts it (§2.3(1)). Ships with P1. | S |
| **P3** | `_cmd_join` persists `--name` through `rename_self`, and the join block becomes `identity.name` unconditionally | Makes the join and the rename agree at the source, so a fresh node never has the two-copy split at all; also makes the approval-card name land in the identity, not only in the row. | S |
| **P4** | `member_renamed` audit on the receiving side (device id, old, new) and a receipt on the renaming side | The trail §4.3 promises; cheap and makes check 10 reviewable. | S |
| **P5** *(optional)* | Mesh tab "rename this device" control: a `POST` route that reaches **this device's** relay, plus a rendered frame of the new name | Only if the product wants the tab to own the action; it cannot rename a peer (§4.1) and must not pretend to. | M (other repo) |

**Not addressed by this note, deliberately:** no `member set-name`; no name
uniqueness; no new wire op; no new capability string; no `MESH_PROTOCOL_VERSION`
move; no change to `members_digest_of`; no signed name statement; no removal of a
name (`--clear`).

---

## 8. Change classification: **C2 (standard iteration)**

Justification against the escalation test (`minerva-software-development`
`references/sdlc.md`, "High-Risk Escalation Triggers"):

- **Posture** — it does not establish, replace or weaken a security architecture.
  It *tightens* two existing rules: a name is believed only from the device it
  names (Rule N1), and a name may not resolve to an ambiguous target (P0). Names
  remain cosmetic, exactly as `mesh-transport-identity.md` §3.1 states today.
- **Breadth** — one service (the mesh relay/CLI), one reversible metadata write,
  no schema change, no migration, no cross-product boundary.
- **Materiality** — failure or reversal costs display labels and a human's
  recognition of a device; no tenant isolation, confidentiality, integrity,
  availability or data-recoverability property moves. The one confidentiality-
  adjacent path (P0's `credential share` mistarget) is *closed* by this change,
  not opened.
- Rollback is a rename back, or restoring the record/identity file.

**What would make it C5, and does not apply here:** making the name an
authority-bearing identity fact — signed rename statements accepted from third
parties, approval-gated renames, or name-based resolution for grants, shares and
revocations. That is "redesigns the standard inter-service security model" rather
than "a component within it", and it is the alternative §10 rejects.

Two **genuine identity/protocol** flags, so nobody mistakes them for labels:

1. `identity rotate` (not this change) remains the only operation that changes
   *identity* — new key, new id, signed continuity statement
   (`identity.py:325-393`, `cli.py:8371`). A rename must never be routed
   through it, and this note does not.
2. Rule N1 is a new **wire semantic** — "a name may change after admission" —
   carried inside an existing payload. It is not a protocol change (no op, no
   version, no capability), but it is a behavioural change on a trust path and
   deserves review attention beyond a cosmetic PR.

---

## 9. Risks to watch during rollout

1. **The epoch-replace seam** — the single way this feature can silently regress.
   Watch acceptance check 10 on every candidate build; if `_keep_local_authority`
   is missed, every rotation reverts every rename and nobody notices until a
   member leaves.
2. **The two-process write class.** `store.py:36-52` already names `_cmd_rename`
   and `_cmd_identity_rotate` as writers outside the in-process lock. Adding a
   direct CLI write must follow the `net_member_caps` precedent (relay when
   answering, `store.mutate` when not) or the class grows by one by accident.
3. **Stale in-memory identity.** If the CLI writes the identity file behind a
   running relay, `status`/`peers`/the `unattended` remedy keep the old name until
   restart. The relay path in P1 exists for exactly this; a build that takes the
   CLI path unconditionally is the regression to look for.
4. **`_resolve_device` shipped late.** If P0 trails P1, the feature widens a hole
   that already exists. They go out together or P0 first.
5. **`_superseded` display churn** (§2.3(3)) — one extra revoked node after a
   rename; screenshot it, do not "fix" it by weakening the heuristic.
6. **Old peers revert on their own copy** (§5) — expected, cosmetic, no digest
   mismatch. If a build starts *refusing* frames over a name difference, the
   digest or the merge rule has been changed by mistake.
7. **Token churn** (§2.3(5)) — a `/new remote <oldname>` after a rename answers
   "No peer named …". Expected; the sentence is already correct and names the
   listing.

---

## 10. Rejected alternatives

| Alternative | Why rejected |
|---|---|
| `lop network member set-name <device> <name>` (relabel a peer) | It cannot be true: the peer's own assertion arrives on the next pull (Rule N1) and would overwrite it, so it would either lie or need a second field that silently disagrees with what the device says about itself. And it is the impersonation surface §4.2 describes — an operator renaming a peer to their own machine's name. Rejected on correctness first, security second. |
| A local-only label with no propagation | §2.3(1): any peer rotation reverts it. It would need `_keep_local_authority` anyway, and then two devices would silently disagree about who is who with no UI distinction. |
| A new wire op (`net_member_name`) instead of the pull rule | More surface for the same result: an `OP_CAPABILITY` row, a capability string, a dispatch entry, a version gate for old peers, and a push path that can fail silently. The wire already has "exactly one membership read" (`relay.py:5134-5160`); a rename is a membership fact and belongs in it. |
| Bump the epoch / write a new `NetworkRecord` revision for a rename | A rotation is authority and secret; a name is neither (§2.2). An epoch bump for a label would invalidate links and rotate a secret for cosmetics. |
| A signed rename statement accepted from anyone | The only way to make the name provable *transitively*; it also makes it authority-bearing, needs a verifier, a doc revision and an acceptance rule, and would be C5 (§8). Not needed while names stay cosmetic. |
| Route the rename through `identity rotate` (which already takes `name=`) | Rotates the key, changes the device id, rewrites every peer's row and needs the continuity statement — for a label. The opposite of the smallest change. |
| Make names unique | Unenforceable across devices, already rejected by §16 Q8, and blocks the legitimate "two laptops both called macbook" case `mesh-ui.md:938-939` contemplates. |

---

## 11. Evidence I did not gather (and what would settle it)

- **No wire capture.** I did not run a two-device rig; the convergence claim
  rests on `refresh_membership`'s schedule and `_learn_members_from`'s call site,
  which are cited above. It is proven by acceptance checks 6, 9, 11.
- **No rendered Mesh tab.** I did not screenshot the tab; the rendering claim
  rests on `network_topology`'s payload and `deviceLabel`. It is proven by check 8
  (and a design screenshot if P5 is taken).
- **No judgement from the operator on rename visibility** (§4.3). A rename that
  is *invisible as a change* on the receiving tab is the one product question this
  note cannot derive from the code; the audit record (P4) is the substrate either
  answer needs.
