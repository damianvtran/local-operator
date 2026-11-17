# Mesh listener authentication (G1, closing G2/G3/G4)

Status: design, source-only. Read at `origin/main` 1d493869f388 (2026-11-17).
Author: architect (lopdev team). Inputs: the trust-gap audit (G1–G4, `mitm_spike.py`),
`docs/design/mesh-transport-identity.md` §6.2. All `file:line` below are at
1d493869 and are relative to `local_operator/network/`.

**Operator direction (verbatim, relayed):** "trust the mesh network and have refresh tokens,
etc. across the mesh to keep connections long lived and alive but ensure that connections
can be revoked as needed … prevent man in the middle attacks or any other sort of injections".

## 1. Problem, as found in code

The handshake is `hello → challenge → auth → welcome` (handshake.py:8-12). Only the dialer
proves anything:

- The listener verifies the dialer properly (§2 below).
- The dialer copies `device_id` out of an unsigned, un-MACed `challenge` (handshake.py:525-541).
- `read_welcome` (557-576) checks only the frame name and the link version. It then takes
  `phase` and `endpoints` from a plaintext frame that sits outside the transcript, and calls
  `establish()`. Link keys come only from the ephemeral X25519 exchange and the transcript
  (792-834), so whoever answers holds matching keys.
- The docstring at 578-587 claims the opposite and is false.
- Relay dial path (relay.py:13585-13625): `send_hello → read_challenge → send_auth →
  read_welcome → establish → PeerLink → _admit_link(outbound=True) → _note_peer_endpoints
  → _pull_members`.
  - `_admit_link` (7861-7963) checks only stop/trust.
  - `expected_device` (13508) only feeds audit rows (13640, 13646). It is never compared
    with the peer.
- The CLI join dialer (cli.py:6121-6160) has the same gap.

Consequences (all confirmed by reading the code):

- **G2:** `_pull_members` runs on that unverified link. `adopt_members` (relay.py:1595-1610)
  says "the link already authenticates the answering peer", which is untrue for outbound
  links. A claimed admin may introduce rows (1652-1653, 1674).
- **G3:** `_broadcast_epoch` (13393-13421) sends `epoch_frame(target_device_id=link.device_id)`.
  That frame carries `secret` for any active id (1940-1950), including the eviction rotation.
- **G4:** welcome endpoints (handshake.py:574, 884-888) go through `_note_peer_endpoints`
  (relay.py:6654-6683) onto the claimed member's row, which durably redirects dials.
- **Also:**
  - An impersonated outbound link enters `self.links` and gets the claimed member's
    capabilities (forwards, credential borrow).
  - The dialer's duplicate fence (7887-7922) would DISPLACE the real outbound link to that
    device.

I did not re-run the spike. The audit reports it RED on a0e458d43, and the code path above
is unchanged at 1d493869.

## 2. Symmetric check: the listener side is already correct

`verify_auth` (handshake.py:692-788) checks, in order:

1. frame shape;
2. `device_id` matches the hello (704);
3. mode;
4. epoch ∈ {current, previous} (711-716);
5. ACTIVE member row in the current table, before any key is used (722-727);
6. join: the key's fingerprint equals the claimed id (729-738);
7. Ed25519 signature against the STORED key, label `lop-mesh-auth-v1` (741-745);
8. HMAC under the named epoch key or the invite key, using `hmac.compare_digest` (750-769).

Only then does it set `auth_verified` (777).

Two small notes:

- `accept_hello` checks the version pre-auth (604), despite the comment in wire.py:1108. This
  is harmless.
- Self-link is refused at hello (616-620).

No change is needed here beyond reusing the same helpers for the reverse direction.

## 3. The handshake change (decision)

### What the listener sends

`welcome` gains one field, `proof: {"sig": b64u, "mac": b64u}`. The listener computes it in
`welcome_frame` (handshake.py:857), only after `verify_auth` has succeeded (so
`self.credential` is the matched credential).

```
T          = transcript(hello, challenge, auth_core)          # existing, unchanged (133-162)
W          = canonical_json(welcome minus "proof")            # phase, epoch, device_id,
                                                              # instance_id, session_protocol,
                                                              # members_digest, capabilities,
                                                              # endpoints, nets, network, inviter
digest_L   = sha256(b"lop-mesh-accept-v1\x00" + lp(T) + lp(W))
proof.sig  = identity.sign(b"lop-mesh-accept-sig-v1\x00" + digest_L)     # DeviceIdentity.sign, identity.py:190
proof.mac  = HMAC-SHA256(self.credential.key, b"lop-mesh-accept-mac-v1\x00" + digest_L)
```

- **Signing key:** the listener's device Ed25519 key (`DeviceIdentity`). The dialer verifies it
  against `MemberRecord.public_key` (types.py:1498) of the ACTIVE row whose `device_id` equals
  `challenge.device_id`, read from the dialer's own record.
- **MAC key:** the SAME `Credential` the dialer MACed with.
  - Member mode: the epoch key of the epoch the auth named, i.e. `wire.epoch_key(secret, net,
    record.epoch)`, relay.py:13583.
  - Join mode: the invite key, relay.py:13580 / cli.py:6147.
- The MAC proves current-secret possession, so a removed device still holding its own
  Ed25519 key fails once the dialer has rotated. That is the revocation property the
  operator asked for.

### What the proof covers, and what it is checked against (security condition 3)

- **Coverage:** the proof covers the FULL transcript `T` and the welcome core `W`. `T` holds
  both hellos, both nonces, both ephemeral X25519 keys, the dialer's and listener's device
  ids, `network_id`, both epochs, caps and builds, and the auth core. `W` holds `endpoints`,
  `phase`, `epoch`, `device_id`, `instance_id`, `members_digest`, `nets`, `network` and
  `inviter`.
- **G4:** the welcome `endpoints` are therefore under both the signature and the MAC. A
  MITM's endpoint redirect fails at step 7 or 8, and nothing from `W` is read before step 9.
- **Identity check against the stored key:**
  - `peer == expected_device` is decided against the dialer's STORED member key, never the
    challenge's claim. The `challenge.device_id` comparison at step 5 only selects WHICH
    stored row to verify against.
  - The proof must then verify under that row's `MemberRecord.public_key` (step 6-7). A
    claim of `expected_device` without that device's private key fails `bad_signature`.
  - The challenge's claimed id is never itself the evidence, and no key is ever taken from
    the frame. The single exception is join: there the key comes from `welcome.inviter` and
    is accepted only when its fingerprint equals the TOKEN's `inviter_device_id` (§5).

### Why the MAC uses the epoch or invite key and not the derived link key

The link keys are HKDF(Z, th) (wire.py `link_keys`). `th` is inside `digest_L`, and both
ephemeral keys are inside the signed T, so the signature already binds the link keys.
Adding a key-confirmation MAC under `k_l2d` would prove nothing more. It would also mix a
transport key into an identity proof, which is a key-confusion risk.

### Transcript and SAS stay unchanged

T is unchanged, so the SAS, the link keys and the `link_id` derivation (812-817) are
byte-identical to today. The proof is an extra tag, not a new transcript element.

### Dialer verification order

Implement this as `Handshake.verify_welcome(frame, expect)`, called by `read_welcome` before
anything is copied from the frame. `expect` is a small `DialerExpectation(device_id,
public_key_for(id)->str|None, credential)`, alongside `ListenerPolicy` (338-361). Each step
raises `HandshakeRefusal`, the caller closes with no reply, and the dialer audits.

1. `net == "welcome"` and `check_link_version` (existing, 560-567).
2. `proof.sig` and `proof.mac` are present and well-formed → else `listener_unproven`.
3. `welcome.device_id == challenge.device_id` and `welcome.epoch` is an int → else
   `malformed_frame`.
4. `challenge.device_id != self_device_id` → else `self_link`.
5. Identity expectation:
   - member mode: `challenge.device_id == expect.device_id`. Make `expected_device` REQUIRED
     for member-mode `dial()`; both callers already pass it (relay.py:12553, 16097). Failure
     → new code `peer_identity_mismatch`.
   - join mode: `challenge.device_id == envelope.inviter_device_id` → else
     `invite_wrong_inviter` (reuse the `invite.REASON_INVITER` spelling, invite.py:65).
6. Key lookup:
   - member mode: the ACTIVE row's `public_key` for that id from the dialer's current record.
     A missing or tombstoned row → `not_a_member`.
   - join mode: `welcome.inviter.public_key` (new field; see §5) whose fingerprint
     (`identity.device_id_for`) equals `envelope.inviter_device_id` → else
     `invite_device_mismatch`.
7. `_verify_signature`-equivalent with the accept label → `bad_signature`.
8. `hmac.compare_digest` on the MAC → `bad_mac`.
9. Only now:
   - set `auth_verified = True` on the dialer side;
   - `peer_public_key = key` (this replaces the empty `_peer_key_from_welcome`, 578-587;
     delete that function and its false docstring);
   - `phase` and `peer_endpoints` from the verified frame;
   - `establish()`.

### HandshakeResult

`HandshakeResult` (types.py:2106) gains `peer_verified: bool`. `PeerLink` exposes it as
`PeerLink.peer_verified`. The name was fixed by security reviewer 60334 for both lanes.

**Semantics:** `peer_verified` is True ONLY when this handshake verified the peer's Ed25519
signature against the STORED key.

- **Inbound (listener):** the dialer's sig and MAC (`verify_auth`, set where `auth_verified`
  is set, handshake.py:777).
- **Outbound (dialer):** the listener's welcome proof AND `peer == expected_device` (step 9).
- **Join:** the key bound by the token fingerprint (§5).

`_admit_link` refuses any link whose `peer_verified` is False (§6).

### Refusal wording (dialer side, audit `detail` and dial reason)

- **`listener_unproven`:** "<name|id> at <host> did not prove its identity: its welcome
  carried no signature. It is running an older mesh build (<challenge.build.version>) or it
  is not the device it claims to be. This device no longer links to peers it cannot verify.
  Update that device: `lop network update <name>`, or run `lop-update` on it."
- **`peer_identity_mismatch`:** "the device at <host> answered as <got>, not <expected>; the
  address on <expected>'s member row is stale or redirected. No link was made."
- **`bad_signature` / `bad_mac`** reuse the existing sentences, with "listener" substituted.

### Audit cause map

Add the new codes to the cause map in `_audit_handshake_refusal` (relay.py:7759-7771):

| Code | Cause |
|---|---|
| `listener_unproven` | `protocol_mismatch` |
| `peer_identity_mismatch` | `wrong_device` |
| `invite_wrong_inviter` | `wrong_device` |

Keep the existing actor rule (7772-7790): a refused listener is a claim, never an actor.

## 4. Version and no-downgrade: recommend a capability, NOT a `MESH_PROTOCOL_VERSION` bump

This deviates from the audit's recommendation. The reason is in the code.

`check_link_version` is strict equality (wire.py:1105-1119). It runs on the listener at
**hello** (handshake.py:604) and on the dialer at welcome (567). Bumping `types.py:42`
from 1 to 2 therefore has two effects:

- **(a) Both directions between old and new builds fail.** New→old fails as a *silent EOF*:
  the old listener refuses at hello with no reply, and the dialer records
  `peer_closed_silently` (relay.py:13717), not a version message.
- **(b) The rollout channel breaks.** `lop network update` rides a mesh link (`net_update`,
  meshupdate.py:1-30). The target is the origin's own installed version
  (`origin_target`, meshupdate.py:217-235), so the origin updates first. Once its relay
  restarts on v2, it cannot reach any v1 member to update it. The relay does restart on
  update (verified, §4 "What happens to live links").

No-downgrade does not need a version number. It comes from the DIALER's rule being
unconditional: **a new dialer requires `proof` on every member and join welcome, whatever
the challenge advertises.** There is nothing to negotiate, so an attacker has nothing to
strip.

### Rule set

| Dialer \ Listener | old listener | new listener |
|---|---|---|
| **old dialer** | as today | Accepted. The new listener always attaches `proof`; the old dialer ignores unknown welcome keys (it reads only `phase`/`endpoints`, 568-574). The new listener still fully verifies the old dialer, so this costs the listener nothing. The old dialer stays vulnerable, which only updating fixes. |
| **new dialer** | REFUSED `listener_unproven`, naming the peer's build from `challenge.build` | verified |

- Advertise `listener-proof-v1` in `wire.LINK_CAPABILITIES`. This is for diagnostics only
  (`peers` can say "verifies listeners"). It never gates enforcement.
- Leave `MESH_PROTOCOL_VERSION = 1`. That is the documented rule: link features are
  capability strings (types.py:12-20).

### Rollout: how an updated device reaches a not-yet-updated member (security condition 1)

All citations are at `origin/main` 1d493869.

**What happens to live links when one side updates.** They do not survive: (b) is false.

- `lop update` (and `/update`) installs the new build, then `update.refresh_service_daemons_after_upgrade`
  (update.py:8306) restarts the relay. On macOS that goes through `relay.refresh_plist_if_stale`
  (relay.py:17073), and the relay restarts when its running build differs from the installed one
  (`launchd.restart_if_build_moved`). On Linux it goes through `relay.refresh_unit_if_stale`
  (17135).
- A member updated by `net_update` restarts its own relay after replying
  (`meshupdate.roll_services_after_reply`, meshupdate.py:1013-1037).
- The process exits, so every link on that relay drops. The far side gets EOF and runs
  `link_closed`, then `redial.lost` (relay.py:8170). Its `_redial_loop` (7139-7187) then
  re-dials **forever** with capped jittered backoff (redial.py:89-91, `wire.RECONNECT_MAX_S`).
  There is no give-up.

**Do running tasks survive the relink? Yes, for the turn itself.**

- A session's turn runs in its own runtime process, not in the relay.
- `_op_forward` (relay.py:9942-9985) is a per-op carrier. Each forwarded frame is a fresh
  `_dial_owned(session_id, …)` plus `exchange` to the local runtime (9964-9971), so no turn
  state lives on the link.
- The generation layout keeps a busy runtime on the tree it loaded until it is idle
  (meshupdate.py module docstring, "On the immutable GENERATION layout").
- What is lost is only an op in flight on the dropped link. Its requester gets a timeout or
  `session_unreachable` and re-issues after the redial. This is the same behaviour as any
  relay restart today.
- Verified by reading only. No new test is needed beyond the existing relay-restart coverage.

**Which side dials, and which op carries the update.**

1. `lop update` opens a rollout record (`rollouts.open_after_update`, rollouts.py:1386).
2. The record is driven by the **origin's relay**, which is already the NEW build, since it
   restarted above. The driver thread (`mesh-rollout`, rollouts.py:1-30) calls
   `meshupdate.update_peer` (rollouts.py:1011).
3. `update_peer` (meshupdate.py:1509-1557) calls `server._ensure_link_with_reason(device_id)`.
   - That first returns ANY live link to the device via `_link_for`, inbound or outbound
     (relay.py:12385-12396, 12434-12436).
   - Only if there is none does it dial.
   - On a live link it sends ONE request, `op: net_update` (meshupdate.py:1597-1605).
4. If there is no link, it returns `code: "unreachable"`. The driver parks the member and
   re-asks every `CADENCE_S` = 300 s (rollouts.py:94-96, 798-810) until the record's window
   closes. Nobody is prompted.

**Chosen path: (a) the old member dials in, so the roll rides the inbound link. It works with
the code as it stands.**

- The old member's relay lost its link when the origin restarted, so its `_redial_loop`
  re-dials the origin within the backoff ladder.
- That is an OLD dialer meeting a NEW listener, which this design accepts (Rule set). The new
  listener verifies it fully (§2), so the inbound link has `peer_verified=True` and passes
  `_admit_link`.
- The driver's next 300 s re-ask finds that link through `_link_for` and sends `net_update`
  over it. Links are bidirectional; the inbound link's request path is the same `PeerLink`.
- The member installs, replies, restarts, and comes back as a new listener. From then on the
  origin's own dials verify.

**Required code changes for (a):** none in the roll. Two small mappings:

- `update_peer`'s unreachable sentence must carry the `listener_unproven` gloss. It already
  appends `peer_reason_words(dial_reason)` (meshupdate.py:1546-1557), so this needs only the
  gloss table entry (Surfacing, below).
- `classify_answer` already treats `unreachable` as "re-ask at cadence", which is correct.

**Where (a) cannot work: the member can never dial the origin.** This happens when the origin
is a dial-only install (it advertises loopback, `advertise_endpoints`, and members record
`waiting_for_peer`, relay.py:7172-7178), or when a firewall blocks member→origin. With (a)
alone, such a member never updates without a hand step, and that violates "no repeated
bother". For that case only, add **(c): a one-shot, unadmitted update exchange**, scoped as
follows:

- **Trigger:** only inside `update_peer`, only when `_ensure_link_with_reason` returned
  `listener_unproven`. No other caller can construct it. The `allow_unproven` flag is a
  keyword on a private `RelayServer._legacy_update_exchange(device_id, endpoint, frame,
  timeout)`, not on `dial()`.
- **What it does:**
  - re-dials that endpoint;
  - runs the handshake through `send_auth`;
  - reads `welcome` WITHOUT requiring `proof`, and builds the codec;
  - seals exactly ONE `net_update` request (the target identity `{version, source_ref}`, which
    is public on PyPI, plus the rollout id);
  - reads sealed frames until the reply with the matching `req` or `UPDATE_OP_DEADLINE_S`;
  - discards and does not dispatch ANY other frame;
  - closes.
- **What it never does:** no `PeerLink`, no `_admit_link`, no `self.links` entry,
  `_note_peer_endpoints`, `_pull_members`, `redial.connected`, link-up hooks, or inbound op
  dispatch. It cannot carry a prompt, an ask, a credential, an epoch frame or a member row,
  because no code path reaches those senders without a `PeerLink`.
- **Operator authority: none.** It uses the device's own rollout grant, which the member's
  authorizer already checks, and no human input.
- **The reply is UNVERIFIED.** The rollout record moves to `applying` (rollouts.py:1004-1008),
  never `done`, on this reply. `done` comes only from the next verified contact answering
  `already_on_target`, which is the existing re-probe path. An impersonator can therefore
  learn a public version string and delay that member's line by one ask. It cannot mark the
  member done, write anything durable, or receive anything secret.
- **Audit:** `update_member_state` with `cause: "legacy_listener"`, naming the endpoint.
- **Sunset:** delete (c) in the first minor release after this one. Every member that has
  been updated presents `listener-proof-v1`.

**Evidence that settles whether (c) is needed:** on the operator's origin, whether
`advertise_endpoints` is loopback-only and whether each member's row for the origin holds a
dialable endpoint. If every member can dial the origin, ship (a) alone and drop (c). The
manager or coder should check this read-only, because it is live config. If it cannot be
checked, ship (c): it is the path with no hand step.

Option considered and rejected: (d), a stated manual step. It is correct but fails "no
repeated bother", and the rollout record already exists to avoid it.

### Surfacing (security condition 2): a named, sticky line, never hidden by redial

Where dial-failure state lives today:

- `RedialTracker.failed(device_id, cause=reason)` (redial.py:152-162) stores `last_cause`.
  `snapshot()` (179-200) exposes it as `last_error`.
- `_redial_loop` feeds it (relay.py:7186-7187).
- `peers()` copies the snapshot into each row's `redial` block (relay.py:15746-15750). The
  doctor's per-endpoint probe row carries `detail=refusal.code` (16091-16100).
- Human lines gloss reason tokens through `resume.peer_reason_words` (resume.py:3179; used by
  `peers`, the sidebar and meshupdate.py:1548). Doctor uses `doctor_detail_words` (named in
  that docstring).
- The row's `build` is filled only from a live link (relay.py:15758-15762), so a refused peer
  shows `{}`.

**The problem:** `last_cause` is overwritten by every later failure (`cause or last_cause`,
redial.py:162). After one `listener_unproven`, a later `connect_failed:…` or `not_attempted`
would replace it. The specific line would then flap or vanish under the reconnect loop.

**The fix:**

1. **A sticky per-device fact, separate from the redial tracker.**
   `RelayServer._unproven_listeners: dict[device_id, {"build": challenge.build, "endpoint":
   host, "at": ts}]`.
   - Written in `dial()`'s `MeshRefusal` arm when `refusal.code == "listener_unproven"`.
   - Cleared ONLY when `_admit_link` admits a link to that device with `peer_verified=True`
     in the OUTBOUND direction, i.e. the peer has now proved itself as a listener.
   - An inbound link from an old member does not clear it, because the member is still old
     as a listener. That inbound link's hello `build` then refreshes the stored build: the
     hello is verified by the listener, so the build is no longer a bare claim.
   - Not persisted; it re-learns after a restart on the first dial.
2. **The build named.** For the dialer, the peer's build arrives in the **challenge**
   (`build_challenge(build=…)`, handshake.py:281-303; read at 540). The challenge is
   unauthenticated, so the line says "reports build X".
   - When an inbound link from that member exists, use its **hello** build instead
     (handshake.py:628), which the listener verified. Say "runs build X".
   - Render `version` only (the `update.BuildStamp` shape), never `source_ref` paths.
3. **`peers()` row:** add `"listener_proof": {"state": "unproven"|"verified"|"unknown",
   "build": {...}, "at": ts}` next to `redial`. It is present on every row, in the same
   absent-value rule as `build` (15751-15757).
4. **Human lines:**
   - `lop network peers` prints, for every row with state `unproven`, a dedicated line that is
     not part of the reachability parenthetical: `<name> runs an older build (<version>) that
     cannot prove its identity, so this device will not dial it. Update it with: lop network
     update <name>`.
   - When an active rollout record covers the member, append `(rollout <id>: waiting for it
     to connect)`.
   - `lop network doctor` emits the same sentence as a named finding, severity warn, on the
     member.
5. **Gloss table:** add `listener_unproven` and `peer_identity_mismatch` to
   `peer_reason_words` and `doctor_detail_words`. Their tests walk the vocabulary from the
   source, so a missing entry fails CI by itself (resume.py docstring, "The tests walk that
   vocabulary").
6. **Never hidden by backoff:** the line is driven by (1), not by `redial.last_error`. A
   reconnect loop or the `down` escalation therefore cannot change or drop it until a
   verified outbound link exists.
7. **No new prompt, notification or repeat:** the line is listing and doctor output only.
   The audit row is coalesced by the existing `_refusal_rows.admit` key (relay.py:7790-7800),
   which is one row per window.

### Option B (if the manager insists on a bump)

Bump to 2, but make the listener accept hello `v ∈ {1, 2}` and echo the dialer's `v` in its
welcome. The dialer requires `v == 2` plus `proof`. This works, but it costs two spellings
of one fact for no security gain over the capability rule.

## 5. First contact (join): the token already binds the listener

The invite envelope carries `inviter_device_id` (invite.py:83, 103, 255). It is the
fingerprint of the inviter's Ed25519 key (`identity.device_id_for`), so no token format
change is needed.

**Inviter side (`welcome_frame` pair branch, handshake.py:881-883):** add
`inviter.public_key = identity.public_key`. It is under `proof.sig`, and its fingerprint is
checked against the token.

**Joiner side:**

- `proof.mac` is under the invite key. A token holder can produce it, and so can **every
  member of the network**: the invite key derives from the network secret plus the
  plaintext `invite_id` (`invite.invite_credential_for`). The MAC therefore proves "a token
  holder or a member", not "the inviter" (corrected in round 1, review N-1).
- `proof.sig` must verify under a key whose fingerprint equals the token's inviter id. **This
  is what join security rests on.**
- To fake the inviter, an attacker therefore needs the inviter's private key (plus the token
  or membership for the MAC).
- This newly protects `--automated` joins, where `typed = result.sas` (cli.py, the
  `automated` branch) means no human compare happens. Today the SAS is the only MITM
  defence, and automated joins skip it.

**Pin:** the joiner keeps the verified inviter key. When `_finish_pairing` persists the
`pair_result.members` list (handshake.py:978-982), it refuses (`invite_wrong_inviter`) if
the inviter's row `public_key` differs.

**The honest SAS path is unchanged:** T is unchanged, so the digits are unchanged.

**Callers to update:**

- `cli.py:6121-6160`: pass `DialerExpectation(device_id=envelope.inviter_device_id, …)`.
- relay.py `dial(mode="join")` returns before use (13594-13601), but it should still verify
  for uniformity.

## 6. G2 / G3 / G4

### One structural gate instead of per-consumer gates (G2, G3, forwards, credentials)

`_admit_link` (relay.py:7861) is documented as "THE ONE WAY INTO `links`". At its top, before
the duplicate fence, add:

```
if not getattr(link.result, "peer_verified", False): return "unverified_link"
```

This closes, by construction, every link consumer: `_pull_members`, `_broadcast_epoch`,
`_flush_outboxes`, `_op_forward`, credential lending, and the fence. A refused link never
reaches `redial.connected` or the link-up hooks (7929, 7954).

### Defence in depth, kept small

1. `adopt_members` (1595): add a parameter `supplier_verified: bool`. It returns
   `(False, [])` when that is false, and its docstring claim becomes true. Plumb it from
   `_adopt_member_rows` (≈6919-6931) using `link.result.peer_verified`.
2. `epoch_frame` callers in `_broadcast_epoch` (13414-13416): skip and QUEUE any link without
   `peer_verified`. This gives the regression test a precise assertion, and a skipped link
   costs only a queue entry.

Do not add any other gates.

### G4: endpoints only after proof

`_note_peer_endpoints` on the dial path (13618) already runs after `read_welcome`. After
this change, `read_welcome` raises before endpoints are set, so ordering alone closes G4.
Add an assertion-style guard: `_note_peer_endpoints` takes `verified: bool` and returns
early when it is false.

### G4: re-learning poisoned rows

There is no provenance field to sweep by. `MemberRecord` has `endpoints` and
`observed_endpoint` but no source (types.py:1509, 1519). Only welcome-sourced `endpoints`
could have been poisoned:

- `observed_endpoint` is written only from inbound links, which were always verified
  (6685-6730).
- Hello endpoints are inside the transcript.

**Recommendation:** no migration and no sweep. Heal on negative proof, but delete an address
only on AUTHENTICATED evidence (revised in round 1: review M1, security S-2):

- **`bad_mac` only demotes.** It is reachable only after a signature that verified under M's
  stored key. In `dial()` it removes E from `M.endpoints` under `store.mutate`, never when E
  is `observed_endpoint` and **never M's last address**. Audited as `endpoint_demoted`.
- **`peer_identity_mismatch` and `bad_signature` are unauthenticated** (a bare challenge, or
  64 random bytes), so they write nothing durable. They put E in the relay's in-memory
  `_suspect_endpoints[M]`. `_ensure_link_with_reason` then dials M's other addresses first and
  falls back to the suspects within the same attempt. A verified outbound link at E clears it.
  - This keeps the original concern: `probe_candidates` takes the FIRST connect, so a nearby
    attacker or a stale address would otherwise win every race and starve the real candidate.
    It now does so without letting one frame strand a member that has a single address.
- **`listener_unproven` does NOT demote.** That is the expected answer of a legitimate
  not-yet-updated peer, and demoting would strand it after it updates. A row an attacker
  holds by omitting proof stays a DoS only (no link). It heals when M dials in (hello
  endpoints, authenticated) or a verified peer relays M's row (`_adopt_relayed_metadata`).
- Every demotion is audited with the endpoint, so the operator can see whether any
  poisoning ever happened.

Evidence that would change this: if the audit log shows past `_note_peer_endpoints`
writes from an unexpected host, add a one-shot `doctor` listing rather than a silent sweep.

## 7. Test matrix

RED-on-main means the assertion fails at 1d493869.

| # | Level | Cell | main | fixed |
|---|---|---|---|---|
| T1 | handshake | `mitm_spike` shape: fake listener with no epoch key and no member key claims victim id | accepted, matching `k_d2l` (RED) | `listener_unproven` |
| T2 | handshake | fake listener signs `proof` with its OWN key and claims victim id | accepted (RED) | `bad_signature` |
| T3 | handshake | genuine member key but no current epoch key (removed device, dialer has rotated) | accepted (RED) | `bad_mac` |
| T4 | handshake | MITM rewrites `welcome.endpoints` / `phase` after a valid proof | accepted, endpoints adopted (RED) | `bad_signature` |
| T5 | handshake | reflection: dialer's own `auth.sig`/`mac` replayed as `proof` | n/a | `bad_signature` (label separation) |
| T6 | relay | honest member B answers at A's address, `expected_device=A` | link as B, endpoint written to B's row from A's dial (RED) | `peer_identity_mismatch`, A's E marked suspect in memory, row unchanged (round 1) |
| T7 | relay | impersonated admin listener | link in `links`, `_pull_members` adopts rows (RED) | no link, record bytes unchanged, `adopt_members` not called |
| T8 | relay | `member rm` rotation while an impersonated link would exist | `net_epoch` with `secret` sent (RED) | nothing sent to it; real member's frame queued |
| T9 | relay | existing verified outbound link to A, then an impersonated dial to A | real link DISPLACED (RED) | real link untouched, no `redial.connected` |
| T10 | compat | new listener ↔ old dialer (dialer ignoring `proof`) | — | link up |
| T11 | compat | new dialer ↔ old listener (welcome without `proof`) | accepted (RED) | `listener_unproven` naming `challenge.build.version` |
| T12 | join | fake inviter with no token | proceeds to SAS (RED in `--automated`) | `bad_mac` |
| T13 | join | token holder with a different key (fingerprint ≠ token inviter) | proceeds (RED) | `invite_device_mismatch` |
| T14 | join | honest join, same SAS digits as before (pin a vector) | pass | pass (SAS unchanged) |
| T15 | join | `pair_result` inviter row key ≠ verified welcome key | adopted (RED) | `invite_wrong_inviter` |
| T16 | unit | `_admit_link` with `peer_verified=False` | admitted (RED) | `unverified_link` |
| T17 | unit | `adopt_members(..., supplier_verified=False)` | n/a | `(False, [])` |
| T18 | relay | old dialer (no proof check) → new listener; origin `update_peer` finds the INBOUND link via `_link_for` and sends `net_update` | works | works (cond. 1, path a) |
| T19 | relay | new origin, old listener, member cannot dial in: `update_peer` → `listener_unproven` → `_legacy_update_exchange` sends exactly one `net_update`; assert no `self.links` entry, no endpoint/member write, other sealed frames ignored, record stays `applying` | n/a | pass (cond. 1, path c) |
| T20 | relay | impersonator answers the legacy exchange with a fake `done` receipt | n/a | record not `done`; audit `legacy_listener` |
| T21 | relay | `listener_unproven`, then 3× `connect_failed` from the redial loop: `peers()` row still has `listener_proof.state == "unproven"` with the challenge build; human line printed once | n/a (RED: no field) | pass (cond. 2) |
| T22 | relay | a verified outbound link later admitted clears the sticky state; an inbound-only link does not | n/a | pass |
| T23 | unit | `peer_reason_words` / `doctor_detail_words` vocabulary walk includes `listener_unproven`, `peer_identity_mismatch` | RED (unknown code) | pass |

**New test file:** `tests/unit/network/test_handshake_listener_auth.py` (already started in this worktree; extend it) (T1-T5, T10-T17, T23; T18-T22 as relay cells; T1 mirrors the
audit's `mitm_spike.py`).

**Relay cells T6-T9:** in `test_relay_e2e.py` style, or the same new file using `RelayServer`
pairs on loopback with synthetic keys.

**Neighbour files to run serially** (scripts/run_bounded.py, -n0, isolated HOME):

- `test_handshake.py`
- `test_relay_e2e.py`
- `test_join_automated.py`
- `test_join_park.py`
- `test_pair_offer.py`
- `test_pairing_confirm.py`
- `test_membership.py`
- `test_membership_convergence.py`
- `test_membership_change_push.py`
- `test_member_rm_already_revoked.py`
- `test_endpoint_probe.py`
- `test_dial_allowlist.py`
- `test_dial_journal_order.py`
- `test_relay_stop.py`
- `test_wire_crypto.py`
- `test_reason_surfaces.py` (gloss vocabulary walk)
- `test_rollouts.py` (`classify_answer`, driver re-ask) and any `meshupdate` test covering `update_peer` (grep `update_peer` under tests/unit/network)

These hand-built listener/welcome fixtures (they grep for `read_welcome|welcome_frame`) WILL
break until they produce a proof. That is expected. Fix the fixtures with a shared helper;
never weaken the dialer.

## 8. Risks a reviewer must check

1. **Constant-time compare.** Use `hmac.compare_digest` for the MAC, and base64-decode errors
   → refusal, not a 500. Ed25519 `verify` raises `InvalidSignature`; catch `ValueError` too,
   as 913 does.
2. **Domain separation.** Use three distinct labels (`accept-v1` digest, `accept-sig-v1`,
   `accept-mac-v1`), never the dialer's `lop-mesh-auth-v1` / `lop-mesh-mac-v1`, so neither
   side's tag can be reflected (T5).
3. **Key confusion.** The signing key is the device Ed25519 key, never the epoch secret. The
   MAC key is the epoch or invite key, never `k_d2l`/`k_l2d`. The verification key comes
   from the DIALER's stored row (or the token fingerprint for join), never from the frame.
   The single exception is join `inviter.public_key`, and it is only accepted after the
   fingerprint check.
4. **Canonicalisation.** W is `canonical_json` of the parsed frame minus `proof`, as the
   listener already does for `auth_core` (703). Watch for non-JSON-stable values (floats,
   key order) in `nets`/`capabilities`. Add a test that round-trips a real `welcome_frame`.
5. **Replay.** T contains the dialer's fresh nonce and ephemeral key, so a recorded proof
   never verifies on a new handshake. Do not cache `digest_L`.
6. **Ordering.**
   - Nothing may read `phase`, `endpoints`, `inviter` or `nets` before step 9.
   - `peer_device_id` may be set at `read_challenge`, but `auth_verified` and audit actors
     must stay "unknown" until step 9 (relay.py:7772-7790).
7. **Duplicate fence (7887-7922).** The refusal must happen before `PeerLink` exists, so
   `displaced` is never computed for an unverified link (T9). Also check that cross-dial
   behaviour is unchanged.
8. **Epoch skew.** A dialer still at epoch N-1 may accept a removed device that holds key N-1
   until the dialer learns the rotation. This is the same window the listener direction
   already has, so it is not a regression. Flag it, don't fix it here.
9. **Line budget.** `proof` adds about 150 B to welcome. `MAX_HANDSHAKE_LINE` is 16 KiB
   (wire.py:61), so this is fine. Assert with a test.
10. **Rollout.** Enforcement is immediate, and links from updated → not-updated devices
    drop. Follow the §4 runbook order. Expect `listener_unproven` rows in `peers` until the
    fleet updates. They are the intended signal, not a regression.

## 9. Round-1 dispositions (PR #4179)

The implementation added one step beyond §3: **a challenge naming an unexpected device is
refused before the dialer sends `auth`** (`Handshake.refuse_unexpected_peer`). An honest
member at a stale address would otherwise verify our auth, register the link, and its
duplicate fence would evict our real link to it. It also keeps our auth tag away from the
wrong party.

| Item | Disposition |
|---|---|
| G1D-1 / S-1: key rotation while a peer was offline | **Fixed, rotation-proof design.** A rotated listener sends its `rotation_proof` (its self row's statement, signed by its previous key) in the challenge as `rotation`. When `challenge.device_id` ≠ expected, the dialer continues to `auth` only if the statement names the expected id as `old_device_id`, the challenge id as `new_device_id` and this network, and verifies against the dialer's STORED key for the expected id (`verify_rotation_statement`). The welcome proof must then verify under the statement's `new_public_key`. `dial()` applies the hop through `apply_device_rotation` (the `net_identity_rotate` path, shared with `_op_identity_rotate`) before admitting the link. The statement is inside the transcript, so the proof covers it. It is public data, so replaying it gains an attacker only the auth frame a matching-id impostor already gets, and it then fails `bad_signature`. Limit: one hop. A device that rotated twice while the peer was away still needs the table route. With the hop applied, A' → D also links (the second half of S-1). |
| G1D-3 / M1 / S-2: unauthenticated demotion | **Fixed.** See §6 "re-learning poisoned rows": only `bad_mac` deletes, never the last address. Mismatch and `bad_signature` only reorder dials in memory. |
| G1D-5 / U1 / m1 / S-3 / Q-1: sticky unproven state | **Fixed.** An inbound link with `listener-proof-v1` in its MAC-covered caps clears the record. `peers`/`doctor` report `verified` for a live outbound or proof-capable link before reading the record. A doctor probe does not re-set the record while such a link is live. |
| G1D-2 / U2 / S-4 / S-5 / N2 / N-2: remedies | **Fixed.** A join through an old inviter says to run `lop update` on the inviting device and mint a new invite there, with no token tail. Member wording names both `lop network update` (shell-quoted) and `lop update` on the device itself, and no literal placeholder remains. |
| G1D-6 / N-1: invite MAC claim | **Corrected** in §5 and at the `cli._join_one` comment. |

