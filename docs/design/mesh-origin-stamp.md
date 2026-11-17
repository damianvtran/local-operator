# Mesh origin stamp (G5) and relay→runtime proof (G6)

Status: PROPOSED · lane: remote-parity (79bddc262663) · security reviewer: 60334a0b4c12
Base: origin/main `1d493869f`; every `file:line` below is at that HEAD.
Scope: produce and preserve a stamp and a proof; nothing consumes them, and
today's behaviour stays until G1/G7 land (operator, relayed by 60334). One
stricter side effect (M-12): the steer/prompt merge (§4) keeps `lop_remote_prompt`
on harness-injected remote rows, which `_note_run_input` reads to refuse card-free
profile saves (`session.py:417-423`). It can only refuse more.

## 1. Problem

- Remote prompt: one boolean `provider_payload["lop_remote_prompt"]`
  (`local_operator/session/session.py:424-430`), set when `owner_typed=False`
  (`session.py:10615-10616`, `10791-10792`), i.e. `authority.locality != "local"`
  (`session/runtime/server.py:9081-9088`, `9128-9130`).
- Remote ask answer: `answered_by.surface="remote"` for ANY non-local connection
  (`asks/surfaces.py:117-128`, via `server.py:162-197`); no device/network/link is
  stored (`asks/store.py:839`, `891`).
- With `harness_injected`, prompt (`session.py:10626`) and steer
  (`session.py:10793-10800`) OVERWRITE `provider_payload`; spool keeps only
  `{"remote_origin": True}` (`session/runtime/serving.py:629-637`, `process.py:5324-5325`).
- `_peer_block` = `{device_id, name}` (`network/relay.py:9892-9899`) reaches the
  runtime as self-declared auth fields (`network/dial.py:369-384`) behind only
  `control_key` (`server.py:5288-5292`, `5315-5330`), readable by any same-uid
  process (`asks/surfaces.py:7-11`).

## 2. Stamp contract — PROPOSED (names provisionally agreed with 60334)

One object under one new key in both places:
row → `message.provider_payload["lop_mesh_origin"]`;
answer → `record["answered_by"]["mesh_origin"]`.

| key | type | meaning (all set by the RECEIVING runtime) |
|---|---|---|
| `origin_device_id` | str | the link's peer `device_id` (`relay.py:9897-9899`) |
| `network_id` | str | `link.network_id`; new in `_peer_block` |
| `link_id` | str (hex) | `link.link_id`, handshake-derived (`network/handshake.py:813-818`) |
| `capabilities` | list[str], sorted | set the relay resolved from the live member row (`relay.py:4976-4989`), sent at `dial.py:377-378` |
| `verified` | bool | `proof_ok AND block.peer_authenticated` (§2 rule 3) on THIS connection |
| `stamped_by` | str | `device_id` of the device whose runtime minted the stamp (this install's `identity.load().device_id`, `network/identity.py:237`). Added by M-4. |
| `operator_origin` | bool, OPTIONAL, absent by default | G7 slot (60334); this PR never writes it. Constraints in §2.2. |

Rules:
1. **Never from a frame.** Built from `_ConnectionAuthority` (`server.py:2119-2160`).
   A frame-supplied `lop_mesh_origin`/`mesh_origin` is stripped from prompt, steer
   and answer frames, and `_stamp_peer_sender` (`server.py:2158-2183`, which today
   pops only the device keys at `2172-2173`) also pops `mesh_origin` and
   `lop_mesh_origin` (M-2).
2. **Unproved non-local connection:** stamp written with `verified:false` and the
   claimed identity (audit only). Local connections write no stamp.
3. **Peer authentication is a per-link fact in the PROVED block (M-1).** `PeerLink`
   gains `peer_authenticated`, copied from `handshake.auth_verified`
   (`handshake.py:440`), set only by `verify_auth` after it checks the PEER's
   signature and MAC (`handshake.py:777`): today inbound links only, since a dialer
   reads the listener's id unverified (`handshake.py:531-536`); G1 sets it on the
   dialer after verifying the listener. It is MAC-bound in the G6 block (§3.1) and
   `verified = proof_ok and block["peer_authenticated"] is True`. No runtime-side
   constant, so a post-G1 runtime dialled by a pre-G1 relay still gets `false`.
4. **Answer merge (M-8).** The store keeps the FIRST answer's `by` and folds
   revisions only into `surface` (`asks/store.py:878-891`). The mesh stamp is
   folded too: `verified` = AND over the first answer and every revision, where a
   revision with NO stamp (a local unproven dial) or `verified:false` counts as
   false, and differing `origin_device_id`/`stamped_by` ⇒ false. A verified "No"
   revised to "Yes" by an unproven dial therefore reads `verified=false`.
5. **Legacy:** a row or record without the key is UNVERIFIED (all pre-PR rows,
   older-build spool lines).
6. **`verified` = "this DEVICE over an authenticated link", never "a human" (M-3).**
   A phone or tunnel never verifies as itself; relayed through device B it verifies
   as B: B's phone → `admit_command` (`mobile/remote_sessions.py:574-614`) →
   `open_remote_viewer` (`session/remote_open.py:259`) → A's `_accept_stream`/
   `_op_forward` → `dial_owner(peer=_peer_block(link))` (`relay.py:12235-12241`,
   `9966`). Likewise B's `_pilot_act` (`network/cli.py:8091`), `network sessions
   --send` and B's agents' `send`/`sessions`. A's own phone daemon (non-`attach`, no
   relay proof, `asks/surfaces.py:123-128`) and tunnel (relay harness port only,
   `tunnels/gateway.py:854-888`) never stamp `verified`. Only G7 (§2.2) separates
   a human from an agent on B.

### 2.1 Consent-time re-derivation (60334 binding note 1)

The stored `verified` is evidence, not a verdict. New stdlib-only helper in
`local_operator/network/origin.py`:

```python
def mesh_origin_counts(stamp: Mapping | None, *, root: Path, self_device_id: str) -> bool:
    """True iff a delivery stamp still counts as an authenticated mesh device NOW."""
```

It reads the LIVE record itself with `store.load(network_id, root)`
(`network/store.py:886`; not `StoreView.network`, which catches only
`FileNotFoundError`, `relay.py:3958-3962`); ANY exception ⇒ False (M-7). True only if:
- `stamp["verified"] is True` and `stamp["stamped_by"] == self_device_id` (M-4);
- `record.trust == "active"` (the `_refusal_reason` term, `relay.py:7834-7859`);
- `row = record.member(origin_device_id)` exists, `row.device_id == origin_device_id`
  (M-7: `member()` also matches `previous_ids`, `network/types.py:1903-1907`, and a
  rotation may answer a compromise) and `row.active` (`types.py:1543`; false once
  `removed_at` is set, `types.py:1539`);
- `origin_device_id not in record.removed_ids`.

`record.epoch` is logged, not gated: revocation is the tombstone + `removed_ids`
(`relay.py:2365-2410`), and an unrelated rotation must not void an active member's
rows. **Consumer:** the guard (60334). This PR ships the helper + tests, no caller.

### 2.2 G7 requirements (60334) — constraints on the `operator_origin` slot only

(a) Per FRAME, not per connection (a stream pumps many frames, `relay.py:12220-12250`).
(b) Content-bound: digest of (op, session_id, text or ask_id+answers, command_id).
(c) B's relay sets it only when B's local client proved operator authority on that
    frame (`request_proof_ok` shape, `harness/approval.py:781-795`); never from
    `send`, `_pilot_act` or the mobile daemon unless the phone proved its own.
(d) Inside link-AEAD frames, folded into A's G6 MAC; trust bound: B's relay is honest.
(e) Optional end to end: the surface signs the frame digest with a non-biometric key
    certified under the operator anchor (`server.py:2366-2385`, `_device_cert_point`
    `6942`).
(f) At rest it inherits M-4/M-5 unless consumed in memory at delivery or gated on
    `stamped_by == self`.

## 3. G6 — relay proves itself to the runtime, per connection

### 3.1 Construction

Reuses the `request_proof_ok` shape (`harness/approval.py:781-795`):
`_proof` = `hmac_sha256(cap, label || client_nonce || ":" || server_salt)`
(`approval.py:678`); fail-closed constant-time `_proof_ok` (`approval.py:732-779`);
nonces from `operator_nonce()`, shape-checked by `is_wire_hex` (`approval.py:622-643`);
domain-separated labels (`approval.py:648-649`). New label, key, and one bound input:

```
relay_proof = hmac_sha256(K_relay, b"lop-relay-attest-v1" || relay_nonce || ":" ||
              relay_salt || ":" || wire.canonical_json(block))   # network/wire.py:378
block = {device_id, name, network_id, link_id, peer_authenticated, capabilities}
```
Canonical form (M-11): exactly those six keys; the first four are `str`,
`peer_authenticated` is `bool`, `capabilities = sorted(set(str(c) for c in caps))`,
matching `dial.py:378`. The runtime holds caps as a frozenset (`server.py:5323-5328`)
and must re-sort the same way before recomputing.

Flow, mirroring the operator handshake (`server.py:10885-10896`, `10950-10966`):
1. `dial_owner` auth frame adds `relay_nonce` (`dial.py:369-384`).
2. Runtime mints `relay_salt` per connection, even without an operator cap (unlike
   `_welcome_operator_proof`, `server.py:10959-10962`), and returns it in welcome.
3. Relay sends `{"op":"relay_attest","proof":…}` as its first post-welcome frame.
4. Runtime verifies, recomputing `block` from the auth frame's `peer` and
   `capabilities`; on success sets `conn.mesh_verified=True` and keeps
   `conn.peer_authenticated` from the proved block. `_ConnectionAuthority` gains both,
   filled at `server.py:8957-8970`.
5. Frames before a valid attest run as today and stamp `verified=false`.
6. **Position pinned (M-9):** `relay_attest` is accepted ONLY as the first
   post-welcome frame, refused (no state change) later; `dial_owner` consumes its
   reply before handing the dial on, and `_accept_stream` (`relay.py:12193-12250`)
   drops any `relay_attest` from a remote viewer.
7. **No link, no attest (M-6):** linkless `_dial_owned` (`relay.py:11973` set_model,
   `12090` create's first prompt) sends no peer block ⇒ `verified=false`; create
   SHOULD pass `link` so a remotely created session's first prompt can verify.

Salt from the runtime so a captured auth frame cannot be replayed; same reason as
the operator salt (`server.py:2356-2365`).

### 3.2 Key: derived, no new file

`K_relay = HKDF-SHA256(ikm = device Ed25519 seed, info = b"lop-relay-runtime-v1", L=32)`.
The seed is this install's mesh identity (`network/identity.py:153-166`, `load()` at
`identity.py:237`); relay and runtime share the config root, so each derives it.
No new secret: the device key already IS the trust root, and deriving inherits its
deny rule. The runtime derives `K_relay` once and discards the seed; the relay derives
per dial, so a rotated identity is picked up (a stale key fails closed) (M-10).
Location `<config>/network/identity/`, private dir (`identity.py:141-149`), file
`FILE_MODE = 0o600` (`identity.py:60`, `606`).

### 3.3 Keeping model-authored shells out — and where that fails

Mechanism: `network/identity` is in `_CONFIG_SECRET_SUBPATHS`
(`tools/confinement.py:787-795`, entry `789`), compiled into a kernel
`(deny file-read* …)` for the active and default config roots
(`confinement.py:1027-1053`), applied via `sandbox-exec` (`confinement.py:223`)
to any shell under `ShellWriteConfinement`.

Residual risk, stated plainly:
1. **Only write-confined shells are covered.** `ShellWriteConfinement` exists only
   when the context carries `shell_write_root` (`confinement.py:1100-1110`), set for
   read-only/scout roles and their descendants (`session/session.py:9438-9447`,
   `harness/subagent.py:1227-1252`). An ordinary session's `bash` is NOT confined
   and can read `device.json` (the residual `confinement.py:736-748` already states).
2. **Linux has no kernel confinement**; it is a named follow-up
   (`confinement.py:219-222`), and confined shells there are refused, not sandboxed.
3. **What G6 buys:** forging a mesh-verified connection moves from "read the 0600
   runtime record" (any shell, `asks/surfaces.py:7-11`) to "read the device
   private key", the bar for impersonating this device to the whole mesh. It is
   NOT a boundary against an unconfined same-uid process.
4. **Therefore** the guard must never take `verified` alone as consent (§2.2); full
   receiving-side closure needs a separate uid or OS keystore: out of scope.
5. **No integrity at rest (M-5).** §2.1 re-checks membership and trust, not
   authenticity: any writer of the session directory (journal, inbox, ask log) can
   mint `verified:true`. Today that is practically the seed bar: confined shells can
   neither write config nor connect to loopback (`confinement.py:729-734`), and
   unconfined processes can already read `device.json`.
6. **No channel binding (M-9).** The proof covers the connection's start only; anything
   that redirects the relay's dial (rewriting the runtime registry record, squatting a
   dead runtime's port) can sit in the middle and inject frames. Same bar as the seed.

### 3.4 Compatibility

- Older relay (no `relay_nonce`/attest, or no `peer_authenticated` in its block):
  works as today; stamps `verified=false`.
- Older runtime: ignores unknown auth keys (`server.py:5315-5330` reads named keys
  only) and sends no `relay_salt`; relay skips attest; no stamp ⇒ unverified.
- No mesh wire change, no `MESH_PROTOCOL_VERSION` bump (`network/types.py:42`).

## 4. Preservation paths

- **Prompt/steer** (`session.py:10615-10626`, `10791-10800`; binding note 2, M-6):
  both gain `mesh_origin: Mapping | None`, merged beside `lop_remote_prompt`; both
  harness-injection writes become a merge,
  `provider_payload = {**(provider_payload or {}), RENDERED_INJECTION_KEY: True}`.
- **Spool (M-2, M-6):** NOT in `InboxLine.sender` (`session/runtime/inbox.py:245`),
  which peer rows share: `_stamp_peer_sender` (called at `server.py:9524`) adds the
  same `remote_origin` key `_owner_sender` uses (`serving.py:633-637`). `InboxLine`
  (`inbox.py:208-250`) gains `mesh_origin: dict | None = None`, written ONLY at the
  four owner-spool `SOURCE_USER` sites (`serving.py:4484`, `4535`, `5805`, `5849`)
  and read on both drain arms, prompt (`process.py:5324-5325`) and steer
  (`process.py:5375-5378`), only when `line.source == SOURCE_USER`. Copied, never
  raised; no at-rest authenticity is claimed (§3.3 residual 5).
- **Handover/move (M-4):** the move copies the session and promotes it with one
  `os.replace` (`network/mobility.py:8-19`; `_promote`, `mobility.py:2554`). Source-
  minted stamps keep `stamped_by` = source, so they never count (§2.1), and `_promote`
  also rewrites each foreign-minted stamp to `verified=false` in place (original kept
  for audit). Replica promotion (`network/sync.py:3874`) gets the same downgrade.
  At-rest trust of a moved ask log's `answered_by.surface` is pre-existing, out of scope.
- **Ask answer:** `_stamp_ask_answer` (`server.py:162-197`; calls `9385`, `9462`) also
  returns the stamp; the store writes `answered_by.mesh_origin` (`asks/store.py:839`)
  and folds revisions at `878-891` per §2 rule 4.

## 5. Tests (all new; coder runs only these files)

1. Forged `peer`/`capabilities`/`locality:"remote"` + valid `control_key`, attest
   missing/wrong label/wrong salt/other connection's ⇒ `verified=false`.
2. Valid proof, `peer_authenticated=true` ⇒ `verified=true`; stamp = proved block;
   `stamped_by` = this device.
3. Valid proof, `peer_authenticated` false or absent (pre-G1 outbound; skew) ⇒ false.
4. Proof over another block (`device_id` or `peer_authenticated` changed) ⇒ false.
5. Frame-supplied `lop_mesh_origin`/`mesh_origin` stripped on prompt, steer, answer,
   and `peer_message` live AND spooled (`InboxLine.mesh_origin` stays `None`).
6. Spool round trip at all four sites and both drain arms: stamp byte-equal on
   `SOURCE_USER`; a forged `mesh_origin` on a `SOURCE_PEER` line is ignored.
7. Steer AND prompt keep the stamp, plain and with `harness_injected=True`.
8. Move and replica promote: source-minted rows/answers read `verified=false`, keep
   `stamped_by` = source; `mesh_origin_counts` False on the destination.
9. This device's phone daemon (`kind != attach`) and tunnel answers never verify.
9b. Relayed from device B via its mobile daemon, pilot, `network sessions --send` or
   an agent `send` ⇒ `verified=true`, `origin_device_id=B`, no `operator_origin`.
10. Legacy rows/records without the key ⇒ `mesh_origin_counts` False.
11. `mesh_origin_counts` rejects untrusted network, tombstoned member, `removed_ids`,
    missing record, `verified=false`, `stamped_by != self`; accepts an active member
    after an unrelated rotation.
12. Merge of writes from different devices ⇒ false. 12b: a verified answer revised
    by an unstamped or `verified:false` write ⇒ false (M-8).
13. Confinement (darwin-only, `confinement.py:729-734`): under
    `ShellWriteConfinement`, reading `<config>/network/identity/device.json` is
    refused; without `sandbox-exec`, assert the documented refusal.
14. Compatibility: old relay (no `relay_nonce`) admitted, unverified; old runtime (no
    `relay_salt`) ⇒ no attest, dial succeeds.
15. Linkless `_dial_owned` ⇒ no attest, false; create passing `link` ⇒ verifiable.
16. Late `relay_attest` refused; one pumped from a viewer via `_accept_stream`
    dropped; the attest reply never reaches a viewer (M-9).
17. Corrupt/unreadable record ⇒ False; stamp naming a `previous_ids` entry ⇒ False.

## 6. Files to change (estimates)

| File | Change | ≈ lines |
|---|---|---|
| `local_operator/network/origin.py` (new) | `derive_relay_key`, `relay_proof`, `relay_proof_ok`, `mesh_origin_counts` | +170 |
| `network/relay.py:4842-4904, 9892-9899` and the PeerLink build sites (dial and register) | `PeerLink.peer_authenticated` from `handshake.auth_verified`; `_peer_block` + `network_id`, `link_id`, `peer_authenticated` | +20 |
| `network/dial.py:369-400` | `relay_nonce` in auth; read `relay_salt`; send `relay_attest` | +35 |
| `session/runtime/server.py:2119-2183, 2356-2365, 5315-5330, 8957-8970, 10885-10896` | `mesh_verified`/`peer_authenticated`; salt in welcome; `relay_attest` op; stamp builder; strip frame copies | +115 |
| `server.py:9081-9130, 162-197` | pass `mesh_origin` to prompt/steer; return it with the answer surface | +30 |
| `session/session.py:10596-10626, 10780-10800` | `mesh_origin` kwarg; prompt and steer merge, not overwrite | +30 |
| `session/runtime/inbox.py:208-250`, `serving.py:4484/4535/5805/5849`, `process.py:5324-5325, 5375-5378` | `InboxLine.mesh_origin`, written and read on `SOURCE_USER` only | +30 |
| `network/mobility.py:2554`, `network/sync.py:3874` | `_promote`/`promote_replica` downgrade foreign-minted stamps | +35 |
| `network/relay.py:12090, 12193-12250` | pass `link` on create; drop viewer `relay_attest` | +10 |
| `asks/store.py:839, 891` | store and merge `mesh_origin` | +20 |
| tests (§5) | `tests/unit/{network,session,asks,tools}` | +500 |

≈ 495 product + 500 test lines, one PR.

## 7. Rollout

- **Inert:** nothing reads the stamp or calls `mesh_origin_counts` until 60334's guard
  change, so this can merge before G1. Only new payload keys appear, which providers
  never see (`session.py:417-423`).
- **Order:** this PR → G1 (lane 7c), which sets `peer_authenticated` on dialer links
  after listener verification → G7 `operator_origin` (60334) → the guard consumes.
- **Watch:** one extra round trip per forwarded op (`relay.py:9964-9971`); attest
  failures logged as facts only (like `server.py:183-196`); no `verified=true` from an
  outbound link before G1.

## 8. Changes from the provisional contract (flag to 60334)

1. Container keys: `provider_payload["lop_mesh_origin"]`, `answered_by["mesh_origin"]`.
2. `direction` replaced by a MAC-bound `peer_authenticated` in the G6 block, folded
   into `verified`; not a stamp field (M-1).
3. New stamp field `stamped_by` (M-4).
4. No `epoch` field (§2.1); additive int if wanted.
5. Unproved non-local connections stamp `verified=false` rather than omitting it.

## 9. Security review round 1 — disposition

Reviewer: 60334a0b4c12 (opus-5-5) · Scope: `1d493869f..5a4d7175d` · CHANGES REQUESTED
(4 MAJOR, 5 MINOR, 3 NIT); G6 crypto core judged sound.

| Finding | Sev | Disposition | Where |
|---|---|---|---|
| M-1 runtime constant downgradable by version skew | MAJOR | fixed: per-link `peer_authenticated` from `auth_verified`, MAC-bound; constant dropped | §2 r3, §3.1, §5.3-4 |
| M-2 spool stamp in peer-controlled `InboxLine.sender` | MAJOR | fixed: `InboxLine.mesh_origin`, `SOURCE_USER` only; `_stamp_peer_sender` pops both keys | §2 r1, §4, §5.5-6 |
| M-3 rule 6 false for B's phone/pilot/agents | MAJOR | fixed: "device, never human"; test 9b; G7 requirements | §2 r6, §2.2, §5.9b |
| M-4 move/replica imports foreign stamps | MAJOR | fixed: `stamped_by`; own stamps only count; `_promote`/`promote_replica` downgrade | §2, §2.1, §4, §5.8 |
| M-5 at-rest integrity overstated | MINOR | fixed: claim removed; residual 5 | §3.3 r5, §4 |
| M-6 carriage list incomplete (4 spool sites, steer drain, prompt overwrite, linkless dial) | MINOR | fixed: all sites named; prompt merge; linkless ⇒ unverified, create passes link | §3.1.7, §4, §5.6-7, §5.15 |
| M-7 store-read failure; `previous_ids` match | MINOR | fixed: any exception ⇒ False; exact `device_id` required | §2.1, §5.17 |
| M-8 revision without stamp keeps verified | MINOR | fixed: unstamped/false revision ⇒ false | §2 r4, §5.12b |
| M-9 no channel binding; attest position | MINOR | fixed: position pinned, viewer attest dropped; binding stated as residual 6 | §3.1.6, §3.3 r6, §5.16 |
| M-10 runtime keeps signing seed | NIT | fixed: derive once, discard; relay derives per dial | §3.2 |
| M-11 canonical block form | NIT | fixed: exact keys/types, sorted set | §3.1 |
| M-12 "no guard change" not exact | NIT | fixed: stricter side effect stated | header |
