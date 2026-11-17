# Mesh origin stamp (G5) and relay→runtime proof (G6)

Status: PROPOSED · lane: remote-parity (79bddc262663) · security reviewer: 60334a0b4c12
Base: origin/main `1d493869f`; every `file:line` below is at that HEAD.
Scope: produce and preserve a stamp and a proof. **No guard decision changes**;
today's behaviour stays until G1/G7 land (operator, relayed by 60334).

## 1. Problem

- A remote prompt carries one boolean, `provider_payload["lop_remote_prompt"]`
  (`local_operator/session/session.py:424-430`), set when `owner_typed=False`
  (`session.py:10615-10616` prompt, `10791-10792` steer), which the runtime derives
  from `authority.locality != "local"` (`session/runtime/server.py:9081-9088`, `9128-9130`).
- A remote ask answer is `answered_by.surface="remote"` for ANY non-local
  connection, mesh or phone (`asks/surfaces.py:117-128` via `_stamp_ask_answer`,
  `server.py:162-197`); the store keeps no device/network/link (`asks/store.py:839`, `891`).
- Steer with `harness_injected` OVERWRITES `provider_payload` (`session.py:10793-10800`).
- Spool keeps only `{"remote_origin": True}` (`session/runtime/serving.py:629-637`),
  mapped back to `owner_typed=False` (`session/runtime/process.py:5324-5325`).
- The relay's identity, `_peer_block` = `{device_id, name}` (`network/relay.py:9892-9899`),
  reaches the runtime as self-declared auth fields (`network/dial.py:369-384`) read
  behind nothing but `control_key` (`server.py:5288-5292`, `5315-5330`), which any
  same-uid process can read (`asks/surfaces.py:7-11`).

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
| `verified` | bool | true iff the G6 proof checked on THIS connection AND the link is authenticated both ways (inbound today; any link after G1) |
| `operator_origin` | bool, OPTIONAL, absent by default | G7 slot (60334): sender asserts a user-typed turn / own-surface answer, never an agent `send`. This PR never writes it. |

Rules:
1. **Never from a frame.** Built from `_ConnectionAuthority` (`server.py:2119-2160`).
   A frame-supplied `lop_mesh_origin`/`mesh_origin` is stripped, as
   `_stamp_peer_sender` drops frame device keys (`server.py:2158-2190`).
2. **Unproved non-local connection:** stamp written with `verified:false` and the
   claimed identity (audit only). Local connections write no stamp.
3. **Direction before G1.** The relay adds `direction: "inbound"|"outbound"` to the
   peer block from `link.outbound` (`relay.py:4904`, set at `7910`). Outbound ⇒
   `verified=false` until G1 flips one constant, `MESH_OUTBOUND_AUTHENTICATED`.
   Reason: outbound `device_id` is the unauthenticated challenge field
   (`handshake.py:531-536`); inbound is signature-verified (`handshake.py:718-742`).
   `direction` is bound by the proof (§3), not stored.
4. **Answer merge.** `asks/store.py:891` merges writes with `weakest_surface`; the
   mesh stamp merges likewise: `verified` = AND over writes; differing
   `origin_device_id`s ⇒ `verified=false`, first kept.
5. **Legacy:** a row or record without the key is UNVERIFIED (all pre-PR rows,
   older-build spool lines).
6. **Phone/tunnel never verify.** The phone daemon dials as non-`attach` and without
   a relay proof (`asks/surfaces.py:123-128`); the tunnel gateway reaches only the
   relay harness port (`tunnels/gateway.py:854-888`). Neither can derive the relay
   key (§3.2). `surface=="remote"` alone never means mesh.

### 2.1 Consent-time re-derivation (60334 binding note 1)

The stored `verified` is evidence, not a verdict. New stdlib-only helper in
`local_operator/network/origin.py`:

```python
def mesh_origin_counts(stamp: Mapping | None, record: NetworkRecord | None) -> bool:
    """True iff a delivery stamp still counts as an authenticated mesh member NOW."""
```

True only if, against the LIVE record (`store_view.network(network_id)`):
`stamp["verified"] is True`; the record exists with `trust == "active"` (the
`_refusal_reason` term, `relay.py:7834-7859`); `record.member(origin_device_id)`
exists (`network/types.py:1903`) and `.active` (`types.py:1543`; false once
`removed_at` is set, `types.py:1539`); and `origin_device_id not in record.removed_ids`.
`record.epoch` is logged, not gated: revocation is the tombstone + `removed_ids`
(`relay.py:2365-2410`), and an unrelated rotation must not void an active member's
rows. **Consumer:** the guard (60334). This PR ships helper + tests; no caller.

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
block = {device_id, name, network_id, link_id, direction, capabilities: sorted}
```

Flow, mirroring the operator handshake (`server.py:10885-10896`, `10950-10966`):
1. `dial_owner` auth frame adds `relay_nonce` (`dial.py:369-384`).
2. Runtime mints `relay_salt` per connection, even without an operator cap (unlike
   `_welcome_operator_proof`, `server.py:10959-10962`), and returns it in welcome.
3. Relay sends `{"op":"relay_attest","proof":…}` as its first post-welcome frame.
4. Runtime verifies, recomputing `block` from the auth frame's `peer` and
   `capabilities`; on success sets `conn.mesh_verified=True`. `_ConnectionAuthority`
   gains `mesh_verified: bool = False`, filled at `server.py:8957-8970`.
5. Frames before a valid attest run as today and stamp `verified=false`.

Salt from the runtime so a captured auth frame cannot be replayed; same reason as
the operator salt (`server.py:2356-2365`).

### 3.2 Key: derived, no new file

`K_relay = HKDF-SHA256(ikm = device Ed25519 seed, info = b"lop-relay-runtime-v1", L=32)`.
The seed is this install's mesh identity (`network/identity.py:153-166`, `load()` at
`identity.py:237`); relay and runtime share the config root, so each derives it.
Why not a new secret: the device key already IS the trust root (whoever reads it
impersonates this device to every peer); a second secret would have the same
exposure and no extra protection, and deriving inherits the existing deny rule.
Location `<config>/network/identity/`; directory repaired to private mode
(`identity.py:141-149`), file written `FILE_MODE = 0o600` (`identity.py:60`, `606`).

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
4. **Therefore** the guard must never take `verified` alone as consent. Intended
   rule: `mesh_origin_counts(...) and operator_origin is True`, with
   `operator_origin` proved on the SENDING side by the in-memory operator capability
   (`approval.py:781-795`), never on disk. Owner 60334 (G7). Full closure on the
   receiving side needs a separate uid or OS keystore: out of scope.

### 3.4 Compatibility

- Older relay (no `relay_nonce`/attest): works as today; stamps `verified=false`.
- Older runtime: ignores unknown auth keys (`server.py:5315-5330` reads named keys
  only) and sends no `relay_salt`; relay skips attest; no stamp ⇒ unverified.
- No mesh wire change, no `MESH_PROTOCOL_VERSION` bump (`network/types.py:42`).

## 4. Preservation paths

- **Prompt** (`session.py:10615-10616`): `prompt()` gains `mesh_origin: Mapping | None`,
  merged into `provider_payload` beside `lop_remote_prompt`.
- **Steer** (`session.py:10791-10800`; binding note 2): the harness-injection write
  becomes a merge,
  `message.provider_payload = {**(message.provider_payload or {}), RENDERED_INJECTION_KEY: True}`.
  Neither stamp erases the other.
- **Spool:** `_owner_sender` (`serving.py:636-637`) gains `mesh_origin`, stored as
  `sender["mesh_origin"]`; the busy spool call (`serving.py:4484-4490`) passes it;
  drain (`process.py:5324-5325`) returns it as `fields["mesh_origin"]`. Copied,
  never raised; a spool editor only produces rows §2.1 re-checks.
- **Handover/move:** the stamp rides `provider_payload`, which the durable journal
  row keeps (`session/transcript.py:517`; noted at `session.py:10617-10624`); a move
  transfers session dir + journal (`network/mobility.py:10-16`), ask records included.
  No code change expected; a test pins it.
- **Ask answer:** `_stamp_ask_answer` (`server.py:162-197`; call sites `9385`, `9462`)
  also returns the mesh stamp; the store writes `answered_by.mesh_origin`
  (`asks/store.py:839`) and merges at `891` per §2 rule 4.

## 5. Tests (all new; coder runs only these files)

1. Forged `peer`/`capabilities`/`locality:"remote"` with valid `control_key`, no or
   bad attest (missing, wrong label, wrong salt, other connection's proof) ⇒ `verified=false`.
2. Valid proof, inbound block ⇒ `verified=true`; stamp equals the proved block.
3. Valid proof, `direction="outbound"` ⇒ `verified=false` (pre-G1 constant).
4. Proof over a different block (`device_id` swapped) ⇒ `verified=false`.
5. Frame-supplied `lop_mesh_origin`/`mesh_origin` stripped on prompt, steer, answer.
6. Spool round trip (`serving.py:4484` → `process.py:5324`): stamp byte-equal.
7. Steer keeps the stamp, plain and with `harness_injected=True`: both keys present.
8. Move/handover: stamped row and answer survive journal reload on the destination.
9. Phone daemon (`kind != attach`) and tunnel-origin answers never `verified=true`.
10. Legacy rows/records without the key ⇒ `mesh_origin_counts` False.
11. `mesh_origin_counts` rejects untrusted network, tombstoned member, `removed_ids`,
    missing record, `verified=false`; accepts an active member after an unrelated rotation.
12. Answer merge of two writes from different devices ⇒ `verified=false`.
13. Confinement (darwin-only, like existing darwin tests, `confinement.py:729-734`):
    under `ShellWriteConfinement`, reading `<config>/network/identity/device.json`
    is refused; on a host without `sandbox-exec`, assert the documented refusal.
14. Compatibility: old-relay auth (no `relay_nonce`) admitted, unverified; old-runtime
    welcome (no `relay_salt`) ⇒ relay sends no attest, dial still succeeds.

## 6. Files to change (estimates)

| File | Change | ≈ lines |
|---|---|---|
| `local_operator/network/origin.py` (new) | `derive_relay_key`, `relay_proof`, `relay_proof_ok`, `mesh_origin_counts`, constants | +170 |
| `network/relay.py:9892-9899` | `_peer_block` + `network_id`, `link_id`, `direction` | +8 |
| `network/dial.py:369-400` | `relay_nonce` in auth; read `relay_salt`; send `relay_attest` | +35 |
| `session/runtime/server.py:2119-2160, 2356-2365, 5315-5330, 8957-8970, 10885-10896` | `mesh_verified`; salt in welcome; `relay_attest` op; stamp builder; strip frame copies | +110 |
| `server.py:9081-9130, 162-197` | pass `mesh_origin` to prompt/steer; return with answer surface | +30 |
| `session/session.py:10596-10620, 10780-10800` | `mesh_origin` kwarg; merge, not overwrite | +25 |
| `session/runtime/serving.py:629-641, 4484-4490` | carry `mesh_origin` in spool sender | +15 |
| `session/runtime/process.py:5324-5325` | hand it back on drain | +4 |
| `asks/store.py:839, 891` | store and merge `mesh_origin` | +20 |
| tests (§5) | `tests/unit/{network,session,asks,tools}` | +450 |

≈ 420 product + 450 test lines, one PR.

## 7. Rollout

- **Inert:** nothing reads the stamp or calls `mesh_origin_counts` until 60334's
  guard change, so it can merge before G1. Only new payload keys appear, which
  providers never see (`session.py:417-423`).
- **Order:** this PR → G1 (lane 7c) flips `MESH_OUTBOUND_AUTHENTICATED` → G7
  `operator_origin` (60334) → guard consumes.
- **Watch:** one extra round trip per forwarded op (`relay.py:9964-9971`);
  per-connection `relay_attest` failures logged as facts only (like `server.py:183-196`);
  no `verified=true` with `direction=outbound` before G1.

## 8. Changes from the provisional contract (flag to 60334)

1. Container keys added, field names unchanged: `provider_payload["lop_mesh_origin"]`,
   `answered_by["mesh_origin"]`.
2. `direction` is NOT a stamp field; it is proof-bound and folded into `verified`.
3. No `epoch` field (§2.1: membership + trust carry revocation); additive int if wanted.
4. Unproved non-local connections write the stamp with `verified=false` rather than
   omitting it, so audit keeps the claimed identity; the guard reads it only through
   `mesh_origin_counts`.
