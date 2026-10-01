# Push fixtures

The **machine-side shapes** of push/ack-sync — ADR 0006 in
[`damianvtran/local-operator-mobile`](https://github.com/damianvtran/local-operator-mobile) —
frozen as data so the three repos that implement the feature build against one
written contract instead of against prose. This is slice **S3**'s deliverable
(`docs/push-plan.md`), and the files here are the pin the other lanes read:

- **core** (this repo) produces the shapes in `registry-*.json` and `emit-*.json`;
- **the app** consumes the register response, renders the list, and maps the three
  refusal codes to copy;
- **the cloud** (`proposal` route shapes, not ours) validates the emit bodies with
  `extra="forbid"`, which is why every field a body may carry is enumerated here
  rather than left to a reader's judgement.

ADR 0003's fixture pattern is followed deliberately (see the mobile repo's
`fixtures/relay/README.md`): **one JSON per shape, a `provenance` object in every
file, and this README as the index**. As there, **`provenance` is the authority** —
the filename and the tables below are conveniences, and if they ever disagree with
a file's own `provenance`, the file wins.

## Provenance

**Every file here is `synthetic`.** None is captured from a running relay, and the
`how` field says why: the machine-side shapes are *contract text* — the register
response, the emit body, the refusal sentences — and there is no route emitting
them yet that a capture would make more truthful. Each file names the ADR ref it
was transcribed from (`damianvtran/local-operator-mobile` @ `b03aeb15`) and the
core ref it was checked against (`damianvtran/local-operator` @ `b77fec9d4`, the
branch base).

They are **hand-authored and reviewed as a contract**, not regenerated: a
generator would let the contract move silently the next time someone ran it. The
one place a value is pinned rather than transcribed —
`emit-idempotency-keys.json`'s digests — is recomputed by the test on every run, so
a drifted digest fails CI rather than being quietly refreshed.

**Two pins, on purpose.** Every shape is transcribed from the ADR at
`b03aeb15` (S3's freeze). The DIGEST rules were amended after that freeze and are
now merged in the same document at **`cc2569a4`** — the third emit type, the
`alert` object, the `alert`-absent-on-attention rule, the `exclude` scope, and
the persisted window's `emit_id` — so the digest file carries a
`digest_rules_ref` naming that ref beside its `adr_ref`. Where the two disagree
about the digest, `cc2569a4` is the one that governs.

## The files

| File | Shape | The one thing it fixes |
| --- | --- | --- |
| `registry-register-response.json` | `POST /api/push/register` → 200 | `{ok, device_id, device_key, registered_at}` and nothing else — the key's only delivery path |
| `registry-register-rotation.json` | the same route, re-registered | `device_id`/`registered_at` stable, `device_key` **new** (Q29: the first key stops matching) |
| `registry-register-refusal-device-revoked.json` | 403 | `device_revoked` + its sentence, refused before anything is written |
| `registry-register-refusal-device-unpaired.json` | 403 | `device_unpaired` + its sentence, which names the **computer** |
| `registry-list-response.json` | `GET /api/push/devices` → 200 | the row allow-list (`state`, the optional `name`/`credential_live`/`last_authenticated_at`) and the `precedence` sentence; **no `device_key`, no token** |
| `registry-unrevoke-success.json` | `POST /api/push/devices/{id}/unrevoke` → 200 | `{ok, device_id}`; every marker cleared, no token and no credential restored |
| `registry-unrevoke-refusal-machine-only.json` | 403 | `machine_only` + its sentence — the operators-only gate |
| `registry-unrevoke-refusal-device-absent.json` | 404 | `device_absent` + the sentence naming the id asked about |
| `emit-completion.json` | `POST /v1/tunnels/{tunnel_id}/push/events` | the §3.2 completion payload and the emit body (payload + `devices`) |
| `emit-attention.json` | the same route | the attention payload, its optional `exclude`, and the absent-`exclude` variant |
| `emit-digest.json` | the same route | the digest payload (`type: "digest"`, the VISIBLE coalesced catch-up), its optional `exclude`, and the absent-`exclude` variant |
| `emit-idempotency-keys.json` | `Idempotency-Key` | §3.4's three recipes with worked vectors, including the heal and the digest |
| `payload-forbidden-fields.json` | deny list | what may never appear in any of the above, and why each one |

## How to read them

- **Shape files** carry the allow-list next to the literal: `required` /
  `optional` (for a payload: `payload_required` / `payload_optional`, and the
  report block's own pair, plus `alert_required` on the two shapes that carry an
  alert) with a type per field, plus `forbidden` where the shape has one and
  `absent_fields` where a shape declares a field it must never carry. A field is
  either named in a shape block **or** in
  `payload-forbidden-fields.json` — never both, never neither. The two shapes
  with an alert also file `alert_count_one`: the same alert at a count of one,
  where the count term is singular (`1 conversation needs you`), so no reader of
  this tree only ever sees the plural.
- **`same_as`** means *this file does not restate the shape* — it names the file
  whose `required`/`optional`/`forbidden` blocks govern it too. `registry-register-
  rotation.json` uses it for the register response, so there is one home for that
  allow-list; the test resolves the pointer and asserts the inheritance, so a
  restatement that disagrees is refused rather than ignored.
- **`example` / `payload` / `body`** are the concrete literal. `body` is the whole
  emit body (payload plus the `devices` report block); `payload` is the object the
  core builds before the block is attached. **Every filed `example` is validated
  against the blocks that govern it** — required present, optional allowed,
  nothing else, `forbidden` absent — because the literal is the text the app and
  the cloud copy from.
- **`notes`** carry the constraint, not the description: why a field is optional,
  what a code means, which ADR rule a shape exists to satisfy.

## Using these in tests

- `tests/unit/mobile/test_push_payload.py` asserts the core's payload builder and
  its two keys against `emit-*.json`;
  `tests/unit/mobile/test_push_wire_contract.py` drives the **real** registry
  builders and routes and asserts their responses against `registry-*.json`.
  Both compare against the filed literals — the equality is the guard, so adding a
  field to a builder fails, and removing one fails too. The emit cells' **input**
  is the test's own (`DEVICE_ROWS` / `COMPLETION_INPUT`), never read back from the
  file they check: an input taken from the expected output makes the comparison a
  tautology.
- **The `devices` block's pair is enforced per row**, on both sides — the rows the
  test builds and the rows the file files — against `report_block_required` /
  `report_block_optional`. A block row with an extra field, a retyped timestamp or
  a missing `credential_live` fails, and a file with no deny list and no closed
  body fails as an unguarded shape.
- **Check `provenance.kind` in a test, not the path.** A suite that reads a fixture
  by name after a rename is the case that field exists for (the mobile repo's rule).
- The refusal files are asserted **exactly**: `code` and `error` are copy the app
  renders, so a reworded sentence is a contract change.

## What is deliberately NOT here

- **The APNs/FCM envelope** (`aps`, `badge`). The machine sends the `data` object
  and stops; the phone-facing envelope is the cloud's fan-out and a proposal.
  `aps.badge` is never sent at all (§1.5). The machine DOES compose the
  user-visible `alert` for the two visible types (the lane's ruling of
  2026-10-01), and the cloud delivers that text verbatim and wraps it; the
  attention form deliberately carries none, because it is a silent wake.
- **The cloud's own route shapes** (`POST <cloud>/v1/push/register`, the
  heartbeat) — proposals, owned by the cloud lane's `docs/push-cloud-ops.md`. The
  one shape they share is the `devices` report block, which rides the emit body
  here and is spelled the same way on all three carriers (§3.2 #1–#3).
- **The report block's builders.** Its rows are S4c's
  (`local_operator/mobile/push_credentials.py`, merged as PR #1881) — this tree
  freezes the block's literal, not its producer, and **names the block's key by
  importing that module's `REPORT_DEVICES_FIELD`** rather than spelling a second
  string beside it.
- **The tunnel-gateway allowlist guard (QA row Q31).** The cells asserting that
  `X-Lop-Operator-Key` is absent at the relay and that the allowlist entries stay
  lowercase are S4c's, landed with PR #1881 — this slice cross-references the rule
  and deliberately does not write a second, differently-worded copy of it.

## Versioning

`payload_version` is `1` (§3.2's `v`). A change to the wire shape means a **new
version and a new file**, never a silent edit in place: the field is on the body
precisely so a reader that does not know the version can refuse it.
