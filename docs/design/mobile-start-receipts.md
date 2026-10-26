# Mobile relay: durable session-creation receipts (protocol v1)

Authoritative contract for `POST /api/sessions/start` when a client supplies a
creation key. Implementation: `local_operator/mobile/start_receipts.py` and the
keyed branch of the start route. Client UI/outbox behaviour is a separate lane;
this document defines only the wire and server guarantees.

## Why creation needed its own protocol

Existing command admission deduplicates retries that still carry their original
command ID and body. Creation had none: the route minted `uuid4().hex[:12]` on
every request, so a retry after a lost response produced a second conversation
and the client had no way to learn which one was real. This protocol makes the
identity authoritative **before** the side effect and replays it afterwards.

## Capability

Authenticated `GET /api/sessions` and the `sessions` SSE frame carry:

```json
"capabilities": {
  "session_start": { "version": 1, "durable_idempotency": true, "receipt_capacity": 4096 }
}
```

- **Absent** means the relay predates this contract. An older server silently
  ignores unknown start-body fields, so a client MUST NOT send
  `create_request_id` unless this object is present.
- `receipt_capacity` is the hard bound on recorded keys for that installation.
- A server may refuse a *new* key when full (`create_capacity`) even while
  advertising the capability: the field is a build fact, not a writability
  promise. Storage faults are reported the same way, as a typed refusal.
- The object never identifies an account. It is not a signed-in principal, a
  password verifier or a cross-login recovery claim, and must not be used as
  one.

## Request

```json
{ "create_request_id": "<uuid>", "cwd": "…", "provider": "…", "model_id": "…" }
```

- `create_request_id` is optional. **Omitting it preserves the shipped
  behaviour exactly**: no receipt is read or written, the response is unchanged,
  and duplicates remain possible. Legacy callers keep working untouched.
- When present it must be a UUID. The canonical form (lowercase, hyphenated) is
  the key; `{…}`/uppercase input is normalised before use.
- `cwd`, `provider` and `model_id` are optional strings. Keyed requests are
  refused with `400 invalid_start_request` for unknown top-level fields, so a
  later field cannot silently alias an earlier admission.
- The **fingerprint** is the canonical JSON of
  `{cwd, provider, model_id}` exactly as requested. Omitted values are not part
  of it.

## Response

Success keeps the existing shape:

```json
{ "ok": true, "pid": 12345, "session_id": "abcdef123456" }
```

A replay of a completed key returns the same `session_id` and the **current**
`pid` of that session. A `pid` is never replayed from the receipt: after a
restart or PID reuse it could address an unrelated process, so the server
reconciles the live owner of the exact `session_id`; when no owner exists it
answers `create_indeterminate` (below) instead of a stale number.

## Refusals

| Status | `code` | Meaning |
|---|---|---|
| 400 | `invalid_create_request_id` | Not a UUID, or longer than 36 characters |
| 400 | `invalid_start_request` | Unknown top-level field, or a non-string value longer than 4096 characters |
| 400 | `invalid_start_directory` | `cwd` is not an existing directory the spawn gate admits |
| 400 | `invalid_start_selection` | The resolved provider/model pair is missing or unusable |
| 409 | `create_conflict` | Same key, different fingerprint or input |
| 409 | `create_indeterminate` | Uncertain outcome; carries the authoritative `session_id` |
| 503 | `create_capacity` | The receipt bound is full; new key refused before any side effect |
| 413 | `start_request_too_large` | Body over 151,552 bytes; refused before it is parsed, keyed or not |
| 503 | `create_receipts_unavailable` | Store corrupt, unreadable or unwritable |

`create_indeterminate` is a **hold, not a failure to retry as new**. The
response includes `session_id`; a client may resume or query that exact session,
but must not mint a replacement key for the same intent. Known keys keep
replaying while the store is full.

## Guarantees

1. **Auth and origin precede everything.** The route's existing gate runs
   before both a new claim and a replay; idempotency is never an authentication
   bypass. Receipts are bound to the server's own config root, never to a
   caller-provided account string.
2. **Resolved intent is durable before the side effect.** On first admission the
   server resolves `cwd` (request value or the shared default start directory),
   `provider` and `model_id` through the same birth-precedence resolver the
   session constructor uses, writes them with the newly minted `session_id`, and
   commits before calling `spawn_session`.
3. **Retries freeze the original intent.** A replay never re-reads changed
   defaults, never re-resolves a relative directory and never re-validates a
   provider the user has since changed.
4. **One key, one constructor.** Same-process concurrency coalesces on an
   in-flight claim; the durable row provides cross-process exclusion. The spawn
   uses the existing `spawn_session(..., resume=session_id)` path, so the
   existing exact-owner adoption and constructor exclusion still apply.
5. **A cancelled request keeps its claim.** The receipt lifecycle is shielded
   from client disconnection, so an abandoned request still completes or holds
   its key. A later retry observes the same identity.
6. **Restart reconciliation is conservative.** A pending row is completed only
   when the exact `session_id` has a verifiable live owner. Absence of an owner
   is inconclusive: the server answers `create_indeterminate` with that
   `session_id` and never starts a second constructor. No "most recent session"
   lookup exists anywhere in this path.
7. **The store fails closed.** A corrupt, unreadable or unwritable store refuses
   work (`create_receipts_unavailable`) rather than treating itself as empty; an
   empty answer would make every recorded key re-runnable.
8. **No expiry, no eviction.** Unlike transfer receipts, these rows are never
   TTL-expired or evicted. Forgetting a key would convert a legitimate retry
   into a second conversation. Capacity is the only bound, and it refuses new
   keys only.

## Bounded limitations (honest)

- **Capacity is finite** (4096 keys by default). Once full, new keyed starts are
  refused until the store is extended or an explicit, versioned epoch migration
  exists. There is no TTL reclamation, and none should be added without a
  contract change that makes a retired epoch's retries visibly refused rather
  than silently re-run.
- **A pending row can remain unresolved indefinitely** if the original child
  never becomes adoptable. That is deliberate: it preserves the identity the
  user's intent already owns, at the cost of requiring an operator/client
  decision instead of an automatic second attempt.
- **This protocol does not provide draft or outbox persistence**, encrypted
  recovery across an auth-expiry redirect, or any account-scoped recovery. Those
  are client-lane and separately reviewed concerns; the relay's existing 401 /
  logout purges are unchanged here.
- **Receipt storage is local to the installation's config root.** Two servers,
  or the same server after a config-root change, do not share keys. A client
  must scope its own retry identity accordingly.
- The relay proves the same owner authority as before; it does **not** prove a
  distinct signed-in account.
- **The whole request body is capped at 151,552 bytes** (`START_BODY_MAX_BYTES`)
  before any JSON parsing. The cap is checked on `Content-Length` and again on
  the bytes actually read, so a chunked or lying request stops at the cap and
  is never buffered whole. It derives from the field bounds (3 fields × 4096
  characters × 12 bytes for the widest JSON escape, plus 4 KiB), so no request
  the schema admits is refused by it. It applies to unkeyed callers too: the
  route cannot tell a body is keyed until it parses it.

## Client obligations

- Advertise-gate on `capabilities.session_start`; without it, keep the
  conservative legacy flow and never send a key.
- Generate one key per user intent and **retain the key and body** until
  acknowledged; reuse them for every retry, including after an app restart.
- Treat `create_indeterminate` as "identity known, liveness unproven": show the
  refusal, offer resume of the returned `session_id`, never a fresh start under
  a new key.
- Treat `create_conflict` as a client bug (two intents shared a key), and
  `create_capacity` / `create_receipts_unavailable` as retry-later conditions.

## Evidence

`tests/unit/mobile/test_start_receipts.py` exercises the real authenticated HTTP
route against the real receipt store with only child construction and owner
discovery controlled: duplicate starts, coalescing, lost response, restart
replay, PID reconciliation, cancelled request, mismatch, auth/origin, capacity,
corruption, unwritable store, frozen defaults and the legacy response.
