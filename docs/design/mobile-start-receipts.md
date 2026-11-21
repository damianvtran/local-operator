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
  "session_start": {
    "version": 1, "durable_idempotency": true, "receipt_capacity": 4096, "effort": true,
    "agent": true, "team": true
  }
}
```

- **Absent** means the relay predates this contract. An older server silently
  ignores unknown start-body fields, so a client MUST NOT send
  `create_request_id` unless this object is present.
- `receipt_capacity` is the hard bound on recorded keys for that installation.
- `effort: true` means the start body accepts `effort` (below). A client MUST
  NOT send `effort` unless it reads `true` here: an older relay refuses it on a
  keyed request (`invalid_start_request`) and silently drops it on an unkeyed
  one, which would start the session at the configured default instead.
- `agent: true` / `team: true` mean the start body accepts `agent` / `team`
  (below) AND the relay serves `GET /api/start-targets`. Same rule, higher
  stakes: an older relay silently drops the field on an unkeyed start, so a
  client that sent it blind would start an **unbound** chat while showing the
  team it picked. Absent flag: no picker, no field.
- A server may refuse a *new* key when full (`create_capacity`) even while
  advertising the capability: the field is a build fact, not a writability
  promise. Storage faults are reported the same way, as a typed refusal.
- The object never identifies an account. It is not a signed-in principal, a
  password verifier or a cross-login recovery claim, and must not be used as
  one.

## Request

```json
{ "create_request_id": "<uuid>", "cwd": "…", "provider": "…", "model_id": "…", "effort": "medium" }
```

- `create_request_id` is optional. **Omitting it preserves the shipped
  behaviour exactly**: no receipt is read or written, the response is unchanged,
  and duplicates remain possible. Legacy callers keep working untouched.
- When present it must be a UUID. The canonical form (lowercase, hyphenated) is
  the key; `{…}`/uppercase input is normalised before use.
- `cwd`, `provider` and `model_id` are optional strings. Keyed requests are
  refused with `400 invalid_start_request` for unknown top-level fields, so a
  later field cannot silently alias an earlier admission.
- `effort` is an optional birth reasoning level, keyed or not: one of
  `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` (case-insensitive;
  `null` or blank means "the configured default", exactly as omitting it). Any
  other value is `400 invalid_start_effort`, before a receipt or a spawn. A
  level the chosen model's ladder lacks is clamped to its nearest rung by the
  session, as every other birth level is; the configured default the phone
  should display is `GET /api/models` → `default.effort`, and the levels to
  offer per model are that row's `efforts` (below).
- `agent` / `team` are optional, **mutually exclusive** launch targets, keyed
  or not: a profile or team name of at most 128 characters, trimmed (`null` or
  blank means "no target"). The name is validated against the same global
  registries and with the same validator as the desktop's new-chat target
  (`validate_target`, which canonicalises case: `LOPDEV` binds `lopdev`; a team
  is checked through its whole roster). The session's `attachment.json` is
  written and read back **before** the child spawns, so the session restores
  that identity at construction and its first turn already runs as the agent,
  or as the team's manager with the team brief and roster. The target does
  **not** change the model, exactly as on the desktop.
- The **fingerprint** is the canonical JSON of
  `{cwd, provider, model_id}` exactly as requested, plus `effort`, `temporary`,
  `agent` or `team` only when chosen — so a request without them fingerprints
  byte-for-byte as it did before the fields existed. Omitted values are not part
  of it. The target is fingerprinted as the caller spelled it (trimmed) and the
  canonical name is frozen in the resolution, so `"LOPDEV"` and `"lopdev"`
  under one key are `create_conflict`. The sidecar is written on the fresh
  start only: a replay never rewrites it, so it cannot undo a later `/team clear`.

### Listing targets: `GET /api/start-targets`

Gated like every route; licensed by `session_start.agent/team`.

```json
{
  "agents": [{ "name": "coder", "label": "Coder", "kind": "role", "source": "builtin",
               "description": "…", "agent_id": null, "avatar_sha256": null }],
  "teams":  [{ "name": "lopdev", "label": "lopdev", "description": "…",
               "manager": "manager", "member_count": 7 }],
  "degraded": false
}
```

- Rows are the desktop menu's own catalogues (`profile_catalogue`,
  `team_catalogue`) over the global registries, so the phone offers exactly the
  names the desktop offers and the start route accepts. Project-scoped profiles
  are out of scope.
- `label` is pre-painted through the shared display rules; render it as is.
- `kind` is `role` or `specialist`; `source` is `builtin` (a packaged seed not
  yet installed, `agent_id: null`), `installed` or `custom`.
- `avatar_sha256` is the session rows' source (`agent_avatar.agent_index`);
  fetch the image with `GET /v1/agents/{agent_id}/avatar?v=<sha>&size=64`.
  Teams carry no avatar; join on `manager` to show the manager's.
- A registry failure answers `200` with empty lists and `degraded: true`:
  show "couldn't load", not "no agents".

### Picking a model and effort: `GET /api/models`

The sheet that feeds this body reads `GET /api/models` (authenticated):

```json
{
  "models": [
    { "selector": "anthropic/claude-opus-5-5", "provider": "anthropic",
      "model_id": "claude-opus-5-5", "name": "…", "label": "…",
      "connected": true, "aggregated": false,
      "efforts": ["low", "medium", "high", "xhigh", "max"] },
    { "selector": "anthropic/claude-haiku-4-5-20251001", "…": "…", "efforts": [] },
    { "selector": "radient/auto", "…": "…" }
  ],
  "default": { "selector": "anthropic/claude-opus-5-5", "provider": "anthropic",
               "model_id": "claude-opus-5-5", "effort": "medium" }
}
```

- `efforts` is the ordered ladder (lowest first) the effort picker may offer
  for that row: `build_model_spec(...).reasoning_efforts` — the spec the child
  session itself is built from (provider-listing ladder where the relay holds
  it, then the aggregator router ladder, the direct DeepSeek ladder, then the
  offline effort table) — built on the offline path (static registry or
  in-process memo, no network per row), and the same ladder `default.effort` is
  clamped to. `[]` means the model has **no** effort knob (the spec has no
  ladder and the shipped registry describes the model, including an undated
  alias of a shipped snapshot): hide or disable the picker and send no
  `effort`. An **absent** key means the relay does not know the ladder (an
  aggregator or listing-only id whose listing it has not read), exactly what an
  older relay that never sends the field means: offer the whole vocabulary and
  let the session clamp. It is never `null`.
- `default` names what a start with no `provider`/`model_id`/`effort` runs.
  `null` means this relay resolved no default (render "unavailable"); an absent
  key means an older relay. `default.effort` is the configured `model_effort`
  clamped to the model's `efforts` (`resolve_effort_in` against the
  `build_model_spec` ladder and its `reasoning_default_effort`, the clamp the
  session applies), or that spec default when nothing is configured, `null` for a model whose `efforts` is `[]`, and the configured
  value **unclamped** for a model whose ladder is unknown (the session clamps
  it against the provider's listing at birth).

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
| 400 | `invalid_start_effort` | `effort` is not a string in the level vocabulary (keyed and unkeyed) |
| 400 | `invalid_start_target` | `agent`/`team` not a string, over 128 characters, **both** set, or a team roster with a cycle or past the nesting limit (keyed and unkeyed) |
| 404 | `unknown_start_target` | The name, or a team's manager/member, does not resolve; carries `unresolved: {team, slot, member}` when known |
| 503 | `start_target_registry_unavailable` | The profile/team registry could not be read |
| 503 | `start_target_unsaved` | The binding did not read back; nothing spawned. Keyed: carries `session_id` (retry answers it, as for any post-claim failure) |
| 400 | `invalid_start_selection` | The resolved provider/model pair is missing or unusable |
| 409 | `create_conflict` | Same key, different fingerprint or input |
| 409 | `create_indeterminate` | Uncertain outcome; carries the authoritative `session_id` |
| 413 | `start_request_too_large` | Body over 153,152 bytes; refused before it is parsed, keyed or not |
| 503 | `create_capacity` | The receipt bound is full; new key refused before any side effect |
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
- **Downgrading past the `effort` field breaks keyed starts until the store
  is moved aside.** A relay from before `effort` validates every stored
  receipt on each keyed start and rejects a row whose stored intent carries
  `effort`, so after a rollback over a store holding one, every keyed start
  answers `503 create_receipts_unavailable`. Because rows never expire or
  evict (guarantee 8), this does not heal on its own. Unkeyed starts are
  unaffected. Recovery, on the downgraded machine (the store is opened per
  request, so the relay need not be stopped):

  ```sh
  cd "${LOCAL_OPERATOR_CONFIG_DIR:-$HOME/.local-operator}"
  mv mobile-start-receipts.sqlite3 "mobile-start-receipts.sqlite3.pre-downgrade.$(date +%s)"
  ```

  The next keyed start creates a fresh store. The cost is that every key
  recorded before the move is forgotten, so a retry of one of those keys
  starts a new conversation rather than replaying; let in-flight phone starts
  settle first. Upgrading again needs no action. Do not move the old file back
  afterwards: it would drop every key recorded in the fresh store since.
- **The same applies to `agent`/`team`.** A relay from before those fields
  rejects a stored intent carrying them, so a rollback over a store holding a
  target receipt answers keyed starts with `503 create_receipts_unavailable`
  until the store is moved aside as above.
- **A failed spawn after a target is bound leaves an orphan directory**
  (`sessions/<sid>/attachment.json`, no transcript), the same exposure the
  temporary mark already has.
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
- **The whole request body is capped at 153,152 bytes** (`START_BODY_MAX_BYTES`)
  before any JSON parsing. The cap is checked on `Content-Length` and again on
  the bytes actually read, so a chunked or lying request stops at the cap and
  is never buffered whole. It derives from the field bounds (3 fields × 4096
  characters × 12 bytes for the widest JSON escape, plus 4 KiB, plus one
  128-character target name × 12 bytes + 64), so no compactly
  or ordinarily formatted UTF-8 body within the field bounds is refused by it. A
  body padded with more insignificant whitespace than that slack, or sent as
  UTF-16/32 JSON, can exceed it; no real client sends either. It applies to
  unkeyed callers too: the route cannot tell a body is keyed until it parses it.
  A body naming **both** `agent` and `team` at full escaped length can draw
  `413` before the `400 invalid_start_target` it would otherwise earn; either
  refuses it before anything is written.

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
