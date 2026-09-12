# Desktop control API

The desktop control plane is additive. It reuses provider registry/auth-store and
`settings_io` authorities; it does not replace legacy chat, transcript, or SSE
contracts. Feature negotiation, not just the installed package version, decides
which controls a client may offer.

## Transport and trust

`GET /v1/capabilities` is public and returns a `CRUDResponse` with
`desktop_contract`, `desktop_available`, `desktop_auth`, and a versioned `features`
map. Version 1 advertises `auth`, `settings`, `commands`, `catalogues`, `lifecycle`,
`mcp` and `radient`. These version backend HTTP subsystems, not renderer completion
or third-party authorization. See [DESKTOP_CONTROLS.md](DESKTOP_CONTROLS.md) for the
all-command acceptance matrix and the new control routes. A missing route means an older backend;
show an update/setup action rather than falling back to an unprotected write.

Electron **main**, not the renderer, generates a random 32-byte token for each
managed backend lifetime. Supply it only through `LOCAL_OPERATOR_DESKTOP_TOKEN`
in that child's environment. Never put it in argv, logs, config files, build
variables, renderer storage, or URLs. Main's typed IPC adapter supplies
`Authorization: Bearer <token>` for allowlisted operations, verifies its owned
main-frame sender and rejects redirects. Browser development uses a server-side
proxy with the isolated token; client JavaScript must not receive it.

New routes return 503 if the backend was not started with a token, 401 for a
missing/wrong bearer, and 403 for an unapproved Origin. No-Origin requests still
need the token. Origins are rejected unless they exactly match the comma-separated
`LOCAL_OPERATOR_DESKTOP_ORIGINS` environment setting. `null` is never allowed.

In explicit desktop-token mode the legacy `/v1/config`,
`/v1/config/system-prompt`, and `/v1/credentials` reads/writes also require that
bearer. Otherwise they would bypass the new central-control boundary. Unmanaged
legacy servers retain their old behavior; this is not a redesign of every legacy
route's security. Central Radient credential consumers (`/v1/models`, speech,
transcription and agent ZIP upload) also require this boundary in managed mode;
see DESKTOP_CONTROLS.md for their compatibility resolver and media-relay obligations.
Legacy chat and SSE remain unchanged. Sensitive responses are
`Cache-Control: no-store`; rejected input is not echoed by validation responses.

## Providers and accounts

All following responses use `{status,message,result}`. No status response
contains an API key, access token, refresh token, or complete stored grant.

- `GET /v1/auth/providers`: `result.providers` contains canonical provider rows
  with `id`, `name`, `storage_id`, `search_aliases`, `auth_methods`, `local`,
  `accepts_api_key`, `configured`, `stored_credentials`, and non-secret metadata.
  Each method carries `id`, `label`, `kind` (`browser`, `device`, `api_key`),
  `requires_secret_input`, and `paste_fallback`. Aliases are methods on their
  storage provider. The mock test transport is not an end-user provider.
  `configured` means credential presence, **not** a successful connection test.
- `GET /v1/auth/status`: redacted stored account identities and credential types.
  Environment credentials are not removable stored accounts.
- `POST /v1/auth/login` with `{provider: <method id>}` starts a login operation.
- `GET /v1/auth/operations/{id}` returns `id`, `provider`, `state`, `message`,
  `auth_url`, `instructions`, `input_required`, `prompt_id`, and `expires_in`.
  States: `starting`, `waiting`, `input_required`, `succeeded`, `failed`,
  `cancelled`, `expired`. Terminal responses clear the authorization URL.
- `POST /v1/auth/operations/{id}/input` with `{prompt_id,value}` answers only the
  currently pending prompt. Stale/repeated prompt IDs conflict (409). Input is
  ephemeral, masked by the UI, and must never enter the chat transcript.
- `DELETE /v1/auth/operations/{id}` cancels and joins the flow, closing callback
  listeners and pending input. Closing a status poll is not cancellation.
- `PUT /v1/auth/providers/{id}/key` with `{value}` stores an API key through the
  same AuthStore login tier as the terminal. Existing credential precedence is
  preserved; adding a key does not silently delete an OAuth account.
- `DELETE /v1/auth/providers/{id}/credentials` removes stored provider grants.
  Require explicit confirmation in the UI. Environment keys are unchanged.

Only one login may be active per backend: provider aliases share fixed callback
ports and credential targets. Operations time out after 15 minutes and only 32
receipts are retained in memory. A backend restart invalidates those receipts.
Provider OAuth implementations still own PKCE/state/callback validation and
refresh. The desktop injects a per-flow no-op browser opener and forwards the
published authorization URL to main to open **once**, not twice. Only HTTPS
provider destinations and HTTP loopback URLs are accepted. Device instructions
are display/copy content; input-required prompts are paste controls. The
QwenCloud usage-OAuth method also requires its inference API key; a device grant
alone is not an inference credential.

## Settings

- `GET /v1/settings`: `result.sections` has `name`, `title`, `scope`, `description`;
  `result.settings` has every registered key, label, help, kind, choices/members,
  bounds, current/default values, `is_default`, `empty_unsets`, and `redacted`.
  Search this complete projection by label/help/key. Scope comes from its section.
  Desktop theme and terminal `tui.theme` are different scopes, not synonyms.
- `PATCH /v1/settings/{key}` with `{value}` writes one typed value. Integers are
  actual integers (not booleans/fractions), numbers are finite, enum choices
  preserve type identity, lists use declared string members. Unknown keys are
  404; invalid/read-only values are 422. Empty unsetting fields delete the key.
- Cascade edits use `{value: <chain map>, base: <original chain map>}`. The base
  is mandatory because unchanged rows from a stale screen must not replace
  concurrent terminal edits or flatten their stored effort metadata.
- `POST /v1/settings/{key}/reset` deletes the stored value and returns the
  authoritative row. Literal dotted keys retain the registry's exact paths.

Only registered keys are serialized, never arbitrary config mappings. Endpoint
URLs with userinfo/query/fragment content have `redacted: true` and `value: null`;
they must not be blindly saved back as null. This surface accepts only endpoints
without inline credentials/query parameters; secrets need their own masked
credential flow. A broken config returns 409 without renaming/replacing it. The
single-process write lock serializes this HTTP server's merge/write operations;
it is not a cross-process compare-and-swap guarantee.

## Canonical viewer protocol

The existing runtime attach protocol remains `client: "attach"`; desktop
adapters additionally send `surface: "desktop"` only after discovering
`desktop-watch-v1` in the runtime record's capabilities. An old runtime is refused
before dialing, since it would otherwise mistake the HTTP proxy for a person at
a terminal. Terminal and phone handshakes retain their existing defaults.

`desktop_watch` renews the connection's `{visible, can_notify}` lease for 45
seconds. Visible means the selected conversation is actually in a focused,
visible window. Delivery capability is separate: a background Electron main
process can notify without making the model believe a person is watching. Main
must not set `can_notify` when native notifications are unavailable or denied.
Expired desktop leases count neither as interactivity nor as idle-runtime
residency. The runtime's existing heartbeat re-evaluates expired leases and
restores parked-gate OS fallback. A valid desktop notification lease suppresses
that fallback, so one gate does not produce both an Electron and runtime toast.

`AttachedSession(surface="desktop")` carries this metadata through its existing
runtime binding/recovery path. It goes cold rather than becoming the runtime
in the HTTP process; reconnect does not resurrect an expired desktop lease. Its
`bind_runtime`, `update_desktop_watch`, and identity-checked `answer_gate` helpers
are host adapters, not another session daemon. Detach still closes only the
viewer, never the canonical runtime.

The command registry and resolver now live in `local_operator/slash_commands.py`.
The TUI imports that same registry; canonical frontend capabilities derive from
it without importing the Textual application. A desktop command UI must consume
these names, aliases, argument and prompt-consumption semantics rather than
maintain a second vocabulary.

## Canonical HTTP sessions (implementation checkpoint)

These additive routes use the same bearer/Origin boundary and `no-store` policy
as the adapters above. Command/native-action, MCP and lifecycle adapters are now
specified in [DESKTOP_CONTROLS.md](DESKTOP_CONTROLS.md). The Electron streaming,
attachment and renderer surfaces still require their own acceptance gate. Do not
enable the production composer merely because these routes exist. Legacy chat/SSE
shapes are unchanged.

`DesktopSessions` shares one `DesktopSessionBridge` per canonical 12-lowercase-hex
session ID. An agent profile is not a session ID. Creation is explicit and writes
a small `sessions/<id>/desktop.json` draft marker with the chosen cwd; it starts
no runtime. This makes a blank desktop conversation reopenable after an HTTP
restart without inventing a new session per request. Ordinary transcript/session
origin metadata remain authoritative; no `desktop` origin hides the session from
terminal/phone lists. Older sessions use their saved frontend checkpoint cwd,
falling back to the parent of the config root when no cwd was retained.

| Endpoint | Request | Result inside `CRUDResponse.result` |
| --- | --- | --- |
| GET `/v1/desktop/sessions` | `limit` 1..500, default100 | `{sessions: [...]}` canonical rows plus explicit desktop drafts |
| POST `/v1/desktop/sessions` | `{request_id, cwd}` | `{session_id}`; cwd must exist |
| GET `/v1/desktop/sessions/{id}` | — | snapshot frame below |
| GET `.../{id}/history` | optional `before_id`, `limit` 1..500 | `{entries,has_more,cursor_missing}` |
| POST `.../{id}/messages` | `{request_id,text,images?,mode?:prompt|steer}` | `{status:admitted,command_id,duplicate,detail,replayed?}` |
| POST `.../{id}/commands` | `{request_id,command,args?,images?}` | `{command,result:SlashResult,replayed?}` |
| POST `.../{id}/answers` | `{epoch,request_id,value,question_index}` OR `{epoch,request_id,approved}` | runtime receipt; stale runtime/request/question409 |
| GET `.../{id}/events` | optional `epoch`, `after_seq` | authenticated SSE, `data: <DesktopSessionFrame>` |
| POST `.../{id}/watch` | `{subscription_id,visible,can_notify}` | `{lease_seconds:45}`; disconnected/wrong-session ID404 |
| POST `.../{id}/notified` | `{completion_token}` | `{claimed:bool}`; cold, never marks read |

Create/message/command `request_id` is a canonical lowercase UUID string, reused
for a retry of the **same** operation. Answer `request_id` is instead the pending
gate's opaque ID, and answer `epoch` is the **runtime** epoch from frontend state,
not the HTTP stream epoch. Approval booleans and question indices are strict.
Answer bodies are never retained in the HTTP receipt journal or echoed back.

Images use the runtime's `{data_b64,mime_type}` shape (png/jpeg/gif/webp), at most8;
the encoded message/command body must fit900,000bytes. Empty prompts without an
image, invalid base64 and slash text on `/messages` return422 before runtime binding.
The existing Electron request transport currently has a smaller262,144byte body
budget: this checkpoint does not claim larger native image uploads work.

The command endpoint now accepts every shared canonical command and alias.
Runtime controls return actual SlashResult data; native/interactive controls return
typed destination, arguments and form/source/submit metadata, never fake execution
success. See [DESKTOP_CONTROLS.md](DESKTOP_CONTROLS.md) for all 35 rows and explicit
frontend responsibilities. On a `team_attached` or `agent_attached` result, the
runtime's `data.request` plus images is admitted **once by the bridge**
under the original request UUID; an added `result.admission` records that fact.
The renderer must not independently re-submit that consumed request.

### Admission and retry semantics

A200 message receipt means the canonical runtime acknowledged admission, not
that the model succeeded or the turn completed. The runtime's canonical events
and durable history are the authority for completion and side effects. Explicit
mutations use `AttachedSession.bind_runtime` / existing `engage_runtime` lease
arbitration. A cold read or stream attaches only to an already-live runtime.
HTTP shutdown/last-reader cleanup only disposes the viewer; it never stops work.

The private0600 `desktop-receipts.db` stores request fingerprints and completed
non-secret responses, not raw request or answer bodies. Reusing a UUID with
changed input returns409. A completed receipt survives HTTP restart. A control
interrupted between durable reservation and result commit is **indeterminate**:
a retry returns409 and requires state reconciliation, not another side effect.
Only natural prompt/steer admissions can retry an indeterminate receipt, because
the runtime already reserves those UUIDs durably. This is at-most-once control
execution with honest crash ambiguity, not a claim of transactional exactly-once
execution across the HTTP worker and the canonical runtime. Receipts currently follow
the retained session lifetime; no automatic deletion/expiry is claimed.

### Stream ordering and lifecycle

Frames carry `{session_id,epoch,seq,type,payload}`; `heartbeat` and overflow `gap`
carry only `{session_id,type}`. This outer epoch/seq is the HTTP **semantic receipt
cursor**, independent of the inner canonical frontend `{epoch,sequence}`.

1. `open` supplies `{subscription_id,gap,watch_ttl_seconds}`. Its seq is connection
   metadata, **not** permission to discard replay up through that number.
2. If retained, ordered frames after the supplied receipt cursor are replayed.
   This includes semantic `event` frames already covered by newer paint state.
3. `snapshot` follows replay, with `{frontend:FrontendSync,history,cold}`. Its
   history page ends inclusively at the captured canonical history cursor, not
   at a newer tail read after an await. Missing replaced cursor produces
   `cursor_missing:true` and an empty page; use `/history` to reconcile that gap.
4. New frames continue in receipt order: `frontend.update` is a canonical field
   delta, and `event` carries a typed canonical AgentEvent. Apply the snapshot
   after replay so an old cumulative record cannot repaint newer snapshot text.
   Preserve runtime sequence/epoch checks independently of semantic event dedupe.
5. `notification` carries one bridge-composed banner:
   `{contract,kind,title,status,body,body_is_snippet,body_is_failure,`
   `title_is_session_name,dedupe_key,completion_token,session_name,`
   `focus_policy}`. It is NOT an `AgentEvent` and must not be painted into the
   transcript; a renderer that does not know the type ignores it and still
   advances its receipt cursor.

### The `notification` frame

Emitted **if and only if** the bridge observes a newly published, unseen
`completions` row whose kind is `complete` or `error`, on a bridge that has
already taken a baseline attention reading. No engine event triggers one —
in particular `turn_end` is ONE MODEL CALL and `agent_end` routinely arrives
while `task` children still run, so neither is the turn-over signal. The
authority is `Session._publish_attention_outcome`, which makes that decision
inside the process owning the job manager; the bridge observes it rather than
re-deriving it, which is why a frontend cannot disagree with the TUI.

`ask`/`approval` never travel as `notification`: they already reach the app as
`pending_gate` in the snapshot and update frames, and a second channel for the
same card would duplicate it. `interrupted` is suppressed (the user pressed the
key themselves).

The frame is **live-only, never replayed**. An edge whose whole value is
timeliness must not arrive hours after a reconnect, and a retained notification
would push a real transcript event out of the 256-frame budget. Its `seq` still
advances, which keeps `seq` monotonic across both publish paths; the gap
arithmetic is unaffected because `gap` is computed against the oldest RETAINED
frame. The durable signal for a missed completion is `attention` plus the
sidebar's unseen mark, both of which survive a reconnect.

`body` says what happened rather than that something happened. For `complete`
it is the last assistant line (`body_is_snippet:true`); for `error` it is the
session's own recorded failure text (`body_is_failure:true`), read from the
`session_incident` record's unrendered `raw` so the banner names a cause the
user can act on instead of only "Stopped with an error". Both degrade to the
house sentence when unavailable, and BOTH are suppressed entirely when
`display.notification_session_name` is off — one flag over every
session-derived fact, decided in the backend. The two flags are never both
true. An `interrupted` or `complete` body is never the failure text and an
`error` body is never the last assistant line: a failing turn's last sentence
routinely reads as a success.

Parked gates do **not** travel as `notification`. They keep using
`pending_gate` in the snapshot and `frontend.update` frames, which the app
already receives; a second channel for one question is how one gate becomes
two banners. `pending_gate` gained an **additive, optional** `session_name` so
a gate banner can be triaged — "Waiting for approval" with several sessions
open names none of them. It is `""` when
`display.notification_session_name` is off, and absent/empty on an older
backend, so it is additive in both skew directions and an unchanged renderer
keeps drawing the anonymous card it draws today.

`payload.contract` (`1` today) is advertised as `features.notification_contract`
in `/v1/capabilities`. Additive fields do not bump it. A renderer seeing the
capability owns every completion banner and must stop toasting on `agent_end`,
or one turn produces two banners; a renderer not seeing it is talking to a
backend that composes nothing.

`POST .../{id}/notified` claims the right to raise ONE banner for a completion,
through `AttentionStore.claim_delivery(..., backend="desktop")`. Exactly one
surface wins per `completion_token`, so a TUI observer and the desktop watching
the same idle session produce one toast between them. Claim-then-deliver means
the claimant must BE the deliverer: call it immediately before constructing the
OS notification and never before the focus gate, because a claim taken for a
banner that is then suppressed is delivered to nobody, permanently. It is cold
(no bridge acquire, no runtime spawn) and it **never** advances the read
watermark: `unseen` and the sidebar mark survive it untouched.

A cold reconnect, HTTP restart, detached interval, expired replay cursor or future
cursor requires a gap snapshot. One live shared bridge retains at most256frames
and8MiB; each subscriber has the same backlog bounds. Overflow emits `gap` and
closes instead of silently losing semantic events. Max32subscribers per session,
64cached bridges; only idle bridges are evicted. Job trajectories stay out of
snapshots and deltas, through existing canonical serializers. Per-job trajectory
retrieval is not exposed by this HTTP checkpoint yet.

Watch ownership belongs to an active SSE subscription, not an arbitrary window
ID. Visibility and native-notification delivery are aggregated independently
across its unexpired leases. Renew while the surface is genuinely alive; no
heartbeat automatically grants presence. Expiry clears runtime presence, and ASGI
disconnect cleanup is shielded from the cancelled request scope so lease revoke
and detach actually reach the runtime. A stream alone means neither visible nor
notification-capable. Electron must set can_notify=false until native delivery
really exists.

Notification CONTENT and cross-surface arbitration are now backend concerns: the
`notification` frame above carries composed `{title,status,body}` and
`/notified` arbitrates who may raise the banner. What remains outside these
routes is the renderer's own business — its local TTL dedupe map (keyed on the
backend-minted `dedupe_key`), its focus gate, and notification-click behaviour.
The backend still never DELIVERS an OS toast for a leased desktop surface; it
composes for a frontend that does.

### Verification

`tests/e2e/test_desktop_sessions.py` drives real loopback HTTP and the production
Session/ServingSessionHandle/RuntimeServer/AttachClient with only the provider
stream scripted: same-session terminal controls, consumed team prompt, durable
single admission, actual runtime ask/approval futures, invalid/stale answers,
ordered replay, session isolation, disconnect/watch cleanup and reopen. Two of
its cases assert the `notification` contract on the ordered frame log, which is
the only shape in which a COUNT is checkable: a real multi-step turn (a tool
call, then an answer) emits several `turn_end` frames, one `agent_end` and
exactly one `notification`; and a turn that ends with a registered `task` child
still running emits none at all until the child settles and its re-entry turn
completes. `tests/unit/server/test_desktop_notifications.py` covers the frame's
gating, the replay exemption, the gap arithmetic and the `/notified` claim.
`tests/e2e/test_desktop_spawn.py` additionally executes the real detached process
launcher using the built-in test provider, then recreates the HTTP lifespan and
checks stable identity/title, persisted receipts, authoritative history and epoch
reset. The test-owned runtime exits through its own stop protocol. Neither test
uses operator credentials or connects to operator sessions. Unit tests exercise
receipt crash ambiguity, byte/count overflow, lease aggregation, cache isolation
and the inclusive history paging boundary.
