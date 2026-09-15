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
or third-party authorization. Later keys join the same map as they are added —
`diagnostics` (the `/info` host read and the `/session` ledger report) is one — and a
key's presence is the whole negotiation: a control whose key is absent must show an
update action rather than call a route the backend does not have. See
[DESKTOP_CONTROLS.md](DESKTOP_CONTROLS.md) for the
all-command acceptance matrix and the new control routes. A missing route means an older backend;
show an update/setup action rather than falling back to an unprotected write.

Electron **main**, not the renderer, generates a random 32-byte token for each
managed backend lifetime. Supply it only through `LOCAL_OPERATOR_DESKTOP_TOKEN`
in that child's environment. Never put it in argv, logs, config files, build
variables, renderer storage, or URLs. Main's typed IPC adapter supplies
`Authorization: Bearer <token>` for allowlisted operations, verifies its owned
main-frame sender and rejects redirects. Browser development uses a server-side
proxy with the isolated token; client JavaScript must not receive it.

The bearer has **two sources**, and they exist for two different daemons:

1. **The environment token** above, for a backend main started itself.
2. **The claim handshake**, for a daemon main did *not* start — the one the
   user's TUI, a terminal or launchd started, which has no capability in its
   environment. Such a daemon mints `claim_key = secrets.token_urlsafe(32)` at
   startup and publishes it in its **discovery record** at
   `<config>/run/serve/<pid>.json`, `0600` inside a `0700` directory — the same
   boundary that already protects a session's `control_key`. Main reads the
   record, finds the daemon, confirms `instance_id`, and presents the key once
   to `POST /v1/desktop/claim` as `Authorization: Bearer <claim_key>`. The key
   is never returned by a route, never logged, and never written anywhere but
   the record. A daemon the app started publishes no key at all (the
   environment already governs it, and its claim is refused with `409`).

`POST /v1/desktop/claim` is the **only** desktop route not behind the bearer,
because it is the door that bearer stands in front of — gating it would mean no
app could ever attach to a discovered daemon. It answers `200` with
`{"claimed": true, "instance_id": …}` once; `409` if the plane is already
governed (the latch is one-way — one app owns a daemon for its lifetime); `503`
if the daemon published no record and therefore no key; `401` for a missing or
wrong key.

Its body is optional and carries at most `origins`: a list of the origins the
caller's **renderer** is served from, which the claim adds to the allowlist.
Each entry must be a plain `http(s)` origin with no path, query or fragment;
`null`, `*`, a non-string, an unparseable value, or an origin the operator's own
`LOCAL_OPERATOR_DESKTOP_ORIGINS` list excludes is refused, each with its own
error detail. The field exists because a packaged renderer loads from `file://`,
whose origin is the literal `null`, which is never admittable — so without a
declaration the app could not admit the dev-server origin it is developed
against. A native caller may also simply send an `Origin` header.

**A browser can never claim**, even holding the key: a claim request carrying
`Sec-Fetch-Site` is refused. That header is attached by the browser's own fetch
stack and is on the forbidden-header list, so page script cannot forge or remove
it, while the intended caller (main) sends none. Defence in depth rather than a
capability boundary — the request that installs an Origin puts the *spender* on
the allowlist, so a leaked key must not be spendable by page script.

New routes return 503 if the backend was not started with a token **and no claim
has been accepted**, 401 for a missing/wrong bearer, and 403 for an unapproved
Origin. No-Origin requests still need the token. Origins are rejected unless
they exactly match the allowlist in force: the comma-separated
`LOCAL_OPERATOR_DESKTOP_ORIGINS` environment setting, plus whatever the accepted
claim installed. `null` is never allowed.

**A daemon is `503` or governed; there is no third state.** Accepting a claim
puts the daemon into the same posture the app imposes when it starts the backend
itself — `/v1/capabilities` flips `desktop_available` to `true`, the legacy
control surface (`/v1/config`, `/v1/credentials`, `/v1/agents`, `/v1/jobs`,
`/v1/schedules`, `/v1/models`) becomes bearer-gated for every other local
caller, and an `Origin` that is not on the allowlist loses its CORS grant
altogether. That last part — the echo, not the bearer check — narrows only when
the allowlist in force is **non-empty**: an `Origin` that is not admitted gets
no grant, and an allowlist-less daemon keeps the historical wildcard CORS. That
residual is deliberate and measured: the shipped app's renderer is loaded from
`file://`, so it sends the opaque origin `null`, and it reads `/health`
directly as its "server offline" signal, so suppressing the echo there made the
app report a healthy daemon as down. This is why the docs do **not** claim that
a claim removes the wildcard echo on every daemon: it removes it where the claim
admits an origin, and the control half above is what protects the rest. The cost
is deliberate and named: an operator's own `curl` script against a claimed
daemon starts seeing `401`, and the daemon logs an audit line naming the
instance and the origin the claim installed so that is diagnosable.

In managed mode — a `LOCAL_OPERATOR_DESKTOP_TOKEN` in the environment **or** an
accepted claim, which are one posture — the legacy `/v1/config`,
`/v1/config/system-prompt`, and `/v1/credentials` reads/writes also require that
bearer. Otherwise they would bypass the new central-control boundary. Unmanaged
legacy servers retain their old behavior; this is not a redesign of every legacy
route's security. Central Radient credential consumers (`/v1/models`, speech,
transcription and agent ZIP upload) also require this boundary in managed mode;
see DESKTOP_CONTROLS.md for their compatibility resolver and media-relay obligations.
Sensitive responses are `Cache-Control: no-store`; rejected input is not echoed
by validation responses.

Two surfaces are deliberately **outside both gate families**, so no bearer and no
claim gates them: `/v1/chat`, `/v1/sse`, `/v1/ws`, `/v1/static`, and the
`/v1/models/...` sub-paths (the gate matches the exact `/v1/models` template).
This predates the claim handshake and is unchanged by it; on a daemon whose
allowlist admits an origin, a claim still removes their CORS grant from every
other origin, so a page can no longer *read* them cross-origin, while an
allowlist-less daemon keeps the echo (see the `null`-origin residual above) and
an unauthenticated local caller can still reach them either way. Widening the
gate to cover them is a separate, larger decision and is recorded rather than
made here.

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

A lease is also the one signal that CREATES residency rather than only
preserving it. While a live lease says `visible`, the HTTP bridge starts the
session's runtime in the background (`refresh_watch`), because that is term 3 of
the residency predicate's own premise: a user looking at the conversation is
about to type, and paying the child spawn inside their first click is the stall
this removes. A lease that says only `can_notify` creates nothing: it is
delivery reachability, not attention, and it never did more than keep an
already-running runtime warm.

The bounds are the ones already in place rather than new ones, and they are
worth stating exactly, because the trigger is now "a window nobody is clicking
on":

* **One runtime per session** — the engage path is shared, so a lease-driven
  warm, an explicit `POST /warm`, and a command all serialise on the same lock
  and the losers return at their own `is_cold` check. One WINNER, rather than
  one spawn: the launch loop lets every contender spawn a candidate inside the
  ~300 ms before the transcript lease is taken and the losers exit 0, so a
  command landing in that window starts a doomed second candidate. That is the
  launch loop's designed behaviour and predates this change.
* **The lease is the lifetime.** The constant that decides how long the runtime
  is kept is the runtime-side `DESKTOP_WATCH_LEASE_S` (45 s today), which the
  bridge's own subscription lease `WATCH_TTL` happens to equal — same value by
  agreement, not by construction, and they must not drift now that one of them
  governs creation. Stop heartbeating and the lease expires, term 3 stops
  counting the viewer, and the runtime's existing idle drain reaps it. The warm
  itself dies with the bridge: it lives only while some user holds it (an open
  event subscription), so a lease that arrives with no subscription is recorded
  and creates nothing, exactly as `POST /warm` without one does.
* **The aggregate is the number of FOCUSED windows, not one runtime.** `visible`
  is `visibilityState === "visible" && hasFocus()` and the app mounts one
  lease-bearing chat view per focused window, so one warm exists at a time
  (~82 MB idle measured for the runtime, against the ~283 MB the reaper budgets
  for one) while every session touched in the last ~48 s may hold one
  transiently. This is why no cap is needed today — and the assumption is
  load-bearing: a future that mounts several lease-bearing views at once (a
  split pane, a per-pane lease, a "watching" surface that asserts `visible`
  without focus) multiplies the residency by the number of panes and needs a
  real cap.
* **A failure is paced, not repeated per heartbeat.** The bridge keeps a live
  lease's warm intent across attempts, but an attempt that actually ran waits
  out a backoff (30 s doubling to a 120 s ceiling) before the next one, so a
  session whose runtime cannot start does not spawn a child every beat. The
  pace follows the ATTEMPT rather than its outcome, which is what covers a
  runtime that comes up and then dies on every beat (a late boot failure): the
  charge stands while the window is still looking, and the next beat finds the
  viewer cold with the deadline still ahead. An attempt that was refused before
  doing any work — the viewer is in owner recovery, or another engage is
  already in flight — is retried at a 1 s poll instead, which is what carries
  the warm across the recovery window a single attempt used to lose to. The
  pace is dropped as soon as no live visible lease remains (the window was
  hidden, navigated away from, or closed), so a viewer who returns is not
  charged for the last intent's failures.
* **A deliberately stopped session stays stopped.** `lop stop` (or a `/stop`
  from another surface) ends the runtime whatever the presence says, and the
  lease-driven warm refuses while the session is stopped — the same fact
  `_recover_runtime` refuses on, so a focused window's beat cannot resurrect
  the session the user just ended; `/resume` re-opens it, and only a user
  action does. The limit of the guard is the marker's own: it is this facade's
  own flag, OR the durable `stopped_at`, which is written only for a session
  that HAS wakes — a stop from another surface on a wake-less session leaves no
  trace, so the guard closes the common cases rather than the whole class. An
  explicit `POST /warm` is still not gated: it is the keystroke path, a user
  action, and it is the route that has always kept a stopped session's
  `/resume` prompt honest by routing the send into it.

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

**A new conversation can be born on a chosen model and reasoning level.**
`model` is an **optional, additive** field of both `POST /v1/desktop/sessions`
and its `preview` twin, carrying the same three fields the canonical frontend
state publishes for a conversation's model — `{provider, model_id,
reasoning_effort}` — so a picker row can be handed back unmodified.
`reasoning_effort: null` means "no level chosen", not "this model has no
ladder": the conversation is then born on the configured `model_effort` (clamped
to the model's ladder), so a level nobody picked is never stored and never
replaces the machine's configured one.

- **Omitted or null ⇒ unchanged.** The body, the `desktop.json` marker (still
exactly `{version, cwd}`) and the launch are byte-for-byte what an earlier
client produced.
- **A pick of a model with no level is not a pick of a level.** The marker
records `reasoning_effort: null`, and the pane, the child's construction and the
first turn's admission row all resolve the machine's configured `model_effort`
(clamped into the picked model's ladder) — exactly what a launch that named no
model resolves. The model's own default rung is a seed, not a choice: it is never
STORED. It IS carried when the machine configures no level, because it is then the
machine's own resolution — and it has to travel explicitly rather than be left
out, because the birth sample rides the owner's model RPC as well as the spawn
environment and that RPC rebuilds the spec from the model's metadata, reseating
the conversation on the seed.
- **Rendered before it is refused, never after.** An unknown provider, a model
id the provider's catalogue does not serve, a pair that cannot be resolved into
a spec, or a level the model's ladder does not offer is refused with `422`
(`detail.code` ∈ `provider_unknown`, `model_unknown`, `model_unavailable`,
`effort_unsupported`) on **both** routes, which apply the same admissions in the
same order — working directory, model, target — and so answer the same refusal
for the same body. Refusals happen before anything the REQUEST would write,
including the create route's receipt claim, so a corrected retry of the same
`request_id` still creates. Two scopes are stated rather than implied: the
directory and model admissions are pure reads (a `stat`; the catalogue and the
metadata cache), while the target admission builds the registries `create` would
build anyway, and on a fresh root that materialises `<config>/agents` — the same
carve-out the `preview` route documents. A retry of a request that SUCCEEDED is
answered from its receipt without re-running any admission, so a working
directory that has since vanished does not turn a success into a failure. A
catalogue that cannot be enumerated offline (an aggregator on a cold cache, a
local endpoint) is not a refusal: an unserved pair then surfaces at the first
turn, exactly as it does today.
- **Preview answers the readings for the conversation that would be created** —
the identity *and* the spec the first turn will run on, from the same synthesis a
cold open uses — while staying session-less and side-effect free: no directory, no
marker, no receipt row. With a `model` that is the picked pair at the level it will
run at; with **no** `model` it is the CONFIGURED pair, resolved through the same
model metadata, so the effort ladder and the level are answered for an unpicked
draft too. (A config-only projection answered an empty ladder and no level, which
hid the desktop strip's effort chip and left its picker unreachable on every new
conversation.) Which field is whose, stated precisely: the LADDER
(`reasoning_efforts`) and the LEVEL are MODEL-derived and answered here; the
WINDOW is model-derived here but ACCOUNT-derived in a real cold open, which may
apply account metadata (`resolve_context_metadata`) and a window the account's plan
scopes. A draft must not read account metadata at all (a synthetic stickiness key
would move a real account's stickiness), so an account-scoped window can still
differ between this payload and the first cold frame.
- **Create stores the choice in the marker** (additively, under `model`), and
the FIRST turn is born on it: the plane seeds the cold viewer from the marker,
which carries it into the spawn (`LOP_MOBILE_CHILD_PROVIDER`/`_MODEL`/`_EFFORT`
plus the explicit-selection override), so the constructed spec, the first
frontend snapshot, the first provider request and the selection row journalled
at admission all agree. The choice is stored in **normalised** form (provider
alias resolved, level lowercased).
- **The journal wins.** The seed applies only while the conversation has no
selection of its own; once its leased owner has journalled one — including a
later `/model` switch — re-opening or resuming resolves from the journal and the
birth seed is not re-applied. A marker naming a pair or level that no longer
exists degrades to the configured default (or the nearest rung of the model's
remaining ladder) rather than failing an open.
- Advertised as `features.draft_selection: 1`. A client that does not see it must
keep a draft's model and effort chips **inert** rather than dispatching into a
`422`; `features.draft_preview` alone means the backend can only *report* the
readings.

| Endpoint | Request | Result inside `CRUDResponse.result` |
| --- | --- | --- |
| GET `/v1/desktop/sessions` | `limit` 1..500, default100 | `{sessions: [...]}` canonical rows plus explicit desktop drafts |
| GET `/v1/desktop/sessions/search` | `q` (<=256 chars), `limit` 1..500, default100 | `{sessions:[{id,name,mtime,forked,rank,body_match}],query,limit}`, best match first |
| POST `/v1/desktop/sessions` | `{request_id, cwd, target?, model?}` | `{session_id}`; cwd must exist |
| POST `/v1/desktop/sessions/preview` | `{request_id, cwd, target?, model?}` | `{frontend: <wire sync payload>}` for a session that does not exist |
| POST `.../{id}/working-directory` | `{request_id, cwd}` | `{cwd,label,outcome:cold\|rebound\|unchanged,will_wait}`; gated by `features.session_move >= 2` AND `features.frontend_replace >= 1` |
| GET `/v1/desktop/sessions/{id}` | — | snapshot frame below |
| GET `.../{id}/history` | optional `before_id`, `limit` 1..500 | `{entries,has_more,cursor_missing}` |
| POST `.../{id}/messages` | `{request_id,text,images?,mode?:prompt|steer}` | `{status:admitted,command_id,duplicate,detail,replayed?}` |
| POST `.../{id}/commands` | `{request_id,command,args?,images?}` | `{command,result:SlashResult,replayed?}` |
| POST `.../{id}/answers` | `{epoch,request_id,value,question_index}` OR `{epoch,request_id,approved}` | runtime receipt; stale runtime/request/question409 |
| GET `.../{id}/events` | optional `epoch`, `after_seq`, `frontend_replace=1` | authenticated SSE, `data: <DesktopSessionFrame>` |
| POST `.../{id}/watch` | `{subscription_id,visible,can_notify}` | `{lease_seconds:45}`; disconnected/wrong-session ID404 |
| POST `.../{id}/notified` | `{completion_token}` | `{claimed:bool}`; cold, never marks read |
| POST `.../{id}/seen` | `{completion_token}` | `AttentionState`; 409 when the token is not this conversation's current completion |

`POST .../{id}/seen` is the read receipt, and **a 2xx from it means this
conversation is read**: `unseen: false`, the receipt advanced through the token's
sequence, both computed inside the same write transaction that admitted the call.
Two 409s are refusals that move nothing, and both mean the caller is looking at a
result the conversation has moved past: `unknown completion token` for a token
this conversation never published, and `superseded_completion_token` (a machine
`code` in the body, see `local_operator.session.attention.SUPERSEDED_TOKEN_CODE`)
for a real but no-longer-current one. Neither refusal carries state — the remedy
is for the caller to re-read its own attention state and acknowledge the token
that names, which is the projection it is already subscribed to. The receipt
contract itself, including the anchors and the mobile parity, is in
[ATTENTION.md](ATTENTION.md).

Create/message/command `request_id` is a canonical lowercase UUID string, reused
for a retry of the **same** operation. Answer `request_id` is instead the pending
gate's opaque ID, and answer `epoch` is the **runtime** epoch from frontend state,
not the HTTP stream epoch. Approval booleans and question indices are strict.
Answer bodies are never retained in the HTTP receipt journal or echoed back.

### Moving a live session (`POST .../{id}/working-directory`)

A live session's working directory can be changed with the route the terminal's
`/move` is built on: one shared implementation, so both surfaces answer "where
does this session work" the same way. `cwd` may be relative or carry `~`, and it
resolves against the **session's** current directory (not the server process's),
so `../sibling` means what the user sees. `outcome` is `cold` (nothing was
running; the next engage spawns in the new directory), `rebound` (a runtime was
retired and its successor is owed) or `unchanged` (the session was already
there; nothing was written and nothing was retired).

**At most once.** The request is receipted with `retry_safe=False`. A row that
FINISHED replays its stored receipt; a row still PENDING is answered `409
Request outcome is indeterminate. Reconcile session state before issuing a new
request` and is never re-executed — a relative target re-resolved against the
directory the first attempt may already have moved to is a second move
(`child/child`), and an absolute one can undo a later accepted move. A client
that still wants the move mints a NEW request id; it must not re-id an
indeterminate relative path.

**Refusals.** An ordinary refusal is a `409` carrying the session's own
sentence: mid-turn, a runtime too old to be moved, work that arrived during the
retire, an absent or unenterable directory, a marker that cannot be read or
written, another client attached (below), or an older desktop window (below). An
UNKNOWN owner outcome is NOT a refusal: if the retire request left the process
and no definitive answer came back, the owner may already have accepted, so the
answer is `503 {"detail": {"code": "move_outcome_unknown", "message": ...}}` — the
named-condition shape this ladder already uses (the `code`/`message` pair sits
under `detail`, which is what the shipped error handler and the desktop client
both read) — nothing is rolled back, and the
client reconciles before claiming either directory. A client must not treat
every move failure as a 409 refusal.

**Exclusivity, and why it is bounded.** A move is honoured by retiring the
runtime, and every facade attached at that moment engages a successor from its
OWN cwd, so a second attached client would ask for a runtime in the directory IT
believes in and the loser of that race decides where the session works. The route
therefore refuses while another actual attach is registered (`409 … open in
another terminal or attached client. Disconnect that client, then move again.`)
and while a mounted desktop viewer cannot render the replacement frame below
(`409 … Update the desktop app, then move again.`). Several desktop windows
behind ONE bridge are one attach and are unaffected: the owner counts attach
CONNECTIONS, not windows or visibility. The owner enforces the fence itself and
only for a caller that saw `exclusive-move-v1` in its attach capability list; an
owner that does not advertise it gets the `/reload` update guidance rather than a
silent unguarded retire. A TUI-initiated `/move` keeps the legacy,
non-exclusive shape — mixed-viewer target propagation is a non-goal of this
release, not something it solves.

Two facts a client must not assume away. **Durability:** the new directory is
written to this session's `desktop.json` marker and to the bridge's own `cwd`
*before* any runtime is retired, so a server restart or a bridge eviction resumes
the session where it was moved to; a refused move restores both. Two designed
exceptions, both reported as indeterminate rather than hidden: an unknown owner
outcome keeps the target, and a rollback that could not run leaves the marker at
the target while the facade returns to the owner's directory. A later move
settles that state against the LIVE OWNER'S OWN RECORD — never against the marker
alone, which is a copy the failed operation wrote — and repairs the durable copy
when the owner's record proves it stale. **Re-engage:** the successor is started by the retire
frame on the bridge, not by this request — the runtime leaves by the `retiring`
route (never `stopping`), which flips the viewer cold and engages a replacement
easily, and the successor's own bind is what publishes the new `frontend.cwd`
that settles the working-directory chip. A client therefore paints its own
optimistic value and lets the stream confirm it, rather than treating this
receipt as proof that the successor is up.

**The replacement frame.** A move installs the accepted directory on the facade
without the owner's epoch or sequence moving, so it cannot travel as an ordinary
`frontend.update`: that delta carries the owner's UNCHANGED clock, and a renderer
that requires a newer sequence drops it — leaving a mounted viewer painting the
old directory forever, with no successor frame to repair it in the cold case. The
bridge instead publishes, synchronously and exactly once per accepted move,
`frontend.replace`: `{session_id, epoch, seq, type, payload:{frontend, cold}}`.
The outer `epoch`/`seq` are the BRIDGE's own cursor (so the frame is ordered
against every delta around it), `payload.frontend` is the whole bounded
`FrontendSync` carrying the true owner epoch and sequence, and there is no
`history` field — this replaces the paint projection, it does not reset the
conversation. It is replayable, and a renderer applies it only when the outer
epoch matches the active stream and the outer `seq` is newer than the cursor it
held BEFORE the frame; duplicates and out-of-order copies are ignored.
Negotiation is additive: the events stream accepts `frontend_replace=1` and
`features.frontend_replace: 1` advertises it.

The route is advertised as `features.session_move`, which is **2** from the
exclusivity fence and the replacement frame onwards. A renderer must offer the
move controls only with `features.session_move >= 2` AND
`features.frontend_replace >= 1`; a move refuses before mutating while any
mounted subscriber has not negotiated the flag, so no successful move can leave
an already-mounted viewer showing the old directory. A renderer that does not see
those keys keeps its read-only working-directory chip; a typed `/move <path>`
must report the same degradation rather than firing a request such a backend
answers with a 404. `move` itself is NOT an owner command: a bare `/move` still answers a
`native_action` with destination `session.move`, which asks the renderer to open
its picker and claims nothing ran.

`GET /v1/desktop/sessions/search` is the CLI's `/resume` search over HTTP: the
same `local_operator.session.session_search` implementation the TUI picker and
the phone daemon run, so a query that finds a conversation on one surface finds
it on the others. It matches the session's displayed name, its id, an exact
case-insensitive substring of the conversation body (from the cached digest
index, re-digested only for transcripts that changed), and — when the query is
not already answered precisely — a bounded soft tier (prefix, word-order, edit
distance <= 2 on words of 4+ characters). `rank` is the relevance tier
(0 name, 1 id, 2 body, 3 soft) and `body_match` says the conversation is why the
row surfaced, so a client can label a row it would otherwise show with no
visible reason. `rank` is NOT MEANINGFUL when `q` is empty: an empty query is
the store listing (every session, newest first) and nothing matched anything, so
the tier is the constant 0 rather than "this matched on its name". A client that already holds the catalogue should merge these
results into the rows it has rather than replacing the list, and must skip the
whole call when `/v1/capabilities` does not advertise `features.session_search`
— an older backend answers 404 there.

Images use the runtime's `{data_b64,mime_type}` shape (png/jpeg/gif/webp), at most8;
the encoded message/command body must fit900,000bytes. Empty prompts without an
image, invalid base64 and slash text on `/messages` return422 before runtime binding.
The existing Electron request transport currently has a smaller262,144byte body
budget: this checkpoint does not claim larger native image uploads work.

The command endpoint now accepts every shared canonical command and alias.
Runtime controls return actual SlashResult data; native/interactive controls return
typed destination, arguments and form/source/submit metadata, never fake execution
success. See [DESKTOP_CONTROLS.md](DESKTOP_CONTROLS.md) for all 35 rows and explicit
frontend responsibilities.

**Every action receipt the desktop viewer declares is admitted here, once.** The
runtime defers that submit to the client whose auth frame declared the receipt
type (`SLASH_ACTION_RECEIPTS`), and the desktop viewer declares all of them, so
the bridge admits the request — for `team_attached`, `agent_attached` **and
`goal_set`** — under the original request UUID, with the body's images, and
records it in `result.admission`. The renderer must not independently re-submit
that consumed request. A receipt whose `data.request` is empty carries no action
(a detach, a listing, a status), and staged images do not turn one into a turn.

`admission.status` is a ONE-WORD answer to "did the owner take this text", and it
is the field a renderer branches on:

| `status` | means | `detail` |
|---|---|---|
| `admitted` | the owner acknowledged the admission | the owner's own sentence (`prompt admitted`, `steering queued`) |
| `pending` | the acknowledgement had not arrived when the receipt was sent; the request was written to the owner's connection, unacknowledged | `pending; the owner has not acknowledged it`, or `pending; the steer into the turn already running is not acknowledged yet` when the text went to the steer path |
| `failed` | the owner, or the transport, answered with an error — the request was **not** admitted | a vetted sentence naming the reason (`failed; the owner did not answer in time`, `failed; the session owner could not be reached`, an enumerated admission refusal) |

The wait before `pending` is bounded (2 s, well inside the client's 15 s ack
deadline) and returns the instant the owner answers, so a receipt is never
withheld for a running turn's duration and the ack deadline is never reached.
`admitted` and `pending` are therefore both prompt answers to the same question;
they differ in whether the owner had answered yet, and neither implies the turn
completed — the canonical events and durable history remain the authority for
that. On `failed`, the text was NOT delivered: re-issue it under a NEW request
id (`pending`/`failed` are honest answers for THIS id, so a retry of the same id
replays this receipt rather than re-delivering the text).

**A `pending` admission that fails LATER is announced, not logged.** The receipt
has been sent by then, so the failure has no caller left to reach; the host
publishes an `admission.failed` frame on the SESSION's stream
(`{request_id,command,status,detail}`, the same vetted `detail`) where the
mounted viewer reads it, and a viewer that has not connected yet receives it
from replay. A `pending` admission that SUCCEEDS announces nothing: the user row
appearing through the event relay is the confirmation, exactly as for any other
admitted prompt.

Set `/goal <text>` while a turn is running therefore behaves as it does on one
Enter in the terminal: the text is steered into the turn in flight rather than
parked, and the reply is never withheld for the running turn's duration.

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
   history page is the transcript's durable tail — the newest ≤100 entries,
   `limit` unchanged — read once for this frame, which is the same unbounded read
   `/history` serves. `has_more` means *older rows exist below the page*, not that
   the page is partial or lossy: `before_id` paging reaches them, and this is also
   the flag that can now flip to `true` on a reopen where it previously read
   `false` with the older rows silently absent. `frontend.snapshot.history_cursor`
   / `live_cursor` are the DEDUPE watermark for the paired state, NEVER a
   visibility boundary for the page: truncating the page at them silently dropped
   durable rows whenever a turn was mid-flight (the steer-drain case) or the
   viewer's state predated the rows, with no `cursor_missing` to signal it.
   Because nothing is cut, the snapshot can no longer report
   `cursor_missing: true` — an evicted or replaced cursor is not a state this
   frame can be in. An EMPTY page still means "reconcile through `/history`":
   that is now exactly the case where the paired state carries no `history_cursor`
   at all (no frontend refresh or checkpoint yet), and readers depend on that
   signal, so it is preserved deliberately rather than inferred.
4. New frames continue in receipt order: `frontend.update` is a canonical field
   delta, and `event` carries a typed canonical AgentEvent. Apply the snapshot
   after replay so an old cumulative record cannot repaint newer snapshot text.
   Preserve runtime sequence/epoch checks independently of semantic event dedupe.
   `frontend.replace` is the desktop-only REPLACEMENT of that projection,
   published once per accepted move and ordered by the BRIDGE's outer `seq`
   rather than the owner's clock (see "The replacement frame" under the move
   route). It carries no `history` field and neither creates a gap nor
   invalidates a history cursor.
5. `notification` carries one bridge-composed banner:
   `{contract,kind,title,status,body,body_is_snippet,body_is_failure,`
   `title_is_session_name,dedupe_key,completion_token,session_name,`
   `focus_policy}`. It is NOT an `AgentEvent` and must not be painted into the
   transcript; a renderer that does not know the type ignores it and still
   advances its receipt cursor.
6. `admission.failed` carries `{request_id,command,status,detail}` for a
   `/commands` admission whose receipt answered `pending` and which then failed
   (see "Every action receipt..." above). It is published LIVE AND RETAINED, so a
   viewer that connects after the failure still learns the text was not
   delivered. Like `notification` it is NOT an `AgentEvent` and must not be
   painted into the transcript — it is a notice about a request, not a turn —
   and a renderer that does not know the type ignores it and still advances its
   receipt cursor.

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

## The machine-wide event feed

`GET /v1/desktop/events` — authenticated SSE, the same bearer/Origin boundary as
every other route on this plane (503 when the backend was not started with a
token, 401 for a missing or wrong bearer, 403 for an unapproved `Origin`).

WHY A SECOND STREAM. Every notification channel before this one was per SESSION:
a `notification` frame rides the SSE stream of a session's bridge, and a bridge
exists only while a route holds one. The app holds exactly one — the session it
is displaying — so a completion in session B while the app showed session A
produced **no composed frame at all**, and on a machine with no TUI running a
finished turn was announced by nobody. This stream is one socket for the whole
process, and it composes for sessions that have no bridge.

**It acquires no bridge and spawns no runtime.** Its two dependencies —
`AttentionStore` and `compose()` over `sessions/<id>/` — are both
bridge-independent, and `tests/unit/server/test_desktop_feed.py` asserts the
absence directly (`DesktopSessions.bridges` stays empty across a feed cycle).
Reusing a bridge's composer instead would make watching a 200-row catalogue
build 200 cold facades and 200 SQLite poll loops, and would take the
`BRIDGE_COUNT` ceiling with it.

### Frames

The envelope's SHAPE is the session stream's (`epoch`, `seq`, `type`, `payload`,
plus `session_id` on the types that concern one session) so the relay needs no
new parser. `session_id` is absent on `open` and `catalogue`: the feed is not a
session, and a fabricated id would make `observe(sessionId, frame)` look like it
had one to attribute a catalogue event to.

```jsonc
{"epoch":"…","seq":12,"type":"open",
 "payload":{"subscription_id":"…","heartbeat_seconds":15,"lease_seconds":45,
            "watch_ttl_seconds":45,"catalogue_revision":98123,
            "attention":{"session/<id>":{ /* AttentionState */ }}}}
{"epoch":"…","seq":13,"type":"attention","session_id":"<12 hex>","payload":{ /* AttentionState */ }}
{"epoch":"…","seq":14,"type":"notification","session_id":"<12 hex>","payload":{ /* per-session payload */ }}
{"epoch":"…","seq":15,"type":"catalogue","payload":{"revision":98124}}
{"epoch":"…","seq":16,"type":"heartbeat","payload":{"ts":1699999999.5}}
{"epoch":"…","seq":17,"type":"gap","payload":{"reason":"overflow","subscription_id":"…"}}
```

- **`notification.payload` is the per-session payload**, built by the SAME
  function the bridge uses (`notifications.compose.notification_payload`), so
  `dedupe_key` is byte-identical and the app's local claim map collapses the pair
  into one banner. The ONE field the two derive differently is `focus_policy` —
  a ROUTING field, not content:

  | Where the completion is | `focus_policy` |
  |---|---|
  | the session a desktop window is displaying | **no feed frame at all** — rung 1 applied, the card is in band |
  | any other session | `always` |

  Reusing the bridge's `when_unfocused` here was a defect: the app suppresses
  that value whenever ANY window is focused, so in the commonest state of all —
  the user in the app on session A while B finishes — rung 2 had already
  silenced the runtime and the TUI and the frame was then suppressed by the
  app's own focus gate. The completion reached nobody. The window's focus says
  nothing about whether the user wants to hear that a DIFFERENT conversation
  finished, so the feed says `always`.
- **`attention` is live-only**, with the authoritative snapshot in `open`. It
  carries no `supported` key: only a live runtime can answer that, and the
  renderer's merge preserves the value it already holds rather than letting a
  catalogue-shaped frame clear it (otherwise a frame arriving for the session on
  screen disables its read receipt).
- **Neither `notification` nor `attention` is replayed.** The connection's first
  read is a BASELINE: it records the store's current revision and announces
  nothing that predates the connection. A reconnect therefore does not flood,
  and a completion missed while the app was away is recovered by the durable
  unseen mark in the `open` snapshot — never by a late banner.
- **The snapshot excludes the catalogue's rows.** Those carry a preview read per
  row; the client already has them, and putting that scan on the feed would move
  the sidebar's cost rather than remove it.
- **`catalogue`** is a cheap invalidation token over the sessions directory (its
  `(inode, mtime_ns)` plus the directory-name set, on a 1 s cadence), so the
  sidebar stops polling `sessions.list` on a 5 s clock. The 30 s safety poll and
  a refetch on window focus remain as drift insurance. An in-place transcript
  append does not move the token — deliberately, since noticing that would mean
  walking the store on every tick, which is the cost the poll was retired for.
- One live subscriber backlog bound (256 frames / 8 MiB), 32 subscribers;
  overflow emits `gap` and closes.

### The polling cadence

One task per process, started with the first subscriber and stopped with the
last. Each tick is TWO `os.stat` calls — `attention.db` and its journal, the
doorbell `config_watch` already uses for `config.yml` — with a SQLite read only
when the store actually moved. That is what takes background detection from the
1 s poll's floor to a p50 of roughly one tick. The revision gate stays the
AUTHORITY and the delta read is only an optimisation: a heal moves neither
`MAX(sequence)` nor `SUM(acknowledged)`, only the `mutations` counter, so a tick
that trusted the delta alone would miss it.

### The burst ceiling

At most `desktop_feed.BURST_LIMIT` (3) individual banners per tick. The
remainder is not dropped: it is announced as ONE digest frame carrying

- `burst_count` — the whole remainder, an absolute number rather than one
  relative to the ceiling;
- `session_ids` — the members, so the click can land on the catalogue rather
  than on an arbitrary member of the set;
- `completion_token: null` — no single completion owns the frame, so a client's
  own claim step skips the digest itself;
- `member_tokens` — `[{session_id, completion_token}, ...]`, the field the
  members are arbitrated through and the marker that identifies a digest frame
  to a client. A member is claimed one level down, through the same
  `POST /v1/desktop/sessions/{session_id}/notified` a single banner uses,
  immediately before the digest is delivered — never when the frame is merely
  queued, because a client may suppress the banner by its own focus rule and a
  claim taken then would burn a completion nothing announced. A member another
  surface claimed first is simply not this digest's to announce, and the count
  stays the backend's statement of what happened.

A DIGEST IS EMITTED FOR ANY NON-EMPTY OVERFLOW, INCLUDING A SINGLE MEMBER: four
completions in one tick are three individual banners plus a digest with
`burst_count: 1` and one `member_tokens` entry. So "is this a digest?" is a
question about `member_tokens` (or `session_ids`) being PRESENT, not about a
count greater than one — a client that keys on the count reads the smallest
digest as a private banner, finds no `completion_token` to claim, and takes no
claim at all, leaving that member open for a second surface to banner a second
time. The boundary is the smallest overflow, not the largest.

The TUI's own per-tick cap is the same number, asserted equal by a test rather
than left to convention.

## Desktop delivery presence

`POST /v1/desktop/presence` — the machine-wide answer to *"can a desktop app on
this host attempt a banner, right now?"*

```jsonc
// request
{"subscription_id":"<from the open frame>","can_notify":true,
 "can_notify_kinds":["complete","error"],"session_id":"<12 hex or omitted>",
 "window":{"exists":true,"focused":true,"visible":true,"minimized":false}}
// response
{"result":{"lease_seconds":45}}
```

A ROUTE RATHER THAN A FILE THE APP WRITES. The app may be paired to a backend on
another host, so it cannot write to that host's filesystem; the SERVER aggregates
what its live feed subscriptions report and materialises it as ONE RECORD PER
SERVE PROCESS at `<config_dir>/run/desktop/delivery/<instance_id>.json` (0700
directory, 0600 staged write), which every sibling process reads and unions.
Local and remote apps then behave identically — which is the whole reason this is
server-side.

ONE FILE PER PUBLISHER, because presence is a per-PROCESS assertion (one serve
process owns one set of live feed subscriptions) and a single machine-wide file
got that wrong in both directions: while every publisher wrote
`run/desktop/delivery.json`, the last WRITER decided the whole machine's answer —
a second server advertising `can_notify:false` revoked a live one's lease on
every beat — and the first process to EXIT deleted the file out from under the
other. The single file is still READ when it is present, because a sibling
started before this change writes it and a reader that stopped looking would go
blind to that sibling for the life of its process; **nothing writes it any
more**, and the records directory is the contract.

- Reachability (`can_notify`) and `can_notify_kinds` are UNIONED across records:
a banner either live backend can raise does reach this machine, and intersecting
them would let the weaker sibling veto the stronger one.
- The window state is TAKEN, not unioned — "which conversation is on screen" has
exactly one answer, and it comes from the freshest record that has a window (a
windowless sibling never blanks a window that is genuinely on screen).
- A record is believed only while its `pid` is alive and its `heartbeat_at` is
inside `PRESENCE_TTL_S`. A reader reaps a stale record in its ANSWER and does not
unlink it — the file belongs to a process that may be starting up again — so a
PUBLISHER sweeps the siblings it can PROVE dead (a dead pid, or silence past
`DEAD_RECORD_AGE_S`, twice the TTL) during its own write, re-identifying each entry
first so it cannot unlink a record a live sibling replaced in that window.
Without the sweep every process that died without running its exit path left a
file behind for good, and readers paid a read for it on the announce path and on
every banner decision.

- **The lease is held against the SSE subscription.** An unknown
  `subscription_id` is a 404, and a dropped socket REVOKES its claim. A presence
  withdrawn only on a missed heartbeat would leave a 45 s window in which every
  runtime on the machine stays silent for a banner nobody can raise.
- **`PRESENCE_TTL_S` (45 s) and `PRESENCE_BEAT_S` (15 s)**, matching `WATCH_TTL`,
  `DESKTOP_WATCH_LEASE_S` and `VIEWER_HEARTBEAT_TIMEOUT_S`. Three missed beats
  expire a claim, and a dead pid is reaped on read — the same two reaping rules
  as `scan_viewers`.
- **`can_notify` means "can ATTEMPT delivery"**, nothing stronger.
  `Notification.isSupported()` knows nothing about macOS Focus/DND, Windows
  Focus Assist, or a permission the user denied. A suppression costs a banner,
  never the durable `unseen` mark — a claim never advances the read watermark
  (`docs/ATTENTION.md`).
- **`can_notify_kinds` narrows the claim.** The feed carries completions only
  (`complete`/`error`), so a machine-wide presence must NOT suppress a parked
  `ask`/`approval`: nothing machine-wide would carry it, and the gate would go
  silently unannounced. Gates keep the per-session lease and the per-session
  toast they have today. Carrying gates on the feed is a follow-up.
- **`window` is the app's REAL window state**, reported by Electron main. The
  renderer's `document.visibilityState`/`hasFocus()` cannot answer it (a window
  behind another app reports `visible`; a throttled window reports `hidden`),
  and the desktop backend reads this field as "somebody is looking at this
  window" for the visibility rung.
- **`window.exists:false` discards `session_id`.** Closing the last window on
  macOS leaves the app alive in the dock, and its record may still name the
  conversation it was showing; treating that as "on screen" would suppress the
  banner for the one conversation the user cannot see.

### The producer contract: which fields buy which rung

THIS IS A CROSS-REPO CONTRACT AND NEITHER SIDE'S OWN TEST SUITE CAN SEE IT BREAK.
The backend derives rung 1 and rung 2 from the beat, so a producer that sends
only `{subscription_id, can_notify}` gets no rung at all — not an error, not a
fallback, just a quiet machine. The app's first implementation did exactly that,
and the observable result was the one defect this whole feature exists to fix:
rung 2 never engaged, the RUNTIME's rung 4 fired early and claimed the delivery,
and the feed then composed nothing for it — so a completion in the conversation
the user was LOOKING AT raised a banner.

| Field | What it buys | If omitted or false |
|---|---|---|
| `subscription_id` | binds the lease to a live feed socket | 404; nothing is published at all |
| `can_notify` | the app can ATTEMPT a banner (rung 2's eligibility) | rung 2 ineligible, rung 4 speaks |
| `can_notify_kinds` containing the kind | rung 2 eligible **for that kind** | **claims NOTHING** — a missing or empty list is not "all kinds". The feed carries `complete`/`error`, so the app must send `["complete","error"]`; a kind it does not send stays with the runtime, which is what keeps a parked gate's per-session toast alive |
| `window.exists:true` | there is a window that could be displaying something | `session_id` is discarded, and no session reads as attended |
| `window.focused` AND `visible` AND NOT `minimized` | rung 1's desktop arm: somebody is actually looking at this window | the session is not "watching", so a completion for it still banners (correct for an occluded window) |
| `session_id` | WHICH session that window displays | the desktop contributes no visibility for any session |

The beat must be renewed (`FEED_PRESENCE_BEAT_S`, 15 s) and the socket kept
open: the lease is revoked on disconnect, so a beat that stops is a machine
that believes nobody can deliver.

## The notification eligibility ladder

First match wins. It is a decision, not a race; the loser is never eligible
rather than "suppressed", and `claim_delivery` remains the tie-break AMONG the
eligible.

1. **A surface is WATCHING the session** — a TUI attached to it, a phone, or a
   desktop window genuinely displaying it. The card is in band; no OS banner.
   The predicate is the VISIBILITY one (`RuntimeServer.watching_surfaces()` →
   `_visible_attach_surfaces()`), NEVER `notification_surfaces()`. The latter
   answers "could a banner reach this person somewhere", which is a different
   question, and using it to suppress meant "this machine can banner" read as "a
   human is reading X": with the panel on X and the window behind another app,
   every OS surface went quiet while nobody was looking.
2. **A notify-capable desktop app** on this host claims the completion kind —
   the feed above composes it, so the runtime and the TUI stay silent.
3. **A TUI is running anywhere on this machine** — its 1 s background announcer
   raises it. Its viewer record is the signal, so a crashed TUI does not keep
   every runtime silent forever.
4. **Nothing** — the SESSION RUNTIME raises it, through
   `tui.notify.detached_notify` and the shared composed vocabulary, claiming
   through `claim_delivery(..., backend="runtime")` and handing the claim back
   if the spawn reported nothing went out. Its banner carries `session_id`, so
   the click lands in the app via the ladder below.

Rung 4 is a GATE rather than a race, because the runtime learns about a
completion earlier than every other surface (at turn settle, against the feed's
100 ms poll and the TUI's 1 s tick): announcing unconditionally would win every
completion and make both richer paths dead.

### The click ladder

`lop resume-click <id>`, first rung that works. THE ORDER IS THE OPERATOR'S
STATED REQUIREMENT rather than a preference between measurable options: the
requested destination is the DESKTOP UI, and the terminal is the fallback.

1. **A running desktop viewer** — switch it and raise its window. A record
   reporting `has_window:false` is a rung-1 candidate TOO: the app is alive with
   its last window closed, and its `resume_session` recreates the window and then
   navigates, which is exactly the "app is alive, its window is closed" click.
   Within the chosen surface, a viewer already displaying the target is preferred
   (a no-op switch), then the most recently focused, then the lowest pid.
2. **The desktop app is installed but not running** — launch it with the session
   id. Discovery order: the `desktop.launch_command` setting when set (argv with
   `{session}` substituted), else `local-operator-ui` on `PATH`, else
   `open -b com.local-operator --args --open-session <id>` on macOS. Each
   candidate is tried and abandoned only on a NON-ZERO EXIT, and a launcher
   still alive after the probe counts as success: a missing bundle exits 1 after
   starting fine, so "a child started" is not evidence that anything ran.
   `pnpm dev` and a repository checkout are deliberately undiscoverable and fall
   through.
3. **A running TUI viewer** — switched in place.
4. **Nothing suitable is running** — open a terminal. It reports a LANDING, not
   a spawn: the backend the registry detected is tried, then on macOS the
   AppleScript Terminal backend, which LAUNCHES Terminal.app and needs no
   terminal around this process — detection is the wrong question on a rung
   that only ever runs where none is discoverable — and nothing else. THAT
   macOS CANDIDATE IS WITHHELD OVER SSH, where there is no window server for its
   `tell application` to reach: osascript would accept the script and the spawn
   would report a landing that never happened, so an ssh session reports failure
   instead. A rung that opened no window returns False.

   The failure is then REPORTED TWICE, because the two reports have different
   readers. `lop resume-click` prints the `lop --resume <id>` receipt to stderr
   — what a hand-run gets, and what makes the path debuggable from a terminal —
   and the ladder posts the same sentence through `tui.notify.detached_notify`
   as well. A real click has no terminal to print into: the notifier is spawned
   with all three streams on `/dev/null` and its `NSTask` inherits them, so an
   ssh session, a non-darwin host and a hand-edited `desktop.launch_command`
   typo were otherwise clicks that did nothing and said nothing.

Rung 3 is asked for "whatever is left" rather than for a named surface, which is
what keeps the fallback identical for viewer types this build does not have; the
desktop's own preference lives in rung 1, and the surface filter is what narrows
this rung when the launch is refused.

THE DESKTOP IS A RUNG, NOT A TIE-BREAK, AND IT IS THE FIRST ONE. As a tie-break
it only ever decided between two viewers the user had never focused, so a TUI
focused once — ever — outranked it for good, and a terminal that happened to be
open swallowed every click before discovery ran: the user asked for the app and
got their terminal, decided by nothing but incidental focus history. Focus
history is not a policy for a destination the user named.

`LOCAL_OPERATOR_NO_DESKTOP_LAUNCH=1` TAKES THE APP OUT OF THE LADDER, not only
out of the launch: rungs 1 and 2 are skipped AND rung 3 is narrowed to non-desktop
viewers, so a running app cannot become the destination of a click whose launch
the user forbade. With nothing else running, the click falls through to the
terminal — the same place it lands when the launch is allowed and the app is
absent.

## Capability keys

| Key | Version | Advertises | Absent means |
|---|---|---|---|
| `desktop_feed` | 1 | `GET /v1/desktop/events`, `POST /v1/desktop/presence` and their frame/lease shapes | the app opens no feed, beats no presence, and keeps its 5 s catalogue poll and its per-session notification path verbatim |
| `desktop_presence` | 1 | the backend reads the per-publisher records under `run/desktop/delivery/` (plus the legacy `run/desktop/delivery.json` while an older sibling writes it) and defers its own completion banner to a notify-capable desktop | nothing is suppressed on the strength of a lease nobody publishes |

Neither bumps `notification_contract`, which stays 1: the payload is unchanged
except for the derived `focus_policy` routing field, which the client already
special-cases, and a renderer that ignores the new frame types keeps working. In
both skew directions the new behaviour is a no-op: a new backend with an old UI
never sees a presence file (so rung 4 raises the banner), and an old backend
with a new UI advertises no keys (so the UI keeps the poll and the per-session
path).
