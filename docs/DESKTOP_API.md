# Desktop control API

The desktop control plane is additive. It reuses provider registry/auth-store and
`settings_io` authorities; it does not replace legacy chat, transcript, or SSE
contracts. Feature negotiation, not just the installed package version, decides
which controls a client may offer.

## Transport and trust

`GET /v1/capabilities` is public and returns a `CRUDResponse` with
`desktop_contract`, `desktop_available`, `desktop_auth`, and a versioned `features`
map. Version 1 currently advertises `auth` and `settings`. Unadvertised features
are not implemented by this contract. A missing route means an older backend;
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
route's security. Legacy chat and SSE remain unchanged. Sensitive responses are
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
