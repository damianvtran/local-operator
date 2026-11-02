# Native human account profile relay

The desktop edits the supported Radient account profile, not a local agent and
not a second copy of the account service. Upstream source of truth:
`agent-server/docs/ACCOUNT-PROFILE.md`, `internal/controllers/account_profile.go`
and `internal/responses/account_profile.go` in the Radient server repository.

## Closed desktop contract

All routes require the desktop control-plane bearer. The renderer never receives
Radient OAuth credentials. `features.account_profile = 1` advertises this contract.

| Route | Input | Result |
|---|---|---|
| `GET /v1/desktop/account/profile` | `expected_account_id` query | `{result:{scope,profile}}` |
| `PATCH /v1/desktop/account/profile` | same query plus `scope`; JSON partial fields | same normalized profile snapshot |
| `PUT /v1/desktop/account/avatar` | same query plus `scope`; raw image bytes | same snapshot; upstream multipart `file` |
| `DELETE /v1/desktop/account/avatar` | same query plus `scope`, `mode=uploaded|provider`, `confirmed=true` | same snapshot |

Unknown/duplicate query fields fail closed. Supported PATCH fields are
`display_name` (64 code points), `title` (80), `description` (500, at most eight
lines), and `timezone` (IANA, 64). Absent means unchanged; `""` means clear; null,
empty patches, unknown fields and pronouns are not accepted by this editor.
Upstream owns Unicode cleaning/normalization and full image validation.

Avatar bodies stop at 1 MiB before relay allocation. The upstream identifies
PNG/JPEG/WebP by content, checks dimensions before full decoding (4096 maximum
edge, 16 million pixels), rejects trailing data and re-encodes metadata-free
64/256 derivatives. The renderer also admits image headers before its own
bitmap allocation. Delete is always an explicit separate transaction: removing
the uploaded slot may reveal the saved provider image and removing the provider
slot may reveal an upload. Neither operation promises that both slots disappear.

## Authority and result admission

`TeamBridgeClient` is reused only for person OAuth, pinned credential selection
and transport, **not** enrollment, tenant status or membership. The cache-neutral
`_account_authority` captures config/destination, exact credential row, expected
account and durable login incarnation before any refresh await. The opaque
SHA-256 `scope` receipt contains no row id/token and is only a freshness value,
never an alternative to desktop authentication.

Writes compare that receipt, refresh the captured person row, require its exact
credential id, check again immediately before the HTTP request (after client
entry), and recheck after the bounded response. There is one upstream dispatch:
401 never retries under a refreshed/new login. A post-dispatch switch refuses
local success; **it cannot undo a write already committed upstream**.

The profile path does not admit, evict or populate private image caches merely
to check authority. Image readers retain their existing cache-admission wrapper.
All successful route responses are private/no-store. Only presentation fields
and the current account id are projected from upstream; no arbitrary account
metadata or pronouns are returned.

## Verification boundary

`tests/unit/team_bridge/test_account_profile.py` exercises synthetic HTTPX
transport: partial JSON bytes, normalized projection, stale/A→B→A receipts,
refresh/client-entry races, post-dispatch suppression, fixed multipart/remove
modes, no 401 replay, wrong-account replies and compressed-response rejection.
The existing avatar transport tests are direct regression neighbors for the
shared authority/streaming factor. These are not live account-service evidence.
Independent real-loopback HTTP QA, current-head CI and UI design/interaction
rounds are required before merge. Never mutate the operator's account to test.
