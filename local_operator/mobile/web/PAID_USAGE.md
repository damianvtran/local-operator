# Cached provider usage on the paired phone

`/usage` opens a phone-local read-only sheet; it is not sent as a terminal
frontend-local command to a session runtime. The sheet GETs `/api/usage` through
the existing paired-phone authentication transport.

The gate runs before any cached data read. The backend public authority
`ProviderController.read_only_for_persisted_usage()` followed by the instance
`cached_persisted_usage_reports_read_only(provider=None)` (closing its owned
authentication handle in `finally`)
owns metadata-only admission and read-only cache access. Only deliberately
persisted accounts may contribute; ambient/environment/config-override accounts
must not be selected. No provider refresh, credential refresh, cache migration,
mkdir, chmod, WAL/schema initialization or store write is permitted.

The response includes normalized display amounts, allowlisted unit, provider ID,
paid eligibility/payability, family-cap scope and timestamp. It omits account
identities, credential IDs, labels/details, arbitrary notes and raw vendor data.
Window display identifiers are exact allowlists: 5-hour, 7-day, monthly and
lifetime metadata; family identifiers are Opus, Sonnet, Fable or ZRead only.
Unknown strings stay unknown; numeric reset timestamps may distinguish them.
No raw window/family label is forwarded. Included-plan continuation requires an
explicit shared, non-family included window to be exhausted. A family cap alone
never claims the shared plan is spent; its restriction remains independently.

New phone copy is authored in the existing `relay.shell` English catalogue and
context sidecar; desktop new copy uses `ui.chat`. Their existing generators must
run under a separately allocated lease before compilation: generated key maps
and the phone bundled catalogue are deliberately not edited by hand.

The route consumes the real `ReadOnlyUsageSnapshot.reports`, never iterates
the snapshot as a list. Its envelope preserves `may_be_stale` and epoch-ms
`cached_at`; the phone discloses the immutable-cache timestamp and any WAL
staleness caveat. `unavailable=True` is a controlled 503, not an empty report.
An absent cache means an empty display, never a measured zero-credit balance;
an admission/read failure is a controlled unavailable response, not empty data.

Payability is the backend's fraction-first `UsageLimit.paid_usable()` result.
The phone does not normalize vendor payloads, infer eligibility from funding,
or invent denominators/percentages/currency conversion. A blocked wallet keeps
its actual balance and says `Spend unavailable`. Eligible paid headroom can
continue after included quota is exhausted; model-family restrictions remain
unless the provider explicitly says paid allowance supersedes them.

## Refresh

`GET /api/usage` stays write-free. A separate, authenticated and
origin-checked `POST /api/usage/refresh` (body `{"force": bool}`) is the only
phone path that reaches vendors. The sheet paints the GET, then posts
`force:false` on open (served by the 5-minute cache, the cross-process lease
and per-account backoff, so reopening costs nothing) and `force:true` from the
refresh button.

- **Who is probed.** `ProviderController.for_persisted_usage_refresh()` targets
  persisted logins only. Environment, runtime and config keys are neither
  targets nor the API key a stored login uses
  (`AuthStore.get_persisted_api_key`).
- **What it writes.** The usage cache row under the exact persisted key the GET
  reads (`_persisted_account_fingerprint`), its lease rows, and OAuth tokens
  rotated by the vendor in `auth.db` — the writes the TUI's background warmer
  already makes.
- **Floor and single flight.** One refresh runs per daemon; a concurrent POST
  joins it. A forced refresh inside `desktop_quota.REFRESH_FLOOR_MS` (15 s)
  runs unforced and answers `refresh.status: "throttled"` with
  `retry_after_ms`.
- **Bound.** The route waits up to 8 s, then answers with the cached projection
  and `status: "timeout"`. The task is shielded, never cancelled: cancelling
  mid OAuth rotation could lose a refresh token the vendor already replaced.
  A failure answers the cached projection with `status: "failed"`.
- **Response.** The GET envelope plus `refreshed_at` (the OLDEST report's
  `fetched_at`), `refresh: {status, forced, retry_after_ms}`, and `source:
  "live"` only when a report was fetched after the refresh started. The POST
  answers from its own results because the GET's `immutable=1` read cannot see
  WAL frames a refresh just wrote.

Both routes add, without changing earlier keys: per report `provider_name`,
`account_index`/`account_count` (login order, assigned before identity is
stripped — no email or hashed id crosses the tunnel), `state`
(`providers.usage.usage_report_state`, shared with the desktop catalogue) and
`credential_invalid`; per limit `remaining_fraction` (clamped 0..1) and `tone`.

Dependency order: paid-health contract PR, strict read-only authority PR, then
these panels. Source tests mocking that public method are not proof that the
method exists or that store writes are absent; real authorized HTTP and
filesystem side-effect checks plus rendered mobile frames are required before
merge. No live credentials or vendor requests are needed for those fixtures.
