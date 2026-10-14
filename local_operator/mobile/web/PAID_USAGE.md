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

An absent cache means an empty display, never a measured zero-credit balance;
an admission/read failure is a controlled unavailable response, not empty data.

Payability is the backend's fraction-first `UsageLimit.paid_usable()` result.
The phone does not normalize vendor payloads, infer eligibility from funding,
or invent denominators/percentages/currency conversion. A blocked wallet keeps
its actual balance and says `Spend unavailable`. Eligible paid headroom can
continue after included quota is exhausted; model-family restrictions remain
unless the provider explicitly says paid allowance supersedes them.

Dependency order: paid-health contract PR, strict read-only authority PR, then
these panels. Source tests mocking that public method are not proof that the
method exists or that store writes are absent; real authorized HTTP and
filesystem side-effect checks plus rendered mobile frames are required before
merge. No live credentials or vendor requests are needed for those fixtures.
