# Radient account retention prerequisite

This standalone change targets current `main`, not the Team Bridge planner or
old single-organization enrollment trigger. It changes credential storage only;
it is not an onboarding executor, service lifecycle, receiver factory or owner
set implementation.

## Why

AuthStore's former org-first dedupe lets account B replace account A's row when
both share an organization. The old constant-key adoption also turns an
unidentified historical row into a new account without proof. Neither is safe
for a caller holding an exact credential ID.

Radient's proposed identity is a JSON tuple of account ID and exact stored API
endpoint. Default organization is mutable metadata and does not create another
credential. Other providers retain their existing identity policy. API keys
and CLI-imported keys retain their existing no-dedupe rules.

Legacy adoption requires a unique match with nonempty stored hub metadata.
Unknown account/hub metadata is never inferred from the current default. Rows
are retained rather than automatically assigned to another account or deleted.
**Remediation candidate — authored, not yet validated:** generic metadata writes
no longer admit quarantine and Radient same-key writes preserve existing disabled
causes. The actual authorization-code/PKCE flow now verifies its freshly exchanged
bearer with `GET https://api.radienthq.com/v1/me`. The committed server contract
at `8c2a35ac9529940c78101acd81161304facc1240` (`internal/routes/routes.go:422`,
`controllers/auth.go:MeHandlerEcho`, `responses/auth.go:MeResponse`) authenticates
the bearer and queries its account. Admission requires matching, nonempty
`result.identity.account_id` and `result.account.id`; JWT labels are not this
account proof.

The flow captures the canonical token-endpoint/API-hub mapping before awaiting
anything. Unmapped custom token endpoints refuse before dispatch. Token and
profile requests disable redirects; timeouts are 30 and 10 seconds respectively.
A failed, malformed, inconsistent, unauthenticated or cancelled profile response
produces no completion for the host to persist. Changing the current configured
default cannot supply a hub or account identity.

An ephemeral, payload-bound, single-use completion survives only in-process
between the real flow and CLI/controller persistence. It expires after 60 seconds
and rechecks cancellation immediately before persistence. JSON/dictionary copies,
arbitrary metadata fields and refresh results cannot be consumed as admission;
it is not a persisted verification flag or a boundary against arbitrary Python
execution. Both hosts keep/recheck their cancellation guards after the profile
await. `AuthStore.complete_radient_login` is the only quarantine transition.
It preserves historical payload/id/key, skips known accounts and foreign-hub
rows, and preserves unrelated explicit disabled causes. A matching verified
completion may restore its own automatic `identity_unattributed` or existing
`invalidated-token` state; generic writes, refresh/retry and success bookkeeping
cannot release either. Existing post-write notifications are deferred until the
completion reaches its final active state. A preserved explicit disable yields
an unavailable outcome, without changing model defaults or operator profile.
Refresh preserves the existing account/hub rather than accepting new JWT account
labels or dispatching a grant to a changed/unmapped issuer.

The earlier audit incorrectly named credential sync as an OAuth exploit path:
`network/credentials/sync.py:_copyable_payload` forces `type="api_key"`, so that
specific path does not enter the OAuth hook. Generic metadata authority and
unconditional generic-upsert revival were the actual flaws.
Namespace collisions and ambiguous stored matches are explicit refusals.
An exact stored endpoint is only a storage partition; it does not establish
canonical-hub authority or permit sending a bearer to an arbitrary URL.

## Safe retained status and recovery

Usable credential inventories remain filtered. A separate read-only query returns
only retained Radient row IDs, never bearer payloads or inferred identity labels.
The shared CLI/TUI `login status` formatter distinguishes retained-but-unavailable
records from absence. Its restore line is **host-neutral by default**; the CLI
(`local-operator login-status`) adds the terminal command because its reader is
in a shell, while the TUI and the desktop receive the neutral sentence rather
than a slash or shell instruction they cannot follow. It explicitly says older
records have not been deleted or assigned to the account being signed into.
CLI/controller completion receipts use the same notice. An existing explicit
disabled state is reported as **stays disabled** — an outcome of a verified
sign-in, not a failure — and is never described as automatic quarantine.

This is a semantic implementation, **not visual approval**. Actual CLI output,
real styled TUI frames and affected desktop rendering still require the granted
validation/capture lane and independent design review.

## Validation boundary

The authored tests use the real AuthStore SQLite path, synthetic credentials,
close/reopen, existing exact-row refresh and a synthetic pre-send failure.
The store tests stub the parked-tunnel rearm hook to prevent synthetic logins
from starting services. Completion-host tests additionally stub existing profile,
model-default and cache-notification side effects; their flow producer, request
validation and SQLite persistence remain real against synthetic HTTP transport.
Explicit client pins and DEFAULT post-login selection
are distinct assertions; an explicit-pin pass does not prove default selection.
The latter is a required negative control before this change is publishable.

Existing OAuth row serialization at this source head is JSON, not an encrypted
row/AAD codec. Reopen evidence must not be described as encrypted-row proof.
No real credentials, keychain interaction, network enrollment, services, models,
proactive arming or production flags belong in this validation.

Actual baseline reproduction used main `44c9024588` through public
`AuthStore.upsert_credential` and real temporary SQLite: both same-org account
retention and unknown-legacy preservation failed. Both assertions passed on
published head `2b05de4f02`. The required default-selection negative initially
failed separately; after the quarantine correction, the two focused test
modules passed 24 cases, including reopen and synthetic refresh recovery.
Scoped Black/isort, flake8 and pyright passed on all three changed Python files
at published head `2b05de4f02`. Those 29 scoped passes did **not** prove admission
provenance; code/QA identified R1/R2 and design identified D1. Current uncommitted
remediation and new admission/revival/recovery/status negatives have not run.
The next gate must retain raw stdout/stderr, not only JSONL exit-code receipts.
Commands, frozen head and resource figures belong in the PR evidence. This
remains a draft prerequisite, not an independently reviewed completed onramp.
It does not claim fresh/upgrade/later-join onboarding,
account-isolated CoS history, shared-loop transport, service health or executor
idempotency. Those require separately reviewed implementation and real-path QA.
