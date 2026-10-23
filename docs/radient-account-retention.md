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
After a login with a known account and nonempty stored endpoint, unidentified
Radient OAuth history is quarantined with the existing visible disabled cause
`identity_unattributed`; its row ID, identity key and payload remain unchanged.
This prevents an unidentified bearer from shadowing the new login in both the
default OAuth cascade and an unpinned Bridge client. Known accounts are not
quarantined. A payload compare-and-swap protects a row attributed by another
login during the scan. No grant revocation, credential deletion, or account/hub
substitution is inferred.
Namespace collisions and ambiguous stored matches are explicit refusals.
An exact stored endpoint is only a storage partition; it does not establish
canonical-hub authority or permit sending a bearer to an arbitrary URL.

## Validation boundary

The authored tests use the real AuthStore SQLite path, synthetic credentials,
close/reopen, existing exact-row refresh and a synthetic pre-send failure.
Only the parked-tunnel rearm hook is stubbed to prevent synthetic test logins
from starting services. Explicit client pins and DEFAULT post-login selection
are distinct assertions; an explicit-pin pass does not prove default selection.
The latter is a required negative control before this change is publishable.

Existing OAuth row serialization at this source head is JSON, not an encrypted
row/AAD codec. Reopen evidence must not be described as encrypted-row proof.
No real credentials, keychain interaction, network enrollment, services, models,
proactive arming or production flags belong in this validation.

Actual baseline reproduction used main `44c9024588` through public
`AuthStore.upsert_credential` and real temporary SQLite: both same-org account
retention and unknown-legacy preservation failed. Both assertions pass on the
changed implementation. The required default-selection negative initially
failed separately; after the quarantine correction, the two focused test
modules passed 24 cases, including reopen and synthetic refresh recovery.
Scoped Black/isort, flake8 and pyright passed on all three changed Python files.
Commands, frozen head and final resource figures belong in the PR evidence.
This remains a prerequisite, not an independently reviewed completed onramp. It does not claim fresh/upgrade/later-join onboarding,
account-isolated CoS history, shared-loop transport, service health or executor
idempotency. Those require separately reviewed implementation and real-path QA.
