# Conversation mutation owner receipt

`team_bridge_conversation_owner_guard: 1` is an additive desktop-runtime
capability. It closes a stale-action race: a conversation displayed for account
A can still have an action queued when the desktop signs in as B. Response
filtering is too late for a write. The runtime compares the captured account
with the exact OAuth access snapshot used to dispatch, before contacting the
messaging service; it does not guard and then resolve a second credential.

## Renderer contract

A new renderer must require the capability before enabling these mutations and
send **both** `expected_account_id` and `tenant_id`. Capture them from the data
owning the action, not from the latest global account/organization picker. An
idempotent send retry retains the original action's receipt and target tenant.
The runtime accepts omission for legacy clients; that compatibility does not
make old clients protected from stale-action replay.

| Desktop operation | Receipt location |
| --- | --- |
| POST conversations (create) | JSON object |
| POST conversations/{id}/messages (send/retry) | JSON object |
| POST conversations/{id}/join | JSON object |
| POST conversations/{id}/read (cursor, muted, notify) | JSON object |
| POST conversations/{id}/members/{account}/consent | JSON object |
| POST conversations/{id}/messages/{message}/admit | JSON object |
| POST conversations/{id}/archive or unarchive | Query |
| POST conversations/{id}/leave | Query |
| POST conversations/{id}/members (add members) | Query; body remains a member list |
| DELETE conversations/{id}/members/{account} | Query |

Every path above is under `/v1/desktop/team-bridge/`. A guarded request needs
an explicit nonempty target `tenant_id`. The optional receipt is shared by the
six object-body models and by validation of the bodyless/list-body query.
A mismatched owner returns HTTP 409 with `account_mismatch` and makes no
messaging-service request. An invalid body is rejected by model validation;
an invalid query receipt returns HTTP 422 `invalid_request`.

`expected_account_id` is local anti-replay metadata, never an upstream field.
The target tenant is forwarded to the messaging service (body for object
writes and member invitations; query for archive/unarchive, leave and remove).
It is **not** compared with OAuth `org_id`: the same person may legitimately
write to another organization. The service still authorizes target membership
and every operation's existing permissions.

## Client and refresh boundary

`_guarded_access(expected_account_id, *, access=None)` preserves a supplied
`OAuthAccess` object and compares that same object. `_send` performs the guard
before dispatch, including when an internal caller supplies access. Without a
receipt, supplied snapshots and legacy credential resolution retain their
previous behavior. The 401 refresh path stays pinned to the original row and
account; a replacement account cannot inherit a retry.

## Exact coverage and exclusions

This capability covers create, send/retry, join, read/notification policy,
consent, admit, archive/unarchive, member add/remove and leave. It does not
claim to protect the existing GET/list/body read surfaces. Threads and reactions
are absent from this base runtime. Feature PRs #2757 and #2760 must independently
adopt the same captured-owner and explicit-target contract for their mutations
before claiming compatible coverage; the capability is not an automatic guard
on a future endpoint.

## Validation boundaries

`tests/unit/server/test_conversation_owner_guard.py` drives the actual ASGI
router, desktop authentication, request models, client and disposable SQLite
AuthStore through a real account logout/login switch. The messaging service is
a controlled HTTP transport with a recorded mutation ledger. It checks every
operation above, omitted legacy receipts, malformed tenants, independent
desktop/upstream denial, multi-organization acceptance, retry and snapshot pins.
This is not a claim of renderer validation or a deployed-service integration;
those remain separate evidence. A baseline replay of the archive-query case
must fail by dispatching the stale action, rather than merely failing to import
or construct the test fixture.
