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
claim to protect the existing GET/list/body read surfaces. The capability is
not an automatic guard on a future endpoint.

## Threads and reactions: REQUIRED receipts

`team_bridge_threads: 1` and `team_bridge_reactions: 1` advertise the thread
read and the human reaction routes, and each key PROMISES the same atomic
captured-owner guard. Unlike the optional receipt above, these routes REQUIRE it:
no shipped renderer called them before the guard existed (the desktop UI held
both surfaces closed for exactly this reason), so there is no legacy caller to
stay compatible with, and an optional guard regresses the first time a caller
forgets it.

| Desktop operation | Receipt location | Capability |
| --- | --- | --- |
| GET conversations/{id}/messages/{message}/replies | Query | `team_bridge_threads` |
| POST conversations/{id}/messages with `thread_root_id` (reply) | JSON object | `team_bridge_threads` |
| PUT conversations/{id}/messages/{message}/reactions/{emoji} | JSON object | `team_bridge_reactions` |
| DELETE conversations/{id}/messages/{message}/reactions/{emoji} | Query | `team_bridge_reactions` |
| PUT conversations/{id}/agent-notify (owner's own agent wake level) | JSON object | `team_bridge_agent_notify` |

An absent, blank, repeated or malformed `expected_account_id` or `tenant_id`
on the replies read, a threaded send, or either reaction route is refused (422)
before any messaging-service request; a mismatched owner is `409 account_mismatch`,
also with no request. An ordinary top-level send (no `thread_root_id`) keeps the
legacy optional receipt under `team_bridge_conversation_owner_guard`.

One adjacent route is deliberately NOT covered by either key, and callers must
not assume otherwise: `GET conversations/{id}/messages/{message}/body` is the
agent-classification body fetch (body + binding hash), it predates this work,
and it takes no owner receipt. No renderer op reaches it (the desktop contract
has no `team_bridge.messages.body`), so it feeds no human thread surface today;
closing it is follow-up work, not part of this promise.

A second pre-existing hole is named here for the same reason: `POST
/v1/desktop/team-bridge/scheduled` accepts `thread_root_id` (`ScheduledCreateIn`)
and carries **no** `expected_account_id` at all, so a scheduled thread reply is
created as one account and fires against whoever is signed in when it comes due.
Neither capability promises anything about that path; it needs its own change
and is listed so nobody reads `team_bridge_threads` as covering it.

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
