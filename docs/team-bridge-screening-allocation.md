# Team Bridge screening allocation

The durable screening ledger has two independent hourly ceilings: **40 logical
calls** and **40 actual HTTP requests** per receiving account at a canonical hub.
Endpoint, tenant, credential and enrollment changes do not create another budget.
Reservations are spend, even if the request's outcome is uncertain. A cached
primary-model answer spends neither bucket; every actual retry or fallback send
spends the HTTP bucket.

Inside each ceiling, one verified sending account may spend at most **20 units**.
Human/agent kind changes do not split that account. Existing nested allocation
reserves remain: ambient work can spend 30, all non-human-addressed work together
can spend 35, and only human-addressed work can reach the final five units.
These are intersecting ceilings, not additional capacity.

## Provenance and batching

Pair allocation uses the server's stored `message.sender` participant, not the
top-level display sender. Conversation allocation uses `sender_account_id` and
`sender_kind` from the existing gated classification body read. No history,
display label, recipient-binding check or durable feed ref substitutes for that
read. The account and kind are registered in the immutable input proof and job;
the ledger rechecks them before every HTTP dispatch, including each fallback.

A coalesced call counts once, irrespective of message count. Its logical rows
share a batch id. Each HTTP send likewise has a distinct shared unit id and a
row for every participating sender. The dispatch fence checks all retained
batch proofs, including non-carrier messages; picking the first message never
lets the other senders ride free.

## Old servers and old ledger rows

Missing identity is supported, but cannot be free capacity: an unattributed unit
counts against **every sender's allowance**. New unattributed work sees **all
account spend** against its 20-unit ceiling. Thus an old server can screen at
most 20 unknown calls per bucket per hour, and a new sender's available share
is reduced by unknown historical spend until it ages out. A mixed batch with
unknown provenance uses the same conservative rule. This intentional capacity
loss prevents identity stripping from buying a new bucket. Migration adds nullable
allocation columns without deleting, refunding or reassigning existing charges.

Deploy the server's additive identity response first to minimize that capacity
loss. Old clients ignore the fields; old servers remain safe with new clients.
No configuration change or flag flip is needed.

## Input filtering versus outages

Only the production `ProviderError` request-kind input-refusal predicate marks a
provider input-filter refusal. That marker is retained across wire sanitization;
provider prose is not stored. It neither increments nor clears the outage
breaker, and it is not cached as an answered verdict. Authentication errors,
rate limits and service failures keep their existing failure semantics.
