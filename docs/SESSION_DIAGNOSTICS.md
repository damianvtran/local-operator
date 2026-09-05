# Current-session diagnostics

`/session` opens a read-only, scrollable snapshot of the current session. It
accepts no arguments, does not call a model, and does not append a prompt or a
receipt to conversation history. Escape or `q` closes the view and returns focus
to the composer. Close and reopen to refresh; arrow, page, Home and End keys
navigate the report.

## Scope

The usage source is the shared local analytics SQLite ledger, filtered by the
**exact current session ID**, not a transcript traversal or an ID prefix. A
resumed session includes that ID's retained records. Child sessions have their
own IDs and are excluded. A fork's copied conversation history does not become
new ledger spend. Older rows can be pruned by analytics retention, and in-flight
requests or asynchronous recorder writes may not have reached the ledger yet.
The report is therefore recorded, retained usage, not a lifetime invoice.

An absent database or an exact ID with no rows is an empty report. An unreadable,
corrupt or incompatible ledger is explicitly unavailable. Opening diagnostics
does not create or migrate a database. Legacy optional fields are projected as
unknown (timings, request IDs, purpose/outcome, usage reporting) or unattributed
(estimated components), rather than fabricated observations.

## Accounting

- Input context is the normalized full input count, including cached context.
  Total tokens are input context plus output. Reasoning is a **subset** of
  output, never an additional charge or extra token total.
- Cache hit rate is cache-read tokens divided by full input context. Cache
  writes are shown separately and are not added on top of normalized context.
- Input components are the existing character-weighted estimates recorded by
  analytics. Tool inventory, schemas, results and conversation are distinct
  components. Their token split is estimated, not provider itemization.
- Combined costs are summed exactly in integer micro-USD, grouped by both
  provider and model ID. Costs are provider-reported where available, otherwise
  list-price estimates. The stored ledger does not retain provenance per row,
  so the view does not claim to distinguish those sources. Unknown costs are
  not free; a `+` marks a partial known sum, and a known zero remains `$0.0000`.
- Separate input/output/tool dollar amounts cannot be recovered from this
  ledger. In particular, assigning dollar shares by token ratio would confuse
  discounted cached input with full-price output and is deliberately not done.
- Timings are logical-request observations, with a separate sample count for
  each mean/range. Unknown historical timings are excluded, not zero-filled.
  Duration includes retries and consumer backpressure; first output is the
  first text or tool-call delta, not a provider-internal compute measurement. Recent rows (12 by default, hard cap
  50 in the API) identify logical requests, not retry attempts.

## Runtime snapshot and implementation boundary

The view captures name, full session ID, selected/effective model identity and
streaming state before yielding to the ledger worker. When the canonical
frontend snapshot is available it also reads its context occupancy, measurement
limit and turn generation. Reduced session facades can omit those fields; the
view reports unavailable rather than reading private prompts/context or adding
new protocol requirements. Compaction count is not currently exposed and is
shown as unavailable.

`AnalyticsStore.session_report` uses one short-lived, read-only connection and
an explicit SQLite read transaction. Totals, grouping, timings and bounded recent
rows all observe the same snapshot even while a WAL writer commits new calls.
The UI runs that read off the event loop. Before displaying the result it checks
the captured session object, ID and (when exposed) mirrored owner epoch, so
`/new` or `/resume` cannot surface a stale report over another session, including
an owner replacement behind the same remote facade.

Remote terminal frontends use the same shared local analytics database and
mirrored session identity/runtime scalars. The command is frontend-local, like
`/analytics`; no analytics RPC or transport schema is added. Slash autocomplete,
help and the daemon's mirrored command registry discover it from the standard
command table.

The report does not render prompts, tool schemas/payloads/results, credentials,
provider URLs or raw exception text. Its only tool-related data is the existing
aggregate estimated token count.
