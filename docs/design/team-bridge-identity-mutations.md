# Team Bridge: explicit identity mutation boundary

This is a narrow runtime UX exception to Team Bridge RFC §5.1's status cache
floor, not a change to its ordinary read discipline. A warm status answer lasts
at least 300 seconds; reads at app start, sign-in, focus and surface-open still
coalesce per account and never reconcile from a cached answer. A null
`agent_name` before that boundary is not evidence that publishing failed.

## Explicit desktop settings writes

After a successful `PATCH /v1/settings/{key}` or
`POST /v1/settings/{key}/reset`, only an **effective bridge identity change** to
`team_bridge.session` or `aida.name` permits immediate reconciliation. Effective
identity is the normalized designation plus the canonical CoS display name;
unchanged edits, unrelated settings and failed persistence do not fetch status.
Shared settings writers, config watchers, CLI/TUI writes and timers do not gain
network work. The existing explicit bridge endpoint settings PUT already carries
the canonical name and does not need another status read.

The async route releases the settings write lock before awaiting one bounded,
best-effort reconciliation. It uses the existing account-pinned status reader and
its per-account lock, not a separate client or an invalidate-all/cache-read hook.
Mutation revisions coalesce pending edits onto the latest persisted values; an
edit arriving during an older read cannot be mistaken for a completed revision.
Each fresh reconciliation uses one status read and at most one guarded name PUT
(subject to the transport's existing single auth-refresh retry). No polling,
delayed retries, enrollment, auto-designation or new consent are introduced.

All existing publisher guards remain: supported status, eligible selected tenant,
active endpoint, `agent_name` capability key, committed local owner for the same
account and endpoint, and a designation that identifies the recorded Aida
session. Clearing or designating a different session never publishes its title.
Names come from the canonical resolver, never the request's raw text.

## Honest completion and failure

The response acknowledges the persisted local setting, **not a confirmed remote
identity echo**. The status answer was fetched before any name PUT and remains
unchanged in the cache; there is no fabricated echo or second confirmation GET.
A timeout, refusal or unavailable bridge does not turn a saved setting into an
HTTP write failure. Diagnostics contain only safe failure categories, never
credentials, settings values, names or transport exception prose. Later ordinary
fresh boundaries retain the existing retry behavior. This work does not prove
recipient wake/delivery or change directory counts/self-listing behavior.
