# Screen bench: protocol, corpora, templates, results

The screen-only bench itself is `scripts/eval_bridge_screen.py`; its usage and the
rates it reports are described in
[`../README.md#screen-only-bench-the-inbound-screener-as-a-classifier`](../README.md#screen-only-bench-the-inbound-screener-as-a-classifier).

- [`PROTOCOL.md`](PROTOCOL.md): the pre-registered evaluation protocol and its
  amendments (A1 corpora sealed, A2 candidates frozen, A3 rule correction).
- [`corpus/`](corpus/): the sealed `adv`, `legit` and `replay` corpora.
- [`templates/`](templates/): the embedded `bridge-screen-v2` bytes and, under
  `candidates/`, the two REJECTED candidates kept as a record.
- [`results/2026-10-31/`](results/2026-10-31/): the runs, summaries and write-ups.

## What a false block means over time

A legitimate message that was not delivered is a false block. Read
`false_hold_rate` and `false_bounce_rate` separately; a disposition alone does not
say whether time or another retry can change the outcome.

### Retrospective clarification — 2026-10-31

The answer-hold cache is **shipped**, not planned:
[PR #3220](https://github.com/damianvtran/local-operator/pull/3220) landed in
[v0.94.2](https://github.com/damianvtran/local-operator/releases/tag/v0.94.2).
This is a policy/product clarification of the current implementation, not a change
to the sealed runs or their adoption decisions.

- **Cached model-answer holds do not expire with time or retries.** The cacheable
  codes are `model_hold`, `model_refused`, `reply_truncated`, `parse_failed` and
  `bounce_without_category`. Only an answer from the requested primary model is
  remembered; a fallback model's answer decides that pass but remains uncached.
  A cache hit reuses the hold instead of making another model call. A displayed
  `retry_at` is retry-scheduling metadata, not a promise that the cached hold will
  release when that time arrives.
- **Reuse is bounded by the screening tuple and live authority, not a timer.** The
  lookup binds owner, body hash, template hash, requested model, admission revision
  and envelope digest; the latter binds the screened sender/display/organization,
  message type, request and attachment set. A changed tuple is a cache miss. The
  read also checks the live published template/model/revision, policy epoch,
  enrollment generation and screening engine version. Template/policy republish,
  engine-version changes or re-enrollment can therefore invalidate an answer; mere
  elapsed time cannot. Invalidated rows need not be physically deleted to stop
  being reusable.
- **No TTL or manual per-message release is promised.** The operator decision was
  relayed by author session `3e304d8d39bf` as the answer to ask `a-d9c6`: keep cached
  model-answer holds until republish, with no TTL/manual-release path. The original
  [#3220 deferred-findings note](https://github.com/damianvtran/local-operator/pull/3220#issuecomment-6541276533)
  records why expiry would reopen the re-draw defect and why a per-message release
  would need a separate product/surface decision. The relay settles that product
  choice; it does not remove the implemented invalidation checks above.
- **Not all holds are cached or permanent.** Provider/route failures remain
  uncached and retryable, so an outage can recover. Other authority/state holds
  must be re-derived rather than replayed as model answers. In particular, a
  `bounce` is not universally a terminal transport rejection: for an already
  confirmed upstream read, both fresh and cached rejects take the
  `local_quarantine` hold path rather than creating a conflicting transport
  reject. The answer-hold cache is not itself admission authority.

These details follow the cache classification and primary-model guard in
[`screening.py`](../../../local_operator/team_bridge/screening.py) and the tuple,
policy/epoch/generation/engine checks in
[`store.py`](../../../local_operator/team_bridge/store.py).

### Reading the pre-cache results

The sealed 2026-10-31 study predates this cache. At that time a model-answer hold
could be re-driven through a fresh model draw; it might later deliver a legitimate
message, but could also eventually admit an adversarial one. The study's split
false-block interpretation included that re-draw exposure. Keep that historical
interpretation when reading the results; do not treat those repeated bench trials
as evidence that a cached production hold now expires or re-draws on retry.
