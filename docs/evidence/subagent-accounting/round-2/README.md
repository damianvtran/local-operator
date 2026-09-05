# Round 2: authoritative cold ledger and both table-priced models

F1: a child's roster sidecar can be committed after the last parent frontend
checkpoint, or before that checkpoint/transcript exists. The cold facade now
reads the supported sidecar once, overlays restored rows, and replaces its
canonical subagent total from the complete validated accounting checkpoint.
Only persisted receipts/estimates are priced here; no discovery. Malformed or
legacy accounting leaves the checkpoint fallback; a valid empty ledger replaces
it. Retained rows are not summed, so swept-only money survives.

Independent QA ran real billed background children and ownerless RemoteSession,
then reattached the same facade to a real owner. Actual before/after:

| State | Before cold total | After cold / rendered / bound total |
| --- | --- | --- |
| stale parent checkpoint .125, sidecar .25 | .125 | .25 / .25 / .25 |
| swept child rows, lifetime .25 | .125 | .25 / .25 / .25 |
| no frontend checkpoint, sidecar .25 | None | .25 / .25 / .25 |

The automated regression matrix covers checkpoint present/absent, swept/visible
rows, valid/empty/malformed/legacy ledger (16 combinations), through actual
`RemoteSession.cold`. Cold/remote plus model regression suite: 45 passed.

## Explicit Fable and Astra regression

Both observed models are TABLE-PRICED, not provider-USD receipt successes. Tests
use the exact identities anthropic/claude-fable-5-1 and openai/gpt-6-astra. Fable's
input/cache-read/cache-write buckets are separate. Astra's cache-read tokens are
a subset of input and must be deducted from uncached input, not charged twice.

Coder tests use controlled fixture rates, explicitly not vendor-price claims,
and observed token/cache shapes: Fable input116/output52767/read4282533/write164387;
Astra input5935061/output7159/read4740864. Both pass shared `cost_for_usage`, call
aggregation, wire persistence, recorded-only offline totals, mixed known/unknown,
and an additional provider receipt control. No model-specific production branch
was added.

Independent QA also exercised actual background runtime/remote/OperatorApp
with each model's table-priced usage and provider-cache shape, in warm and offline
processes. Recorded examples: Fable 2.69306415; Astra 5.8679652 (QA used its later
Astra usage sample). Row, canonical total and rendered footer match. Those figures
are fixture accounting outputs, not verified real vendor invoices.

PNG frames here were rendered from QA's real styled OperatorApp SVG captures
and viewed. They show Fable `$2.69`, Astra `$5.87`, and the stale-checkpoint fix
`$0.250`. The landing header reports v0.46.12 because the worktree venv's package
metadata was installed at initial setup; editable runtime source is the reviewed
worktree, not an installed release. No screenshot was modified to hide this.

No visual layout, navigation or control changes. Existing design round remains
subject to reviewer confirmation against these recaptured state frames. All
application runs use isolated HOME/LOCAL_OPERATOR_CONFIG_DIR; no live operator
session or credential was controlled.
