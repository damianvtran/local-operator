# Fixed-source live task results

All **20 sessions passed independent artifact checks**, with clean final provider
stops and stable runtime source hashes. Five matched repeats per arm/task ran
on native OpenAI `gpt-5.6-sol` at low effort. The baseline was committed
`765c619bd9c7603241dc0bcdc2773ffbb2e9c9fa`; the candidate runtime was
`1a7bd26b61414cdfdb1efc6ba7ef43879e40ee3a`.

The sample shows no consistent task-time or step regression, and does not
establish a general end-to-end speedup. Candidate latency was lower in three
of five pairs for each task. The complete results, including the slower cells,
are retained in [live-final.json](live-final.json). See
[the method](live-method.md) and the two reproducible benchmark scripts for
scope, isolation, source checks, budgets, and acceptance predicates.

One fixture limitation affects baseline repair repeat 5: its `git status`
discovered an ancestor repository and exposed unrelated workspace filenames in
recorded tool output. No unrelated file contents or credentials appear in that
output. The trial remains in every reported aggregate as an uncontrolled-state
caveat. Future workers initialize a synthetic Git root before fixture seeding;
that correction was verified offline, with no replacement live run. Scoped
synthetic inputs and prompt instructions are not a filesystem sandbox.

| Metric | Repair baseline | Repair candidate | Aggregate baseline | Aggregate candidate |
| --- | ---: | ---: | ---: | ---: |
| Accepted sessions | 5/5 | 5/5 | 5/5 | 5/5 |
| Mean elapsed seconds | 40.661 | 39.695 | 35.573 | 36.261 |
| Median elapsed seconds | 40.880 | 37.375 | 34.970 | 32.484 |
| Mean model requests | 6.2 | 6.2 | 4.2 | 4.0 |
| Mean tool calls | 7.8 | 6.8 | 4.8 | 4.8 |
| Mean input tokens | 61,511 | 60,153 | 47,375 | 47,455 |
| Mean output tokens, including reasoning | 1,441 | 1,352 | 1,264 | 1,283 |
| Cache-read fraction of input | 42.0% | 48.0% | 52.7% | 28.6% |
| Mean time outside provider streams, seconds | 0.957 | 0.662 | 1.112 | 0.757 |
| Next-input estimate median absolute error, tokens | 455 | 29 | 568 | 28 |

Repair's observed mean elapsed time decreased 2.4%; aggregation increased 1.9%.
Across both tasks, the candidate made 51 model requests versus 52 for baseline.
The time outside provider streams is a wall-time residual covering harness
preparation, tools, and scheduling; it is not an isolated CPU benchmark. Both
its lower means and the task latency differences remain descriptive observations
with only five samples per group.

The aggregation outlier remains in the means: candidate repeat 1 took 54.135
seconds versus baseline's 32.987. Its final provider request took 23.19 seconds
for 77 output tokens, compared with 3.54 seconds for 105 baseline output tokens.
Both sessions made four provider requests. This locates most of that observed
gap inside a provider stream; it does not prove which source behavior or remote
condition caused the delay.

Extra steps also occurred in both arms. Candidate repair repeat 1 removed a
redundant generated test entrypoint; baseline repair repeat 2 needed seven
requests while the candidate completed in six. Model-authored extra work does
not automatically exonerate the harness: prompt or state changes can change
strategy. The complete paired sample, rather than an individual explanation,
is the basis for the bounded non-regression assessment.

## Cache and context evidence

OpenAI cache results are mixed and do **not** demonstrate a universal cache-rate
gain. The aggregation imbalance is already visible before native reasoning is
replayed: all five baseline initial requests hit an 8,320-token prefix, while
only one of five candidate initial requests hit its 8,448-token prefix. That
accounts for approximately 58% of the aggregate cached-token gap. System and
tool-schema hashes stayed identical within each arm, but differed between arms;
prior remote prefix exposure and request routing therefore remain confounds.
Home catalogue/cache state was intentionally shared and these were not cold
cache experiments. Cached input is a subset of input, not an additional bucket.

The context estimator has more direct evidence. Comparing each continuation's
unscaled admission estimate with its own authoritative next input, maximum
absolute error fell from 1,955 to 69 tokens for repair and from 4,514 to 73 for
aggregation. There were 26 measured continuation requests per arm for repair,
16 for baseline aggregation and 15 for candidate aggregation. Native-bound
candidate hints include reported reasoning only when its encrypted item is
actually replayed under the selected protocol/account. They remain estimates:
small tokenization and wire-overhead differences persist. The protocol and
boundary regression tests cover large reasoning counts, new user boundaries,
credential/protocol changes, partial outputs, and resumed transcripts separately.

## Earlier exploratory runs

An earlier two-repeat sample used evolving candidate source and showed slower
candidate means. One candidate aggregation cell changed semantic source during
execution. Those exploratory results were not pooled with this matrix. Their
raw local artifacts and investigation notes remain available for audit, including
the explicit correction to an earlier claim that every exploratory cell was
source-stable. A separate high-effort diagnostic exposed the missing reasoning
contribution that was fixed before the committed matrix.

This evidence supports correctness and shows no consistent regression in this
sample of two tasks. It does not establish performance across every model, long conversation,
subagent workload, or cache state. Deterministic structural benchmarks and the
separate provider cache experiment support those specific mechanisms.
