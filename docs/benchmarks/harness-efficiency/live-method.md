# Live task benchmark method

The benchmark runs real `Session` tool loops with native OpenAI authentication,
`gpt-5.6-sol`, fixed low effort, the same synthetic tasks and seeded input data.
Only `read`, `write`, `edit`, `bash`, `eval`, `glob`, and `grep` are available.
Each run receives a fresh task directory, config directory, and transcript.
There are no external MCP servers or real user tasks in these sessions.

The native auth database stays in its original location and is opened through
`AuthStore`; credentials are neither copied into the benchmark nor printed.
The home-based model catalogue/cache is intentionally shared. These are **not
cold-cache runs**. Model resolution and session construction are outside the
reported `session.prompt` elapsed time. Remote provider cache contents, load,
and sampling are uncontrolled, and there is no model sampling seed parameter.
Fixed fixture seeds make task inputs comparable, not model behavior deterministic.

The worker requires an exact committed runtime SHA and rejects uncommitted
runtime changes. Each checkout must own its own real editable venv. Workers run
outside both repositories and verify that the package imports from the intended
checkout. Runtime AST/content hashes before and after each run detect source
changes during execution; unstable cells remain in the raw results but are
excluded from the matrix's stable-run aggregates.

```sh
python3 scripts/bench_live_matrix.py \
  --baseline ~/workspace/repos/lo-baseline \
  --candidate ~/workspace/repos/lo-candidate \
  --baseline-sha <full-baseline-commit> \
  --candidate-sha <full-candidate-commit> \
  --repeats 5 --output /tmp/lo-live-results
```

Five repeats across two tasks and two arms produce 20 sessions. The arm order
alternates within each pair. Each session has a 14-request budget and a
240-second turn timeout, plus bounded disposal and a 285-second process guard.
Retries, usage-based account selection, and fallback are disabled for matching.
The worker requests an 8192-token output cap; the native OAuth endpoint may
remove that unsupported field, so the call/time guards bound the actual trial.

- **Repair:** fix invoice deduplication, filtering, decimal rounding, currency
  grouping, and one-pass iterators; add tests and run them. A separate verifier
  generates 240 seeded rows and checks results, iterator behavior, immutability,
  and empty input independently of the model's editable tests.
- **Aggregate:** join 72 customers with 640 seeded events, filter eligibility,
  produce sorted exact JSON fields, and verify independently. The external
  verifier compares the whole output and original input hashes.

Each result includes acceptance, failures, actual provider calls, token/cache
counts, request hints, tool events, timing, and source/prompt/schema hashes.
OpenAI cached input is a subset of total input, so cache fraction is
`sum(cache_read_tokens) / sum(input_tokens)`. Output tokens already include
reasoning tokens; those counters must not be added together. Native-bound
context hints are captured after protocol/account selection on the candidate.
Synthetic transcripts and generated files stay in the chosen output directory
for investigating extra steps or failed acceptance checks.

Report all runs and task-specific variation. A small stochastic sample can
demonstrate task completion and reveal overhead, but does not by itself establish
that code caused a latency or cache-rate difference. Protocol fixtures and
deterministic structural benchmarks provide separate evidence for specific
mechanisms. In particular, retaining reasoning may add billed input while
preserving continuation state; that is a protocol correctness improvement and
must not be presented as an automatic token reduction.
