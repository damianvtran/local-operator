# Session efficiency validation

The session changes preserve system-prefix bytes while standing instructions
remain unchanged, publish changing session state at the conversation tail, refresh knowledge on admitted user
boundaries, and batch durable transcript writes off the event loop. Their
publication rules also cover cancellation, atomic file replacement, aside
injection and parent/child communication receipts.

## Measurements

Baseline is committed `5cbea141cca4dde7f6c31cc209aa4ea7d08889f9`, using its own
editable virtual environment. The changed tree uses its own editable environment.
Each JSON artifact records SHA-256 hashes of the six sources used by the probes;
these identify the measured code even while an implementation branch is still
uncommitted. These are local offline observations on macOS, not provider cache
hit rates or fleet task-completion forecasts.

| Probe | Baseline | Changed | What it establishes |
| --- | ---: | ---: | --- |
| Durable 48-message paired flush, median of 7 | 4.55 ms | 1.35 ms | Less local durable-write work |
| fsync calls in each flush | 48 | 1 | One durability barrier per closed batch |
| fsync calls on the event-loop thread | 48 | 0 | Disk stalls cannot block sibling event loops |
| 100 prompt builds over 5,000 transcript rows, median thread CPU of 7 | 109.78 ms | 0.256 ms | Indexed pointers and unchanged-block reuse remove repeated scans/rendering |
| Matching serialized Anthropic prefix after goal change | 11.46% | 99.984% | Conversation prefix remains eligible for reuse |

The first local run showed the same mechanisms with different timings: paired
flush 3.29 → 1.85 ms and prompt preparation 70.75 → 0.250 ms. Wall timings vary
with disk and machine load; the structural assertions are the regression guards.
The wire probe uses the real `AnthropicClient._build_body` and compares tools,
system and messages in cache-hierarchy order. Matching characters do not prove
provider token-cache hits, account routing, expiry or a quality improvement.

Data: [baseline](benchmarks/session-efficiency/baseline.json),
[changed](benchmarks/session-efficiency/changed.json).

Reproduce from the changed checkout, supplying the baseline's own interpreter:

```sh
BASELINE_REPO=~/workspace/repos/lo-harness-baseline
"$BASELINE_REPO/.venv/bin/python" scripts/bench_session_efficiency.py "$BASELINE_REPO"
.venv/bin/python scripts/bench_session_efficiency.py .
```

The script uses temporary transcripts and mock provider streams. It reads no
credentials, calls no live model, and does not alter the stable `lop` runtime.

## Regression coverage

The expanded session/factory/comms set passed 1,204 tests in 72.95 seconds:

```sh
.venv/bin/python -m pytest tests/unit/session tests/unit/test_session_factory.py \
  tests/unit/harness/test_comms.py tests/unit/harness/test_subagent_concurrency.py -n 2 -q
```

Subsequent targeted additions cover cancellation during file replacement and a
reply held until its durable receipt commits. The session efficiency and
transcript edge set passed 34 tests; the final efficiency file passed 9 tests.
The final naming/persistence/efficiency subset passed 56 tests in 5.83 seconds,
including the outside-loop naming coroutine leak fix.
Root release validation runs the complete suite and exact CI lint/type gates.

Meaningful guards include:

- One worker-thread fsync per batch; cancellation keeps the journal lock until
  the write or replacement finishes and memory agrees with disk.
- Goal/date changes reach the same request without rewriting its system blocks;
  resume reuses the frozen prefix and does not duplicate state messages.
- Updated standing instructions start a new durable prefix epoch on live calls
  and resume. Helpers follow the new authority immediately; cache reuse never
  keeps obsolete repository, custom, or packaged instructions active.
- Main calls and helpers share the state-delta renderer and restore complete
  labelled state after compaction; helpers do so without journaling or consuming
  the main turn's publication obligation.
- A state-journal I/O failure prevents sending a request with an obsolete goal.
- Asides are durable before injection. An answered child question is not
  acknowledged before its communication receipt commits.
- An answer already committing survives child detach and parent cancellation.
  Refused or cancelled-before-start saves fail the exact waiter, and late
  commits cannot answer a replacement question.
- Each child uses its own stream handle, retains repository guidance and a
  bounded knowledge directory, and does not copy the parent's conversation.
- A completed answer stays completed after compaction; length-interrupted
  answers retain the existing recovery-band continuation.
- Summarization carries an explicit output budget and separate purpose/effort;
  standalone helpers cannot overwrite ordinary-turn context accounting.
- Slow presentation observers can opt into bounded ordered delivery with text
  coalescing. Overflow disconnects explicitly and requires snapshot resync.

## Audit dispositions and limits

Finding 11 was an architectural risk, not a measured current production stall:
TUI, SSE and headless listeners already use synchronous projection/queue pushes.
The new `subscribe_presentation` contract isolates slow async SDK viewers without
changing ordered critical subscribers. Its test proves isolation and bounded
queues; no existing production latency improvement is claimed for it.

Finding 21 did not justify changing the default compaction threshold, cache TTL,
or ordinary-turn effort. Existing cache ratios were already strong, and no
controlled live task-quality dataset establishes that more aggressive defaults
would preserve successful completion. Those defaults remain unchanged. The
summary-specific budget is the concrete finding 16 fix, separate from that
unproven tuning proposal.

Earlier compaction, a smaller verbatim window, or different TTL/effort policies
should be adopted only after repeated same-model task trials report accepted
completion, total tokens/cost, p50/p95 latency and post-compaction rediscovery.
The offline prefix/CPU probes are intentionally not presented as that evidence.
