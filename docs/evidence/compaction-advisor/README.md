# Evidence: speculative compaction advisor (BETA)

Captured 2026-08-28 on macOS 25.6.0 against the real session transcript at
`~/.local-operator/sessions/bda7b76d34e0/transcript.jsonl`, and (for the cache
arm) live Anthropic `claude-opus-4-8` calls on the configured OAuth credential.

| File | What it shows |
|---|---|
| `cache-measurement.txt` | Live provider cache counters for the advisor request shape, including the placement finding that changed the code |
| `replay-severance.txt` | Counterfactual replay: passes, severance rate, and replay cost for recency vs task-aware preserve at three triggers |

Both are the verbatim stdout of committed scripts, so they are re-derivable:

```sh
.venv/bin/python scripts/measure_advisor_cache.py            # needs an Anthropic credential
.venv/bin/python scripts/bench_advisor_replay.py <transcript.jsonl>
```

## What the cache measurement proves

The advisor reads the WHOLE conversation on every call, which is only
affordable if that read is a cache HIT. That claim is the feature's entire
economic case, and it was asserted before it was measured.

```
2 ADVISOR isolated=False : cache_read=14024 cache_write=568   cache_hit=96.1%
6 ADVISOR system-block   : cache_read=0     cache_write=14590 cache_hit=0.0%
```

**The shipped shape reads 96.1% from cache**, so the advisor costs roughly 2.6%
of the bill rather than the 25.6% a cold prefix would cost.

**Arm 6 changed the implementation.** Carrying the advisor's instructions as an
extra *system block* — the obvious placement, and what the original design
specified — diverges the cache prefix at a point ahead of the messages and pays
a full cache write, 0% hit. The instructions therefore ride inside the appended
user turn, and the request is strictly append-only relative to the turn's
prefix. See `ADVISOR_SYSTEM_PROMPT` in `local_operator/compaction/advisor.py`.

**Arm 4 is a caveat, not a contradiction.** On Anthropic `isolated=True` also
reads 100% from cache, because that provider keys caching on prefix *content*
rather than on `prompt_cache_key`. Isolation is still declined, since the key
does govern the OpenAI-compatible wire and this path must be correct on every
provider a session may run on — but on Anthropic specifically, the isolation
trade-off costs nothing.

## What the replay proves

The headline is **severance** (the fraction of passes whose cut lands inside
the task in flight), not raw token savings. Compacting constantly saves tokens;
the cost of doing so is invisible in a token count and very visible to a user
whose in-progress work was summarized away.

```
  trigger  rule        passes  severed  sever%  replay Mtok  vs 600k
  600,000  recency          4        3     75%        2,249    0.0%
  600,000  task-aware       4        0      0%        2,428   -7.9%
  300,000  recency          9        5     56%        1,296   42.4%
  300,000  task-aware      10        1     10%        1,373   39.0%
```

At the shipped 600k trigger, task-awareness costs ~8% more replay and takes
severance from 75% to 0%. At a 300k trigger it costs ~3 points of savings and
takes severance from 56% to 10%.

Active-task spans at the four 600k passes were 32.3k / 0.9k / 101.5k / 63.1k
tokens (p50 47.7k) against a `keep_recent_tokens` of 20k, which is the direct
measurement of why a recency-shaped cut severs a task-shaped session.

### Note on drift

This transcript is a LIVE session and keeps growing: it read 7,663 entries when
the PR was opened and 8,005 here, so pass counts and cost percentages move
slightly between runs (the 600k task-aware cost read -4.2% earlier, -7.9% now).
The severance comparison — 75% to 0% at 600k, and task-aware never worse than
recency at any trigger — has been stable across every run. Re-running against a
grown transcript is expected to shift the cost column, not the conclusion.
