# Evidence: speculative compaction advisor (BETA)

Captured 2026-08-28 on macOS 25.6.0 against the real session transcript at
`~/.local-operator/sessions/bda7b76d34e0/transcript.jsonl`, and (for the cache
arm) live Anthropic `claude-opus-4-8` calls on the configured OAuth credential.

| File | What it shows |
|---|---|
| `cache-measurement.txt` | Live provider cache counters for the advisor request shape, including the placement finding that changed the code |
| `replay-severance.txt` | Counterfactual replay: passes, severance rate, and replay cost for recency vs task-aware preserve at three triggers |
| `token-benchmark.txt` | Cache-aware token AND dollar accounting, advisor-off vs advisor-on, swept over cadence / floor / trigger / accuracy / cache-hit rate |
| `cache-integrity.txt` | 15 checks that nothing except a real compaction pass breaks the prompt cache |
| `aside-tool-choice-measurement.txt` | Live re-measurement of the aside/advisor shape at ~37k tokens: `tool_choice: none` vs `auto` reads the prefix equally well, and the fleet's head-only cache events are better explained by TTL expiry (2026-09-01) |

All four are the verbatim stdout of committed scripts, so they are
re-derivable without a paid run (only the first needs a credential):

```sh
.venv/bin/python scripts/measure_advisor_cache.py            # needs an Anthropic credential
.venv/bin/python scripts/bench_advisor_replay.py <transcript.jsonl>
.venv/bin/python scripts/bench_advisor_tokens.py <transcript.jsonl>
.venv/bin/python scripts/check_advisor_cache_integrity.py
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

## What the token benchmark proves, and where it does NOT

`bench_advisor_replay.py`'s `replay Mtok` column is a CONTEXT-token count and
must not be read as money: it prices a cache read the same as a cache write,
and those differ by 12.5x. `bench_advisor_tokens.py` does the cache-aware
accounting, including the term a naive argument omits — a compaction pass
invalidates the prefix, so the next turn re-writes most of what survived.

That penalty is measured from this session's own provider counters rather than
assumed: at each of its passes the following turn re-read only a ~23.4k
system+tools head and re-wrote 80.7–85.4% of the residual (~25.6% of the
pre-pass context). Those constants are re-derived at runtime by `calibrate()`.

Headline, shipped defaults (trigger 600k, cadence 20, floor 200k):

```
strategy      passes  adv calls  prompt Mtok    cost $     vs off
snapcompact        5          0        2,846  1,480.28
  advisor ON      10         39        1,623    869.06     -41.3%
context-full       5          0        2,865  1,528.13
  advisor ON      12         20        1,653    976.39     -36.1%
```

So the feature does reduce net consumption on this session, and the advisor's
own calls are a rounding error (\$7.79 of \$869) because they ride the warm
prefix. **Three caveats keep this from being a blanket endorsement:**

1. **Most of the saving is not unique to the advisor.** A static 300k trigger
   costs \$842.02 against the advisor's \$869.06 — the advisor is 3.2% *worse*
   on tokens alone. What it buys is that the ceiling stays at 600k for the
   turns that need it (peak context 457k vs 315k), which a static trigger
   cannot express. The token case is a tie; the case rests on severance and
   retained headroom.
2. **An advisor that never finds a boundary is pure cost.** At accuracy 0.0 it
   spends its full 200-call ceiling for +3.2%. That is the honest worst case,
   and it bounds the downside at roughly 3% rather than at nothing.
3. **`context-full` is the weaker regime.** There, each extra pass also buys a
   summarisation call (\$99.83 vs \$42.43 off), so the margin narrows from
   41.3% to 36.1%. All nine passes in this session were `snapcompact`, which
   makes an extra pass cost only its cache re-write.

The saving grows as the cache-hit rate falls (43.2% at 50% hits), because a
smaller context is worth more when every turn re-writes it. Cadence and floor
are nearly flat between n=5 and n=80 and between 100k and 300k, so those knobs
are not where the money is; a floor of 400k degrades to -26.6% by pushing the
advisor's band above where the boundaries are.

## What the cache-integrity check proves

`check_advisor_cache_integrity.py` runs 15 checks, all passing. The load-bearing
ones drive the REAL `Session.advise_compaction` (via the session suite's own
harness) rather than a reconstruction, so a future change that appended the
advisor's question to `_context.messages` fails the check instead of passing it:

- the live history is unchanged by identity and content after a real call, and
  the question exists only on the request tail (append-only against the prefix);
- system blocks are byte-identical, with a control arm proving the rejected
  system-block placement *does* diverge them — so the check can fail;
- `cache_control` markers stay within `MAX_CACHE_BREAKPOINTS` (4) and never
  move LEFT into the shared prefix. They *do* shift right by one, because
  `_message_cache_breakpoints` marks the last message and the second-to-last
  user turn and the advisor appends a user turn. Moving right is safe (reads
  match the longest cached prefix); the live proof is arm 3 of
  `cache-measurement.txt`, an ordinary turn immediately after an advisor call
  measuring `cache_read=14024 cache_write=0`, a 100% hit;
- `prompt_cache_key` survives, which is what `isolated=False` is for;
- off by default is inert: a pre-feature config resolves identically and the
  gate returns `None`.

Wire attribution: the live counters in `cache-measurement.txt` are **Anthropic**
(`claude-opus-4-8`, OAuth). The breakpoint and system-block checks are
Anthropic-wire specific (`AnthropicClient._build_body`). The `prompt_cache_key`
conclusion is the **OpenAI**-wire one — Anthropic keys on prefix content and
ignores the key, which is why `isolated=True` measured 100% there and is still
declined for the wires where it would matter.

### Note on drift

This transcript is a LIVE session and keeps growing: it read 7,663 entries when
the PR was opened, 8,005 for the severance replay and 9,418 for the token
benchmark, so pass counts and cost percentages move slightly between runs (the
600k task-aware cost read -4.2% earlier, -7.9% now; the advisor's net saving
read -42.7% at 9,012 entries and -41.3% at 9,418).
The severance comparison — 75% to 0% at 600k, and task-aware never worse than
recency at any trigger — has been stable across every run. Re-running against a
grown transcript is expected to shift the cost column, not the conclusion.
