# Compaction: carry the live task on the archive tail

Status: implemented (`compaction/snapcompact.py`, `compaction/cutpoint.py`,
`Session._dense_tail_cut`). Snapcompact strategy only.

## Problem

A pass that fires at the 400k trigger often left the context far above what
"compacted" suggests. Over 1,858 real passes that started at or above 300k
(21 days of one operator's sessions), the provider-reported context of the first
request after the pass was 97k / 149k / 269k at p10 / p50 / p90, and 140k at
p10 on Anthropic models. A pass that lands at 270k is back at the
trigger after 130k of work, and each pass re-sends the whole rebuilt context
uncached, so shallow passes are both frequent and expensive.

Decomposing 61 of those passes by replaying the journal, and then re-running 25
real compaction points through `Session.compact_now`, gave three causes:

| Cause | Size | Where |
|---|---|---|
| The task floor keeps the live task raw, up to a 100k cap | 42k-100k raw tokens kept in 16 of 25 passes | `task_boundary_floor` |
| The archive's head edge re-sends the session's oldest three pages | ~10.8k tokens, 99.9% identical to the previous pass's head | `compact_to_archive` |
| A resumed subagent's launch brief is preserved once per resume | up to the 100k preserved-turn cap; 8 of the 11 sessions whose block exceeded 20k | `extract_preserved_user_turns` |

The local token ruler also under-reads a raw window relative to the provider
(1.4-1.7x on an Anthropic model, measured per session), which is why 134k local
shows up as 250k+. That is not addressed here.

## What was tried and rejected

The obvious fix is a narrower raw window. It was measured on the same 25 points,
with a retention probe: for every assistant step, assistant message, tool-result
first line and long-result last line that today's behaviour keeps raw, is it
still present in the rebuilt context?

| Policy (task span over the cap, n=6) | After, local | Steps | Assistant text | Result heads | Long-result ends |
|---|---|---|---|---|---|
| Today (floor, cap 100k) | 134k | 100% | 100% | 100% | 99% |
| Floor capped at 50k | 85k | 77% | 91% | 77% | 47% |
| Plain recency | 57k | 49% | 67% | 55% | 20% |
| This change | 87k | 100% | 100% | 100% | 98% |

A narrower window compacts more by forgetting the first half of the task the
agent is still executing, which is the failure the floor was added to prevent.
The archive does not absorb what the window drops: its middle is a fixed number
of pages, dropped oldest-first.

## Design

1. **Dense tail.** When the task floor (not an advisor hint) widened the raw
   window and the strategy is snapcompact, the pass cuts at plain recency
   instead. The span the floor would have kept raw is handed to
   `compact_to_archive(protected_tail_messages=N)`, which widens the archive's
   verbatim tail edge to hold its whole serialization. It is never paged and
   never dropped. Guard: `dense_tail_saves` requires the serialized span to be
   smaller than the raw span on the archive's own ruler, so the change of
   carrier cannot enlarge a pass; otherwise the floor's raw window is kept.
2. **Tool results keep their end.** The archive serializer cut a long result
   head-only at 2,000 characters. It now keeps the first 1,500 and the last 500
   (`_truncate_middle`), same budget. The end is where the verdict is: exit
   summary, test totals, the exception line, the stderr block. Without this,
   24% of long-result ends in a dense-carried task survived; with it, 98%.
3. **One-page head edge.** `HEAD_EDGE_FRAMES = 1`; the tail keeps
   `HQ_EDGE_FRAMES`. The two pages no longer held verbatim join the middle,
   where they are budgeted like all other old history and mined into the carry
   when their page is dropped. This is a real removal of old text from the
   context, not a deduplication: only 4% of the head was present elsewhere.
4. **Launch briefs carried once.** `collapse_relaunch_briefs` keeps the newest
   `subagent-launch:` turn verbatim and reduces each older one to the lines the
   newest does not contain. Keyed on the id the harness mints, never on text.
   On the 8 affected real blocks: 62-86% smaller, zero lines lost from the
   block, and the cap no longer evicts distinct resume notes to keep copies.

## What is given up

Inside the moved span only: the middle of a tool result longer than 2,000
characters, tool-call arguments past 500 characters, results flagged useless,
and provider reasoning blocks. Everything else in the span is verbatim. Two
pages of the session's oldest archived text leave the verbatim head. The
operator's own turns, the carried tier, and the newest `keep_recent_tokens` of
raw history are unchanged.

## Measured result

25 real compaction points, real `compact_now`, private transcript copies. The
baseline is this tree with all four levers switched off, which reproduces the
unmodified tree's results on 24 of 24 shared cases; the new arm is
deterministic across runs (25 of 25).

| Group by live-task span | n | After, today | After, new | Change |
|---|---|---|---|---|
| Recency window only (span <= 20k) | 9 | 57.3k | 50.8k | -11% |
| Task 20-50k | 3 | 82.3k | 54.9k | -33% |
| Task 50-100k | 7 | 108.0k | 69.5k | -36% |
| Task over the 100k cap | 6 | 134.4k | 86.8k | -35% |
| All | 25 | 98.8k | 63.2k | -36% |

Worst case 154.7k -> 108.9k; no case larger; per-case reduction 11-53%.
Operator turns verbatim after the pass: 151 of 151 in both arms. CPU per pass:
10.9 s vs 10.8 s summed over the 25 passes (median 0.43 s vs 0.44 s).

Retention of what today keeps raw, all 25 points: steps 1345/1345, assistant
messages 418/418, result heads 99%, long-result ends 97% (98% today).

Limits of this evidence: sizes are on the local ruler, so the provider-side
figure is an estimate until real passes are observed after release; the 25
points are stratified, so the per-group rows transfer and the overall median is
not a fleet forecast; the probe tests line-start presence, not usability, and
cannot see inside pages replayed as image frames.

## Not changed

Context-full (text-only models) keeps the floor's raw window: it has no
verbatim tail, and an LLM summary would paraphrase the task. An
advisor-widened window stays raw. The mint site that re-stamps the launch
brief on resume (`harness/subagent.py`) is untouched; the collapse is the
compaction-side bound that holds whatever the mint site does.
