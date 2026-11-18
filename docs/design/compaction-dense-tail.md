# Compaction: carry the live task on the archive tail

Status: implemented (`compaction/snapcompact.py`, `compaction/cutpoint.py`,
`Session._dense_tail_cut`). Snapcompact strategy only.

## Problem

A pass that fires at the 400k trigger often left the context far above what
"compacted" suggests. Over 1,858 real passes that started at or above 300k
(21 days of one operator's sessions), the provider-reported context of the first
request after the pass was 97k / 149k / 269k at p10 / p50 / p90, and 140k at
p10 on Anthropic models. A pass that lands at 270k is back at the trigger after
130k of work, and each pass re-sends the whole rebuilt context uncached, so
shallow passes are both frequent and expensive.

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
still present as text in the rebuilt context?

| Policy (task span over the cap, n=6) | After, local | Steps | Assistant text | Result first lines | Long-result last lines |
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
   verbatim tail edge to hold its serialization. It is never paged and never
   dropped. Guard: `dense_tail_saves` requires the serialized span to be
   smaller than the raw span on the archive's own ruler, so the change of
   carrier cannot enlarge a pass; otherwise the floor's raw window is kept.
2. **The same task, one pass later.** A dense pass archives the task's opening
   request with the task, so the next pass's floor has no operator turn to
   measure from. The archive therefore records how much of its text is a task
   the next pass should go on protecting (`Archive.live_tail_chars`), and while
   no new operator turn has arrived the next pass keeps that text on its tail
   together with what it archives itself. The horizon is deliberately the one
   the raw floor has: it protects a task while the task fits its cap, keeps the
   newest cap's worth on the pass where the task outgrows it, and is plain
   recency after that. So the tail is bounded (`Session._dense_tail_cap`, three
   quarters of the floor cap), and a tail the bound had to trim, or one chosen
   by a floor that was already clamped, is carried on that pass and recorded as
   `0`: the pass after it is an ordinary recency pass. A new operator turn
   gives the floor an anchor again and releases the earlier task.
3. **Tool results keep their end.** The archive serializer cut a long result
   head-only at 2,000 characters. It now keeps the first 1,500 and up to the
   last 500, opening the kept end on a line start (`_truncate_middle`), same
   budget. The end is where the verdict is: exit summary, test totals, the
   exception line, the stderr block. Without this, 24% of long-result ends in
   a dense-carried task survived; with it, 98%.
4. **One-page head edge.** `HEAD_EDGE_FRAMES = 1`; the tail keeps
   `HQ_EDGE_FRAMES`. The two pages no longer held verbatim join the middle,
   where they are budgeted like all other old history and mined into the carry
   when their page is dropped. This is a real removal of old text from the
   context, not a deduplication: only 4% of the head was present elsewhere.
5. **Launch briefs carried once.** `collapse_relaunch_briefs` keeps the newest
   `subagent-launch:` turn verbatim and reduces each older one to the lines the
   newest does not contain. Keyed on the id the harness mints, never on text.
   On the 8 affected real blocks: 62-86% smaller, zero lines lost from the
   block, and the cap no longer evicts distinct resume notes to keep copies.

## What is given up

Inside the moved span only:

- the middle of a tool result longer than 2,000 characters;
- tool-call arguments past 500 characters;
- results flagged useless, with the call that produced them;
- images and documents, which become `[image]` / named-document placeholders
  (the raw window kept them as images; the newest `keep_recent_tokens` still
  does);
- provider reasoning blocks.

Everything else in the span is verbatim. Outside it:

- two pages of the session's oldest archived text leave the verbatim head;
- when the protected span is wider than the default three-page tail, the text
  just OLDER than the span sits in the middle pages instead of on that tail.
  The archive therefore reaches about three pages less far back, and that
  text may replay as an image frame rather than as text. On the two real tasks
  wider than the floor cap this was 22 of 113 and 18 of 146 steps no longer
  findable as text (see below).

The operator's own turns, the carried tier, and the newest `keep_recent_tokens`
of raw history are unchanged.

If the snapcompact branch itself raises, the pass falls back to the context-full
LLM summary as before, and the span that would have ridden the tail is
summarized with the rest: the cut was already made at recency. That path is
unchanged in mechanism and now covers more text.

## Measured result

Real `compact_now` on private transcript copies, isolated `HOME`, no model call.
The baseline is this tree with every lever switched off, which reproduces the
unmodified tree's results on 24 of 24 shared cases. The change is deterministic
across two runs in both benches.

### One pass (25 real compaction points)

| Group by live-task span | n | After, today | After, new | Change |
|---|---|---|---|---|
| Recency window only (span <= 20k) | 9 | 57.3k | 50.1k | -13% |
| Task 20-50k | 3 | 82.3k | 54.3k | -34% |
| Task 50-100k | 7 | 108.0k | 68.8k | -36% |
| Task over the 100k cap | 6 | 134.4k | 86.9k | -35% |
| All | 25 | 98.8k | 63.9k | -35% |

Worst case 154.7k -> 108.4k; no case larger; per-case reduction 11-54%.
Operator turns verbatim after the pass: 151 of 151 in both arms. CPU per pass:
10.9 s vs 10.6 s summed over the 25 passes (median 0.42 s vs 0.43 s).

Retention of what today keeps raw, all 25 points (today -> new): steps
1345/1345 -> 1343/1345, assistant messages 418/418 -> 418/418, result first
lines 1407/1412 -> 1404/1412, long-result last lines 313/320 -> 309/320. The
two missing steps are two-call messages where one call's result was flagged
useless, so that call is dropped from the line; the other call is present.

### Two consecutive passes (14 real pairs)

Pass 1 at a real compaction point, the session's real continuation appended,
then pass 2. The reference is the live task in the context the session really
had before its second pass, so it includes task history older than the floor
window. Figures are after pass 2; steps are those still findable as text.

| What happened between the passes | n | After, today | After, new | Steps, today | Steps, new |
|---|---|---|---|---|---|
| Same task, fitted the floor at pass 1 (continued) | 2 | 136.8k | 108.5k | 284/415 | 284/415 |
| Not continued (see below) | 7 | 57.8k | 50.1k | 352/1308 | 366/1308 |
| New operator turn before pass 2 | 5 | 110.5k | 75.4k | 299/351 | 259/351 |
| All | 14 | 69.4k | 55.0k | 935/2074 | 909/2074 |

No pass 2 is larger than today's. Without step 2 of the design, the continued
row was 49.0k with 127/415 steps: the protection lasted one pass.

The seven not continued: in three the floor was already clamped at pass 1, in
three the task's request had been archived before pass 1, and in one a new
operator turn arrived inside the recency window. Today's behaviour is plain
recency at pass 2 in all seven as well.

The tail bound was sized on the continued row. At the full floor cap the pass
lands where today's does (129k-131k against 135k-139k); at three quarters it
keeps the same number of steps as today at 108k-109k; at 0.6 and 0.5 it keeps
fewer than today (258 and 237 of 415).

The new-operator-turn row loses 40 steps as text, all in the two tasks wider
than the floor cap (22 of 113 and 18 of 146). In those two the archive dropped
about 38k more characters of its oldest text than today's did and replayed 5
and 8 of its 8 middle pages as image frames, which this probe cannot read. The
mechanism is the last item under "What is given up"; how much of the 40 is
dropped text and how much is image-replayed was not separated.

### Limits of this evidence

- Sizes are on the local ruler, so the provider-side figure is an estimate
  until real passes are observed after release.
- The points are stratified, so the per-group rows transfer and the overall
  medians are not a fleet forecast. The continued row is two sessions.
- The probe tests line-start presence as text, not usability, and cannot see
  inside pages replayed as image frames, which a vision model can read.

## Not changed

Context-full (text-only models) keeps the floor's raw window: it has no
verbatim tail, and an LLM summary would paraphrase the task. An
advisor-widened window stays raw. The mint site that re-stamps the launch
brief on resume (`harness/subagent.py`) is untouched; the collapse is the
compaction-side bound that holds whatever the mint site does.

A task that runs across many passes is, from its third pass, at plain recency
with a default tail, as it is today. Holding more of such a task costs about
1k tokens of context per 0.7% of its steps on the pairs above (87% of steps at
130k, 55% at 84k), and is a separate decision.
