---
name: highlights
description: What the Highlights block under an answer is — the file callouts and generated graphics added after a turn — and how to switch it off, with privacy, cost and fixes.
---

# Highlights

After a turn that produced something real — a file, a block of numbers — Local Operator can add a short **Highlights** block under the final answer: callouts for the files the turn made, and a small generated graphic when the numbers justify one. It appears only under a real user turn (never for wakes, monitors, peer messages or `lop exec`), always after the answer, and never in the model's context: the block sits in the transcript beside the answer, not in it. **Most turns show nothing** — no deliverable files and no numeric evidence means the turn is skipped before anything leaves the machine.

## What appears

- **File callouts** — up to four files the turn wrote or named (`reports/latency.md`, `data/bench.csv`, …) with an "N more" line for the rest; paths are session-relative, never absolute. This half costs nothing.
- **Graphics** — a chart or table drawn from the turn's own numbers. Spends model tokens; on a stock install it is **off** until the surfaces that can render a graphic are out.

## Turning it on and off

Edits are scoped to new sessions — start one after a change (the TUI's `/settings → Highlights` shows the same section):

```bash
lop config edit supplements.enabled false     # the master switch
lop config edit supplements.files false       # file callouts only
lop config edit supplements.graphics true     # once a renderer ships
```

In the file every key is `values.supplements.<key>`; the CLI spells it `supplements.<key>`, which is what `lop config list` prints:

| key | default | what it changes |
| --- | --- | --- |
| `enabled` | `true` | master switch |
| `files` | `true` | file callouts |
| `graphics` | `false` | generated graphics (spends tokens) |
| `model` | `auto` | the generator's model ladder; `session` forces this session's model |
| `maxTurns` | `2` | generator turns per job (1–4) |
| `maxOutputTokens` | `6000` | output budget per generator turn |
| `timeoutS` | `90` | whole-job wall clock |
| `maxCostUsd` | `0.20` | soft per-job spend cap |
| `maxFeatured` | `4` | files shown before "N more" folds the rest |
| `denyPrefixes` | `[]` | folders never listed or sent |

`LOP_SUPPLEMENTS=0` removes the feature from a process entirely — read once, at start; only `0`, `false`, `no` or `off` disable it.

## Privacy

- **Never listed or sent:** credentials and secrets (`.env`, keys, tokens, cloud and browser credential files), the harness's config dir and scratchpads, databases, `docker-compose` files, build outputs — plus anything under a prefix in `supplements.denyPrefixes`.
- **What leaves the machine for the decision:** each file's name, size and writing tool — no directories, no contents — plus your message and the final answer, bounded and redacted like the classification layer. The decision runs on the classification cascade (Radient → TypeSafe → OpenRouter); with no vendor it falls back to a local heuristic and graphics are skipped.

## Cost

The decision is one small classification call, logged with its cost in the session log. A generator job is bounded by `supplements.maxCostUsd` (default `$0.20`; the design's per-job envelope is $0.04–0.08) and its spend is recorded under a `supplement_render` purpose, so `/session`'s by-purpose rows can show it.

## Troubleshooting

1. **Nothing appeared.** Most turns legitimately show nothing; if it is never anything, check `supplements.enabled` (and start a new session after an edit) and read the session log — it says why (`supplements: prefilter … skipped=…`, or the decision's `vendor=`).
2. **Graphics skipped.** Off in this build; even when on, a graphic is drawn only when the turn's evidence contains numbers — a model "yes" without evidence is overruled.
3. **A file is missing.** Only files the turn itself wrote or named can be called out; excluded folders (above), paths already linked in the answer, and files beyond the featured set ("N more") do not appear.

## Not in this release

- No image components — generated output is HTML; raster charts are not in v1.
- No mesh transfer — a file another device holds is named ("on <peer>") but not previewed.
- The phone app's WebView frame is pending its own lane.
- Static preview URLs are not yet token-authenticated (a recorded follow-up).
