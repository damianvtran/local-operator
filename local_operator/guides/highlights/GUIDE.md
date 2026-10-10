---
name: highlights
description: What the Highlights block under an answer is — the file callouts and generated graphics added after a turn — and how to switch it off, with privacy, cost and fixes.
---

# Highlights

After a turn that produced something real — a file, a block of numbers — Local Operator can add a short **Highlights** block under the final answer: callouts for the files the turn made, and a small generated graphic when the numbers justify one. It appears only under a real user turn (never for wakes, monitors, peer messages or `lop exec`), always after the answer, and never in the model's context: the block sits in the transcript beside the answer, not in it.

**Nothing paints it yet.** The decision and the row ship first: an eligible turn journals a `supplement_v1` row, readable by tooling and API consumers (the session log, a history read), but no surface renders a Highlights block at this head — the TUI, UI, relay and native renderers are their own lanes, so until one of them lands a correct turn still shows nothing anywhere, whatever the settings say.

**Most turns have nothing to report.** With no file in the turn at all — and, once graphics are on, no numbers to draw from — the turn stops before anything leaves the machine. A source file the turn wrote is still offered to the decision: only the decision can tell "write me a script" from "this is the deliverable".

## What appears

- **File callouts** — up to four files the turn wrote or named (`reports/latency.md`, `data/bench.csv`, …) with an "N more" line for the rest; paths are session-relative, never absolute. Spends no model tokens.
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
- **What leaves the machine for the decision:** each offered file's base name, its kind and size, the tool that wrote it, and that tool call's one-line intent — no directories, no contents — plus your message and the final answer, each clipped to a character bound. Everything sent passes the project's egress scrub first (`redaction_shapes.scrub_shapes`, the same pass the classification layer's outbound state takes): credential-shaped strings are masked and absolute paths removed, while a file sitting under a sensitive directory component (keys, credential stores, the config dir) is never offered as an option at all. The base name is kept deliberately — the decision needs it to judge which file is the deliverable. The decision runs on the classification cascade (Radient → TypeSafe → OpenRouter); with no vendor it falls back to a local heuristic and graphics are skipped.

## Cost

The decision is one small classification call per question — one for the files, plus a second, concurrent one once graphics are on (`decide_many`, one call for both, is a follow-up). Each is logged with its cost in the session log. A generator job is bounded by `supplements.maxCostUsd` (default `$0.20`; the design's per-job envelope is $0.04–0.08) and its spend is recorded under a `supplement_render` purpose, so `/session`'s by-purpose rows can show it.

## Troubleshooting

1. **Nothing appeared.** Expected today: no surface paints the block yet, so no settings change brings it back (see the top of this guide). Once a renderer ships, most turns still legitimately show nothing; if it is then never anything, check `supplements.enabled` (and start a new session after an edit) and read the session log — it says why (`supplements: prefilter … skipped=…`, or the decision's `vendor=`).
2. **Graphics skipped.** Off in this build; even when on, a graphic is drawn only when the turn's evidence contains numbers — a model "yes" without evidence is overruled.
3. **A file is missing.** Only files the turn itself wrote or named can be called out; excluded folders (above), paths already linked in the answer, and files beyond the featured set ("N more") do not appear.

## Not in this release

- **No surface renders the block yet** — the TUI, UI, relay and native renderers are separate lanes; what exists today is the journaled row, readable by tooling and API consumers.
- No image components — generated output is HTML; raster charts are not in v1.
- No mesh transfer — a file another device holds is named ("on <peer>") but not previewed.
- The phone app's WebView frame is pending its own lane.
- Static preview URLs are not yet token-authenticated (a recorded follow-up).
