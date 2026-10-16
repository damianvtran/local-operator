---
name: highlights
description: "Highlights under an answer: callouts for a turn's deliverables (reports, PDFs, HTML previews), charts drawn from its numbers — how to switch it off, with privacy, cost and fixes."
---

# Highlights

After a turn that produced something real — a file, a block of numbers — Local Operator can add a short **Highlights** block under the final answer: callouts for the files the turn made, and a small generated graphic when the numbers justify one. The block draws no heading or label — no title, badge or separator announces it — and appears seamlessly directly under the answer's text: its position, not any caption, tells the reader it belongs to the turn. It is added only under a real user turn — a message you typed, or your answer to an ask the agent raised (never for wakes, monitors, peer messages, a timed-out or declined ask, or `lop exec`), always after the answer, and never in the model's context: the block sits in the transcript beside the answer, not in it.

## For the agent

The block is your delivery surface: a file the turn wrote or named surfaces under your answer as a file callout on its own, so name each deliverable and its path — and never tell the user you cannot attach, send or share a file. A standalone `.html` document you write can render as an inline preview on the desktop UI, and a turn's structured numbers can be drawn as a graphic: never deny a file or a visual that can be produced.

## Where it shows

Each surface paints the block from the row a turn journaled, and they do not sync with one another. Two ship it today:

- **TUI** (`lop`), since v0.68.27 — the file callouts, the `N more files` disclosure and the state line. It does **not** paint components: a terminal cannot run one, so an HTML component is skipped **silently**, with no placeholder box and no "open this in the app" note (a stub that admits it cannot render the thing is worse than the thing simply not being there). A turn whose only highlight is a graphic therefore shows nothing in the TUI.
- **Desktop UI** (local-operator-ui), since v0.34.14 — the same callouts and state line, plus every component mounted in a sandboxed frame (`allow-scripts`, opaque origin, no network, no file access, no popups). The component is body-only — no title line or caption restating the answer, only the data labelling it needs — and each mounted component carries exactly one `Source: …` caption naming where its numbers came from. A frame that dies or hangs is replaced by one quiet line where it stood (`Couldn't render this graphic`, with a `Retry` when the row can offer one), never by a rewritten block; a turn collapsed by the fold condenses the block with it, to one line of counts (`2 files · 1 graphic`) once the row has settled.

Not yet: the **relay / phone web** renderer is in review (core PR #2222), so a phone shows nothing until that lands; the **phone app's own native renderer** is a separate lane and is not shipped. Where a surface cannot render something, it shows nothing rather than an apology.

**Most turns have nothing to report.** With no file in the turn at all — and no numbers to draw from — the turn stops before anything leaves the machine, and a settled row with nothing in it paints no block at all. A source file the turn wrote is still offered to the decision: only the decision can tell "write me a script" from "this is the deliverable".

## What appears

- **File callouts** — up to four files the turn wrote or named (`reports/latency.md`, `data/bench.csv`, …), one row each with its name, the path as the row carries it (session- or `~/`-relative) and its size: openable in the desktop UI, and in the TUI clickable wherever the terminal renders links (plain, copyable text otherwise). Past the featured four, the rest fold into an `N more files` line that expands in place (bounded at the 20 stored). A finish with two files or fewer and no component is ONE line instead — `2 files: latency.md, bench.csv`. The path is never absolute — session-relative when the file sits under the session's cwd, `~/`-relative otherwise. Spends no model tokens.
- **Graphics** — a chart or table drawn from the turn's own numbers. Spends model tokens, and is **on by default**: the desktop UI renders it and the TUI skips it. It still costs nothing on most turns — the graphics question is not even asked unless the turn holds numbers to draw (see Cost). Turn it off with `supplements.graphics` (below).
- **Prepared HTML previews** — a standalone `.html`/`.htm` the turn successfully wrote can be mounted directly, with no classifier or generator call: the exact content the write executed (at most three documents per turn, 48 KiB each) is wrapped in the same validator and sandboxed frame as a graphic. Only the standalone-document profile is supported — `<!doctype html>`, one `<head>`/`<body>`, one plain-text `<title>`, inline style/script; external or sidecar scripts and stylesheets, CSS imports, module loaders, base URLs and custom meta are refused and stay file callouts. A later edit of that file, or any shell/eval step in the turn, invalidates a captured document. The desktop UI mounts it; the TUI skips components. Rides the `supplements.graphics` switch — off means file callouts.
- **The state line** — one quiet line while a job runs, and on the arms that never settle: `Preparing highlights…`, `Highlights cancelled`, `Couldn't prepare highlights`. The desktop UI draws those arms' controls too (`Adjust…`, `Cancel`, `Retry`); the TUI's actions are the `/highlights cancel | retry | adjust <text>` slash commands instead, because a terminal line advertising a click it cannot honour is a lie.

## Turning it on and off

Edits are scoped to new sessions — start one after a change (the TUI's `/settings → Highlights` and the desktop UI's Settings render the same section, from the same registry):

```bash
lop config edit supplements.enabled false     # the master switch
lop config edit supplements.files false       # file callouts only
lop config edit supplements.graphics false    # no generated graphics (they spend tokens)
```

In the file every key is `values.supplements.<key>`; the CLI spells it `supplements.<key>`, which is what `lop config list` prints:

| key | default | what it changes |
| --- | --- | --- |
| `enabled` | `true` | master switch |
| `files` | `true` | file callouts |
| `graphics` | `true` | generated graphics (spends tokens) |
| `model` | `auto` | the generator's model ladder; `session` forces this session's model |
| `maxTurns` | `2` | generator turns per job (1–4) |
| `maxOutputTokens` | `6000` | output budget per generator turn |
| `timeoutS` | `90` | whole-job wall clock |
| `maxCostUsd` | `1.00` | soft per-job spend cap, checked between turns |
| `maxFeatured` | `4` | files shown before "N more" folds the rest |
| `denyPrefixes` | `[]` | folders never listed or sent |

`LOP_SUPPLEMENTS=0` removes the feature from a process entirely — read once, at start; only `0`, `false`, `no` or `off` disable it.

## Privacy

- **Never listed or sent:** credentials and secrets (`.env`, keys, tokens, cloud and browser credential files), the harness's config dir and scratchpads, databases, `docker-compose` files, build outputs — plus anything under a prefix in `supplements.denyPrefixes`.
- **What leaves the machine for the decision:** each offered file's base name, its kind and size, the tool that wrote it, and that tool call's one-line intent — no directories, no contents — plus your message and the final answer, each clipped to a character bound. Everything sent passes the project's egress scrub first (`redaction_shapes.scrub_shapes`, the same pass the classification layer's outbound state takes): credential-shaped strings are masked and absolute paths removed, while a file sitting under a sensitive directory component (keys, credential stores, the config dir) is never offered as an option at all. The base name is kept deliberately — the decision needs it to judge which file is the deliverable. The decision runs on the classification cascade (Radient → TypeSafe → OpenRouter); with no vendor it falls back to a local heuristic and graphics are skipped.

## Cost

The decision is one small classification call per question — one for the files, plus a second, concurrent one only when the turn's evidence holds structured numbers (a table, delimited block, JSON array, four numbers sharing a unit in one paragraph, or a list-structured answer with four distinct operational counts offered as a labelled table); a turn without them asks the files question alone (`decide_many`, one call for both, is a follow-up). Neither delays the answer: the decision and any generator job run after it. Each is logged with its cost in the session log. A generator job is bounded by `supplements.maxCostUsd` (default `$1.00` — a soft between-turns cap sized to cover roughly one Opus-class turn, deliberately not derived from the measured Sonnet-class envelope of $0.04–0.08 per job) and its spend is recorded under a `supplement_render` purpose, so `/session`'s by-purpose rows can show it.

## Troubleshooting

1. **Nothing appeared.** Most turns legitimately show nothing (above), and a settled row with no files and no components paints no block at all — that is the normal case, not a failure. Two surfaces show nothing by design rather than because something broke: the TUI never draws a component, and the phone draws nothing until the relay lane lands. If it is never anything on the TUI or in the desktop UI, check `supplements.enabled` (and start a new session after an edit) and read the journal — a turn that actually produced something carries the decision's `vendor=` in its row, while a turn with nothing to show writes no row at all: a prefilter skip's reason (`supplements: prefilter … skipped=…`) and a decision that found nothing are only on DEBUG lines, so run with `LOG_LEVEL=DEBUG` to see either.
2. **Graphics skipped.** Charts need classifier confidence ≥0.7; a validated four-entity exact-count summary table uses ≥0.6 (distinct labels, no inferred trend or comparable bars). No vendor still means no generated graphic. A graphic is drawn only when the turn's evidence contains structured numbers — a model "yes" without evidence is overruled, and the question is not asked at all without it — and the TUI skips components whether or not one was drawn. Check `supplements.graphics` has not been switched off (it is on by default).
3. **A file is missing.** Only files the turn itself wrote or named can be called out: a file written by `write`, or named by a shell/python step (redirect, `tee`, `cp`/`mv` destination, `open(..., 'w')`, `to_csv(...)`, in the directory the command `cd`'d to) — a path the command builds in a variable is found by a one-level look at the directory it worked in, for files modified during the turn. A deliverable the answer itself links is still called out (the row carries `linked_in_answer`); a file the answer merely names without the turn having written it, excluded folders (above), source code the turn wrote, and files beyond the featured set ("N more") do not appear.

## Not in this release

- No image components — generated output is HTML; raster charts are not in v1.
- No mesh transfer — a file another device holds is named ("on <peer>") but not previewed.
- The relay lane and the phone app's own renderer — see "Where it shows".
- Static preview URLs are not yet token-enforced: tokens are minted and accepted, but a request without one is not refused (a recorded follow-up).
