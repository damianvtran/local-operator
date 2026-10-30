# The `$skill` tray — rendered evidence for the round-1 fixes

Frames for the review round that moved the tray's cap into column units (design
D1 / agent review M2), fixed the RTL description direction (D2), added the
"more below" cue (D3), and replaced SKM-D5's IME composition guard (M1).

## How they were captured

One browser instance, throwaway context, DPR 2. The **real components** —
`Composer`, `AboveComposerStack`, `PendingCard` — rendered inside a column that
reproduces `screens/session-view.tsx`'s layout (header, transcript `flex-1`,
the above-composer slot, the composer), with only the data layer stubbed
(`GET /api/skills` answering a fixture catalogue, `useCapabilities` answering
`skill_invocation.available = true`, an empty draft). The locale goes through the
product's own bootstrap: `resolveInitialLocale()` fed a `{"language":"ur"}` frame
on `/api/config/events`, so the Urdu frame uses the shipped copy and the real
`<html dir>`.

**The keyboard is reproduced the way the product sees it**, not by shrinking the
viewport: the column is pinned to a `--lo-vvh` height while `vh` stays at the
device height (844). The session view's own handler makes exactly these two
writes; a real keyboard shrinks only the visual viewport, so `vh` is untouched.

The `*before*` frame paints the round-1 cap on the SAME frame
(`max-height: min(45vh, 22rem)`), so the pair differs by one declaration.

Rig: a Vite config whose `resolveId` points the composer's `../api` and
`../store` imports at stub modules; the height, `--lo-vvh`, catalogue size,
draft and approval card are query parameters. Not committed because it needs a
`node_modules` symlink into `local_operator/mobile/web`; the frames and the
numbers below are what a re-render should reproduce.

## The numbers (read from the DOM, CSS px, column-relative)

| frame | column | tray cap | tray client / content | composer top→bottom | clipped | send disc |
|---|---|---|---|---|---|---|
| `keyboard-508-before.png` | 508 | 352 (`min(45vh, 22rem)`) | 352 / 360 | 432 → 586 | **78** | 496 → 540, 12 of 44 px visible |
| `keyboard-508-after.png` | 508 | **228.6** (`0.45 × column`) | 229 / 360 | 346 → 500 | **0** | 410 → 454, fully visible |
| `keyboard-553-after.png` | 553 | 248.85 | 249 / 360 | 391 → 545 | 0 | 455 → 499, fully visible |
| `column-470-with-approval.png` | 470 | 211.5 | 212 / 360 | 352 → 462 | 0 | 416 → 460, fully visible |
| `8rows-390x844.png` | 844 | 379.8 | 360 / 360 | 682 → 836 | 0 | 746 → 790, fully visible |

The before row is the review's measurement, reproduced exactly (432 → 586
against a 508 foot, 78 px clipped, 12 of the send disc's 44 px on screen).

## What each frame shows

- `1row-390x844.png` — draft `$res`, one matching row.
- `3rows-390x844.png` — draft `$`, three rows, `hidden` tag on `incident-retro`.
- `8rows-390x844.png` — the whole fixture catalogue; at this column all eight
  rows fit (360 px of content inside a 379.8 px cap), so nothing is clipped.
- `8rows-ur-rtl-390x844.png` — Urdu: rows mirror, `$name` stays LTR, the tag is
  `پوشیدہ`. Every description keeps its opening words and now truncates at its
  END (`Read the sources before changing any code…`), because the description
  carries its own first-strong direction.
- `3rows-320x568.png` — the narrow column; the composer's existing fit is
  unchanged.
- `keyboard-508-before.png` / `keyboard-508-after.png` — the blocker, before and
  after, on the same column.
- `keyboard-553-after.png` — a smaller keyboard; the sixth row fades out under
  the `lo-fade-y` cue while the composer keeps its whole foot.
- `column-470-with-approval.png` — a landscape/short column with a pending
  approval. The composer's foot is fully visible. The approval card is still
  squeezed to its decision row (slot 52.5 px): at a 470 px column `0.45 × column`
  equals the old `45vh`, so this state is unchanged by the cap — see the PR
  comment's note under D1.
