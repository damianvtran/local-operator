# The `$skill` tray — rendered evidence

Frames for the `$skill` picker: the column-unit cap (design D1 / agent review
M2), the RTL description direction (D2), the "more below" cue (D3), the tray
under a waiting question (D5), and the sharded test rig's keep-the-foot rule at
keyboard heights.

Every frame here is a render of the **shipped palettes** — `localOperatorLight`
(`*-light`) or `localOperatorDark` (`*-dark`) — set through the app's own
`data-theme`. An earlier set was a forced-colors render: grayscale, so it could
evidence geometry and direction but not surfaces, the accent, the `hidden` tag's
colour, or contrast (design round 2, D6). Geometry is identical between the two
palettes; the pairs are there for the colour that is now real.

## How they were captured

One browser instance, throwaway profile, DPR 2. The **real components** —
`App` → `SessionScreen` → `AboveComposerStack` → `Composer` → `SkillTray` —
against the app's own stylesheet, with only the wire stubbed: `/api/*` answers
are built in the harness (`GET /api/skills` a fixture catalogue,
`useCapabilities`/`skill_invocation.available = true`, the projection frame with
or without `pending`). The locale goes through the product's own bootstrap
(`resolveInitialLocale()` fed a `{"language":"ur"}` frame on
`/api/config/events`), so the Urdu frame uses the shipped copy and the real
`<html dir>`.

**The keyboard is reproduced the way the product sees it**, not by shrinking the
viewport: the column is pinned to a `--lo-vvh` height while `vh` stays at the
device height (844). `screens/session-view.tsx` makes exactly those two writes;
a real keyboard shrinks only the visual viewport, so `vh` is untouched.

The rig is the same one round 1 and round 2 used (a Vite config whose `resolveId`
points `@lo` at the PR worktree's `src/`, plus a `node_modules` symlink into
`local_operator/mobile/web`), with the theme, column height, catalogue size,
draft and pending card as query parameters. It is still not committed because it
needs that symlink; the frames and the numbers below are what a re-render
reproduces.

## The numbers (read from the DOM, CSS px, column-relative)

| frame | column | tray cap | tray client / content | above-composer slot | composer top→bottom |
|---|---|---|---|---|---|
| `keyboard-508-before-light` | 508 | 352 (`min(45vh, 22rem)`) | 352 / 360 | — | 432 → 586, **78 px clipped** |
| `keyboard-508-after-light` / `-dark` | 508 | **228.6** (`0.45 × column`) | 229 / 360 | — | 346 → 500 |
| `keyboard-553-after-light` / `-dark` | 553 | 248.85 | 249 / 360 | — | 391 → 545 |
| `8rows-390x844-light` / `-dark` | 844 | 379.8 | 360 / 360 | — | 682 → 836 |
| `3rows-320x568-light` | 568 | 255.6 | 140 / 140 | — | 406 → 560 |
| `ask-508-before-light` / `-dark` | 508 | 228.6 | 229 / 360 | **29.4** | 346 → 500 |
| `ask-508-light` / `-dark` | 508 | **96** (two rows) | 96 / 360 | **162** | 346 → 500 |
| `ask-553-light` / `-dark` | 553 | 96 | 96 / 360 | 207 | 391 → 545 |
| `ask-470-light` / `-dark` | 470 | 96 | 96 / 360 | 168 | 352 → 462 |
| `ask-320-light` / `-dark` | 568 | 96 | 96 / 360 | 222 | 406 → 560 |
| `ask-844-light` / `-dark` | 844 | 96 | 96 / 360 | 271.1 | 682 → 836 |
| `ask-dock-508-light` / `-dark` | 508 | **96** (two rows) | 96 / 360 | **44** (32 px dock inside) | 346 → 500 |

The before row is round 1's measurement reproduced exactly (432 → 586 against a
508 foot, 78 px clipped, only the top 12 of the send disc's 44 px on screen).

## What each frame shows

**Without a question waiting** — the cap is 45 % of the column:

- `keyboard-508-before-light` / `keyboard-508-after-light` — the D1/M2 blocker
  and its fix on the same column, one declaration apart.
- `keyboard-553-after-light` / `-dark` — a smaller keyboard; the sixth row fades
  under the `lo-fade-y` cue while the composer keeps its whole foot.
- `8rows-390x844-light` / `-dark` — the whole fixture catalogue; all eight rows
  fit (360 px of content inside a 379.8 px cap), so nothing is clipped.
- `8rows-ur-rtl-390x844-light` / `-dark` — Urdu: rows mirror, `$name` stays
  LTR, the tag reads `پوشیدہ`. Every description keeps its opening words and
  truncates at its END, because it carries its own first-strong direction.
- `3rows-320x568-light` — the narrow column with a three-row list: the cap is
  above the content (140 px), so the tray is its own height and the composer's
  fit is untouched.

**With an approval waiting** — the cap drops to two rows (design D5):

- `ask-508-before-light` / `-dark` — the blocker: at a 508 px column the 45 %
  tray took 228.6 px, the above-composer slot fell to **29.4 px**, and the card
  scrolled its own question (title at y −24.6) above the column, leaving the
  approve/deny row sliced at the slot's edge.
- `ask-508-light` / `-dark`, `ask-553-light` / `-dark`, `ask-470-light` /
  `-dark`, `ask-320-light` / `-dark`, `ask-844-light` / `-dark` — the same
  columns with the two-row cap: the tray keeps `$name` discoverable and
  scrolls, the slot grows to 162 / 207 / 168 / 222 / 271.1 px, and the card's
  meta row, title and decision row are all inside it (`textVisibleInSlot`
  true in every one of these runs, false in the before frame).

At 844 the tray is two rows as well: with an ask on screen the list yields the
column everywhere, not only where it had to, so the slot measures 271.1 px and
the card has room to spare. That is the one state where the round-2 design
report's target said "unchanged" (it measured a 234 px slot with the eight-row
tray, the state this PR's 45 % cap produces without an ask); the verdict — whole
card, nothing clipped — is the same either way, and only the number moves.

**With a queued ask waiting — a dock, not a card (Q-R4-1):**

- `ask-dock-508-light` / `-dark` — a real open ask: the runtime publishes the
  `asks` row and mirrors it into `pending`, so `blockingPending` draws no card,
  but the 32 px dock is on screen. The two-row cap applies (96 px), the slot
  keeps **44 px** and the dock sits whole inside it, and the composer's send
  disc is whole (410 → 454 against the 500 px composer foot). Before the fix
  the 45 % cap applied: tray 228.6 px, slot 29.4 px, and the dock (78 → 110)
  running past the slot's 101.4 bottom edge — clipped out of view. QA's
  390×508 fixture measured slot 4.5 px and 0 px of dock visible; the numbers
  move with the fixture's card height, the state does not. The asks sheet, which
  opens by default on a cold load, is dismissed before the frame so the measured
  surface is the dock itself.
- Measured, not framed: the same state at 390×844 is tray 96, slot 44, dock
  32 px inside it (the 45 % cap did not clip there; the defect was the keyboard
  column).

## What is not evidenced here

- A real phone keyboard, IME, or on-screen keyboard: the column is pinned, not
  squeezed by a real inset.
- The keyboard-closed layout below 470 px with an ask, and the RTL ask card.
- Colour contrast ratios: the frames are rendered with the shipped palettes, so
  contrast can be judged from them, but no ratio is asserted here — the
  contrast contract is `scripts/contrast-contract.mjs` in the web app.
