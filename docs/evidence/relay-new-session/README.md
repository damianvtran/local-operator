# Evidence — relay one-tap new session

Frames captured 2026-10-04 by the lopdev manager session, driving the **real mobile
relay surface** (the built web bundle served by an isolated `lop mobile serve`
instance on 127.0.0.1:4499, throwaway HOME, real child runtimes) in the Local
Operator desktop app's browser host.

Two capture rounds:

- **v1** at `6abef9783` (the PR's first head): the flow's before/after story.
- **v2** at `c165705a1` (the round-1 remediation head): refreshed after the review
  rounds — chip affordance/target/overflow fixes, sheet contrast/wrap fixes, and
  the split refusal copy. v2 replaced the frames whose surfaces changed and added
  the long-path pair and the two refusal sentences.

Themes: "Local Operator Dark" (the default) and "Local Operator Light". Every
frame's theme is named in its caption below.

## Before (origin/main build, same isolated-instance method) — v1

- `before-list.png` — the list, empty state (footer "new session" opened a picker)
- `before-new-session-picker.png` — **the intermediate screen this PR removes** (`#/new`):
  working-directory quick-picks + model + Start
- `before-session.png` — a session started from the picker (no directory visible in the composer)

## After — the flow

- `after-fresh-list-empty.png` — fresh install, empty list (v1)
- `after-one-tap-composer.png` (dark, v2) — **ONE tap on "new session" landed here**
  (`#/s/…`, no intermediate); the chip shows `~/work/proj-b` (the registry's most
  recent directory — the default); the control now carries the app's chip ground,
  the control border and the 44 px floor
- `after-one-tap-typed.png` (dark, v2) — first characters ("hi") typed; no tap was
  needed in the session view
- `after-directory-sheet.png` (dark, v2) — the sheet: `current` / `home` / `tmp` /
  `recent` rows, un-dimmed (`aria-disabled` contrast rule), free-text field at
  `border-control`
- `after-recents-default.png` (light, v2) — the same one-tap, light theme
- `after-recents-sheet.png` (light, v2) — the sheet in light, with recent rows
- `after-recents-changed.png` (dark, v2) — moved to `~/work/proj-a` from the
  sheet; same session id, same route, the draft "hi" survived the retire/respawn

## After — the long-path pair (design round 1, D1)

`before-longpath-chip-{dark,light}.png` are from the **design round's live pass**
(credit: review-design round 1): the chip at a deep cwd measured `499.17 × 32` in a
`424`-wide row — overflowing, clipped, 65 px untappable, no ellipsis.
`after-longpath-chip-{dark,light}.png` (v2) are the same surface on the fixed head:
the chip measures **`420 × 44` — inside its row**, the label ellipsises, the whole
control is tappable.

`after-longpath-sheet-{dark,light}.png` (v2) — the sheet on the same deep path: the
`current` row WRAPS so the full path is readable (`break-all`, D6); the chip stays
single-line truncated above.

## After — refusal copy (UX round 1, U2)

- `after-refusal-notallowed.png` (dark, v2) — `a session can only work inside your home folder or the tmp root: /etc`
- `after-refusal-missing.png` (dark, v2) — `there's no directory at ~/nope-dir-9z`

The two sentences replace the single "that working directory can't be used"
sentence, which named neither cause (both render in-sheet, `role="alert"`).

## After — light theme

- `after-light-list.png` (v1)
- `after-light-session.png` (v2) — chip `~/work/proj-a` + draft, light
- `after-light-sheet.png` (v2) — the sheet in light

## Numbers behind the frames

- Chip: `~`-state `43.56 × 44` and `~/work/proj-b`-state `112.91 × 44` (v2; v1's `~`-state was `37.56 × 32`); deep-path state `420 × 44`
  inside a 424 px row (v2, was `499.17 × 32` overflowing); geometry via CDP computed styles.
- Autofocus pinned by `src/session-start.navigation.test.tsx` ("one tap on the list
  starts a session with no cwd and focuses the composer"): the POST body is `{}` AND
  `document.activeElement` is the "Message…" textarea; negatives assert resume /
  deep-link navigation never steal focus.
- Server-side cross-check (v1 and v2): `GET /api/directories` → `default` and
  `POST /api/sessions/start {}` land in exactly that directory; fresh-install
  default = `home`.
- `node scripts/contrast-contract.mjs` → "Contrast contract holds: 1333 assertions
  across 31 themes" on the remediation head (D2/D5).
