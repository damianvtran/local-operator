# Evidence — relay one-tap new session

Frames captured 2026-10-04 by the lopdev manager session, driving the **real mobile
relay surface** (the built web bundle served by an isolated `lop mobile serve`
instance on 127.0.0.1:4499, throwaway HOME, real child runtimes) in the Local
Operator desktop app's browser host. PR head at capture: `6abef9783`.

Themes: "Local Operator Dark" (the default) for the main pass, "Local Operator
Light" for the light pass.

## Before (origin/main build, same isolated-instance method)

- `before-list.png` — the list, empty state (footer "new session" opened a picker)
- `before-new-session-picker.png` — **the intermediate screen this PR removes** (`#/new`):
  working-directory quick-picks + model + Start
- `before-session.png` — a session started from the picker (no directory visible in the composer)

## After

- `after-fresh-list-empty.png` — fresh install, empty list
- `after-one-tap-composer.png` — **ONE tap on "new session" landed here** (`#/s/…`, no
  intermediate); the chip shows `~` — the fresh-install default (home), visible
- `after-one-tap-typed.png` — first characters ("hi") typed; no tap was needed in the
  session view
- `after-directory-sheet.png` — chip → sheet (`~ current`, `tmp`)
- `after-changed-to-tmp.png` — moved to `/private/tmp`: same session id, same route,
  the draft "hi" survived the retire/respawn
- `after-recents-default.png` — with used directories seeded in the agent registry, a
  one-tap start lands in `~/work/proj-b` (most recent), visible in the chip
- `after-recents-sheet.png` — the sheet with current / home / tmp / recent rows
- `after-recents-changed.png` — changed to `~/work/proj-a` from the composer; same id
- `after-light-list.png` / `after-light-session.png` / `after-light-sheet.png` — the
  same surfaces in "Local Operator Light"

## Numbers behind the frames

- Chip hit box **37.56 × 32 px** at its `~` state (12px mono), measured via CDP
  computed styles.
- Autofocus is pinned by `src/session-start.navigation.test.tsx` ("one tap on the list
  starts a session with no cwd and focuses the composer"): asserts the POST body is
  `{}` AND `document.activeElement` is the "Message…" textarea; negative cases assert
  resume / deep-link navigation never steal focus.
- Server-side cross-check in the same pass: `GET /api/directories` →
  `{"default":"…/work/proj-b", …}` and `POST /api/sessions/start {}` landed exactly
  there; fresh-install default = `home`.
