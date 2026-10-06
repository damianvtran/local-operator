# Evidence — issue #2016: mobile one-gesture "mark all as read" (PR #2023)

This branch carries the runnable evidence for PR #2023 (`feat/mobile-mark-all-read`;
head `f46ee0cf8` for the original slice, `522fc80e6` for the round-1 remediation,
`842490005` for the round-2 remediation).
It contains **no product code** and is never merged; the files
exist so the PR thread can link frames and raw logs at a pinned commit, and so QA
can re-execute the rigs.

## Provenance

- **Fixture**: the repo's own `scripts/mobile_overflow_fixture.py`, wrapped by the
  copy in this branch (one seam patched: `build_app`), run on an isolated HOME +
  `LOCAL_OPERATOR_CONFIG_DIR` sandbox (the repo's `scripts/probe_isolation.py`),
  `dial_registrants=False`, port 4216 (non-default), synthetic 12-hex ids only.
  Teardown verified: port free, no fixture processes, run sandboxes removed.
- **Client**: the worktree's own build of `local_operator/mobile/web` (served by the
  fixture daemon).
- **Captures**: Chrome driven by the repo's `scripts/mobile_overflow_capture.py` —
  `headless=new`, device metrics 390x844 (dpr 2), `--use-mock-keychain`, one instance
  per run, teardown asserts zero leftover processes.
- No secrets: the login password lives only in a per-run env file that was deleted
  after the runs; the transcripts redact it and the cookie value.

## Layout

### `repro/` — before the fix (the reproduction the operator gate cleared first)

- `scripts/` — the three repro scripts (fixture wrapper, capture, transcript).
- `artifacts/01-list-390x844.png` — the list with two unread rows and **no** mark-all
  control anywhere.
- `artifacts/dom-list-initial.json`, `dom-session.json` — control inventories proving
  zero bulk controls (list: 21 controls / 0 matches; session screen: 8 / 0).
- `artifacts/transcript.log` / `.json` — `POST /api/attention/seen` → **404** (the gap);
  the single-session route works (200); the badge read moves by exactly one (2 → 1).

### `postfix/` — after the fix (the PR's head)

- `scripts/` — the updated rig. The fixture wrapper adds two fixture-only hooks:
  `POST /fixture/publish` (settle a newer completion mid-run — stages the
  not-a-sweep case against a real store write) and `GET /fixture/state` (the shared
  store's own read). The capture script adds a **real pointer tap** (CDP
  `Input.dispatchMouseEvent` pressed+released at the control's measured centre).
- `artifacts/01-list-unread-with-control-390x844.png` — both unread rows **+ the
  control** ("mark all as read", measured 100x44 at 12,92).
- `artifacts/02-list-marked-390x844.png` — after ONE tap: marks gone, receipt
  **"Marked 2 read."** above the footer, control unmounted.
- `artifacts/dom-list-initial.json` / `dom-list-marked.json` — the control found by
  name and its rect; then `marks: 0`, control absent after the tap.
- `artifacts/03-session-390x844.png`, `dom-session.json` — the session screen still
  carries no bulk control (scope: "anywhere in the client").
- `artifacts/transcript.log` / `.json` — the wire pass on a fresh fixture generation:
  counts **2 → 1 → 1 → 0**; a mixed batch answers `read=[…] , unknown=["deadbeef1234"]`
  (per-item, still 200); the **NOT-A-SWEEP** attempt with a token superseded after the
  render answers `superseded`, clears nothing, and `GET /fixture/state` shows the
  conversation still unseen on the NEWER token; the finish clears it; an all-unknown
  no-op is still 200.
- `artifacts/store-state.json`, `publish-result.json`, `attention-unread-*.json`,
  `sessions-*.json` — the raw payloads behind those readings.
- `logs/` — the two fixture runs' stdout (generation A: capture; generation B:
  transcript — the capture consumes the unread pile, hence two generations).

### `round1/` — round-1 remediation (head `522fc80e6`)

Five states at 390x844, each a rendered frame plus the numbers behind it
(`artifacts/capture-report.json`), captured with the same rig:

- `artifacts/01-control-with-count-390x844.png` — the control now states the pile
  size: **`mark all 2 read`**, measured **366x44 at (12, 92)** — stretched to the
  column like the search field (design D4/UX U6; it was a 99.5px shrink-wrapped
  pill).
- `artifacts/02-scrolled-sticky-390x844.png` — at the list's **own maximum scroll
  (`scrollTop` 271 of 1014)**, both unread marks are visible near the foot and the
  control is still on screen: the band is sticky. **Its own measurement in
  `capture-report.json` is `top: 40`** — the unscrolled state reads `top: 92`, and
  the two were transposed in the round-1 prose (agent round 2, NIT-1); the band
  rect is 40–92 because the search field above it has scrolled away. Before this
  round the control measured **y=-179** — entirely off-screen — in exactly this
  state (design D5/UX U7).
- `artifacts/03-receipt-near-gesture-390x844.png` — after one real pointer tap:
  marks 0, and the receipt **"Marked 2 read."** renders **in the band at the top
  (y=92, with a Dismiss control)** instead of 646px away beside the footer
  (design D3/UX U1).
- `artifacts/04-degraded-receipt-390x844.png` — **the MAJOR fix.** With the
  unread read genuinely failing (the fixture raises from
  `AttentionStore.state_many_and_revision` — the seam the repo's own unread test
  injects, so this is the daemon's real degraded aggregate), the control is still
  mounted over two painted marks and the press answers, in danger ink,
  **"Could not read what is unread — nothing was cleared. Try again."** — never
  "Nothing unread." (agent MAJOR-1 = design D1). No POST is made.
- `artifacts/05-failure-line-390x844.png` — with the bulk write blocked at the
  network layer, the marks stay at 2 and the alert reads **"Nothing was cleared —
  the daemon could not be reached. Try again."**, naming the recovery instead of
  echoing `Failed to fetch` (UX U5).
- `artifacts/transcript.log` / `.json` + `shape-readings.json` — the wire pass on a
  fresh generation: badge **2 → 0** on the happy path, and the SHAPE layer pinned —
  `zzz-not-hex`, `ABCDEF012345`, an 11-char id, a non-UUID token and an empty token
  each answer **422** with the same sentence (agent NIT-2 / QA Q2/Q3), while a
  well-formed but unknown id keeps the per-item **`unknown`** verdict at 200.
- `logs/` — the two fixture generations' stdout (capture, then transcript).

### `round2/` — round-2 remediation (head `842490005`)

Four states at 390x844, each a rendered frame plus the numbers behind it
(`artifacts/capture-report.json`). Same rig, same fixture wrapper as `round1/`
(one fixture, two capture generations — the r2 script is the only new file):

- `artifacts/01-band-separator-max-scroll-390x844.png` — the pinned band at the
  list's own maximum scroll (`scrollTop` 272 of 1015), now carrying its own edge:
  computed **`border-bottom-width: 1px`** (was `0px`, `box-shadow: none`), band
  rect **40–93 (`h 53`)** — one pixel taller than the pre-fix band (`40–92`,
  `h 52`) because the edge is part of the box (design round 3, D9) — with two
  unread marks still visible beneath it. **Design D6.**
- `artifacts/02-focus-ring-390x844.png` — keyboard activation with a query that
  registers no card: after a real `Tab` the focus is the control, and after a real
  `Enter` the batch clears and the fallback lands on the band **with a visible
  ring** — `document.activeElement` is the band `DIV`, computed **`outline-style:
  solid`, `outline-width: 2px`, `outline-color: rgb(56, 201, 106)`** (the accent
  ring). **Design D7 / QA Q-2 / agent NIT-3.**
- `artifacts/03-degraded-no-count-390x844.png` — with the unread read failing for
  real, the control now reads the plain **`mark all as read`** (the count it cannot
  stand behind is gone) above the danger alert, both marks still painted, and the
  step's **`/api/attention/seen` request count is 0** — read off the wire by a
  page-side `fetch` counter installed for every document (agent NIT-2's artifact;
  the same counter reads **1** on the steps that do post, so the zero is
  discriminating rather than vacuous). **UX U11.**
- `artifacts/04-receipt-window-after-absence-390x844.png` — a route away and back
  with **`elapsedAwaySeconds: 15.8`** (longer than the 12s TTL): the receipt is
  **still present on return** at (12, 92) 314x28, because the window is the
  reader's own viewing time and the clock pauses while the screen is unmounted.
  **UX U9.**

`logs/capture.log` is the rig's stdout for this pass. The counter and key-event
helpers live in `scripts/mobile_2016_capture_r2.py`; the fixture wrapper it runs
is `round1/scripts/mobile_2016_fixture_r1.py` (unchanged).

## Re-running (QA)

From a worktree of the **PR head** (never the shared root checkout), with this
branch's scripts referenced by path (they import `scripts.*` from `PYTHONPATH`, which
must point at the PR-head worktree — that is how they were run here):

```sh
SCRATCH=<your scratchpad dir>
mkdir -p "$SCRATCH/artifacts"
python3 -c 'import secrets; open("'"$SCRATCH"'/pw.env","w").write("LOP_MOBILE_FIXTURE_PASSWORD=" + secrets.token_urlsafe(16) + "\n")'
chmod 600 "$SCRATCH/pw.env"
set -a; . "$SCRATCH/pw.env"; set +a

# 1) fixture (needs port 4216 free; run from the PR-head worktree)
PYTHONPATH=. LOP_2016_SEED_OUT="$SCRATCH/artifacts/seed.json" \
  .venv/bin/python <this-branch>/evidence/mobile-2016/postfix/scripts/mobile_2016_fixture.py 4216

# 2) capture (same fixture generation; consumes both unread rows via the tap)
PYTHONPATH=. .venv/bin/python <this-branch>/evidence/mobile-2016/postfix/scripts/mobile_2016_capture.py 4216 "$SCRATCH/artifacts"

# 3) transcript (RESTART the fixture first — fresh generation)
.venv/bin/python <this-branch>/evidence/mobile-2016/postfix/scripts/mobile_2016_transcript.py 4216 "$SCRATCH/artifacts"
```

The repro-phase scripts under `repro/scripts/` run the same way (their transcript
asserts the pre-fix contract: the bulk route absent).

The remediation generations run against the SAME fixture wrapper:

```sh
# round-1 rig (five states)
PYTHONPATH=. .venv/bin/python <this-branch>/evidence/mobile-2016/round1/scripts/mobile_2016_capture_r1.py 4216 "$SCRATCH/artifacts"

# round-2 rig (four states; re-arms its own pile, so it can be re-run against a
# store an earlier pass already cleared, and it counts /api/attention/seen
# requests per step)
PYTHONPATH=. .venv/bin/python <this-branch>/evidence/mobile-2016/round2/scripts/mobile_2016_capture_r2.py 4216 "$SCRATCH/artifacts"

# the wire transcript (shape layer + happy path)
PYTHONPATH=. .venv/bin/python <this-branch>/evidence/mobile-2016/round1/scripts/mobile_2016_transcript_r1.py 4216 "$SCRATCH/artifacts"
```

All three expect the fixture started from
`evidence/mobile-2016/round1/scripts/mobile_2016_fixture_r1.py` on the same port
(it carries the `/fixture/publish`, `/fixture/state` and `/fixture/degrade`
hooks); the degraded states need `/fixture/degrade?on=1` set more than 1 s after
the last badge read, or the 1 s summaries cache answers before the seam.

The round-1 rigs add one capability hook each and run identically:

```sh
# fixture with the degraded seam + the out-of-band publish hook
PYTHONPATH=. LOP_2016_SEED_OUT="$SCRATCH/artifacts/seed.json" \
  .venv/bin/python <this-branch>/evidence/mobile-2016/round1/scripts/mobile_2016_fixture_r1.py 4216

# five captures (same generation; each tap consumes the pile, so the script
# republishes and reloads before the degraded and failure states)
PYTHONPATH=. .venv/bin/python <this-branch>/evidence/mobile-2016/round1/scripts/mobile_2016_capture_r1.py 4216 "$SCRATCH/artifacts"

# wire transcript (RESTART the fixture first — fresh generation)
.venv/bin/python <this-branch>/evidence/mobile-2016/round1/scripts/mobile_2016_transcript_r1.py 4216 "$SCRATCH/artifacts"
```
