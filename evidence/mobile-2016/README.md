# Evidence — issue #2016: mobile one-gesture "mark all as read" (PR #2023)

This branch carries the runnable evidence for PR #2023 (`feat/mobile-mark-all-read`,
head `f46ee0cf8`). It contains **no product code** and is never merged; the files
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
