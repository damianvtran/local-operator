# Evidence — issue #2017 (mobile focus-zoom, wide view included)

Temporary capture drivers, frames and reports for the fix PR. **Not for merge**:
this branch exists so the PR description can pin links to the frames, per the
repo's convention (AGENTS.md, "Evidence goes on the PR") — the artifacts never
ride the feature branch and never land on `main`.

## What is here

- `capture_focus_zoom.py` — drives the REAL built SPA (served by
  `scripts/mobile_overflow_fixture.py`) out of headless Chrome at 390x844 and
  360x780, recording the viewport meta, `visualViewport.scale`,
  `--lo-fit-scale` and every field's computed font-size, plus one PNG per
  surface.
- `probe_focus_zoom.py` — focuses each field class and reads the scale before
  and after, in both modes: it shows exactly what this host's Chromium does
  (fit scale + stable font metrics) and does not do (WebKit iOS focus-zoom).
- `probe_st_second.py` — the one-off DOM dump behind the ask-card finding
  (free-text asks sit on rows the app renders as terminal, so no answerable
  ask-card input materialises in this fixture; st-second is a dead session).
- `before-*` / `after-*` — frames + reports of the same drivers on the same
  fixture, pre-fix and post-fix, same viewports.
- `served-assets-checks-*.txt` — served HTML/CSS/JS token scans.
- The measured numbers table is in the PR body; the raw values are in
  `before-focus-zoom-report.json` / `after-focus-zoom-report.json`.

## Re-run (the invocation this evidence was produced with)

From a worktree with the fix built (`corepack pnpm install --frozen-lockfile &&
corepack pnpm build` in `local_operator/mobile/web`, node 24 + pnpm 11.22.0 via
corepack) and its own venv (`uv venv --python 3.12 .venv && uv pip install -e
".[all,dev]" --python .venv/bin/python`):

```sh
cd <worktree>
export LOP_MOBILE_FIXTURE_PASSWORD=$(python3 -c 'import secrets;print(secrets.token_urlsafe(16))')
PYTHONPATH=. .venv/bin/python scripts/mobile_overflow_fixture.py 4317
# second shell, same env var — the drivers read it and never print it:
PYTHONPATH=. .venv/bin/python docs/evidence/mobile-2017-focus-zoom/capture_focus_zoom.py <outdir> 4317
PYTHONPATH=. .venv/bin/python docs/evidence/mobile-2017-focus-zoom/probe_focus_zoom.py <outdir> 4317
```

The drivers import `scripts.probe_isolation` first, which re-homes `HOME` and
`LOCAL_OPERATOR_CONFIG_DIR` (and drops `CMUX_*`), so the fixture cannot reach
the operator's live daemon or sessions; `dial_registrants=False` disables
registrant sockets entirely. No credentials are stored in this directory.

## What this cannot show

Chromium does not implement WebKit iOS's `_zoomToFocusRect:` focus-zoom, and
this host has no simulator/Xcode tooling (`xcrun simctl list runtimes` →
`unable to find utility "simctl"`; recorded on PR #1913). The drivers verify
the fit scale and the computed field fonts — the two terms WebKit's rule
consumes — not the zoom itself. Device verification (real iPhone, 360pt and
390pt) is handed to QA.
