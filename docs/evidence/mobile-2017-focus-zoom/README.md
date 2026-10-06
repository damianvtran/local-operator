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
  surface, per mode. Frames are named `<vp>-<default|wide>-<surface>`; the
  round-1 design review is why (D3): the first version named frames without the
  mode, so the wide pass overwrote the default pass's frames and the
  default-mode field set was lost.
- `probe_focus_zoom.py` — focuses each field class and reads the scale before
  and after, in both modes: it shows what this host's Chromium does (fit scale +
  stable metrics) and does not do (WebKit iOS focus-zoom).
- `probe_st_second.py` — the one-off DOM dump behind the ask-card finding
  (free-text asks sit on rows the app renders as terminal, so no answerable
  ask-card free-text input materialises in this fixture).
- `before-*` / `after-*` — the same drivers, the same fixture, the same
  viewports, at the base build and at the fixed head.
- `served-assets-checks-*.txt` — served HTML/CSS/JS token scans.

## Which build each artifact was captured on

| artifact | build / head |
|---|---|
| `before-{vp}-*.png` (composer, `default-*`, `wide-*`) | base `7d612a2db` (pre-fix) |
| `before-focus-zoom-report.json`, `before-focus-behavior-report.json` | base `7d612a2db` (pre-fix) |
| `after-{vp}-*.png` (composer, `default-*`, `wide-*`) | head `a0abf8a82` (round-1 remediation) |
| `after-focus-zoom-report.json`, `after-focus-behavior-report.json` | head `a0abf8a82` |
| `before-st-second-card.png` | base `7d612a2db` |

The `after-*` set was first captured at `ae758cbe9`; its non-composer frames
were the overwritten (wide-only) ones the round-1 review flagged (D3) and have
been replaced by this phase-labelled set, re-captured on `a0abf8a82`. The
`before-*` set was likewise re-captured phase-labelled from a throwaway
worktree at the base.

## Frames per mode

Each viewport carries one frame per surface in BOTH modes: `default` (wide view
off — the mode a user gets before opting in, and where the 12/14 → 16px field
change lands) and `wide`. Surfaces: `list` (the session-list search's own
screen), `model-sheet`, `directory-sheet`, `directory-longvalue` (a real long
path typed in — the populated case an empty-with-placeholder frame cannot
show), `pair`, `past`, `projects-create`, `asks-sheet`; `composer-default` /
`composer-wide` are the phone screen in each mode.

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

For the `before-*` set the tree is a throwaway worktree at the base
(`git worktree add --detach <dir> 7d612a2db`), its web bundle built with the
dependency tree APFS-cloned from the feature worktree (`cp -Rc`, never a
re-install), and the fixture/driver run there on the feature worktree's venv
with `PYTHONPATH=<base worktree>` — verify the shadowing first
(`local_operator.__file__` must name the base tree) and that the served CSS is
the base hash, which carries neither new rule.

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
390pt), including the rotation case (boot in landscape with wide on → rotate to
portrait → focus a field), is handed to QA.
