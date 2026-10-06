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
- `served_assets_check.sh` + `served-assets-checks-{before,after}.txt` — what the
  fixture actually serves (viewport meta, the focus-zoom tokens, the two CSS
  rules as built). The script writes the SERVED ASSET CONTENT HASHES into the
  file's own header, so each scan proves which bundle it read.

## Which build each artifact was captured on

| artifact | build / head | served bundle (recorded in the artifact) |
|---|---|---|
| `before-{vp}-*.png` (composer, `default-*`, `wide-*`) | base `7d612a2db` (pre-fix) | `index-D_Q6hwPa.css` / `index-BamXnLrM.js` |
| `before-focus-zoom-report.json`, `before-focus-behavior-report.json` | base `7d612a2db` | `"provenance"` field: `index-D_Q6hwPa.css` |
| `after-{vp}-*.png` (composer, `default-*`, `wide-*`) | head `a0abf8a82` (round-1 remediation) | `index-CHPDfYVB.css` / `index-CRwSKFzN.js` |
| `after-focus-zoom-report.json`, `after-focus-behavior-report.json` | head `a0abf8a82` | `"provenance"` field: `index-CHPDfYVB.css` |
| `served-assets-checks-before.txt` | base `7d612a2db` | named in its own header |
| `served-assets-checks-after.txt` | head `a0abf8a82` | named in its own header |
| `before-st-second-card.png` | base `7d612a2db` | — |

## Round-2 correction (provenance)

The first revision of `served-assets-checks-after.txt` was a scan of the round-1
head `ae758cbe9` — `index-CUqFJHkk.css`, and the wide rule as the bare
`calc(16px / var(--lo-fit-scale))` — while this README claimed the `after-*` set
was captured on `a0abf8a82`. The file was byte-identical across the two evidence
commits, so nothing inside it could contradict the claim; that is the defect,
not just the stale numbers. Both `after-*` report JSONs had the same weakness in
kind: they recorded measurements and nothing about the build they came from, so
"regenerated on `a0abf8a82`" was not checkable either (they were in fact
re-run on `a0abf8a82`, and stayed byte-identical because neither change moves a
computed font or the scale — which is exactly why the assertion needed proof
rather than inspection).

Fixed at the source, not in the sentence:

- Both drivers now record the served asset hashes **inside every report**
  (`"provenance": {"served_css": [...], "served_js": [...]}`), read from the
  SPA's own `<link>`/`<script>` at capture time — a report can no longer assert a
  build it did not read.
- `served_assets_check.sh` is now the committed scan and puts the same hashes in
  the file header; it also counts the bare pre-guard rule explicitly (must be
  `0` on the fixed build).
- Every `before-*`/`after-*` artifact was regenerated in a single-build rig run
  (`after-*` on `a0abf8a82`, `before-*` on `7d612a2db`) and re-copied as a set,
  so all six dozen artifacts of a side share one run and one build.

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
LOP_MOBILE_FIXTURE_PASSWORD=$LOP_MOBILE_FIXTURE_PASSWORD \
  docs/evidence/mobile-2017-focus-zoom/served_assets_check.sh 4317 <outfile> "(<label>)" local_operator/mobile/web/src
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
