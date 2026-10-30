# Per-locale rendered frames — relay SPA, `hi` + `es` (#3166)

QA evidence only, on a branch separate from the PR, so no product source is touched.

- Commit under test: `baa9df9ef6108dd5ab0995ca512179a97a4203dd` (PR #3166 head, read from `gh pr view 3166 --json headRefOid`)
- Control commit: `3ed15106efdc184a1fee42175925656606c6b3b7` (`origin/main` at capture time, no `bundled.hi` chunk)
- Method: same rig as #3165's per-locale round (`docs/evidence/locale-frames` on `docs/locale-frames-evidence`), with the
  repo path parameterised (`LF_REPO`), `hi`/`es` markers added to `analyse.py` (quoted from `bundled.{hi,es}.gen.ts`),
  and the fixture printing which `daemon.py`/`dist` it served plus the `bundled.*` assets present.
- Real `MobileDaemon` + `build_app` serving the `vite build` output of each worktree, `language` written via
  `ConfigManager` in an isolated HOME/config (`scripts.probe_isolation`), `dial_registrants=False`, synthetic sessions.
- One headless Chrome (`--use-mock-keychain`, absolute binary path), 390x844 DPR 2, one tab per capture.

`run_*.analysis.json` hold every per-frame number; `rig/` has the scripts and plans.
