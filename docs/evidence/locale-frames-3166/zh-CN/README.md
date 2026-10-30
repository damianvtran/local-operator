# Relay SPA rendered frames — `zh-CN` on `main` (post-merge, after #3166)

QA evidence only. No product source is touched.

- Commit under test: `origin/main` = `19f36be40feb604eb8e577d52fd9155d49830b0c`
- Rig: the parent folder's `rig/` (`locale_frames_capture.py` and `locale_fixture.py` unchanged). `zh-CN/rig/analyse.py` only adds
  `zh-CN` markers, quoted from `bundled.zh-CN.gen.ts`. The plans are here.
- Real `MobileDaemon` + `build_app` serving `vite build` of main, with `language: zh-CN` written via `ConfigManager` under
  `scripts.probe_isolation`. Synthetic sessions, `dial_registrants=False`, one headless Chrome (`--use-mock-keychain`) at 390x844 DPR 2.
- Control: the `zh-CN` chunk is held 2500 ms server-side. That is beyond `LOCALE_BOOTSTRAP_BUDGET_MS` (1500 ms).
