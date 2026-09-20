# Design review frames — local-operator PR #1341 (round 1)

Captured on 2026-09-19/20 for the design round on PR #1341 (head `2683360e`).

- `after-*.png` — the branch's own build (`pnpm build` in a detached worktree of
  `2683360e`), served from a sandbox static server + mock daemon API and rendered
  in the operator's browser inside a 390x844 CSS px iframe (dpr 2). `after-390x844-dark.png`
  is stitched from two pans of the frame, because the browser viewport is 720 CSS px tall.
- `before-degenerate-390x844-dark.png` — the same app bytes with a stylesheet built
  without the `@source` globs (31,052 B, md5 `c948fe74cbb3bdb474195065372bbd92`, byte-identical
  to a build of the pre-fix sources in an installed-shaped tree).
- `harness/` — the mock server and the phone-frame page used for the captures; the
  reproduction is: `python3 harness/mockserver.py <port> <dist-dir> <mode-file>` then
  open `http://127.0.0.1:<port>/__frame.html?off=0&w=390&h=844`.

The operator's live daemon (port 4098) was never contacted; no `lop-update` was run.
