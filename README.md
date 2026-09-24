# PR #1436 — design review round 1 frames

Real `OperatorApp` (`scripts.visual_capture.save_capture`), worktree at 02cfee996, isolated HOME/config per capture, rendered with `rsvg-convert -w 1400`.

- `real-100x30-*`: a REAL `lop exec --hosting test --workstream --name ws-design-probe "Audit the ingest pipeline for dropped rows"` run (agent-shell env, isolated), copied into a store beside one operator row.
- `basic-*`: 1 workstream + 1 operator row + hidden agent-shell + hidden subagent requester (seeded via mark_session_origin / write_session_title).
- `mixed-*`: 3 workstreams interleaved with 3 operator rows with look-alike titles.
- `empty-*`: no sessions.
