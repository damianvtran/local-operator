# PR #1436 — round-1 remediation evidence

Real `OperatorApp` via `scripts.visual_capture.save_capture`, fresh isolated HOME/config per capture,
store seeded with 5 sessions (operator's own `dddd…`, subagent requester `aaaa…`, agent-shell `cccc…`,
workstream `bbbb…` opened_by coder, workstream `cdcd…` with all-null opened_by). `rsvg-convert -w 1400`.

- `*-before.*`: head `02cfee996` (round-1 reviewed head)
- `*-after*.*`: head `e202ecad2` (remediation, rebased on main `1e629f801`)
- `wire-before.json` / `wire-after.json`: raw bytes of `GET /v1/desktop/sessions?limit=100` from a real
  `lop serve` over the same seeded store (sha256 `7d4b9434…a54f` both).
