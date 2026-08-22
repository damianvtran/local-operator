# /agent + /team polish (0.20.1) — visual evidence

Deferred follow-ups from PR #237's review rounds. Frames are rendered SVGs of
the REAL `OperatorApp` (via `run_test` + `save_screenshot`), not stills from a
CSS-less test host. Capture states are driven through the real `/agent` and
`/team` command paths.

## U2 — persistent active-agent / active-team band segment
- `U2-band-plain.svg` — no attachment: band unchanged from before U2.
- `U2-band-agent.svg` — `◉ auditor` after `/agent auditor`.
- `U2-band-team.svg` — `◫ feature-release` after `/team feature-release`.
- `U2-band-both-120.svg` — both segments: `◫ feature-release › ◉ auditor`,
  placed between effort/model and cwd, cool-blue `signal` ink.

Design call: two segments (not one), placed just after `effort` in the drop
ladder (later = kept longer), team shedding before agent. Both are BOUNDED
rungs capped at `AGENT_PROFILE_CELLS` (20). See the `team`/`agent` rungs in
`status_line.py` for the full rationale.

## D1 — section header outranks its entries (both listings)
- `D1-agent-list-before/after.svg`, `D1-team-list-before/after.svg` — the
  `agents`/`teams` header is now bright bold `fg`; entries stay muted.

## D2 — picker description reclaims slack from an empty detail column
- `D2-picker-theme-before/after.svg` — the win: `/theme` rows with no detail
  (34 of 35) now render fuller descriptions instead of reserving a column only
  the `← current` row fills.
- `D2-picker-agent-after.svg` — `/agent` unchanged (every row carries a
  kind-tag detail, so the shared scan edge is preserved by design).
- `D2-picker-credential/mcp-after.svg` — no regression on other shared-widget
  commands.

Per-row reclaim for a SHORT-but-nonempty detail is deliberately NOT done: it
would break the single scannable left edge the state column exists for
(`/login`, `/theme`, `/mcp logout`). That half stays deferred.

## D4 — noun standardized on "agent"
- `D4-agent-attach-after.svg` — attach notice: "agent auditor is ready…".
- `D4-agent-clear-after.svg` — detach notice: "no agent active; this session
  uses its base instructions." (was "no agent profile active…").

## D3 — narrow-width frames
- `D3-band-both-60.svg` / `D3-band-both-48.svg` — the band sheds team then
  agent gracefully (geometry verified: virtual_size == size, no scrollbar).
- `D3-agent-list-60/48.svg`, `D3-picker-agent-60/48.svg` — listings and picker
  at <80 cols.
