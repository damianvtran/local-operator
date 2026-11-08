# PR #3583 — TUI sidebar visual evidence (evidence-only branch)

Orphan branch holding the rendered before/after frames for
https://github.com/damianvtran/local-operator/pull/3583
("background work keeps a session in Running"). Kept off the product
branch per AGENTS.md "Do not commit PR evidence artifacts".

All frames: Textual app SVG export (`app.save_screenshot`) rasterised with
`rsvg-convert`; isolated HOME/config, CMUX_* stripped, synthetic session ids.

- `coder/` — implementer's capture. `before-*` = base, `after-*` = PR head;
  `*-working` = parent turn done with children still running,
  `*-finished` = after the children finish.
- `qa/` — independent QA capture (round 1), `sidebar-{base,head}-{working,finished}`.
