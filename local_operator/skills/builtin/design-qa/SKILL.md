---
name: design-qa
description: "Deterministic design review for UI/UX on web and terminal screens: measure contrast, spacing, overlap, clipping and copy defects, then judge the rest. Use when reviewing a screen, dashboard, or interface's visual design."
---

# Design QA

A quality gate for user-visible interfaces, web or terminal. Two lanes, in order: measure, then judge. Every claim carries evidence.

## Lane 1: measure

Run the deterministic checks for the surface before forming any opinion, and cite each number's actual source (computed style, bounding box, log line). Never report an unmeasured number; if you cannot measure it, say so.

- Web: `references/checks-web.md`
- Terminal: `references/checks-tui.md`
- Copy: `references/checks-copy.md`

## Lane 2: judge

Judge only what measurement cannot decide: hierarchy (does the eye land in the right place), flow (is the next step obvious), does the screen answer its own question, do the states tell the truth. Judgement stands on the lane 1 numbers. Where you must eyeball, say that you did.

## Matrix: states x viewports

Cover every state the surface has, at narrow, standard, and wide. Themes and personalization axes the product supports (dark mode, display sizes) are extra columns.

| | narrow | standard | wide |
|---|---|---|---|
| loading | | | |
| empty | | | |
| error | | | |
| populated | | | |

For terminal UIs, "viewports" are terminal sizes; include the smallest width the product promises. A single happy-path screenshot is not coverage.

## Procedure

1. Get rendered frames, never source alone. Web: drive the harness `browser` tool (background tab, logins intact; never install or script a browser engine). Terminal: drive the real app that loads the shipped styles, not a bare test host, and export with the app's own capture facility.
2. Run the deterministic checks for the surface; keep each check's real output.
3. Capture a frame per matrix cell. Where anything animates, compare consecutive frames: a first frame that differs from the settled frame is motion the user sees.
4. Form judgements against the brief and the states, not against taste.
5. Report findings with severity and evidence. Where the round format uses D-prefixed ids (D1, D2, ...), keep them stable across rounds and verify fixes only on changed surfaces.

## Honest degrade

When a check cannot run (no browser tool, no render path), report: `static-inspected only, unverified: <the checks that could not run>`. Never let could-not-check read as a pass.

## Red flags

- Installing or scripting a browser engine to get a frame; use the harness `browser` tool or say frames are unavailable.
- A finding with an eyeballed number ("about 12px apart").
- One viewport, one state, or one theme presented as coverage.
- Skipping lane 1 and judging from taste, or skipping lane 2 and reporting only metrics.
- Reviewing source and calling it a design review.
