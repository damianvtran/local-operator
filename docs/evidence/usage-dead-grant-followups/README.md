# Usage panel: dead-grant follow-ups from #618 round 2 (D6, D7)

Rendered evidence for the two design follow-ups deferred from #618 (issue
#626). Both are narrow-width polish on the `sign-in expired` surface that PR
introduced; the code-side follow-ups in the same PR (R10–R12, Q6) are covered
by mutation-checked tests rather than frames.

`panel_shot.py.txt` drives the REAL `OperatorApp` (so `local_operator.tcss` is
applied — the `_PanelHost` in `test_usage_panel.py` declares no `CSS_PATH`)
and seeds the mounted `UsagePanel` through the same `start_fetch` /
`show_reports` calls the `/usage` worker makes, so every line in the frames
comes out of the production title and note code. Renamed `.py.txt` so the
repo's linters skip one-off tooling.

```sh
cd <worktree>
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/usage-dead-grant-followups/panel_shot.py.txt alibaba 40 out.svg
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/usage-dead-grant-followups/panel_shot.py.txt mixed 48 out.svg
```

`before-*` frames were captured on the clean branch point (`origin/main`
`f4e9613a9`) before any file was touched; `after-*` on this branch. Stills
are the SVG exports rendered with `rsvg-convert` and downscaled 50%.
`rendered-lines.txt` is the plain text of every frame (`render_lines_for_test`),
before and after, including the 48-col alibaba and 56-col mixed frames.

| artifact | width | what it shows |
|---|---|---|
| `before-alibaba-32.png` / `-40.png` | 32, 40 | **D6.** The note row under `alibaba-token-plan-oauth` reads `alibaba-token-plan-oauth` — a verbatim duplicate of the heading, in the attention colour, on the row the reader uses for the remedy. No state, no verb. |
| `after-alibaba-32.png` / `-40.png` | 32, 40 | The same rows read **`sign-in expired`**. The id is still whole on the heading directly above; the state — the one thing no other row carried — is now on the note. 15 cells against the 25-cell floor budget, no ellipsis. |
| `before-mixed-40.png` / `-48.png` | 40, 48 | **D7.** A mixed set (one fresh, one transiently stale, one dead) titles as `· 1 stale · 1…` and `· 1 stale · 1 sign-in…`: the passive clause leads and the actionable one is what the ellipsis eats. |
| `after-mixed-40.png` / `-48.png` | 40, 48 | `· 1 sign-in ex…` and `· 1 sign-in expired ·…`. The actionable clause leads; `stale` reaches the screen only once the expired count is on it whole. |
| `before-mixed-56.png` / `after-mixed-56.png` | 56 | The first width where both clauses fit. Before: `1 stale · 1 sign-in expired`; after: `1 sign-in expired · 1 stale`. Same cells, swapped order — the `wanted` budget is order-independent, so nothing else on the card moved. |

Unchanged, and checked against the text dump rather than assumed: the
per-block notes (`last known 2h ago`, `/login anthropic`), the meters, the
heading rows, and the 48-col alibaba frame, where `/login
alibaba-token-plan-oauth` fits and the last rung is never reached.
