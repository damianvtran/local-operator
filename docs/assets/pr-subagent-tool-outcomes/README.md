# Subagent tool-outcome header evidence

Issue #406 reported a running child whose two failed tool calls still summarized as
`2 tools`. These frames drive the real `OperatorApp` with two folded `edit` rows
whose end events carry `is_error=True`.

## Frames

| Width | Before | After |
| --- | --- | --- |
| 100×30 | [`before/normal.svg`](before/normal.svg) | [`after/normal.svg`](after/normal.svg) |
| 62×24 | [`before/narrow.svg`](before/narrow.svg) | [`after/narrow.svg`](after/narrow.svg) |

The before header says `2 tools`; the after header says `2 tools · 2 failed`.
At 62 columns the existing truncation ladder sheds the disposable `coder` role
before losing the outcome, so the label, live status, elapsed time, attempts,
and failures all remain whole.

## Geometry

The JSON beside each SVG records the rendered title and geometry. At both widths
`max_scroll_y` remains 0 and body virtual height equals viewport height (11/11 at
100×30, 9/9 at 62×24), so the longer title neither introduces a scrollbar nor
changes transcript geometry.

Captured with the existing real-app harness:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
  docs/assets/pr-subagent-view-followups/shot.py failed OUT.svg 100x30
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
  docs/assets/pr-subagent-view-followups/shot.py failed OUT.svg 62x24
```

The PNG files are browser-rendered inspections of the SVGs, retained as proof
that both exported frames were viewed rather than accepted as markup.
