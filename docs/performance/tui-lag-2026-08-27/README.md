# TUI lag evidence, 2026-08-27

Reproduction and profiling evidence for long transcripts, many subagents, full-screen tool cards, streaming, and canonical frontend-state updates.

## Reproduce

Run the portable benchmark from any directory with the repository interpreter:

```sh
env -u NO_COLOR TERM=xterm-256color \
  /path/to/local-operator/.venv/bin/python \
  /path/to/local-operator/scripts/benchmark_tui_lag.py
```

Capture the real `OperatorApp` frame with:

```sh
env -u NO_COLOR TERM=xterm-256color \
  /path/to/local-operator/.venv/bin/python \
  /path/to/local-operator/scripts/tui_lag_shot.py /tmp/tui-lag.svg
```

The benchmark reports timings but does not enforce machine-dependent thresholds. Unit tests assert the stable structural contracts: snapshots share immutable retained event state, a one-job update carries only its appended event, unchanged updates publish nothing, and repeated frontend notifications coalesce before repaint.

## Measurements

Measured on the same machine and checkout before and after the change:

| Scenario | Before | After |
| --- | ---: | ---: |
| Canonical state read, 100 jobs x 500 events | 60.77 ms median | 0.001 ms median |
| Unchanged scalar mutation, same state | 22.40 ms median | <0.001 ms median |
| Changed scalar mutation, same state | 45.91 ms median | 0.004 ms median |
| Unchanged jobs refresh, 100 x 500 | 29.04 ms median | 0.002 ms median |
| One-job trajectory append, 100 x 500 | not isolated | 0.47 ms maximum |
| Subagent fold, 500 events | 2.18 ms median | 1.37 ms median |
| Mounted band refresh, 100 children | 47.31 ms median | 44.53 ms median |

The portable post-change run additionally measured 1,000-block replay at 773 ms, one-row refresh in a 500-card ledger at 43 ms median, and streaming near 100k/500k characters over 1,000 retained blocks at 114/207 ms median. Profiling attributed those remaining frame costs to Textual whole-container arrange/reflow when content adds rows. Virtualization and reflow redesign were explicitly outside this bounded change; the release removes the frontend-state stalls that compounded those existing render costs without hiding live progress or lowering cadence.

## Rendered evidence

`lop-perf-before.svg` and `lop-perf-after.svg` were captured from the real `OperatorApp` at 120x40 with 92 blocks, including a screen-full tool ledger. Both measured:

- transcript content size: 117x28
- transcript virtual size: 116x105
- screen size and virtual size: 118x38
- transcript vertical scrollbar: visible

Both SVGs were rendered to PNG locally and inspected. Content, spacing, scroll position, and geometry were unchanged; only the checkout path in the status line differed. SVG source is retained because it is the Textual-native artifact and Git can review it as text; the redundant 255 KiB PNG pair is omitted.
