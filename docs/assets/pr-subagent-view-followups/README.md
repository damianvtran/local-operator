# Subagent-view follow-ups (#405, #407)

Rendered frames for the four deferred follow-ups from PR #404. All four
drive the real `OperatorApp` (so `local_operator.tcss` applies) via
`shot.py`:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/assets/pr-subagent-view-followups/shot.py <dupe|failed|narrow|trunc> out.svg [WxH]
```

- `dupe` (100x30) — two genuine consecutive identical error notices.
  Before: two byte-identical `✗` rows. After: the same two rows, marked
  `(1/2)` and `(2/2)`.
- `failed` (100x30) — two `edit` calls, both errored. Header still reads
  `2 tools` (issue #406, left as the honest start-count; the body already
  shows both failures).
- `narrow` (62x24) — landing scroll position on a wrapping error. Before:
  first transcript line is a hanging continuation (`payload that also
  failed…`) with no glyph. After: the pinned truncation note, then `✗`
  at a row head.
- `trunc` (62x24) — the same overflow scrolled home. The truncation note
  is page chrome, so it is on screen at both the tail (`narrow`) and the
  head (`trunc`).
