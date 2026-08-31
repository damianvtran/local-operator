# Deferred-history band segment removal evidence

The at-rest `▾ N older` deferred-history counter was removed from the status
band by operator decision: the bounded state of a resumed transcript is still
signalled by the in-transcript "older messages above" notice on scroll, so the
band counter is not needed. The paging behaviour itself (bounded first paint,
lazy older pages, in-transcript notice, ctrl+home/ctrl+end) is untouched.

## Frames

| State | Width | Before | After |
| --- | --- | --- | --- |
| bounded resume (150 messages, 72 deferred) | 100×30 | [`before/bounded.svg`](before/bounded.svg) | [`after/bounded.svg`](after/bounded.svg) |
| bounded resume (150 messages, 72 deferred) | 62×24 | [`before/narrow.svg`](before/narrow.svg) | [`after/narrow.svg`](after/narrow.svg) |
| ordinary empty session | 100×30 | [`before/plain.svg`](before/plain.svg) | [`after/plain.svg`](after/plain.svg) |

Before, the bounded band painted `▾ 72 older` in the right group at rest;
after, the segment is absent and the freed cells go to the tail (`!` alarm and
the oldest-mounted-row preview). At 62 columns the ladder sheds cleanly without
the rung. The ordinary session's band is byte-for-byte identical before and
after (the JSON probes diff empty): the segment only ever existed when a
resume held a deferred head.

## Geometry

The JSON beside each SVG records the band's rendered plain text and screen
geometry. At every size `virtual_size` equals `size` and
`show_vertical_scrollbar` is false, so removing the rung neither introduces a
scrollbar nor changes transcript geometry.

Captured with the checked-in real-app harness:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
  docs/assets/pr-drop-older/shot.py OUT.svg 100x30
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
  docs/assets/pr-drop-older/shot.py OUT.svg 62x24
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
  docs/assets/pr-drop-older/shot.py OUT.svg 100x30 --plain
```

The PNG files are browser-rendered inspections of the SVGs, retained as proof
that both exported frames were viewed rather than accepted as markup.
