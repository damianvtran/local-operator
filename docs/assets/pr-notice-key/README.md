# Duplicate subagent notice rows

Rendered frames for the fix in `fold_trajectory`: one error notice on a child
at `TRAJECTORY_CAP` re-printed itself on every refresh, because the row key was
the event's offset in a window that evicts from the front.

Both frames come from `shot.py`, which drives the real `OperatorApp` (so
`local_operator.tcss` applies), seeds a trajectory sitting exactly at the cap,
and then simulates ten 1 Hz refreshes with front-eviction between them:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python docs/assets/pr-notice-key/shot.py out.svg
```

- `before/` — captured from a detached worktree of `origin/main` @ 0ed1051e.
  Eleven identical "Invalid arguments" rows; the child's instruction, its prose
  and its tool card are pushed off the top of the page.
- `after/` — the same script on this branch. One error row, with the
  instruction, the assistant's sentence and the running `edit` card all visible.

The header clock is pinned to a fixed offset from `time.time()` so the pair
differs only by the fix under test.
