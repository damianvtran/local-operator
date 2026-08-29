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

- `before/` — captured from a detached worktree at the branch's merge base,
  `0008afef`. Eleven identical "Invalid arguments" rows; the child's
  instruction, its prose and its tool card are pushed off the top of the page.
- `after/` — the same script on this branch. One error row, with the
  instruction, the assistant's sentence and the running `edit` card all visible.

Both frames land their chrome on identical rows (the hint line, the subagent
panel, the composer and the status line), so the pair differs by the row
collapse and nothing else.

## The capture is deterministic on purpose

Two runs of this script have to produce the same bytes, or the artifact stops
being evidence: a reader cannot tell a real change from a timing difference.
Everything that would otherwise vary is pinned, and each of these was a real
source of drift before it was (review round 2, D6):

- **Every displayed field lives on the JOB, not just in the `show()` call.**
  The app polls subagents once a second and re-`show()`s the page from the job
  itself, so a value passed only as an argument survives exactly one tick and
  the capture then depends on which paint landed last.
- **The elapsed clock is re-anchored on every tick**, because `job_seconds`
  measures a *running* job against `time.time()`.
- **Shimmer is disabled** via `LOCAL_OPERATOR_NO_SHIMMER`, the working line's
  own supported still-frame mode, rather than by reaching into its timers.
- **The spinner is stopped last**, after the final `show()` — every `show()`
  restarts it for a running job.
- **The layout is settled** to a fixed point before the shot, since the body
  applies its rows and its tail scroll over several refreshes.
- **The body is anchored at its tail**, explicitly rather than by asking for a
  scroll. The before-frame overflows its viewport — eleven duplicate rows do
  not fit, which is the point of the artifact — and the page could settle at
  either of two stable offsets, showing 7 or 11 of the duplicates. That is a
  difference in the headline number with no relation to the fix.
- **The live clocks are re-anchored last**, with nothing awaited afterwards:
  the tool card and the working line each run their own `time.monotonic()`
  counter, and anything awaited after pinning them lets the counter tick.

The one thing that legitimately differs between the two frames is the working
directory in the status line: the before-frame is captured from a throwaway
worktree at the merge base and the after-frame from the branch checkout, so
each names its own path. Nothing above the status line depends on it.
