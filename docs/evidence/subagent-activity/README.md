# Subagents band: `thinking` until a child actually streams text

Evidence behind the fix for a band that said `responding` for every running
child the moment its model call began, and kept saying it until a tool started.

Both scripts run against the worktree they are launched from (`sys.path` is
the cwd), and both drive the real code rather than seeded strings: the band
frame's row text comes out of `_make_relay` fed the exact event the loop yields
at the top of a provider call, and the sequence capture runs a real parent
`Session` with a scripted provider stream through the production launch path
(`_launch_subagent` → `run_subagent` → `_make_relay` → `SubagentProgressEvent`).
Renamed `.py.txt` so the repo's linters skip one-off tooling.

```sh
# Band frame: a child whose model call is in flight and nothing has streamed.
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/subagent-activity/band_shot.py.txt out.svg

# Progress-word sequence for a tool-only call followed by a prose call.
.venv/bin/python docs/evidence/subagent-activity/progress_capture.py.txt LABEL
```

| artifact | what it shows |
|---|---|
| `before.svg` / `.png` | `origin/main` (`e0c27ea9`). A child that has only emitted `message_start` — no token streamed — reads **`responding`**. |
| `after.svg` / `.png` | This branch. The same state reads **`thinking`**; nothing else in the band moved. |
| `before-sequence.txt` | The parent-stream progress words on `origin/main`: `responding -> responding -> thinking -> Sleeping briefly -> thinking -> responding -> thinking`. The two leading `responding`s are `message_start` and `message_end` of a call that produced **only a tool call**. |
| `after-sequence.txt` | This branch: `thinking -> thinking -> Sleeping briefly -> thinking -> thinking -> responding -> thinking`. `responding` appears exactly once, on the first text delta, and later deltas add nothing to the stream. |

The trajectory printed above each sequence is the child's own event list from
the same run, so a reader can line each progress word up with the event that
produced it.
