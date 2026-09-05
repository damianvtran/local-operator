# `/model` receipt on a session whose label lands asynchronously

Frames for the fix that derives the receipt's destination from the resolved
spec instead of re-reading `session.model_label` after `set_model`, and for
the cold-viewer refusal that keeps that receipt honest when nothing is bound.

Captured with `shot.py.txt`, which imports the SAME `_AsyncLabelSession` the
regression tests in `tests/unit/tui/test_app_pilot.py` assert against — a
fake whose `set_model` records the request and never moves the label, which is
how `RemoteSession` reads for a terminal attached to another owner's session
until the owner's frontend-state sync arrives. Run from anywhere; the script
locates the repo root from its own path:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/model-receipt/shot.py.txt out.svg        # attached viewer
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/model-receipt/shot.py.txt out.svg cold   # cold viewer
```

- `before.*` — origin/main `0f2e02322`: `model: anthropic/claude-opus-5 →
  anthropic/claude-opus-5 (this session)` for a switch that did happen.
- `after.*` — this branch: `model: anthropic/claude-opus-5 →
  anthropic/claude-fable-5-1 (this session)`.
- `before-cold.*` — origin/main, viewer with no runtime (`is_cold`): the same
  `old → old` line; the request was silently dropped.
- `after-cold.*` — this branch: `no runtime is running for this session; send
  a message to start one, then run /model again` and no receipt, because the
  switch reached nothing (QA round 1, Q4). The boot splash stays, as it does
  for every command that changed nothing (`_system_notice`).

The status band is unchanged between each pair on purpose: it repaints from
the owner's frontend-state sync, which the fake never delivers. The band's cwd
is `/private/tmp` because the script `chdir`s there before painting, so no
worktree path lands in a committed frame.
