# `/model` receipt on a session whose label lands asynchronously

Frames for the fix that derives the receipt's destination from the resolved
spec instead of re-reading `session.model_label` after `set_model`.

Captured with `shot.py.txt` (run from the repo root as
`env -u NO_COLOR TERM=xterm-256color .venv/bin/python shot.py out.svg`) against
`_AsyncLabelSession`-style fake whose `set_model` lands the label 0.5 s later,
which is how `RemoteSession` behaves for a terminal attached to another owner's
session.

- `before.*` — origin/main `0f2e02322`: `model: anthropic/claude-opus-5 →
  anthropic/claude-opus-5 (this session)` for a switch that did happen.
- `after.*` — this branch: `model: anthropic/claude-opus-5 →
  anthropic/claude-fable-5-1 (this session)`.

The status band is unchanged between frames on purpose: it repaints from the
owner's frontend-state sync, which the fake never delivers.
