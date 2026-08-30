# Echo de-duplication by message id (#228, #231)

Evidence for replacing text-based user-echo de-duplication with an id-based
registry on both front ends.

## Reproductions

Both scripts probe for the new argument, so they run **unchanged** against
`origin/main` and against the fix — the two arms differ only in the code under
test. Run with the tree on `PYTHONPATH` (the repo `.venv` is editable and would
otherwise resolve `local_operator` to the main checkout):

```sh
PYTHONPATH=<tree> .venv/bin/python repro_231.py
env -u NO_COLOR TERM=xterm-256color PYTHONPATH=<tree> .venv/bin/python repro_228.py <tree>
```

- `repro_228.py.txt` — a TUI steer `continue`, then the phone's own `continue`
  as a distinct message. Base paints 1 row (the phone's message is swallowed);
  head paints 2.
- `repro_231.py.txt` — a phone steer, four assistant rows to push its echo past
  the 3-entry window, then the drain's announcement. Base paints 2 rows (the
  steer is repainted); head paints 1.

## The id contract, against the real engine

`verify_live.py.txt` drives a real `Session` with a scripted stream and observes
the emitted `MessageStartEvent`s, rather than reading the source. It asserts the
property the registry depends on: the id handed to `prompt(message_id=)` and the
id of the object queued by `steer_message` are the ids the announcements carry.

## Frames

Captured from the real `OperatorApp` at 100x30 per AGENTS.md "Visual
validation", so `local_operator.tcss` is applied. `shot_228.py.txt` is the
capture script; it takes an output path and the tree root.

- `frames/before_collision.svg` — base: one `continue` row and its queued
  notice. The phone's message never appears.
- `frames/after_collision.svg` — head: two `continue` rows, each with its gutter
  rule and the correct `gap-above` spacing.
- `frames/after_true_duplicate.svg` — head, the other direction: the steer's own
  delivery follows, the notice settles to `sent — the agent has it now`, and no
  third row is added.
