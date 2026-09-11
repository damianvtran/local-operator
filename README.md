# Frames: a harness-injected row painted as the user's words

Evidence for the PR that stops a live-context-only injection from surviving the
compaction rebuild, and stops every human-facing fold from painting one as the
user's own words. Nothing here is loaded by the code or the tests; the frames
exist for the review.

## Capture

`scripts/harness_injection_shot.py` renders both frames — the real `OperatorApp`
(so `local_operator.tcss` applies), the operator's session shape replayed through
the app's own resume path, then the failover receipt posted the way the live
event bridge posts it:

```sh
# after (this branch)
cd <worktree> && env -u NO_COLOR TERM=xterm-256color \
    .venv/bin/python scripts/harness_injection_shot.py /tmp/after.svg

# before (the same script against a pre-fix checkout, where the leaked rows paint)
git worktree add --detach /tmp/lo-before HEAD && ln -s ~/local-operator/.venv /tmp/lo-before/.venv
cd /tmp/lo-before && env -u NO_COLOR TERM=xterm-256color \
    .venv/bin/python scripts/harness_injection_shot.py /tmp/before.svg

rsvg-convert /tmp/before.svg -o before.png && rsvg-convert /tmp/after.svg -o after.png
```

`before-*` was captured at `60273bb79` (pre-fix), `after-*` on the PR head.

## What the pair shows

- **before:** the four `[model switch]` notices a compaction pass baked into
  session `835fbcafdc27` are painted as user rows (the `▌` gutter), directly under
  the operator's own prompt — the frame he reported.
- **after:** those four rows mount nothing; the operator's prompt, the assistant
  line and the failover receipt (the retry notice, the fallback notice, and the
  band naming `Grok 4.6`) are unchanged.

The only other difference between the two images is the band's working-directory
segment, which prints the checkout the frame was captured from.

## Geometry behind the frames

Both frames, from `.geometry.json`:

| reading | before | after |
|---|---|---|
| `css_path` | `['local_operator.tcss']` | `['local_operator.tcss']` |
| `screen.size` | `[118, 44]` | `[118, 44]` |
| `screen.virtual_size` | `[118, 44]` | `[118, 44]` |

`css_path` is the check that the real stylesheet was applied rather than a
default Rich presentation, and `virtual_size == size` on both frames says
neither one made the screen itself scrollable (the `before` frame's scrollbar is
the transcript's own, from the extra content).
