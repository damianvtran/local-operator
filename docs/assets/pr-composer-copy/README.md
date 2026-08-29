# Composer multi-click selection — rendered evidence

Frames exported with `save_screenshot` from the real `OperatorApp` (the host
that loads `local_operator.tcss`), driven with `pilot.click(times=...)` so the
gesture under test is the click chain a terminal actually sends. Regenerate
with:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/assets/pr-composer-copy/shot_composer_copy.py <outdir> <repo-root>
```

`before/` is the same script run against the parent commit, so each pair
differs only by the change under review.

**Both themes are captured, and that is not decoration.** The selection band
set a ground and no ink, so selected text fell back to Textual's near-white
default; on the light ramp that was 1.003:1 against the band and the selection
ERASED the text instead of highlighting it (design round 1, D1). A dark-only
capture cannot show that defect, which is how the first round of frames missed
it. The theme must be passed to the `OperatorApp` **constructor** — it
re-applies its `theme_name` argument over the module-level ramp, so a bare
`set_theme()` beforehand is silently overwritten and every "light" frame comes
out byte-identical to its dark twin.

| File | Gesture | What it shows |
| --- | --- | --- |
| `01-double-click` | double-click inside `ingest` | the word highlighted |
| `02-double-click-ctrl-c` | …then Ctrl+C | `copied 6 characters`, draft intact |
| `03-triple-click` | triple-click | the whole line highlighted |
| `04-triple-click-ctrl-c` | …then Ctrl+C | `copied 32 characters`, draft intact |
| `05-blank-line-double-click` | double-click the blank row between two paragraphs | a live range on the line break (D2) |
| `06-blank-line-ctrl-c` | …then Ctrl+C | `copied 1 character`, **draft intact** — before the fix this frame is the empty composer and `draft cleared — ↑ to recover` |
| `07-trailing-blank-double-click` | double-click the blank LAST row after two shift+enters | nothing painted — **deliberately identical** to the frame before the gesture (D2) |
| `08-trailing-blank-ctrl-c` | …then Ctrl+C | the draft **still there**; on the parent this frame is the empty composer and `draft cleared — ↑ to recover` |

**`07` and `08` are byte-identical to each other on this branch, and that is
the evidence rather than a duplicate.** The last row genuinely has no line
break to take, so the gesture paints nothing and the press that follows is
absorbed: two frames that do not move is exactly what "the draft survived" looks
like here. The pair that carries the finding is `before/08` against `after/08`,
which differ — the parent's is an empty composer carrying `draft cleared — ↑ to
recover` (design review round 2, D2). Contrast R1-7, which was about presenting
two identical captures as two independent observations; these two are one
observation, stated as one.

`light-*` are the same six on the paper ramp. All twelve frames are distinct
files with distinct contents; an earlier evidence set presented two
byte-identical captures as two observations (agent review round 1, R1-7).
