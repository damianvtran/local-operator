# Designer round-1 frames for PR #993

Reviewer: `designer` (round 1), scope `17138cf2..b8d98b2f`.
Captured on the PR worktree `~/workspace/repos/lo-switch-leak` at head
`b8d98b2f`, 2026-09-11, with the app's own resume path and the real
`local_operator.tcss` (`css_path == ["local_operator.tcss"]` in every
`.geometry.json` beside these files).

Nothing here is loaded by the code or the tests; the frames exist for the
review.

## Capture

`capture_design_review.py` (copied here) drives the real `OperatorApp` through
`run_test(size=(120, 46))`, seeds the shape of the reported session, then posts
the events each mode needs:

```sh
cd <worktree>
for m in after control resume firstrow tooladj tooladjcontrol onlyinjections \
         liveorder realreceipt hops4; do
  env -u NO_COLOR -u CMUX_WORKSPACE_ID TERM=xterm-256color \
      .venv/bin/python capture_design_review.py "$m" "/tmp/$m.svg"
done
rsvg-convert -w 960 /tmp/<mode>.svg -o <mode>.png
```

Each mode is captured twice, one `pilot.pause()` apart (`<mode>-settled.svg`);
the pairs are **byte-identical** for every mode, so nothing animates or reflows
on the settle.

## What each frame answers

| frame | question it settles |
|---|---|
| `liveorder.png` | the REAL live adjacency — prompt, four fallback receipts, then the answer: the receipt sits between the prompt and the reply, so it reads as this turn's, not as a floating aside |
| `realreceipt.png` | the receipt's actual wording (`<reason> — falling back to <model>` per hop, then the recovery edge `back to <primary>`), i.e. what the live path paints |
| `hops4.png` | the same cascade as the `retry_start` line would print it (four `retry 1:` rows) — see finding D2 |
| `resume.png` | a REOPENED session whose leaked rows are now hidden: prompt + answer only, no trace of the fallback at all |
| `firstrow.png` | a transcript whose FIRST row is a leaked notice: no leading gap, and the band's provisional title reads the user's real prompt |
| `tooladj.png` / `tooladjcontrol.png` | four leaked rows sitting between a tool card and the assistant line, with and without those rows in the history — **byte-identical**, so hiding leaves no gap and orphans no adjacency |
| `onlyinjections.png` | a transcript whose rows are ALL injections: the app's empty/boot composition (welcome view), not a blank pane |

## The comparison that matters

`after.png` (four hidden leaked rows + receipt) and a matched control (the same
receipt over a history with no leaked rows at all) differ by **nothing**:

```
$ cmp after.png control.png && echo byte-identical
byte-identical
```

Both are 34,241 bytes, 960x782, and their widget trees agree exactly
(`UserBlock@2 h1`, `AssistantBlock@4 h1`, `NoticeBlock@6 h1`,
`NoticeBlock@7 h1`; `screen.size == screen.virtual_size == [118, 44]`;
transcript `scrollbar [false, false]`). The same holds for the tool-adjacent
pair. Hiding a stamped row is a true no-op on the frame — the rows behave as if
they had never existed, which is the strongest available form of "no visual
artifact".
