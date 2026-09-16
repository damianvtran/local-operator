# `/links` — rendered frames for the TUI link opener

Emitted by `scripts/link_shot.py`, which drives the REAL `OperatorApp` through
the real command (the line is typed into the editor and submitted), so the
extraction walk, the card and the app's hand-off are all in frame.
`scripts.visual_capture.save_capture` writes the SVG at native 8x17 cells; the
PNGs are that SVG rasterized full-bleed for viewing (macOS `qlmanage`, cropped to
the painted region — the SVGs are the source of truth).

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/link_shot.py docs/evidence/tui-open-links/after-100x30.svg 100x30 default
```

## The frames

| frame | what it shows |
| --- | --- |
| `before-100x30` | `main`, the same conversation, `/links` typed and submitted. It is not a command there, so the line is submitted as a PROMPT (the user row). This is the defect: there is no way to open the URL. |
| `after-100x30` | the card over the same conversation: three URLs — the markdown link, the bare URL and the autolink — newest message first, `❯` on the first row, sender hint right-aligned, footer in the other cards' order. |
| `after-parens-100x30` | the review-round-1 case. ONE row for `[Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar))` with the balanced parentheses intact, and ONE for `**https://a.test/bold**` with the emphasis markers stripped. Before the fix this painted two rows for the first link, with the cursor on the truncated `…/Foo_(bar`. |
| `after-long-140x40` | a URL longer than the card: the row is cut with an ellipsis, the head (the host) is what survives, and the whole URL is what `enter` opens. |
| `after-many-60x20` | more links than fit: an eleven-row window, the newest first, `showing 1–11 of 24`, and the footer still on screen at the bottom. |
| `after-small-44x10` | a terminal with room for one link row: one row, the counter, and the footer — the card never grows past its box. |

## What each frame is evidence OF

The card's geometry, from the capture's own digest (printed by the script):
every frame is a `LinkPickerScreen` with `vscroll=False` — a scrollable Screen is
always a bug in this app — and its card box inside the screen box. At 100x30 the
card is 71x9 in a 98x28 content box; at 60x20 it is 43x18 in a 58x18 one, which
is the whole box and therefore the case the row budget exists for.

## What these frames are NOT evidence of

* **They do not show the terminal's own gesture.** No frame can: cmd+click is
  decided by Ghostty, and the reason it does nothing is that lop holds mouse
  reporting. The hand-off to the opener is asserted in the tests instead, with a
  spy on `local_operator.mcp.auth.open_browser_quietly`.
* **They do not prove the browser opened.** No test and no capture launches one.
* **They are not a font-fidelity claim.** `docs/VISUAL_CAPTURE.md` covers the
  rasterization limits; these are ASCII rows and colour bands, which is what the
  change is about.
