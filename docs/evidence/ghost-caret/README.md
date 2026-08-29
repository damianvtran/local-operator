# Testing evidence: the ghost caret returns to the insertion point

Rendered frames and geometry for the fix to #378's `_paint_ghost_caret`, which
moved the composer's block caret one cell LEFT so it sat on the last character
the user typed instead of on the ghost's first cell.

## The defect, as a user reported it

With `/resume` fully typed the only completion is the trailing space, so the
ghost is `" "`. #378 painted the block one cell left, over the `e` of `resume`.
A block ON a committed character is the vi-normal-mode / overwrite idiom, so
the frame said "your next keystroke replaces this `e`" about text that was
fully committed and a caret that genuinely belonged after it.

The contradiction the user hit is between the composer's GHOSTED and UN-GHOSTED
states, not between `/resume` alone and `/resume` mid-draft. Any buffer with no
completion on offer — ordinary prose, or a command the width gate refuses —
puts the caret after the text, where it belongs; every buffer carrying a ghost
pulled it one cell back. `hi /resume` is a ghosted state too (its completion is
also the lone trailing space), so it was displaced as well and is fixed the same
way. See the note under the frame table.

## Research verdict (the rationale for the rule)

Every autosuggestion implementation surveyed keeps the caret at the TRUE
insertion point, which is exactly where the ghost starts, and with a block
cursor the block therefore lands on the ghost's first character:

| system | where the caret sits |
|---|---|
| fish | suggestion rendered after the cursor, dim `fish_color_autosuggestion` |
| zsh-autosuggestions | `CURSOR == $#BUFFER`, ghost lives in `POSTDISPLAY` |
| prompt_toolkit `AppendAutoSuggestion` | ghost appended as fragments, cursor stays at `cursor_position` |
| Textual `Input.render_line` | builds `value + suggestion[len:]`, then applies the cursor style ON TOP of the first ghost cell |
| VS Code / Copilot, JetBrains | ghost after a bar caret at the model position |

There is no precedent, terminal or GUI, for retreating the caret a cell. Browser
URL bars use the OTHER coherent convention (insert the completion and SELECT
it, which does mean "the next keystroke replaces this") — a displaced block
caret is neither convention.

## Frames

Captured with `shot.py` against the REAL `OperatorApp` (it declares `CSS_PATH`,
so `local_operator.tcss` applies — the lightweight test hosts do not load it and
cannot show a style change). BEFORE is `origin/main` at **`0008afef`** in a
detached worktree; AFTER is this branch rebased onto that same commit, so both
sides share a baseline.

The frames were originally shot against `e641a888` and re-shot after rebases
picked up #388 (double/triple-click composer selection, which touches
`editor.py` and `local_operator.tcss`), #401, and the 0.43.3 release commit. Re-shooting rather than
carrying the old stills forward is the point: a baseline that predates a change
to the composer's own stylesheet cannot prove what the composer renders now. The
defect reproduces identically on both baselines — same ghost, same caret column
in all seven states — so the rebase changed nothing this evidence rests on.

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/ghost-caret/shot.py . /tmp/ghost-after
```

`caret-*.png` are the load-bearing frames: the composer row cropped to the
caret cell at 8x, before stacked above after. The full-screen
`before-*` / `after-*` pairs are there to show nothing else on the screen moved.

`native-*.png` are the same three multi-character states cropped at **1x**, so
the dimmed cell can be judged at the size a user actually sees rather than at
the 8x that flatters it (design review round 2, D4). The covered glyph survives:
`e` in `/resum`, `g` in `/mcp lo` and `o` in `/mcp login n` are each identifiable
at native scale and each still reads as preview rather than as committed text.
Read the 8x crops for the one-cell caret move, and these for whether the result
is legible in practice.

| frame | state | what it shows |
|---|---|---|
| `caret-resume-full.png` | `/resume`, picker open | The user's report. BEFORE: block over the `e`. AFTER: gate 4 withholds the whitespace-only ghost and the block sits after the text, on a blank — the ordinary end-of-line caret. |
| `caret-resum-partial.png` | `/resum` → ghost `e ` | BEFORE: block on the typed `m`, previewed `e` dim beside it. AFTER: block at the insertion point on the `e`, and that `e` is still dim ink. |
| `caret-mcp-lo.png` | `/mcp lo` → ghost `gin` | Enum-tail ARGUMENT slot, a different completion path. Block moves from the typed `o` to the previewed `g`. |
| `caret-mcp-login-n.png` | `/mcp login n` → ghost `otion` | The compound server row, the hardest completion to predict. Block moves from `n` to the previewed `o`. |
| `caret-mid-draft.png` | `hi /resume` | **Also fixed, not a control.** A command token with a draft in front of it yields the same whitespace-only ghost, so it carried the same one-cell displacement (col 9 → col 10) and gate 4 withholds it the same way. The user called this state correct because, with the ghost withheld, it now looks the way they expected — see the note below. |
| `before/after-narrow-width.png` | `/analytic` at 18 cols | Gate 2 withholds the ghost (content box 10 cells, `/analytic` + `s ` needs 11). Unchanged. |
| `before/after-prose-no-ghost.png` | `hello there` | No command token, no ghost. **The genuine untouched control**, pixel-identical before and after. |

**What the two whitespace-ghost frames can and cannot show (design review round
2, D1).** In `caret-resume-full.png` and `caret-mid-draft.png` gate 4 withholds
the ghost, so the after panel is a block on an empty cell — the correct render,
and the one the user asked for. As EVIDENCE, though, those two frames cannot by
themselves distinguish "the whitespace ghost was suppressed and the caret moved
to the insertion point" from "the completion engine returned nothing at all":
the proof that the caret moved lives in the geometry table below (col 6 `'e'` →
col 7 `' '`), not in the pixels. The three multi-character states
(`resum-partial`, `mcp-lo`, `mcp-login-n`) carry the visual argument on their
own, because there the block visibly steps one cell right onto a glyph that
stays dim. Read the five as two different demonstrations, not five equivalent
ones.

**On `mid-draft`, corrected (design review round 1, D1).** An earlier version of
this README and of the PR body called this state "the one the user calls
correct" and offered it as the control proving the fix is scoped to the picker
case. That was wrong, and the geometry table below always recorded the truth:
`hi /resume` produces the whitespace-only ghost too, so the caret was
displaced there as well (col 9 → col 10). What the user actually reported is
that the mid-draft state *looked* right — which it now genuinely is, for the
same reason as every other ghosted state. The states this change does not reach
are `prose-no-ghost` (no command token) and `narrow-width` (gate 2 already
withholds the ghost); those are the controls.

`shot.py` also saves a `.frame2.svg` per state one settle later; if the settled
frame differed from the first painted one, the row would be reflowing where the
user can see it. All 14 (7 states x 2 trees) are byte-identical. The SVGs
themselves are not committed — they are duplicates of frames already here — but
the check is no longer an unverifiable claim: `settle-frames.txt` records the
per-state result and the one-line loop that regenerates it (design review round
2, D5).

### The pairs are directly diffable (design review round 1, D3/D4/D5)

The two sides are shot from different worktrees, so the app's real cwd differed
between them and appeared in BOTH the banner and the status line. Every pair
therefore differed for a reason unrelated to the change, and at the 18-column
`narrow-width` size the two paths wrapped to different line counts and pushed
the composer down by 434px, so that pair could not be compared as-shot at all.

`shot.py` now pins `os.getcwd()` to one string for the capture, which is the
honest fix rather than cropping the banner away: the frames still show the real
surface, and any remaining difference within a pair is attributable to the
change. Measured with `magick compare -metric AE` on the committed PNGs:

| pair | differing pixels |
|---|---|
| `narrow-width` | **0** |
| `prose-no-ghost` | **0** |
| `resume-full` | 1409 |
| `resum-partial` | 1299 |
| `mcp-lo` | 1045 |
| `mcp-login-n` | 1371 |
| `mid-draft` | 1411 |

The two controls are now byte-identical, so "diff the pair and see nothing move"
works as the check the full-screen frames exist for. The five that differ are
exactly the states carrying a ghost, and the difference is confined to the
composer row.

`caret-mid-draft.png` was also re-shot with a shorter draft prefix (`hi ` rather
than `check this `), because at the longer seed the caret cell fell on the right
edge of the crop window and was clipped in the before panel and lost entirely in
the after one (D5). The state under test — a command token with a draft in front
of it — is unchanged by the shorter prefix.

## The ink of the cell under the block

Judged from the rendered frames, not from the style code, and it needed a fix.
Textual applies `text-area--cursor` over the whole caret cell, so with the caret
back at the insertion point the ghost's first character was painted in the
caret's full ink — measured `#1e1a14` on `#e9e5db`, **13.76:1**, byte-identical
to a caret on a character the user actually typed. In `caret-resum-partial.png`
that showed as a solid black-on-cream `e` while its own trailing space stayed
dim: the committed/previewed boundary landed INSIDE the ghost.

`Editor._paint_ghost_ink` re-inks that ONE cell: the caret keeps its background
(the block is load-bearing — the boot composer, the read-only composer and the
attachment chip all assert an inverted caret cell) and the suggestion's own
foreground goes back on top. That is **3.29:1** against the cursor ground —
legible, and visibly not the 13.76:1 of committed text — so the cell reads as
"caret, on previewed text". The caret does not move.

## Geometry behind the pixels

`geometry-before.json` / `geometry-after.json`, written by the same run that
saved the frames. The **column** of the cell carrying the cursor background is
the number that matters; a strip's `cell_length` cannot show it, because the
composer pads the row out to its box width (69 cells at w=100 in every state,
ghost or no ghost, before and after).

| state | buffer | ghost before | caret cell before | ghost after | caret cell after |
|---|---|---|---|---|---|
| `resume-full` | `/resume` | `' '` | col 6 `'e'` | `''` | col 7 `' '` |
| `resum-partial` | `/resum` | `'e '` | col 5 `'m'` | `'e '` | col 6 `'e'` |
| `mcp-lo` | `/mcp lo` | `'gin'` | col 6 `'o'` | `'gin'` | col 7 `'g'` |
| `mcp-login-n` | `/mcp login n` | `'otion'` | col 11 `'n'` | `'otion'` | col 12 `'o'` |
| `mid-draft` | `hi /resume` | `' '` | col 9 `'e'` | `''` | col 10 `' '` |
| `narrow-width` | `/analytic` (w=18) | `''` | col 9 `' '` | `''` | col 9 `' '` |
| `prose-no-ghost` | `hello there` | `''` | col 11 `' '` | `''` | col 11 `' '` |

In every ghosted state the caret column moves from `len(buffer) - 1` to
`len(buffer)`, the insertion point. The two no-ghost states are untouched, which
is what pins that this change reaches only the ghosted path.

Screen geometry is unchanged between the trees, checked directly rather than
inferred from the stills: editor content box `(69, 1)`, screen `(98, 22)`,
`virtual_size` `(98, 22)` — equal to the screen, so nothing made it scrollable —
and `show_vertical_scrollbar` False, in both trees over three consecutive runs.
