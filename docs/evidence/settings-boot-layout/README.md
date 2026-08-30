# `/settings` opened from the boot screen — rendered evidence

Frames for the v0.43.0 report that `/settings` **fails to take over the view**
when it is opened from the boot/splash screen: the boot layout stays applied
underneath it, so the page shares the screen with a docked, width-clamped,
centred input card that belongs to another layout. Depending on the terminal
size that costs the page rows, horizontal span, or both. Captured with
`scripts/settings_shot.py`, which drives the real `OperatorApp` (the only host
that loads `local_operator.tcss`) against a scratch config dir.

The `boot` state is the one this fix added, and it exists because none of the
existing frames could show the bug: every other state seeds a conversation
first, which retires the splash and with it the whole second layout that was
colliding with the page.

Reproduce either side:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/settings_shot.py OUT.svg 140x40 boot
# the two list ends the clamp makes users dwell on:
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/settings_shot.py OUT.svg 100x30 top
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/settings_shot.py OUT.svg 100x30 retired
```

**The collision is not the same shape at every terminal size, and the two pairs
below show different halves of it.** This matters: an earlier version of this
page described both pairs as "compression", which the 100x30 frames do not show
(design round 1, D1).

| file | what it shows |
| --- | --- |
| `before-boot-100x30.svg` | before, and the failure here is **horizontal**: the page gets its full 21 rows, but the input card keeps its width clamp — 73 cells starting at column 12 — so it floats inset and centred over a full-width page |
| `after-boot-100x30.svg` | after: the same 21 rows, but the dock now spans the page (96 cells from column 1) as the plain bar the conversation layout gives it. **The rows are identical on both sides; the card is the whole change** |
| `before-boot-140x40.svg` | before, at a large terminal, where the failure is **vertical** as well: 26 of 38 rows for the page, the card clamped to 94 cells at column 22 and lifted off the bottom by the composition's four reserved rows |
| `after-boot-140x40.svg` | after, at a large terminal: 31 rows instead of 26 — exactly what the same page gets over a conversation — and the whole `Appearance` section is now on screen |
| `after-splash-restored-140x40.svg` | the splash immediately after leaving the page, proving the boot composition comes back intact |
| `before-top-100x30.svg` / `after-top-100x30.svg` | the top of the list after travelling to it by held `up`. Before, the viewport settled one row down and the `Model` header owning the highlighted row sat off screen; after, the header is visible (UX round 1, U1) |
| `before-retired-100x30.svg` / `after-retired-100x30.svg` | the read-only last row the clamp parks users on. Before, the footer advertised `enter change · r default` on a row that honours neither; after, both are shed (UX round 1, U2) |

## The numbers behind the frames

Each capture prints its own geometry. BOTH dimensions are reported, because
reporting rows alone is what made the 100x30 pair look like it showed nothing:
`view.height` is the row count, `boot-card` is the card's width clamp, and
`shell.width`/`shell.x` are what that clamp does. `dock.outer` rather than
`dock.height` carries the composition's reserve — the inner height is 5 on both
sides at every size.

```
                  boot   boot-card  shell.w  shell.x  dock.outer  view.h
before  100x30    True   True        73       12       5           21
after   100x30    False  False       96        1       5           21
before  140x40    True   True        94       22       9           26
after   140x40    False  False      136        1       5           31
after   140x40  over a conversation (the reference)    5           31
```

At 100x30 the boot composition reserves **zero** rows, so the page height is
identical on both sides and the whole defect is the card's clamp and offset —
which is what the operator's screenshot showed. Vertical compression proper
appears at 140x40, where the reserve costs the page five rows.

The screen is never made scrollable on either side
(`virtual_size.height <= size.height`), so the compression at 140x40 was rows
being withheld from the page rather than content overflowing it.

## The restore is exact, not approximate

The boot layout is **suppressed** while the mode is up rather than saved and
re-applied, so the frame taken after leaving has to be checked rather than
assumed. It is byte-identical to the same frame captured at `origin/main`
(`a892aaf5`), at both sizes, once the SVG's generated element ids and the
scratch config path are normalised:

```sh
sed -E 's/terminal-[0-9]+/terminal-X/g' before2/boot-140x40.splash.svg > b.svg
sed -E 's/terminal-[0-9]+/terminal-X/g' after2/boot-140x40.splash.svg  > a.svg
diff b.svg a.svg   # no output, at 100x30 and at 140x40
```

That is the splash, its composition, the card's clamp and the rows reserved
below it all coming back unchanged.
