# `/settings` opened from the boot screen — rendered evidence

Frames for the v0.43.0 report that `/settings` **compresses** when it is opened
from the boot/splash screen instead of taking over the view. Captured with
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
```

| file | what it shows |
| --- | --- |
| `before-boot-100x30.svg` | before: the page squeezed into the rows above the boot input card, which keeps its clamp and its centred position |
| `before-boot-140x40.svg` | before, at a large terminal, where the failure is plainest: 26 of 38 rows for the page, the card floating mid-screen, dead space beneath it |
| `after-boot-100x30.svg` | after: the page takes the view and the dock is the plain full-width bar the conversation layout gives it |
| `after-boot-140x40.svg` | after, at a large terminal: 31 rows instead of 26, which is exactly what the same page gets over a conversation, and the whole `Appearance` section is now on screen |
| `after-splash-restored-140x40.svg` | the splash immediately after leaving the page, proving the boot composition comes back intact |

## The numbers behind the frames

Each capture prints its own geometry. `view.height` is the row count the bug is
about; `boot` is whether the second layout is still applied.

```
before  140x40   boot=True   dock.height=5  view.height=26   screen.height=38
after   140x40   boot=False  dock.height=5  view.height=31   screen.height=38
after   140x40   over a conversation (the reference)  view.height=31
```

The screen is never made scrollable on either side
(`virtual_size.height <= size.height`), so the compression was rows being
withheld from the page rather than content overflowing it.

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
