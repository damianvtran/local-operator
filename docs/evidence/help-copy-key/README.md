# The composer copy key in `/help` — rendered evidence

`ctrl+c` copies a highlighted range in the composer, but had no durable
surface. It is not a slash command, so the `/help` command table could not
carry it, and its only advertisement was one row in `welcome.TIPS` — a splash
whose `WelcomeView.display` goes False after the first message. The key was
therefore taught to users who had not started working yet and was unreachable
by the mid-draft user who wants it (#169). `welcome.py`'s own note records that
limit and names `/help` as the place to close it.

These frames are why the row is two lines rather than one. What users get wrong
is not the key but what the SECOND press does: a live range makes every
`ctrl+c` a copy, and the key only returns to the draft and interrupt rungs once
the caret moves and the highlight collapses. A bare "ctrl+c copies" row would
leave a user holding a highlight believing the interrupt had disappeared, which
is why `esc` — which stops the agent whether or not anything is highlighted —
is named on the same row.

`before.svg` is `origin/main` (`75f867bf`): the key reference under the command
table is a single `ctrl+v` row, and nothing in the durable help names the copy
key. `after.svg` is this change: `ctrl+c` sits directly above it in the same
block, left-aligned in the same key column, with its continuation line indented
under the description via the `ljust(name_width)` continuation its neighbours
use. Composed widths are 64 and 71 cells against a **74-cell ceiling** at 80
columns, so neither line wraps to column 0 — the defect the paste note's own
comment in `_help_block` records, and the bound
`test_help_documents_the_composer_copy_key_and_its_release` now pins.

The 74 is measured from the painted compositor, not arithmetic: at 80 columns
the painted row may be 78 cells (the screen's own 2-cell inset) and the block
adds a further 2-cell spine indent, so a composed row of 74 fits and **75 wraps
to two lines**, putting its tail at the key gutter where it reads as another
key row. That is #402's design round 1 D2, the defect `paste_note` was
shortened for. The `cmd+v` row below these sits at exactly 74 with zero
headroom and its comment says so emphatically; this README, that comment, the
`ctrl+c` comment and the test now state the one number in one voice, after an
earlier revision of this change asserted a looser 76 that could not have fired
at the boundary it guards (review round 2, F3).

Capture either side:

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    scripts/help_keys_shot.py OUT.svg 100x30
```

The rows under test sit at the BOTTOM of the help block, below the whole
command table, so the script jumps to the end of the transcript after running
`/help`; a capture without that lands on the commands and misses the key
reference entirely.
