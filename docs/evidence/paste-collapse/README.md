# Evidence: large-paste consolidation in the composer

Captured on macOS (Darwin 25.6.0, arm64) on 2026-09-02, against
`dev-paste-collapse` at 100x30.

| File | What it shows |
|---|---|
| `before.svg` | 500 log lines pasted at `origin/main`: 501 document rows, the user's question scrolled out of the 8-row field |
| `after.svg` | The same paste on this branch: 1 row, `why does this build fail? here is the log:[Paste #1, 501 lines]` |
| `composer_shot.py` | Reproduces both frames (run from the repo root) |
| `verify_end_to_end.py` | Drives the real `OperatorApp`: the collapse decision across six payload classes, and the payload arriving at `session.prompt` |

## What is proven

**The defect is legibility, not performance.** The before-frame is the whole
case: eight rows of anonymous payload tail, with the question the paste was
meant to support scrolled 493 rows out of view. Nothing is slow — measured,
20,000 lines insert in 0.35s and keystroke latency stays flat at ~0.09s,
because `Editor { max-height: 8 }` decouples render cost from document size.

**The payload reaches the model intact.** `verify_end_to_end.py` submits
through the real `on_editor_submitted` path and reads what arrives at
`session.prompt`:

```
composer  : 'why does this build fail?[Paste #1, 501 lines] '
sent chars: 19915  lines: 501
payload intact : True
no chip leaked : True
question kept  : True
```

**The collapse decision, across six payload classes:**

```
500-line ERROR log                    chip=True   rows=1
200-line mypy log (path-prefixed)     chip=True   rows=1
300-file refused drag (D5)            chip=False  rows=300
20000-file refused drag (D5)          chip=False  rows=20000
20000-line find dump (accepted cost)  chip=False  rows=20000
short 3-line snippet                  chip=False  rows=3
```

The third and fourth rows are D5: a multi-file drag that `_attach_pasted_images`
refused must stay readable *whatever its size*, because those paths are what the
user needs in order to see which file was rejected. A drag and a bare-path
listing are indistinguishable by text and by route, so the fifth row is the
accepted cost of honouring D5 — a `find` dump does not collapse. Tool output
that is merely path-*prefixed* is unaffected, because `error:` is not a path.
