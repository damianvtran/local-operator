# Subagent page: the "history unavailable" latch

Frames behind the fix for a footer that claimed `history unavailable` while the
child's history was on screen.

Both shot scripts drive the real `OperatorApp` through `run_test`, so
`local_operator.tcss` is applied and the footer is the one a user reads. Neither
patches a clock or pokes a flag: the refreshes are the same `show()` the 1 Hz
job poll issues, so what the footer says here is what it says in the product.

Run them (renamed `.py.txt` so the repo's linters skip one-off tooling):

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
    docs/evidence/subagent-history-latch/shot.py.txt <repo-root> out.svg
```

| frame | what it shows |
|---|---|
| `before.svg` / `.png` | `origin/main`. The child's transcript appeared on disk while the page was open; the footer still says `history unavailable` and the durable row is missing. |
| `after.svg` / `.png` | This branch. The same sequence loads the durable row and the footer settles on `start of transcript`. |
| `after-no-directory.svg` / `.png` | The honest negative: a child that never started a durable session keeps `no saved transcript`, unchanged across 12 polls. |

Both `after` frames were re-captured after design round 1 changed the footer
copy. `after-no-directory` **is** finding D1: it previously read `history
unavailable` under a fully rendered trajectory — the same contradiction this
branch exists to remove, reached by a different cause, because a reader maps
"history" onto the rows directly above the note rather than onto the durable
file the page means. `no saved transcript` names what is actually absent and
measures identically (31 cells with `read-only`), so it inherits the old shed
boundary.

`shot.py.txt` reproduces the launch race — `SubagentComms.attach` binds
`session_dir` before the child's first append lands, so the page's opening read
finds a directory with no `transcript.jsonl` in it. `shot_no_directory.py.txt`
covers the case that must NOT self-correct, and prints the distinct footer texts
it observed so a reader can see the note does not flap.
