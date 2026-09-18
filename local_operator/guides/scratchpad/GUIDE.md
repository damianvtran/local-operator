---
name: scratchpad
description: Use `scratchpad://` for your own scratch files — notes, data, a one-off script, a benchmark run — instead of writing them into the user's working directory.
---

# Session scratchpad (`scratchpad://`)

`scratchpad://` is your own scratch area for this session, in this session's own
`scratchpad/` folder under the local-operator store. It exists so that
intermediate work has somewhere to live that is not the user's working
directory.

The rule is one line: **your own scratch goes in `scratchpad://`; anything the
user asked for as an output goes in the working directory.** Their tree should
end up holding deliverables and the files they named, and nothing else.

## Use it for

Every kind of text file you would otherwise drop into their tree:

- **A one-off script or snippet** — a `.sh`, a `.py`, a SQL fragment, a
  throwaway harness. Write it here, then run it by the absolute path the tool
  printed (`bash`, `eval` and `grep` all take that path; none of them can resolve
  a scheme).
- **Data you are still shaping** — a `.csv`/`.tsv` extract, a `.json` payload, a
  scratch list of rows you will filter next turn.
- **A benchmark or perf run** — raw numbers, timings, before/after tables. Keep
  the measurements and re-read them instead of re-running the work.
- **Wake and scheduled-run bookkeeping** — what a run did, what it must not
  repeat.
- **A long tool result worth keeping** across turns, when a `spill://` handle is
  not enough because you need to grep, edit or re-read it.

None of this is the user's business, and left in their tree it is litter they
cannot tell apart from output.

## Use something else for

- Anything the user asked for as an output → the working directory, at the path
  they would expect.
- Anything that must outlive this conversation → the scratchpad is this
  session's own folder, and it is deleted with the session.
- Binaries → text only: markdown, JSON, CSV/TSV, TXT, YAML, logs, script
  sources.

## The protocol

Five calls, all through the tools you already have:

| Call | Effect |
|---|---|
| `read` with `scratchpad://` | list the scratchpad (one level) |
| `read` with `scratchpad://logs/` | list one subdirectory |
| `read(path="scratchpad://logs/run.md", range="40-80")` | read a file, or a line range of it |
| `write(path="scratchpad://logs/run.md", content="…")` | create or overwrite it (parent folders are made for you) |
| `edit(path="scratchpad://logs/run.md", edits=[…])` | change it in place with SEARCH/REPLACE hunks |

Every result begins with the URL as you typed it, followed by the resolved
absolute path it maps to (`<url> -> <path>`), so the two can never be confused.
That real path is what `bash`, `ls`, `grep`, `glob` and `eval` need — a shell
cannot resolve a scheme — so use it whenever you step outside the tools. A
listing prints its entries as relative names and carries the resolved folder in
its header.

## File names and types

Give every file a real extension: `perf-2026-09-17.md`, `rows.csv`,
`wake-log.json`, `probe.sh`, `shape.py`. The desktop app opens known extensions
in its canvas and shows a tile for them — markdown as a document, CSV/TSV as a
spreadsheet, `.sh`/`.py`/`.json` in the code editor — and an extensionless file
gets no tile and no viewer. One file per subject; subdirectories are free
(`scratchpad://logs/run.md`).

## Rules the tool enforces

- `..`, absolute paths and dotfiles are refused outright, as are `?` and `#`
  (they open a query or fragment — percent-encode as `%3F`/`%23`).
- One directory level per listing, and the whole folder is bounded, so a wide
  directory cannot flood the transcript.
- Paths are resolved and must stay inside the scratchpad; a symlink pointing out
  is refused rather than followed.
- Any other URL scheme in a path argument is refused: a scheme the tools do not
  own is not silently reinterpreted as a relative path.

## When it is unavailable

Some hosts have no session folder (a `--train` agent directory, a bare tool
context). The tool says so. Use a real temporary directory (`bash mktemp -d`)
and keep the path in your working notes — do not put scratch in the user's
working directory to work around it, which is the litter this protocol exists to
prevent.
