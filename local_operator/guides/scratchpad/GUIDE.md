---
name: scratchpad
description: Use `scratchpad://` for your own scratch files — notes, data, a one-off script, a benchmark run — instead of writing them into the user's working directory.
---

# Session scratchpad (`scratchpad://`)

`scratchpad://` is your own scratch area for this session, in this session's own
`scratchpad/` folder inside the session directory. It exists so that intermediate
work has somewhere to live that is not the user's working directory.

The rule is one line: **your own scratch goes in `scratchpad://`; anything the
user asked for as an output goes in the working directory.** Their tree should
end up holding deliverables and the files they named, and nothing else.

The test to apply in one line: is this file for YOU to keep working from, or is
it the answer the user asked for? Working material goes here; the answer does
not.

## Use it for

Text, and rendered frames:

- **A one-off script or snippet you write for yourself** — a `.sh`, a `.py`, a
  SQL fragment, a throwaway harness. Here, and only here, when it is your own
  means to an end: a script the user asked you to deliver is an output and goes
  in the working directory. Run it by the absolute path the tool printed
  (`bash`, `eval` and `grep` all take that path; none of them can resolve a
  scheme).
- **Data you are still shaping** — a `.csv`/`.tsv` extract, a `.json` payload, a
  scratch list of rows you will filter next turn.
- **A rendered frame or a still** — a PNG, a JPEG/GIF/WebP, or a screenshot of a
  UI under test. These are first-class here, not a special case: a raster image
  written into the pad by `bash` reads back through the scheme as a VIEWABLE
  image. (An SVG is stored here just as happily, but the reader does not decode
  vector formats — it comes back as its own text, so render it to a PNG first if
  what you need is to LOOK at the frame. The case that really needs a temp dir is
  narrower and is set out below.) The standing rule for
  any user-visible change is before/after frames, so the pad is where the
  capture script, the frames and the numbers behind them belong together.
- **A benchmark or perf run** — raw numbers, timings, before/after tables. Keep
  the measurements and re-read them instead of re-running the work.
- **Wake and scheduled-run bookkeeping** — before a scheduled run, read
  `scratchpad://wake-log.json` first and decide from it instead of re-running the
  whole check; add each item to it as you report it, so the next wake does not
  repeat it.
- **A long tool result worth keeping** across turns, when a `spill://` handle is
  not enough: a handle is read-only and cannot be edited in place. Read it and
  write what you need into `scratchpad://`, then work on it there.

None of this is the user's business, and left in their tree it is litter they
cannot tell apart from output.

## Use something else for

- Anything the user asked for as an output → the working directory, at the path
  they would expect.
- Anything that must outlive this conversation → not here — nothing here outlives
  it. The store is deleted with the session (it lives INSIDE the session's own
  directory, so deleting the session takes the whole folder, and the automatic
  cleanup pass does too when its policy is switched on). An output that has to
  survive belongs in the working directory, or wherever the user wants it.
- **A NON-image binary** (an archive, a model file, a `.bin`) → not here: it
  has no text to return, so the reader refuses it as text and there is nothing
  useful to do with it. (`bash mktemp -d` with NO template, which lands in
  `$TMPDIR`, the per-user temp directory, is the home for those.) An IMAGE is
  not in this class — measured 2026-09-21: a PNG written into the pad by `bash`
  reads back through the scheme as a viewable image, so rendered frames belong
  here with everything else. Do NOT pass a template that carries
  the `/tmp` directory in its path — that is what puts you back in the one
  directory macOS reaps. `/tmp` belongs to the system cleaner:
  `/usr/libexec/tmp_cleaner` (launchd `com.apple.tmp_cleaner`, run daily)
  prunes entries under `/tmp` older than three days, and `$TMPDIR` is not on
  that list — so scratch there can be deleted out from under a session that is
  still running, while `$TMPDIR` survives it. Then write that absolute path into
  the scratchpad itself (`scratchpad://dirs.txt`): the path is what `bash` and
  `read` need to reach the files, and a path that lives only in an old
  transcript is a path already lost.

## The protocol

Five calls, all through the tools you already have:

| Call | Effect |
|---|---|
| `read` with `scratchpad://` | list the scratchpad (one level) |
| `read` with `scratchpad://logs/` | list one subdirectory |
| `read(path="scratchpad://logs/run.md", range="40-80")` | read a file, or a line range of it |
| `write(path="scratchpad://logs/run.md", content="…")` | create or overwrite it (parent folders are made for you) |
| `edit(path="scratchpad://logs/run.md", edits=[…])` | change it in place with SEARCH/REPLACE hunks |

A result that succeeds names the URL as you typed it and the resolved absolute
path it maps to (`<url> -> <path>`), so the two can never be confused, and a
write's verb says whether it created or overwrote the file. That real path is what
`bash`, `ls`, `grep` and `eval` need — a shell cannot resolve a scheme — so use
it whenever you step outside the tools. A listing prints its entries as relative
names and carries the resolved folder in its header.

**You do not have to write a file first to learn that path.** `bash` and the
`eval` kernel are each given this session's pad as `$LOCAL_OPERATOR_SCRATCHPAD`,
an absolute path, so a shell can create directly into it:
`mkdir -p "$LOCAL_OPERATOR_SCRATCHPAD/logs"`, `> "$LOCAL_OPERATOR_SCRATCHPAD/x.log"`.
The folder is created for you the first time anything needs it — `write`/`edit`
make it, and a shell call makes it before the command runs — so a bare redirect
and a `mktemp` template both work on their first use, with no `mkdir` first.
The idiom for a rig that wants a private subdirectory is
`mktemp -d "$LOCAL_OPERATOR_SCRATCHPAD/rig.XXXXXX"` — the template keeps the
directory inside the pad, so the files stay readable through the scheme and
survive the session, which a `/tmp` one does not. The name is set only inside an
agent's own shell and kernel; it is unset in the user's terminal, where the
`scratchpad://` calls are the way in.

## File names and types

Give every file a real extension: `perf-2026-09-17.md`, `rows.csv`,
`wake-log.json`, `probe.sh`, `shape.py`. The desktop app opens known extensions
in its canvas and shows a tile for them — markdown as a document, CSV/TSV as a
spreadsheet, `.sh`/`.py`/`.json` in the code editor — and an extensionless file
gets no tile and no viewer. One file per subject; subdirectories are free
(`scratchpad://logs/run.md`).

## Rules the tool enforces

- `..`, absolute paths and dotfiles are refused outright, as are `?` and `#`
  (they open a query or fragment — percent-encode as `%3F`/`%23`). A name
  containing `://` is refused too: a second scheme inside a name is a URL, not a
  file name.
- One directory level per listing, and a listing is bounded, so a wide directory
  cannot flood the transcript. Reading a very large file comes back truncated,
  with the `read` call that continues it.
- The folder itself is NOT size-capped: a single write is not refused for its
  size, so keep the files to what you need. Nothing caps it later either — the
  only thing that removes it is the session going away, by the cleanup pass (when
  its policy is switched on) or by the session being deleted.
- Paths are resolved and must stay inside the scratchpad; a symlink pointing out
  is refused rather than followed.
- Any other URL scheme in a path argument is refused: a scheme the tools do not
  own is not silently reinterpreted as a relative path.

## When it is unavailable

Some hosts have no session folder (a `--train` agent directory, a bare tool
context). The tool says so. Use a real temporary directory — `bash mktemp -d`
with NO template, so it lands in `$TMPDIR` and not in `/tmp` (see above) — and
keep the absolute path in your working notes; do not put scratch in the user's
working directory to work around it, which is the litter this protocol exists to
prevent.
