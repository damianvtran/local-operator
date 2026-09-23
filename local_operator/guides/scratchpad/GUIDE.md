---
name: scratchpad
description: Use `scratchpad://` for your own scratch files — notes, data, a one-off script, a benchmark run — instead of writing them into the user's working directory.
---

# Session scratchpad (`scratchpad://`)

`scratchpad://` is your own scratch area for this session, in this session's own
`scratchpad/` folder inside the session directory. It exists so intermediate work
has somewhere to live that is not the user's working directory.

The rule is one line: **your own scratch goes in `scratchpad://`; anything the
user asked for as an output goes in the working directory** — their tree should
end up holding the files they named and nothing else. Ask of each file: is it
for YOU to keep working from, or the answer they asked for?

## Use it for

Text, and rendered frames:

- **A one-off script or snippet you write for yourself** — a `.sh`, a `.py`, a
  SQL fragment, a throwaway harness. Here, and only here, when it is your own
  means to an end: a script the user asked you to deliver is an output and goes
  in the working directory. Run it by the absolute path the tool printed
  (`bash`, `eval` and `grep` all take that path).
- **Data you are still shaping** — a `.csv`/`.tsv` extract, a `.json` payload, a
  scratch list of rows you will filter next turn.
- **A rendered frame or a still** — a PNG, a JPEG/GIF/WebP, or a screenshot
  of a UI under test. These are first-class here: a raster image written into
  the pad by `bash` reads back through the scheme as a VIEWABLE image. (An SVG
  is stored here too, but the reader does not decode vector formats — it comes
  back as its own text, so render it to a PNG first if what you need is to LOOK
  at the frame. The case that needs a temp dir is below.) The standing rule for
  a user-visible change is before/after frames, so the capture script, the
  frames and the numbers behind them belong here together.
- **A benchmark or perf run** — raw numbers, timings, before/after tables. Keep
  the measurements and re-read them instead of re-running the work.
- **Wake and scheduled-run bookkeeping** — before a scheduled run, read
  `scratchpad://wake-log.json` first and decide from it; add each item to it as
  you report it, so the next wake does not repeat it. The pad rides out a
  restart, so a wake finds the bookkeeping it left here.
- **A long tool result worth keeping** across turns, when a `spill://` handle is
  not enough: a handle is read-only and cannot be edited in place. Write what you
  need into `scratchpad://` and work on it there.

Left in their tree, this is litter they cannot tell apart from output.

## Use something else for

- Anything the user asked for as an output → the working directory, at the path
  they would expect.
- Anything that must outlive this conversation → not here; an output that has
  to survive belongs in the working directory, or wherever the user wants it.
  But do not confuse that with the pad being EPHEMERAL — it is not (measured
  2026-09-22: a session that read its lifetime as "like a temp directory" kept
  a duplicate copy of its state outside the pad for an hour). The pad lives
  INSIDE the session's directory, so it **survives runtime restarts and
  rollovers**, and it is cleared only when the session is deleted or expires —
  deleting the session (or the cleanup pass, when its policy is on) takes the
  whole folder. A long-running loop or a multi-turn state file has no reason to
  keep a second copy anywhere else.
- **Do not put these here: build trees, dependency trees, compiled artefacts,
  archives, anything the shell built.** `write`/`edit` refuse them by NAME — a
  `node_modules`/`target`/`dist`/`out` directory, a
  `*-build`/`_build`/`*-cache`/ `cmake-build-*`/`bazel-*` tree, a
  `libfoo.so.1.2`, a `foo.tar.gz` — and refuse a write that would take the
  WHOLE pad past its cap. Build it in a git worktree instead (`git worktree add
  <path>`), where the output is wanted and can be rebuilt from the commit
  rather than carried as bytes. A pad is the wrong home for it both ways: it is
  billed to a disk shared with every other session, and it ends with the
  session. **This rule is enforced at the TOOLS and not in a shell.** `write`
  and `edit` carry their payload inline and are checked; `bash` and the `eval`
  kernel get this pad as a path (`$LOCAL_OPERATOR_SCRATCHPAD`) and are NOT
  policed, so a `pnpm install`, a `cargo build` or a redirect into the pad
  still succeeds and still lists back through the scheme. The refusal is a
  nudge at the tool surface: it cannot undo what a shell has already put there.
- **A NON-image binary** (an archive, a model file, a `.bin`) → not here: it
  has no text to return, so the reader refuses it as text. (`bash mktemp -d`
  with NO template lands in `$TMPDIR`, the per-user temp directory; a template
  carrying `/tmp` puts you back in the directory macOS reaps, where
  `/usr/libexec/tmp_cleaner` (launchd `com.apple.tmp_cleaner`, daily) prunes
  entries older than three days — and `$TMPDIR` is not on that list, so scratch
  there can be deleted out from under a running session.) An IMAGE is not in
  this class: measured 2026-09-21, a PNG written into the pad by `bash` reads
  back through the scheme as a viewable image, so rendered frames belong here
  too. Then write that absolute path into the scratchpad itself
  (`scratchpad://dirs.txt`): a path that lives only in an old transcript is
  already lost.

## A directory named `tmp` or `scratch` is not scratch space

The `[scratch]` advisory fires for a second shape, the one this protocol's own
convention makes easiest to fall into: a file whose PARENT directory is named
`tmp`, `.tmp`, `temp`, `scratch`, `.scratch`, `scratchpad` or `.scratchpad` —
inside the working directory, a repository, or beside one. The name is the
whole test, in any case, and it has no extension gate. A file dropped into such
a folder follows the convention whether or not anyone decided to.

**The shell channel needs the path NAMED, not related.** A bare `> tmp/x.md` is
relative, and the scan has no working directory to resolve it against, so it goes
unnoticed — while the same write through `write`/`edit` IS resolved and does fire.
A home path is not that case: a LEADING `~/`, `$HOME/` or `${HOME}/` is expanded
to the real home directory before the scan looks at the target, so it fires
exactly as its absolute spelling does. Leading alone, and the three spellings
only — `~other/tmp/x.md` is another user's home and is left alone, as is a `~`
that is not at the front. The expansion reads the token, not its quoting, exactly
as the `$TMPDIR` spelling always has: `'~/x'` is a literal name to the shell and
is expanded here anyway, so the line can name a file the command did not create.
Spell the whole path and both channels see it.

A scratch-named directory inside a temp directory is not this advisory's
business: `/tmp` and `$TMPDIR` are the system's own area, and anything deeper
is a build or rig folder. Only a file DIRECTLY in one of them gets a line of
its own, and the two lines differ — `/tmp`'s names the cleaner that reaps it,
`$TMPDIR`'s says only that it is not this session's own area.

Three things are wrong with it:

- **It is not session-scoped.** Nothing removes it when the session ends, and it
  is on no cleaner's list — so what lands there is still there tomorrow, and still
  there for the next session, which never wrote it and cannot tell whose it is.
- **It is inside the user's tree**, whose only job is to hold deliverables and the
  files they named; a session's working file in there is litter wearing the wrong
  label.
- **The user cannot tell it apart from their output.** A file in a folder called
  `tmp` reads as something they made.

A folder of that name INSIDE your own pad is a different thing and the advisory
stays silent for it: the pad is session-scoped, and `tmp/` in there is the pad.
Containment, not the name, is what decides.

**The honest limit.** This is one line appended to a tool result, and a line is
not a rule: it cannot refuse the write, and it cannot clean up what is already
there. Nothing sweeps a `tmp`-named folder in the user's tree, so moving earlier
scratch — possibly your own — out of one is yours to do.

## A file another agent or session must read

Your pad is per-session and another session cannot read it — but the PATH can
be handed over: write the file to the pad and pass a subagent the absolute path
the result prints, and the same holds for a path a later turn of your OWN
session needs. The pad is private and the path is ordinary, and it is the path
— not the pad — that crosses the boundary.

The opposite gesture looks identical and is the mistake to watch: handing a
path to something that has to outlive this session. The pad is session-SCOPED,
not ephemeral, so a file that must outlive the session is an output and goes
where the user asked for it. A `tmp`-named folder in their tree is neither: it
is not the pad, so nothing takes it away, and it is not where they asked, so
nothing in it is theirs to read.

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
path it maps to (`<url> -> <path>`), and a write's verb says whether it created
or overwrote the file. That real path is what `bash`, `ls`, `grep` and `eval`
need — a shell cannot resolve a scheme — so use it whenever you step outside
the tools. A listing prints its entries as relative names and carries the
resolved folder in its header.

**You do not have to write a file first to learn that path.** `bash` and the
`eval` kernel are each given this session's pad as `$LOCAL_OPERATOR_SCRATCHPAD`,
an absolute path, so a shell can create directly into it:
`mkdir -p "$LOCAL_OPERATOR_SCRATCHPAD/logs"`, `> "$LOCAL_OPERATOR_SCRATCHPAD/x.log"`.
The folder is created on first use — by `write`/`edit`, or before a shell call
runs — so a bare redirect and a `mktemp` template both work first time. Some
results also carry a one-line `[scratch]` reminder when a command writes into a
temp root; the result carries the path whole. The idiom for a rig that wants a
private subdirectory is `mktemp -d "$LOCAL_OPERATOR_SCRATCHPAD/rig.XXXXXX"`
— the template keeps the directory inside the pad, so the files stay readable
through the scheme and survive the session, which a `/tmp` one does not. The
name is set only inside an agent's own shell and kernel; it is
unset in the user's terminal, where the `scratchpad://` calls are the way in.

## File names and types

Give every file a real extension: the desktop app shows a tile only for a known
one — `perf-2026-09-17.md` as a document, `.csv`/`.tsv` as a spreadsheet,
`.sh`/`.py`/`.json` in the code editor — and an extensionless file gets no tile
and no viewer. One file per subject; subdirectories are free
(`scratchpad://logs/run.md`).

## Rules the tool enforces

- `..`, absolute paths and dotfiles are refused outright, as are `?` and `#`
  (they open a query or fragment — percent-encode as `%3F`/`%23`), and a name
  containing `://` is refused: a second scheme inside a name is a URL, not a file
  name.
- One directory level per listing, and a listing is bounded, so a wide directory
  cannot flood the transcript; a very large file comes back truncated, with the
  `read` call that continues it.
- A pad is capped THREE times, because it is disk shared with every other
  session on the machine and is reclaimed only when its session ends: a SINGLE
  write at 32 MiB (33,554,432 bytes); the whole pad at 256 MiB (268,435,456
  bytes), past which every write is refused whatever filled it — no list of
  names knows every shape of build tree, so the last rule is the pad's own
  total; and, as a tree rather than a pad, a folder holding over 20,000 entries.
  Overwriting the file that took the pad over DOES still work — a replaced file
  is not counted twice, and with nothing here deleting anything that is the only
  way back under. Reading is deliberately NOT gated, so a pad written before
  these rules can still be listed, read and cleaned up. Nothing removes the
  folder later either — only the session going away does that, by the cleanup
  pass (when its policy is switched on) or by the session being deleted.
- Paths are resolved and must stay inside the scratchpad; a symlink pointing out
  is refused rather than followed.
- Any other URL scheme in a path argument is refused: a scheme the tools do not
  own is not silently reinterpreted as a relative path.

## When it is unavailable

Some hosts have no session folder (a `--train` agent directory, a bare tool
context). The tool says so. Use a real temporary directory — `bash mktemp -d`
with NO template, so it lands in `$TMPDIR` and not in `/tmp` (see above) — and
keep the absolute path in your working notes; do not put scratch in the user's
working directory to work around it.
