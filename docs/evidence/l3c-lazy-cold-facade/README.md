# Deferring the cold desktop facade's message list

Evidence for the L3c slice of `CORE-PROFILE-SUPPLEMENT.md` §2: a durable
metadata record lets a desktop cold open rebuild the same first frontend state
without parsing the journal, deferring the message list to an explicit hydration.

Commits: `feat/l3c-lazy-cold-facade`, branch cut from `origin/main`
(`fd94f589b4`). Numbers below were taken on 2026-10-25 on this host, which runs
~25 concurrent agent sessions; CPU is `time.process_time()` (the judge metric),
never wall time.

## The fixture

One synthetic journal, built through the real writer (`Transcript`): 1,000
turns × (user + assistant-with-receipt + tool result + spend row), one
`frontend_state_checkpoint_v1` row at the tail. **8,634,662 bytes, 3,000
messages.** It lives in a temp config root; the operator's store is never read.
Both sides of every comparison get their own APFS clone of the same seed, so the
bytes are identical and only the code under test differs.

## What a cold facade open costs, miss versus hit — IN ONE PROCESS

**Read the scope off the label: these are four facade opens inside ONE process
per side, with only the replay cache cleared between samples.** They are not
fresh-process numbers. The process stays warm in every way a process can —
imports, the transcript index cache, config, the roster and wake reads — so the
hit figure below is the cost of opening a facade with an empty replay cache, and
it is *not* what a cold restart of the desktop costs. (An earlier revision of
this file called these "fresh process" numbers. That was wrong, and the
fresh-process measurement in the next section is what the restart claim rests
on.)

Baseline = `origin/main`.

| side | sample | backend CPU | journal passes | journal bytes |
|---|---|---|---|---|
| origin/main | 0 | 462.66 ms | 1 | 8,634,662 |
| origin/main | 1 | 217.76 ms | 1 | 8,634,662 |
| origin/main | 2 | 118.85 ms | 1 | 8,634,662 |
| origin/main | 3 | 223.36 ms | 1 | 8,634,662 |
| branch | 0 (MISS, writes the record) | 448.97 ms | 1 | 8,634,662 |
| branch | 1 (HIT) | 10.30 ms | **0** | **0** |
| branch | 2 (HIT) | 10.35 ms | **0** | **0** |
| branch | 3 (HIT) | 10.26 ms | **0** | **0** |

Read as two claims, and only two:

* **The first open of a journal is unchanged.** Sample 0 on the branch is the
  same parse, the same bytes and the same CPU as `origin/main`; it is that read
  which writes the record. Nothing here makes a first-ever open cheap, and a
  journal whose record is missing, stale, corrupt, unknown-schema or from an
  older build takes exactly this path.
* **An open with a matching record reads no journal bytes at all** — 0 passes
  and 0 bytes against 8.6 MB, at ~10 ms of CPU against 119-218 ms *in the same
  in-process scope*. Those ~10 ms are the synthesised state, the config/wake
  reads and the restore, not a hidden parse. What this does NOT say is that a
  cold restart opens in ~10 ms; see the fresh-process section for that number.

## Real process, real route

`local-operator serve` on its own temp config and port (one per side), the real
`GET /v1/desktop/sessions/<id>` snapshot route and the real `/events` stream,
reaped by exact pid (`pgrep` after the run: none).

Twenty snapshot reads per round, server CPU from `ps -o cputime` before and
after — one request is below macOS's reporting resolution, twenty are not:

| side | round 1 | round 2 | record on disk |
|---|---|---|---|
| origin/main | 35.5 ms/request | 12.0 ms/request | absent |
| branch | 21.0 ms/request | 15.5 ms/request | 3,468 bytes |

**These steady-state numbers do NOT show the win, and that is the honest
reading.** Within one process the replay cache already answers the second and
later reads, so neither side pays a parse per request after the first: the
round-2 figures are the same cost on both sides.

## Fresh process — what independent QA actually measured

The restart question, answered by a measurement that starts a NEW process for the
open rather than reusing a warm one (independent QA, evidence id 6403162115):

| | backend CPU |
|---|---|
| fresh-process **hit** (record valid, journal unchanged) | **314.6 ms** |
| of which **session enter** | **267.8 ms** |

The byte claim survives at this scope: **0 replay passes, 0 journal bytes** — the
replay-message parse and attachment resolution really are gone from the open.
What does not survive is any reading of the in-process table as a restart cost.
A cold restart with a valid record still costs ~315 ms of backend CPU, and the
dominant remaining term is **session enter** (state synthesis, config, the
index, the sidecars) — work this slice does not touch and does not claim to have
improved.

So: **`<300 ms` for a cold start is not demonstrated**, and nothing in
this change should be read as promising it. What is claimed is narrower and
measured: the journal is not replayed, and the first paint's frontend state is
identical.

## Wire equality

The `/events` frames captured from those two `serve` processes were compared
canonically — recursively stripping only the fields two independent processes
cannot make equal (`epoch`, the serve clocks, opaque cursors/tokens), with the
list in one place — over four journal shapes seeded into both sides:

* settled conversation with a checkpoint,
* checkpointless (pre-ledger) journal,
* journal whose last tool call has no result yet,
* compacted journal whose head was cut.

Both sides emitted `open → snapshot → heartbeat`, and for the real-process pair
the canonical snapshot payload was **IDENTICAL, field for field**. The in-process
test suite asserts the same equality per shape, and additionally that the
deferred open performed **no** journal pass while the eager open performed one.

## Residual named, not solved

The witness is the replay cache's own: session directory, `(st_ino, st_size,
st_mtime_ns)` plus the request's cut and wanted custom types, with a re-stat
after the record read. That proves this codebase's writers (append, or atomic
replace) did not touch the journal. It does **not** prove the bytes are
unchanged: `Transcript._write_entries` restores the mtime after a bookkeeping
batch, and its rollback restores size and mtime, so a same-size interior rewrite
inside one mtime instant is undetected. That is recorded in the module docstring
rather than papered over, and the payload is metadata only for that reason — a
missed rewrite can restore stale roster/todo/spend facts, never a wrong
transcript. Closing it properly needs a durable writer-maintained generation and
is a separate change.

## Reproduce

* In-process numbers and the fixture: `l3c_measure.py` (`seed`, `inproc`).
  `inproc` opens N facades inside ONE process and clears only the replay cache
  between them — it does not measure a fresh process, and its hit figure must
  not be quoted as a restart cost.
* Real process, real routes, frame capture and canonical diff: `l3c_run.sh`
  with `l3c_frames.py` and `l3c_cpu.py`. Every process is reaped by pid and every
  root is a temp directory. Both need `L3C_WORKTREE` (and `L3C_ROOT_BASE` /
  `L3C_ROOT_BRANCH` for the shell driver) pointing at the two checkouts.

## The TUI's cold-boot frame is unchanged

The TUI never passes `deferred_history` — its cold facade, the peer viewers, the
mobile daemon, `/resume`, a new session and the sidebar's saved preview all take
the eager path — so the frame to prove unchanged is the TUI's own cold boot.
Captured with the repo's own recipe (`scripts/eager_boot_shot.py --isolated`,
real `OperatorApp`, isolated `HOME`/config, both checkouts run from the SAME cwd
and rendered with `rsvg-convert`):

| artifact | sha256 |
|---|---|
| `tui-cold-open-before.svg` (`origin/main`) | `6fbca07611c0e7210dc2b8ddc547f3c4da4eb3bb73adb5decd1588fef2cd47c0` |
| `tui-cold-open-after.svg` (branch) | `6fbca07611c0e7210dc2b8ddc547f3c4da4eb3bb73adb5decd1588fef2cd47c0` |

Byte-identical SVG, and `compare -metric AE` on the two rasters reports **0**
differing pixels. Both frames were looked at: splash, version, cwd line, the
slash hints, the tip line and the composer band all paint as before.

An earlier uncontrolled pair (each side run from its own checkout) differed in
exactly the two fields that track the run rather than the code — the printed cwd
and the composer's settle state — which is why this pair is taken from one cwd
and compared byte for byte.

## The order the renderer uses, and what a deferred read installs

Two behaviours the review asked to be pinned rather than argued:

* **`/history` before `/events` needs no hydrated history.**
  `test_the_history_page_before_events_needs_no_hydrated_history` drives the real
  route, then the real handshake, and asserts the page equals the eager page
  *and* that the facade is still deferred afterwards — the page is answered from
  the journal index, so nothing in that path touches `_history`.
* **Hydration installs the journal as of the READ, not as of the open.**
  `through_id=None` is the same cut the eager cold open uses, so deferral moves
  the moment and never the cut;
  `test_hydration_reads_the_journal_as_of_the_hydration_not_the_open` appends a
  row while the facade sits deferred and asserts it is installed exactly once,
  that `_cold_painted_ids` derives from that same list, and that the result
  equals what an EAGER facade reads at the same instant. The record's witness is
  cache-validation evidence and never a fence on that read.
