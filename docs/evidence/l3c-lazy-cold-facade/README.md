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

## What a cold open costs, miss versus hit

Fresh process per side, four opens each, replay cache cleared between samples —
i.e. the restart case a durable record exists for. Baseline = `origin/main`.

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
* **A later process's open of an unchanged journal reads no journal bytes at
  all** — 0 passes and 0 bytes against 8.6 MB, at ~10 ms of CPU against
  119-218 ms. The remaining ~10 ms is the synthesised state, the config/wake
  reads and the restore, not a hidden parse.

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
round-2 figures are the same cost on both sides. The durable record's benefit is
the *cross-process* one measured in the table above (a restart, which is what a
desktop server does between requests), not an intra-process one.

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
* Real process, real routes, frame capture and canonical diff: `l3c_run.sh`
  with `l3c_frames.py` and `l3c_cpu.py`. Every process is reaped by pid and every
  root is a temp directory. Both need `L3C_WORKTREE` (and `L3C_ROOT_BASE` /
  `L3C_ROOT_BRANCH` for the shell driver) pointing at the two checkouts.
