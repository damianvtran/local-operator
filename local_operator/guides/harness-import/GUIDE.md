---
name: harness-import
description: "Bring conversations and memory from Claude Code, Codex or omp into Local Operator safely: list, preview, select explicitly, confirm, import; read-only on the source."
---

# Importing another harness

Use the `harness_import` tool. It is read-only on the other harness and only
creates Local Operator sessions and memory records after an explicit preview and
confirmation. No filesystem path is accepted; source names and source session
ids are the whole addressing surface.

## Safe flow

1. `op="sources"` — see which harnesses exist on this machine and how many
   sessions each lists.
2. `op="sessions", source="omp"` — inspect titles, dates, message/compaction
   counts and availability. Source paths are never returned.
3. `op="preview", source="omp"` — see the estimated session/message/compaction
   counts and any warnings. This writes nothing. The result carries a
   `preview_token`.
4. Ask the operator which conversations to bring over. Always show the preview
   before asking; never select silently on their behalf.
5. `op="import", confirm=true, preview_token="…"` with either
   `session_ids={"omp": ["…"]}` or `all=true`. There is no default-all. The
   request must match the previewed selection and options exactly.
6. Poll `op="status", job_id="…"` until done/cancelled/failed. Re-running the
   same import is safe: an unchanged source is a no-op, a grown source resumes,
   and a rewritten source creates a new target instead of splicing histories.
7. Read the result's warnings and `instruction_review_path`. Re-run is safe.

## What it does not do

- It never modifies, locks, checks out, or writes inside the foreign harness.
- It never reads `auth.json`, credential stores, `.env`, settings, or plugin
  stores. Credential-shaped text in conversations is redacted before writing;
  counts are reported, values are not.
- It does not import silently, auto-select every session, or accept a path.
- It does not append a rewritten Claude branch or Codex rolled-back turn to an
  existing conversation; it creates a new target with a recorded predecessor.

## Memory and instructions

Curated foreign memory becomes Local Operator facts under the core scope rules:
durable personal facts are global, work state tied to an imported conversation
is session-scoped, and untethered work state goes to a review artifact.
Instruction files (`CLAUDE.md`/`AGENTS.md`) are written to a review artifact
under `<config>/imports/instructions/`; they are never appended to live
instructions automatically. Tell the operator where the review copy is and let
them choose what, if anything, to adopt.

`include_memory=false` or `include_instructions=false` skips those stages.
`include_subagents` and `include_cleared_segments` are not implemented in this
slice; preview/result report an explicit unsupported-option warning rather than
silently ignoring them.

## Cancel and failures

`op="cancel"` with a `job_id` stops a running import between write batches. A
cancelled target is left with a partial manifest; importing the same source
again resumes it. Per-session failures are reported in the result and do not
abort the other sessions. If a source is oversized or unreadable, that row is
marked `unsupported` and its siblings remain importable.
