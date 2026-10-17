# LOH-14 — Harness import: mapping design and slice plan

Status: proposed for implementation · 2026-10-17
Owner: LOH-14 delivery manager (`6b4a77d5d906`) · Requested by: hackathon intake (`b290eb6429d7`)
Scope source: operator brief, 2026-10-17. This note is the gate before code.

## 0. Decision summary

Build one offline, read-only import service in `local_operator/harness_import/` that
converts Claude Code, Codex and omp conversations and memory into Local Operator's
existing durable formats:

- **Conversations** become normal `~/.local-operator/sessions/<id>/transcript.jsonl`
  sessions, replayed by the existing `Transcript.build_llm_history()`.
- **Compaction** becomes a real lop `compaction` entry whose `summary` is the source
  summary, whose `first_kept_entry_id` is the source's first retained message, and
  whose earlier source rows remain on disk before that anchor. Replay therefore sees
  the same "summary + kept window" the source saw, while the replaced span is still
  inspectable and re-importable. No LLM call is made during import.
- **Memory** becomes atomic `MemoryStore` facts in the global bank, with provenance
  and source timestamps; instruction files (`CLAUDE.md` / `AGENTS.md` / `MEMORY.md`)
  are converted to reviewable candidate facts by default, never silently appended to
  the operator's live instructions.
- **Every surface** (TUI, desktop UI, agent tool, optional CLI) calls the same
  service. No surface re-implements parsing, dedupe or compaction mapping.

The import is additive: foreign stores are opened read-only, target sessions are new
directories with deterministic IDs, and re-running the same import is a no-op or a
resume, never a duplicate session and never an append to a user-edited transcript.

## 1. Scope boundary

Carried from the operator's brief:

1. Core: one import service/API for **memory** and **conversations**, preserving
   roles/order, tool-call structure, timestamps, session boundaries and compaction
   semantics.
2. TUI: import slash commands (list harnesses/conversations, select one/all, progress
   and results).
3. UI: Settings → Import (browse, select one/all, progress and results) with rendered
   evidence and design/UX rounds.
4. Agent-usable: the same capability as a tool so Aida can guide a first-time user.

Explicitly **not** in this workstream, although the tracker description mentions them:
importing skills, MCP server definitions/plugins or general settings. Those are a
separate workstream if the operator expands scope. This note does include the three
harnesses named by the tracker; the adapter interface leaves room for others.

## 2. Evidence: real stores on this machine (read-only, 2026-10-17)

All probes were read-only. No foreign file, SQLite database or directory was modified.
No message bodies or secrets are reproduced here.

| Store | Version / path | What was actually present | Consequence for the design |
|---|---|---|---|
| Claude Code | 2.1.274; `~/.claude` | No `~/.claude/projects/` and no transcript `.jsonl` files. `history.jsonl` has 322 prompt lines across 39 session ids. `~/.claude/CLAUDE.md` is 3,480 bytes. The native binary contains `parentUuid`, `isSidechain`, `isCompactSummary`, `compactMetadata` (with `preservedMessages`, `allUuids`, `anchorUuid`, `excludeUuids`), `compact_boundary`, `logicalParentUuid`, `toolUseResult`, `MEMORY.md`, `agent-memory/`, `agent-memory-local/` and `memory/` path strings. | Listing can use `history.jsonl`; bodies are **metadata-only/unavailable** on this machine. The parser is designed from the verified field names and the omp fork's independent Claude importer, and is tested with synthetic fixtures. Do not fabricate a conversation from prompt history. |
| Codex | 0.147.0; `~/.codex` | `state_5.sqlite` has 1,203 threads (1,189 `legacy`, 14 `paginated`). Every sampled `rollout_path` under `~/.codex/sessions/...` is **missing**; `~/.codex/archived_sessions/` holds two real `rollout-*.jsonl.zst` files (88 and 42 records). `memories_1.sqlite` has the `stage1_outputs` memory table but **0 rows**. `AGENTS.md` is 3,480 bytes. The native binary contains `compacted`, `replacement_history`, `compaction_summary`, `encrypted_content`. | Most Codex conversations are index-only on this machine: list them with an explicit "body unavailable" state, do not silently omit. The two archived rollouts are the only real body fixtures available; `.jsonl.zst` is supported when the runtime has `compression.zstd` (the installed lop runtime is Python 3.14; the repo test venv is 3.12, so that test is skipped there). |
| omp | 18.11.2; `~/.omp/agent` | 9 session `.jsonl` files, 913 messages and 521 custom entries across them. Sample: `session` header, `title`/`title_change`, `model_change`, `thinking_level_change`, `message` (roles `user`/`assistant`/`toolResult`; blocks `text`/`thinking`/`toolCall`/`image`), `custom_message`, `custom`, `credential_pin`, `service_tier_change`, `model_usage`. `session-index.db` is stale (1 row); `history.db` has titles/history. No `compaction` entry exists locally, but the fork's `CompactionEntry` schema is: `summary`, `shortSummary`, `firstKeptEntryId`, `tokensBefore`, `details`, `preserveData`, `fromExtension`, `warning`. No `~/.omp/agent/memories/` tree exists; no `learned.md` found. | omp is the best real end-to-end fixture: full messages, tool calls, timestamps and branches. Compaction mapping is tested with synthetic entries built from the verified schema. Memory adapter is file-based and handles an absent tree. |
| lop target | `~/.local-operator` | 4,091 session `transcript.jsonl` files. Entry shape `{"id","ts","type","payload"}` with `type` in `message` / `compaction` / `custom` / `prune`; compaction payload includes `summary`, `first_kept_entry_id`, `tokens_before`, `preserve_data`, `preserved_user_turns`, `preserved_turns_cap`. `Message` roles are `user`/`assistant`/`tool`; content union is text/image/audio/attachment (**no thinking block**). Session sidecars: `created_at.json`, `title.json` (`{text,user_set,names}`), optional `origin.json`. `memory/memory.db` has 12,513 current facts; `MemoryStore.put` requires atomic title+body (≤160 title, ≤256 tokens/8 KiB body), validated source metadata and global/session/team/agent scopes. | Target mapping is concrete: write `TranscriptEntry` rows with source `ts` and deterministic IDs; write `created_at.json` before materialising; write `title.json` through `resume.write_session_title`; store imported memory through `MemoryStore.put`, never direct SQLite writes. |

Independent reference: `~/oss/oh-my-pi/packages/coding-agent/src/session/claude-session-store.ts`
and `codex-session-store.ts` already parse both formats for omp's own importer. They
are a cross-check for field names and edge cases (tool pairing, Codex `event_msg`
duplication, `thread_rolled_back`, `compacted.replacement_history`), not a dependency
and not code to copy blindly: our target types and compaction semantics differ.

## 3. Canonical import model

Every adapter converts its source into the same small model. Mapping to the lop
transcript happens once, in `mapping.py`.

```text
SourceSession
  source: "claude" | "codex" | "omp"
  source_id: str
  path: Path
  title: str | None
  cwd: str | None
  created_ts: float          # source time, never import time
  updated_ts: float
  message_count: int | None  # metadata estimate only
  compaction_count: int
  parent_source_id: str | None
  kind: "conversation" | "subagent"
  availability: "full" | "metadata_only" | "unsupported"
  warnings: list[str]

SourceEvent (streamed, one at a time)
  source_event_id: str
  parent_source_event_id: str | None
  ts: float
  kind: "user" | "assistant" | "tool_result" | "compaction" |
        "model_change" | "title" | "reset" | "unknown"
  text: list[str]            # text blocks, source order
  images: list[ImageContent] # inline images only; URL-only images are skipped
  thinking: list[str]        # counted, not rendered in v1 (lop has no thinking block)
  tool_calls: list[ToolCall] # id, name, arguments
  tool_result: ToolResult | None
  compaction: CompactionInfo | None
  model: str | None
  raw_flags: dict            # e.g. isMeta, isSidechain, synthetic

CompactionInfo
  summary: str
  first_kept_source_id: str | None   # source id of the first retained message
  tokens_before: int
  details: dict                       # source counts/window ids, never secret values
  preserve_data: dict                 # namespaced under "source_snapshot" if kept
  preserved_user_turns: list[str]

MemoryRecord
  source_id: str
  title: str
  body: str
  tags: list[str]
  created_ts: float
  source_time: float | None
  uri: str
  status: "candidate" | "active"
```

Adapters are generators. They never call `read_text()` on a whole session, never load
a full JSON array, and never write to the source. `list_sessions()` reads only metadata
(paths, titles, timestamps, row counts); bodies are read only when the user previews
or imports.

Per-source parse and branch rules are in §4; target row construction is in §5.

## 4. Per-harness mapping

### 4.1 Claude Code

Discovery roots (in order): `CLAUDE_CONFIG_DIR` if set, else `~/.claude`.
Conversation files: `<root>/projects/<encoded-cwd>/<session-uuid>.jsonl`.
Metadata fallback: `<root>/history.jsonl` (prompt history keyed by `sessionId`);
`~/.claude.json` project entries are only a cwd/trust index, not conversation bodies.
If a session id appears in `history.jsonl` but has no transcript file, list it with
`availability="metadata_only"` and a reason; import refuses it rather than inventing a
conversation from prompts.

| Claude record | Use |
|---|---|
| `type: "user"` with string/array `message.content` | user text; `tool_result` blocks become separate `tool` messages, `tool_use_id` retained |
| `type: "assistant"` with `message.content` | assistant text + `thinking` + `tool_use` blocks; `tool_use` becomes `tool_calls` |
| `type: "system"`, `subtype: "compact_boundary"` | compaction boundary; `compactMetadata` is the compaction record |
| `type: "summary"` (`summary`, `leafUuid`) | compaction summary; anchor is `leafUuid` or the first retained uuid |
| `isCompactSummary: true` on a user record | the synthetic summary row; do not import it as a normal user turn |
| `isMeta: true` | skip (harness bookkeeping) |
| `isSidechain: true` | subagent thread; skip by default, import as a separate session only when `include_subagents` |
| `uuid` / `parentUuid` | branch tree; `logicalParentUuid` is recorded in the manifest only |
| `custom-title` / `ai-title` | source title (custom wins); `user_set` true only for `custom-title` |
| `timestamp`, `cwd`, `sessionId`, `gitBranch` | timestamps, session metadata, manifest provenance |

Branch policy: choose the newest leaf (highest timestamp; tie → last record), walk
`parentUuid` back to the root, reverse, and emit that active path. Inactive branches
are recorded in the manifest as IDs only. A `summary` record with a `leafUuid` outside
the active path is a branch summary; skip it with a warning rather than applying it to
the active conversation.

Compaction: prefer a `system/compact_boundary` record with `compactMetadata`; else the
`summary` record; else an `isCompactSummary` user row. `tokens_before` comes from
`compactMetadata.preTokens` or is estimated from the summarized text. `preservedMessages`/`anchorUuid` identify
the first retained message; if absent, use the first message after the compaction row.
The summary row itself is not imported as user text.

### 4.2 Codex

Discovery roots: `CODEX_HOME` if set, else `~/.codex`.
Bodies: `<root>/sessions/**/rollout-*.jsonl` and
`<root>/archived_sessions/**/rollout-*.jsonl(.zst)`. Metadata:
`<root>/state_5.sqlite` (`threads`), `<root>/session_index.jsonl` (id → title).
The `threads` row gives id, title, cwd, created/updated, source, model provider,
rollout path and archive state; it is the listing source when rollout files are absent.
Read SQLite through the private snapshot described in §7, with explicit columns only.

| Codex record | Use |
|---|---|
| `session_meta` | session id, cwd, timestamp, cli version, model provider, git commit |
| `turn_context` | current model; a change becomes a model-change metadata event |
| `response_item` `message` role user/developer/assistant | user/assistant text; `developer` messages are system instructions and are skipped |
| `response_item` `function_call` / `custom_tool_call` | assistant tool call; coalesce with adjacent assistant text in the same turn |
| `response_item` `function_call_output` / `custom_tool_call_output` | tool result |
| `response_item` `reasoning` | counted, not stored (lop has no thinking content) |
| `response_item` `web_search_call` / `tool_search_call` and outputs | tool call/result with a synthetic name |
| `event_msg` `user_message` / `agent_message` / `agent_reasoning` | used only when no canonical `response_item` text with the same text exists; otherwise dedupe by exact text |
| `event_msg` `thread_name_updated` | title |
| `event_msg` `thread_rolled_back` (`num_turns`) | drop the last N user turns and everything after their start, matching source semantics; count the rollback in the manifest |
| `compacted` | compaction: `payload.message` summary, `replacement_history` first retained message, `window_id`/`window_number`/`previous_window_id` in preserve data |
| `encrypted_content` / provider-encrypted compaction item | never stored; record presence, byte count and a hash only |

Tool pairing: a `function_call` and its output can be separated by other records. Keep
a map call-id → tool name/arguments; emit the assistant message with all calls in the
turn, then tool results in source order. If a call has no output, emit the assistant
message with the call and no result; mark the session partial.

Compaction: if the rollout contains the pre-compaction messages, they are imported as
normal archived rows before the compaction marker. If the rollout was pruned and only
`replacement_history` remains, import the retained replacement messages first (marked
`synthetic="replacement_history"`) and anchor at the first real message after the
summary; otherwise anchor at the compaction row itself (summary-only replay). The
encrypted compaction item is not decrypted — there is no key and no network.

Zstd: use `compression.zstd` inside a guarded import. On Python < 3.14, list `.zst`
sessions as unsupported with a clear reason; do not shell out to `zstd`.

### 4.3 omp (oh-my-pi)

Discovery root: `OMP_HOME` if set, else `~/.omp`.
Bodies: `~/.omp/agent/sessions/<encoded-cwd>/*.jsonl`; `<encoded-cwd>` is the
`--path-with-dashes--` form (e.g. `--private-tmp--`). `session-index.db` and
`history.db` are optional metadata; the JSONL is authoritative.

| omp entry | Use |
|---|---|
| `session` header | source id, cwd, timestamp, version; `parentSession` means a child/subagent session |
| `title` / `title_change` | latest title; `source: "user"` sets `user_set` |
| `model_change` / `thinking_level_change` / `service_tier_change` | metadata; model changes become metadata entries |
| `message` role `user`/`assistant`/`toolResult` | text/image/toolCall/toolResult; assistant `toolCall` blocks become `tool_calls` |
| `message` content `thinking` | counted, not stored |
| `message` content `image` | inline base64 only; URL-only images are skipped with a warning (no network) |
| `custom_message` `attribution: "user"` | user-visible injected text; import as user text with a source prefix |
| `custom_message` agent/system (todo nudges, async results, launch notices) | skip; harness chrome |
| `custom` `tool_execution_start` / `tool_execution_update` | skip; duplicate of message toolCall data |
| `custom` other (`system-prompt-digest`, `session_exit`, `todo_hud_state`) | metadata/ignored |
| `credential_pin` | skip always |
| `compaction` | full mapping per §5.4 (`summary`, `firstKeptEntryId`, `tokensBefore`, `preserveData`, `details`) |
| `branch_summary` / `reset_boundary` | branch/reset semantics: reset is a hard session boundary; active segment imported, earlier segment optional |
| `parentId` | branch tree; active leaf policy as Claude |

Branch policy: choose the newest leaf by timestamp; walk `parentId` to the root. A
`session` header with `parentSession` is a child session: skip by default; when
included, import it as its own lop session and record the parent in the manifest.
Inactive branches and branch summaries are recorded as metadata, not merged into the
active conversation.

### 4.4 Memory

Memory is not "more transcript". Lop's memory guide explicitly forbids bulk-ingesting
transcripts, environment variables or credential paths; imported memory must therefore
be atomic, provenance-tagged and reviewable. The adapter emits `MemoryRecord`s and the
service writes them through `MemoryStore.put` (never direct SQLite writes).

| Source | Artifact | Mapping |
|---|---|---|
| Claude Code | `~/.claude/CLAUDE.md`, `~/.claude/CLAUDE.local.md` | instruction text; default: split into section-sized **candidate** facts with `source.kind="reported"`; optional append to `~/.agents/AGENTS.md` only after explicit confirmation |
| Claude Code | Claude's auto-memory index and its linked files: the global `agent-memory/MEMORY.md` when present, plus each project's `projects/<encoded>/agent-memory/`, `agent-memory-local/` and legacy `memory/` | parse index bullets/headings and linked files; one fact per bullet/section; title from the heading, body bounded to 256 tokens/8 KiB; `source.uri` = file path |
| Codex | `~/.codex/AGENTS.md` | same instruction handling as Claude's `CLAUDE.md` |
| Codex | `~/.codex/memories_1.sqlite` → `stage1_outputs` (`raw_memory`, `rollout_summary`, `rollout_slug`, `generated_at`, `usage_count`) | one or more atomic facts per row; title from `rollout_slug` or thread title; body split on paragraph boundaries; tags `codex`, `imported`, thread id; `source_time=generated_at`; `source.kind="summary"` |
| omp | `~/.omp/agent/memories/<encoded>/{MEMORY.md,memory_summary.md,learned.md}` | parse bullets/sections; `learned.md` bullets are facts; `MEMORY.md` index entries point at detail; `memory_summary.md` is a consolidated summary |
| omp | remote `hindsight` backend, or a `mnemopi` bank not on disk | skip with `remote_backend_no_network`; no HTTP call |
| Any | `auth.json`, `auth.db`, `settings.json`, `.env`, credential/plugin stores | never read; not an import source |

Target records use `MemoryStore.put("global", fact)` with:

```text
title:       <=160 chars, source heading/slug or first sentence
body:        atomic, <=256 tokens/8 KiB, redacted
tags:        ["imported", "source:<claude|codex|omp>", <source tag>] (<=12)
source:      {kind: "reported"|"summary", uri: "harness-import:<source>:<id>",
              author: "harness-import", session: "<target session id>"}
             # session is omitted for memory derived from an instruction/memory
             # file, which has no target conversation.
created_at:  source generated time when known, else source file mtime
source_time: source timestamp
status:      "candidate" for instruction-derived or low-confidence records;
             "active" for curated memory rows (Codex stage1_outputs, omp MEMORY.md)
```

Dedupe is `MemoryStore.put`'s exact provenance-aware dedupe (same body/source/created_at
returns `deduplicated`). Per-item refusals are counted and skipped; they never fail the
whole import. The global quota is 16k active tokens: preview shows projected tokens and
the service imports in source priority order (curated memory first, instruction-derived
candidates last), reporting skipped records if the quota is reached. Imported sessions
set `capture_memory=false` in their manifest so lop's automatic capture does not re-mine
migrated transcripts unless the operator opts in.

## 5. Target mapping: lop session and transcript

### 5.1 Session identity and sidecars

```text
target_session_id = sha256("lop-harness-import/v1\0" + source + "\0" + source_session_id)[:12]
```

12 hex characters matches the existing session-id shape. On collision with a different
source id, extend the hash deterministically; never reuse a session directory that
already has a different `harness-import.json`.

Before the transcript exists, create the directory and write:

- `created_at.json` — source `created_ts` (atomic write; then construct
  `Transcript(defer_materialise=True)` so the constructor does not stamp import time).
- `title.json` — via `resume.write_session_title(session_dir, title, user_set=..., past_names=[])`.
- `harness-import.json` — manifest (below).

A session with no title falls back to the first user message's provisional title; an
empty conversation gets the source id as a last resort. Titles are whitespace-normalised,
secret-redacted and bounded to lop's title rules.

Manifest is durable, not LLM-visible:

```json
{
  "v": 1,
  "source": "omp",
  "source_id": "01a14564-...",
  "source_path": "/Users/.../.omp/agent/sessions/...",
  "source_fingerprint": {"size": 3490, "mtime_ns": 0, "sha256_prefix": "..."},
  "source_created_at": 1760629588.0,
  "imported_at": 1760...,
  "status": "complete|partial|metadata_only",
  "active_leaf_source_id": "...",
  "last_source_offset": 3490,
  "imported_entries": 10,
  "counts": {"messages": 4, "tool_calls": 0, "tool_results": 0,
             "compactions": 0, "skipped": 6, "redactions": 0},
  "compaction_anchors": {"source-compaction-id": "target-entry-id"},
  "cwd": "/tmp",
  "parent_source_session_id": null,
  "model_history": [{"ts": 1760..., "model": "..."}],
  "warnings": [],
  "capture_memory": false,
  "target_prefix_sha256": "..."
}
```

`target_prefix_sha256` is the transcript's byte hash after import, used to detect a
user-edited target before a resume.

### 5.2 Entry IDs and timestamps

```text
entry_id = sha256("lop-harness-import/v1/entry\0" + source + "\0" +
                  source_session_id + "\0" + source_event_id + "\0" + ordinal)[:16]
```

`source_event_id` is the source's uuid/entry id; when absent (Codex line events), use
`ordinal:type:timestamp`. Re-importing the same source event produces the same target
id. Message entries reuse the message id as the entry id, as lop already requires for
`first_kept_entry_id`; adapters therefore build the `Message` with this id.

`ts` is always the source timestamp (ISO 8601 → Unix seconds). Missing timestamps are
filled monotonically: previous source ts + 1 ms, never `time.time()`. The session's
`created_at.json` uses the source session birth; `updated_at` is the last entry ts.

### 5.3 Rows

| Canonical event | Lop transcript row |
|---|---|
| user text | `type="message"`, `payload={"kind":"message","role":"user","content":[...]}` |
| assistant text/tool calls | `type="message"`, `role="assistant"`, `content` text blocks, `tool_calls` list |
| tool result | `type="message"`, `role="tool"`, `tool_call_id`, `tool_name`, `is_error`, content |
| compaction | `type="compaction"`, payload per §5.4 |
| model change / title / branch summary | `type="custom"` with `custom_type="import_model_change"` / `import_title` / `import_branch_summary`; replay ignores custom entries |
| reset boundary | split into segments (§5.5) |

Use `encode_message_payload()` for message serialization so the imported rows are the
same slim shape as live rows. Add a narrow, tested `Transcript.append_imported(entries)`
method (or an equivalent writer) that accepts pre-built `TranscriptEntry` objects with
their own `ts` and IDs, validates unique IDs and compaction anchors, and commits them in
one fsync batch. Import never goes through `append_message()` because that stamps
`time.time()`.

Images: inline base64 is carried as `ImageContent` and externalised by the existing
attachment write path; URL-only images are skipped with a count (no network). Images
larger than the attachment cap are truncated/skipped with a warning. Thinking blocks are
counted and not stored (lop's content union has no thinking block); they remain in the
foreign store.

### 5.4 Compaction row

```json
{
  "summary": "<source summary>",
  "first_kept_entry_id": "<target id of first retained message>",
  "tokens_before": 12345,
  "preserve_data": {
    "source": "codex",
    "source_compaction_id": "...",
    "window_id": 7,
    "window_number": 3,
    "previous_window_id": 6,
    "tokens_after": 4096,
    "replacement_history_count": 12,
    "encrypted_item_sha256": "...",
    "encrypted_item_bytes": 1234
  },
  "preserved_user_turns": [{"id": "...", "text": "..."}],
  "preserved_turns_cap": 100000,
  "preserved_turns_dropped": 0,
  "preserved_turns_shed": 0
}
```

Rules:

1. The summary text is the source summary, verbatim after redaction; never regenerated.
2. `first_kept_entry_id` is the target id of the first source message retained after the
   cut. omp gives it directly; Codex uses the first replacement-history message; Claude
   uses `anchorUuid`/`preservedMessages`/`leafUuid`. If no retained message exists, the
   anchor is the compaction row's own id and replay is summary-only.
3. `preserved_user_turns` carries every user-authored text in the summarized span,
   excluding synthetic summary/meta rows, in source order. The cap is 100k tokens (lop's
   default); overflow records `preserved_turns_dropped`/`shed` and keeps the earliest
   constraints. This is what prevents a compacted conversation from losing the user's
   own instructions to paraphrase.
4. `preserve_data` holds source-specific metadata only. Never store `encrypted_content`
   or a provider-encrypted compaction blob; store its hash and size. Never use the key
   `snapcompact` for foreign data — lop replay interprets that key — namespace it under
   `source_snapshot`.
5. Multiple compactions are written in source order. Lop replay uses the **latest**
   compaction, exactly like the source; earlier compactions and all pre-anchor rows stay
   on disk as the archive of what each span replaced. `Transcript.compact_file()` only
   folds prune entries, so it does not delete those rows.
6. A compaction with no summary and no kept messages is skipped with a warning; the
   messages are still imported linearly.

### 5.5 Session boundaries and branches

- One source **top-level conversation** becomes one target session. Subagent/sidechain
  threads are skipped by default; `include_subagents` imports each as its own target
  session with `parent_source_session_id` in the manifest. Never merge a subagent into
  its parent transcript.
- A `reset_boundary` or equivalent hard reset splits the source into segments. Only the
  active segment is imported by default; `include_cleared_segments` imports earlier
  segments as separate target sessions linked in the manifest. This preserves "what was
  cleared" without replaying it as live context.
- Branch alternates are not merged. The active branch is imported; inactive branches are
  recorded as IDs/paths in the manifest. `branch_summary` rows become metadata, not
  fake user messages.
- Codex `thread_rolled_back` is applied before emission: dropped turns never enter the
  target, and the count is recorded. This matches the source's own active history.

### 5.6 Idempotency and resume

| Situation | Behaviour |
|---|---|
| Manifest complete, source fingerprint unchanged | no-op; return `skipped_already_imported` with the existing target id |
| Source JSONL grew, target transcript prefix unchanged | resume from `last_source_offset`/event ordinal, append only new events; update manifest |
| Source branch changed or source file rotated/replaced | do not append; create a new target session with a deterministic suffix and record the old target as predecessor |
| Target `target_prefix_sha256` differs (user edited/compacted it) | never append to the user's session; import a new copy with a suffix and warn |
| Partially written target (crash) | replay manifest; if the transcript is a prefix of the expected rows, resume; otherwise create a new copy |
| Same source session selected twice in one job | dedupe by target id before starting |
| Import into a different config root | target IDs are stable, but manifests are per-root; no cross-root claim |

A small lazy index `<config>/imports/index.json` (rebuilt from session manifests,
never authoritative) makes "already imported" checks O(1) across thousands of
sessions. It is updated atomically after a successful import.

Memory import uses the same rule at the fact level: a re-run with an unchanged
source fingerprint is skipped, and source timestamps plus the `source` URI make
`MemoryStore.put`'s exact provenance-aware dedupe collapse repeated facts. A
changed source appends only facts whose `(source.uri, source_time, body)` tuple is
new; no memory entity is rewritten or deleted. The per-source fingerprint lives in
`<config>/imports/memory-<source>.json`.

## 6. Core service and API

Package: `local_operator/harness_import/`

```text
types.py        SourceKind, SourceSession, SourceEvent, CompactionInfo,
                MemoryRecord, ImportRequest, ImportPreview, ImportProgress,
                ImportResult, ImportWarning
sources/base.py SourceAdapter protocol: detect()/list_sessions()/read_session()/read_memory()
sources/claude.py, codex.py, omp.py
mapping.py      canonical events -> Message/CustomMessage/TranscriptEntry;
                compaction anchoring and preserved-user-turn extraction
redact.py       credential-shape detection + per-entry redaction (reuses
                local_operator.redaction_shapes; adds PEM/bearer/env-assignment cases)
manifest.py     harness-import.json read/write; source fingerprint; prefix checksum
store.py        read-only path/SQLite helpers; size caps; stable IDs; import lock
service.py      HarnessImportService: sources(), list_sessions(), preview(),
                import_sessions(progress=...), cancel()
memory.py       foreign memory -> MemoryStore.put, chunking/quota/dedupe
```

Service surface (in-process, async where I/O is involved):

```python
sources() -> list[SourceInfo]
list_sessions(source, *, include_subagents=False) -> list[SourceSession]
preview(request) -> ImportPreview        # counts, sizes, projected memory tokens,
                                         # warnings, target ids, no writes
import_sessions(request, progress) -> ImportResult
cancel(job_id) -> bool
```

`ImportRequest` carries: selected sources/session ids, `all`, `include_subagents`,
`include_cleared_segments`, `include_memory`, `include_instructions` (default false),
`memory_status` (`candidate`/`active`), and optional bounds. There is no arbitrary
path parameter: adapters use environment overrides/known roots only.

HTTP API (desktop-only, same `require_desktop` guard as other `/v1/desktop` routes):

```text
GET    /v1/desktop/imports/sources
GET    /v1/desktop/imports/sessions?source=omp
POST   /v1/desktop/imports/preview
POST   /v1/desktop/imports/jobs          -> {job_id}
GET    /v1/desktop/imports/jobs/{id}     -> progress/result
DELETE /v1/desktop/imports/jobs/{id}     -> cancel
```

Jobs are in-memory (bounded, newest 20) and survive no restart; the manifest makes a
re-run safe. Progress is polled by the UI; the existing SSE channel may carry
`import.progress` later, but polling is enough for v1. One import job at a time per
process, guarded by `<config>/imports/.lock` (pid + stale timeout).

Optional CLI for QA/headless: `lop import list|preview|run|cancel` calls the same
service; the TUI and UI never shell out to it.

## 7. Security, privacy and resource bounds

- **Read-only source access.** JSONL opened `r`; no `chmod`, lock files or temp files in
  foreign roots. SQLite is opened through a private snapshot: copy the database and its
  `-wal`/`-shm` files (when present) under `<config>/imports/snapshots/<hash>/`, open the
  copy with `immutable=1`, and delete the snapshot after listing. A plain `mode=ro`
  connection can create a `-shm` side file in a WAL database, which would violate the
  read-only contract. Symlinks are resolved and confined to the detected root; a link
  escaping the root is skipped.
- **No secrets.** Never read `auth.json`, `auth.db`, `settings.json`, `.env`, credential
  or plugin stores. Skip `credential_pin`, encrypted compaction content and auth-like
  records. Redact credential shapes in text/tool arguments/tool results before writing;
  record counts, never values. Reuse `local_operator.redaction_shapes` and the memory
  guard; refused memory facts are skipped, not forced.
- **No network.** The package imports no HTTP/provider client and accepts no URL to
  fetch. URL-only images are skipped. Tests block socket creation and assert no
  outbound attempt; the operator's networking/LAN pause is respected (no transport work).
- **Bounded memory/CPU.** Stream line-by-line; cap line size (16 MiB), entries per
  session (200k), total bytes (512 MiB), memory records (5k), and batch transcript
  writes (200 rows / 4 MiB). A parent map stores only ids/parents, not bodies. No full
  file reads, no `json.load` of a whole session, no model calls.
- **No source modification ever.** QA hashes every foreign file before/after and compares;
  SQLite `-wal`/`-shm` mtimes must not change. A read-only root test must still import.
- **Target safety.** Imported titles/notes are sanitised; manifest is written atomically;
  `created_at.json`/`title.json` use existing atomic helpers; no overwrite of an existing
  session with a different source identity.
- **Memory safety.** Atomic facts only, never transcripts; instruction files become
  candidate facts unless the operator explicitly chooses append; quota is reported before
  import; automatic capture is off for imported sessions. The capture path
  (`local_operator/memory_capture.py`) must consult each session's `harness-import.json`
  and skip a session whose `capture_memory` is false; this is a named change in slice 2,
  not an assumption.

## 8. Surface integration

### 8.1 Agent tool + Aida

Add `harness_import` to the tool registry (`local_operator/tools/`), backed only by the
core service. Actions: `sources`, `sessions`, `preview`, `import`, `status`, `cancel`.
It exposes no filesystem paths; `import` requires an explicit confirmation argument and
runs with the normal write-approval policy. The tool description says: "Import
conversations and memory from Claude Code, Codex or omp; read-only on the source; preview
first."

Add a packaged guide `guide://harness-import` with the safe flow. Extend Aida's seed
(`local_operator/agent_seeds/aida.md`, "First contact") with a fifth optional step:
offer to bring over conversations/memory, list first, show what will be imported, never
modify the other harness. Bump the seed version. The agent path is the first-run path:
Aida can call `harness_import` during setup without the TUI/UI being open.

### 8.2 TUI

Add `/import` (alias `/migrate`) to `SLASH_COMMANDS` in `local_operator/slash_commands.py`
with subcommands: bare = source picker; `list`; `preview <source> [id]`; `run <source>
[id|all]`; `status`; `cancel`. Reuse the existing full-page mode/picker patterns
(`/settings`, `/resume`) rather than inventing a new widget stack. The page shows:
source cards with detected path/status/count; session rows with title, cwd, date,
message/compaction counts, availability; space to select, `a` all, enter to import; a
progress view with per-session rows and a result summary; warnings/errors and a
"re-run is safe" note. Add i18n catalogue entries (`wire.slash`) and the echo-policy
test row. Design/UX reviews use rendered SVG stills from the real app.

### 8.3 Desktop UI

Add a Settings section "Import conversations" in `~/local-operator-ui`:
source cards (Claude Code, Codex, omp), session list with search/select-all, preview
pane, import button, progress, results and warnings. It calls the desktop API only.
Add Storybook stories for loading/empty/error/populated states and a live-app browser
flow with screenshots for the same states. Design review (`D` findings) and UX review
(`U` findings) are required before merge; re-capture after visual fixes.

## 9. Test and QA strategy

Unit/integration tests (isolated `HOME`/`LOCAL_OPERATOR_CONFIG_DIR`, copied fixtures):

- Per-source fixtures: Claude synthetic records exercising user/assistant/tool_use/
  tool_result/thinking/sidechain/summary/compactMetadata; Codex synthetic rollout with
  session_meta/turn_context/response_item/event_msg/compacted/replacement_history/
  thread_rolled_back; omp real-shaped session with title/model_change/messages/
  custom_message/compaction/branch. Fixtures contain fake text only, no real user data.
- Compaction: summary/anchor/tokens/preserved-user-turns; multiple compactions; anchor
  missing; summary-only; replay result equals expected "summary + kept window"; pre-cut
  rows remain on disk; `compact_file` does not delete them.
- Tool structure: call/result pairing, multiple calls, orphan call, error result,
  interleaved user text.
- Timestamps/order: exact source times, monotonic fill, session created_at; no import
  time in any row.
- Idempotency/resume: twice = one session; append = only new rows; rotated source = new
  copy; user-edited target = no append; crash-prefix resume.
- Memory: chunking to atomic facts, candidate/active status, quota projection/skip,
  provenance/dedupe, secret refusal, no transcript bulk-ingest, instruction append
  requires confirmation.
- Security: source hash unchanged; SQLite WAL mtimes unchanged; socket block; redaction
  false-positive controls; symlink escape refused; malformed/truncated lines skipped.
- Bounds: large synthetic session stays under the RSS/entry caps; cancellation stops
  between batches.

Real-path QA per slice:

- Core: import a real omp session from a **copy** under an isolated root; run
  `lop`/`build_llm_history` on the target; inspect the transcript rows; re-run and
  compare; source hash unchanged.
- Agent/Aida: run `lop exec` in an isolated root with a stub source; call the tool
  actions; verify preview, approval, import, re-run no-op.
- TUI: `run_test` on the real app, drive `/import` through empty/populated/error states,
  `save_screenshot` SVGs, render and inspect; progress/cancel; re-run.
- UI: browser tool against the live app with fixture stores; screenshots for loading,
  empty, error, populated, progress, results; design/UX rounds.
- CI: affected tests/lints per iteration; full suite once at terminal frozen head; CI
  supplies the whole-tree pass. Independent QA is not replaced by CI.

## 10. Slice plan (serial; one PR per slice)

### Slice 1 — Mapping design note (this PR)
Docs-only. Acceptance: the note matches the real stores, the operator scope and the
slice order. Reviewer checks the mapping/compaction reasoning; QA re-checks the evidence
counts and paths. No code.

### Slice 2 — Core import service + API + fixtures/tests
New `local_operator/harness_import/`, `Transcript.append_imported`, server routes,
`lop import` CLI for QA. No TUI/UI/agent tool yet. Acceptance: import a copied omp
session end-to-end in an isolated root; replay matches source compaction semantics;
re-run no-op; source unchanged; unit matrix green; API preview/job/cancel works.
Independent reviewer + QA on the PR branch.

### Slice 3 — Agent tool + Aida setup
`harness_import` tool, `guide://harness-import`, Aida seed first-contact step, seed
version bump. Acceptance: tool lists/previews/imports in an isolated root; approval
required for writes; Aida prompt test pins the step; no network/source writes. Review +
QA.

### Slice 4 — TUI `/import`
Slash registry, picker/page, progress/results, i18n, echo policy. Acceptance: real TUI
frames for empty/populated/progress/error/results; select one/all; cancel; re-run safe.
Code review + QA + design/UX rounds with rendered SVG evidence.

### Slice 5 — UI Settings import
Desktop API client, Settings section, Stories, live-app screenshots. Acceptance: browse
Claude/Codex/omp; import one/all; progress/results; empty/error/populated states;
design/UX rounds clean on current head. Code review + QA + design/UX.

### Slice 6 — Release handoff
No version bump in any PR. After all merges, hand the PR list, merge SHAs and impact to
the current release-window owner. LOH-14 does not cut a tag or run `lop-update` itself.

Dependencies: 2 blocks 3/4/5. 3 and 4 both depend only on 2, but the operator asked for
serial slices; keep that order unless a scheduling reason appears. 5 needs the server
route from 2.

## 11. Risks and open questions

1. **Claude bodies are absent on this machine.** Fixture fidelity is the main risk. We
   verify field names against the installed Claude Code 2.1.274 binary and the omp
   fork's independent parser; QA uses a synthetic Claude store. If the operator later
   has real Claude transcripts, re-run the real-path matrix before calling the adapter
   complete. Do not claim real Claude import coverage from that machine.
2. **Most Codex bodies are missing here.** 1,203 thread rows but no rollout files; only
   two archived `.zst` bodies. The adapter must be honest about `metadata_only` sessions.
   `.zst` support depends on `compression.zstd` (runtime 3.14 yes; test venv 3.12 skip).
3. **Branching is inherently lossy in a linear target.** We import the active branch and
   preserve alternate-branch IDs in the manifest. If the operator expects every
   regeneration/edit branch as a separate conversation, that is a follow-up, not v1.
4. **Thinking blocks are dropped.** Lop has no thinking content type; they remain in the
   source store and are counted. Revisit if lop adds a thinking content block.
5. **Secret redaction is heuristic.** Conservative patterns plus the memory guard;
   preview shows redaction/refusal counts. An unlabelled secret can still pass, so the
   preview must say "review before importing" and never bulk-ingest transcripts.
6. **Memory quota.** The global bank is 16k active tokens; large foreign memory sets will
   be partially imported. Preview reports projected/quota and skipped counts; instruction
   files default to candidate facts, not active instructions.
7. **Scope drift.** The tracker text mentions skills/MCP/plugins/settings; the operator's
   brief does not. This workstream stays on memory + conversations. Confirm if the
   tracker should be narrowed or a sibling workstream opened.
8. **No release.** No PR bumps the version; the release owner handles the window.

## Appendix A — Evidence commands (re-runnable, read-only)

```sh
# Claude: no transcripts, prompt history only
ls ~/.claude/projects 2>&1
python3 - <<'PY'
import json,os,collections
c=collections.Counter()
for line in open(os.path.expanduser('~/.claude/history.jsonl')):
    try: c[json.loads(line).get('sessionId')]+=1
    except Exception: pass
print(len(c), sum(c.values()))
PY

# Codex: thread counts, rollout existence, memory rows
sqlite3 "file:$HOME/.codex/state_5.sqlite?mode=ro" \
  "select history_mode,count(*) from threads group by history_mode;"
sqlite3 "file:$HOME/.codex/state_5.sqlite?mode=ro" \
  "select id,rollout_path from threads order by updated_at desc limit 3;"
sqlite3 "file:$HOME/.codex/memories_1.sqlite?mode=ro" \
  "select count(*) from stage1_outputs;"

# omp: session files and entry shapes
python3 - <<'PY'
import glob,json,os,collections
fs=glob.glob(os.path.expanduser('~/.omp/agent/sessions/*/*.jsonl'))
c=collections.Counter()
for f in fs:
    for line in open(f):
        try: c[json.loads(line).get('type')]+=1
        except Exception: pass
print(len(fs), dict(c))
PY

# lop target: transcript count, sidecars, memory schema
find ~/.local-operator/sessions -name transcript.jsonl | wc -l
sqlite3 "file:$HOME/.local-operator/memory/memory.db?immutable=1" ".tables"
```

## Appendix B — Source field cross-check

- Claude: `uuid`, `parentUuid`, `logicalParentUuid`, `sessionId`, `timestamp`, `cwd`,
  `type`, `message.role`, `message.content` (`text`/`thinking`/`tool_use`/`tool_result`),
  `isMeta`, `isSidechain`, `isCompactSummary`, `compactMetadata` (`trigger`, `preTokens`,
  `preservedMessages`, `allUuids`, `anchorUuid`, `excludeUuids`), `summary`, `leafUuid`,
  `custom-title`, `ai-title`, `toolUseResult`.
- Codex: `session_meta` (`id`/`session_id`, `timestamp`, `cwd`, `originator`,
  `cli_version`, `source`, `model_provider`, `git`), `turn_context` (`model`),
  `response_item` (`message`, `function_call`, `function_call_output`, `custom_tool_call`,
  `custom_tool_call_output`, `reasoning`, `web_search_call`, `tool_search_call`),
  `event_msg` (`user_message`, `agent_message`, `agent_reasoning`, `thread_name_updated`,
  `thread_rolled_back`, `token_count`), `compacted` (`message`, `replacement_history`,
  `window_id`, `window_number`, `previous_window_id`), `compaction_summary`
  (`active_context_tokens_before/after`, `retained_image_count`,
  `compaction_summary_tokens`).
- omp: `session` (`id`, `timestamp`, `cwd`, `parentSession`), `title`/`title_change`,
  `model_change`, `message` (`role`, `content` blocks, `timestamp`, `parentId`),
  `custom_message` (`customType`, `content`, `attribution`), `custom`, `credential_pin`,
  `compaction` (`summary`, `shortSummary`, `firstKeptEntryId`, `tokensBefore`,
  `details`, `preserveData`, `warning`), `branch_summary`, `reset_boundary`.
