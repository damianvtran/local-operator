# Design: a first-class `sessions` tool (lifecycle ops + bounded transcript peek)

Status: proposal for implementation. Author: architect (lopdev team, deepseek-flash).
Base: worktree `sessions-tool-0b39`, branch `feat/sessions-tool-0930`, cut from `origin/main` @ `144a6fa86` (v0.64.9). All file:line references are against that tree.
Companion reading: `docs/design/peer-send.md` (the `send` tool's design note — this note mirrors its structure), `docs/EXEC.md` (the exec contract), `AGENTS.md` ("What an agent may start", "The tool-surface footprint ladder", "Visual validation", "Timing, flakes…").

## 1. Summary

Ship **one new built-in tool, `sessions`**, present only where a session can hold the delegation surface (createIf rung 3 — zero schema in sessions that cannot use it), with six ops: `list`, `info`, `spawn`, `resume`, `stop`, `peek`. `spawn` opens a parallel session **visible by default** (`--workstream`, carrying the opener), because the invisible-by-default CLI shape is the incident this tool exists to remove; `visibility="ephemeral"` is the explicit opt-out. The tool rides the existing exec guard at `cli.main` rather than duplicating it, returns structured receipts (session id, job id, pid when published, state, origin, sidebar visibility), and never overlaps `send` (no message delivery, no steer op), `task`/`hub` (no subagents), `wake`/`monitor`, or `jobs`. `peek` gives token-bounded transcript inspection — tail/head/cursor windows, a search-then-window mode, and a compact digest — built on the existing bounded readers ([`read_transcript_page`](local_operator/session/transcript.py), `read_replay_suffix`) and the hub-peek rendering discipline, never a whole-file parse. Advertising lands in the system prompt, `guide://agents`, `guide://peer-messaging`, the aida seed, and the docs site; visuals land in the TUI (glyph + summary), the mobile web app, and one UI-repo change (trace-row glyph). No version bump (release windows own that).

## 2. Problem and evidence

**The incident.** A dispatcher session spawned parallel work with the raw CLI and omitted `--workstream`. Because the flag is opt-in and `origin.json` is written once and immutable, the run was hidden from the operator's sidebar for its entire life and **could not be promoted afterwards** — the documented remedy is "start it as a workstream instead" (`docs/EXEC.md`:222-236). The invisible disposition is the default; the visible one is one easy-to-miss word. Parallel-session delegation should be impossible to get wrong.

**The current teaching surface points at the raw CLI** (grep over the worktree, 2026-09-30):

- `local_operator/agent_seeds/aida.md`:42-43 — teaches `lop exec --workstream <name> "<task>"`, which does not even parse: `--workstream` is `store_true` (`exec_startup.py`:186-196) and the name belongs to `--name`. The most privileged seed teaches a malformed invocation of the exact flag the incident missed.
- `local_operator/guides/agents/GUIDE.md`:54-77 — "Separate sessions: `lop exec`, only when the user asked".
- `local_operator/guides/peer-messaging/GUIDE.md`:241-290 — "`lop sessions` — what is running and what it costs".
- `local_operator/prompts_md/system.md`:173-180 — "`lop sessions` (via `bash`) lists them" (the peer-messaging paragraph the model reads every turn).
- `local_operator/tools/builtin.py`:12635 — the `send` tool's own description says "`lop sessions` lists what is running", pushing agents to bash for the list the `send` schema assumes.
- `docs/EXEC.md` (full flag table) and the docs site pages (§9) teach `lop exec` throughout.

**What already exists** (the reason this is a small tool, not a new subsystem): the flag, the stamp, the guard, the listings, the registry, the stop/resume machinery, the transcript readers, and the search index all exist; the tool is a typed front door over them with the visibility default flipped. §4 maps each op to its machinery.

**Footprint context** (measured for this note; cl100k via the repo's own tiktoken pin, `compaction/tokens.py`:26-33 — same ruler compares to same ruler): `send` params 446 tokens / 1,908 chars; `hub` 567 / 2,327; `task` 500 / 2,161; `wake` 389 / 1,511. The full default surface as built in a plain `ToolContext` (19 tools, name+description+parameters) measures **7,649 tokens / 33,113 chars** — so a new ungated tool is a real, permanent per-request tax, and the gate decision in §3 follows from it.

## 3. Op set, schema, and the gate

### 3.1 Tool identity

- Name: **`sessions`** (matches `lop sessions`; no existing tool collides — `tools/registry.py`:37-96).
- Label: `Sessions`.
- One tool, six ops — mirrors `hub`'s "ONE tool rather than eight" decision (`builtin.py`:21220-21227) instead of six one-op tools.

| op | what it does | approval tier |
|---|---|---|
| `list` | live + stored sessions, lean rows | read |
| `info` | one session: state, where it lives, origin/visibility/opener | read |
| `spawn` | open a parallel session (visible workstream by default) | write |
| `resume` | reopen a stored/stopped session headlessly | write |
| `stop` | end a running session (graceful ladder) | exec |
| `peek` | bounded transcript read: tail/head/cursor/search/digest | read |

`restart` is deliberately **not** an op: restarting a live session is `stop` + `resume`, two independently approved and independently audited calls; a compound op would hide a destructive step inside a convenience. `steer` is deliberately **not** an op — see §7.

### 3.2 Parameters (single flat schema, as `hub`/`send` do)

```jsonc
{
  "op": "list|info|spawn|resume|stop|peek",            // required
  "session": "a1b2c3d4e5f6",                            // info/resume/stop/peek: exact id
  "target": "release-crew",                             // info/resume/stop/peek: name/cwd substring
  "pid": 48213,                                         // info/stop: exact
  "include_stored": false,                              // list only
  "limit": 20,                                          // list only
  "query": "flaky shard",                               // list: filter via store search; peek: locate
  "prompt": "…",                                        // spawn required
  "name": "night-audit",                                // spawn
  "team": "release", "profile": "reviewer", "model": "anthropic/claude-sonnet-5",  // spawn
  "background": true,                                   // spawn/resume (default true)
  "visibility": "workstream|ephemeral",                 // spawn; default "workstream"
  "steps": 12, "head": 20, "before_id": "…", "around_id": "…",  // peek windows
  "regex": false,                                       // peek: query as regex
  "digest": false                                       // peek: compact fold instead of steps
}
```

Validation rules (fail with a legible refusal, never a guess):

- Addressing is **exactly one** of `session`/`target`/`pid`, resolved with the `send` tool's own resolver so the disambiguation text is byte-compatible with what models already know: `mobile/peer_send.py`:190 (`resolve_peer_target`), :474 (`resolve_cold_session`), :550 (`resolve_stored_target`), :850 (`candidate_lines`).
- `spawn` requires `prompt`; the other spawn-only fields are refused on other ops (and vice versa) — the tool must not silently ignore a misspelled intent.
- `visibility` is only meaningful for `spawn`; on `resume` it is refused with the immutability sentence (§5.3).
- `peek`: `steps` XOR `head` XOR (`before_id`/`around_id`); `regex` requires `query`; `digest` ignores windows.
- `stop` defaults to the graceful ladder only (`force=False` in v1 — §6).

### 3.3 The gate — rung 3, predicate `context.subagent_launcher is not None`

**Recommendation: rung 3 (`createIf` in `TOOL_BUILDERS`), factory returns `None` unless `context.subagent_launcher is not None`, and the name joins `SESSION_CAPABILITY_TOOLS` so the session's own merge re-runs the builder.** The population trim then comes from the two mechanisms that already exist (details below); spawn itself is additionally refused per call by the exec guard (§6.2).

Why not the candidates as stated:

- **`context.may_delegate` cannot be the build predicate as-is.** It is derived from the session's *pre-merge* inventory (`session.py`:12123: `any(tool.name == "task" for tool in self._tools)`), and **the contexts that decide the factory inventory all leave it False**: `session_factory.py`:4074-4088 (factory context) and `subagent.py`:2574-2608 (child construction context) use the field's default (`harness/types.py`:1362), while the merge that re-runs the builder asks a context whose answer is derived from the *pre-merge* list (`session.py`:4540-4542 reads `may_delegate` from `self._tools` at :12123, before the merge has added `task`; the front ends' own rebuilds read the same live list the same way, `tui/app.py`:28670-28685). So a `may_delegate` gate would strip the tool from **every** session that has not already got `task` — including every manager and every plain child of a delegating parent (`subagent.py`:2913-2921 keeps `task` for exactly that child) — and re-plumbing the merge order to fix it is a larger change than this tool justifies.
- **Unconditional presence like `send`** would put ~870 schema tokens (measured, §3.4) into every subagent run, the population that can never use `spawn`. That is what the ladder exists to prevent.
- **`subagent_launcher`** (`harness/types.py`:1343) is the session-owned launcher — the same prerequisite `task`/`wait`/`jobs` already gate on — and it is **present at merge time on every real Session** (`session.py`:12112). The result: the tool is built for top-level sessions and children alike, and the *existing* trim rules take over:
  - non-delegating children lose it in the prune that already strips `task` (`subagent.py`:2881-2921, "who may delegate is the role's decision, not the depth's");
  - a declared inventory (`--tools`) trims it through `_filter_declared` (`session.py`:4555-4561) unless the declaration names it;
  - a role allowlist that omits it keeps it only where the merge always has (`hub` is the precedent: `subagent.py`:2559-2564).

Trade-off, stated: a **top-level** session that cannot delegate (e.g. `lop exec --profile reviewer`) still carries the schema and sees `list`/`info`/`peek` (usable) plus a `spawn` that the guard refuses with the standard refusal text. That is the deliberate cost of not inventing a second gating convention; if a future review wants it tighter, tightening the predicate needs the merge-order refactor above, not a new mechanism.

Implementation notes: the builder lives in `tools/builtin.py`; the registry entry appends at the end of `TOOL_BUILDERS` (`registry.py`:37-96) and `DEFAULT_TOOL_NAMES` (`registry.py`:100-128) — appending never shifts the provider-visible array prefix the prompt cache keys on (the reason `project`/`monitor` appended, `registry.py`:88-95). `SESSION_CAPABILITY_TOOLS` (`session.py`:573-581) gains `"sessions"`. Existing pins: `test_default_set_builds_all_builtin_tools` (`tests/unit/tools/test_registry.py`:129-137) keeps passing unchanged because its `_engine_context()` already carries `subagent_launcher` (:93); add ONE delta test in the `test_default_set_drops_monitor_without_scheduler` shape (:150-153) asserting the tool's absence when the launcher is `None`; `test_default_names_cover_builder_table` (:338) stays true because the name joins both tables. The merge behaviour is covered where `task`/`hub` are.

### 3.4 Schema cost (measured; trim during implementation)

Draft measured for this note (18 params + description; counted with cl100k_base via tiktoken — the repo's own estimator ruler, `compaction/tokens.py`:23-33, and to be compared only against other figures from the same ruler): **params 781 tokens / 3,440 chars; description 92 tokens / 421 chars → ~873 tokens total**. That is above `hub` (567) and `send` (446). **Budget: hold the shipped params ≤ 700 tokens** by tightening Field descriptions (the draft's is deliberately verbose), and re-measure with `/context` on a real session in PR A (the ladder's own instruction: "when a tool's cost is in doubt, measure it", `AGENTS.md`:3506). The gate keeps this cost off every non-delegating child regardless.

## 4. Mapping to existing machinery (no new subsystem)

- **spawn / resume → the `exec` front end.** The tool does not mint sessions itself; it runs the same `lop exec` path every other spawn uses (a subprocess, §6.2), with argv fields mapped to `ExecArgs` (`exec_mode.py`:127-168; `workstream` at :147-149 is carried into the factory namespace at :728 and across the `--background` boundary by `STARTUP_FIELDS`, `exec_startup.py`:18-50). Detached worker, job record, and the bounded readiness receipt are `exec_mode._spawn_background` (`exec_mode.py`:542+; receipt prints job id, status, session id, `--status` handle and log path, :620-660). `resume` is the same path with `--resume <id>`; a live runtime is refused rather than raced (`docs/EXEC.md`:39).
- **visibility → the stamp.** `--workstream` selects `ORIGIN_AGENT_WORKSTREAM` at creation; the write happens in one place, `session_factory._prepare` (`session_factory.py`:3869-3900) via `agent_shell.stamp_agent_shell_session` (`agent_shell.py`:356-419), with opener attribution read at stamp time (`agent_shell.py`:422-484). Absent the flag the stamp is `ORIGIN_AGENT_SHELL` — hidden and silent. Nothing new to build; the tool just stops omitting the flag.
- **list / info → the published row shapes.** `info.collect.session_rows` (`info/collect.py`:614-740) is already extracted as `lop sessions --json`'s contract and reused by `/info` — the tool reuses it rather than deriving a second row shape (the CLI comment at :4840-4850 explains why the extraction happened). Live rows come from the registry scan (`collect.py`:271-330); stored rows from `_stored_lines` (`collect.py`:563-612); `state` semantics (live/wedged/stale/stored) are pinned there. For `info` only, add what a single-row read can afford: the session directory, the transcript path, `origin` (via the marker read path the catalog already uses, `session/catalog.py`:1460-1520) and `opened_by` (`resume.workstream_opened_by`, `resume.py`:3670-3710; keys `OPENED_BY_KEYS`, :3667). The desktop row already publishes exactly this shape (`server/models/desktop_sessions.py`:189-196) — follow it, do not invent a third spelling.
- **visibility predicate → `resume.is_user_session`** (`resume.py`:1253-1274): an allow-list over `USER_ORIGINS` (:142), the one shared fact every listing funnels through. The tool reports `sidebar_visibility: "listed" | "hidden"` from it, never from a re-derived rule.
- **stop → the existing kill switch.** `session.runtime.control.stop_session` (`control.py`:1271-1310; the ladder documented there), the same call `lop stop` makes (`cli.py`:7084). Never a raw signal from the tool; never `force=True` in v1.
- **live state → the runtime registry.** `registry.record_path` (:143), publish (:163), `classify` (:513), `scan` (:582); record fields incl. `busy` (`runtime/types.py`:1018-1195). This is what makes `list`'s state honest for a run that has not published yet vs one that has.
- **peek → the bounded readers** (§8), plus the step renderer `_render_transcript_steps` (`harness/comms.py`:3445+).
- **advertising → the classification roster** (§9): guides are candidates by frontmatter description (`guides/discovery.py`:19-64; roster built at `session_factory.py`:1846-1941; rendered `<resource_recommendations>` at `classification/recommend.py`:291-376).

## 5. Visibility semantics (the correctness core)

1. **`spawn` → `--workstream` by default.** The session is *created* by this call, so the stamp applies (`agent_shell.py`:356-419: `created_here` true, agent shell true). The tool only exists where delegation is allowed at all, and delegation happens when the operator asked for parallel work — so published-with-opener is the default, and the invisible disposition becomes the explicit choice. This is the incident fix.
2. **`visibility="ephemeral"`** → pass nothing → stamped `agent-shell`, hidden everywhere and silent. The tool text says what it is for: a throwaway run the operator did NOT ask to see. The description must carry the same condition the guides teach ("reach for parallel sessions when the user asked"), because the tool cannot verify intent — but unlike the CLI, the default is the safe direction.
3. **`resume` never re-stamps.** `created_here` is false for an adopted directory, and re-marking is refused by construction (`agent_shell.py`:405; `docs/EXEC.md`:231-236). The result reports the session's **unchanged** origin and visibility; if it was hidden it stays hidden — `origin.json` is immutable (`docs/EXEC.md`:222-230). The tool never writes or edits `origin.json`; a hidden session is re-opened as a **new** workstream instead, and the refusal/result text says so.
4. **Result fields per op make the property checkable, not inferred.**
   - `spawn`: `{session_id, job_id, name, state:"starting", origin:"agent-workstream", sidebar_visibility:"listed", opener:{agent,label,session}, log_path}` — pid is added when the discovery record is already published (`registry.record_path`), `null` until then; the receipt is readiness, not completion (`docs/EXEC.md`:40).
   - `ephemeral` spawn: `origin:"agent-shell", sidebar_visibility:"hidden"` and the text says the run is silent and how the operator can still reach it (`lop sessions` lists the live run; the agent's route back is `lop exec --resume <id>` — `docs/EXEC.md`:196-200).
   - `resume`: `origin`/`sidebar_visibility` echoed from disk, plus `{"visibility_changed": false}` made explicit in the text.
5. `--workstream` outside an agent shell stamps nothing (`exec_startup.py`:194) — irrelevant here because spawn runs with the agent marker set (§6.2), and that is what makes the stamp fire.

## 6. Approval tiers, safety, and the guard

### 6.1 Tiers

- Static `approval_tier="exec"` (the highest any op needs — `stop` ends a process), with a per-call override exactly like `hub`'s (`builtin.py`:22009-22018: "The gate is per TOOL, not per op, so the tier is the highest any op needs"): `call_approval_tier = read for {list, info, peek}, write for {spawn, resume}, exec for stop`.
- Rationale for `write` on spawn/resume: `hub`'s own comment — "`resume` starts a child session, which is exactly what `task` asks the user to approve" (`builtin.py`:22011-22014). `spawn` is that commitment one level up (a top-level session + process).
- `describe_approval` (model after `_describe_send_approval`, `builtin.py`:12590-12614 — consequence not mechanism, discriminator first, no tool-name repeat, ellipsis in cells):
  - spawn: `open "<name>" as a listed workstream (<team|profile|model>): <prompt…>`
  - stop: `stop <name> (<id8>, pid <pid>): ends its current run and releases the session lease`
  - resume: `resume <name> (<id8>): reopens its transcript headlessly`
- What never happens: no `--yolo`, no `--control`, no `--tools` pre-approval passthrough (a spawned run's approval posture is the operator's to set; record as a v1 exclusion, revisit only with evidence); no SIGKILL (`force=False`); no `origin.json` writes; no interactive launch — the interactive path stays refused for every agent shell (`docs/EXEC.md`:154-157) and this tool does not go near it.

### 6.2 The exec guard: ride it, never duplicate it

The spawn/resume subprocess gets a child environment built the way the `bash` tool builds one:

- `shell_env.child_environment(policy, parent, injections)` is THE helper for this ("so the next tool that spawns a child has a function to call rather than a pattern to copy", `tools/shell_env.py` module docstring) — call it, do not hand-copy.
- Injections: `NON_INTERACTIVE_ENV` (`builtin.py`:422-508) — which already includes the agent-shell marker `AGENT_SHELL_ENV: "1"` (:474) — plus the **three-arm delegation allowance** the bash tool writes (`builtin.py`:3794-3798): `MAY_DELEGATE_ENV="1"` when `context.may_delegate`, the empty string only to clear an inherited value, and the name omitted otherwise. The name is `agent_shell.MAY_DELEGATE_ENV` (`agent_shell.py`:152) — one name, one writer; if the three-arm block is worth a tiny shared helper during PR A, extract it so `bash` and `sessions` cannot drift (the code's own rule, builtin.py:3742-3743).
- Deciding remains `cli.main`'s job: the guard reads the marker and refuses with the standard text (`cli.py`:10744-10762; refusal copy `docs/EXEC.md`:161-166). The tool adds **no** predicate of its own at call time — it forwards the session's answer and reports the refusal verbatim if the guard declines. Fail-closed on both shapes of "no answer" (`context is None` / duck-typed context), per the bash tool's stated contract (:3787-3793).
- The subprocess launches the real CLI (`lop` entry point as installed; the nested-CLI resolution tests are `tests/unit/test_agent_shell_guard.py` — follow the way they and the harness rigs resolve it; do not hardcode a path).

### 6.3 Structured results

Every op returns this repo's `ToolResult` shape: a bounded text body plus a `details` mapping (cf. hub peek's `details={op, job_id, status, total, shown}`, `builtin.py`:21685-21691). Per-op details keys are pinned in §5.4 (spawn/resume) and §8 (peek); `list`/`info` return the row's published keys (`info.collect.session_rows` order; `info` extends with directory/transcript_path/opened_by). Any oversized body goes through `spill_truncate` (`builtin.py`:981-1008), which merges `{"spill": {...}}` into `details` and leaves the elided text recoverable via the harness spill handle — the same pattern the operator asked to reuse.

## 7. Authority boundaries

- **vs `send`**: `send` owns delivery (mailbox/wake/now), model switches, and their receipts (`execute_send`, `builtin.py`:12804+; approval descriptor :12590). `sessions` never delivers a message. **There is no `steer` op**: `send(now=True)` already steers and opens a turn, and the tool's description plus `guide://peer-messaging` point at it in so many words ("steering mid-turn is `send` with `now=True`" — matching `system.md`:176-177). A second steering door would drift from the peer protocol's ack/wake semantics on its first change.
- **vs `task`/`hub`**: subagent children. A subagent never publishes a `SessionRecord` of its own (`info/collect.py`:453-455), so children are structurally absent from `list` and unaddressable by this tool; `hub` stays the only child surface (`builtin.py`:21959+). The tool description states this ("top-level sessions and stored conversations only; subagents are `hub`'s").
- **vs `wake`/`monitor`**: scheduling is untouched; a spawned session may arm its own — that is its business, not this tool's.
- **vs `jobs`/`wait`**: those observe *this* session's background work. `sessions list` observes other sessions; `sessions stop` ends another runtime via the kill-switch ladder, never `jobs`' cancel.
- **vs `project`**: no project linkage in v1 (aida/manager flows keep using `project`).
- **vs the CLI**: the CLI remains for humans and for hosts without the tool (network verbs, scripts, cron). Guides teach tool-first, CLI-as-fallback; nothing is deprecated and no CLI flag changes.

## 8. Transcript inspection (addendum 2)

Design principle, borrowed from the harness's own instruction: make the cheap path the obvious path — the same philosophy as `hub op='peek'` for subagents (`builtin.py`:21639-21692: "the op exists so a parent can check progress without a transcript dump landing in its own context") and the spill pattern (`builtin.py`:981-1008; `spill://` handles).

### 8.1 Two-level location

1. **Which session?** `list` with `query` uses the existing store search — `session_search.search_store` (`session_search.py`:355) over digests built by `search_index.build_index` (:297; `digest_transcript_read` :174-226). Honest limit to document: the digest is a bounded head read (`SCAN_BYTES` 256,000, `DIGEST_CHARS` 4,000; roles user/assistant only — `search_index.py`:96-149), so the index **locates sessions, not positions inside them**. The tool says so when a match is returned from the index half.
2. **Where in the session?** `peek query=<text>` walks the transcript **backward from EOF** — the reader's own direction (`read_transcript_page` docstring, `transcript.py`:960-1006) — decoding rows until one contains the needle (literal; `regex=true` switches to `re.search` over decoded rows within the same budget). It then reads the window around the match with the existing anchored page mode (`around_id`/`before`/`after`; `validate_page_request` :644-679; `_PAGE_LOCATE_WINDOW_BYTES` 16 MiB, :693). Report `scanned_bytes`; when the budget is exhausted without a hit, return the honest miss plus a pointer to widen (`around_id` from a known id) — never scan a 262 MB journal silently (the deep-page cost data is in the reader's own docstring, :991-1006).

### 8.2 Windows: tail / head / cursor (and why ids, not hub-style indices)

- `steps=N` — last N steps (default 12; max 50). Sources from the tail page read (cheap: "the tail page costs the chunk that carries it").
- `head=N` — first N steps, for "how did this start".
- `before_id` / `around_id` — cursor paging using the entry ids the read returns (`TranscriptPage`, `transcript.py`:471-490: ids are stable across compaction; byte offsets are not). The reply's footer carries the continuation hint (`before_id=<oldest id> for earlier`) and `has_older`/`has_newer` from the page.
- **Deviation from `hub` peek, stated as a decision**: hub numbers steps 1..N with stable absolute indices (`comms.py`:274-292, `_resolve_peek_range` :3325) because it parses the child's whole transcript — subagent journals are small, and it says so in-code ("rendering is O(window) but parsing is O(total)", :2064-2077). Operator journals are not small (262 MB measured, `transcript.py`:999-1006), and a counted `total` would need a full scan — exactly what the bounded readers exist to avoid. So `sessions peek` windows by cursor id and reports booleans, not absolute indices. If the manager requires numeric ranges, that decision has a cost attached and must be taken explicitly.
- Rendering reuses `_render_transcript_steps`/`_render_message_step` (`comms.py`:3445+) and its clip discipline (`PEEK_STEP_CHARS` 600, `_clip` keeps head AND tail — :241, :3390+), fed from the bounded entries (not from a fresh `Transcript`, which is what hub peek builds).

### 8.3 Digest mode

`peek digest=true` reads the tail window (~40 entries, bounded by the same readers), folds it to ≤10 lines: counts per kind (user/assistant/tool), the newest user ask (≤80 chars), the newest assistant line (≤120), the last ≤8 tool calls as `name · summary`, and live state (busy/pending/has_older). `details={"op":"peek","mode":"digest","steps_seen":40,"tool_calls":7,"has_older":true}`. Precedent for the discipline: `digest_transcript`'s bounded, flattened, role-filtered fold (`search_index.py`:157-226) — but note it is a *search* digest; this is a *reading* digest, so the two stay separate functions with separate budgets.

### 8.4 Token budgets (per op; default invocations)

| op | shape | budget |
|---|---|---|
| `list` | up to 20 rows × ~12 clipped fields | ≤ ~600 tokens |
| `info` | one row + ≤8 extra fields | ≤ ~200 tokens |
| `spawn`/`resume` | one receipt line + details | ≤ ~120 tokens |
| `stop` | outcome sentence + details | ≤ ~60 tokens |
| `peek` (steps) | 12 steps × ≤600 chars | ≤ ~1,800 tokens |
| `peek digest` | ≤10 lines | ≤ ~150 tokens |

Anything over its budget is `spill_truncate`d (`{"spill": {...}}` in details), and the body carries the head+tail the repo's clipper keeps (`clip_head_tail`, `builtin.py`:517-521). The budgets are asserted by a test (§11.10).

## 9. Advertising plan

**Mechanism found, cited** — how resources are recommended today: the classification layer semantically matches the user's message against skills/guides/MCP servers/projects and injects a `<resource_recommendations>` block (`session_factory.py`:1846-1941 builds the roster; `classification/recommend.py`:291-376 renders it; `classification/service.py`:305 "Recommends skills, guides and MCP servers for one session's prompts"). **There is no "tool" candidate kind** — so a tool is advertised by (a) its own schema/description riding every request where present, (b) standing system-prompt text, and (c) the **guides' frontmatter descriptions**, which ARE the routing signals (`guides/discovery.py`:19-64: "Their frontmatter descriptions are semantic routing signals"). Follow (b) and (c); do not invent a tool-recommendation kind.

Edits (PR D):

1. `local_operator/prompts_md/system.md`:173-180 — extend the peer paragraph: `Other lop sessions … use the 'sessions' tool to list them, inspect one, or spawn a parallel session — the bare CLI ('lop sessions'/'lop exec --workstream') remains the fallback. The 'send' tool hands a message to one …`. One paragraph, same voice; the delegation condition ("when the user asked") stays attached to spawn, as in the guides today.
2. `local_operator/guides/agents/GUIDE.md`:54-77 — rewrite "Separate sessions" as: tool first (`sessions` op=`spawn`, visible workstream by default, when the USER asked), `lop exec --workstream` as the fallback for hosts without the tool; keep the guard rules and the "stamped" paragraph, now naming what `spawn` defaults to. Update the guide's frontmatter `description` to name the sessions tool.
3. `local_operator/guides/peer-messaging/GUIDE.md`:241-290 — add a tool-first lead to "`lop sessions` — what is running": use the `sessions` tool (`list`/`info`/`peek`); the CLI section stays (it is the human path). Update the frontmatter description likewise.
4. `local_operator/agent_seeds/aida.md`:41-51 — replace the launcher sentence with the `sessions` tool (`op=spawn`), fix the malformed `--workstream <name>` spelling (:42-43), keep the approvals sentence.
5. `local_operator/tools/builtin.py`:12635 — `send`'s description: "`lop sessions` lists what is running" → "the `sessions` tool lists what is running (`lop sessions` via bash as a fallback)".
6. `local_operator/guides/browser/GUIDE.md`:135 — stale `lop exec --resume` pointer; fix to name the tool where it fits (small).
7. **Docs site** — `~/local-operator-docs` (Next.js content repo; its own AGENTS.md governs): mirror the tool-first framing into `content/getting-started/choose-your-surface.md`:19-30 (the surface table), `content/capabilities/workflow-automations.md`:27, `content/capabilities/always-on.md`:139, `content/build/exec-mode.md` (the exec guide page), `content/about/why.md`:45. Human CLI stays; agent-first framing is added. Separate PR in that repo.
8. No new `guide://sessions` in this scope — the two updated guides already carry the triggers, and a third guide risks router noise; flagged as an open decision in §13 if the manager wants a dedicated surface.

## 10. Visual plan

Rules of the road: every surface below gets rendered **before/after frames** and a design round (per the standing rule and `AGENTS.md` "Visual validation": real `OperatorApp` for TUI, browser tool for web, never a CSS-less test host). The `sessions` tool must have a **distinct, consistent icon/label** everywhere tool activity renders, and no second mark where the session marker already exists.

1. **TUI tool-call rows (this repo, PR C).**
   - `local_operator/tui/glyphs.py`: add `"sessions"` to `NERD_TOOL_ICONS` (:82-136) and `PLAIN_TOOL_ICONS` (:147-189). Constraints are in-code: FA block only, one cell (`_single_cell` gate :199-217), and it must not collide with neighbours — `task`/`agent` already own `nf-fa-users` (:125-126) and `send` owns `nf-fa-paper_plane` (:127). Proposal: `nf-fa-window_restore` (two stacked windows = "a second session") with `nf-fa-clone` as alternate; plain fallback `□` (WGL4, unused; `wake`/`monitor` hold the circles). Final pick on rendered frames by the designer.
   - `local_operator/tui/widgets/tool_card.py`: add `"sessions"` to `_TOOL_CATEGORY` (:256-276) as `tool.row.name_meta` (coordination; reuse of an existing binding — no new CSS, no new binding-table entry). Add a `_summary_from_args` branch (:798-821) like the `send` special case (:808-813): lead with the discriminator — `spawn · workstream · <name>` / `stop · <target>` / `peek · <target> · last 12` — because the row truncates from the right and the mode must survive (the `_send_summary` lesson, :725-746).
2. **TUI session surfaces: NO new mark — a decision, not an omission.** A spawn writes exactly `ORIGIN_AGENT_WORKSTREAM`, and workstream rows are already marked everywhere the TUI lists them: `· agent-opened` in the picker (`session_picker.py`:1672-1683; literal `AGENT_OPENED_MARK` = `"agent-opened"`, `session/preview.py`:129) and `opened by <role>` in the sidebar (`session_sidebar.py`:1700-1705; `opener_role` `preview.py`:132). A second "spawned by sessions tool" mark would say the same thing twice; the opener already answers "whose work is this".
3. **Mobile web tool rows (this repo, PR C).** The phone row is state glyph + monospace tool name + summary (`mobile/web/src/components/tool-row.tsx`:15-24, :153-158) — **there is no per-tool iconography on mobile**, so the distinct treatment is (a) the name itself and (b) a summary special case mirroring the TUI: `mobile/projection.py` `_summarize_args` (:345-362) deliberately mirrors `_summary_from_args` ("Mirrors the TUI's contract", :346-352) — the two must change in the same PR, and the mobile tests (`tool-row.queued.test.tsx`, `session-view.*`) run under the repo's own vitest config.
4. **Mobile session list: add the agent-opened mark (recommended).** The phone list renders workstream rows (they are user-origin by `USER_ORIGINS`) but carries **no opener/origin** today (grep: no `opened_by`/`origin` anywhere under `local_operator/mobile/`), while TUI and desktop both mark machine-opened rows — the 2026-09-18 confusion class (a machine row read as the operator's) applies to the phone too. Recommended: daemon row shape gains `opened_by`/`origin` (mirror `server/models/desktop_sessions.py`:189-196) and `session-list.tsx` renders a small "agent" chip/subtitle; design round on frames. If the manager prefers not to widen scope, the alternative decision is an explicit "no mark" — flag (§13).
5. **Desktop UI (separate repo `~/local-operator-ui`).** Trace rows need a `sessions` tool glyph/label mapping (coordinate with the active `fix/trace-tool-labels` lane there — the TUI's `project` glyph comment already cross-references that work, `glyphs.py`:104-110). Session-list marker already exists on the desk; the backend needs no change (tool name rides the trace automatically; `opened_by` is already on the desk wire). One PR in that repo, not ours.
6. **Backend surface list for 1-5**: tool name `sessions` in trace rows (automatic), summary text (TUI + mobile special cases), glyphs (TUI only), session-row origin/opener (already desk-side; needed mobile-side if (4) lands).

## 11. Tests

Homes: `tests/unit/tools/test_sessions_tool.py` (new; model on `test_send_tool.py` — real in-process substrate, no mocks of the tool's own logic), `tests/unit/test_agent_workstream.py` (visibility/stamp cases — the file that already pins this machinery), `tests/unit/session/` (peek/reader cases), `tests/unit/tui/test_tool_card.py` + the glyph coverage gate, mobile vitest files for the web rows. Matrix:

1. **Surface/gating**: default `create_tools(ToolContext())` excludes `sessions`; a context with a launcher includes it; a non-delegating child loses it through the prune; a declared inventory drops it unless named. Fail direction: assert the gated absence (like the existing carve-outs `test_registry.py`:135-165).
2. **Spawn builds the right call**: executor-level assertion on the constructed argv/env — `--workstream` present by default; `AGENT_SHELL` injected; `MAY_DELEGATE` three-arm truth table (may_delegate true → `"1"`; false with inherited name → `""`; false, not inherited → absent). Prove-can-fail: delete the `--workstream` write → the test must fail.
3. **Spawn end-to-end** (isolated `HOME`+`LOCAL_OPERATOR_CONFIG_DIR`; real nested CLI against test hosting): session directory created, `origin.json` == `agent-workstream`, opener fields present, row passes `is_user_session`; with `visibility="ephemeral"` → `agent-shell`, hidden; receipt fields (session id/job id) round-trip.
4. **Guard passthrough**: a session whose context says may_delegate False gets the standard refusal text and creates nothing (fail-closed on `context is None` too).
5. **list/info shapes**: pinned `details` key order; `include_stored` behaviour; live/stored/wedged state strings (fixture records); `info` on a workstream returns `origin`/`opened_by`/`sidebar_visibility`; ambiguous target returns the shared candidate lines.
6. **stop**: per-op tier table (`call_approval_tier`), describe_approval sentence pinned character-level (pattern: `tests/unit/tools/test_approval_descriptions.py`), refusal paths (no target / pid mismatch / ambiguous), graceful-only (`force` never passed).
7. **resume**: never passes `--workstream`; origin.json bytes unchanged for a hidden session (prove-can-fail: flip the flag, bytes change); live runtime → refusal sentence.
8. **peek range on a large transcript**: fixture journal sized to make a whole-file parse detectable; assert the tail read's touched bytes stay under a bound derived from the reader's own chunking (structural assert — bytes/IO, not elapsed time, `AGENTS.md` "Timing"); head read likewise; two-page cursor round-trip via `before_id`.
9. **search-then-window**: seeded needle deep in the fixture returns the matching step with `scanned_bytes` ≤ cap; a needle beyond the cap returns the honest miss (prove-can-fail: move the needle past the cap).
10. **digest shape**: pinned fold for a seeded transcript; counts correct; empty/one-turn cases; and the **per-op token-budget guard**: each op's default output measured against §8.4 (character proxy + optional tiktoken, mirroring `compaction/tokens.py`'s fallback discipline) — fails on regression.
11. **Visuals**: before/after frames for the TUI tool row and the picker/sidebar mark; mobile row frames; design round (`### Design review — round N`, D-findings) on those frames; UX round only if the phone flow changes shape.
12. **Isolation and flake discipline**: every test above runs under `env -i HOME=$ISO LOCAL_OPERATOR_CONFIG_DIR=$ISO/.local-operator` (AGENTS.md "Isolating a run"), TUI shots under `env -u NO_COLOR TERM=xterm-256color`, never the operator's live sessions, `CMUX_*` unset for anything that boots a TUI (team rules), kills scoped to own pids. Waits subscribe to events (`ChangeSignal`), never clock sleeps.

## 12. PR split & sequencing

- **PR A — core lifecycle tool.** `tools/registry.py` (append to both tables), `tools/builtin.py` (builder gate, schema, execute for `list`/`info`/`spawn`/`resume`/`stop`, env signing via `shell_env` + the three-arm helper), `session/session.py` (`SESSION_CAPABILITY_TOOLS` + the merge-pin updates), tests 1-7. Smallest self-contained slice; lands first.
- **PR B — peek.** `tools/builtin.py` (peek + digest + spill + budgets) and tests 8-10. Extends A's schema; sequential in the same file — one review round while A's is still fresh is acceptable, but it must not merge before A.
- **PR C — visuals.** `tui/glyphs.py`, `tui/widgets/tool_card.py`, `mobile/projection.py`, `mobile/web/src` (+ `mobile/types.ts`/daemon row if the phone mark lands), tests 11. File-disjoint from A/B; lands after A (the summary branch assumes the tool); design round required.
- **PR D — advertising/docs.** `prompts_md/system.md`, `guides/agents/GUIDE.md`, `guides/peer-messaging/GUIDE.md`, `agent_seeds/aida.md`, the `send` description line, browser guide fix; docs-site PR in `~/local-operator-docs` separately. Lands after A (its text teaches a tool that must exist).
- Ordering rationale: A is the correctness fix (visibility default), B the context-economy half (addendum 2), C/D the surface/teaching half. Each PR carries its own tests; no PR carries a version bump (`AGENTS.md` release rules).

## 13. Risks / open decisions for manager

- **Gate predicate**: `subagent_launcher`-presence (recommended; works with the existing merge/prune/declaration filters) vs `may_delegate` (structurally False at every construction context today — `session_factory.py`:4074-4088, `subagent.py`:2574-2608 omit it; `session.py`:12123 derives it from the pre-merge list). If the coder finds a cheaper accurate predicate, take it **only if** the role-less-child case stays correct (a plain child of a delegating parent keeps `task`, `subagent.py`:2913-2921 — it must keep the tool too).
- **Default-visible spawn**: sign off on the exact description/guide wording that carries the "user asked for it" condition, and on whether `visibility="ephemeral"` ships in v1.
- **Peek indexing**: cursor-id windows instead of hub-style numeric step ranges — sign off on the deviation (cost honesty vs familiarity); numeric ranges imply a counted total or a persisted count.
- **Phone agent-opened mark**: add (recommended) vs an explicit no-mark decision.
- **Nested CLI resolution**: how the tool finds the `lop` launcher (console script vs `sys.executable`) across the uv-tool install and editable venv — follow the nested-spawn precedent in `tests/unit/test_agent_shell_guard.py`; verify in PR A on both install shapes.
- **Schema trim**: 873 measured tokens (draft) → hold ≤700 params before A merges; re-measure via `/context` on a live session.
- **Spill fit**: confirm `spill_truncate` applies cleanly to peek bodies at these sizes (it is the builtin-level helper; verify during B).
- **`restart` as two calls** and **no `steer` op** — confirm both read as intended.
- Optional: a dedicated `guide://sessions` if the manager wants a recommendation-engine surface of its own (recommended against, §9.8).

## 14. Manager decisions (locked 2026-09-30)

Resolutions for §13, locked by the coordinating manager before implementation:

1. **Gate predicate — approved as recommended** (`context.subagent_launcher is not None`). The per-call exec guard remains the enforcement; a top-level session that cannot delegate carrying the schema (and a refused `spawn`) is the accepted cost of not inventing a second gating convention. If the coder finds a cheaper *accurate* predicate, it may adopt it ONLY if the role-less-child case keeps both `task` and the tool.
2. **Default-visible spawn — approved; `visibility` ships, default `"workstream"`, `"ephemeral"` as the explicit opt-out.** The default direction is the fix; an explicit `ephemeral` is a deliberate act (not an omission), and it keeps throwaway runs on the tool path instead of pushing them back to the raw CLI. The spawn description carries the "when the USER asked for parallel work" condition, and every spawn/resume result names the visibility it produced.
3. **Peek windows by cursor id — approved.** No counted total on operator-size journals; `steps`/`head` still give last/first N; report `has_older`/`has_newer`/`scanned_bytes`; an honest miss over a silent slow scan.
4. **Phone agent-opened mark — ADD** (daemon row gains `opened_by`/`origin` mirroring the desktop row shape, small chip/subtitle on the phone session list; frames + design round).
5. **Nested CLI resolution — follow the `tests/unit/test_agent_shell_guard.py` precedent; verify both install shapes (uv-tool + editable venv) in PR A; never hardcode a path.**
6. **Schema trim — hold shipped params ≤ 700 tokens; state the re-measured number and its method in the PR A body.**
7. **Spill — apply `spill_truncate` to over-budget op bodies; verify fit at PR B.**
8. **`restart` = `stop` + `resume` (two auditable calls); no `steer` op — confirmed** (documented pointer to `send(now=True)`).
9. **Add a dedicated `guide://sessions`** (overturns §9.8's recommendation-against): its frontmatter description is the routing signal for session-work queries; `agents`/`peer-messaging` keep the narrative and point at it; harness-agnostic wording; if review finds router competition, fold it back into the agents guide.

Process riders (binding on the implementation):
- Four PRs as §12: A (core lifecycle) → B (peek), C (visuals; carries the design/UX rounds), D (advertising/docs; includes the docs-site PR in `~/local-operator-docs`, coordinated).
- No version bump in any PR; every PR body carries a collector-shaped `Release:` line (A argues `minor` — a new user-noticeable surface, window owner decides; B/C/D argue `patch`).
- PRs: assigned `damianvtran`, no reviewer requests, no `@`-tags, opened non-draft.
- Aida is flagged at the design note, each PR open, and each merge.
