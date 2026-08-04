# omp Recon: Skills + Compaction Subsystems

> **Delivery note:** the assignment asked me to write this to `local://scout-omp-skills-compaction.md`. No `write` tool was provisioned to this scout (read/grep/glob only), and the `node_repl` MCP is sandboxed read-only — `writeFile` returns `EPERM` for the session `local/` dir, `~/tmp`, `/tmp`, and `os.tmpdir()` alike. The full report is therefore delivered inline here; persist verbatim to that path if the file is still wanted. All findings are complete.

Scope: read-only map of `~/oss/oh-my-pi` (TypeScript, Bun monorepo). **(A) Skills** — discovery, prompt injection, `skill://` reads. **(B) Compaction** — triggers, flow, history replacement, prompt-cache interaction. Paths are relative to `~/oss/oh-my-pi` unless absolute.

---

## 1. Skill on-disk format

### Layout

A skill is **a directory containing `SKILL.md`**. The directory name is the default skill name; frontmatter `name` overrides it.

```
<skills-root>/
  <skill-dir-name>/
    SKILL.md          <- REQUIRED. frontmatter + markdown body
    references/       <- optional, free-form; e.g. references/sdlc.md
    scripts/          <- optional; e.g. scripts/probe.ts
    templates/        <- optional
    agents/           <- optional; e.g. agents/openai.yaml (observed in the wild)
    <anything else>
```

Only `SKILL.md` is parsed. **Every other file/subdir is opaque** — no manifest, no index, no registry of `references/`. The model reaches them purely by path, via `skill://<name>/<relative-path>` or an absolute path under the skill's `baseDir`. That *is* the progressive-disclosure mechanism (§3).

Real examples on this machine:
- `.omp/skills/tool-prompt-optimization/{SKILL.md, scripts/probe.ts, scripts/probe-builtin.ts}`
- `.omp/skills/semantic-compression/SKILL.md` (single file)
- `~/.omp/agent/skills/minerva-software-development/{SKILL.md, references/sdlc.md, agents/openai.yaml}`

### SKILL.md frontmatter

YAML frontmatter delimited by `---`, parsed by `parseFrontmatter` (`@oh-my-pi/pi-utils`) inside `scanSkillsFromDir` (`packages/coding-agent/src/discovery/helpers.ts:365`). Recognized keys (`SkillFrontmatter`, `packages/coding-agent/src/capability/skill.ts:11`):

| key | type | effect |
|---|---|---|
| `name` | string | Invocation name. Falls back to `path.basename(path.dirname(skillPath))` when absent/blank. |
| `description` | string | The **only** thing besides the name injected into the system prompt. |
| `enabled` | boolean | `false` → skill skipped entirely at scan time. |
| `hide` | boolean | Loaded and reachable via `skill://` and `/skill:<name>`, but **omitted from the system-prompt listing**. |
| `disable-model-invocation` | boolean | Agent Skills standard equivalent of `hide`; normalized to `disableModelInvocation` and OR-ed into `hide`. |
| `globs` | string[] | On the type; consumed by the *rules* surface, not by skill injection. |
| `alwaysApply` | boolean | Same — declared, not consumed by skill injection. |
| (anything else) | unknown | Retained on `frontmatter`, ignored. |

The body is stored as `Skill.content` at scan time but is **never injected into the system prompt** — read on demand only.

`requireDescription: true` is passed by the OMP-native, `.agent[s]`, `github`, and `omp-plugins` providers: a `SKILL.md` with no `description` is **silently dropped**. Claude/Codex/opencode do not require it (empty-string fallback).

### Sample (real)

```markdown
---
name: semantic-compression
description: Aggressively remove grammatical scaffolding LLMs reconstruct while
  preserving meaning-carrying content. Output may be fragments. Use when compressing
  text for prompts, reducing token count, preparing context for LLM input, ...
---

# Semantic Compression

LLMs reconstruct grammar from content words. ...
```

Description convention: **what it does + explicit "Use when ..." trigger phrases**. That string is the entire routing signal — there is no embedding index.

### Discovery roots (multi-provider)

`loadCapability("skills")` fans out over registered providers, each with a `priority`; results dedupe by `name`, highest priority winning.

| provider id | roots | file |
|---|---|---|
| `native` | `<ancestors>/.omp/skills/` (walk-up cwd→repoRoot), `~/.omp/agent/skills/` | `discovery/builtin.ts:282` |
| `managed-skills` | `~/.omp/agent/managed-skills/` (auto-learn writes) — **lowest** priority, always discovered | `discovery/builtin.ts:314` |
| `agents` | `.agent/skills`, `.agents/skills` (project walk-up + `~`) | `discovery/agents.ts:64` |
| `claude` | `~/.claude/skills`, `<ancestors>/.claude/skills` | `discovery/claude.ts:167` |
| `claude-plugins` | `~/.claude/plugins/cache/**`, honoring `marketplace.json` `skills` (additive to the `skills/` fallback) | `discovery/claude-plugins.ts:194` |
| `codex` | `~/.codex/skills`, `.codex/skills` | `discovery/codex.ts:237` |
| `opencode` | `~/.config/opencode/skills`, `.opencode/skills` | `discovery/opencode.ts:228` |
| `github` | `.github/skills/<name>/SKILL.md` | `discovery/github.ts:283` |
| `omp-plugins` | `<extension-package>/skills/` | `discovery/omp-plugins.ts:54` |
| `custom` | `skills.customDirectories` setting (tilde-expanded) | `extensibility/skills.ts` |

Scan is **non-recursive**: only `<dir>/<child>/SKILL.md` (plus `<dir>/SKILL.md` when `includeSelf`, Claude-plugin manifests only). Dot-prefixed entries skipped. Symlinked dirs followed; `fs.realpath` dedupe prevents double-loading one file through two roots.

### Post-scan filtering + ordering (`loadSkills`, `extensibility/skills.ts:118`)

1. Master gate `skills.enabled`.
2. Per-source toggles: `enableCodexUser`, `enableClaudeUser`, `enableClaudeProject`, `enablePiUser`, `enablePiProject`, `enableAgentsUser`, `enableAgentsProject`; unnamed third-party providers fall through to "any third-party toggle on". `managed-skills` is exempt.
3. `disabledExtensions` entries of the form `skill:<name>`.
4. `ignoredSkills` / `includeSkills` — **glob patterns over the skill name** (`new Bun.Glob(pattern).match(name)`). CLI `--skills a,b` overrides `includeSkills`; `--no-skills` empties the list.
5. Name collision: first (highest-priority) wins; loser emits a `SkillWarning`.
6. Sort: `compareSkillOrder` — name case-insensitive, then exact name, then path. Comment: **"Deterministic ordering for prompt stability."**

Result: `{ skills: Skill[], warnings: SkillWarning[] }`, `Skill = { name, description, filePath, baseDir, source, hide?, _source? }` (`baseDir` = `filePath` minus trailing `/SKILL.md`). Stashed in a **process-global** snapshot via `setActiveSkills(...)` / `getActiveSkills()` (`extensibility/skills.ts:44-57`), which the `skill://` handler reads.

---

## 2. How skill names + descriptions get into the system prompt

### Injection point — `packages/coding-agent/src/system-prompt.ts`

- **L716-721**: skills come from `options.skills` (pre-loaded, the normal path) or lazily from `loadSkills({ ...skillsSettings, cwd })`. Wrapped in `withDeadline("loadSkills", ..., fallback [])` — **best-effort and time-boxed**; a slow disk degrades to no skills rather than blocking startup.
- **L836-840**, the render filter:
  ```ts
  const hasRead = toolNames.includes("read");
  const filteredSkills = hasRead ? skills.filter(skill => skill.hide !== true) : [];
  ```
  → **no `read` tool ⇒ no skills listed at all** (the model could not fetch them).
- **L870**: `skills: filteredSkills` into template data. **L899**: `prompt.render(...)`.

### Format — `prompts/system/system-prompt.md:26-34`

```
# Skills & Rules
{{#if skills.length}}
Skills are specialized knowledge. If one matches your task, you MUST read `skill://<name>` before proceeding.
<skills>
{{#each skills}}
- {{name}}: {{description}}
{{/each}}
</skills>
{{/if}}
```

Wire output: one flat line per skill — `- <name>: <description>` — inside a `<skills>` tag. **No body, no file path, no token budget, no truncation.**

The custom-system-prompt variant (`prompts/system/custom-system-prompt.md:31-41`) uses a richer element and a different lead-in ("Scan descriptions for your task domain."):

```
<skills>
<skill name="{{name}}">
{{description}}
</skill>
</skills>
```

The `skills` array is referenced again at `system-prompt.md:184` ("Read relevant skills and rules first").

### Accounting

`estimateSkillsTokens` (`modes/utils/context-usage.ts:75`) counts `name` + `description` per skill and subtracts that from the system-prompt bucket so the TUI shows a separate "Skills" line. `renderedSkills()` (L67) mirrors the `buildSystemPrompt` filter exactly — mirror this in a port, since drift makes the accounting lie.

---

## 3. How `skill://<name>` reads work (progressive disclosure)

### Handler — `internal-urls/skill-protocol.ts`

`SkillProtocolHandler`, `scheme = "skill"`, `immutable = true` (cacheable, never revalidated).

`resolve(url, context)`:
1. Skills from `context.skills ?? getActiveSkills()`.
2. `skillName = url.rawHost || url.hostname`; empty → error.
3. Exact-name lookup. Miss → error listing **all** available names (`"Unknown skill: X\nAvailable: a, b, c"`) — the model's self-correction path.
4. No path → `skill.filePath` (the `SKILL.md`), or `skill.baseDir` when `context.pathOnly === true` (tools wanting a filesystem path, e.g. bash).
5. With path → `path.join(skill.baseDir, decodeURIComponent(urlPath.slice(1)))` after `validateRelativePath()` (rejects absolute paths and any `..` segment), **plus** a second `path.resolve` containment check against `baseDir`.
6. `fs.stat`: directory → `buildDirectoryResource(...)` (so `skill://name/references` enumerates); file → full text via `Bun.file(targetPath).text()`.
7. Returns `InternalResource { url, content, contentType ("text/markdown"|"text/plain"), size, sourcePath, notes: [] }`.

`complete()` returns `{ value: name, description }` per active skill (TUI autocomplete only).

### What gets loaded when — four paths

| trigger | loaded | wrapper |
|---|---|---|
| model `read("skill://<name>")` | `SKILL.md` full text as a tool result | none — raw file content |
| model `read("skill://<name>/references/x.md")` | that file only | none |
| user `/skill:<name> [args]` | `SKILL.md` body, frontmatter stripped | `prompts/skills/user-invocation.md` |
| autoload (agent frontmatter `autoloadSkills`) | same body | `prompts/skills/autoload.md` |

`buildSkillPromptMessage(skill, args, invocation)` (`extensibility/skills.ts:487`) reads `skill.filePath`, strips frontmatter with `content.replace(/^---\n[\s\S]*?\n---\n/, "").trim()`, renders one of the two templates, returns `{ message, details: { name, path, args?, lineCount } }`. Appended as a `custom_message` entry of type `SKILL_PROMPT_MESSAGE_TYPE` (`modes/skill-command.ts`), `attribution: "user"` on the user path.

`parseSkillInvocation(text)` supports leading `/skill:foo bar` **and** mid-prompt `fix the bug /skill:foo focus on auth` (surrounding prose collapses into `args`). Mid-prompt detection is suppressed when the draft starts with another slash command or a local-execution sigil (`!`, `!!`, `$ `, `$$ `).

`user-invocation.md` verbatim:
```
[IMPORTANT: The user has invoked the "{{name}}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]

{{body}}

---

[Skill directory: {{baseDir}}]
Resolve any relative paths in this skill (e.g. `scripts/foo.js`, `templates/config.yaml`) against that directory using its absolute path: read referenced assets and templates, and run scripts with the terminal tool when the skill's instructions call for it.
{{#if userArgs}}
User: {{userArgs}}
{{/if}}
```

`autoload.md` is deliberately minimal (provenance only; must not claim the user invoked it):
```
{{body}}

---

Skill: {{filePath}}
{{#if userArgs}}
User: {{userArgs}}
{{/if}}
```

### The ladder (3 rungs, no machinery)

1. **Always in context**: `- name: description` (~20-60 tokens each).
2. **On demand, model-chosen**: `SKILL.md` body via `read("skill://name")`.
3. **On demand, skill-chosen**: `skill://name/references/*.md`, `scripts/*` — the `SKILL.md` body simply *tells the model* to read them. Nothing enumerates or preloads.

Two reinforcements make rungs 2/3 stick:
- System-prompt hard rule: **"If one matches your task, you MUST read `skill://<name>` before proceeding."**
- Prune/shake protection: `skill` tool results and `read` results whose path starts with `skill://` are protected, so a loaded skill is never truncated out from under the model — `DEFAULT_PRUNE_CONFIG.protectedTools = ["skill", isSkillReadToolResult]` (`compaction/pruning.ts:54`), predicate at `compaction/tool-protection.ts:38`; likewise in `DEFAULT_SHAKE_CONFIG` / `AGGRESSIVE_SHAKE_CONFIG` (`shake.ts:47,58`).

### Writing skills back (auto-learn)

`tools/manage-skill.ts` + `autolearn/managed-skills.ts`: `create|update|delete` a `SKILL.md` under `~/.omp/agent/managed-skills/` **only**. Frontmatter generated from `name` + `description`. `isNameClaimedByAuthoredSkill()` refuses names already taken by an authored skill (managed skills resolve last and would be silently shadowed). Memory consolidation (`prompts/memories/consolidation.md`) emits the same shape: `skills[].name → skills/<name>/SKILL.md`, plus `scripts/`, `templates/`, `examples/` buckets.

---

## 4. Semantic search over skills? **No.**

Verified by grepping `embedding|semantic|vector|similarity|fuzzy` across `extensibility/`, `discovery/`, `internal-urls/` — zero skill-related hits.

**Pure static listing**: every non-hidden skill's name+description sits in the system prompt every turn; selection is the model reading those descriptions. The only search-ish things nearby are (a) `Bun.Glob` name matching for `includeSkills`/`ignoredSkills` config filters and (b) client-side fuzzy filtering of `complete()` candidates for TUI autocomplete (`internal-urls/types.ts:187`: "The caller fuzzy-filters the returned set against the partially typed query") — not model-facing.

Implication: prompt cost is O(total skills) forever. omp's counterweights: the description *is* the whole payload; `hide` / `disable-model-invocation` for opt-in skills; glob include/ignore config; deterministic sort so the block is byte-stable across turns (prompt-cache friendly).

---

## 5. Compaction: triggers, flow, replacement

Reference: `docs/compaction.md` (444 lines, current and authoritative).

### 5.1 Session entry model

Compaction is a **first-class session entry**, not a message rewrite.

```ts
CompactionEntry {
  type: "compaction";
  summary: string;
  shortSummary?: string;
  firstKeptEntryId: string;                 // the compaction boundary
  tokensBefore: number;
  details?: { readFiles: string[]; modifiedFiles: string[] };
  preserveData?: Record<string, unknown>;   // snapcompact frames, openai remote payloads
  fromExtension?: boolean;
}
BranchSummaryEntry { type: "branch_summary"; fromId; summary; details?; fromExtension? }
```

History is **append-only**. Nothing is deleted; `buildSessionContext()` just stops replaying:
1. Latest compaction on the active path → one `compactionSummary` message.
2. Entries from `firstKeptEntryId` onward re-included verbatim.
3. `branch_summary` → `branchSummary` message; `custom_message` → `custom` message.
4. `convertToLlm()` renders `compactionSummary`/`branchSummary` as **user** messages via static templates (`compaction/prompts/compaction-summary-context.md`, `branch-summary-context.md`); `custom` passes through as a developer message.

`compaction-summary-context.md` verbatim:
```
Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that model used. You MUST build on the work already done and NEVER duplicate it. Here is that summary:

<summary>
{{summary}}
</summary>
```

The **display transcript** is separate (`buildSessionContext({ transcript: true })`): the TUI shows all entries chronologically with the compaction as an inline `── 📷 compacted · ctrl+o ──` divider. Only the LLM context resets.

### 5.2 Triggers (six) — `session/session-maintenance.ts`

1. **Manual** `/compact [instructions]` → `AgentSession.compact(...)` → `SessionMaintenance.compact()` (L574).
2. **Overflow recovery** — assistant error classified as context overflow, not older than the latest compaction. Failing message removed from active state; **context promotion tried first** (`#tryContextPromotion` L1415 — switch to a larger-context sibling model, retry without compacting); else compaction with `reason: "overflow", willRetry: true`. `handoff` is disallowed here (its request would reuse the overflowing input). On success, `agent.continue()`.
3. **Incomplete-output recovery** — assistant `stopReason === "length"`. Same shape, `reason: "incomplete"`; `handoff` **is** allowed (input still usable).
4. **Post-turn threshold** — successful assistant message whose adjusted context exceeds `resolveThresholdTokens(...)` (`checkCompaction()` L1194).
5. **Mid-turn threshold** — `maintainContextMidRun()` (L1068) before the next provider request inside a tool loop, gated by `compaction.midTurnEnabled !== false`, only at safe tool-loop boundaries; plus `runPrePromptCompactionIfNeeded()` (L1012).
6. **Idle** — `runIdleCompaction()` (L974) when not streaming and not compacting; `reason: "idle"`, never auto-continues.

### 5.3 Threshold math — `packages/agent/src/compaction/compaction.ts`

```ts
DEFAULT_RESERVE_TOKENS = 16384;
MAX_SUMMARY_TOKENS     = DEFAULT_RESERVE_TOKENS;   // hard cap on a generated summary

effectiveReserveTokens(w, s)   = max(floor(w * 0.15), s.reserveTokens ?? 16384)   // L304
resolveBudgetReserveTokens(w, s)                                                  // L320
  // if the reserve was DEFAULTED and is impossible for this window, or >= window,
  // use max(1, floor(w * 0.15)); an EXPLICIT reserve is always honored
resolveThresholdTokens(w, s)                                                      // L359
  // 1. s.thresholdTokens > 0   -> clamp(s.thresholdTokens, 1, w - 1)
  // 2. s.thresholdPercent > 0  -> floor(w * clamp(pct, 1, 99) / 100)
  // 3. else                    -> clamp(w - resolveBudgetReserveTokens(w, s), 0, w - 1)
shouldCompact(ctxTokens, w, s) = s.enabled && s.strategy !== "off" && w > 0
                                 && ctxTokens > resolveThresholdTokens(w, s)      // L334
```

Token source: `compactionContextTokens(providerCtxTokens, storedEstimate) = max(...)` (L355) — provider usage is ground truth **floored by the agent's own estimate of stored history**, because a `before_provider_request` compression hook (or inline snapcompact) can deflate what the provider sees while real history grows unbounded.

`calculateContextTokens(usage)` prefers `usage.contextTokens`, else `totalTokens` (or `input+output+cacheRead+cacheWrite`) **minus provider orchestration tokens** (billed but never replayed into the prefix).

Local estimation: `estimateTokens(message, { excludeEncryptedReasoning? })` (L407), cl100k_base via a native tokenizer, `IMAGE_TOKEN_ESTIMATE = 1200`, memoized per message (§6.7).

Hysteresis: `COMPACTION_RECOVERY_BAND = 0.8` (`session-maintenance.ts:166`). A pass counts as having created headroom only when residual ≤ `0.8 × threshold`; without it, a pass that shaves just under the line every turn sustains an auto-continue dead loop (issue #2275) and the snapcompact re-fire thrash.

### 5.4 Boundary + cut point

`prepareCompaction(pathEntries, settings, activeModel)` (`compaction.ts:1212`):
1. `undefined` if the last entry is already a compaction.
2. Scan backwards for the previous compaction, **skipping** any whose remote/native `preserveData` the active model cannot replay (`remotePreserveReusable`) — re-expand and locally summarize rather than strand history behind an opaque placeholder.
3. `boundaryStart = prevCompactionIndex + 1`, `boundaryEnd = entries.length`.
4. `tokensBefore` from the last assistant usage.
5. **Adaptive keepRecent**: `ratio = promptTokens / estimatedTokens`; if `> 1`, `keepRecentTokens = max(1, floor(keepRecentTokens / ratio))` — corrects local under-estimation against real billing.
6. `findCutPoint(entries, start, end, keepRecentTokens)` (`compaction.ts:623`):
   - Walk **backwards** accumulating `estimateTokens` until `>= keepRecentTokens`, then snap to the nearest valid cut point at or after that index.
   - Valid cut points: message entries with role `user | assistant | bashExecution | hookMessage | branchSummary | compactionSummary`, plus `custom_message` and `branch_summary` entries. **Hard rule: never cut at a `toolResult`** (orphans a tool call).
   - Then walk backwards over adjacent **non-message** entries (`model_change`, `thinking_level_change`, labels …) pulling them into the kept region; stop at any message or compaction boundary.
   - `isSplitTurn` when the cut is not a `user` message and a turn start was found (turn starts: `user`, `bashExecution`, `custom_message`, `branch_summary`).
7. Partition: `messagesToSummarize` (discarded), `turnPrefixMessages` (split-turn), `recentMessages` (kept).
8. `undefined` when nothing to summarize (no-op guard).
9. Carry `previousSummary` + `previousPreserveData` and cumulative `fileOps`.

### 5.5 Summary generation (`compact(...)`, `compaction.ts:1396`)

1. `convertToLlm()` the messages.
2. `serializeConversation()` — dialect-aware. Tool results truncated to `TOOL_RESULT_MAX_CHARS = 2000` with `[... N more characters truncated]`; `useless`-flagged tool results **and their paired calls** dropped entirely; `thinking` blocks dropped for the `anthropic` dialect (its classifier refuses input reproducing its own reasoning).
3. Wrap in `<conversation>…</conversation>`; optionally `<previous-summary>…</previous-summary>`.
4. Optional `<additional-context>` lines from the `session.compacting` hook and the memory backend.
5. Execute with `SUMMARIZATION_SYSTEM_PROMPT` via `instrumentedCompleteSimple`.

Prompt selection (`packages/agent/src/compaction/prompts/`): first compaction → `compaction-summary.md`; iterative → `compaction-update-summary.md`; split-turn second pass → `compaction-turn-prefix.md`; short UI summary → `compaction-short-summary.md`; handoff → `handoff-document.md`.

`compaction-summary.md` mandates a fixed structure — `## Goal`, `## Constraints & Preferences`, `## Progress` (`### Done` / `### In Progress` / `### Blocked`), `## Key Decisions`, `## Next Steps`, `## Critical Context`, `## Additional Notes` — plus explicit rules to preserve any unanswered question verbatim and keep exact file paths, function names, error messages, tool outputs, and repo state.

Split-turn merge:
```
<history summary>

---

**Turn Context (split turn):**

<turn prefix summary>
```

**File-operation appendix**: `extractFileOpsFromMessage` scans assistant tool calls (`read` → read set, `write`/`edit` → modified set), strips read selectors (`:50-200`, `:raw`, `:conflicts`) via `stripReadSelector`, drops `scheme://` paths, folds prior compaction details, and `upsertFileOperations` appends a `<files>` tag: a prefix-folded directory tree with `(Read)`/`(Write)`/`(RW)` markers, capped at 20 files with `[…N files elided…]`. Legacy `<read-files>`/`<modified-files>` tags are stripped first so old summaries self-heal.

### 5.6 Strategies (`compaction.strategy`)

| strategy | mechanism |
|---|---|
| `context-full` | The LLM summarization above. |
| `snapcompact` (**default**) | Local, deterministic. No LLM/API key/network. Discarded history serialized, whitespace-collapsed, rendered onto **PNG bitmap frames of pixel-font text** that vision models read back. |
| `shake` | Mechanical elision: eligible tool results and large fenced/XML blocks → `artifact://` refs. `DEFAULT_SHAKE_CONFIG: protectTokens 16_000, minSavings 4_000, fenceMinTokens 400`. Falls through to context-full when it cannot reclaim enough (idle excepted). |
| `handoff` | Writes **no** `CompactionEntry`. `generateHandoff(...)` produces a document; `AgentSession.handoff()` starts a **new session** and injects it as a visible `custom_message` (`customType: "handoff"`). |
| `off` | Disabled. |

Provider-native paths tried before local summarization: **V2 streaming compaction** (`compaction-v2-streaming.ts`) appends a `compaction_trigger` to a normal Responses stream; the returned compaction item plus retained real user messages (bounded by `compaction.v2RetainedMessageBudget`, default 64000) become replacement history under `preserveData.openaiRemoteCompaction`. Then the native `/responses/compact` endpoint (`compaction/openai.ts`). Then a configured `compaction.remoteEndpoint` — either `{ systemPrompt, prompt }` to an omp summarizer or OpenAI-compatible `{ model, messages, stream:false }` when the path ends in `/chat/completions` (llama.cpp / vLLM as compactor).

**snapcompact specifics** (`packages/snapcompact/src/snapcompact.ts`, ~2k lines):
- `compact(preparation, options)` (L1886) → `CompactionResult` with `preserveData["snapcompact"] = Archive`.
- `Archive { frames, text?, textHead?, textTail? }` — **bounded source text plus rendered frames**; later compactions re-render from `Archive.text`, not by carrying old PNGs forward.
- `historyBlocks(archive, { maxFrameDataBytes })` (L1681) rebuilds ordered blocks each context rebuild: plain text at the oldest edge → imaged middle → plain text at the newest edge; a large middle **foveates** (HQ/LQ/HQ, `HQ_EDGE_FRAMES = 3`).
- Shape table is **model-aware** (`resolveShape` L384): Anthropic `11on16-bw` (8x13 glyphs on an 11px advance; 1932px for Opus 4.7+/Fable/Mythos under the 4,784 visual-token cap, else 1568px); Google `8on22-bw` @2048 (Gemini bills a flat ~1,120 tokens/image at any size); OpenAI `8on22-bw` @1568 with `detail: "original"`.
- Budgets: `TOOL_RESULT_MAX_CHARS 2000`, `TOOL_ARG_MAX_CHARS 500`, `TOOL_CALL_MAX_CHARS 2000`, `TRUNCATE_HEAD_RATIO 0.6` (errors land in the tail); tool output printed in dim gray ink (`DIM_ON`/`DIM_OFF` = `\u000e`/`\u000f`). `MAX_FRAMES_DEFAULT 80`, `FRAME_TOKEN_ESTIMATE 5024`, `FRAME_DATA_BYTES_ESTIMATE 170_000`, `FRAME_DATA_BYTES_BUDGET 3_000_000`, `providerImageBudget` (anthropic/bedrock 90, openai 200, unknown 5).
- Requires a vision model (`model.input` includes `"image"`); else falls back to context-full with a warning. Manual `/compact` honors the strategy unless custom instructions are given.

### 5.7 Persist and reload

Once a summary exists (generated or hook-provided), `AgentSession`:
1. `appendCompaction(...)` → new `CompactionEntry` (handoff instead creates a new session).
2. `buildDisplaySessionContext()` from the active leaf.
3. Replace live agent messages with the rebuilt context.
4. Sync todo phases from the rebuilt branch; close provider sessions whose history was rewritten (`closeCodexProviderSessionsForHistoryRewrite`, `resetCodexProviderAfterCompaction`).
5. Emit the `session_compact` hook event.

Auto-continue: on post-turn threshold success with `compaction.autoContinue !== false`, schedule an agent-authored developer prompt from `prompts/system/auto-continue.md` — **only when the recovery band was cleared**. Mid-turn never schedules one (the core loop already owns the next request).

Hooks (`extensibility/hooks/types.ts`): `session_before_compact` (cancel, or supply a full `CompactionResult`), `session.compacting` (override prompt / add `<additional-context>` / set `preserveData`), `session_compact` (post notification); `session_before_tree` / `session_tree` for branch summaries.

### 5.8 Pre-compaction pruning

`pruneToolOutputs(entries, config)` (`pruning.ts:305`) with `DEFAULT_PRUNE_CONFIG` (L54):
- `protectTokens: 40_000` (newest tool-output tokens untouched); `minimumSavings: 20_000` (else no-op).
- `MIN_PRUNE_TOKENS = 50` — never blank a result smaller than the `[Output truncated - N tokens]` placeholder (~8 tokens); doing so *grows* context and churns the cache for nothing.
- `protectedTools: ["skill", isSkillReadToolResult]` (+ the active plan reference file added by `AgentSession`; + `isArtifactRecoveryToolResult` for shake).

`pruneSupersededToolResults(entries, config)` (L249) — per-turn stale/useless pass:
- **Superseded reads**: a later `read` of the same file+selector key (`readToolSupersedeKey`) makes the earlier result dead weight.
- **Useless results**: tools self-flag via `AgentToolResult.useless` / `ToolResultBuilder.useless()` (zero-match search, timed-out `hub` wait, empty inbox); never set together with `isError` (errors win). Blanked to `USELESS_NOTICE` = `[Uneventful result elided]`. Gated by `compaction.dropUseless` (default on).
- Flagged pairs are **blanked in place, never removed**, so tool-call/result pairing and provider-native replay stay intact. The flag never reaches provider wire formats.

---

## 6. Cache-breaking: compaction vs prompt-cache invalidation

The most transferable part of the design. omp treats the provider prompt cache as a **first-class cost** and gates every history mutation on where it lands relative to the warm prefix.

### 6.1 Stable-prefix discipline
- **Append-only history.** Compaction never rewrites earlier messages; it appends an entry and changes only *where replay starts*. One intentional whole-prefix reset, not a drip of small edits each costing a re-write.
- **Deterministic skill ordering** (`compareSkillOrder`), explicitly "for prompt stability": the `<skills>` block must be byte-identical turn to turn or the system prompt itself busts the cache.
- **`MIN_PRUNE_TOKENS = 50`**: a prune saving less than the placeholder costs is pure cache churn — encoded as a hard floor.

### 6.2 Warm-suffix guard — per-turn prunes stay in the cheap tail
`session-maintenance.ts:145`:
```ts
/** Per-turn prune cache window. A tool result whose all-message suffix exceeds
 *  this is in the warm, already-sent prompt-cache prefix: re-writing it costs the
 *  cacheWrite premium on the whole suffix. */
const PRUNE_CACHE_WARM_SUFFIX_TOKENS = 8_000;
```
Mechanics (`pruning.ts`): `computeMessageSuffixTokens(entries)` (L143) precomputes, per index, the estimated tokens of **all messages strictly after it** — exactly the cached content the provider must re-write if that entry is mutated. `PruneConfig.cacheWarmSuffixTokens` (L51): when `suffix[i] > cacheWarmSuffixTokens`, the entry is in the warm prefix → **skipped**, even if superseded or useless (which otherwise bypass `protectTokens`). Those victims are deliberately left "for compaction/shake, which rebuild the cache anyway". `SupersedePruneConfig` applies the same idea with `DEFAULT_SUFFIX_TOKEN_LIMIT = 8_000` (the read→edit→read tail case).

### 6.3 Idle flush — mutate freely once the cache is provably cold
`session-maintenance.ts:153`:
```ts
/** Idle gap after which the supersede pass may flush the whole sent region (the
 *  provider cache is cold, so re-writing it is free). MUST exceed the maximum
 *  Anthropic prompt-cache TTL — "long" retention (the OAuth default) is 1h — or a
 *  still-warm prefix is busted by the flush. 90 min leaves margin over the 1h TTL. */
const PRUNE_IDLE_FLUSH_MS = 90 * 60_000;
```
(`pruning.ts`'s own default is the more conservative `DEFAULT_IDLE_FLUSH_MS = 30 * 60_000`; the coding-agent overrides to 90 min because it knows the Anthropic long-retention TTL.) When `now - lastMessageTimestamp >= idleFlushMs`, `pruneSupersededToolResults` ignores the suffix guard entirely and prunes **every** still-sent candidate at/after the compaction boundary.

### 6.4 Never churn what is not sent
Both prune passes take `keepBoundaryId` (= latest `CompactionEntry.firstKeptEntryId`) and skip everything before it: summarized-away entries are never transmitted, so mutating them is pure I/O with zero token benefit.

### 6.5 Cache-preserving handoff
`generateHandoff(...)` (`compaction.ts:1056`) deliberately does **not** build a fresh one-shot prompt. It sends the **live system prompt, tool array, and real message history** — the exact warm cache prefix — then appends one agent-attributed `user` message carrying the handoff prompt, with `toolChoice: "none"`. The call *reads* the cache instead of writing a new one.

### 6.6 Timing choices that are really cache choices
- **Mid-turn compaction only at safe tool-loop boundaries** — a mid-loop rewrite would invalidate the cache in the middle of a burst of cheap cached turns.
- **Post-turn threshold maintenance under `handoff`** schedules a *post-prompt* task instead of compacting inline, so it does not race (and re-write) the next turn; pre-prompt and mid-turn checks run inline because they already sit at a boundary.
- **Context promotion before compaction** (`#tryContextPromotion`) avoids the rewrite entirely — a cold cache on the new model, but history intact.
- **`COMPACTION_RECOVERY_BAND = 0.8`** prevents compaction thrash, i.e. repeated compactions each resetting the cache.
- Threshold maintenance runs **after a completed turn**, so the cache write happens once, where the next request was going to be a fresh prefix anyway.

### 6.7 Local token-estimate memoization (different cache, same file family)
`packages/agent/src/compaction/message-cache.ts` memoizes per-message token estimates and LLM conversions keyed on **object identity**, with two invariants worth porting:
1. **Settle gate** — a streaming assistant is mutated under one identity while `usage`/`stopReason` are provisional; only settled assistants (`usage.totalTokens > 0`, terminal `stopReason` that is not `aborted`/`error`) are cached.
2. **Owner invalidation** — every in-place rewriter (`pruneToolOutputs`, `pruneSupersededToolResults`, `applyShakeRegion`, `stripImagesFromMessage`) must call `invalidateMessageCache(message)`; cross-package consumers subscribe via `registerMessageCacheInvalidator`.
Two split caches (`estimateCacheDefault` / `estimateCacheFloored`) because `excludeEncryptedReasoning` yields a different number for the same message. Deliberately `WeakMap`, not symbol-tagged properties, so object spreads (`{ ...message, content: truncated }`) do not inherit a stale estimate.

---

## 7. Key types / functions with file paths

### Skills

| symbol | path |
|---|---|
| `Skill { name, description, filePath, baseDir, source, hide?, _source? }` | `packages/coding-agent/src/extensibility/skills.ts:18` |
| `SkillWarning`, `LoadSkillsResult` | `extensibility/skills.ts:35,39` |
| `getActiveSkills()` / `setActiveSkills()` / `resetActiveSkillsForTests()` | `extensibility/skills.ts:47,54,60` |
| `isNameClaimedByAuthoredSkill(name)` | `extensibility/skills.ts:73` |
| `loadSkillsFromDir({ dir, source })` | `extensibility/skills.ts:86` |
| `loadSkills(options)` | `extensibility/skills.ts:118` |
| `getSkillSlashCommandName(skill)` | `extensibility/skills.ts:399` |
| `parseSkillInvocation(text)` → `{ name, args }` | `extensibility/skills.ts:~440` |
| `buildSkillPromptMessage(skill, args, invocation)` | `extensibility/skills.ts:487` |
| `SkillFrontmatter`, capability `Skill`, `skillCapability` | `packages/coding-agent/src/capability/skill.ts` |
| `scanSkillsFromDir`, `compareSkillOrder`, `parseAgentFields` (`autoloadSkills`) | `packages/coding-agent/src/discovery/helpers.ts:365,357,~300` |
| provider registrations | `discovery/{builtin,agents,claude,claude-plugins,codex,opencode,github,omp-plugins}.ts` |
| `SkillProtocolHandler`, `validateRelativePath` | `packages/coding-agent/src/internal-urls/skill-protocol.ts` |
| `buildSystemPrompt` (filter L836-840, data L870, render L899) | `packages/coding-agent/src/system-prompt.ts` |
| `<skills>` prompt block | `packages/coding-agent/src/prompts/system/system-prompt.md:26-34` |
| `<skill name="">` custom-prompt block | `prompts/system/custom-system-prompt.md:31-41` |
| `user-invocation.md`, `autoload.md` | `packages/coding-agent/src/prompts/skills/` |
| `estimateSkillsTokens`, `renderedSkills` | `packages/coding-agent/src/modes/utils/context-usage.ts:67-83` |
| `manage_skill` tool + managed-skill dir | `tools/manage-skill.ts`, `autolearn/managed-skills.ts`, `prompts/tools/manage-skill.md` |
| `SKILL_PROMPT_MESSAGE_TYPE`, `SkillPromptDetails` | `packages/coding-agent/src/session/messages.ts` |
| slash-command dispatch / TUI component | `modes/skill-command.ts`, `modes/components/skill-message.ts` |

### Compaction

| symbol | path |
|---|---|
| `CompactionSettings`, `DEFAULT_COMPACTION_SETTINGS`, `DEFAULT_RESERVE_TOKENS`, `MAX_SUMMARY_TOKENS` | `packages/agent/src/compaction/compaction.ts` |
| `calculateContextTokens`, `calculatePromptTokens`, `getLastAssistantUsage` | `compaction.ts:~222-290` |
| `effectiveReserveTokens`, `resolveBudgetReserveTokens`, `shouldCompact`, `compactionContextTokens`, `resolveThresholdTokens` | `compaction.ts:304,320,334,355,359` |
| `estimateTokens`, `IMAGE_TOKEN_ESTIMATE` | `compaction.ts:407,~392` |
| `CutPointResult`, `findCutPoint` | `compaction.ts:599,623` |
| `prepareCompaction`, `CompactionPreparation`, `CompactionResult`, `CompactionDetails` | `compaction.ts:1212,~150` |
| `compact(preparation, model, apiKey, customInstructions?, signal?, options?)` | `compaction.ts:1396` |
| `generateHandoff`, `generateHandoffFromContext`, `AUTO_HANDOFF_THRESHOLD_FOCUS` | `compaction.ts:1056,1022` |
| `shouldUseProviderNativeCompaction` | `compaction.ts:~205` |
| file-ops + serialization helpers, `SUMMARIZATION_SYSTEM_PROMPT` | `packages/agent/src/compaction/utils.ts` |
| `DEFAULT_PRUNE_CONFIG`, `pruneToolOutputs`, `pruneSupersededToolResults`, `computeMessageSuffixTokens`, `MIN_PRUNE_TOKENS`, `DEFAULT_SUFFIX_TOKEN_LIMIT`, `DEFAULT_IDLE_FLUSH_MS`, `USELESS_NOTICE` | `packages/agent/src/compaction/pruning.ts` |
| `ShakeConfig`, `DEFAULT_SHAKE_CONFIG`, `AGGRESSIVE_SHAKE_CONFIG`, `collectShakeRegions` | `packages/agent/src/compaction/shake.ts` |
| `isSkillReadToolResult`, `isProtectedToolResult`, `isArtifactRecoveryToolResult` | `packages/agent/src/compaction/tool-protection.ts` |
| `invalidateMessageCache`, `isEstimateCacheable`, `read/writeEstimateCache`, `registerMessageCacheInvalidator` | `packages/agent/src/compaction/message-cache.ts` |
| V2 streaming compaction | `packages/agent/src/compaction/compaction-v2-streaming.ts` |
| remote/native OpenAI compaction | `packages/agent/src/compaction/openai.ts` |
| branch summarization | `packages/agent/src/compaction/branch-summarization.ts` |
| `CompactionEntry`, `BranchSummaryEntry`, `SessionEntry` | `packages/agent/src/compaction/entries.ts` |
| `createCustomMessage`, `createBranchSummaryMessage`, `ConvertToLlm` | `packages/agent/src/compaction/messages.ts` |
| compaction prompts | `packages/agent/src/compaction/prompts/*.md` |
| `SessionMaintenance` (compact L574, runIdleCompaction L974, runPrePromptCompactionIfNeeded L1012, maintainContextMidRun L1068, checkCompaction L1194, runAutoCompaction L2133, shake L460, dropImages L409) | `packages/coding-agent/src/session/session-maintenance.ts` |
| `PRUNE_CACHE_WARM_SUFFIX_TOKENS` (8k), `PRUNE_IDLE_FLUSH_MS` (90 min), `COMPACTION_RECOVERY_BAND` (0.8) | `session-maintenance.ts:145,153,166` |
| `Shape`, `SHAPES`, `SHAPE_VARIANTS`, `resolveShape`, `Archive`, `Frame`, `historyBlocks`, `getPreservedArchive`, `serializeConversation`, `normalize`, `wrap`, `geometry`, `render`, `renderMany`, `frames`, `compact` | `packages/snapcompact/src/snapcompact.ts` |
| settings schema (`compaction.*`, `snapcompact.*`, `branchSummary.*`, `skills.*`) | `packages/coding-agent/src/config/settings-schema.ts` |
| authoritative narrative doc | `docs/compaction.md` |

### Settings defaults

```
compaction.enabled                    true
compaction.strategy                   "snapcompact"   // context-full | handoff | shake | off
compaction.reserveTokens              unset  (floor 16384, and >= 15% of window)
compaction.keepRecentTokens           20000
compaction.autoContinue               true
compaction.midTurnEnabled             true
compaction.thresholdPercent           -1     // <= 0 => reserve-based
compaction.thresholdTokens            -1     // > 0 wins over percent
compaction.handoffSaveToDisk          false
compaction.remoteEnabled              true
compaction.remoteEndpoint             undefined
compaction.remoteStreamingV2Enabled   true
compaction.v2RetainedMessageBudget    64000
compaction.idleEnabled                false
compaction.idleThresholdTokens        200000
compaction.idleTimeoutSeconds         300
compaction.supersedeReads             true
compaction.dropUseless                true
snapcompact.systemPrompt              "none"  // agents-md | all
snapcompact.toolResults               false
snapcompact.shape                     "auto"
branchSummary.enabled                 false
branchSummary.reserveTokens           16384
```

---

## Porting notes to Python

Ordered by value/effort for `~/local-operator`.

### Tier 1 — port nearly verbatim (high value, low risk)

1. **Adopt `<dir>/SKILL.md` with YAML frontmatter `name` + `description` verbatim.** It is the emerging cross-tool standard (Claude, Codex, opencode, GitHub Copilot all read it), so you inherit an ecosystem of existing skills for free. Python: `pathlib` + `python-frontmatter`, or a 10-line `---` splitter plus `yaml.safe_load`. Keep `enabled`, `hide` / `disable-model-invocation`, and the `frontmatter.name or dir.name` fallback.
2. **Two-line prompt injection. Do not over-engineer:**
   ```
   Skills are specialized knowledge. If one matches your task, you MUST read `skill://<name>` before proceeding.
   <skills>
   - <name>: <description>
   </skills>
   ```
   Sort deterministically (`key=lambda s: (s.name.lower(), s.name, str(s.path))`) — a **cache-correctness requirement**, not cosmetics.
3. **`skill://` resolver with hard path containment.** Port `validateRelativePath` plus the resolve-then-`is_relative_to(base_dir)` check exactly: reject absolute paths and any `..` segment *before* joining, then re-check after resolving (defense in depth against symlinks). Return a directory listing for directory targets so `skill://x/references` is enumerable. The miss error **must** list all available names — it is the model's self-correction path.
4. **Progressive disclosure with zero machinery.** Three rungs: description in prompt → `SKILL.md` on demand → `references/*` named by the `SKILL.md` prose. Resist building an index; the point is that rung 3 is just filesystem plus instructions.
5. **Protect loaded skills from truncation.** Whatever your context-reduction pass is, exempt tool results whose path starts with `skill://`. Otherwise the model re-reads the same skill in a loop.
6. **Never cut at a tool result.** The single hardest correctness rule in `findCutPoint`. Orphaned tool calls break every provider. Encode it as an assertion, not a comment.
7. **Append-only compaction entry.** Store `summary`, `first_kept_entry_id`, `tokens_before`, `details`, `preserve_data` as a session entry; rebuild LLM context as `[summary_message] + entries[first_kept:]`. Do **not** mutate history in place. This also buys a free display transcript (show everything; render the compaction as a divider) and resumable sessions.
8. **The two prompt-cache constants.** The highest-leverage idea here:
   - `PRUNE_CACHE_WARM_SUFFIX_TOKENS = 8_000` — precompute suffix-token totals per entry (`itertools.accumulate` over reversed estimates) and only mutate entries whose suffix is under the window.
   - `PRUNE_IDLE_FLUSH_MS = 90 min` (must exceed the ~1h Anthropic long-retention TTL) — once idle past the TTL, mutate freely.
   Each is ~15 lines and directly reduces cacheWrite spend.
9. **`MIN_PRUNE_TOKENS` floor and `COMPACTION_RECOVERY_BAND = 0.8`.** Two constants that prevent, respectively, negative-value prunes and compaction thrash loops. Cheap insurance; omp added the band only after a live dead-loop bug (#2275).
10. **Structured summary prompt.** Copy `compaction-summary.md`'s section list (Goal / Constraints / Progress{Done, In Progress, Blocked} / Key Decisions / Next Steps / Critical Context) and the "preserve any unanswered question verbatim" rule. Plus the `<files>` appendix — a folded path tree with `(Read)`/`(Write)`/`(RW)` markers is the highest-signal-per-token part of a coding-agent summary.

### Tier 2 — port with adaptation

11. **Multi-provider discovery** is only worth it if local-operator users already have `~/.claude/skills` etc. Otherwise start with two roots (`<project>/.<app>/skills` walk-up, `~/.<app>/agent/skills`) but keep the provider abstraction (priority + dedupe-by-name + collision warning) so adding roots later is additive. Keep `os.path.realpath` dedupe for symlinked roots.
12. **Adaptive `keep_recent_tokens`** (`keep / (prompt_tokens / local_estimate)` when the ratio exceeds 1). `tiktoken`'s `cl100k_base` drifts from Anthropic billing by the same ~5-10% omp acknowledges, so this correction earns its keep. Also port `context_tokens = max(provider_reported, local_estimate)` — otherwise any payload-compression layer lets real history grow unbounded until it overflows.
13. **Recovery ladder**: overflow → try a larger-context model → compact → retry. Cheaper and lossless versus compacting first. Needs a model registry that knows sibling context windows.
14. **Split-turn handling.** Needed once you allow cutting at assistant messages. The merged `<history summary> --- **Turn Context (split turn):** <prefix summary>` shape is simple and works.
15. **Useless-result flagging.** Let tools mark their own results contextually useless (zero-match search, timed-out wait). Python: a `useless: bool` field on the tool-result dataclass, honored by the prune pass and excluded from summarization input. Keep the invariant that `useless` and `is_error` are mutually exclusive, and blank in place rather than deleting (pairing must survive).
16. **Estimate memoization** keyed on message identity with a settle gate. Python: `weakref.WeakKeyDictionary` (message objects must be weakref-able — plain classes, or `__slots__` classes that also declare `__weakref__`). Alternatively key on a monotonic message id and invalidate explicitly. Every in-place mutator must invalidate.

### Tier 3 — probably skip (for now)

17. **snapcompact (bitmap-image archival).** Fascinating, but it depends on a Rust native renderer (`crates/pi-natives/src/snapcompact.rs`), bundled bitmap fonts, an eval-derived per-model shape table, and vision-capable models. A Python port via `Pillow` (`ImageDraw.text` with a bitmap `ImageFont`, PNG → base64) is feasible, but the shape table is empirical — do not copy the numbers without re-running evals for your model mix. Ship `context-full` summarization first; treat snapcompact as an experiment.
18. **Provider-native compaction** (OpenAI `compaction_trigger` / `/responses/compact`). Only worth it on the Responses API, for zero-cost summarization. Note the trap omp documents: a prior native compaction's `preserve_data` must be **skipped and re-expanded** when the active model cannot replay it, or that history is stranded behind an opaque placeholder.
19. **Branch summaries / `/tree` navigation.** Only if local-operator has a session tree.

### Python-specific gotchas

- **`WeakMap` → `weakref.WeakKeyDictionary`**; omp's rationale (avoiding stale values riding along object spreads) maps to `dataclasses.replace` / `copy.copy` — identity-keyed caches remain the right call.
- **Frontmatter strip**: omp uses `/^---\n[\s\S]*?\n---\n/`. In Python use `re.DOTALL` with a non-greedy body anchored at string start.
- **Token counting**: `tiktoken.get_encoding("cl100k_base")` matches omp's estimator. Charge a flat ~1200 tokens per image and exclude provider orchestration tokens from context sizing.
- **Async discovery**: omp scans all roots with `Promise.all` and time-boxes skill loading (`withDeadline("loadSkills", ..., fallback [])`). Mirror with `asyncio.gather` + `asyncio.wait_for` — a slow NFS home directory must degrade to "no skills", not hang startup.
- **Glob filters** (`includeSkills` / `ignoredSkills`) match on the **skill name**, not the path; `fnmatch.fnmatch` is the direct equivalent of `Bun.Glob.match` here.
- **Frontmatter key normalization**: accept both `disable-model-invocation` (spec, kebab) and `disableModelInvocation`; omp ORs them into one `hide` flag.

