# Audit report — `es` / `wire.errors.sdk_startup`

**Reader 1 · first full independent native audit · 4 retained keys · no re-authoring**

| Field | Value |
|---|---|
| Locale | `es` (neutral Spanish, wave 1) |
| Namespace (file group) | `wire.errors.sdk_startup` |
| Mode | `first_full_independent_native_audit` (retained translations audited as-is; target not edited) |
| Audit date | 2026-11-03 |
| Audit id | `es-ssu-reader1-20261103` |
| Auditor | agent `i18n-auditor`, session `dd6c62eeb8a7` |
| Requested route | `radient/auto` (selector only) |
| Served underlying model | **UNKNOWN** — no usage/provider payload reachable under the strict source-only scope; the `Model: radient/auto` header is a selector, not execution proof |
| Fallback | **UNKNOWN** — no failure/incident event exposed this turn; not reported as `NONE` because that is not proven |
| Independence | Translator agent `i18n-translator`, session `804764f54d82`; auditor session `dd6c62eeb8a7`. No shared authoring; `independent_from_translator_session: true` |

## Inputs (hashed from the frozen files named in the binding)

| Input | Path | SHA-256 | Bytes |
|---|---|---|---|
| English source | `…/sdk-startup-followon/en.json` | `5390ecb2413e0978c34541a9a0db6c2a004ae86446d1c0fa0d2e697e4f9ca6bd` | 772 |
| English context sidecar | `…/sdk-startup-followon/context.json` | `82f193fd3c3e84526cbfaa0498fc47b776dd84df128cea3442df6d13369bb13e` | 2 298 |
| Spanish target | `…/sdk-startup-followon/es.json` | `2f60ae79a4350194cf1a22af6bcbbcc01452b0316835e8d7607f67a46790cfcd` | 830 |

Cross-checks performed (byte equality, not prose):

- Frozen English and frozen context are **byte-identical** to the worktree originals (`…/.worktrees/startup-owned-copy/local_operator/i18n/catalogues/en/wire.errors.sdk_startup{,.context}.json`) — same hashes.
- Frozen `es.json` is **byte-identical** to the translator's original at `…/sessions/804764f54d82/scratchpad/es-sdk-startup-4-0e7cde3/wire.errors.sdk_startup.json` (`2f60ae79…`). So the retained translation is what is audited.
- `local_operator/i18n/catalogues/es/wire.errors.sdk_startup.json` **does not exist** in the PR-3378 worktree — this group is staged, not shipped. No shipped claim is made anywhere below.

## Dependency state

Source and context belong to **unmerged core PR #3378**, head `0066245ce4f90109933c59e74f1ad21b86d647bd`. Read-only code inspection was limited to `local_operator/sdk.py` (attach/build path, lines 545–745) to confirm the display contract; no source or caller change is made or implied by this audit.

## Context delta (recorded, not a translation defect)

| | Value |
|---|---|
| Old context (staged ES packet, built before this round) | `ee0bbde5e99de22c04368e6bbfedf913d758ef6ba5075984ddf7d795b1ee7822` (1 890 B) — `candidates/es/context.json` |
| Current context (binding) | `82f193fd3c3e84526cbfaa0498fc47b776dd84df128cea3442df6d13369bb13e` (2 298 B) |
| Diff scope | **only** `attach_hosting_unconfigured.note`. `role`/`tone` and the notes of the other three keys are unchanged. |

What the delta does: it removes the stale *"English-only child namespace falls back under RFC §6 …"* framing (the false English-only claim) and adds the localized-cause rule — the `{error}` slot carries either the caller's **separately localized** `wire.errors.hosting.not_configured` sentence or a foreign producer's opaque reason, so **the frame itself must contain no English cause wording**.

ES consequence: the retained frame names the remedy in Spanish and carries no cause wording of its own, so the delta does not invalidate any of the four retained values. The stale-context fact is provenance (the ES packet's manifest records `ee0bbde5…`), not a finding against the translation.

## Prior audits and independence of this pass

- **No prior ES audit of this group.** `i18n/audits/evidence/es/wire.errors.sdk*` → 0 matches. This is the first full native pass for `es`.
- No verdict borrowed from `wire.errors.hosting` or `wire.errors.sdk_spec` audits. The only sibling read was the **English** `wire.errors.sdk_spec.json`, for house style (lowercase-start diagnostic frames), and the ES catalogue for terminology consistency.
- No parallel reader's report was read.
- FR/ZH prior full audits were not used: their judgment is unaffected by the context delta, and compatibility here is established by my own byte and contract checks, not by their verdicts.

## Method actually executed (and what was not)

Executed in-process, stdlib only (`json`, `hashlib`, `difflib`, `unicodedata`), inside this session's Python kernel — the command outputs in this session are the evidence:

1. SHA-256 of all three frozen inputs + the four comparison files above.
2. JSON parse; key-set identity (4/4/4, identical key order); no key added or dropped.
3. Placeholder extraction and comparison per key: names, counts and **order** identical — `{resume}`, `{error}`, `{profile}`, `{spec_tools}`, `{policy_tools}`.
4. ICU structure: brace balance and max depth 1 in every value; **no `plural`/`select` and no comma inside any brace**, so there is no plural message and no CLDR category set to check (not applicable, not skipped).
5. ICU apostrophe parity: EN and ES `attach_runtime_missing` each carry exactly 4 apostrophes = two doubled pairs → both render `mode='own'` (2 in ES `tool_inventory_mismatch` vs 0 is correct — ES writes «de la ejecución», no possessive apostrophe needed).
6. Byte-level token identity per key: `config.yml`, `spawn_session()`, `deliver()`, `spec.tools`, `ApprovalPolicy.declared(`, `mode=''own''` — all 1:1.
7. Whitespace/invisible scan: no leading/trailing space, no double space, no NBSP/NNBSP/ZWSP/LRM/RLM/TAB, no newline, no backtick, no `#`. Non-ASCII is only `á é í ó ú`.
8. Width measurement (`unicodedata.east_asian_width`, combining marks skipped): 225 / 181 / 39 / 144 cells. No wide glyphs, no combining marks, so cell width == character count.
9. Context diff (old vs current) with `difflib`.
10. Code contract read: `sdk.py` 545–745 — the four frames are raised as `SessionSpecError(<rendered frame>)`, i.e. the frame *is* the user-visible text; `resume`/`profile` are `repr()`-quoted, `spec_tools`/`policy_tools` are `str(tuple)`; `startup_cause_text()` substitutes the localized hosting sentence only for the exact owned type + exact catalogued English source, otherwise passes the foreign `str(error)` through verbatim.

**NOT run — explicit gaps, not passed checks:**

- Repo validator `scripts/i18n/check.py` / `generate.py --check`: **NOT RUN**. The strict source-only scope overrides the role's mechanical-validator default (no process/shell/Node/network). Reason recorded; cannot be read as a mechanical pass issued by the repo tooling.
- Pseudo-locale export/render, runtime/rendered fit, screenshot or terminal frame capture: **NOT RUN**.
- Overflow judgement: **NOT RUN** — the context sidecar carries **no `maxCells`/`maxChars` for any of the four keys** (confirmed: each entry has only `role`, `tone`, `note`), and inventing a budget would be fabrication. No whole-frame or runtime clearance is implied by this source-only pass.
- RTL: not applicable to `es`.
- Ledger: **not written** — the strict scope forbids repo edits; the requester records the row citing this record.
- No translation edited, no PR comment posted, no PR/repo state claimed.
- **Scope disclosure:** after the checks had completed, one `bash -c true` no-op was launched by mistake under the strict scope. It returned no output, fed no value into any finding or verdict, and touched no file — recorded here rather than omitted.

## Mechanical result

PASS on every check in items 2–8 above. No BLOCKER-class mechanical defect.

## Per-key native review (accuracy · idiom · register · glossary · safety)

| Key | Verdict | Note |
|---|---|---|
| `attach_hosting_unconfigured` | PASS (1 MINOR-adjacent NIT, 1 NIT) | Meaning complete: refusal, both facts (no live runtime; root cannot start one), cause slot, then a full Spanish remedy. Register correct (`tú`: «Configura … activa … conéctate»). ICU-quoted identifier preserved. |
| `attach_runtime_missing` | PASS | «ábrelo con `mode='own'` para ejecutarlo en este proceso» vs «conéctate cuando exista un runtime» keeps the own-vs-attach distinction exactly; «o actívalo antes» renders "engage it first" without loss. |
| `profile_attach_failed` | PASS | Terse refusal, initial capital preserved, no trailing period (same as source), `{profile}` untouched. |
| `tool_inventory_mismatch` | PASS (1 MINOR) | Both API tokens and the tuple placeholders in source order; the run's reach is rendered «el alcance de la ejecución» and "once" as «una sola vez». |

**No untranslated English human wording.** Every Latin run in the ES values was enumerated and named: `runtime` (kept-English, es guide wave-1), `config.yml` (path), `id` (see NIT-2), `mode='own'` / `spawn_session()` / `deliver()` / `spec.tools` / `ApprovalPolicy.declared` (API, named in the context note), plus the placeholders. **`hosting` is translated**: the frame says «el alojamiento», and the catalogued ES cause sentence is «La plataforma de alojamiento no está configurada.» (`wire.errors.hosting.not_configured`, read in the PR-3378 worktree — catalogue state, not a shipped claim) — same head noun, so the operator's "no English anywhere" requirement holds for this group's human words.

**Glossary / style conformance checked and clean:** `runtime` kept (§ style guide wave-1 list); session/id as machine tokens kept; tú address; sentence case with source-mirrored lowercase starts (the English `sdk_spec`/`sdk_startup` family uses lowercase-start diagnostics — e.g. `output enforcement cannot be set…` — so lowercase here is **source-aligned**, not a typographic slip, and I do not request a divergence); Spanish punctuation (no space before `:` `;`, no `¿…?` needed, no `!`); no Title Case; no region-marked lexis.

**Safety review:** every value is a refusal with its remedy intact; none of the four omits a permission/destructive caveat or inverts polarity. The only "do something" instructions are the source's own (`config.yml`, `spawn_session()/deliver()`, `mode='own'`), preserved verbatim.

## Findings

### ES-SSU-M1 — MINOR — `wire.errors.sdk_startup.tool_inventory_mismatch`

- **Current:** «spec.tools {spec_tools} y ApprovalPolicy.declared({policy_tools}) **nombran** inventarios distintos; declara el alcance de la ejecución una sola vez»
- **Problem:** «nombrar» on two spec expressions is a calque of English *name*. It is understandable and has corpus neighbours («--tools no nombra ninguna herramienta» in `wire.exec.tools_unreachable`, «nombrar la sesión» in `wire.settings`), so this is not a wrong-meaning defect — but in a diagnostic the native verb for a declaration that *specifies* a set is «especificar/designar/definir».
- **Suggested wording:** «… **especifican** inventarios distintos; declara el alcance de la ejecución una sola vez» (or «designan»).
- **Gating:** no. Minor only; meaning, tokens and punctuation are unaffected.

### ES-SSU-N1 — NIT — `wire.errors.sdk_startup.attach_hosting_unconfigured`

- **Current:** «Configura el alojamiento para esta raíz (**su** config.yml)»
- **Problem:** the sentence addresses the user as `tú`, so «su» momentarily reads as "your"; the intended referent is the root. Both readings happen to be true here, so nothing is misleading.
- **Suggested wording:** «(el config.yml de esa raíz)» — longer; keeping «su» is acceptable.

### ES-SSU-N2 — NIT — `wire.errors.sdk_startup.attach_hosting_unconfigured`

- **Current:** «luego activa el **id** con spawn_session()/deliver()»
- **Problem:** the ES catalogue writes the noun as `ID` elsewhere («ID del equipo del hub al que vincularlo», «IDs de trabajo y de sesión»); lowercase `id` reads as the parameter token. Both are defensible; one spelling would be cleaner.
- **Suggested wording:** «activa el **ID** con spawn_session()/deliver()» (code tokens stay byte-identical).

No further MINOR or NIT is raised; findings are capped deliberately rather than padded.

## Verdict

**`passed`** — 0 BLOCKER, 0 MAJOR, 1 MINOR (`ES-SSU-M1`), 2 NIT (`ES-SSU-N1`, `ES-SSU-N2`). Under the `passed` rule (no unresolved blocker/major) this group passes; the minor and nits are non-gating and may be folded into a later remediation wave or deferred.

**Explicitly not claimed:** mechanical validation by the repo tooling, rendered/terminal fit, pseudo-locale behaviour, or any runtime clearance. This is a source-only native audit of 4 retained keys against the current frozen English source and context at PR #3378 head.

## Record and companions

- `record.json` — structured audit record (same directory).
- `receipt.json` — identity + hashes of this report and the record.
- `record.json` carries `audit_notes_not_execution_log: true`: the checks above ran as in-process stdlib computations in the auditor session; **no validator, scanner, generator, render or app execution exists to log**; the one `bash -c true` no-op disclosed above produced no output and no effect, and these notes must not be read as such a log.
