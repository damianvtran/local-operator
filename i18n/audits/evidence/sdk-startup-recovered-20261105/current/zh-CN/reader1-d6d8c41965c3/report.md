# zh-CN — `wire.errors.sdk_startup` — context-delta confirmation (priority reader 1)

- **Locale / namespace:** zh-CN / `wire.errors.sdk_startup` (4 keys)
- **Mode:** context-delta confirmation only. Target bytes are unchanged from the two prior independent audits, so no re-authoring and no literary re-audit of unaffected strings.
- **Date:** 2026-11-03
- **Auditor:** `i18n-auditor`, session `d6d8c41965c3`, requested route `radient/auto`.
  Served underlying model: **UNKNOWN** (the route label is a header value, not served-model proof).
  Fallback: **UNKNOWN** — no fallback event exposed; `NONE` not claimed.
  Independent of translator session `7fce46e29bee` and of prior readers `e55f54b6a26e` / `6433b7a5b3e9`.
- **Dependency:** EN source + context live on **unmerged** PR #3378 head `0066245ce4f90109933c59e74f1ad21b86d647bd`. No shipped / activated / release-eligible claim.
- **Verdict: PASSED** — 0 BLOCKER, 0 MAJOR, 2 NIT (pre-existing, non-gating). `changes_requested` not warranted.

## Inputs (all re-hashed here, from the frozen files)

| input | sha256 |
|---|---|
| `en.json` | `5390ecb2413e0978c34541a9a0db6c2a004ae86446d1c0fa0d2e697e4f9ca6bd` |
| `context.json` (new) | `82f193fd3c3e84526cbfaa0498fc47b776dd84df128cea3442df6d13369bb13e` |
| `zh-CN.json` (target, 817 B) | `fcff2fac99196a9e4dbd03957fb959c2af69013b1471532f819ddbc705293e02` |
| `i18n/style-guides/zh-CN.md` | `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c` |
| `i18n/glossary.md` | `0eb34437a8c3d6f564f278ef957a603b1d9f51bb07d5ddae5ea8f37deba962f8` |
| old `context.json` (superseded) | `ee0bbde5e99de22c04368e6bbfedf913d758ef6ba5075984ddf7d795b1ee7822` |

Binding value `82f193fd…` is correct; the `82b…` string passed around in chat is a typo and was not used.
The binding was treated as authoritative and every file above matched it byte-for-byte.

**Equalities against the prior audits:** EN unchanged, target **byte-identical** (`fcff2fac…` = the prior audits' recorded input hash), style guide and glossary unchanged. Only the context changed.

## Context delta

One field changed: the `note` for `attach_hosting_unconfigured`. All roles/tones and the other three notes are byte-identical.

- **Old (stale):** `error` is opaque producer text; the namespace is English-only and falls back per-carried.
- **New:** the cause is *not always* opaque — when it is the product's own not-configured hosting error (exact type + published sentence) the **caller substitutes the separately localized cause sentence**, so the frame must never carry English cause wording of its own; any other producer's reason stays opaque and verbatim; the frame renders in the resolved locale, never an English pin; per-namespace English fallback still applies to locales that do not carry the namespace.

**Checked against the code at the dependency head** (`local_operator/sdk.py`, read-only, in the PR worktree):

- `startup_refusal_locale()` resolves the configured `language` through `resolve_language` (587–596) — no English pin.
- `startup_cause_text()` renders `wire.errors.hosting.not_configured` in the resolved locale **only** when `type(error) is HostingNotConfiguredError` *and* `str(error)` equals the catalogued English source; otherwise it returns `str(error)` verbatim (603–632).
- The frame is wired with `error=startup_cause_text(error)` and rendered through `startup_refusal_text` (660–666); `attach_runtime_missing` uses the same path (690–694).

So the removed English-only assertion was indeed false, and the new note describes the code rather than asserting a plan. **The old raw-English-cause finding is handled: the cause is now a separately localized sentence produced at the caller, not by this catalogue.**

## Prior findings disposition

| id | prior severity | disposition |
|---|---|---|
| `Z2-1` (reader `6433b7a5b3e9`) | MINOR — raw English `{error}` inside a zh-CN frame | **Closed caller-side** by PR #3378 (owned message + caller substitution), exactly the remedy both prior auditors named. No target text change was or is owed; target bytes unchanged. Not re-opened. |
| `F1`, `F2` (reader `e55f54b6a26e`) | MINOR — same English-opaque-cause class, recorded as residual gap `waived: false` | Same closure as above. Prior wording approvals stand. |

(Note: the second record attributes the gap to `attach_runtime_missing`, while the English-cause slot is `attach_hosting_unconfigured`'s `{error}`. Same single gap, one corrected path for both frames. This is a provenance nit in the prior record, not a quality issue in the target.)

## Mechanical (computed in-session, Python stdlib only)

No repo validator, scanner, generator, Node or process was run — this task is strict source-only; that is recorded as NOT RUN, not waived.

- Valid JSON; key sets equal across EN / target / new context / old context (4 keys, no additions or drops).
- Placeholders identical in name, count and order per key — `resume`+`error`; `resume`; `profile`; `spec_tools`+`policy_tools`.
- No `plural`/`select` structure in source **or** target; the four messages carry plain arguments, so no CLDR plural categories are required for zh-CN here (its single `other` category is moot with no plural).
- No Rich/HTML markup or backticks to preserve.
- Literals byte-identical in target: `config.yml`, `spawn_session()`, `deliver()`, `mode=''own''`, `spec.tools`, `ApprovalPolicy.declared(`, `ID`. Doubled apostrophes are ICU escaping and are retained in the catalogue bytes.
- Typography: full-width `，。：（）；` for prose; ASCII `.` / `'` appear only inside `config.yml`, `spec.tools`, `mode=''own''`; no CJK↔Latin space insertion; no NBSP/ideographic space/zero-width; no `您`, no `！`; full sentences keep `。`, label-form strings carry none.
- Sidecar width hints: **none** (`maxCells`/`maxChars` absent from all four entries — verified programmatically). No budget exists to exceed and no budget is invented. Cell counts (zh: 170 / 143 / 23 / 110) are informational only, not a fit verdict.

## Native pass (fresh reader, compatibility with the new context)

All four strings remain acceptable zh-CN product copy; the delta owes no wording change.

- `attach_hosting_unconfigured` — both facts and remedy order preserved; the colon introduces a **complete** Chinese remedy sentence; the frame has no English cause wording of its own, so both the substituted localized cause and a passed-through foreign cause read correctly after `{error}`.
- `attach_runtime_missing` — disposal-then-refusal, `mode='own'` remedy and engage-then-attach order all preserved; nothing added about disposal or exception attributes.
- `profile_attach_failed` — could-not + quoted profile preserved; `配置档` matches shipped zh-CN usage elsewhere rather than `sdk_spec`'s `配置档案`.
- `tool_inventory_mismatch` — two named inventories and the state-once remedy preserved; no inventory widening implied; the space before `{spec_tools}` is Latin-to-Latin, not a CJK–Latin gap.
- Terminology/register: no glossary row governs `运行时`/`托管`/`附接`/`清单`; terse diagnostic voice, no `请`, no pronoun repetition. `附接` is an SDK-family coinage, consistent across the four keys and approved by both prior readers — noted, not raised (changing it now would be an out-of-scope re-author).

## Findings (none gating)

`ZD-1` **NIT** — `attach_hosting_unconfigured`: `运行` appears three times (`没有正在运行的运行时` / `无法启动一个` / `运行时存在`), slightly heavy. Optional: `没有活跃运行时为{resume}提供服务…`.

`ZD-2` **NIT** — `attach_hosting_unconfigured`: `启动` renders both *start a runtime* and *engage the id*. Optional: `…然后用spawn_session()/deliver()接入该ID…`, keeping engage distinct from start and from `附接`.

Both sit on unchanged, previously approved bytes; neither gates, and neither justifies a translation wave under this task.

## Not run (explicit gaps)

Rendered fit/design · pseudo-localization · RTL (n/a for zh-CN) · live execution of the SDK refusal under a real zh-CN config · repo validator and generator. No whole-frame or runtime English-free clearance is claimed: the foreign-cause passthrough is verbatim **by contract**. Source/context depend on an unmerged PR head, so nothing here is a shipped claim.

## Record

`record.json` alongside this report (same directory), with hashes, identities, delta evidence, dispositions and the verification matrix. Intended canonical root once a tracked bundle is cut: `i18n/audits/evidence/zh-CN/wire.errors.sdk_startup/d6d8c419-zhctxdelta1/`. No repo edits, no ledger row, no PR comment were made.
