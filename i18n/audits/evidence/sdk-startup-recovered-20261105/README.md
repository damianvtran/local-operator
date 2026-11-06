# SDK startup translation recovery — evidence, not a new audit

This directory retains completed work for the four human-facing
`wire.errors.sdk_startup` refusals. It does not change or translate the raw
`wire.errors.sdk_spec` machine boundary.

`recovery-manifest.json` is the original recovery inventory (SHA-256
`084fbaf1197d2bb124baa95c516bfdd0424c5b185271e30c6f7b7bc0b5219b6e`).
All 106 payloads were rehashed. The 78 unique payloads are retained unchanged;
`retention-map.json` maps every original occurrence and reference to its retained
bytes, and explicitly links the three translator records, target hashes, four
prior full audits and six current approvals/reports. `binding-resolution.json`
preserves how original scratch references were recovered. Duplicate bytes are
not duplicated merely to recreate obsolete scratch paths.

## Scope and provenance

- French and Chinese: two prior full audits each, followed by two current-context
  confirmations each. A context-only confirmation is **not** a new full audit.
- Spanish: two full current-context audits.
- Original dates, input hashes, verdicts, model disclosures and historical notes
  about the then-unmerged #3378 are immutable. That prerequisite is now merged
  and an ancestor of the recovery branch; the original notes are not rewritten.
- Recorded translator router labels are `radient/auto`. They do not identify a
  concrete underlying model. Admission rows state that distinction explicitly.
- Current auditors' concrete served models and fallbacks are UNKNOWN. The
  canonical rows use `auditor.model: null` with explicit source disclosures,
  rather than inventing GPT, another model, or a no-fallback claim. Original
  header/router labels remain in the untouched records. Chinese translator date
  is also null where the original packet does not establish one.
- `advisory-dispositions.json` lists all 19 original advisory/observation entries
  with source-scoped IDs and reasons. Their disposition is non-gating deferral
  without changing approved wording or inventing follow-up issues.

Six admission rows were derived from the six actual passed current approvals,
then appended using the official `ledger.py append` expected-history CAS writer.
English, context and target hashes were checked against each approval. The
immutable baseline/marker files were not rewritten. These derived admission
rows and this map are coordinator metadata, **not** replacement auditor records.

## Execution limits

The recovered audit notes remain source/linguistic evidence, not a claimed new
runtime or rendered test. Recovery validation exercises the actual SDK startup
producer with isolated provider/factory resources, real FR/ES/zh-CN config,
owned-cause localization and opaque foreign-cause preservation. Existing raw,
pre-root, fallback, mutation-oracle and lazy-import checks remain separate.
No new translation, SDK spec migration, SDK production-code change, native
rendering, CLDR-wide, bidi, hardware or provider/network clearance is claimed.
