# Round 2 — deduplicated receipt ledger

Pinned head: `4880031aeed59ef17ae6ffccc15d28e1fc0db104`. Integrator: `4d5d48953a65`, actual `openai/gpt-6.1-sol` (host quota fallback).

**HOLD at K:** 14 original receipts opened, parsed, hashed and byte-preserved; eight unique major `(locale,key)` entries remain. Three VI glossary minors belong in the same failed-VI repair wave. Five clean locale pairs are retained, not re-audited. No native/rendered/shipping pass is asserted.

## Exact receipt coverage (per auditor; not additive across auditors)

| Locale | A | B | Fresh R2 keys each | Retained compatible keys each | Namespace total each |
|---|---|---|---:|---:|---:|
| zh-CN | PASS_WITH_NON_GATING_MINORS | PASS_WITH_NON_GATING_MINORS | 55 | 840 | 895 |
| fr | PASS_WITH_NON_GATING_MINORS | PASS_WITH_NON_GATING_MINORS | 40 | 855 | 895 |
| es | FAIL | FAIL | 47 | 848 | 895 |
| ru | PASS_WITH_NON_GATING_MINORS | PASS_WITH_NON_GATING_MINORS | 54 | 841 | 895 |
| vi | PASS_WITH_NON_GATING_MINORS | FAIL | 65 | 830 | 895 |
| hi | PASS_WITH_NON_GATING_MINORS | PASS_WITH_NON_GATING_MINORS | 80 | 815 | 895 |
| ur | PASS_WITH_NON_GATING_MINORS | PASS_WITH_NON_GATING_MINORS | 89 | 806 | 895 |

Every raw receipt retains its own schema and verdict. Fresh keys match the exact readerbrief scope; retained keys are the disjoint complement. Source/target/context per-key hashes and text were checked against pinned Git K; retained values were additionally compared to the old Git head. Five clean catalogue working files were byte-identical to K at integration. Raw receipt model provenance attributes all 14 audits to GPT fallback, and their distinct sessions exclude the seven authors; partner-blindness is original auditor attestation, not retrospective observation.

## Frozen accepted major ledger

### R2-M01 — `es` / `wire.settings.providers_openrouter_allow_fallbacks.help`

- State: human help references raw English options rather than actual visible localized Choice.label.
- Intended transition: Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label.
- Authority: fresh author `87094744ea9d`; integrator authors no translation.
- Original finding citations: `i18n/audits/evidence/approval-repair-r2-originals/es-A.json#/findings/0` (ES-A-R2-M1), `i18n/audits/evidence/approval-repair-r2-originals/es-B.json#/findings/0` (ES-B-R2-M1)
- Original proposals are quoted verbatim in the JSON ledger; they are not authoritative replacements. Actual pinned choice-label evidence is included per entry. Stable API/config tokens remain code, not human-label substitutes.

### R2-M02 — `es` / `wire.settings.providers_openrouter_data_collection.help`

- State: human help references raw English options rather than actual visible localized Choice.label.
- Intended transition: Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label.
- Authority: fresh author `87094744ea9d`; integrator authors no translation.
- Original finding citations: `i18n/audits/evidence/approval-repair-r2-originals/es-A.json#/findings/1` (ES-A-R2-M2), `i18n/audits/evidence/approval-repair-r2-originals/es-B.json#/findings/1` (ES-B-R2-M2)
- Original proposals are quoted verbatim in the JSON ledger; they are not authoritative replacements. Actual pinned choice-label evidence is included per entry. Stable API/config tokens remain code, not human-label substitutes.

### R2-M03 — `es` / `wire.settings.providers_openrouter_enforce_distillable_text.help`

- State: human help references raw English options rather than actual visible localized Choice.label.
- Intended transition: Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label.
- Authority: fresh author `87094744ea9d`; integrator authors no translation.
- Original finding citations: `i18n/audits/evidence/approval-repair-r2-originals/es-A.json#/findings/2` (ES-A-R2-M3), `i18n/audits/evidence/approval-repair-r2-originals/es-B.json#/findings/2` (ES-B-R2-M3)
- Original proposals are quoted verbatim in the JSON ledger; they are not authoritative replacements. Actual pinned choice-label evidence is included per entry. Stable API/config tokens remain code, not human-label substitutes.

### R2-M04 — `es` / `wire.settings.providers_openrouter_require_parameters.help`

- State: human help references raw English options rather than actual visible localized Choice.label.
- Intended transition: Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label.
- Authority: fresh author `87094744ea9d`; integrator authors no translation.
- Original finding citations: `i18n/audits/evidence/approval-repair-r2-originals/es-A.json#/findings/3` (ES-A-R2-M4), `i18n/audits/evidence/approval-repair-r2-originals/es-B.json#/findings/3` (ES-B-R2-M4)
- Original proposals are quoted verbatim in the JSON ledger; they are not authoritative replacements. Actual pinned choice-label evidence is included per entry. Stable API/config tokens remain code, not human-label substitutes.

### R2-M05 — `es` / `wire.settings.providers_openrouter_sort.help`

- State: human help references raw English options rather than actual visible localized Choice.label.
- Intended transition: Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label.
- Authority: fresh author `87094744ea9d`; integrator authors no translation.
- Original finding citations: `i18n/audits/evidence/approval-repair-r2-originals/es-A.json#/findings/4` (ES-A-R2-M5), `i18n/audits/evidence/approval-repair-r2-originals/es-B.json#/findings/4` (ES-B-R2-M5)
- Original proposals are quoted verbatim in the JSON ledger; they are not authoritative replacements. Actual pinned choice-label evidence is included per entry. Stable API/config tokens remain code, not human-label substitutes.

### R2-M06 — `es` / `wire.settings.providers_openrouter_zdr.help`

- State: human help references raw English options rather than actual visible localized Choice.label.
- Intended transition: Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label.
- Authority: fresh author `87094744ea9d`; integrator authors no translation.
- Original finding citations: `i18n/audits/evidence/approval-repair-r2-originals/es-A.json#/findings/5` (ES-A-R2-M6), `i18n/audits/evidence/approval-repair-r2-originals/es-B.json#/findings/5` (ES-B-R2-M6)
- Original proposals are quoted verbatim in the JSON ledger; they are not authoritative replacements. Actual pinned choice-label evidence is included per entry. Stable API/config tokens remain code, not human-label substitutes.

### R2-M07 — `es` / `wire.settings.shell_environment_mode.help`

- State: human help references raw English options rather than actual visible localized Choice.label.
- Intended transition: Reference actual current localized human Choice.label; preserve any explicitly named stable API/config token as code, never substitute it for the human label.
- Authority: fresh author `87094744ea9d`; integrator authors no translation.
- Original finding citations: `i18n/audits/evidence/approval-repair-r2-originals/es-A.json#/findings/6` (ES-A-R2-M7), `i18n/audits/evidence/approval-repair-r2-originals/es-B.json#/findings/6` (ES-B-R2-M7)
- Original proposals are quoted verbatim in the JSON ledger; they are not authoritative replacements. Actual pinned choice-label evidence is included per entry. Stable API/config tokens remain code, not human-label substitutes.

### R2-M08 — `vi` / `wire.settings.display_nerd_icons.help`

- State: changed visible human help retains ordinary English terminal without named exception.
- Intended transition: Fresh Vietnamese LLM translator replaces ordinary English terminal with native term; technical IDs unchanged.
- Authority: fresh author `f0f530be0d7f`; integrator authors no translation.
- Original finding citations: `i18n/audits/evidence/approval-repair-r2-originals/vi-B.json#/findings/0` (VI-B-R2-M1)
- Original proposals are quoted verbatim in the JSON ledger; they are not authoritative replacements. Actual pinned choice-label evidence is included per entry. Stable API/config tokens remain code, not human-label substitutes.

## Same-wave Vietnamese non-gating minors

- `wire.settings.providers_openrouter_enforce_distillable_text.help` — `VI-B-R2-m1`; source `i18n/audits/evidence/approval-repair-r2-originals/vi-B.json#/findings/1`.
- `wire.settings.providers_openrouter_zdr.help` — `VI-B-R2-m2`; source `i18n/audits/evidence/approval-repair-r2-originals/vi-B.json#/findings/2`.
- `wire.settings.web_fetch_render_backend.help` — `VI-B-R2-m3`; source `i18n/audits/evidence/approval-repair-r2-originals/vi-B.json#/findings/3`.

## Preservation and retention

- `approval-repair-r2-originals/`: 14 byte-identical R2 originals; source paths and SHA-256 values are in the JSON ledger and artifact manifest.
- `approval-repair-r2-referenced-support/`: only six explicitly referenced external JSON artifacts, including the approved English capture and named audit/control evidence. Existing in-tree R1 originals and sourcebinding/readerbrief/retention artifacts are indexed without modification.
- `approval-repair-r2-clean-stream-retention.json`: all 895 key bindings for each of zh-CN/fr/ru/hi/ur, identifying fresh R2 vs compatible R1 provenance and exact A/B original references. Root must prove actual K2 compatibility before retention; this is not a blind re-pin.
- Raw R1 and R2 FAIL verdicts are preserved. VI A PASS and VI B FAIL remain distinct; the locale does not become clean by majority or summary.
- All other clean-stream minor/width findings are deferred; no clean locale or minor-only cycle is reopened.

## Exit / limitations

Round 3 authors only ES seven helpers + VI four helpers (one major and three same-wave glossary minors), followed by independent A/B delta audits plus direct reference neighbors. No fresh full-895 audit. No product edits, new translations, tests/native/CI/network/dependency work, commits or pushes were performed by the integrator. All artifacts remain untracked.
