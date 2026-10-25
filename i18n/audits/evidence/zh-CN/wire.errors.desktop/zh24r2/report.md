# Independent zh-CN re-audit — wire.errors.desktop — zh24r2

- Round: 2 (post-remediation re-audit at current bytes)
- Audited revision: `d9f70f622d9f0f1600c667cfe918571a408e3cb6` (branch `i18n/zh-cn-core-rebased`)
- Fix commit under review: `cb716031e335aa27e632227f1a0303e143dcd19f` ("fix(i18n): apply the zh-CN core audit findings")
- Supersedes: `i18n/audits/evidence/zh-CN/wire.errors.desktop/zh24a1/record.json` (independent audit, pre-fix target `a8cb0db6bbc8…`); `i18n/audits/evidence/zh-CN/wire.errors.desktop/zhca1/record.json` (carried-namespace attestation, same pre-fix target). Those records carry pre-fix target bytes and are not current-byte evidence.
- Auditor: translation-auditor (lopdev team), session `b1d9d628481a`, model `deepseek/deepseek-flash`, 2026-10-24. Independent of the implementer: this session did not author or edit any catalogue value, and every claim below was re-derived from the current bytes.
- Scope: 7 changed key(s) — `wire.errors.desktop.aside_adopt_unconfirmed`, `wire.errors.desktop.aside_limit_reached`, `wire.errors.desktop.aside_max_turns`, `wire.errors.desktop.aside_not_adoptable`, `wire.errors.desktop.aside_request_reused`, `wire.errors.desktop.aside_still_running`, `wire.errors.desktop.aside_unavailable` — plus a regression read of all 84 keys in the namespace.

## Applied-fix verification (executed)

`fix-verification.json` compares `cb716031e3^` with the current target, parses the exact proposal recorded in the pre-fix `zh24a1` report, and re-derives ICU arguments from the current English source. Every changed value was found, matches its proposal byte-for-byte, and keeps the English argument set:

| key | proposal found | applied = proposal | args = en | render | rendered/max cells |
|---|---|---|---|---|---|
| `wire.errors.desktop.aside_adopt_unconfirmed` | yes | yes | yes | ok | no budget |
| `wire.errors.desktop.aside_limit_reached` | yes | yes | yes | ok | no budget |
| `wire.errors.desktop.aside_max_turns` | yes | yes | yes | ok | no budget |
| `wire.errors.desktop.aside_not_adoptable` | yes | yes | yes | ok | no budget |
| `wire.errors.desktop.aside_request_reused` | yes | yes | yes | ok | no budget |
| `wire.errors.desktop.aside_still_running` | yes | yes | yes | ok | no budget |
| `wire.errors.desktop.aside_unavailable` | yes | yes | yes | ok | no budget |

No key outside this set changed between `cb716031e3^` and `d9f70f622d` in this namespace.

## Mechanical results (executed)

- Key parity: en 84 / zh 84; missing 0; extra 0; key order identical.
- Non-NFC 0; control characters 0; bidi controls 0; zero-width 0; fullwidth Latin/digits 0; CJK↔Latin spaces 0.
- ICU argument mismatches 0; plural-structure errors 0. Only the locale's `other` arm is present where English has `one`/`other`, as the zh-CN guide requires; `#` stays bound to `{count}`.
- Technical tokens dropped 0. The zh24r2 probe narrows two analyzer false positives rather than any catalogue value: `env_or_constant` now requires an underscore-delimited constant, and `abs_path` no longer captures a trailing sentence period; a numeric sample was added for `{max, number}`. Catalogue bytes are unchanged by these probe refinements.
- Width: 0 changed key(s) carry a `maxCells` budget, 0 over budget.
- Render: all changed keys render with representative values; 0 failures. The namespace-level render count and plural samples are in `runtime-mechanical.json`.
- `scripts/i18n/check.py --catalogues-only`: `check.py: catalogue checks: 0 problem(s)` (exit 0).

## Findings

No new finding in the fix set or the regression read.

## Regression read

All 84 values were read side by side with the English source and, for the changed keys, their context notes. All seven aside errors now use 题外问答, the term already used by the feature title and the unsaved-asides summary; the other terms in the namespace (profile → 配置档, tag → 标签, session → 会话) hold. The remaining 77 values were re-read, including the claim, library and session-file refusal families. `{max, number}` renders with a numeric sample and keeps its ICU kind.

## Verdict

- Namespace: PASSED — clean.
- Blocker 0 · major 0 · minor 0 · nit 0; unresolved blocker/major count 0.
- `current_report_verdict`: `passed`.
- This record changes no catalogue byte, ledger row, allowance, shipped set or hold state.

## Caveats

- String-level audit: no native terminal, relay SPA, Electron or rendered-frame pass was performed, so truncation in a real paint, font-cell width and interaction behaviour are not claimed.
- zh-CN remains in `SECURITY_HELD_LOCALES` (`scripts/i18n/generate.py`) and out of `data/shipped.json`; this record neither activates it nor waives the hold.
- Translator provenance: the tree records no zh-CN provenance file (only `i18n/provenance/ur-core.translator.json`); agent/model/date are left null rather than invented. The fix commit is attributed to a shared account, so independence is claimed for this audit session and its unedited inputs, not from the commit author field.

## Evidence files

- `record.json`
- `report.md`
- `audit.log`
- `audit.py.txt`
- `fix-verify.py.txt`
- `fix-verification.json`
- `mechanical-raw.json`
- `runtime-mechanical.json`
- `pseudo-evidence.json`
- `wire.errors.desktop.pairs.txt`
- `wire.errors.desktop.compact.txt`
- `receipt.json`
