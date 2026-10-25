# Independent zh-CN re-audit — relay.ask — zh24r2

- Round: 2 (post-remediation re-audit at current bytes)
- Audited revision: `d9f70f622d9f0f1600c667cfe918571a408e3cb6` (branch `i18n/zh-cn-core-rebased`)
- Fix commit under review: `cb716031e335aa27e632227f1a0303e143dcd19f` ("fix(i18n): apply the zh-CN core audit findings")
- Supersedes: `i18n/audits/evidence/zh-CN/relay.ask/zh24a1/record.json` (independent audit, pre-fix target `f57ce2524036…`). Those records carry pre-fix target bytes and are not current-byte evidence.
- Auditor: translation-auditor (lopdev team), session `b1d9d628481a`, model `deepseek/deepseek-flash`, 2026-10-24. Independent of the implementer: this session did not author or edit any catalogue value, and every claim below was re-derived from the current bytes.
- Scope: 2 changed key(s) — `relay.ask.context.messages`, `relay.ask.dismiss.moot_note` — plus a regression read of all 11 keys in the namespace.

## Applied-fix verification (executed)

`fix-verification.json` compares `cb716031e3^` with the current target, parses the exact proposal recorded in the pre-fix `zh24a1` report, and re-derives ICU arguments from the current English source. Every changed value was found, matches its proposal byte-for-byte, and keeps the English argument set:

| key | proposal found | applied = proposal | args = en | render | rendered/max cells |
|---|---|---|---|---|---|
| `relay.ask.context.messages` | yes | yes | yes | ok | 38/60 |
| `relay.ask.dismiss.moot_note` | yes | yes | yes | ok | 18/44 |

No key outside this set changed between `cb716031e3^` and `d9f70f622d` in this namespace.

## Mechanical results (executed)

- Key parity: en 11 / zh 11; missing 0; extra 0; key order identical.
- Non-NFC 0; control characters 0; bidi controls 0; zero-width 0; fullwidth Latin/digits 0; CJK↔Latin spaces 0.
- ICU argument mismatches 0; plural-structure errors 0. Only the locale's `other` arm is present where English has `one`/`other`, as the zh-CN guide requires; `#` stays bound to `{count}`.
- Technical tokens dropped 0. The zh24r2 probe narrows two analyzer false positives rather than any catalogue value: `env_or_constant` now requires an underscore-delimited constant, and `abs_path` no longer captures a trailing sentence period; a numeric sample was added for `{max, number}`. Catalogue bytes are unchanged by these probe refinements.
- Width: 2 changed key(s) carry a `maxCells` budget, 0 over budget.
- Render: all changed keys render with representative values; 0 failures. The namespace-level render count and plural samples are in `runtime-mechanical.json`.
- `scripts/i18n/check.py --catalogues-only`: `check.py: catalogue checks: 0 problem(s)` (exit 0).

## Findings

No new finding in the fix set or the regression read.

## Regression read

All 11 values were read side by side with the English source and, for the changed keys, their context notes. The two fixes are exact. `context.messages` keeps the single `other` arm required by zh-CN, keeps `#` bound to `{count}`, keeps `{time}` as a plain argument that is never reformatted, and renders 38-40 cells against the 60-cell budget for counts 1-100. `moot_note` now names the agent as the actor, so it no longer reads as an instruction to the operator. The remaining 9 values were re-read: dismissal labels, receipts and the `dismiss…` control are consistent with the TUI sibling namespace and the locale's two-mark ellipsis.

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
- `relay.ask.pairs.txt`
- `relay.ask.compact.txt`
- `receipt.json`
