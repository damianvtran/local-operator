# Independent zh-CN audit — tui.notices — zh24a1

- Locale: `zh-CN` (held; not in `shipped.json`)
- Namespace: `tui.notices`
- Revision audited: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (catalogue bytes)
- Assembly revision: `5b5367be1b4acef44d564dcf8cbc1c7c06b209e8` — catalogue diff empty, intervening commits add evidence only
- Auditor: translation-auditor (lopdev team); session `3d1df2689651`; actual model `deepseek/deepseek-flash`; fallback: none
- Independent of the translator: yes (translator session is not the auditor session)
- Keys in namespace: **30**; keys read: **30**
- Delta keys audited: **1** (1 new key)
- Verdict: **clean** (status `passed`); unresolved blocker/major: **0**; minor: 0; nit: 0

## Input hashes (sha256)

- en: `4a159689628113c32a2df75a9e3d98d74aab790c9946848307bb267318731510` (`local_operator/i18n/catalogues/en/tui.notices.json`)
- context: `3f1298de1bdb84a67bde66a5bb937f05376c81e06d866e94180f21b1525c8370` (`local_operator/i18n/catalogues/en/tui.notices.context.json`)
- target: `40af781ff575e4571fbb4caa7f2ae55443f1bdb2c344c2e04137d59198c77c61` (`local_operator/i18n/catalogues/zh-CN/tui.notices.json`)
- policy: `i18n/glossary.md` `ef07b446668778822d0ff4dcd7b380543456ddf7b08e060dce5818cb8fd6fdcd`; `i18n/style-guides/zh-CN.md` `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c`

## Delta keys in scope

- `tui.notices.fast_mode_refused`

## Findings

None. The delta keys and the regression read are clean on the checks and terms below.

## Mechanical results

- Key parity: en 30 / zh 30; missing 0; extra 0.
- ICU argument parity (names and kinds): 30 keys checked, 0 failures.
- Message parse/render: 30 messages rendered with synthetic parameters, 0 failures.
- Plural structure: 0 plural keys; failures 0. zh-CN's other-only arms are expected by the style guide.
- Hard technical tokens (backticks, flags, angle tokens, URLs, paths, %s/%d): 0 failures.
- Width: 0 keys carry `maxCells`; 0 over budget; all delta keys measured against sidecar and note budgets.
- Punctuation: 0 failures; Latin/CJK spacing: 0 failures; 您 occurrences: 0.
- Numbers: 0 unexplained differences.
- Encoding: non-NFC 0; bidi controls 0. RTL checks: not applicable (LTR locale).

## Regression read

All 30 keys in the namespace were read side by side with the English source; the delta keys were judged against their English context notes as well. Findings marked *pre-existing* are in the current bytes but predate `c4ba95919d`; they are recorded because this audit vouches for the namespace, not only the delta.

## Limitations

No native TUI screenshot, relay or desktop render is claimed. Width is measured in cells (CJK/full-width = 2) against the sidecar `maxCells` where present and against note-stated budgets for the delta labels; the +25% growth figure in `pseudo-evidence.json` is a synthetic probe, not a rendered frame. zh-CN remains held and is not made shippable by this audit.
