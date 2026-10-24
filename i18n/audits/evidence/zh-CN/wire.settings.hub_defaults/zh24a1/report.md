# Independent zh-CN audit — wire.settings.hub_defaults — zh24a1

- Locale: `zh-CN` (held; not in `shipped.json`)
- Namespace: `wire.settings.hub_defaults`
- Revision audited: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (catalogue bytes)
- Assembly revision: `5b5367be1b4acef44d564dcf8cbc1c7c06b209e8` — catalogue diff empty, intervening commits add evidence only
- Auditor: translation-auditor (lopdev team); session `3d1df2689651`; actual model `deepseek/deepseek-flash`; fallback: none
- Independent of the translator: yes (translator session is not the auditor session)
- Keys in namespace: **9**; keys read: **9**
- Delta keys audited: **4** (4 re-translations)
- Verdict: **clean** (status `passed`); unresolved blocker/major: **0**; minor: 0; nit: 0

## Input hashes (sha256)

- en: `8720b5719479b638eab24b081f76f97c44ce8d8554cb1b8280f6980182b6740c` (`local_operator/i18n/catalogues/en/wire.settings.hub_defaults.json`)
- context: `a337e7394f3856476044f9accdb8daeb7a864a9a8ecfe0c579ec97ec646712ba` (`local_operator/i18n/catalogues/en/wire.settings.hub_defaults.context.json`)
- target: `b547b9730d78a81e840746de723e8a44597a3dcb175cd7e0200a143bb4972c13` (`local_operator/i18n/catalogues/zh-CN/wire.settings.hub_defaults.json`)
- policy: `i18n/glossary.md` `ef07b446668778822d0ff4dcd7b380543456ddf7b08e060dce5818cb8fd6fdcd`; `i18n/style-guides/zh-CN.md` `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c`

## Delta keys in scope

- `wire.settings.hub_defaults.agents.help`
- `wire.settings.hub_defaults.agents.label`
- `wire.settings.hub_defaults.agents.off.description`
- `wire.settings.hub_defaults.agents.on.description`

## Findings

None. The delta keys and the regression read are clean on the checks and terms below.

## Mechanical results

- Key parity: en 9 / zh 9; missing 0; extra 0.
- ICU argument parity (names and kinds): 9 keys checked, 0 failures.
- Message parse/render: 9 messages rendered with synthetic parameters, 0 failures.
- Plural structure: 0 plural keys; failures 0. zh-CN's other-only arms are expected by the style guide.
- Hard technical tokens (backticks, flags, angle tokens, URLs, paths, %s/%d): 0 failures.
- Width: 2 keys carry `maxCells`; 0 over budget; all delta keys measured against sidecar and note budgets.
- Punctuation: 0 failures; Latin/CJK spacing: 0 failures; 您 occurrences: 0.
- Numbers: 0 unexplained differences.
- Encoding: non-NFC 0; bidi controls 0. RTL checks: not applicable (LTR locale).

## Regression read

All 9 keys in the namespace were read side by side with the English source; the delta keys were judged against their English context notes as well. Findings marked *pre-existing* are in the current bytes but predate `c4ba95919d`; they are recorded because this audit vouches for the namespace, not only the delta.

## Limitations

No native TUI screenshot, relay or desktop render is claimed. Width is measured in cells (CJK/full-width = 2) against the sidecar `maxCells` where present and against note-stated budgets for the delta labels; the +25% growth figure in `pseudo-evidence.json` is a synthetic probe, not a rendered frame. zh-CN remains held and is not made shippable by this audit.
