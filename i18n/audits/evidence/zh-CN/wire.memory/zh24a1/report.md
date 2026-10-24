# Independent zh-CN audit — wire.memory — zh24a1

- Locale: `zh-CN` (held; not in `shipped.json`)
- Namespace: `wire.memory`
- Revision audited: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (catalogue bytes)
- Assembly revision: `5b5367be1b4acef44d564dcf8cbc1c7c06b209e8` — catalogue diff empty, intervening commits add evidence only
- Auditor: translation-auditor (lopdev team); session `3d1df2689651`; actual model `deepseek/deepseek-flash`; fallback: none
- Independent of the translator: yes (translator session is not the auditor session)
- Keys in namespace: **17**; keys read: **17**
- Delta keys audited: **3** (2 new keys + 1 re-translation)
- Verdict: **clean_with_followups** (status `passed`); unresolved blocker/major: **0**; minor: 2; nit: 0

## Input hashes (sha256)

- en: `319e23e6e4a6abee4e9549655142bebb5402563902c6f0497f3ee761fb53aec7` (`local_operator/i18n/catalogues/en/wire.memory.json`)
- context: `852518b5d2bd41f4b8b72d6d89d961d35b1bafaf341726e0406973d12b1c5e7d` (`local_operator/i18n/catalogues/en/wire.memory.context.json`)
- target: `4e39d2bacb0ea88a16f7de720a5ae2208a580ea789ecc135a1c4368d2f3d1e28` (`local_operator/i18n/catalogues/zh-CN/wire.memory.json`)
- policy: `i18n/glossary.md` `ef07b446668778822d0ff4dcd7b380543456ddf7b08e060dce5818cb8fd6fdcd`; `i18n/style-guides/zh-CN.md` `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c`

## Delta keys in scope

- `wire.memory.capture.help`
- `wire.memory.capture.locators.help`
- `wire.memory.capture.locators.label`

## Findings

### ZH-A1 — minor — delta (re-translated key)

“明确存储” renders “explicit stores” with 明确 (“clear/definite”), which reads as an adverb inside the sentence; the locale's established rendering of “explicit” is 显式 (cli.agents_help, cli.exec_help, cli.hub_help, wire.settings all use 显式). “guaranteed route” is also weakened to “可靠途径”; 保障途径 preserves the guarantee the English makes against best-effort capture.

- `wire.memory.capture.help`
  - current: ``每次压缩提交后采集决策、目标和进展；context-full还会复用其已有摘要。明确存储仍是保留目标和决策的可靠途径。``
  - proposed: ``每次压缩提交后采集决策、目标和进展；context-full还会复用其已有摘要。显式存储仍是保留目标和决策的保障途径。``

Note: No placeholder, plural or width change; both substitutions are two characters, so the rendered width is unchanged.

### ZH-A3 — minor — delta (new key)

“locator” here means a path, URL or id (the help text lists exactly those), but “定位信息” is the ordinary Chinese phrase for location/positioning data and can read as a privacy-sensitive geolocation capture. 定位符 is the computing term for a locator (cf. 统一资源定位符 for URL) and matches the help text.

- `wire.memory.capture.locators.label`
  - current: ``采集定位信息上下文``
  - proposed: ``采集定位符上下文``

Note: Current label is 18 cells; the proposed one is 16 cells. No ICU arguments.

## Mechanical results

- Key parity: en 17 / zh 17; missing 0; extra 0.
- ICU argument parity (names and kinds): 17 keys checked, 0 failures.
- Message parse/render: 17 messages rendered with synthetic parameters, 0 failures.
- Plural structure: 0 plural keys; failures 0. zh-CN's other-only arms are expected by the style guide.
- Hard technical tokens (backticks, flags, angle tokens, URLs, paths, %s/%d): 0 failures.
- Width: 0 keys carry `maxCells`; 0 over budget; all delta keys measured against sidecar and note budgets.
- Punctuation: 0 failures; Latin/CJK spacing: 0 failures; 您 occurrences: 0.
- Numbers: 0 unexplained differences.
- Encoding: non-NFC 0; bidi controls 0. RTL checks: not applicable (LTR locale).

## Regression read

All 17 keys in the namespace were read side by side with the English source; the delta keys were judged against their English context notes as well. Findings marked *pre-existing* are in the current bytes but predate `c4ba95919d`; they are recorded because this audit vouches for the namespace, not only the delta.

## Limitations

No native TUI screenshot, relay or desktop render is claimed. Width is measured in cells (CJK/full-width = 2) against the sidecar `maxCells` where present and against note-stated budgets for the delta labels; the +25% growth figure in `pseudo-evidence.json` is a synthetic probe, not a rendered frame. zh-CN remains held and is not made shippable by this audit.
