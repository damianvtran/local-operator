# Independent zh-CN audit — tui.authoring — zh24a1

- Locale: `zh-CN` (held; not in `shipped.json`)
- Namespace: `tui.authoring`
- Revision audited: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (catalogue bytes)
- Assembly revision: `5b5367be1b4acef44d564dcf8cbc1c7c06b209e8` — catalogue diff empty, intervening commits add evidence only
- Auditor: translation-auditor (lopdev team); session `3d1df2689651`; actual model `deepseek/deepseek-flash`; fallback: none
- Independent of the translator: yes (translator session is not the auditor session)
- Keys in namespace: **49**; keys read: **49**
- Delta keys audited: **9** (9 new keys)
- Verdict: **clean_with_followups** (status `passed`); unresolved blocker/major: **0**; minor: 2; nit: 0

## Input hashes (sha256)

- en: `7431acf3288cc3766cf37135badb29511737120ab52fc518bf514311ee488d13` (`local_operator/i18n/catalogues/en/tui.authoring.json`)
- context: `df98bba4457c1080b3941e9a24a04a52e360d11d2a07bdda992861c6cb54928c` (`local_operator/i18n/catalogues/en/tui.authoring.context.json`)
- target: `b26f3ab6cddbf93beff849976a2f453feaa4ee4a828474a2f30e37e9410c7470` (`local_operator/i18n/catalogues/zh-CN/tui.authoring.json`)
- policy: `i18n/glossary.md` `ef07b446668778822d0ff4dcd7b380543456ddf7b08e060dce5818cb8fd6fdcd`; `i18n/style-guides/zh-CN.md` `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c`

## Delta keys in scope

- `tui.authoring.promote_failed`
- `tui.authoring.promote_mixed_plan`
- `tui.authoring.promote_plan`
- `tui.authoring.promote_receipt_created`
- `tui.authoring.promote_receipt_keeps`
- `tui.authoring.promote_receipt_replaced`
- `tui.authoring.promote_replace_plan`
- `tui.authoring.promote_usage`
- `tui.authoring.scope_conversation`

## Findings

### ZH-A2 — minor — delta (new keys)

The three promote plan notices keep English word order: the parenthetical “（以本会话的副本）” is a calque of “with this conversation's copy” placed after the list, “同时复制{created}” is mood-ambiguous in a plan notice (it can read as a command), and “添加`yes`以…” is a literal rendering of “Add `yes` to…”. Chinese states the same plan as 用B替换A and 加上…即可; the rewrite keeps the overwrite-first warning order, both placeholders and the literal `yes`.

- `tui.authoring.promote_plan`
  - current: ``添加`yes`以将{names}复制到你的账户 — 尚未做任何更改。``
  - proposed: ``在命令末尾加上`yes`，即可将{names}复制到你的账户 — 尚未做任何更改。``
- `tui.authoring.promote_mixed_plan`
  - current: ``将替换账户中的现有定义：{replaced}（以本会话的副本）。同时复制{created}。添加`yes`以替换并复制 — 尚未做任何更改。``
  - proposed: ``将用本会话的副本替换账户中的现有定义：{replaced}。这还会复制{created}。在命令末尾加上`yes`，即可替换并复制 — 尚未做任何更改。``
- `tui.authoring.promote_replace_plan`
  - current: ``替换账户中的现有定义：{names}（以本会话的副本）。添加`yes`以替换 — 尚未做任何更改。``
  - proposed: ``将用本会话的副本替换账户中的现有定义：{names}。在命令末尾加上`yes`，即可替换 — 尚未做任何更改。``

Note: Placeholder names/order and the literal `yes` backtick span survive in every proposed line.

### ZH-A6 — minor — pre-existing regression read (not introduced by c4ba95919d)

“Profile” has two Chinese renderings: 配置档 in wire.errors.desktop (profile_name_taken, packaged_profile_not_found, profile_registry_unavailable, requested_not_found) and wire.exec, versus plain 配置 in the authoring page and one settings help line. 配置 already means “configuration”, so using it for an agent profile hides which object is meant.

- `tui.authoring.notice_read_only`
  - current: ``只读：本次会话的运行时位于另一台设备，此处无法修改配置。``
  - proposed: ``只读：本次会话的运行时位于另一台设备，此处无法修改配置档。``
- `tui.authoring.revision_changed`
  - current: ``其他人已修改此配置，重新打开后再编辑。``
  - proposed: ``其他人已修改此配置档，重新打开后再编辑。``
- `wire.settings.subagents_model_choice.help`
  - current: ``由谁选择子智能体的档位；操作员可在其配置中固定。``
  - proposed: ``由谁选择子智能体的档位；操作员可在其配置档中固定。``

Note: The wire.errors.desktop side already uses 配置档 and needs no change.

## Mechanical results

- Key parity: en 49 / zh 49; missing 0; extra 0.
- ICU argument parity (names and kinds): 49 keys checked, 0 failures.
- Message parse/render: 49 messages rendered with synthetic parameters, 0 failures.
- Plural structure: 0 plural keys; failures 0. zh-CN's other-only arms are expected by the style guide.
- Hard technical tokens (backticks, flags, angle tokens, URLs, paths, %s/%d): 0 failures.
- Width: 0 keys carry `maxCells`; 0 over budget; all delta keys measured against sidecar and note budgets.
- Punctuation: 0 failures; Latin/CJK spacing: 0 failures; 您 occurrences: 0.
- Numbers: 0 unexplained differences.
- Encoding: non-NFC 0; bidi controls 0. RTL checks: not applicable (LTR locale).

## Regression read

All 49 keys in the namespace were read side by side with the English source; the delta keys were judged against their English context notes as well. Findings marked *pre-existing* are in the current bytes but predate `c4ba95919d`; they are recorded because this audit vouches for the namespace, not only the delta.

## Limitations

No native TUI screenshot, relay or desktop render is claimed. Width is measured in cells (CJK/full-width = 2) against the sidecar `maxCells` where present and against note-stated budgets for the delta labels; the +25% growth figure in `pseudo-evidence.json` is a synthetic probe, not a rendered frame. zh-CN remains held and is not made shippable by this audit.
