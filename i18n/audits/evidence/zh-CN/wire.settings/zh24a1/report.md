# Independent zh-CN audit — wire.settings — zh24a1

- Locale: `zh-CN` (held; not in `shipped.json`)
- Namespace: `wire.settings`
- Revision audited: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (catalogue bytes)
- Assembly revision: `5b5367be1b4acef44d564dcf8cbc1c7c06b209e8` — catalogue diff empty, intervening commits add evidence only
- Auditor: translation-auditor (lopdev team); session `3d1df2689651`; actual model `deepseek/deepseek-flash`; fallback: none
- Independent of the translator: yes (translator session is not the auditor session)
- Keys in namespace: **818**; keys read: **818**
- Delta keys audited: **18** (18 new keys)
- Verdict: **clean_with_followups** (status `passed`); unresolved blocker/major: **0**; minor: 2; nit: 0

## Input hashes (sha256)

- en: `c1e237c4dd3c2ea236f9e39d15cf4d0b34a231841357aee5a389f34201a73582` (`local_operator/i18n/catalogues/en/wire.settings.json`)
- context: `67662db2835ed0f6db3a65844efc318c79706735457eb974bb7c64cf34f861a9` (`local_operator/i18n/catalogues/en/wire.settings.context.json`)
- target: `d0df5e3e6975719b39d3e81171b2d9416844a1df8a5498f81f398476eb284b55` (`local_operator/i18n/catalogues/zh-CN/wire.settings.json`)
- policy: `i18n/glossary.md` `ef07b446668778822d0ff4dcd7b380543456ddf7b08e060dce5818cb8fd6fdcd`; `i18n/style-guides/zh-CN.md` `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c`

## Delta keys in scope

- `wire.settings.remote_compute.description`
- `wire.settings.remote_compute.title`
- `wire.settings.remote_compute_enabled.choice.false.description`
- `wire.settings.remote_compute_enabled.choice.true.description`
- `wire.settings.remote_compute_enabled.help`
- `wire.settings.remote_compute_enabled.label`
- `wire.settings.session_compression.description`
- `wire.settings.session_compression.title`
- `wire.settings.session_compression_after_days.help`
- `wire.settings.session_compression_after_days.label`
- `wire.settings.session_compression_enabled.choice.false.description`
- `wire.settings.session_compression_enabled.choice.true.description`
- `wire.settings.session_compression_enabled.help`
- `wire.settings.session_compression_enabled.label`
- `wire.settings.session_scratchpad.description`
- `wire.settings.session_scratchpad.title`
- `wire.settings.session_scratchpad_max_bytes.help`
- `wire.settings.session_scratchpad_max_bytes.label`

## Findings

### ZH-A4 — minor — pre-existing regression read (not introduced by c4ba95919d)

“Interpreter” is rendered 解析器 (“parser”) in both help lines while the sibling label bash_shell.label already says bash解释器. 解析器 names a parser, not a shell interpreter; the two help lines are internally inconsistent with the label and technically wrong.

- `wire.settings.bash_shell.help`
  - current: ``bash工具的解析器。留空使用PATH里的bash，否则使用/bin/sh。``
  - proposed: ``bash工具的解释器。留空使用PATH里的bash，否则使用/bin/sh。``
- `wire.settings.bash_shell.help.windows`
  - current: ``bash工具的解析器。留空：PATH里的bash，否则Git的bash.exe。``
  - proposed: ``bash工具的解释器。留空：PATH里的bash，否则Git的bash.exe。``

Note: No placeholder or width change.

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

- Key parity: en 818 / zh 818; missing 0; extra 0.
- ICU argument parity (names and kinds): 818 keys checked, 0 failures.
- Message parse/render: 818 messages rendered with synthetic parameters, 0 failures.
- Plural structure: 0 plural keys; failures 0. zh-CN's other-only arms are expected by the style guide.
- Hard technical tokens (backticks, flags, angle tokens, URLs, paths, %s/%d): 0 failures.
- Width: 7 keys carry `maxCells`; 0 over budget; all delta keys measured against sidecar and note budgets.
- Punctuation: 1 failures; Latin/CJK spacing: 0 failures; 您 occurrences: 0.
- Numbers: 0 unexplained differences. Whitelisted expected transforms present in this namespace: scheduled_count's one-branch literal and/or 55k→55000.
- Encoding: non-NFC 0; bidi controls 0. RTL checks: not applicable (LTR locale).

## Regression read

All 818 keys in the namespace were read side by side with the English source; the delta keys were judged against their English context notes as well. Findings marked *pre-existing* are in the current bytes but predate `c4ba95919d`; they are recorded because this audit vouches for the namespace, not only the delta.

## Limitations

No native TUI screenshot, relay or desktop render is claimed. Width is measured in cells (CJK/full-width = 2) against the sidecar `maxCells` where present and against note-stated budgets for the delta labels; the +25% growth figure in `pseudo-evidence.json` is a synthetic probe, not a rendered frame. zh-CN remains held and is not made shippable by this audit.
