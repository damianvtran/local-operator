# Independent zh-CN audit — wire.errors.desktop — zh24a1

- Locale: `zh-CN` (held; not in `shipped.json`)
- Namespace: `wire.errors.desktop`
- Revision audited: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (catalogue bytes)
- Assembly revision: `5b5367be1b4acef44d564dcf8cbc1c7c06b209e8` — catalogue diff empty, intervening commits add evidence only
- Auditor: translation-auditor (lopdev team); session `3d1df2689651`; actual model `deepseek/deepseek-flash`; fallback: none
- Independent of the translator: yes (translator session is not the auditor session)
- Keys in namespace: **84**; keys read: **84**
- Delta keys audited: **1** (1 new key)
- Verdict: **clean_with_followups** (status `passed`); unresolved blocker/major: **0**; minor: 1; nit: 0

## Input hashes (sha256)

- en: `9a164be42a80d9bf0f60ad2a9f1d76fd1601855a4d19ffac3bf35d6bef11f76d` (`local_operator/i18n/catalogues/en/wire.errors.desktop.json`)
- context: none (no context sidecar)
- target: `a8cb0db6bbc8a33ef041c082d68b9c06dd0b3168a303e447e0f40e0334fc89f4` (`local_operator/i18n/catalogues/zh-CN/wire.errors.desktop.json`)
- policy: `i18n/glossary.md` `ef07b446668778822d0ff4dcd7b380543456ddf7b08e060dce5818cb8fd6fdcd`; `i18n/style-guides/zh-CN.md` `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c`

## Delta keys in scope

- `wire.errors.desktop.fast_mode_unsupported`

## Findings

### ZH-A5 — minor — pre-existing regression read (not introduced by c4ba95919d)

The same product noun “aside” has two Chinese renderings: 题外问答 in the feature title (tui.small.aside.title) and the unsaved-asides summary (tui.notices.unsaved_asides), but 旁支对话 in all seven wire.errors.desktop.aside_* errors. Standardise on the feature title 题外问答 (or 题外对话) across both surfaces; the current split is visible when a user opens the panel and then hits an error from it.

- `wire.errors.desktop.aside_adopt_unconfirmed`
  - current: ``确认将此旁支对话加入会话``
  - proposed: ``确认将此题外问答加入会话``
- `wire.errors.desktop.aside_limit_reached`
  - current: ``关闭一个旁支对话或等待其过期``
  - proposed: ``关闭一个题外问答或等待其过期``
- `wire.errors.desktop.aside_max_turns`
  - current: ``交流16轮后需新建旁支对话``
  - proposed: ``交流16轮后需新建题外问答``
- `wire.errors.desktop.aside_not_adoptable`
  - current: ``无法将此旁支对话纳入会话``
  - proposed: ``无法将此题外问答纳入会话``
- `wire.errors.desktop.aside_request_reused`
  - current: ``此旁支对话请求已使用过``
  - proposed: ``此题外问答请求已使用过``
- `wire.errors.desktop.aside_still_running`
  - current: ``等待旁支对话完成后再关闭``
  - proposed: ``等待题外问答完成后再关闭``
- `wire.errors.desktop.aside_unavailable`
  - current: ``此旁支对话已不可用``
  - proposed: ``此题外问答已不可用``

Note: tui.small.aside.title and tui.notices.unsaved_asides already hold the proposed term and need no change.

## Mechanical results

- Key parity: en 84 / zh 84; missing 0; extra 0.
- ICU argument parity (names and kinds): 84 keys checked, 0 failures.
- Message parse/render: 84 messages rendered with synthetic parameters, 0 failures.
- Plural structure: 0 plural keys; failures 0. zh-CN's other-only arms are expected by the style guide.
- Hard technical tokens (backticks, flags, angle tokens, URLs, paths, %s/%d): 0 failures.
- Width: 0 keys carry `maxCells`; 0 over budget; all delta keys measured against sidecar and note budgets.
- Punctuation: 0 failures; Latin/CJK spacing: 0 failures; 您 occurrences: 0.
- Numbers: 0 unexplained differences.
- Encoding: non-NFC 0; bidi controls 0. RTL checks: not applicable (LTR locale).

## Regression read

All 84 keys in the namespace were read side by side with the English source; the delta keys were judged against their English context notes as well. Findings marked *pre-existing* are in the current bytes but predate `c4ba95919d`; they are recorded because this audit vouches for the namespace, not only the delta.

## Limitations

No native TUI screenshot, relay or desktop render is claimed. Width is measured in cells (CJK/full-width = 2) against the sidecar `maxCells` where present and against note-stated budgets for the delta labels; the +25% growth figure in `pseudo-evidence.json` is a synthetic probe, not a rendered frame. zh-CN remains held and is not made shippable by this audit.
