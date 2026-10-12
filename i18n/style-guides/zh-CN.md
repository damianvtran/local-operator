# Style guide — Simplified Chinese (`zh-CN`)

Read with `i18n/glossary.md`; everything here is zh-CN-specific and binding
for translators and the auditor.

## Formality & tone

- Address: **你** — recorded decision: modern product voice (messaging-class
  apps and developer tools overwhelmingly use 你). 您 reads customer-service
  or government and fights the terse dev tone; 您 is rejected for v1.
- No case to worry about; keep Latin acronyms uppercase (API, CLI, TUI).
- Terse: four-to-eight character lines where possible; verbs for actions
  (保存, 继续, 重试, 取消).
- No sentence-final 。 on short labels and statuses; full sentences keep it.
- No 啦 / 哦 / 呀 particles, no ！ in status copy. "请" only where the English
  says "please" (blocking dialogs).
- No 您/你 repetition: Chinese drops pronouns naturally — prefer verb-forward
  phrasing ("会话已保存" over "你的会话已被保存").

## Product-name policy

- Keep exactly as in glossary §1: Local Operator, `lop`, tool names,
  `--flags`, config keys, paths, identifiers — as Latin runs, never
  transliterated.
- Kept-English technical list: API, CLI, TUI, JSON, YAML, HTTP, URL, GPU, and
  **token** for LLM tokens (auth tokens are 令牌 — note the split).
- Prefer settled Chinese where it exists: 服务器 (server), 文件 (file), 文件夹
  (folder), 网络 (network), 设置 (settings), 设备 (device), 缓存 (cache),
  提交 (commit), 审批 (approvals), 用量 (usage), 费用 (cost).
- Avoid 词元 for LLM tokens (new, low recognition); keep the Latin "token".

## Locale calls (zh-CN)

- session → 会话 · wake → 唤醒 · agent → 智能体 · subagent → 子智能体
- tool trace → 轨迹 · transcript → 记录 (会话记录 when the noun stands alone)
- schedule → 定时 (定时任务 for an item) · monitor → 监控 · relay → 中继
- workspace → 工作区 · prompt → 提示词 · compaction → 压缩 (上下文压缩)

## Plurals

- `other` only — Chinese has no plural agreement; never add one/other
  branches. Example: `{count, plural, other {# 个会话}}`
- Use the noun's natural classifier (个 default; 条 for messages; 项 for
  items); when in doubt, 个.

## Numbers, dates, digits

- Never hand-written — the formatter renders "1,234.5" and dates
  "2026年10月10日"; 24-hour times ("14:30").
- Digits: Latin (v1). Do not convert counts to 万/亿 units unless the source
  does.

## Width rules

- Every CJK character and every full-width punctuation mark is **2 cells**;
  ASCII is 1. A `maxCells` budget is therefore roughly half the characters
  for Chinese text — count with the tooling, not by eye.
- Budgets: `maxCells` from the en sidecar; pinned: PERSIST_HINT 42/43 cells,
  tip pool 59, `/help` ≈ 55.
- Prefer a shorter idiomatic wording over truncation ("重试" over "重新尝试").

## Punctuation & typography

- Full-width punctuation for prose: ，。：；！？、；“”（）. Quotes are “ ”
  (mainland convention; 「」 is not used).
- Ellipsis is …… (two marks), not "...".
- No space inserted between Chinese text and Latin/digit runs (width
  discipline; matches mainland software convention); Latin punctuation only
  inside Latin or code fragments.
- Code spans, flags, paths stay byte-identical — half-width, untranslated.
