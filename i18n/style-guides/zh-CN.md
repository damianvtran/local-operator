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
- No 啦 / 哦 / 呀 particles, no ！ in status copy.
- 请 marks the directive in zh-CN — a blocking dialog, a refusal or error body
  that tells the reader the next step, or a field prompt — whether or not the
  English says "please". English imperatives are unmarked, while a bare
  Chinese imperative in an error body reads curt next to the 请-led bodies
  that surround it (house practice, not a grammar rule); that is also why this
  guide does not carry `i18n/style-guides/en.md`'s ban on "please" outside
  blocking dialogs. Use 请, not a bare imperative, on the first next-step
  clause of such a body; later verbs chained in that clause do not repeat it
  (请在那台电脑上打开Local Operator，确认手机访问已开启，然后轻点“重试”。).
  Decide per value, not per surface — neighbouring values legitimately differ.
- Keep it at the head of the directive clause: value-, ICU-arm- or
  sentence-initial, or after ，；：—— or after a short lead-in (…前, …后,
  因此, 或) or a fronted topic noun (详情请…, 描述请控制在…以内) when a
  statement comes first — 接受前请审核更改内容。 / 恢复连接后请重试。 /
  会话未及时响应——请重试。 / 详情请运行`lop team-bridge receiver status`。 —
  never mid-predicate.
- At most once per clause: one 请 governs a chain of verbs, while a separate
  directive after ； or in a condition clause may carry its own
  (请重试；如果问题持续出现，请重新安装应用。 /
  请重试，若问题仍然存在，请联系支持团队。) — never two 请 in one clause.
- Not on buttons, menu items, bare state labels (已过期, 已连接) or
  single-sentence headings. A status line or a multi-sentence heading that
  carries a next step is an error body and takes it
  (你的Radient登录已过期。请登录以重新连接到{subject}。). A placeholder or
  filter hint inside an input takes no 请 at all (搜索对话……, 输入以筛选);
  "field prompt" means the label, instruction or validation sentence around a
  field (请输入整数。).
- Prohibitions: 请勿 in prose bodies (请勿重复发送); bare 不要 is fine in a
  safety refusal where blunt is intended (不要重试，并告知运行该智能体的人).
  Never write 请不要; a plain 不要 clause may follow a 请 clause when it is the
  "do A, not B" half of the same instruction (请先检查账户再重试，不要假定
  操作已回滚。).
- 请 introducing a request clause is often the verb "ask (someone) to…", not
  the directive 请 above: a third party is the subject and the value reports
  that party's request (都会在Chrome中请你点击Allow允许访问。 /
  智能体读取前会请你批准。 / 会请其他每个活跃成员也移到同一构建), or the
  reader is the subject and it is a plain instruction to ask someone
  (或请成员邀请你). English "Ask X to…" is rendered 请X… (lexical ask, exempt
  from the limits below) or 请让X… (directive 请 + 让: 请让组织所有者邀请你,
  请让手机重新发起配对, which stays under the head-of-clause and
  once-per-clause limits). Keep the 请 either way; do not replace it with a
  bare 让 or with 要求. For the lexical form the position, once-per-clause and
  no-button limits do not apply. A directive to the reader never takes a
  pronoun: 请点击……, not 请你点击……. 请求 (request, asks for) is a different
  word, out of scope for this rule.
- Existing values predate this rule: re-audit its 请 usage when you touch a
  namespace rather than sweeping the catalogues to comply.
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
