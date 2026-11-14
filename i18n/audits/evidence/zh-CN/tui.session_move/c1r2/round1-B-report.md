# c1-au-zh-CN-B — independent audit, locale zh-CN (22 keys)
Auditor: i18n-auditor (c1-au-zh-CN-B); served model: anthropic/claude-sonnet-5-5. Translator: openai/gpt-6.1-sol (not me).
Verdict: clean-with-findings — {blocker 0, major 0, minor 5, nit 3}. Recommend `passed`.

## Mechanical (python3, run here)
- Key set and order identical to keys.json (22/22); no empties; no value identical to English.
- ICU placeholders identical per key ({path},{key},{value},{description},{flag},{reason},{session},{to},{code}).
- Backtick spans byte-identical (cannot_wait, deadline_exceeded). Typed tokens present verbatim: `/approvals auto`, `/move`, `--to`, `--keep`, `--queue`, `--drop-session-roles`, `--wait`, `local`, `<id> <to> <seconds>`. (My first regex pass flagged 3 tokens only because of a CJK lookbehind artifact; direct substring check: all present.)
- No stray English prose. Remaining Latin runs are provider names and the `auto` mode word, as the notes require.
- Terminal width: no cap stated. Longest line is running_sessions_unchanged at 131 cells (may wrap, allowed). list_title is 6 cells (fits the rule).
- Punctuation follows the style guide: full-width ，：；（）; no space between CJK and Latin/code; 你-register avoided via pronoun-drop; no ！.

## Safety lines (meaning fidelity)
- running_sessions_unchanged: PASS. "运行中的会话保持不变，写入配置不会放宽其审批要求" keeps the claim that a config write cannot loosen running sessions. `/approvals auto` is verbatim and the per-session instruction is kept. "新会话以auto模式启动" is kept. See M2 for the one softening.
- deadline_exceeded: PASS. "迁移未在限定时间内完成" says only "did not finish in time", with no failure and no success claim. The re-run command is verbatim and "可查看当前进度" matches "reports where it got to".
- cannot_wait, unsupported_flag, to_needs_device, cli_*: faithful, with no polarity or omission errors. `local` stays English inside <设备|local>.

## Findings
M1 MINOR cli.config.created — "已在{path}创建配置文件". The path sits mid-sentence, glued to CJK text. Terminal path linkifiers and double-click selection treat CJK as word characters, so the selectable or clickable token becomes "…/config.yml创建配置文件". The style guide forbids inserting a space, but moving the path to the end fixes it with no space. Suggest: "已创建配置文件：{path}". Same pattern, weaker (path already last, but undelimited), cli.config.opened "已打开配置文件{path}" → "已打开配置文件：{path}" (shipped convention: "配置文件夹：{path}").
M2 MINOR cli.config.running_sessions_unchanged — "每个/each" is lost: "要让某个会话无需工具审批，需在该会话中输入/approvals auto". This is still per-session, but the safety line should stress it. Suggest: "要让哪个会话无需工具审批，就在那个会话中输入/approvals auto" or "需在每个想放开审批的会话中输入/approvals auto". Also "[已]…保持不变，写入配置不会放宽其审批要求" is fine.
M3 MINOR cli.config.list_desc_hosting — "AI提供商平台". The shipped zh-CN catalogue renders the `hosting` concept as 托管平台 (cli.global_help.hosting, cli.exec_worker_help.hosting, wire.errors.hosting.not_configured). The same setting is described with a different term here. Suggest "AI托管平台（如radient、openai、deepseek、anthropic、openrouter）".
M4 MINOR the four [DEPRECATED …] lines (conversation_length, detail_length, max_learnings_history) — "[已弃用，已由压缩取代]". "已…已" is clumsy, and bare "压缩" is the glossary short form; the glossary/style guide prefer 上下文压缩 when the noun stands alone, and the shipped sibling says "已被压缩引擎取代". Suggest "[已弃用，由上下文压缩取代]" in all three. The classification_notice line "[已弃用]" is correct.
M5 MINOR tui.session_move.ambiguous — "切换工作目录用/move <路径>". Shipped zh-CN wording for the working-directory change is 更改（"更改此会话的工作目录"）; 切换 reads like switching between existing dirs and clashes with "迁移" two clauses later (fine) but not with 更改. Suggest "更改工作目录用/move <路径>，迁移会话用/move <会话> --to <设备>。" (Low-risk; the current text is understandable.)
N1 NIT cli.config.cleared — "将重新使用默认值" is future tense, while the English means it now reads as default. Suggest "已清除{key}：恢复为默认值".
N2 NIT cli.config.list_desc_detail_length — "不作摘要的最近消息数" is OK; "保留原文、不作摘要的最近消息数" would be clearer. Optional.
N3 NIT cli.config.list_desc_classification_notice — "不在对话中显示通知" is slightly wider than "silent in the chat", but it is correct. Shipped sibling: "智能提示在对话中静默；调用记录见日志". Terminology matches (智能提示). Keep.

## Terminology / register checks
智能提示 (shipped), 压缩 (glossary), 审批/会话/设备 (style guide), 配置键/配置项/配置文件 (shipped), 工作目录 (shipped), 已弃用 (shipped) are all consistent. cli.config.list_title "配置项" matches shipped "列出可用配置项及其说明". Register is neutral/terse with no 您 and no particles.

## Commands run
python3 key/order/placeholder/backtick/token/Latin-run/width checks over keys.json vs zh-CN.json; reads of zh-CN style guide, shipped zh-CN catalogues (cli.*, tui.*), and the session_move.py use sites. No MT service used. Nothing edited. This report is at the scratchpad path below.
