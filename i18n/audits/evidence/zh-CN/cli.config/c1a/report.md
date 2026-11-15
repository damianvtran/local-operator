# zh-CN · `cli.config` — independent audit (audit id `c1a`)

- Namespace: `cli.config` — 15 keys of `lop config` output (create/open/edit/list)
- Translation: `openai/gpt-6.1-sol`; auditor `i18n-auditor` (session `0ebc670d4b12`), served model `deepseek/deepseek-flash` — independent of the translator
- Source: `local_operator/i18n/catalogues/en/cli.config.json` (sha256 `40649e74…fda5ac`), sidecar `…cli.config.context.json` (`9831a776…ca90e78`)
- Guides: `i18n/style-guides/zh-CN.md`, `i18n/glossary.md`
- **Verdict: clean-with-findings — 0 BLOCKER, 0 MAJOR, 4 MINOR, 2 NIT** → ledger status `passed`

Mechanical: key set/order 15/15, placeholder sets identical, ICU parses and renders through the repo runtime
(`parse_message`/`render_message`) with argument tuples identical to en, `check.py --catalogues-only` clean, no
empty values, nothing identical to English, no stray braces. Full transcript: `audit.log`.

## MINOR

**M3 — `cli.config.opened` — missing locative; not parallel with `cli.config.created`.**
Current: `已打开配置文件{path}` → renders `已打开配置文件/tmp/config.yml`
Problem: the path is juxtaposed with no 在/于 and no separator, so it reads as if appended; `cli.config.created`
uses the locative pattern `已在{path}创建配置文件`. The source's own note requires the two to use the same noun —
they do (配置文件) — but the pair should also be structurally parallel, since they print one line apart.
Suggested: `已在{path}打开配置文件`.

**M4 — `list_desc_conversation_length`, `list_desc_detail_length`, `list_desc_max_learnings_history` — 压缩 vs shipped 压缩引擎.**
Current: `[已弃用，已由压缩取代] …`
Problem: the same three retired settings appear on the settings page as `已弃用。已被压缩引擎取代。`
(`wire.settings.conversation_length.help`, `.detail_length.help`, `.max_learnings_history.help` — all three
byte-identical there). Two surfaces describe one setting with two names for its successor.
Suggested: `[已弃用，已由压缩引擎取代] …` in all three rows.

**M5 — `list_desc_classification_notice` — narrows "silent in the chat".**
Current: `[已弃用] 智能提示不在对话中显示通知，调用及其费用记录在会话日志中`
Problem: en says Smart hints *is silent* in the chat (it prints nothing); the zh says it does not show a
*notification*, which is narrower and reads like a notification-suppression setting. The settings page for the
same setting says `已弃用。智能提示在对话中静默；调用记录见日志。`
Suggested: `[已弃用] 智能提示在对话中静默；调用及其费用记录在会话日志中`.

**M6 — `running_sessions_unchanged` — modal weakened, "each" lost. This is the group's safety line.**
Current: `运行中的会话保持不变，写入配置不会放宽其审批要求；要让某个会话无需工具审批，需在该会话中输入/approvals auto。新会话以auto模式启动。`
Problem: two separate drifts, neither of which changes the operative meaning:
1. en "a config write **cannot** loosen one" → `不会放宽` ("will not"), where the safety claim is an impossibility. `无法` states it as impossibility, as the source and the code comment do (`loosening_is_authorised` — the CLI's process holds no gate).
2. en "in **each** session you want ungated" → `某个会话` ("a given session"), which loses the iterate-per-session force; a reader with three running sessions can read one `/approvals auto` as enough.
Checks that passed: `/approvals auto` verbatim, `auto` kept as the mode word, per-session instruction present,
`新会话以auto模式启动` faithful. No polarity or permission error.
Suggested: `运行中的会话保持不变，写入配置无法放宽其审批要求；想在哪个会话免审批，就在该会话中输入/approvals auto。新会话以auto模式启动。`

## NIT

**N1 — the four bracket-tag rows — space after a full-width `]`.** `[已弃用，已由压缩取代] 对话历史…` copies the
en space; a full-width `]` already carries its own gap and the guide is otherwise strict about not inserting
spaces. Cosmetic. Drop it or record the source-mirroring choice deliberately.

**N2 — `list_desc_detail_length` — stiffer than its siblings.** `对话历史中不作摘要的最近消息数` stacks three
modifiers before 数; `对话历史中最近多少条消息不作摘要` parses faster and uses 条, the guide's classifier for messages.

## Checked, not defects

`配置项` matches the shipped `cli.config_help.list`; `说明：` matches `cli.team_bridge.outcome.note`;
`AI提供商平台（如…）` keeps provider ids verbatim and uses `、` for the enumeration; `是否自动保存对话` matches the
shipped label 自动保存对话; no 您/你; 。 kept only on the two full-sentence lines that carry it in en.
Pre-existing and not gated: the shipped `cli.sessions_help.move*` family calls the moved object 对话 where the
glossary says session = 会话.
