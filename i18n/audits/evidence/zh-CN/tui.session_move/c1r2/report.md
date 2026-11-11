# c1-a2-zh-CN — Round 2 delta audit, locale zh-CN (22 keys)
Auditor: i18n-auditor (c1-a2-zh-CN), session d9a52b06834b, served model anthropic/claude-sonnet-5-5. Translator of v2: openai/gpt-6.1-sol (not me).
Candidate: c1-prep/zh-CN.v2.json, SHA-256 61adaca428a8f6dda453555b1081edf2c4c11af5b9fdbc81640dc1503b98d1be (re-hashed here with sha256sum; matches the translator).
**Verdict: clean-with-findings — BLOCKER 0, MAJOR 0, MINOR 0, NIT 2. Recommend `passed`.**

## 1. Delta v1→v2 (python3 diff): 13 changed, 9 unchanged; key set/order identical to keys.json
Every changed key judged; all accepted.
- created / opened: "已创建配置文件：{path}" / "已打开配置文件：{path}" — path at end after full-width colon, no CJK-glued path, no inserted space. OK. (NIT-1: source says "new"; 新 dropped — "创建" already implies it; was the same in v1, optional.)
- cleared: "已清除{key}：恢复为默认值" — correct present-state reading. OK.
- running_sessions_unchanged: see §4.
- list_desc_hosting: "AI托管平台（…）" — shipped `hosting` is 托管平台 in 6 values (cli.global_help.hosting, propagated_hosting, exec_worker_help.hosting, wire.errors.hosting.not_configured…); 提供商 for the English "provider platform" would diverge from the same concept elsewhere. OK; provider IDs verbatim.
- conversation_length / detail_length / max_learnings_history: "[已弃用，由压缩引擎取代]" — 压缩引擎 matches the 3 shipped wire.settings.*.help values for the same three settings (recount: exactly 3); no more 已…已. 压缩 is the glossary noun, 压缩引擎 a contextual form. detail_length "保留原文、不作摘要的最近消息条数": 条 = messages classifier per guide; meaning (recent messages left unsummarised) preserved. OK.
- classification_notice: "智能提示在对话中静默；调用及其费用记录在会话日志中" — matches shipped 智能提示 (11 values) and source ("silent in the chat; call and cost in session log"). OK.
- ambiguous: "更改工作目录用/move <路径>，移动会话用/move <会话> --to <对等设备>。" 更改 matches wire.slash.move (shipped); 移动 for session move (recount 16 values with 移动 vs 1 with 迁移, the latter cli.qwencloud_help.migrate = ticket-store data migration); 对等设备 for <peer> (recount 11 values incl. cli.sessions_help.peer/move_to). OK.
- to_needs_device: "--to需指定设备：/move --to <对等设备|local>" — literal `local` preserved; prose 设备 matches EN "device", hint 对等设备 matches EN "<peer|local>". OK.
- cannot_wait: 终端→shell: EN says "from a shell"; shipped keeps shell (19 values). OK.
- deadline_exceeded: see §4.

## 2. Declined finding adjudicated
- A-N1 (drop space after "] " in the four bracket tags, claim: full-width `]`): **overturn the original finding / uphold the decline.** Verified: the brackets in all four values are U+005B/U+005D (ASCII) — hex dump from python3; no full-width bracket exists. The space mirrors the EN source tag delimiter; the guide's "no space between Chinese and Latin/digit runs" rule concerns Latin/digit runs, not a source-mirroring tag delimiter; no shipped zh-CN value has a "] 中文" precedent either way (python3 scan: only [dim]/[red] Rich tags with no following space, a different construct). Cosmetic, not a defect. Also checked Rich: rendering each v2 value through rich.markup.render leaves plain text unchanged for all 22 (brackets are not parsed as markup).

## 3. Mechanical pass on v2 (python3, run here)
- 22/22 keys, order identical to keys.json; no empties; no value identical to English; Cf/Cc (bidi/zero-width/control) characters: none.
- ICU/brace placeholder multiset identical to EN for every key ({path},{key},{value},{description},{flag},{reason},{session},{to},{code}); no plural/select constructs in source.
- Backtick spans byte-identical to EN (cannot_wait, deadline_exceeded).
- Typed tokens present verbatim wherever in EN: /approvals auto, /move, --to, --keep, --queue, --drop-session-roles, --wait, local; placeholders <id> <to> <seconds> in backticks unchanged.
- Remaining Latin: provider IDs, CLI, AI, auto, shell, command/flag tokens — kept literals per guide/notes. No stray English prose.
- Punctuation per guide: full-width ，：；（）, no ！, no 您, no space inserted between CJK and Latin. Fit: no maxCells stated (contexts allow wrapping); longest line running_sessions_unchanged wraps, allowed.
- Use sites: cli.py:3028/3084-3089 and session_move.py:143/214-215 confirm keys are what v2 renders (config output; move errors).

## 4. Safety lines
- cli.config.running_sessions_unchanged v2: "运行中的会话保持不变，写入配置无法放宽其审批要求；需在每个想免除工具审批的会话中输入/approvals auto。新会话以auto模式启动。" — all four claims present: running sessions unchanged ✔; a config write *cannot* loosen (无法, stronger than v1 不会) ✔; /approvals auto must be typed in EACH session (每个…会话中, verbatim command) ✔; new sessions open at auto ✔. Consistent with cli.py:3019-3028 (loosening authorised only in the gate-holding process). PASS.
- tui.session_move.deadline_exceeded v2: "移动超时，结果尚不明确；再次运行`lop sessions move {session} --to {to}`可查看当前进度" — states unknown outcome (结果尚不明确), asserts neither success nor failure; re-run command byte-identical with {session}/{to}. EN only says "did not finish in time … reports where it got to"; adding 结果尚不明确 is a faithful, safe clarification (translator cites session_move.py:208-216: a timed-out subprocess does not prove the owner-side move failed). PASS. (Slight terseness of "移动超时" is fine: it reads as "the move timed out", not as failure.)

## 5. Findings
No BLOCKER/MAJOR/MINOR.
- NIT-1 cli.config.created — EN "Created new configuration file"; 新 omitted ("已创建配置文件：{path}"). Optional "已创建新的配置文件：{path}"; creation implies new.
- NIT-2 cli.config.list_desc_detail_length — "保留原文、不作摘要" is mildly redundant; "不作摘要的最近消息条数" would suffice. Optional.
Neighbour check: unchanged neighbours not contradicted by v2. Pre-existing shipped cli.sessions_help.move_* still says 对话 for "session" (guide says 会话) — outside this slice, not gating; the v2 wording follows the guide.

## Acceptance
Operator's "no English anywhere" bar: no English prose in any of the 22 values; only kept literals remain.

## Commands run
python3: v1/v2/keys.json diff, key-order, placeholder, backtick, token, Cf/Cc, bracket codepoint checks; python3 catalogue recounts over shipped zh-CN (移动 16, 迁移 1, 对等设备 11, 托管平台 6, 提供商 64, 压缩引擎 3, 已弃用 4, 智能提示 11, shell 19, 退出码 6); .venv/bin/python rich.markup.render over all values; reads of zh-CN style guide, cli.py, session_move.py, translator v2-changes. No MT service; nothing edited; no ledger/evidence written.
