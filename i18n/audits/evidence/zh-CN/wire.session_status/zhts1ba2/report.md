# FIRST independent zh-CN audit — wire.session_status

Verdict: **passed** for the supplied candidate bytes. No BLOCKER, MAJOR, MINOR or NIT findings. This is ONE of the two required independent safety-copy audits, not combined approval.

Auditor: i18n-auditor, session `ba2da228c32c`, actual model `openai/gpt-6.1-sol` (host session Model field; no fallback used). Independent from translator session `e9c45b5505eb`; same model family does not imply same session or authorship. I did not translate or edit the target. Native-language reasoning only; no translation/MT service used.

Frozen source revision: `2930492e05a6968a2e1adfe34a091b94802bacd0`.
English: `52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326`.
Context: `5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525`.
Candidate: `6461115924a504b57c11d077a0e735cf08af4bdae59b17997e81b4fc5bf4a649`.

## Method and evidence boundary

Read zh-CN style guide and glossary first, followed by the English namespace and its context sidecar, then the complete candidate. Every key and every cause/command arm was reviewed semantically, not inferred from validator green. Ran the repository's `scripts.i18n.check.main(["--catalogues-only"])` using its existing interpreter against an isolated scratch catalogue containing only this English namespace/context and this candidate. Candidate normal admission and restored admission each returned 0 problems/exit 0. Five negative controls returned exit 1: renamed placeholder, removed key, extra key, forbidden Chinese `one` category, malformed ICU. Independently verified duplicate-free JSON, all 17 keys, argument name/type equality, every select selector/nested argument structure, Chinese `other`-only plural, unchanged code/markup tokens, no embedded newlines, historical retired_label unchanged.

Executed 54 actual ICU renders: 10 plain/legacy cases, 12 command cases (four keys × none, /stop --all, /kill --all), 28 cause cases (two keys × 13 declared selectors plus unknown-selector fallback), four dormant counts (0/1/2/55). See sample JSON and complete execution log. `none` omits attribution; technical command bytes remain exact. Unknown cause selects localized other, not the raw cause token.

First runner attempt reached passing catalogue controls/structure checks but then failed because `wcwidth` is not installed in the repository interpreter. Its actual failed log is retained, not relabeled success. No package installed. Complete rerun uses dependency-free Unicode East Asian W/F=2, combining=0, otherwise=1 measurement, appropriate to these CJK/ASCII samples (not a general emoji/grapheme width algorithm).

No sidecar maxCells/maxChars cap exists. Maximum tested line: 50 estimated terminal cells / 26 codepoints, unseen runtime_killed. This is a fit-risk measurement only: sidebar/picker clipping, actual terminal font/layout, animation and full runtime behavior were NOT exercised or cleared.

## Per-key audit (17/17)

| Key suffix | Current text / covered bodies | Assessment |
|---|---|---|
| retired_label | `{base}——因更新已退役` | Historical value preserved exactly; `base` retained. 退役 is less idiomatic for a running turn, but this carried historical entry is explicitly outside the new formatter. Not a new-copy finding. The legacy English-base demonstration is not a new formatter English leak. |
| interrupted | 已中断 | Natural terse factual status; no invented cause. |
| unseen_interruption | 未读：已中断 | 未读 explicitly marks unread outcome, not a new interruption. Full-width colon and no final stop fit status style. |
| retired | 已中断——为更新而退出 | Update purpose is explicit; no operator-stop attribution. 退出 is more natural than historical 退役 here. |
| unseen_retirement | 未读：已中断——为更新而退出 | Same cause plus correct unread state. |
| signal_stop | 已中断——收到信号后停止 + none/（通过{command}） | Received signal leads to stop, distinct from force-kill. Connector reads naturally and none does not fabricate a command. |
| unseen_signal_stop | 未读：已中断——收到信号后停止 + none/（通过{command}） | Unread signal-stop; both arms preserve meaning and command bytes. |
| forced_kill | 已中断——已强制终止 + none/（通过{command}） | Explicit forceful termination, not a clean signal stop. Both command arms good. |
| unseen_forced_kill | 未读：已中断——已强制终止 + none/（通过{command}） | Correct unread forced-termination outcome, all arms good. |
| complete | 已完成 | Idiomatic completed state. |
| unseen_completion | 未读：已完成 | Correct unread completion; does not turn success into failure. |
| recent | 近期活动 | Natural activity recency label in session-list context. |
| dormant | 已停止（{count, plural, other {#个唤醒已休眠}}） | 唤醒 matches glossary; 已休眠 is inactive/dormant, not canceled and not promised to fire. Natural classifier 个, Latin digits, correct other-only Chinese plural. All 0/1/2/55 outputs checked. |
| error | 出错 | Natural short error status; does not add forensic diagnosis. |
| unseen_error | 未读：出错 | Same error plus unread marker. |
| error_cause | Full cause select, detailed below | All 13 declared arms + actual unknown fallback checked. |
| unseen_error_cause | Same cause select prefixed 未读： | All 13 arms preserve the seen form's cause; unread marker consistently applied. |

### Every cause arm (applies to both error_cause keys)

| Machine selector (unchanged) | Chinese body after 出错—— (or fallback) | Meaning/safety judgment |
|---|---|---|
| user_stop | 用户已停止 | Clearly names user action; does not conflate update retirement. Elliptical object is normal in a session-status row. |
| runtime_stall_bound | 运行时停滞超时后自行退出 | Names stall bound and self-exit, not an outside kill. |
| continuation_limit | 已达续接上限；待处理消息已丢弃 | Preserves reached limit and dropped pending message. 续接 is intelligible technical continuation wording in this context, not a binding-glossary violation. 丢弃 is explicit data-loss wording, not softened. |
| output_limit | 已达模型输出上限；无法继续 | Distinguishes output cap from continuation cap, preserves inability to continue. |
| runtime_retired | 为更新而退出 | Explicit update agency, no user-stop implication. |
| runtime_shutdown | 运行时在本轮中终止 | Preserves during-turn termination without inventing user or update cause. |
| runtime_overdue | 运行时超过进度时限后退出以更新 | Preserves progress bound and leaving to update; does not say the update failed. |
| runtime_update_failed | 更新未完成；运行时仍使用已加载的构建 | Preserves incomplete update and continued use of loaded build, not a successful update/new build. |
| runtime_killed | 运行时未正常退出便消失；未收到停止请求 | Preserves absence of clean exit and absence of stop request; does not assign operator blame or assert a cause. |
| install_mid_update | 安装版本在本轮中被替换 | Correct during-turn installation replacement; 安装版本 is understandable product/technical copy, not an instruction to reinstall. |
| owner_lost | 运行时在本轮中停止响应 | Correct non-response during turn, not proved termination. |
| disposed | 会话在本轮中被释放 | Technical resource-disposal meaning preserved without inventing data deletion. 会话 matches glossary. |
| other / unknown selector | 出错 / 未读：出错 | Generic localized fallback, no raw cause or English base. |

## Scope limits and retention handoff

Unknown nonempty external diagnostic text intentionally retains the legacy English branch OUTSIDE this namespace. This remains an unresolved programme localization gap, not an approved exception and not a finding against these candidate bytes. No claim of all-status-copy completion.

Only scratch artifacts written. No target, repository, generated, history, runtime config, build, install, server or PR changes. Manager must retain this bundle at the proposed canonical root `i18n/audits/evidence/zh-CN/wire.session_status/zhts1ba2/` and append the provided receipt using a freshly measured history hash. The dry-run validates receipt shape/append compatibility only; it does NOT append and does NOT establish tracked retention. Second independent safety-copy audit and real rendered/runtime validation remain separate gates.

This report is `audit_notes_not_execution_log`; only the linked actual `.log` files prove command execution.
