# Independent translation audit — zh-CN / mobile.setup — a3

- PR: #3958 (branch fix/tunnel-remedy-no-commands, stacked on #3945), head 01113e5aa048884a507d9bb4868815f909f63a48; scope `git diff 374037856..01113e5aa -- local_operator/i18n/` (12 new `*_plain` keys; 0 existing keys changed).
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (values committed in 01113e5aa on fix/tunnel-remedy-no-commands; English source by the same coder); model not disclosed to the auditor (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session 9054cd229d3f; not the translator/coder of host session 25fe8dfbf610; edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 1 minor/nit finding(s), none gating.

## Method
Read side by side with en source + context sidecar (12 new keys, 0 changed); glossary agent/approvals/service and the namespace's phone-access, phone-address, sign-in, account and setup terms match #3945's shipped copy; register per style guide; no placeholders in source or target; no command, flag, slash or code-span text (setup._SHELL_COMMAND + slash/flag/backtick scan); NFC; no bidi controls; no maxCells on these keys — the TUI's 50-cell park card renders tunnel.status card_* keys, not these; the *_plain keys reach the TUI only through setup.text(surface='tui') in a wrapping system notice and the desktop's remedy.text. Runtime: setup.plain_remedy({'step': s}) equals the catalogue value for all 12 steps.

## Locale notes
你 register, pronouns mostly dropped; full-width ，。；（）; no CJK–Latin spacing, matching the namespace; 智能体/审批/后台服务/手机界面 consistent.

## Findings
- ZH3-01 nit (cloudflared_plain, mobile_bundle_plain): '会先征求你的审批' — 征求 collocates with 意见/同意 rather than 审批; optional '会先请你审批'.

## Keys read (12)
| key | en | zh-CN |
|---|---|---|
| `mobile.setup.stopped_plain` | Phone access is turned off on this computer. Turn it back on whenever you want to reach this computer from your phone. | 这台电脑的手机访问已关闭。想从手机访问这台电脑时，随时可以重新开启。 |
| `mobile.setup.owner_missing_plain` | The Radient account that set up phone access is no longer signed in on this computer. Sign in with that account to bring phone access back. | 设置手机访问的Radient账户已不再在这台电脑上登录。使用该账户登录即可恢复手机访问。 |
| `mobile.setup.login_plain` | Sign in to Radient with the account that owns phone access, then try again. | 使用手机访问所属的账户登录Radient，然后重试。 |
| `mobile.setup.choose_account_plain` | Choose the Radient account phone access should use. Setup never moves an existing phone address to a different account. | 选择手机访问要使用的Radient账户。设置绝不会把现有的手机地址转到其他账户。 |
| `mobile.setup.account_plain` | Radient could not confirm the account and phone address. Check your internet connection and try again; nothing was changed. | Radient无法确认账户和手机地址。请检查网络连接后重试；未作任何更改。 |
| `mobile.setup.cloudflared_plain` | Phone access needs a helper app called cloudflared, which is not installed on this computer. Ask your agent to install it (it asks for your approval first), then try again. | 手机访问需要一个名为cloudflared的辅助应用，但这台电脑尚未安装。请让智能体安装它（会先征求你的审批），然后重试。 |
| `mobile.setup.mobile_bundle_plain` | The phone interface is not prepared on this computer yet. Ask your agent to prepare it (it asks for your approval first), then try again. Nothing was changed. | 这台电脑尚未准备好手机界面。请让智能体准备它（会先征求你的审批），然后重试。未作任何更改。 |
| `mobile.setup.supervisor_plain` | This computer cannot keep phone access running in the background yet. Ask your agent to set up a background service for it, then try again. Nothing was changed. | 这台电脑暂时无法让手机访问在后台持续运行。请让智能体为此设置一个后台服务，然后重试。未作任何更改。 |
| `mobile.setup.apply_plain` | Everything is ready. Turn on phone access to finish. | 一切就绪。开启手机访问即可完成。 |
| `mobile.setup.verification_plain` | The connection is not confirmed yet. Check again in a moment; completed steps stay saved. | 连接尚未确认。请稍后再检查；已完成的步骤仍然保留。 |
| `mobile.setup.failed_plain` | Phone access setup did not finish. Try again; completed steps stay saved. | 手机访问设置未完成。请重试；已完成的步骤仍然保留。 |
| `mobile.setup.complete_plain` | Open the address above on your phone and sign in with the same Radient account. | 在手机上打开上方地址，并使用同一Radient账户登录。 |
