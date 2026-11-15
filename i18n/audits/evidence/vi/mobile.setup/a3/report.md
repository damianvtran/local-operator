# Independent translation audit — vi / mobile.setup — a3

- PR: #3958 (branch fix/tunnel-remedy-no-commands, stacked on #3945), head 01113e5aa048884a507d9bb4868815f909f63a48; scope `git diff 374037856..01113e5aa -- local_operator/i18n/` (12 new `*_plain` keys; 0 existing keys changed).
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (values committed in 01113e5aa on fix/tunnel-remedy-no-commands; English source by the same coder); model not disclosed to the auditor (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session 9054cd229d3f; not the translator/coder of host session 25fe8dfbf610; edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 3 minor/nit finding(s), none gating.

## Method
Read side by side with en source + context sidecar (12 new keys, 0 changed); glossary agent/approvals/service and the namespace's phone-access, phone-address, sign-in, account and setup terms match #3945's shipped copy; register per style guide; no placeholders in source or target; no command, flag, slash or code-span text (setup._SHELL_COMMAND + slash/flag/backtick scan); NFC; no bidi controls; no maxCells on these keys — the TUI's 50-cell park card renders tunnel.status card_* keys, not these; the *_plain keys reach the TUI only through setup.text(surface='tui') in a wrapping system notice and the desktop's remedy.text. Runtime: setup.plain_remedy({'step': s}) equals the catalogue value for all 12 steps.

## Locale notes
bạn register with pronoun dropping; 'tác nhân', 'phê duyệt', 'dịch vụ chạy nền', 'giao diện điện thoại' consistent.

## Findings
- VI3-01 minor (cloudflared_plain, mobile_bundle_plain): '(tác nhân sẽ xin phê duyệt của bạn trước)' is a calque of 'ask for your approval'. Proposed parenthetical: '(tác nhân sẽ xin bạn phê duyệt trước)'.
- VI3-02 minor (complete_plain): 'Mở địa chỉ ở trên trên điện thoại…' doubles 'trên'. Proposed: 'Dùng điện thoại mở địa chỉ ở trên, rồi đăng nhập bằng cùng tài khoản Radient.'

## Keys read (12)
| key | en | vi |
|---|---|---|
| `mobile.setup.stopped_plain` | Phone access is turned off on this computer. Turn it back on whenever you want to reach this computer from your phone. | Truy cập điện thoại đang tắt trên máy tính này. Hãy bật lại bất cứ khi nào bạn muốn truy cập máy tính này từ điện thoại. |
| `mobile.setup.owner_missing_plain` | The Radient account that set up phone access is no longer signed in on this computer. Sign in with that account to bring phone access back. | Tài khoản Radient đã thiết lập truy cập điện thoại không còn đăng nhập trên máy tính này. Hãy đăng nhập bằng tài khoản đó để khôi phục truy cập điện thoại. |
| `mobile.setup.login_plain` | Sign in to Radient with the account that owns phone access, then try again. | Hãy đăng nhập Radient bằng tài khoản sở hữu truy cập điện thoại, rồi thử lại. |
| `mobile.setup.choose_account_plain` | Choose the Radient account phone access should use. Setup never moves an existing phone address to a different account. | Chọn tài khoản Radient mà truy cập điện thoại sẽ dùng. Thiết lập không bao giờ chuyển một địa chỉ điện thoại hiện có sang tài khoản khác. |
| `mobile.setup.account_plain` | Radient could not confirm the account and phone address. Check your internet connection and try again; nothing was changed. | Radient không thể xác nhận tài khoản và địa chỉ điện thoại. Hãy kiểm tra kết nối internet rồi thử lại; chưa có gì thay đổi. |
| `mobile.setup.cloudflared_plain` | Phone access needs a helper app called cloudflared, which is not installed on this computer. Ask your agent to install it (it asks for your approval first), then try again. | Truy cập điện thoại cần một ứng dụng hỗ trợ tên là cloudflared, nhưng ứng dụng này chưa được cài trên máy tính này. Hãy nhờ tác nhân của bạn cài đặt (tác nhân sẽ xin phê duyệt của bạn trước), rồi thử lại. |
| `mobile.setup.mobile_bundle_plain` | The phone interface is not prepared on this computer yet. Ask your agent to prepare it (it asks for your approval first), then try again. Nothing was changed. | Giao diện điện thoại chưa được chuẩn bị trên máy tính này. Hãy nhờ tác nhân của bạn chuẩn bị (tác nhân sẽ xin phê duyệt của bạn trước), rồi thử lại. Chưa có gì thay đổi. |
| `mobile.setup.supervisor_plain` | This computer cannot keep phone access running in the background yet. Ask your agent to set up a background service for it, then try again. Nothing was changed. | Máy tính này chưa thể duy trì truy cập điện thoại chạy nền. Hãy nhờ tác nhân của bạn thiết lập một dịch vụ chạy nền cho việc này, rồi thử lại. Chưa có gì thay đổi. |
| `mobile.setup.apply_plain` | Everything is ready. Turn on phone access to finish. | Mọi thứ đã sẵn sàng. Hãy bật truy cập điện thoại để hoàn tất. |
| `mobile.setup.verification_plain` | The connection is not confirmed yet. Check again in a moment; completed steps stay saved. | Kết nối chưa được xác nhận. Hãy kiểm tra lại sau giây lát; các bước đã hoàn tất vẫn được giữ. |
| `mobile.setup.failed_plain` | Phone access setup did not finish. Try again; completed steps stay saved. | Thiết lập truy cập điện thoại chưa hoàn tất. Hãy thử lại; các bước đã hoàn tất vẫn được giữ. |
| `mobile.setup.complete_plain` | Open the address above on your phone and sign in with the same Radient account. | Mở địa chỉ ở trên trên điện thoại của bạn và đăng nhập bằng cùng tài khoản Radient. |
