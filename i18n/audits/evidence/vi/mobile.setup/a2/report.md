# i18n audit a2 — vi / mobile.setup (local-operator PR #3945, round-1 remediation)

Head: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes (worktree .worktrees/tunnel-remedies-auto-up; `git diff HEAD -- local_operator/i18n/catalogues/`). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4d11fabf1ca7; not the translating i18n-translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4d11fabf1ca7: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (round-1 remediation values, uncommitted working tree on HEAD 7c0fc7bf2; English drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from translator session c0c9e9803405 transcript: `Model: anthropic/claude-opus-5-5`).
Prior audit: vi mobile.setup a1 (passed, source 6883f92e…, i18n/audits/evidence/vi/mobile.setup/a1/record.json). This round judges the 12 changed/new keys in scope (8 new) and re-reads unchanged keys for consistency only.

**Verdict: passed** — 0 unresolved blocker/major; 1 finding(s) total (VI-01 nit).

Source en sha256 0d245737d539f4d2fe54cf3a2c2174b0aa64820d2eb387e94ea16f65d1b487d7; context sha256 b1c0c11a269077507f7d22256f3874d7f5fd1a363f5c13f412014de70c7f038f; target sha256 2f449bf4a9beb90a5b4b36ff4cf49f38bd358b20858d9d23c5b6646b0d2cc2df; keys 94 (en 94).

Read: i18n/glossary.md (no phone-access/tunnel/credit row; §1 keeps Radient, Local Operator, URLs, commands verbatim), i18n/style-guides/vi.md, en + context sidecar for every in-scope key, HEAD values and prior a-round reports, and every in-scope value in full. Mechanical: check.py --catalogues-only 0 problems; key parity, placeholder parity, verbatim-token counts (/mobile enable, Radient, Local Operator, https://, backticks), NFC — all OK across the whole namespace (audit.log). No ICU plural/select in scope.

Design intent checked on every in-scope key: 'phone access' wording (no 'tunnel'/'remote access' in these sentences — forbidden-term scan clean, audit.log); {url} byte-identical; no command named in phone/desktop/card copy; nothing-is-charged / setup-never-purchases / owner-choice-not-overridden guarantees survive.

## Per-key judgement

| key | value | judgement |
|---|---|---|
| mobile.setup.disabled | Truy cập điện thoại của máy tính này đang bị tắt trong bảng điều khiển Radient. Khi được bật lại, truy cập sẽ tự kết nối lại; thiết lập sẽ không ghi đè lựa chọn đó. | OK: console switch-off, auto-reconnect, setup will not override; URL correctly dropped with EN. VI-01 (nit). |
| mobile.setup.billing_lapsed | Số dư Radient của bạn đã hết, nên truy cập điện thoại đang tạm dừng. Thiết lập không bao giờ tự mua tín dụng; truy cập điện thoại sẽ tự kết nối lại khi số dư được khôi phục. | OK: 'không bao giờ tự mua tín dụng' keeps the no-purchase guarantee; auto-reconnect kept; 'bạn' acceptable on a blocker per guide. |
| mobile.setup.reconnect | Radient đã tạm dừng truy cập điện thoại của máy tính này, dù việc thanh toán vẫn ổn. Kết nối lại sẽ khôi phục truy cập; không mất phí. | OK: 'Kết nối lại sẽ khôi phục truy cập; không mất phí' — nothing-is-charged survives. |
| mobile.setup.network | Máy tính này không kết nối được tới Radient (có thể mạng đã thay đổi hoặc bị mất). Truy cập điện thoại sẽ tự kết nối lại khi có mạng trở lại. Bạn không cần làm gì. | OK: no action asked; 'Bạn không cần làm gì.' matches the EN split. |
| mobile.setup.title_reconnect | Truy cập điện thoại bị Radient tạm dừng | OK. |
| mobile.setup.title_billing_lapsed | Truy cập điện thoại cần thêm tín dụng Radient | OK; 'tín dụng Radient' consistent with corpus. |
| mobile.setup.title_disabled | Truy cập điện thoại bị tắt trong bảng điều khiển Radient | OK. |
| mobile.setup.title_network | Truy cập điện thoại đang chờ mạng | OK. |
| mobile.setup.title_radient_unavailable | Truy cập điện thoại đang chờ Radient | OK. |
| mobile.setup.title_reconnecting | Truy cập điện thoại đang kết nối lại | OK. |
| mobile.setup.action_add_credit | Nạp tín dụng trong bảng điều khiển Radient: {url} | OK; {url} verbatim, no command. |
| mobile.setup.action_open_console | Bật lại truy cập điện thoại trong bảng điều khiển Radient: {url} | OK; full 'truy cập điện thoại'; {url} verbatim. |

## Findings

| ID | severity | key | issue | suggestion |
|---|---|---|---|---|
| VI-01 | nit | mobile.setup.disabled | Same pattern as tunnel.status a4 VI-03: 'Khi được bật lại, truy cập sẽ tự kết nối lại' — bare 'truy cập' as subject; no width budget. | 'Khi được bật lại, truy cập điện thoại sẽ tự kết nối lại; thiết lập sẽ không ghi đè lựa chọn đó.' |

Minor/nit findings do not block; they are offered to the translator for an optional follow-up. The auditor did not edit any value.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Card widths measured with rich.cells.cell_len on the composed line ('! ' prefix; card_paused after '! /mobile enable — '), not on a rendered frame.

## Ledger note

history-sha before append: 0a4df2d794b9e584f277022511e024b809201c2a56341a87a201340a8aa5c693. Row appended with `ledger.py append --dry-run` then `--apply` (dryrun.log).
