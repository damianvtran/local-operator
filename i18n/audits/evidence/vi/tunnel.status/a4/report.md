# i18n audit a4 — vi / tunnel.status (local-operator PR #3945, round-1 remediation)

Head: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes (worktree .worktrees/tunnel-remedies-auto-up; `git diff HEAD -- local_operator/i18n/catalogues/`). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4d11fabf1ca7; not the translating i18n-translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4d11fabf1ca7: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (round-1 remediation values, uncommitted working tree on HEAD 7c0fc7bf2; English drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from translator session c0c9e9803405 transcript: `Model: anthropic/claude-opus-5-5`).
Prior audit: vi tunnel.status a3 (passed, source b368acfb…, i18n/audits/evidence/vi/tunnel.status/a3/record.json). This round judges the 8 changed/new keys in scope (3 new) and re-reads unchanged keys for consistency only.

**Verdict: passed** — 0 unresolved blocker/major; 3 finding(s) total (VI-01 minor, VI-02 minor, VI-03 nit).

Source en sha256 e2bc8c63d3124c363252511306fd3951917ab9f93a2398c49cac8c79302aa8f9; context sha256 78580b806d2acc2c0c84bcf2655b62ea5ae847bf206c43134762e288bba8695d; target sha256 3d545d4f6534d8d5584f5d0eb7fd069a352272dcade75ecec9cb00cc400e1132; keys 24 (en 24).

Read: i18n/glossary.md (no phone-access/tunnel/credit row; §1 keeps Radient, Local Operator, URLs, commands verbatim), i18n/style-guides/vi.md, en + context sidecar for every in-scope key, HEAD values and prior a-round reports, and every in-scope value in full. Mechanical: check.py --catalogues-only 0 problems; key parity, placeholder parity, verbatim-token counts (/mobile enable, Radient, Local Operator, https://, backticks), NFC — all OK across the whole namespace (audit.log). No ICU plural/select in scope.

Design intent checked on every in-scope key: 'phone access' wording (no 'tunnel'/'remote access' in these sentences — forbidden-term scan clean, audit.log); {url} byte-identical; no command named in phone/desktop/card copy; nothing-is-charged / setup-never-purchases / owner-choice-not-overridden guarantees survive.

## Per-key judgement

| key | value | judgement |
|---|---|---|
| tunnel.status.tunnel_inactive.label | bị Radient tạm dừng — đang kết nối lại | OK — 'bị Radient tạm dừng — đang kết nối lại' mirrors 'paused by Radient — reconnecting'; lowercase fragment, spaced em dash per guide; no 'tunnel'. |
| tunnel.status.tunnel_inactive.detail | Radient đã tạm dừng truy cập điện thoại của máy tính này, dù việc thanh toán vẫn ổn. Hãy kết nối lại truy cập điện thoại trong Local Operator trên máy tính đang chạy nó; nó cũng tự thử lại liên tục. Nếu vẫn bị tạm dừng, hãy kiểm tra trong bảng điều khiển Radient tại {url}. | Meaning, 'billing is in order' ('dù việc thanh toán vẫn ổn'), Reconnect-in-Local-Operator remedy, auto-retry, console + {url} all present; no command. VI-02. |
| tunnel.status.tunnel_disabled.detail | Truy cập điện thoại của máy tính này đang bị tắt trong bảng điều khiển Radient. Hãy bật lại tại {url}; truy cập sẽ tự kết nối lại. | Meaning OK: console switch-off, re-enable at {url}, auto-reconnect; 'bảng điều khiển' matches corpus console. VI-03 (nit). |
| tunnel.status.billing_lapsed.detail | Số dư Radient của bạn đã hết, nên truy cập điện thoại đang tạm dừng. Hãy nạp tín dụng tại {url}; truy cập sẽ tự kết nối lại khi số dư được khôi phục. | OK: balance ran out → phone access paused; 'nạp tín dụng' matches corpus credit wording (mobile.setup.balance); auto-reconnect kept. VI-03 (nit). |
| tunnel.status.billing_inactive.detail | Truy cập điện thoại chưa được bật cho tài khoản Radient này. Hãy mở phần thiết lập truy cập điện thoại trong Local Operator trên máy tính đó và chấp nhận mức giá hằng tháng. | OK: 'chưa được bật cho tài khoản Radient này'; monthly price acceptance kept; 'máy tính đó' mirrors EN 'that computer'. |
| tunnel.status.card_paused | bị Radient tạm dừng | OK — 'bị Radient tạm dừng', identical to the label head; line 38 cells. |
| tunnel.status.card_add_credit | Nạp tín dụng Radient để khôi phục truy cập | Meaning OK, 44 cells. VI-01. |
| tunnel.status.card_open_console | Bật truy cập trong bảng điều khiển Radient | Meaning OK, 44 cells; bare 'truy cập' width-forced. VI-01. |

## Findings

| ID | severity | key | issue | suggestion |
|---|---|---|---|---|
| VI-01 | minor | tunnel.status.card_add_credit, tunnel.status.card_open_console | Bare 'truy cập' (access) where the design intent is 'truy cập điện thoại' everywhere; on a standalone terminal card 'Bật truy cập…' does not say access to what. open_console is width-forced: every 'điện thoại' variant measured 51-55 cells with the '! ' prefix (budget 50) unless 'console' stays English, which the vi guide does not keep — accepted there. add_credit has a fitting option. | card_add_credit: 'Nạp tín dụng để khôi phục truy cập điện thoại' (47 cells; trades 'Radient' for the feature name — translator's call). card_open_console: keep as is. |
| VI-02 | minor | tunnel.status.tunnel_inactive.detail | Unresolved from a3 VI-01 and sharper now: 'trên máy tính đang chạy nó; nó cũng tự thử lại liên tục' — two 'nó' with shifting referents (Local Operator / phone access / the computer), after the first sentence already named 'máy tính này'. | '… Hãy kết nối lại truy cập điện thoại trong Local Operator trên máy tính đó; hệ thống cũng tự thử lại liên tục. Nếu vẫn bị tạm dừng, …' |
| VI-03 | nit | tunnel.status.tunnel_disabled.detail, tunnel.status.billing_lapsed.detail | 'truy cập sẽ tự kết nối lại' — bare 'truy cập' as the subject of 'reconnect' reads stilted; no width budget forces the short form here, and the vi guide prefers dropping the subject. | tunnel_disabled: 'Hãy bật lại tại {url}; sau đó sẽ tự kết nối lại.' billing_lapsed: '…; truy cập điện thoại sẽ tự kết nối lại khi số dư được khôi phục.' |

Minor/nit findings do not block; they are offered to the translator for an optional follow-up. The auditor did not edit any value.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Card widths measured with rich.cells.cell_len on the composed line ('! ' prefix; card_paused after '! /mobile enable — '), not on a rendered frame.

## Ledger note

history-sha before append: 7c8e361d48ae58b161e33a2fad3520c4e70f1b607eb19ccfe2fa8aeae47f8f3c. Row appended with `ledger.py append --dry-run` then `--apply` (dryrun.log).
