# i18n audit a2 — zh-CN / mobile.setup (local-operator PR #3945, round-1 remediation)

Head: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes (worktree .worktrees/tunnel-remedies-auto-up; `git diff HEAD -- local_operator/i18n/catalogues/`). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4d11fabf1ca7; not the translating i18n-translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4d11fabf1ca7: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (round-1 remediation values, uncommitted working tree on HEAD 7c0fc7bf2; English drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from translator session c0c9e9803405 transcript: `Model: anthropic/claude-opus-5-5`).
Prior audit: zh-CN mobile.setup a1 (passed, source 6883f92e…, i18n/audits/evidence/zh-CN/mobile.setup/a1/record.json). This round judges the 12 changed/new keys in scope (8 new) and re-reads unchanged keys for consistency only.

**Verdict: passed** — 0 unresolved blocker/major; 0 finding(s) total (none).

Source en sha256 0d245737d539f4d2fe54cf3a2c2174b0aa64820d2eb387e94ea16f65d1b487d7; context sha256 b1c0c11a269077507f7d22256f3874d7f5fd1a363f5c13f412014de70c7f038f; target sha256 27cec626fce0f04be23e98035b3c2ab47ac2dd4b3803be217efa8b37f00dd913; keys 94 (en 94).

Read: i18n/glossary.md (no phone-access/tunnel/credit row; §1 keeps Radient, Local Operator, URLs, commands verbatim), i18n/style-guides/zh-CN.md, en + context sidecar for every in-scope key, HEAD values and prior a-round reports, and every in-scope value in full. Mechanical: check.py --catalogues-only 0 problems; key parity, placeholder parity, verbatim-token counts (/mobile enable, Radient, Local Operator, https://, backticks), NFC — all OK across the whole namespace (audit.log). No ICU plural/select in scope.

Design intent checked on every in-scope key: 'phone access' wording (no 'tunnel'/'remote access' in these sentences — forbidden-term scan clean, audit.log); {url} byte-identical; no command named in phone/desktop/card copy; nothing-is-charged / setup-never-purchases / owner-choice-not-overridden guarantees survive.

## Per-key judgement

| key | value | judgement |
|---|---|---|
| mobile.setup.disabled | 这台电脑的手机访问已在Radient控制台中关闭。重新开启后会自动重新连接；设置不会覆盖这一选择。 | OK: scope, auto-reconnect, '设置不会覆盖这一选择' kept. |
| mobile.setup.billing_lapsed | 你的Radient余额已用完，手机访问因此暂停。设置绝不会购买点数；余额恢复后，手机访问会自动重新连接。 | OK: '设置绝不会购买点数' keeps the guarantee; '点数' consistent with corpus/title. |
| mobile.setup.reconnect | Radient已暂停这台电脑的手机访问，但计费正常。重新连接即可恢复；不会产生任何费用。 | OK: '不会产生任何费用' kept. |
| mobile.setup.network | 这台电脑无法连接到Radient（网络可能已变更或断开）。连接恢复后，手机访问会自动重新连接。无需任何操作。 | OK: '无需任何操作。' asks nothing. |
| mobile.setup.title_reconnect | 手机访问已被Radient暂停 | OK. |
| mobile.setup.title_billing_lapsed | 手机访问需要更多Radient点数 | OK. |
| mobile.setup.title_disabled | 手机访问已在Radient控制台中关闭 | OK. |
| mobile.setup.title_network | 手机访问正在等待网络 | OK. |
| mobile.setup.title_radient_unavailable | 手机访问正在等待Radient | OK. |
| mobile.setup.title_reconnecting | 手机访问正在重新连接 | OK. |
| mobile.setup.action_add_credit | 在Radient控制台中充值：{url} | OK; URL last after '：', no trailing punctuation. |
| mobile.setup.action_open_console | 在Radient控制台中重新开启手机访问：{url} | OK. |

## Findings

None. The auditor did not edit any value.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Card widths measured with rich.cells.cell_len on the composed line ('! ' prefix; card_paused after '! /mobile enable — '), not on a rendered frame.

## Ledger note

history-sha before append: 76e3d964fc0bb846e29b7de88c30f332cfd0867489883af43a918bbb4dc89061. Row appended with `ledger.py append --dry-run` then `--apply` (dryrun.log).
