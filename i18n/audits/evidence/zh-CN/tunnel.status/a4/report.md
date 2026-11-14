# i18n audit a4 — zh-CN / tunnel.status (local-operator PR #3945, round-1 remediation)

Head: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes (worktree .worktrees/tunnel-remedies-auto-up; `git diff HEAD -- local_operator/i18n/catalogues/`). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 4d11fabf1ca7; not the translating i18n-translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 4d11fabf1ca7: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (round-1 remediation values, uncommitted working tree on HEAD 7c0fc7bf2; English drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from translator session c0c9e9803405 transcript: `Model: anthropic/claude-opus-5-5`).
Prior audit: zh-CN tunnel.status a3 (passed, source b368acfb…, i18n/audits/evidence/zh-CN/tunnel.status/a3/record.json). This round judges the 8 changed/new keys in scope (3 new) and re-reads unchanged keys for consistency only.

**Verdict: passed** — 0 unresolved blocker/major; 2 finding(s) total (ZH-01 minor, ZH-02 nit).

Source en sha256 e2bc8c63d3124c363252511306fd3951917ab9f93a2398c49cac8c79302aa8f9; context sha256 78580b806d2acc2c0c84bcf2655b62ea5ae847bf206c43134762e288bba8695d; target sha256 01c73d767048955095b04b2874cd6473522375d9b74751726da1d1f6ee471b37; keys 24 (en 24).

Read: i18n/glossary.md (no phone-access/tunnel/credit row; §1 keeps Radient, Local Operator, URLs, commands verbatim), i18n/style-guides/zh-CN.md, en + context sidecar for every in-scope key, HEAD values and prior a-round reports, and every in-scope value in full. Mechanical: check.py --catalogues-only 0 problems; key parity, placeholder parity, verbatim-token counts (/mobile enable, Radient, Local Operator, https://, backticks), NFC — all OK across the whole namespace (audit.log). No ICU plural/select in scope.

Design intent checked on every in-scope key: 'phone access' wording (no 'tunnel'/'remote access' in these sentences — forbidden-term scan clean, audit.log); {url} byte-identical; no command named in phone/desktop/card copy; nothing-is-charged / setup-never-purchases / owner-choice-not-overridden guarantees survive.

## Per-key judgement

| key | value | judgement |
|---|---|---|
| tunnel.status.tunnel_inactive.label | 已被Radient暂停——正在重新连接 | OK — '已被Radient暂停——正在重新连接'; CJK dash —— correct in Chinese text. |
| tunnel.status.tunnel_inactive.detail | Radient已暂停这台电脑的手机访问，但计费正常。请在运行它的电脑上，于Local Operator中重新连接手机访问；它也会持续自动重试。如果仍处于暂停状态，请在Radient控制台查看：{url}。 | Meaning, billing-in-order, Reconnect remedy, auto-retry, console + {url} present; no command. ZH-01, ZH-02. |
| tunnel.status.tunnel_disabled.detail | 这台电脑的手机访问已在Radient控制台中关闭。请在{url}重新开启；之后会自动重新连接。 | Meaning OK; 'this computer' scope and auto-reconnect kept. ZH-01. |
| tunnel.status.billing_lapsed.detail | 你的Radient余额已用完，手机访问因此暂停。请在{url}充值；余额恢复后会自动重新连接。 | Meaning OK; '充值' natural for add credit. ZH-01. |
| tunnel.status.billing_inactive.detail | 此Radient账户尚未开通手机访问。请在那台电脑上打开Local Operator的手机访问设置，并接受月费。 | OK — '尚未开通手机访问' is the idiomatic service-activation verb; monthly price kept. |
| tunnel.status.card_paused | 已被Radient暂停 | OK — '已被Radient暂停'; line 34 cells. |
| tunnel.status.card_add_credit | 为Radient充值以恢复手机访问 | OK — '为Radient充值以恢复手机访问', full feature name, 29 cells. |
| tunnel.status.card_open_console | 在Radient控制台中开启手机访问 | OK — full feature name, 31 cells. |

## Findings

| ID | severity | key | issue | suggestion |
|---|---|---|---|---|
| ZH-01 | minor | tunnel.status.tunnel_disabled.detail, tunnel.status.billing_lapsed.detail, tunnel.status.tunnel_inactive.detail | {url} is glued to following non-ASCII text: '请在{url}重新开启', '请在{url}充值', '…查看：{url}。'. Whitespace-delimited autolinkers (GFM extended autolink, many terminal URL detectors) can absorb the Han characters or full-width 。 into the link and break the only actionable target; EN ends the URL with ASCII punctuation linkifiers trim. Risk reasoned, not reproduced on a rendered surface. The new action_* lines already end with '：{url}'. | tunnel_disabled: '这台电脑的手机访问已在Radient控制台中关闭。重新开启后会自动重新连接。在此重新开启：{url}'; billing_lapsed: '你的Radient余额已用完，手机访问因此暂停；余额恢复后会自动重新连接。在此充值：{url}'; tunnel_inactive: drop the final 。 after {url}. |
| ZH-02 | nit | tunnel.status.tunnel_inactive.detail | Unresolved from a3 ZH-02: '请在运行它的电脑上，于Local Operator中重新连接手机访问' stacks 在/于 and reads written-formal. | '请在运行它的电脑上打开Local Operator，重新连接手机访问；…' |

Minor/nit findings do not block; they are offered to the translator for an optional follow-up. The auditor did not edit any value.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Card widths measured with rich.cells.cell_len on the composed line ('! ' prefix; card_paused after '! /mobile enable — '), not on a rendered frame.

## Ledger note

history-sha before append: 94d94ec9780d1b2e797784fd943e51cdb39f675e517ae35ae8be90029b63b1f8. Row appended with `ledger.py append --dry-run` then `--apply` (dryrun.log).
