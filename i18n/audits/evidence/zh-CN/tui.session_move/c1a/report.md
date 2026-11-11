# zh-CN · `tui.session_move` — independent audit (audit id `c1a`)

- Namespace: `tui.session_move` — 7 `/move … --to` refusal messages (error tier)
- Translation: `openai/gpt-6.1-sol`; auditor `i18n-auditor` (session `0ebc670d4b12`), served model `deepseek/deepseek-flash` — independent of the translator
- Source: `local_operator/i18n/catalogues/en/tui.session_move.json` (sha256 `f4dd682b…26a913e`), sidecar `…tui.session_move.context.json` (`09ed30a0…cbee573`)
- Use site read: `local_operator/tui/session_move.py` (`parse_move_to`, `run_session_move`) — confirms `ambiguous` covers both the multi-word shape and a path combined with `--to`, and that `{flag}` arrives as a `repr()` (quoted)
- Guides: `i18n/style-guides/zh-CN.md`, `i18n/glossary.md`
- **Verdict: clean-with-findings — 0 BLOCKER, 0 MAJOR, 2 MINOR, 0 NIT** → ledger status `passed`

Mechanical: key set/order 7/7, placeholder sets identical (`{flag}`, `{reason}`, `{session}`+`{to}`, `{code}`),
backticked runnable commands byte-identical in both keys that carry one, ICU parses and renders through the repo
runtime with argument tuples identical to en, `check.py --catalogues-only` clean, nothing identical to English.
Full transcript: `audit.log`.

## MINOR

**M1 — `ambiguous`, `deadline_exceeded` — 迁移 vs the shipped verb 移动.**
Current: `切换工作目录用/move <路径>，迁移会话用/move <会话> --to <设备>。` / `迁移未在限定时间内完成；再次运行…`
Problem: the shipped catalogue names this exact action 移动/移至 — `cli.sessions_help.move` "将对话移至另一设备",
`move_session` "要移动的对话ID", `move_queue` "移动请求…", `move_cancel_queued` "排队移动",
`move_engage_on_arrival` "…不影响移动…移动结果" — 5 shipped strings — plus the shipped error-tier string for a
failed move, `wire.errors.desktop.move_failed` "无法移动到{requested}：{error}". The only shipped 迁移 is data
migration (`cli.qwencloud_help.migrate` "将明文ticket迁移至加密密钥存储"). One user action, two names, one surface
apart — and the surface that most closely matches (a move that did not work) says 移动到.
Suggested: `…移动会话用/move <会话> --to <…>。` and `移动未在限定时间内完成；…`.

**M2 — `ambiguous`, `to_needs_device` — `<设备>` for `<peer>`.**
Current: `… --to <设备>` , `/move --to <设备|local>`
Problem: shipped *peer* is 对等设备 (`cli.sessions_help.peer` "列出一个对等设备持有的会话"; `move_to`
"对等设备（ID或名称），或用`local`…"), while bare 设备 is *device* (`tui.chrome.remote_park` `{device}` → 设备).
The en hint slot says `peer` — a peer's id/name — not any device; collapsing the two loses the distinction the
CLI help keeps. `local` correctly stays byte-identical.
Suggested: `… --to <对等设备>` and `/move --to <对等设备|local>`.

## Checked, not defects

`deadline_exceeded` says 未在限定时间内完成 (did **not finish in time**) and never 失败 — the UNKNOWN-outcome
requirement holds; `再次运行…可查看当前进度` reproduces "again reports where it got to".
`cannot_wait` keeps the whole backticked command, including `<id>/<to>/<seconds>`, verbatim; 终端 for "from a
shell" is an acceptable product-voice gloss.
`unsupported_flag` renders `不支持'--force'` — the quotes come from `repr()` in the use site, half-width inside a
Latin fragment, correct. `cli_unavailable` `（{reason}）` keeps the OS text verbatim in full-width parens.
No 。 on the terse refusals, 。 kept on `ambiguous` (a two-clause sentence, as in en); `；`/`：` full-width.
