# ru/cli.mobile_help — independent i18n audit a1

**Verdict: PASSED.** blocker 0 · major 0 · minor 0 · nit 2; unresolved blocker/major: 0.

- Scope: whole namespace (17 keys) at the new English source hash: the 2 NEW keys (cli.mobile_help.doctor, cli.mobile_help.doctor_json) judged in full; the 15 previously audited keys re-read for consistency with the new keys and with mobile.doctor, and any defect seen is recorded.
- Revision: `a3722be7dadacaf140a2587a0100bd3b26c2ab50` (feat/mobile-doctor, worktree mobile-doctor, node lop-remote-node-2).
- English sha256 `3e1aea77751a6a1286cd0bc525163ac6e90da2f81fd19fff1fc23ae247133267`; context `dab85ceb9e4acf145e434179d866848912a0831bbf827ea196f82c28f855a869`; target `24ade01727f5e5fab0aaaf22dcfa48e82e2e06d63d36b761e1e02b91283aa50c`.
- Translator: i18n-translator subagent, session 398cdb26808f (English drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14.
- Auditor: i18n-auditor, independent `task` subagent (C4 i18n-auditor es fr ru), session 5e5b8d070b27; did not translate or edit any value it judged; anthropic/claude-opus-5-5 (served model reported by the session; no provider fallback observed). Same model family as the translator (disclosed).

## Findings

### RU-H1 — nit
- Keys: cli.mobile_help.doctor
- Issue: «Диагностика доступа с телефона, только чтение (…)»: the appended «только чтение» reads like a list item; meaning (read-only, safe to share) survives.
- Suggested fix: Optional: «Диагностика доступа с телефона в режиме только для чтения (можно безопасно передать)».

### RU-H2 — nit
- Keys: cli.mobile_help.status
- Issue: Pre-existing value: «живые сессии» calques «live sessions».
- Suggested fix: Optional: «активные сессии».


## Translator questions

- Q2 (ru tunnel.duplicates colon pattern): PASS — «туннелей для этого компьютера: {count}» is the established label-colon pattern that avoids plural agreement (same as files.ok «открыто файлов: {open} из {limit}»); natural for count ≥ 2.
- Q5 (ru/es daemon.state_not_running fragment): PASS — «не запущен» (masculine) reads naturally after «auth gate:» (шлюз/демон are masculine).
- Q6 («asleep» renderings): «в спящем режиме» idiomatic; not verified against the phone app's own label.

## Per-key judgement

| key | English | ru | judgement |
|---|---|---|---|
| cli.mobile_help.mobile | Phone access: control this machine's lop sessions | Доступ с телефона: управление сессиями lop на этой машине | ok |
| cli.mobile_help.install | Install the daemon, password and LaunchAgent | Установить демон, пароль и LaunchAgent | ok |
| cli.mobile_help.status | Daemon health, gate and live sessions | Состояние демона, шлюз и живые сессии | RU-H2 |
| cli.mobile_help.daemon_action | {action, select, start{Start} stop{Stop} restart{Restart} other{{action}}} the daemon | {action, select, start{Запустить демон} stop{Остановить демон} restart{Перезапустить демон} other{{action} демон}} | ok |
| cli.mobile_help.logs | Tail the daemon's and the runtimes' logs | Показать конец журналов демона и рантаймов | ok |
| cli.mobile_help.logs_lines | Lines per file (up to two are read) | Строк на файл (читается до двух) | ok |
| cli.mobile_help.logs_follow | Follow by name, so a log created or rotated while you watch is seen (a descriptor-bound follow misses both) | Следить по имени, чтобы был виден журнал, созданный или ротированный во время наблюдения (слежение по дескриптору пропускает оба случая) | ok |
| cli.mobile_help.password | Show or rotate the portal password | Показать или сменить пароль портала | ok |
| cli.mobile_help.devices | List this computer's push devices, or revoke/unrevoke one | Список push-устройств этого компьютера либо отзыв/возврат одного из них | ok |
| cli.mobile_help.devices_command | list (default), revoke <device_id>, or unrevoke <device_id> | list (по умолчанию), revoke <device_id> или unrevoke <device_id> | ok |
| cli.mobile_help.devices_device_id | the device_id `lop mobile devices` lists; required by revoke/unrevoke | device_id, который показывает `lop mobile devices`; обязателен для revoke/unrevoke | ok |
| cli.mobile_help.devices_port | the mobile daemon's port (default: the daemon's own default) | порт мобильного демона (по умолчанию — собственный порт демона) | ok |
| cli.mobile_help.uninstall | Remove the LaunchAgent | Удалить LaunchAgent | ok |
| cli.mobile_help.uninstall_purge | Also delete the password | Также удалить пароль | ok |
| cli.mobile_help.serve | Run the daemon (foreground) | Запустить демон (на переднем плане) | ok |
| cli.mobile_help.doctor | Read-only diagnostics for phone access (safe to share) | Диагностика доступа с телефона, только чтение (можно безопасно передать) | RU-H1 |
| cli.mobile_help.doctor_json | Print the diagnostics as JSON | Вывести диагностику в формате JSON | ok |

## Mechanical checks

- PASS — key parity (same set and order)
- PASS — ICU argument names identical
- PASS — select branch labels identical
- PASS — no command named that English does not name
- PASS — final period mirrors English
- PASS — rendered through Msg.envelope(locale=ru) without error (17/17 keys present in render)
- PASS — no silent English fallback in render

## Not verified

- Native-speaker review; terminal/desktop frames (text render via the runtime only, see rendered.txt); the phone app's own labels (separate repo).
