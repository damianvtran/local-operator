# ru/mobile.doctor — independent i18n audit a1

**Verdict: PASSED.** blocker 0 · major 0 · minor 0 · nit 0; unresolved blocker/major: 0.

- Scope: complete NEW namespace, all 23 keys judged semantically.
- Revision: `a3722be7dadacaf140a2587a0100bd3b26c2ab50` (feat/mobile-doctor, worktree mobile-doctor, node lop-remote-node-2).
- English sha256 `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`; context `c6baa99bc539ef2d2e03bf7b6b11e83edac37139eede8db82034f5fad98d4614`; target `edc8a2061e4496bf731faa0106c044fb2bd2e8faa33b6356039b5551120a8e4a`.
- Translator: i18n-translator subagent, session 398cdb26808f (English drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14.
- Auditor: i18n-auditor, independent `task` subagent (C4 i18n-auditor es fr ru), session 5e5b8d070b27; did not translate or edit any value it judged; anthropic/claude-opus-5-5 (served model reported by the session; no provider fallback observed). Same model family as the translator (disclosed).

## Findings

None.

## Translator questions

- Q2 (ru tunnel.duplicates colon pattern): PASS — «туннелей для этого компьютера: {count}» is the established label-colon pattern that avoids plural agreement (same as files.ok «открыто файлов: {open} из {limit}»); natural for count ≥ 2.
- Q5 (ru/es daemon.state_not_running fragment): PASS — «не запущен» (masculine) reads naturally after «auth gate:» (шлюз/демон are masculine).
- Q6 («asleep» renderings): «в спящем режиме» idiomatic; not verified against the phone app's own label.

## Per-key judgement

| key | English | ru | judgement |
|---|---|---|---|
| mobile.doctor.versions.ok | Local Operator {lop} on Python {python} ({platform}). | Local Operator {lop} на Python {python} ({platform}). | ok |
| mobile.doctor.daemon.not_installed | The phone service is not set up on this computer, so it is not running. | Служба доступа с телефона не настроена на этом компьютере, поэтому она не запущена. | ok |
| mobile.doctor.daemon.not_running | The phone service is set up but not running; the phone app shows this computer as asleep until it starts. | Служба доступа с телефона настроена, но не запущена; пока она не запустится, мобильное приложение показывает этот компьютер в спящем режиме. | ok |
| mobile.doctor.daemon.ok | The phone service is running and answered in {ms} ms. | Служба доступа с телефона запущена и ответила за {ms} мс. | ok |
| mobile.doctor.daemon.slow | The phone service is running but took {ms} ms to answer, which the phone app may show as not answering. | Служба доступа с телефона запущена, но ответила только через {ms} мс; в мобильном приложении этот компьютер может отображаться как не отвечающий. | ok |
| mobile.doctor.daemon.gate_open | The phone service answered without asking for its password: the phone portal is unprotected. | Служба доступа с телефона ответила, не запросив пароль: портал для телефона не защищён. | ok |
| mobile.doctor.daemon.state_not_running | not running | не запущен | ok |
| mobile.doctor.files.ok | The phone service has {open} of {limit} files open. | У службы доступа с телефона открыто файлов: {open} из {limit}. | ok |
| mobile.doctor.files.near | The phone service has {open} of {limit} files open, close to its limit, so new phone connections and database opens can fail. | У службы доступа с телефона открыто файлов: {open} из {limit}, это близко к лимиту, поэтому новые подключения с телефона и открытие базы данных могут завершаться ошибкой. | ok |
| mobile.doctor.files.unknown | The phone service's open files could not be counted on this computer. | Не удалось подсчитать открытые файлы службы доступа с телефона на этом компьютере. | ok |
| mobile.doctor.files.status_line | {open} open of {limit} | открыто {open} из {limit} | ok |
| mobile.doctor.logs.clean | No out-of-files or database-open errors in the last day of logs. | В журналах за последние сутки нет ошибок «слишком много открытых файлов» и ошибок открытия базы данных. | ok |
| mobile.doctor.logs.errors | The last day of logs shows out-of-files errors ({emfile}) and database-open errors ({sqlite}); the phone service may be running out of files. | В журналах за последние сутки есть ошибки «слишком много открытых файлов» ({emfile}) и ошибки открытия базы данных ({sqlite}); возможно, служба доступа с телефона упирается в лимит открытых файлов. | ok |
| mobile.doctor.tunnel.not_configured | Phone access through Radient is not set up on this computer. | Доступ с телефона через Radient на этом компьютере не настроен. | ok |
| mobile.doctor.tunnel.ok | The tunnel is set up and its connector is {state}. | Туннель настроен, его коннектор в состоянии {state}. | ok |
| mobile.doctor.tunnel.problem | The tunnel is set up but its connector is {state} ({reason}). | Туннель настроен, но его коннектор в состоянии {state} ({reason}). | ok |
| mobile.doctor.tunnel.single | This Radient account has one tunnel for this computer. | У этой учётной записи Radient один туннель для этого компьютера. | ok |
| mobile.doctor.tunnel.duplicates | This Radient account has {count} tunnels for this computer; the phone app may be connecting to an old one. | У этой учётной записи Radient туннелей для этого компьютера: {count}; возможно, мобильное приложение подключается к старому. | ok |
| mobile.doctor.tunnel.list_unchecked | The account's tunnel list could not be read, so duplicate tunnels were not checked. | Не удалось прочитать список туннелей учётной записи, поэтому дубликаты туннелей не проверялись. | ok |
| mobile.doctor.login.ok | Signed in to Radient. | Вход в Radient выполнен. | ok |
| mobile.doctor.login.missing | Not signed in to Radient. | Вход в Radient не выполнен. | ok |
| mobile.doctor.files.not_running | The phone service is not running, so it has no open files to count. | Служба доступа с телефона не запущена, поэтому открытых файлов для подсчёта нет. | ok |
| mobile.doctor.daemon.not_answering | The phone service is running but not answering; the phone app shows this computer as not answering. It may have run out of files. | Служба доступа с телефона запущена, но не отвечает; мобильное приложение показывает этот компьютер как не отвечающий. Возможно, она исчерпала лимит открытых файлов. | ok |

## Mechanical checks

- PASS — key parity (same set and order)
- PASS — ICU argument names identical
- PASS — select branch labels identical
- PASS — no command named that English does not name
- PASS — final period mirrors English
- PASS — rendered through Msg.envelope(locale=ru) without error (23/23 keys present in render)
- PASS — no silent English fallback in render

## Not verified

- Native-speaker review; terminal/desktop frames (text render via the runtime only, see rendered.txt); the phone app's own labels (separate repo).
