# Independent translation audit — ru / mobile.setup — a3

- PR: #3958 (branch fix/tunnel-remedy-no-commands, stacked on #3945), head 01113e5aa048884a507d9bb4868815f909f63a48; scope `git diff 374037856..01113e5aa -- local_operator/i18n/` (12 new `*_plain` keys; 0 existing keys changed).
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (values committed in 01113e5aa on fix/tunnel-remedy-no-commands; English source by the same coder); model not disclosed to the auditor (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session 9054cd229d3f; not the translator/coder of host session 25fe8dfbf610; edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 0 minor/nit finding(s), none gating.

## Method
Read side by side with en source + context sidecar (12 new keys, 0 changed); glossary agent/approvals/service and the namespace's phone-access, phone-address, sign-in, account and setup terms match #3945's shipped copy; register per style guide; no placeholders in source or target; no command, flag, slash or code-span text (setup._SHELL_COMMAND + slash/flag/backtick scan); NFC; no bidi controls; no maxCells on these keys — the TUI's 50-cell park card renders tunnel.status card_* keys, not these; the *_plain keys reach the TUI only through setup.text(surface='tui') in a wrapping system notice and the desktop's remedy.text. Runtime: setup.plain_remedy({'step': s}) equals the catalogue value for all 12 steps.

## Locale notes
Clean. вы register; 'доступ с телефона', 'учётная запись', 'одобрение' (glossary approvals), 'фоновая служба' consistent with titles and *_tui keys; ё used.

## Findings
- None.

## Keys read (12)
| key | en | ru |
|---|---|---|
| `mobile.setup.stopped_plain` | Phone access is turned off on this computer. Turn it back on whenever you want to reach this computer from your phone. | Доступ с телефона на этом компьютере отключён. Включите его снова, когда захотите подключиться к этому компьютеру с телефона. |
| `mobile.setup.owner_missing_plain` | The Radient account that set up phone access is no longer signed in on this computer. Sign in with that account to bring phone access back. | Учётная запись Radient, с которой настроен доступ с телефона, больше не авторизована на этом компьютере. Войдите в эту учётную запись, чтобы вернуть доступ с телефона. |
| `mobile.setup.login_plain` | Sign in to Radient with the account that owns phone access, then try again. | Войдите в Radient с учётной записью, которой принадлежит доступ с телефона, и повторите попытку. |
| `mobile.setup.choose_account_plain` | Choose the Radient account phone access should use. Setup never moves an existing phone address to a different account. | Выберите учётную запись Radient для доступа с телефона. Настройка никогда не переносит существующий адрес телефона в другую учётную запись. |
| `mobile.setup.account_plain` | Radient could not confirm the account and phone address. Check your internet connection and try again; nothing was changed. | Radient не смог подтвердить учётную запись и адрес телефона. Проверьте подключение к интернету и повторите попытку; ничего не изменено. |
| `mobile.setup.cloudflared_plain` | Phone access needs a helper app called cloudflared, which is not installed on this computer. Ask your agent to install it (it asks for your approval first), then try again. | Для доступа с телефона нужно вспомогательное приложение cloudflared, но на этом компьютере оно не установлено. Попросите агента установить его (сначала он запросит ваше одобрение), затем повторите попытку. |
| `mobile.setup.mobile_bundle_plain` | The phone interface is not prepared on this computer yet. Ask your agent to prepare it (it asks for your approval first), then try again. Nothing was changed. | Интерфейс телефона на этом компьютере ещё не подготовлен. Попросите агента подготовить его (сначала он запросит ваше одобрение), затем повторите попытку. Ничего не изменено. |
| `mobile.setup.supervisor_plain` | This computer cannot keep phone access running in the background yet. Ask your agent to set up a background service for it, then try again. Nothing was changed. | Этот компьютер пока не может поддерживать доступ с телефона в фоне. Попросите агента настроить для этого фоновую службу, затем повторите попытку. Ничего не изменено. |
| `mobile.setup.apply_plain` | Everything is ready. Turn on phone access to finish. | Всё готово. Включите доступ с телефона, чтобы завершить. |
| `mobile.setup.verification_plain` | The connection is not confirmed yet. Check again in a moment; completed steps stay saved. | Подключение пока не подтверждено. Проверьте ещё раз чуть позже; выполненные шаги остаются сохранёнными. |
| `mobile.setup.failed_plain` | Phone access setup did not finish. Try again; completed steps stay saved. | Настройка доступа с телефона не завершилась. Повторите попытку; выполненные шаги остаются сохранёнными. |
| `mobile.setup.complete_plain` | Open the address above on your phone and sign in with the same Radient account. | Откройте адрес выше на телефоне и войдите в ту же учётную запись Radient. |
