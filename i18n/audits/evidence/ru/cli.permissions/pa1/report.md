# Audit ru / cli.permissions — pa1

Auditor: i18n-auditor (lopdev team), session 741233d88cd4, model anthropic/claude-opus-5-5. Independent of translator job ad99821ab9e1. Worktree @ 50badda034d755ebb049c4f542f3073439bf823e. Date 2026-11-09.

Verdict: **changes_requested** — 2 MAJOR, 5 MINOR, 2 NIT, 0 BLOCKER.

Mechanical: check.py --catalogues-only 0 problems; my own parity check: 28/28 keys, en order, placeholder sets, backtick spans and trailing spaces of [y/N] prompts identical (see audit.log). No plurals in namespace. No maxCells hints; longest value setup.explain 311 cells (wrapping prose, no fixed slot).

Accuracy confirmed: keychain button «Всегда разрешать», «Связка ключей», «Системные настройки» > «Конфиденциальность и безопасность» > «Запись экрана» match Russian macOS; [y/N] kept Latin, which matches _ask accepting only y/yes; cancelled/failed state nothing-changed correctly; register lowercase вы, impersonal; «ёлочки» quotes.

## R1 — MAJOR — 13 keys: help.permissions, help.status, help.setup, status.identity.{team,adhoc,legacy,none}, status.screen.{granted,denied,unknown}, setup.no_identity, setup.explain, setup.failed, setup.done
- Current: «среда/среды выполнения» (e.g. «Идентификатор сред выполнения: {identifier}…»)
- Problem: Glossary violation: ru style guide (Wave-2 recorded calls) fixes runtime → «рантайм»; 16 other ru catalogues use «рантайм», only this file uses «среда выполнения» — same concept named two ways in one CLI.
- Suggested: Use «рантайм» throughout: «Идентификатор рантаймов: {identifier}, …»; «Запись экрана (проверка рантайма): разрешена»; «Подписать рантаймы вашим Developer ID, …»; «…при следующем запуске рантайма»; «Новые рантаймы используют {identifier}…».

## R2 — MAJOR — cli.permissions.setup.done
- Current: Запущенные сеансы сохраняют текущий образ до перезапуска.
- Problem: Glossary/style guide: session → «сессия». «сеанс» is a different term for the same lop concept.
- Suggested: Готово. Новые рантаймы используют {identifier} и общую строку с настольным приложением. Запущенные сессии сохраняют текущий образ до перезапуска.

## R3 — MINOR — cli.permissions.status.optin.on / optin.off / optin.blocked / setup.refused.isolated
- Current: Подпись команды: …
- Problem: «Подпись команды» in a CLI reads first as 'command signature' (команда = command), especially next to «Запустите `lop permissions setup`». Term команда=team is correct; the phrase is ambiguous.
- Suggested: «Подпись ключом команды: включена (…)» / «Подпись ключом команды в этом процессе запрещена ({reason})».

## R4 — MINOR — cli.permissions.status.optin.on
- Current: (идентификатор {identity}, команда {team})
- Problem: {identity} is the signing identity's SHA-1 (permissions_cli.py:117, _setup records sha1), not the runtime identifier; setup.no_identity already calls it «удостоверение подписи». Using «идентификатор» conflates it with the bundle id.
- Suggested: Подпись ключом команды: включена (удостоверение {identity}, команда {team})

## R5 — MINOR — cli.permissions.setup.explain
- Current: macOS может один раз спросить, можно ли codesign использовать этот ключ.
- Problem: Ungrammatical/ambiguous: «можно ли codesign использовать» lacks the dative and parses as 'can one use codesign this key'. Safety copy about a keychain dialog should be unambiguous.
- Suggested: macOS может один раз спросить, разрешить ли codesign доступ к этому ключу.

## R6 — MINOR — cli.permissions.setup.refused.isolated
- Current: …Запустите её из своей обычной оболочки входа.
- Problem: «её» grammatically refers back to «Подпись» (feminine), not the command; «обычной» is an addition.
- Suggested: Подпись ключом команды в этом процессе запрещена ({reason}). Запустите команду из своей оболочки входа.

## R7 — MINOR — cli.permissions.setup.no_identity / setup.failed
- Current: Среды выполнения сохраняют строку «Local Operator Runtime».
- Problem: «сохраняют строку» can read as 'save the row'; English means they keep using it.
- Suggested: Рантаймы по-прежнему используют строку «Local Operator Runtime».

## R8 — NIT — cli.permissions.help.permissions
- Current: …идентификатор конфиденциальности macOS…
- Problem: Calque of 'privacy identity'; acceptable but stiff.
- Suggested: Проверить или настроить, под каким идентификатором macOS видит рантаймы Local Operator в настройках конфиденциальности

## R9 — NIT — cli.permissions.setup.no_app
- Current: …поэтому общих разрешений приложения нет.
- Problem: Slightly shifts 'no app permission to share' to 'there are no shared permissions'.
- Suggested: …поэтому нет разрешений приложения, которыми можно было бы поделиться.
