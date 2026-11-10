# Audit ru / cli.permissions — pb2 (round 2)

Auditor: i18n-auditor (lopdev team), session a6f580403f3a, model anthropic/claude-opus-5-5 (served model per session system prompt; no fallback observed). Independent of the round-2 ru translator (i18n-translator, commit 788cc399410) and of the round-1 auditor (session 741233d88cd4). Worktree @ 788cc399410196b080cfc9f7e7bd558ccb713140. Date 2026-11-09. PR #3691.

Verdict: **changes_requested** — 0 BLOCKER, 2 MAJOR, 1 MINOR, 1 NIT. 38 keys audited, including the 10 new and 5 changed English keys.

## Mechanical (audit.log)

- `check.py --catalogues-only`: 0 problems, exit 0.
- My own parity script (`parity.py.txt`): 38/38 keys, en order, placeholder sets, backtick spans, `[y/N] ` trailing spaces, newline count, the two `- ` bullet lines in setup.explain, and verbatim tokens (`/update`, `guide://permissions`, `HOME`, `LOCAL_OPERATOR_*`, `~/.local-operator`, codesign, Developer ID [Application], Local Operator [Runtime], Python, macOS) are all identical. No plurals and no `maxCells` hints in this namespace.

## Round-1 findings (pa1): all applied

R1 «рантайм» used throughout (13+ keys) · R2 «сессии» (setup.done) · R3 «Подпись ключом команды» (optin.*, refused.isolated) · R4 «удостоверение {identity}» · R5 «разрешить ли codesign доступ к этому ключу» · R6 «Запустите команду из своей оболочки входа» · R7 «по-прежнему используют строку» · R8 help.permissions reworded as suggested · R9 setup.no_app reworded as suggested.

Correction to pa1: pa1 states that «Всегда разрешать» matches Russian macOS. The installed Apple table says otherwise (see P2).

## macOS labels checked against Apple's installed ru tables (macOS 27.2, 26B123)

| Label | Apple table / key | Apple ru | Catalogue | OK |
|---|---|---|---|---|
| Always Allow | SecurityAgent.bundle ConfirmAccess.loctable `100267.title` | Разрешать всегда | «Всегда разрешать» ×3 | **no** |
| Allow | ConfirmAccess.loctable `100268.title` | Разрешить | «Разрешить» | yes |
| Access Control (tab) | Keychain Access.app KeyEditor.loctable `302.label` (also PasswordEditor `257.label`, NoteEditor `295.label`, CertificatePrefEditor `257.label`) | Доступ | «Контроль доступа» | **no** |
| Keychain Access | Keychain Access.app InfoPlist.loctable `CFBundleDisplayName` | Связка ключей | «Связка ключей» | yes |
| private key | Keychain Access.app Localizable.loctable `private key` | Закрытый ключ | закрытый ключ | yes |
| System Settings | System Settings.app InfoPlist.loctable `CFBundleDisplayName` | Системные настройки | «Системные настройки» | yes |
| Privacy & Security | SecurityPrivacyExtension.appex InfoPlist.loctable `CFBundleDisplayName` | Конфиденциальность и безопасность | same | yes |
| Screen & System Audio Recording | SecurityPrivacyExtension.appex Localizable.loctable `SCREENANDAUDIOCAPTURE` / `SCREEN_CAPTURE_HEADER` | Запись экрана и системного звука | same | yes |
| Screen Recording | SecurityPrivacyExtension Localizable.loctable `SCREEN_CAPTURE` | Запись экрана | запись экрана | yes |
| Files & Folders | SecurityPrivacyExtension Localizable.loctable `FILE_ACCESS_COMBINED` | Файлы и папки | «Файлы и папки» | yes |
| Accessibility | System Settings.app Localizable.loctable `Accessibility` | Универсальный доступ | «Универсальный доступ» | yes |

Keychain Access lives at /System/Library/CoreServices/Applications/Keychain Access.app on this Mac; the /System/Applications/Utilities path named in the brief does not exist here.

## P1 — MAJOR — cli.permissions.setup.explain
- Current: …и удалите codesign на вкладке «Контроль доступа».
- Problem: This is the undo instruction for the security-sensitive "Always Allow" grant. Russian Keychain Access names the tab «Доступ» (KeyEditor.loctable `302.label`, next to «Атрибуты»). There is no «Контроль доступа» tab, so the user is sent to a control that does not exist. The context note requires the exact localized macOS label. The translator flagged this label as unverified.
- Suggested: …выберите закрытый ключ этого удостоверения и удалите codesign на вкладке «Доступ».

## P2 — MAJOR — cli.permissions.setup.explain, cli.permissions.setup.always_allow, cli.permissions.setup.already_signed
- Current: «Всегда разрешать» (3 occurrences)
- Problem: The codesign keychain prompt is SecurityAgent's ConfirmAccess dialog, and in Russian its button reads «Разрешать всегда» (`100267.title`; `100268.title` «Разрешить», `100266.title` «Отказать»). The copy tells users which button to click, and the y/N prompt asks whether they clicked it, so the label must match the dialog exactly (context note). The current text reverses Apple's word order.
- Suggested: setup.explain bullet: «- «Разрешать всегда» — вопросов больше не будет, …»; always_allow: «Вы выбрали «Разрешать всегда» (или диалог macOS не появился)? [y/N] »; already_signed: «…Ваш прежний ответ «Разрешать всегда» сохранён.»

## P3 — MINOR — cli.permissions.status.scope
- Current: Проверяется только запись экрана. «Файлы и папки», «Универсальный доступ» и другие службы конфиденциальности используют ту же строку.
- Problem: The sentence lists three macOS privacy services side by side but quotes only two of them, so «запись экрана» reads as a generic phrase and not the pane name.
- Suggested: Проверяется только разрешение «Запись экрана». «Файлы и папки», «Универсальный доступ» и другие службы конфиденциальности используют ту же строку.

## P4 — NIT — cli.permissions.setup.explain
- Current: Этот шаг подписывает рантайм ключом «{name}»…
- Problem: {name} is the signing identity's name, and the same string later calls it «удостоверение» («закрытый ключ этого удостоверения»).
- Suggested: Этот шаг подписывает рантайм удостоверением «{name}»…

## Checked and accepted (no finding)
Every other key reads as accurate, natural, impersonal copy that uses lowercase «вы», «ёлочки» and glossary terms (рантайм, сессия, команда, связка ключей). The safety disclosure in the "Always Allow" bullet is faithful: any program running as your macOS user can sign through codesign with this Developer ID key without being asked. Nothing is weakened or omitted. In status.reason.*, each phrase fits inside the parentheses of optin.blocked and refused.isolated. setup.done_allow_once keeps `/update` and both `lop` commands verbatim.

## Source observation (English; not a ru finding, does not gate this locale)
permissions_cli.py:298-303 prints setup.already_signed ("Your earlier "Always Allow" answer is kept.") whenever the image was already signed, including when `kept` is False (no earlier confirmation, or a different key). In that case it is followed by setup.done_allow_once, so the two lines contradict each other. Raised for the manager; the English owner should decide.
