# Independent translation audit — ru / mobile.setup — a4

- PR: #3958 (branch fix/tunnel-remedy-no-commands), head 139bca9037fda9e936e77dd9f145c99ce1188315; scope `git diff 983511c54..139bca903 -- local_operator/i18n/` (3 new keys, 1 removed key, 0 existing keys changed).
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (values committed in 139bca903 on fix/tunnel-remedy-no-commands; English source by the same coder); model not disclosed to the auditor (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session b3a764eb5573; not the translator/coder (25fe8dfbf610); edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback observed during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 1 minor/nit finding(s), none gating (deferred to the PR thread).

## Method
Read side by side with the en source and context sidecar; compared terminology with the namespace's disabled, billing_lapsed, login_tui, *_plain keys and #3945's tunnel.status/action_* copy. Verified: no placeholders in source or target; NFC; no bidi controls; typographic apostrophes; fr U+202F before ; : ! ?; zh-CN no CJK/Latin space; the *_resume keys carry no automatic-reconnection wording and no command/flag/URL; operation_failed_tui keeps `/login radient` verbatim and no `lop login`; mobile.setup.mobile_bundle_tui absent from the catalogue; key parity with en; runtime `ms.<key>().envelope(locale=…)` equals the catalogue value. No maxCells on these keys (prose result line / remedy card that wraps).

## Locale notes
Accurate, вы register; 'доступ с телефона', 'консоль Radient', 'пополните баланс' (matches action_add_credit/balance), 'кредиты' (matches billing_lapsed). One minor glossary finding (RU4-m1).

## Findings
- RU4-m1 (minor) `mobile.setup.operation_failed_tui` — current: … Radient недоступен или нужно заново войти в аккаунт. — proposed: … Radient недоступен или нужно войти заново. — 'аккаунт' is the only use in ru mobile.setup, which renders account as 'учётная запись' throughout (login_plain, owner_missing_plain); the English names the sign-in, not the account, so dropping the noun removes the drift.

## Keys read (3)
| key | en | ru |
|---|---|---|
| `mobile.setup.operation_failed_tui` | Phone access could not be checked: Radient could not be reached or your sign-in needs renewing. Check your internet connection or sign in with /login radient, then try again. | Не удалось проверить доступ с телефона: Radient недоступен или нужно заново войти в аккаунт. Проверьте подключение к интернету или войдите через /login radient, затем повторите попытку. |
| `mobile.setup.disabled_resume` | Phone access for this computer is turned off in the Radient console, so it stays off. Turn it back on in the console, then turn phone access back on here. | Доступ с телефона для этого компьютера отключён в консоли Radient, поэтому он остаётся отключённым. Включите его в консоли, затем снова включите доступ с телефона здесь. |
| `mobile.setup.billing_lapsed_resume` | Your Radient balance has run out, so phone access stays turned off. Setup never purchases credit; add credit, then turn phone access back on here. | На вашем балансе Radient закончились средства, поэтому доступ с телефона остаётся отключённым. Настройка никогда не покупает кредиты; пополните баланс, затем снова включите доступ с телефона здесь. |
