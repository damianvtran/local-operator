# i18n audit — ru / wire.settings — oh3922r1

PR damianvtran/local-operator#3922, head e2ea7efff2f68a5059f23dea0cdff35e8f549326. Auditor: i18n-auditor (lopdev team), session d25b77f53a23, model anthropic/claude-opus-5-5. Independent of the translator jobs (did not translate; did not read translator notes — none were recorded).

Scope: the 2 keys added by c3cd67df/1c956878 (`network_credentials_offline_hold_s.label`/`.help`), judged against i18n/glossary.md, i18n/style-guides/ru.md and the neighbouring `network_credentials_grant_ttl_s` copy. The other 865 keys are byte-identical to merge-base f1e8d6410770, whose last ledger row was passed on en ffb9cf93…; carried by that proof.

Verdict: **passed** — 0 unresolved BLOCKER/MAJOR.

## Findings
- **I-ru-1 MINOR** `wire.settings.network_credentials_offline_hold_s.label` — current: «Удержание логина в отсутствие владельца (сек)». drops "borrowed" (заимствованного), which the neighbour label and the help both carry; meaning still recoverable from the help. Suggested: «Удержание заимствованного логина без владельца (сек)». Status: open, non-gating (deferred to the PR thread).
- **I-ru-2 NIT** `wire.settings.network_credentials_offline_hold_s.help` — current: «… и выигрывает меньшее.». "выигрывает" is a slight calque of "wins"; the catalogue elsewhere uses "Срабатывает меньшее из …". Suggested: «… и действует меньшее из них.». Status: open, non-gating (deferred to the PR thread).
