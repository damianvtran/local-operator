# i18n audit — ur / wire.settings — oh3922r1

PR damianvtran/local-operator#3922, head e2ea7efff2f68a5059f23dea0cdff35e8f549326. Auditor: i18n-auditor (lopdev team), session d25b77f53a23, model anthropic/claude-opus-5-5. Independent of the translator jobs (did not translate; did not read translator notes — none were recorded).

Scope: the 2 keys added by c3cd67df/1c956878 (`network_credentials_offline_hold_s.label`/`.help`), judged against i18n/glossary.md, i18n/style-guides/ur.md and the neighbouring `network_credentials_grant_ttl_s` copy. The other 865 keys are byte-identical to merge-base f1e8d6410770, whose last ledger row was passed on en ffb9cf93…; carried by that proof.

Verdict: **passed** — 0 unresolved BLOCKER/MAJOR.

## Findings
- **I-ur-1 NIT** `wire.settings.network_credentials_offline_hold_s.help` — current: «اور کم والی غالب رہتی ہے۔». "کم والی" is colloquial for a settings help line. Suggested: «اور ان میں سے کم قدر لاگو ہوتی ہے۔». Status: open, non-gating (deferred to the PR thread).
