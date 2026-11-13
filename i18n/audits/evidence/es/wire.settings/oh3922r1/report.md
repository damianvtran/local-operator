# i18n audit — es / wire.settings — oh3922r1

PR damianvtran/local-operator#3922, head e2ea7efff2f68a5059f23dea0cdff35e8f549326. Auditor: i18n-auditor (lopdev team), session d25b77f53a23, model anthropic/claude-opus-5-5. Independent of the translator jobs (did not translate; did not read translator notes — none were recorded).

Scope: the 2 keys added by c3cd67df/1c956878 (`network_credentials_offline_hold_s.label`/`.help`), judged against i18n/glossary.md, i18n/style-guides/es.md and the neighbouring `network_credentials_grant_ttl_s` copy. The other 865 keys are byte-identical to merge-base f1e8d6410770, whose last ledger row was passed on en ffb9cf93…; carried by that proof.

Verdict: **passed** — 0 unresolved BLOCKER/MAJOR.

## Findings
- **I-es-1 MAJOR** `wire.settings.network_credentials_offline_hold_s.label` — current: «Retención del inicio prestado, propietario ausente (s)». "inicio" alone is not Spanish for login (reads "[redacted] start"); breaks the neighbour term "inicio de sesión prestado" used by grant_ttl_s.label/help and by this key's own help. Suggested: «Inicio de sesión prestado: margen sin el propietario (s)». Status: fixed in e2ea7efff2f6 (translator job 17c3e25545b7); re-audited: passes
- Re-audit of e2ea7efff2f6: «Inicio de sesión prestado: margen sin el propietario (s)» keeps the neighbour term whole, «margen» renders the grace window naturally; 56 cells (existing es label envelope reaches 57, `tools_defer`). Passes.
