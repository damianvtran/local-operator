# i18n audit — vi / wire.settings — oh3922r1

PR damianvtran/local-operator#3922, head e2ea7efff2f68a5059f23dea0cdff35e8f549326. Auditor: i18n-auditor (lopdev team), session d25b77f53a23, model anthropic/claude-opus-5-5. Independent of the translator jobs (did not translate; did not read translator notes — none were recorded).

Scope: the 2 keys added by c3cd67df/1c956878 (`network_credentials_offline_hold_s.label`/`.help`), judged against i18n/glossary.md, i18n/style-guides/vi.md and the neighbouring `network_credentials_grant_ttl_s` copy. The other 865 keys are byte-identical to merge-base f1e8d6410770, whose last ledger row was passed on en ffb9cf93…; carried by that proof.

Verdict: **passed** — 0 unresolved BLOCKER/MAJOR.

## Findings
- **I-vi-1 NIT** `wire.settings.network_credentials_offline_hold_s.help` — current: «… đều áp dụng, giá trị ngắn hơn được dùng.». passive marker missing; "đều được áp dụng" reads more natural. Suggested: «… đều được áp dụng, giá trị ngắn hơn sẽ được dùng.». Status: open, non-gating (deferred to the PR thread).
