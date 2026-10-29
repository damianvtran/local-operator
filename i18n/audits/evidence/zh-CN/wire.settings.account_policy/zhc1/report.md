# zh-CN / wire.settings.account_policy — independent translation audit (round 1)

- Locale: zh-CN · namespace: `wire.settings.account_policy` · keys: 14
- Translator: translator (lopdev team), session 809b6f2773e0, started on anthropic/claude-sonnet-5-5; provider-quota fallbacks through openai/gpt-6-astra and kimi/k3.1-pro before the run completed on radient/auto
- Auditor: translation-auditor (lopdev team), session e139d7f8f30c, started on anthropic/claude-sonnet-5-5; quota fallbacks to openai/gpt-6-astra, then kimi/k3.1-pro, then radient/auto, which produced the findings
- Independence: the auditor is not the translator: translator session 809b6f2773e0, auditor session e139d7f8f30c, neither edited the other's files and the auditor edited nothing
- English source sha256: `1ad13f8198e28cb19fe0ecb2a6ef9b2a6990bc0d047566b71f5dd5322f594b25`
- Context sidecar sha256: `ccee657198d84a0bef6d20ae0343c9a864567e5a622a1b9736b1974ca296a254`
- zh-CN target sha256 at audit time: `51c800cd4fe8bbcfe857b4749f08f1bd836e536fb8b18f88de3426805c769b93`
- Verdict: **passed after one minor remediation** — 0 blocker, 0 major, 1 minor (ZH-SETAP-R1-001), remediated in the same commit as this bundle

## Auditor's finding text

One minor: `reset_soonest.description` read `未知的上报` ("unknown submissions") and dropped the accounts being spread; remediated to `对并列项和上报未知的账户做确定性分散` so the sentence names the accounts whose quota report is unknown. No placeholder, ICU, escape or terminology finding.

## Route and masking

Read route for this bundle: the translator and the auditor read the English source, the context sidecars and the existing zh-CN catalogues directly from the git checkout (`/Users/damian/lo-i18n-zh`) with ordinary file reads. That is permitted: `docs/security/redaction-policy.md` §2.2 classes catalogue copy as public and not secret, and §4 scopes the `read_public_copy` route to copy the shape pass would otherwise mask. `redaction-scan.json` records the measured result for every input file this wave used: 0 redaction markers in the committed bytes. A known display trap (the harness masks `name=value`-looking runs on the way to a transcript) was handled by verifying assignment tokens with ordinals rather than trusting the display; a byte scan of both shipped files for the ASCII marker returned 0 occurrences. No sanctioned-reader receipt is claimed for a read the auditor did not make.

The full auditor transcript is `audit.log`; the marker scan of every input is `redaction-scan.json`.
