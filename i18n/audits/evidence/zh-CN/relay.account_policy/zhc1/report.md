# zh-CN / relay.account_policy — independent translation audit (round 1)

- Locale: zh-CN · namespace: `relay.account_policy` · keys: 13
- Translator: translator (lopdev team), session 809b6f2773e0, started on anthropic/claude-sonnet-5-5; provider-quota fallbacks through openai/gpt-6-astra and kimi/k3.1-pro before the run completed on radient/auto
- Auditor: translation-auditor (lopdev team), session e139d7f8f30c, started on anthropic/claude-sonnet-5-5; quota fallbacks to openai/gpt-6-astra, then kimi/k3.1-pro, then radient/auto, which produced the findings
- Independence: the auditor is not the translator: translator session 809b6f2773e0, auditor session e139d7f8f30c, neither edited the other's files and the auditor edited nothing
- English source sha256: `0019b08106e6f7762194b164b9b9b37054a8cfade759aa2f6490787a73dea890`
- Context sidecar sha256: `7e55e4fadebcde0a6b93d545a62b2743fbb16009498cdf4e2122cb309ecc07f4`
- zh-CN target sha256 at audit time: `71b1cc7b3988bf8ccce82d7d6d599d9c170d9b3644cc76ce662629dc644eb2e2`
- Verdict: **clean** — 0 blocker, 0 major, 0 minor, 0 nit

## Auditor's finding text

Clean. All 13 keys read against the English source and sidecar. The auditor singled out the known-saved vs unknown-status distinction as handled correctly: `error` ("The change could not be confirmed") keeps `无法确认` and never asserts the write failed, while `rejected` does assert `更改未保存`.

## Route and masking

Read route for this bundle: the translator and the auditor read the English source, the context sidecars and the existing zh-CN catalogues directly from the git checkout (`/Users/damian/lo-i18n-zh`) with ordinary file reads. That is permitted: `docs/security/redaction-policy.md` §2.2 classes catalogue copy as public and not secret, and §4 scopes the `read_public_copy` route to copy the shape pass would otherwise mask. `redaction-scan.json` records the measured result for every input file this wave used: 0 redaction markers in the committed bytes. A known display trap (the harness masks `name=value`-looking runs on the way to a transcript) was handled by verifying assignment tokens with ordinals rather than trusting the display; a byte scan of both shipped files for the ASCII marker returned 0 occurrences. No sanctioned-reader receipt is claimed for a read the auditor did not make.

The full auditor transcript is `audit.log`; the marker scan of every input is `redaction-scan.json`.
