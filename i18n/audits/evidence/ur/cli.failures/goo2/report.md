# i18n audit — ur cli.failures (goo2)

Independent re-audit (round 2) of `cli.failures.github_owner_offline` (PR #4364) after the English changed from "which is offline" to "which isn't answering right now"; en source_sha256 `da8725bbc324ad5124716489a31616dc4e7d999ae1ae07f9060d206f0137bc95`.
Auditor: reviewer (lopdev team, 4364-i18n-audit-r2), session 683ecf30b5dc, model anthropic/claude-opus-5-5, no fallback. Not the translator (re-translation by the lane manager in 6d1afe8c3).
Audited worktree commit: 6d1afe8c340e2f3d81e944c7e8cee3e6ce01c3df (round-1 audit 16a10317a, base 981a699bc). Inputs: AGENTS.md "Internationalization is mandatory", docs/I18N.md, i18n/glossary.md, i18n/style-guides/ur.md.

English: GitHub login is lent by {owner}, which isn't answering right now; this session has no saved access to use meanwhile

Verdict: PASS

Previous (round 1): GitHub لاگ اِن {owner} کی طرف سے دیا گیا ہے، جو آف لائن ہے؛ اس سیشن کے پاس اس دوران استعمال کے لیے کوئی محفوظ شدہ رسائی نہیں ہے
As translated:      GitHub لاگ اِن {owner} کی طرف سے دیا گیا ہے، جو ابھی جواب نہیں دے رہا؛ اس سیشن کے پاس اس دوران استعمال کے لیے کوئی محفوظ شدہ رسائی نہیں ہے

Reasoning: 'جو ابھی جواب نہیں دے رہا' renders "isn't answering right now"; the copula-less form is the recorded sibling «جواب نہیں دے رہا» (ur style guide l.82-85; cli.network_uplink.not_answering). Round-1 correction persists: محفوظ شدہ رسائی. RTL: logical order, Urdu ، and ؛, no manual bidi/format controls (checked), GitHub/{owner} as LTR islands. Glossary سیشن.

Checks: placeholder set == {owner} (1 occurrence, exact); only this key changed since 16a10317a, the other five keys are byte-identical; no width budget declared in the context sidecar.
Findings: 0; unresolved BLOCKER/MAJOR: 0.
