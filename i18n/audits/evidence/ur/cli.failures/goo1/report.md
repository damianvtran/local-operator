# i18n audit — ur cli.failures (goo1)

Independent audit of new key `cli.failures.github_owner_offline` (PR #4364), whole namespace re-checked at en source_sha256 `10452313f2d9c33a17496fabefb90870ef27631c1ac446972d7457b8000ce54c`.
Auditor: reviewer (lopdev team, 4364-i18n-audit), session f3778867a369, model anthropic/claude-opus-5-5, no fallback. Not the translator.
Audited worktree commit: ff0d7eef5ad7bb60b5ded0da18a05ad5d95c2e26 (base 981a699bc). Inputs: AGENTS.md "Internationalization is mandatory", docs/I18N.md, i18n/glossary.md, i18n/style-guides/ur.md.

English: GitHub login is lent by {owner}, which is offline; this session has no saved access to use meanwhile

Verdict: FAIL→corrected

As translated: GitHub لاگ اِن {owner} کی طرف سے دیا گیا ہے، جو آف لائن ہے؛ اس سیشن کے پاس اس دوران استعمال کے لیے کوئی محفوظ رسائی نہیں ہے
Corrected:    GitHub لاگ اِن {owner} کی طرف سے دیا گیا ہے، جو آف لائن ہے؛ اس سیشن کے پاس اس دوران استعمال کے لیے کوئی محفوظ شدہ رسائی نہیں ہے

Reasoning: Meaning ambiguity: bare 'محفوظ رسائی' reads as 'secure access' (implying the session's access is insecure) rather than 'saved access'; corpus uses 'محفوظ شدہ' for saved (27 occurrences). RTL: logical order, Urdu ، and ؛, no manual bidi controls (checked), GitHub/{owner} kept as LTR islands; glossary سیشن; لاگ اِن matches corpus (28).
Correction: محفوظ رسائی → محفوظ شدہ رسائی

Checks: placeholder set == {owner} (1 occurrence, exact); the five prior keys are byte-identical to 981a699bc (previously audited); no width budget declared in the context sidecar.
Findings: 1 (fixed in this audit); unresolved BLOCKER/MAJOR: 0.
