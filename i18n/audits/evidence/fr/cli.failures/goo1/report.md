# i18n audit — fr cli.failures (goo1)

Independent audit of new key `cli.failures.github_owner_offline` (PR #4364), whole namespace re-checked at en source_sha256 `10452313f2d9c33a17496fabefb90870ef27631c1ac446972d7457b8000ce54c`.
Auditor: reviewer (lopdev team, 4364-i18n-audit), session f3778867a369, model anthropic/claude-opus-5-5, no fallback. Not the translator.
Audited worktree commit: ff0d7eef5ad7bb60b5ded0da18a05ad5d95c2e26 (base 981a699bc). Inputs: AGENTS.md "Internationalization is mandatory", docs/I18N.md, i18n/glossary.md, i18n/style-guides/fr.md.

English: GitHub login is lent by {owner}, which is offline; this session has no saved access to use meanwhile

Verdict: FAIL→corrected

As translated: La connexion GitHub est prêtée par {owner}, qui est hors ligne ; cette session n’a aucun accès enregistré à utiliser en attendant
Corrected:    La connexion GitHub est prêtée par {owner}, qui est hors ligne ; cette session n’a aucun accès enregistré à utiliser en attendant

Reasoning: Space before ';' was ASCII U+0020; fr guide (Punctuation) requires U+202F before : ; ! ? — 466/509 fr ';' in the corpus use U+202F and the sibling github_borrow_timeout does. Meaning, idiom (connexion GitHub prêtée par…, hors ligne, en attendant), {owner} and glossary (session) otherwise correct.
Correction: U+0020 → U+202F before ';'

Checks: placeholder set == {owner} (1 occurrence, exact); the five prior keys are byte-identical to 981a699bc (previously audited); no width budget declared in the context sidecar.
Findings: 1 (fixed in this audit); unresolved BLOCKER/MAJOR: 0.
