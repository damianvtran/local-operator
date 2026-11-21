# i18n audit — es cli.failures (goo1)

Independent audit of new key `cli.failures.github_owner_offline` (PR #4364), whole namespace re-checked at en source_sha256 `10452313f2d9c33a17496fabefb90870ef27631c1ac446972d7457b8000ce54c`.
Auditor: reviewer (lopdev team, 4364-i18n-audit), session f3778867a369, model anthropic/claude-opus-5-5, no fallback. Not the translator.
Audited worktree commit: ff0d7eef5ad7bb60b5ded0da18a05ad5d95c2e26 (base 981a699bc). Inputs: AGENTS.md "Internationalization is mandatory", docs/I18N.md, i18n/glossary.md, i18n/style-guides/es.md.

English: GitHub login is lent by {owner}, which is offline; this session has no saved access to use meanwhile

Verdict: PASS

As translated: El inicio de sesión de GitHub lo presta {owner}, que está sin conexión; esta sesión no tiene acceso guardado para usar mientras tanto
Shipped unchanged.

Reasoning: Meaning and idiom correct (lo presta {owner}, sin conexión, mientras tanto); no address form needed (tú rule not engaged); glossary sesión; 'inicio de sesión de GitHub' matches corpus; no space before ';'.


Checks: placeholder set == {owner} (1 occurrence, exact); the five prior keys are byte-identical to 981a699bc (previously audited); no width budget declared in the context sidecar.
Findings: 0 (fixed in this audit); unresolved BLOCKER/MAJOR: 0.
