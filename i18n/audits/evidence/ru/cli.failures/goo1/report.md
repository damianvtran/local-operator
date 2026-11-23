# i18n audit — ru cli.failures (goo1)

Independent audit of new key `cli.failures.github_owner_offline` (PR #4364), whole namespace re-checked at en source_sha256 `10452313f2d9c33a17496fabefb90870ef27631c1ac446972d7457b8000ce54c`.
Auditor: reviewer (lopdev team, 4364-i18n-audit), session f3778867a369, model anthropic/claude-opus-5-5, no fallback. Not the translator.
Audited worktree commit: ff0d7eef5ad7bb60b5ded0da18a05ad5d95c2e26 (base 981a699bc). Inputs: AGENTS.md "Internationalization is mandatory", docs/I18N.md, i18n/glossary.md, i18n/style-guides/ru.md.

English: GitHub login is lent by {owner}, which is offline; this session has no saved access to use meanwhile

Verdict: FAIL→corrected

As translated: Вход в GitHub предоставлен устройством {owner}, которое сейчас не в сети; у этого сеанса нет сохранённого доступа на это время
Corrected:    Вход в GitHub предоставлен устройством {owner}, которое сейчас не в сети; у этой сессии нет сохранённого доступа на это время

Reasoning: Glossary violation: 'сеанса' for session; glossary §2 and ru guide fix session → сессия (corpus: 311 сесси* vs 5 сеанс*). Rest (Вход в GitHub предоставлен устройством {owner}, которое сейчас не в сети) idiomatic.
Correction: у этого сеанса → у этой сессии

Checks: placeholder set == {owner} (1 occurrence, exact); the five prior keys are byte-identical to 981a699bc (previously audited); no width budget declared in the context sidecar.
Findings: 1 (fixed in this audit); unresolved BLOCKER/MAJOR: 0.
