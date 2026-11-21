# i18n audit — zh-CN cli.failures (goo1)

Independent audit of new key `cli.failures.github_owner_offline` (PR #4364), whole namespace re-checked at en source_sha256 `10452313f2d9c33a17496fabefb90870ef27631c1ac446972d7457b8000ce54c`.
Auditor: reviewer (lopdev team, 4364-i18n-audit), session f3778867a369, model anthropic/claude-opus-5-5, no fallback. Not the translator.
Audited worktree commit: ff0d7eef5ad7bb60b5ded0da18a05ad5d95c2e26 (base 981a699bc). Inputs: AGENTS.md "Internationalization is mandatory", docs/I18N.md, i18n/glossary.md, i18n/style-guides/zh-CN.md.

English: GitHub login is lent by {owner}, which is offline; this session has no saved access to use meanwhile

Verdict: PASS

As translated: GitHub登录由{owner}借出，该设备已离线；此会话在此期间没有可用的已保存访问权限
Shipped unchanged.

Reasoning: GitHub登录由{owner}借出，该设备已离线；… conveys lent/offline/no saved access; glossary 会话/设备; full-width ，；; no Latin/CJK space; statement with no next step so no 请; no final 。 matching source and sibling.


Checks: placeholder set == {owner} (1 occurrence, exact); the five prior keys are byte-identical to 981a699bc (previously audited); no width budget declared in the context sidecar.
Findings: 0 (fixed in this audit); unresolved BLOCKER/MAJOR: 0.
