# i18n audit — hi cli.failures (goo1)

Independent audit of new key `cli.failures.github_owner_offline` (PR #4364), whole namespace re-checked at en source_sha256 `10452313f2d9c33a17496fabefb90870ef27631c1ac446972d7457b8000ce54c`.
Auditor: reviewer (lopdev team, 4364-i18n-audit), session f3778867a369, model anthropic/claude-opus-5-5, no fallback. Not the translator.
Audited worktree commit: ff0d7eef5ad7bb60b5ded0da18a05ad5d95c2e26 (base 981a699bc). Inputs: AGENTS.md "Internationalization is mandatory", docs/I18N.md, i18n/glossary.md, i18n/style-guides/hi.md.

English: GitHub login is lent by {owner}, which is offline; this session has no saved access to use meanwhile

Verdict: PASS

As translated: GitHub लॉगिन {owner} द्वारा दिया गया है, जो ऑफ़लाइन है; इस सत्र के पास इस बीच उपयोग के लिए कोई सहेजी गई पहुँच नहीं है
Shipped unchanged.

Reasoning: Meaning conveyed (लॉगिन {owner} द्वारा दिया गया है, जो ऑफ़लाइन है); glossary सत्र; 'सहेजी गई पहुँच' unambiguous 'saved'; लॉगिन matches corpus (27); आप register not engaged; Latin digits n/a; no ZWJ/ZWNJ.


Checks: placeholder set == {owner} (1 occurrence, exact); the five prior keys are byte-identical to 981a699bc (previously audited); no width budget declared in the context sidecar.
Findings: 0 (fixed in this audit); unresolved BLOCKER/MAJOR: 0.
