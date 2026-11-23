# i18n audit — vi cli.failures (goo1)

Independent audit of new key `cli.failures.github_owner_offline` (PR #4364), whole namespace re-checked at en source_sha256 `10452313f2d9c33a17496fabefb90870ef27631c1ac446972d7457b8000ce54c`.
Auditor: reviewer (lopdev team, 4364-i18n-audit), session f3778867a369, model anthropic/claude-opus-5-5, no fallback. Not the translator.
Audited worktree commit: ff0d7eef5ad7bb60b5ded0da18a05ad5d95c2e26 (base 981a699bc). Inputs: AGENTS.md "Internationalization is mandatory", docs/I18N.md, i18n/glossary.md, i18n/style-guides/vi.md.

English: GitHub login is lent by {owner}, which is offline; this session has no saved access to use meanwhile

Verdict: FAIL→corrected

As translated: Đăng nhập GitHub được cho mượn từ {owner}, thiết bị này đang ngoại tuyến; phiên này không có quyền truy cập đã lưu để dùng trong lúc chờ
Corrected:    Đăng nhập GitHub được {owner} cho mượn, nhưng thiết bị đó đang ngoại tuyến; phiên này không có quyền truy cập đã lưu để dùng trong lúc chờ

Reasoning: Meaning ambiguity: 'thiết bị này' (this device) reads in a CLI as the local machine, i.e. it could say THIS device is offline; and 'được cho mượn từ {owner}' mixes the lend/borrow direction. Corrected to agentive 'được {owner} cho mượn, nhưng thiết bị đó (that device) đang ngoại tuyến'. NFC verified; glossary phiên/thiết bị.
Correction: …được cho mượn từ {owner}, thiết bị này… → …được {owner} cho mượn, nhưng thiết bị đó…

Checks: placeholder set == {owner} (1 occurrence, exact); the five prior keys are byte-identical to 981a699bc (previously audited); no width budget declared in the context sidecar.
Findings: 1 (fixed in this audit); unresolved BLOCKER/MAJOR: 0.
