# Primary independent audit: zh-CN/providers.vertex_prompts

Audit date: 2026-11-19. Auditor: i18n-auditor / 02e2a562a33a / openai/gpt-6.1-sol (quota fallback from anthropic/claude-opus-5-5). Translator identity and completion time supplied by manager: i18n-translator / 541740ac0ac6 / openai/gpt-6.1-sol, 2026-11-19 16:19:22Z. Different agent sessions; no translation/MT service used.

## Verdict

Passed primary semantic/mechanical audit: all 4 keys read; 0 unresolved BLOCKER/MAJOR, 0 minor/nit findings. This is one audit, not proof the mandatory second audit has completed.

## Every-key semantic review (audit_notes_not_execution_log)

- `providers.vertex_prompts.field.project_id`
  - Current: Google Cloud项目ID
  - Judgment: Google Cloud项目ID preserves the product name and technical ID; adjacent Latin and Han runs follow the explicit no-added-space policy.
- `providers.vertex_prompts.field.service_account`
  - Current: 服务账号JSON或JSON文件路径
  - Judgment: 服务账号JSON或JSON文件路径 accurately distinguishes a service account JSON value from a file path; JSON stays Latin.
- `providers.vertex_prompts.hint.required`
  - Current: 按Enter提交；留空按Enter或按Esc取消登录。
  - Judgment: 按Enter提交；留空按Enter或按Esc取消登录。 accurately distinguishes nonempty submit and empty/Esc login cancellation. Full-width semicolon and Chinese full stop are appropriate. This is a terse key-binding hint describing controls, not an error/validation next-step body requiring 请.
- `providers.vertex_prompts.prompt.field`
  - Current: 请输入{field}（留空取消）： 
  - Judgment: 请输入{field}（留空取消）： uses the required 请-led field instruction, full-width parentheses/colon, unchanged placeholder and preserved final separator space.

## Fit / RTL / pseudo applicability

No maxCells/maxChars budget is declared. audit.log records actual rich.cells.cell_len measurements and direct ICU expansion with both field labels. Runtime formatting is execution, not a rendered terminal frame. The longest hints and service-account labels may truncate in narrow terminals (the context explicitly permits hint truncation). No copy shortening is required absent a concrete budget/rendered failure. CJK wide glyphs and Hindi combining marks require cell rather than code-point measurement; Vietnamese NFC checked. Urdu logical order and absence of manual controls checked; glyph shaping, bidi-capable terminal display, and full UI mirroring were not visually validated here. Pseudo-localization/live rendered fit/fallback/cancel behavior belongs to the PR QA/design lanes; this receipt is not those lanes' evidence. No plural/select branches exist, so locale CLDR categories are not applicable.
