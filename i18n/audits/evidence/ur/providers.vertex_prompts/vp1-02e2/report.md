# Primary independent audit: ur/providers.vertex_prompts

Audit date: 2026-11-19. Auditor: i18n-auditor / 02e2a562a33a / openai/gpt-6.1-sol (quota fallback from anthropic/claude-opus-5-5). Translator identity and completion time supplied by manager: i18n-translator / 541740ac0ac6 / openai/gpt-6.1-sol, 2026-11-19 16:19:22Z. Different agent sessions; no translation/MT service used.

## Verdict

Passed primary semantic/mechanical audit: all 4 keys read; 0 unresolved BLOCKER/MAJOR, 0 minor/nit findings. This is one audit, not proof the mandatory second audit has completed.

## Every-key semantic review (audit_notes_not_execution_log)

- `providers.vertex_prompts.field.project_id`
  - Current: Google Cloud پروجیکٹ ID
  - Judgment: Google Cloud پروجیکٹ ID accurately labels a Google Cloud project ID in logical storage order; both Latin islands remain intact.
- `providers.vertex_prompts.field.service_account`
  - Current: سروس اکاؤنٹ کا JSON یا JSON فائل کا پاتھ
  - Judgment: سروس اکاؤنٹ کا JSON یا JSON فائل کا پاتھ accurately describes service-account JSON or a JSON file path. پاتھ is an intelligible technical loan, not an API-key substitution.
- `providers.vertex_prompts.hint.required`
  - Current: بھیجنے کے لیے Enter دبائیں؛ خالی چھوڑ کر Enter یا Esc دبانے سے لاگ ان منسوخ ہو جائے گا۔
  - Judgment: بھیجنے کے لیے Enter دبائیں؛ خالی چھوڑ کر Enter یا Esc دبانے سے لاگ ان منسوخ ہو جائے گا۔ uses respectful دبائیں and explicitly states login cancellation; Urdu semicolon/full stop are correct. The coordination describes empty submission/Esc cancellation and does not claim optional skip.
- `providers.vertex_prompts.prompt.field`
  - Current: {field} درج کریں (منسوخ کرنے کے لیے خالی چھوڑیں): 
  - Judgment: {field} درج کریں (منسوخ کرنے کے لیے خالی چھوڑیں): follows native object-before-verb order and respectful imperatives. The field placeholder starts the logical string, ASCII parentheses/colon remain appropriate for this input prompt. No hand-authored bidi controls; no arrows to mirror. Latin-island placement still requires the renderer and bidi-capable terminal to display correctly.

## Fit / RTL / pseudo applicability

No maxCells/maxChars budget is declared. audit.log records actual rich.cells.cell_len measurements and direct ICU expansion with both field labels. Runtime formatting is execution, not a rendered terminal frame. The longest hints and service-account labels may truncate in narrow terminals (the context explicitly permits hint truncation). No copy shortening is required absent a concrete budget/rendered failure. CJK wide glyphs and Hindi combining marks require cell rather than code-point measurement; Vietnamese NFC checked. Urdu logical order and absence of manual controls checked; glyph shaping, bidi-capable terminal display, and full UI mirroring were not visually validated here. Pseudo-localization/live rendered fit/fallback/cancel behavior belongs to the PR QA/design lanes; this receipt is not those lanes' evidence. No plural/select branches exist, so locale CLDR categories are not applicable.
