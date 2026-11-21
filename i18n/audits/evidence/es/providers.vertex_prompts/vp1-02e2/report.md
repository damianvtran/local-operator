# Primary independent audit: es/providers.vertex_prompts

Audit date: 2026-11-19. Auditor: i18n-auditor / 02e2a562a33a / openai/gpt-6.1-sol (quota fallback from anthropic/claude-opus-5-5). Translator identity and completion time supplied by manager: i18n-translator / 541740ac0ac6 / openai/gpt-6.1-sol, 2026-11-19 16:19:22Z. Different agent sessions; no translation/MT service used.

## Verdict

Passed primary semantic/mechanical audit: all 4 keys read; 0 unresolved BLOCKER/MAJOR, 0 minor/nit findings. This is one audit, not proof the mandatory second audit has completed.

## Every-key semantic review (audit_notes_not_execution_log)

- `providers.vertex_prompts.field.project_id`
  - Current: ID del proyecto de Google Cloud
  - Judgment: ID del proyecto de Google Cloud is natural neutral Spanish and preserves the technical ID.
- `providers.vertex_prompts.field.service_account`
  - Current: JSON de la cuenta de servicio o ruta del archivo JSON
  - Judgment: JSON de la cuenta de servicio o ruta del archivo JSON is accurate; archivo follows the neutral-Spanish guide rather than region-marked fichero.
- `providers.vertex_prompts.hint.required`
  - Current: Enter para enviar; Enter sin texto o Esc para cancelar el inicio de sesión.
  - Judgment: Enter para enviar; Enter sin texto o Esc para cancelar el inicio de sesión preserves the login cancellation consequence rather than optional skip.
- `providers.vertex_prompts.prompt.field`
  - Current: Introduce {field} (deja vacío para cancelar): 
  - Judgment: Introduce {field} (deja vacío para cancelar) addresses the user in the required tú register; this is a field instruction, not an infinitive button label. Placeholder and trailing space remain intact.

## Fit / RTL / pseudo applicability

No maxCells/maxChars budget is declared. audit.log records actual rich.cells.cell_len measurements and direct ICU expansion with both field labels. Runtime formatting is execution, not a rendered terminal frame. The longest hints and service-account labels may truncate in narrow terminals (the context explicitly permits hint truncation). No copy shortening is required absent a concrete budget/rendered failure. CJK wide glyphs and Hindi combining marks require cell rather than code-point measurement; Vietnamese NFC checked. Urdu logical order and absence of manual controls checked; glyph shaping, bidi-capable terminal display, and full UI mirroring were not visually validated here. Pseudo-localization/live rendered fit/fallback/cancel behavior belongs to the PR QA/design lanes; this receipt is not those lanes' evidence. No plural/select branches exist, so locale CLDR categories are not applicable.
