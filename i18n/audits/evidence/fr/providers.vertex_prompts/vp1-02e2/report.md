# Primary independent audit: fr/providers.vertex_prompts

Audit date: 2026-11-19. Auditor: i18n-auditor / 02e2a562a33a / openai/gpt-6.1-sol (quota fallback from anthropic/claude-opus-5-5). Translator identity and completion time supplied by manager: i18n-translator / 541740ac0ac6 / openai/gpt-6.1-sol, 2026-11-19 16:19:22Z. Different agent sessions; no translation/MT service used.

## Verdict

Passed primary semantic/mechanical audit: all 4 keys read; 0 unresolved BLOCKER/MAJOR, 0 minor/nit findings. This is one audit, not proof the mandatory second audit has completed.

## Every-key semantic review (audit_notes_not_execution_log)

- `providers.vertex_prompts.field.project_id`
  - Current: ID du projet Google Cloud
  - Judgment: ID du projet Google Cloud preserves the project identifier, rather than a project name; Google Cloud remains intact.
- `providers.vertex_prompts.field.service_account`
  - Current: JSON du compte de service ou chemin du fichier JSON
  - Judgment: JSON du compte de service ou chemin du fichier JSON correctly distinguishes pasted service-account JSON from its file path; not an API key.
- `providers.vertex_prompts.hint.required`
  - Current: Enter pour envoyer ; Enter à vide ou Esc pour annuler la connexion.
  - Judgment: Enter pour envoyer; Enter à vide ou Esc pour annuler la connexion accurately states submission and whole-login cancellation; the actual U+202F before the semicolon follows house French typography.
- `providers.vertex_prompts.prompt.field`
  - Current: Saisir {field} (laisser vide pour annuler) : 
  - Judgment: Saisir {field} uses the house infinitive action register; laisser vide pour annuler conveys cancellation, not optional skip. U+202F precedes the colon and trailing input-separator space is retained.

## Fit / RTL / pseudo applicability

No maxCells/maxChars budget is declared. audit.log records actual rich.cells.cell_len measurements and direct ICU expansion with both field labels. Runtime formatting is execution, not a rendered terminal frame. The longest hints and service-account labels may truncate in narrow terminals (the context explicitly permits hint truncation). No copy shortening is required absent a concrete budget/rendered failure. CJK wide glyphs and Hindi combining marks require cell rather than code-point measurement; Vietnamese NFC checked. Urdu logical order and absence of manual controls checked; glyph shaping, bidi-capable terminal display, and full UI mirroring were not visually validated here. Pseudo-localization/live rendered fit/fallback/cancel behavior belongs to the PR QA/design lanes; this receipt is not those lanes' evidence. No plural/select branches exist, so locale CLDR categories are not applicable.
