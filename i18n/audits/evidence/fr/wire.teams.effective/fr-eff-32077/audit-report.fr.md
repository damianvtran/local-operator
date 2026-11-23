# Independent French audit — wire.teams.effective

Date: 2026-11-19. Auditor: i18n-auditor, session 32077ea6da37, actual model openai/gpt-6.1-sol (host Model field; no fallback used). Translator: i18n-translator, session 34a8703ccbff, openai/gpt-6.1-sol (translator receipt; distinct session). No translation/MT service used. Frozen source worktree: /Users/damian/local-operator/.worktrees/i18n-effective-reasons at e70bc818b8553dca1fad007705257817be49ee37.

## Verdict

**passed — independent linguistic/mechanical audit of all 9 keys; 0 unresolved BLOCKER/MAJOR, 0 findings.** No translation edits requested. This is one independent audit, not second-pass certification or complete release/visual approval.

## Mechanical execution

Executed scratch audit_fr.py with the existing primary-checkout Python, -B and PYTHONDONTWRITEBYTECODE=1. Imported validator/parser routines from the frozen worktree, not the installed runtime: scripts.i18n.check._arguments_problem, _check_plural_categories, catalogues.context_entry_problem and local_operator.i18n.runtime.message_arguments/plural_selectors. Actual stdout is audit.fr.log. All pinned source/context/target/translator-receipt hashes verified; strict JSON load rejected duplicate members; exact 9-key membership and context coverage; ordered placeholder names/types; source technical literals; no introduced markup/backticks or bidi controls. Independent regex/literal checks supplemented the repository parser. No plural/select in either catalogue: French one/many/other categories are not required arms on simple interpolation messages.

This is targeted execution of repository mechanical validator routines against the external candidate. The full scripts/i18n/check.py CLI, generation checks, unit suites, builds, installation, servers and UI rendering were NOT run. The candidate has not been integrated; no repository/ledger/generated files were written.

## Complete semantic review

| Key suffix | Result and reasoning |
|---|---|
| team_missing | PASS. « Aucune équipe nommée {role} n’a été trouvée. » preserves absence, opaque team name, and lookup rather than launch semantics. |
| registry_unreadable | PASS. « Impossible de lire le registre des agents. Vérifiez les définitions des agents et réessayez. » preserves uncertainty and the correct remedy without claiming corruption or exposing an exception. |
| config_unreadable | PASS. « Impossible de lire la configuration du modèle. Vérifiez votre configuration et réessayez. » is natural concise French, professional vous imperative, accurate retry remedy. |
| tier_invalid_value | PASS. « La valeur de subagents.models.{tier} est invalide. Utilisez un sélecteur provider/model ou un objet de configuration de modèle. » preserves the two valid alternatives. The context explicitly defines mapping as a configuration object; translating it as a lookup/correspondence table would be wrong here. |
| tier_invalid_mapping | PASS. « L’objet de configuration à l’emplacement subagents.models.{tier} est invalide. Définissez model sur un sélecteur provider/model et, si nécessaire, effort sur un niveau d’effort de raisonnement pris en charge. » preserves mandatory model, optional effort and supported reasoning effort. “Si nécessaire” is an idiomatic discretionary qualification, not a new mandatory field. The instruction remains actionable; literal field names model/effort and provider/model syntax are intact. |
| tier_unconfigured | PASS. « Aucun modèle n’est configuré à l’emplacement subagents.models.{tier}. » accurately identifies absence at the tier path, not an invalid selector. |
| tier_invalid_selector | PASS. « Le sélecteur de modèle à l’emplacement subagents.models.{tier} est invalide. Utilisez le format provider/model. » preserves the required syntax and does not leak or invent the bad raw value. |
| model_unresolvable | PASS. « Impossible de déterminer le modèle correspondant à {selector}. Vérifiez le sélecteur de modèle et la configuration du fournisseur. » conveys selector-to-model resolution rather than the model not existing; both diagnostic checks remain. This is idiomatic French rather than the literal “résoudre le modèle”. |
| unavailable | PASS. « Ce membre de l’équipe est indisponible. Vérifiez la configuration de son modèle et sa définition d’agent. » preserves defensive availability warning and both remedies, without implying an attempted launch. |

## Register, glossary, punctuation and fit

All ordinary diagnostic prose is French. équipe, modèle, fournisseur, configuration and agent read naturally; professional vous appears through Vérifiez and Utilisez, with no tu or bureaucratic Veuillez. model/effort, provider/model and subagents.models.{tier} are explicit syntax, not untranslated UI prose. Apostrophes use U+2019. No prose colon, semicolon, question mark or exclamation mark needs a French U+202F space. There are no quoted passages requiring guillemets, digits requiring number formatting, Rich tags, code spans or commands to preserve beyond the stated identifiers.

Context sidecar specifies natural wrapping and no fixed maxCells/maxChars on all nine keys. The configuration-object remedy is long but complete and semantically justified. No declared width violation can be established. Static inspection finds ordinary Latin-script French and no CJK/combining-control risk. Actual wrapping/truncation in the desktop warning/editor has NOT been verified; long opaque role/selector values remain an integration fit surface.

## Applicability and outstanding integration gates

- **High-visibility second independent audit: REQUIRED.** These are inline error/unavailability reasons plus model-configuration remedies in a saved-team view/editor. Treat them as high-visibility/error copy. This report is one independent pass only. A different independent auditor must complete the other pass; this session cannot manufacture it by rereading itself. No claim is made about another session’s results.
- **Rendered evidence: applicable, UNRUN.** Manager/integrator must supply desktop saved-team/editor frames and long-value/wrapping proof. This report is not rendered/design/UX approval.
- **Pseudo-localization: applicable to client fit, UNRUN.** The core HOWTO advertises no one-command pseudo workflow; no pseudo evidence is invented.
- **RTL: NOT APPLICABLE to this French candidate.** French is LTR, and no bidi controls were introduced. This does not certify Urdu or mirrored-client behaviour.
- **CLDR plural branch execution: NOT APPLICABLE.** There are no plural or select branches in the nine messages.
- **Current ledger append:** a scratch receipt follows current core shape and is schema-validated/read-only dry-run checked. No append --apply or tracked evidence bundle was authorized or performed. The manager must retain these originals and commit a canonical i18n/audits/evidence/fr/wire.teams.effective/<audit-id>/ bundle with matching hashes and receipt note before claiming durable tracked evidence.

## Sources and retention

Normative sources read: guide://i18n, frozen docs/I18N.md (locale pipeline), docs/design/i18n.md §6 (current receipt), i18n/style-guides/fr.md, i18n/glossary.md, source/context sidecar. A requested guide://i18n/coreHOWTO/glossary is not a real guide; guide://i18n directed this audit to the repository HOWTO/glossary instead. All source/target/provenance input SHA-256 values are in record.fr.json. This Markdown is audit reasoning, not a command-execution log; only audit.fr.log and receipt-validation.fr.log record actual commands’ stdout. Scratch retention is explicitly required for this task; no claim these files are tracked, committed or on main.
