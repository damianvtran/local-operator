# French core literals — translation handoff

Translator session: 446e242f9009. Actual translation model: openai/gpt-6.1-sol, following provider-quota fallback notices from the requested anthropic/claude-sonnet-5-5 through anthropic/claude-opus-5-5. No machine-translation service used. Translation only; no independent audit claimed.

## Files and keys

Under `/Users/damian/local-operator/.worktrees/lit-translate/local_operator/i18n/catalogues/fr/`:

- `cli.failures.json` (4): `error_line`, `required`, `invalid_choice`, `invalid_command`.
- `cli.group_help.json` (10): `skills`, `skills_description`, `tunnel`, `secret`, `network`, `disk`, `operator`, `pair`, `search`, `fetch`.
- `tui.setup_cues.json` (2): `setup_placeholder`, `no_provider_cue`.

16 keys total, with each namespace prefix retained. Existing French values were present when read; retained appropriate established wording and refined the skills product name and disk-safety wording. All three assigned files were written in canonical JSON.

## Decisions and terminology

- `error_line` is exactly `{prog}: erreur: {message}`: only `error` translated, preserving the explicitly required CLI wrapper shape, including its ASCII colons. This narrowly follows the task’s wrapper-specific instruction over general French prose spacing.
- Elsewhere colons have U+202F before them, and prose apostrophes are U+2019. Choice quotes are supplied within `{value}` / `{choices}` and preserved verbatim; no guillemets added around technical tokens.
- `skills`: compétence; product name Agent Hub kept exactly. `pull` retained per French style guide/glossary. `bundle` rendered paquet, consistent with the existing target wording; glossary proposal: record paquet for a packaged skill bundle.
- `harness`: runtime, matching the French guide’s recorded kept-English term and the context’s definition as the local lop runtime. Plural runtimes follows the existing French target wording.
- `disk`: replaced “sans risque avéré” (which can mean no *proven risk*) with “ne supprimer qu’après vérification de l’absence de risque”, retaining the stronger requirement that safety be established before reclaiming space.
- `secret`: “sans vous solliciter” means no repeated user prompt, not absence of an AI prompt.
- `operator`: opérateur; approval gate rendered contrôle d’approbation. Glossary proposal: record this term for a live approval gate.
- `fetch`: “sans fenêtre de navigateur” explains headless in ordinary French without inventing a product term.

## Checks actually executed

Read-back verification with Python JSON parsing: all 16 keys present, exact English key order and placeholder order; canonical `ensure_ascii=False`, indent 2, one final newline. No keys added or removed.

Cell widths measured using `/Users/damian/local-operator/.venv/bin/python` and `rich.cells.cell_len`, substituting the actual `/login radient` command:

- `setup_placeholder`: 36 cells — `Saisir /login radient pour commencer`.
- `no_provider_cue`: 54 / 74 cells — `Connecter d’abord un compte IA : saisir /login radient`.

Nothing exceeds the declared budget.

Catalogue validator, generator, tests and git commands were NOT run, as explicitly instructed for this translation slice. No ledger, generated files, code or other catalogues changed. Independent audit remains for the separately assigned auditor.
