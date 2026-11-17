# Audit gbt2 — es / cli.failures (independent, fresh round)

- Worktree commit audited: 80a55b208 (base f0689a27). Keys: 5/5 (new: `cli.failures.github_borrow_timeout`).
- Auditor: i18n-auditor (audit2-es), session b5fe50e51391. Model: started on openai/gpt-6.1-sol, which failed with HTTP 429 at the first tool call; every judgement below was made on the fallback anthropic/claude-opus-5-5.
- Inputs read: `i18n/style-guides/es.md`, `i18n/glossary.md`, `docs/I18N.md`, en catalogue + context sidecar, es catalogue, and the call site (`local_operator/tools/builtin.py::_github_borrow_timeout_note`). No MT service used. No earlier receipts for this key were consulted.
- Verdict: **passed**. 0 blocker, 0 major, 2 minor, 0 nit.

## Mechanical (see audit.log, checks.py.txt)
Key set and order match en; placeholder names and types match (`owner`, `seconds` simple); no ICU plural/select in this group; no backtick spans; valid JSON; `scripts/i18n/check.py --catalogues-only` reports 0 problems. Width: the new line is 83 cells in both en and es; no maxCells; one stderr line.

## Accuracy / safety: `cli.failures.github_borrow_timeout`
Current: `La credencial de GitHub desde {owner} no llegó en {seconds} s; reintenta el comando`
Meaning preserved: a timeout (credential did not arrive within the deadline), tú imperative remedy. It does not imply a denial, revocation, login, an automatic or extra retry, or a token/lease extension. GitHub kept; `credencial` matches the glossary (credentials → credenciales); `{seconds} s` keeps a space before the unit symbol, as Spanish convention expects; no final period, matching the source.

### F1 — MINOR — word order of `desde {owner}`
Problem: "La credencial de GitHub desde {owner}" hangs "desde" off the noun, a calque of "credential from X"; in Spanish, "desde" reads naturally only next to the motion verb. It also stacks with "de GitHub".
Suggested: `La credencial de GitHub no llegó desde {owner} en {seconds} s; vuelve a ejecutar el comando`

### F2 — MINOR — `reintenta el comando`
Problem: transitive "reintentar + object" is understandable, but this catalogue's settled remedy for re-running a command is "vuelve a ejecutar (este) comando" (e.g. two shipped cli strings ending "vuelve a ejecutar este comando"), and elsewhere it uses "reintentar" intransitively ("antes de reintentar"). "vuelve a ejecutar el comando" is more idiomatic and consistent, and stays unambiguous that the user re-runs it (no automatic retry implied).
Suggested: `…; vuelve a ejecutar el comando`

## Other keys (all clean)
- `error_line` `{prog}: error: {message}` — shape kept; "error" is also the Spanish word.
- `required` `se requieren los siguientes argumentos: {names}` — natural argparse-style es.
- `invalid_choice` `argumento {argument}: opción no válida: {value} (elige entre {choices})` — tú, correct.
- `invalid_command` `Comando de {group} no válido: {command}` — only "Invalid"/"command" translated, as the note asks.

## Out-of-scope observation (not a finding)
The en catalogue's new line is indented with 1 space where its siblings use 2; cosmetic, en-side, no effect on parsing.
