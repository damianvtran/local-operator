# ru / cli.failures — independent audit gbt2 (2026-11-16)

Auditor: i18n-auditor (audit2-ru), session 2753becb3d9e. Actual model: started on
openai/gpt-6.1-sol; after one tool call the provider returned quota HTTP 429 and the
harness fell back to anthropic/claude-opus-5-5, which performed the substantive audit.
Translator: i18n-translator (timeout-ru), openai/gpt-6.1-sol with fallback
anthropic/claude-opus-5-5 (as reported by parent), 2026-11-16. Same model family as
the translator's fallback is disclosed; different session; no MT service used.

Scope: all 5 keys of `cli.failures` at worktree commit 80a55b208 (base f0689a27).
Inputs read: i18n/glossary.md, i18n/style-guides/ru.md, docs/I18N.md, en catalogue +
context sidecar, ru catalogue, and the call site in local_operator/tools/builtin.py.
These are audit notes (reasoning), NOT an execution log; see audit.log for real output.

## Mechanical
check.py --catalogues-only: 0 problems. Keys 5/5, same order, placeholder names equal on
every key, no backticks/markup, valid JSON. No plurals in this namespace. No maxCells.

## Per key
- error_line `{prog}: ошибка: {message}` — keeps the `<command>: <error word>: <message>` shape. OK.
- required `необходимо указать следующие аргументы: {names}` — accurate, terse, lowercase like argparse. OK.
- invalid_choice `аргумент {argument}: недопустимое значение: {value} (выберите из {choices})` — accurate; «значение» fits both flag and subcommand use. OK.
- invalid_command `Недопустимая команда {group}: {command}` — accurate. OK.
- github_borrow_timeout `Не удалось получить учётные данные GitHub с устройства {owner} за {seconds} сек; повторите команду`
  - Glossary: credentials → «учётные данные» ✔, device → «устройство» ✔, GitHub kept ✔.
  - «с устройства {owner}» lets the device display name stay nominative/undeclined — good handling of an inserted proper name.
  - «сек» is the style-guide-sanctioned abbreviation; «повторите команду» matches existing ru copy (cli.browser_node). Impersonal, вы-register implicit, no «пожалуйста», no «!». ✔
  - Does not imply denial, revoke, login, an automatic extra retry or token/lease extension. ✔

## Findings
- R1 MINOR — github_borrow_timeout. «Не удалось получить … за {seconds} сек» frames the event as a failed attempt ("could not obtain") rather than the source's neutral timing statement ("did not arrive within"). The time bound keeps it readable as a timeout, so it does not assert denial, but «не удалось» is the generic failure idiom and slightly widens the possible reading. Suggested: «Учётные данные GitHub с устройства {owner} не поступили за {seconds} сек; повторите команду» (or «…не поступили в течение {seconds} сек…»). Optional.

## Observations (not translation findings, not counted)
- O1 — builtin.py renders seconds with `f"{timeout_s:g}"`; a fractional deadline would show a decimal point («7.5 сек») where the ru style guide wants a decimal comma. Only matters if the deadline can be non-integer; code-side, not catalogue.
- O2 — `generate.py --check` exits 1 here (orphaned bundled.<locale>.gen.ts after the shipped.json change in this commit). Implementer gate, unrelated to ru copy.

## Verdict
passed — 0 BLOCKER, 0 MAJOR, 1 MINOR, 0 NIT. No rendered frame and no native-human review.
