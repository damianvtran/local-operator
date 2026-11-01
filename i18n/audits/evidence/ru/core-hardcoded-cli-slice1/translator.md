# Russian core literals

Translator session: 131e937f47aa. Actual translation model: openai/gpt-6.1-sol (provider-quota fallback after anthropic/claude-sonnet-5-5 and anthropic/claude-opus-5-5).

## Files and keys

All three target files already contained Russian translations when read. Retained their translations except two tightened group summaries (`skills`, `operator`); independently derived the final terminology choices from the English/context files, glossary and Russian style guide. No audit or passed status is claimed.

- `local_operator/i18n/catalogues/ru/cli.failures.json` — 4 keys: `error_line`, `required`, `invalid_choice`, `invalid_command`.
- `local_operator/i18n/catalogues/ru/cli.group_help.json` — 10 keys: `skills`, `skills_description`, `tunnel`, `secret`, `network`, `disk`, `operator`, `pair`, `search`, `fetch`.
- `local_operator/i18n/catalogues/ru/tui.setup_cues.json` — 2 keys: `setup_placeholder`, `no_provider_cue`.

## Choices and ambiguities

- Error wrapper: `ошибка`, retaining `{prog}: ошибка: {message}`.
- Missing arguments: impersonal `необходимо указать следующие аргументы: {names}`. The colon introduces verbatim technical names without imposing inflection on either one name or a list.
- Invalid choice: `недопустимое значение: {value}`; the already-quoted value remains untouched. Choices also remain verbatim after `выберите из`.
- Invalid command: `Недопустимая команда {group}: {command}`. The group identifier remains undeclined; a missing verb remains the literal `None` supplied at runtime.
- Skills: `навыки` per glossary; `загрузка` for retrieving a skill from Agent Hub, avoiding an unnecessary parenthetical command token. Agent Hub remains a product name. `пакеты` denotes bundles with additional files; the executable-bit safety warning remains explicit.
- Harness: `рантаймы`, consistent with the Russian guide's recorded runtime terminology. Radient remains unchanged.
- Mesh: retained `Mesh-сети`, a recognizable technical network designation.
- Approval gate: `контроль одобрений`, preserving glossary `одобрения` rather than confusing approval with confirmation. `ослабления` makes the key's effect explicit.
- Headless fetch: `без видимого окна браузера`, explaining the concrete behavior rather than using a literal calque.
- Pairing retains `однократная авторизация вашей подписью`, referring to the one-time pairing authorization, not every subsequent action.

Glossary proposals (not changes): Agent Hub bundle → `пакет`; mesh network → `Mesh-сеть`; approval gate → `контроль одобрений`; headless browser → `браузер без видимого окна`.

## Structural verification and width

Used a small Python inspection (not the repository validator or a test suite) to parse all three final JSON files, confirm exact English key order/count and exact placeholder sequences, and confirm canonical `ensure_ascii=False`, indent=2, one-trailing-newline formatting. Results: 4 + 10 + 2 = 16 keys; all checks OK.

`no_provider_cue` renders as `Сначала подключите аккаунт ИИ: введите /login radient`: **53 terminal cells / 74**, including the interpolated command. Measured by Unicode combining/width properties; every character in this rendered Russian/Latin string is single-width. No string exceeded a declared budget.

Repository validator, generate.py, tests and git commands were NOT run, as explicitly instructed. Ledger, generated assets, allowance, code and tests were untouched. Independent semantic audit remains for the separate auditor.
