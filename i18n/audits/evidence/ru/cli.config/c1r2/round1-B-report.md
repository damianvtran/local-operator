# c1-au-ru-B — independent audit, ru, slice C1 (22 keys)

Auditor: i18n-auditor (lopdev), model anthropic/claude-sonnet-5-5. Independent of translator (openai/gpt-6.1-sol).
Verdict: clean-with-findings — 0 BLOCKER, 0 MAJOR, 4 MINOR, 4 NIT.

## Mechanical (python3, run by me)
- 22 keys, same set and order as keys.json; no empties, none identical to English, no stray whitespace.
- {placeholder} sets identical for all keys.
- Backtick spans byte-identical (cannot_wait, deadline_exceeded). `--to/--keep/--queue/--drop-session-roles/--wait`, `/move`, `/approvals auto`, `local`, `<id> <to> <seconds>` verbatim.
- Angle words `<path> <session> <peer>` → `<путь> <сессия> <пир>` (sanctioned by notes); `<peer|local>` → `<пир|local>` (local Latin, bar/brackets kept).
- Remaining Latin only: provider names, `auto`, shell, CLI, flags/commands. No stray English.
- list_title 22 cells (<30 hint); no other width cap.

## Safety claims — preserved
- running_sessions_unchanged: config write can't loosen running sessions + type /approvals auto in each session + new sessions open at auto: all intact, calm.
- deadline_exceeded: outcome UNKNOWN ("не завершилось за отведённое время … покажет, на каком этапе"), neither success nor failure.
- cannot_wait, unsupported_flag, to_needs_device exact.
- Register «вы»-imperative/impersonal, no "!", no Title Case. Terms: сжатие, «Умные подсказки», выводы, shell/CLI Latin, пир — all match shipped catalogue.

## Findings
MINOR
1. tui.session_move.ambiguous — «для смены рабочей папки». Shipped catalogue uses «рабочий каталог» for this concept (wire.slash.move "Сменить рабочий каталог этой сессии", wire.settings cwd, wire.errors.desktop cwd_*; 7 каталог vs 1 папка). Suggest «Используйте /move <путь> для смены рабочего каталога или /move <сессия> --to <пир> для перемещения сессии.»
2. cli.config.list_desc_model_name — «Модель для общения» drops "specific", renders "interactions" as chat-ish «общение». Suggest «Конкретная модель, используемая для взаимодействия».
3. cli.config.list_desc_auto_save_conversation — EN "Whether to…" boolean phrasing lost. Suggest «Сохранять ли разговоры автоматически».
4. cli.config.cleared — «Значение {key} удалено: снова используется значение по умолчанию»: «значение» twice; «удалено» may read as key deleted. Suggest «{key} сброшен: снова действует значение по умолчанию».

NIT
1. cli.config.list_title «Настройки конфигурации» tautological; «Параметры конфигурации» mirrors "Options" (настройки is glossary-preferred, so acceptable).
2. cli.config.running_sessions_unchanged «требования к одобрениям в них» clunky → «ослабить их требования к одобрениям»; shipped prefers «работающие сессии» though «запущенные» also appears.
3. cli.config.list_desc_classification_notice «вызов» → «вызов модели» (shipped says just «вызов»; optional).
4. «разговор» used consistently here (as cli.sessions_help.move) while wire.settings for same settings says «беседа»; corpus split ~90/89, no glossary row — observation only.

## Commands run
python3 mechanical checks (key set/order, placeholders, backticks, flags, commands, angle brackets, identical/empty, Latin scan); python3/grep over shipped ru catalogues (каталог/папка, запущенные/работающие, беседа/разговор, ИИ, сжатие, Умные подсказки); read ru.md style guide + glossary rows; grep use sites cli.py:3028–3112, tui/session_move.py:54–229.
No ledger row recorded (task asked for report only).
