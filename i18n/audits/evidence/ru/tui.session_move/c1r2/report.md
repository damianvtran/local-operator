# c1-a2-ru — ROUND 2 delta audit (ru), independent auditor
Model served: anthropic/claude-sonnet-5-5 (no fallback, no MT). Input: c1-prep/ru.v2.json vs ru.json vs keys.json; worktree i18n-c1 @ b188079f32e6.
## Verdict: clean-with-findings — 0 BLOCKER, 0 MAJOR, 0 MINOR, 3 NIT (none gating)
## 1. Mechanical (python3, v2 vs keys.json)
- 22 keys, exact source order, same key set as v1; 11 changed, 11 unchanged.
- ICU/{placeholders}: identical sets per key (key, path, value, session, to, flag, reason, code, description). No plural/select in slice.
- Backticked spans byte-identical (cannot_wait, deadline_exceeded). Typed tokens verbatim: /approvals auto, /move, --to, --keep, --queue, --drop-session-roles, --wait, literal `local` (in <пир|local>).
- No empties; none English-identical; no bidi controls. Latin residue only: provider ids (hosting), auto/approvals (typed), shell, CLI (guide kept-Latin).
## 2. Changed keys
- cli.config.cleared "Настройка {key} сброшена: снова действует значение по умолчанию" — OK. Faithful to "Cleared … reads as default again"; avoids gender agreement trap.
- cli.config.list_title "Параметры конфигурации" — OK; equals shipped cli.config_help.list ("параметров конфигурации").
- list_desc_conversation_length / detail_length / auto_save_conversation — OK; беседа matches shipped wire.settings.conversation_length.label "Длина беседы" / auto_save_conversation.label. detail_length "которые не сжимаются в сводку" keeps "unsummarized"; consistent with УСТАРЕЛО … сжатием neighbour (max_learnings_history, unchanged).
- list_desc_classification_notice "вызов модели" — OK; «Умные подсказки» matches wire.settings.classification.title.
- tui.session_move.ambiguous "рабочего каталога" — OK; matches wire.slash.move "Сменить рабочий каталог этой сессии". Tally of shipped ru values: рабочий каталог 7 / рабочая папка 1 (relay chrome composer label; not our slice).
- tui.session_move.to_needs_device "--to требует указать устройство: /move --to <пир|local>" — OK (stiff but correct; flag verbatim).
## 3. Safety lines
- cli.config.running_sessions_unchanged: all 4 claims kept — running sessions unchanged («Работающие сессии не изменились»); config write cannot loosen approvals («запись конфигурации не может ослабить требования к одобрениям в них»; «одобрения» per style guide); /approvals auto typed in EACH session («в каждой сессии»); new sessions open at auto. PASS. Terminology: «Работающие сессии» 7 shipped values vs «запущенные сессии» 5 (cli.permissions.setup.done, wire.slash.info, relay.shell.list.empty): both shipped; not a defect.
- tui.session_move.deadline_exceeded: "за отведённое время результат перемещения не установлен; повторный запуск `lop sessions move {session} --to {to}` покажет, до какого этапа оно дошло" — UNKNOWN outcome stated; neither success nor failure claimed; command verbatim with {session}/{to}. Code path basis: session_move.py:209-217 TimeoutExpired → reaps child, _refusal(..., changed=True), no phase known. PASS.
## 4. Adjudication of DECLINED finding
- A-N4 cli.config.list_unrecognised, keep «неизвестный ключ; программа его не использует»: DECLINE UPHELD (my A-N4 suggestion «его не читает ни одна настройка» overturned). Basis: cli.py:3105-3114 — fragment follows «Описание:» for keys absent from settings_io.SETTINGS (hand-edited unknown top-level keys); the consumer is the program, not a setting. Shipped usage agrees: wire.settings.auto_save_conversation.help «рантайм TUI его не читает». «программа его не использует» is accurate, natural, neutral.
## 5. NITs (optional, no action required)
- N1 tui.session_move.deadline_exceeded: «результат перемещения не установлен» is slightly bureaucratic; «исход перемещения неизвестен» is plainer and states UNKNOWN more directly. Meaning currently safe.
- N2 cli.config.list_desc_model_name «Конкретная модель для взаимодействий» — mildly calqued plural; «Конкретная модель, используемая для взаимодействия» reads smoother. Accurate; fixes A-M3's «общения» imprecision.
- N3 cli.config.list_title «Параметры конфигурации» vs «Настройка {key}» in cleared (style guide lists «настройки» as preferred; shipped count параметр 13 / настройк 59). Title follows shipped cli.config_help.list so it's defensible; no change needed unless the team wants one word.
## Carried but out of scope (FYI): shipped cli.sessions_help.move_* use «разговор» for session id/"busy conversation" whereas session_move.py keys use «сессия» (shipped neighbours, not part of this delta).
## Commands run
python3 diff/mechanical script (keys, placeholders, backticks, tokens, empties, same-as-en, bidi) over keys.json/ru.json/ru.v2.json; python3 catalogue tally over worktree local_operator/i18n/catalogues/ru/ (13 regex term counts); read cli.py:3095-3120, session_move.py:200-220; read style guide ru.md, glossary lines. Did not run check.py (validator on shipped catalogues does not cover scratch v2). No ledger/record written (delta audit; manager records).
