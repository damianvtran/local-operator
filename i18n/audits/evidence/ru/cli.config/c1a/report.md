# i18n audit report — ru · cli.config · audit `c1a`

- Locale: ru · namespace: `cli.config` · keys: 15 · date: 2026-11-11
- Auditor: i18n-auditor, session `e45ca4d0d71c`, actual served model `deepseek/deepseek-flash` (harness model line; no fallback used). Independent of the translator (`openai/gpt-6.1-sol`).
- Verdict: **clean-with-findings** — BLOCKER 0 · MAJOR 0 · MINOR 4 · NIT 4; ledger status **`passed`** (no blocker/major remains).

Inputs, hashes, mechanical results, commands and the ledger-status reasoning are in the full slice report `audit-ru-A.md` (retained beside the manager's c1-prep inputs) and in `audit.log` here. `record.json` lists the hashes; `audit.log` is captured command output.

## Findings

### M1 — MINOR — `cli.config.cleared`

- Current: `Значение {key} удалено: снова используется значение по умолчанию`
- Problem: «значение» twice in one 64-cell line (en 44) and the fronted participle makes a plain confirmation read heavy; the shipped CLI help calls a setting key «ключ» (cli.config_help.edit_key «Обновляемый ключ конфигурации»).
- Suggested: `Ключ {key} очищен: снова действует значение по умолчанию (56 cells)`

### M2 — MINOR — `cli.config.list_desc_detail_length`

- Current: `[УСТАРЕЛО — заменено сжатием] Число последних сообщений в истории разговора, оставляемых без резюмирования`
- Problem: «резюмирование» is not the product's word for what compaction does: shipped ru renders it as «сводка» (wire.settings.compaction.description «Когда беседа сжимается в сводку…», compaction_enabled.help «Сжимать старую историю в сводку…»), so one mechanism gets two vocabularies in the same row family.
- Suggested: `[УСТАРЕЛО — заменено сжатием] Число последних сообщений в истории беседы, которые не сжимаются в сводку (103 cells)`

### M3 — MINOR — `cli.config.list_desc_model_name`

- Current: `Модель для общения`
- Problem: en is «The specific model to use for interactions»: «общение» narrows a model that also drives tool use and drops «specific»; shipped settings copy says «Модель для новых бесед» (wire.settings.model_name.help), so a bare «для общения» reads looser than the product.
- Suggested: `Конкретная модель для взаимодействий (36 cells)`

### M4 — MINOR — `cli.config.running_sessions_unchanged`

- Current: `Запущенные сессии не изменились — запись в конфигурацию не может ослабить требования к одобрениям в них; введите /approvals auto в каждой сессии, где одобрения больше не нужны. Новые сессии запускаются в режиме auto.`
- Problem: Meaning and safety polarity are right (a write does not loosen anything, /approvals auto is per session, new sessions start at auto), but 216 cells against en 148 is +46 %, past the ru guide's 20-30 % band; «не может ослабить требования к одобрениям в них» and «где одобрения больше не нужны» (for «ungated») are the heavy parts. Register, not fit: no cell cap applies.
- Suggested: `Запущенные сессии это не затрагивает — запись конфигурации не ослабит одобрения в них; введите /approvals auto в каждой сессии, где одобрения не нужны. Новые сессии стартуют в режиме auto. (188 cells)`

### N1 — NIT — `cli.config.list_title`

- Current: `Настройки конфигурации`
- Problem: Fine as a heading, but the same CLI's help line renders «Configuration Options» as «параметров конфигурации» (cli.config_help.list); «Параметры конфигурации» is the same 22 cells and keeps one noun for the surface.
- Suggested: `Параметры конфигурации`

### N2 — NIT — `cli.config.running_sessions_unchanged`

- Current: `«запись в конфигурацию»`
- Problem: Shipped ru names the same referent «Запись конфигурации» (wire.settings.approvals.description); one shape for one concept is cheaper to read.
- Suggested: `запись конфигурации`

### N3 — NIT — `cli.config.list_desc_auto_save_conversation (+ the two history rows)`

- Current: `Автоматическое сохранение разговоров`
- Problem: «разговор» here, «беседа» in the shipped labels for the very same settings (wire.settings.auto_save_conversation.label «Автосохранение беседы», conversation_length.label «Длина беседы»). Both nouns ship elsewhere in the catalogue, so either is defensible — but these three rows describe those settings.
- Suggested: `Автоматическое сохранение бесед (and «в истории беседы» in the two rows above)`

### N4 — NIT — `cli.config.list_unrecognised`

- Current: `неизвестный ключ; программа его не использует`
- Problem: «nothing reads it» is about no setting consuming the key; «его не читает ни одна настройка» says that precisely and keeps the fragment shape the row label expects.
- Suggested: `неизвестный ключ; его не читает ни одна настройка (49 cells)`

## Judged and accepted

- Safety semantics of the `cli.config.running_sessions_unchanged` line are correct: a config write does not loosen a running session, `/approvals auto` is per session and verbatim, new sessions start at `auto`.
- The `/move --to` refusal register (lowercase clauses) matches shipped `tui.notices.*`; `<path>`, `<session>`, `<peer>` are translated inside kept angle brackets while `/move`, `--to`, `local` and the backticked shell command stay byte-identical.
- `tui.session_move.deadline_exceeded` never claims failure or success (see M5 for the one nuance).
- No stray English: the only Latin runs are kept identifiers, flags, provider names and `CLI`/`shell`.
