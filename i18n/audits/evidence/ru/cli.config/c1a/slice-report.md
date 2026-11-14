# Independent i18n audit — ru · Slice C1 (core CLI + TUI English paths), suffix **A**

- **Locale:** ru (Russian)
- **File group:** C1 core slice — `cli.config.*` (15 keys, `lop config` output) and `tui.session_move.*` (7 keys, `/move --to` refusals); 22 keys total
- **Audit date:** 2026-11-11
- **Auditor:** `i18n-auditor`, session `e45ca4d0d71c`, **actual served model `deepseek/deepseek-flash`** (from this session's harness model line; no fallback was used or requested)
- **Independence:** the translation was produced by a different agent, `openai/gpt-6.1-sol` (manager brief). This audit read the values only; nothing was edited, and no translation/MT service was used for comparison or back-translation.
- **Verdict:** **clean-with-findings** — BLOCKER 0 · MAJOR 0 · MINOR 5 · NIT 5 → ledger status **`passed`** (no blocker/major remains)
- **Second pass:** performed (this group is settings/error copy). Every safety-bearing line was re-read against its EN source, its context note and the shipped ru catalogue after the first pass; two findings (M4, M5) come from that second pass, and one first-pass candidate (dropping "Successfully" in `cli.config.updated`) was withdrawn as padding.

## Inputs and hashes

| Input | Path | SHA-256 |
|---|---|---|
| EN + context + notes | `…/3ef6295a5545/scratchpad/c1-prep/keys.json` | `ac35dc123a4241a3c1fdbd0f4f012636d9706695bacfb3601aa06eed2cae1ffc` |
| ru translation | `…/3ef6295a5545/scratchpad/c1-prep/ru.json` | `d691d505003ee5dec761e2e94bbb5ba3de4c0ce9f32d69d3147bc20f997ff047` |
| EN catalogue `cli.config` | `local_operator/i18n/catalogues/en/cli.config.json` | `40649e7419d144f7e8a550d5ae3c6d48defa7bb0b23f4a20ceac35f2b0fda5ac` |
| EN context `cli.config` | `local_operator/i18n/catalogues/en/cli.config.context.json` | `9831a776d2f179969666f2271f9bc53ae7de8a18b8567732ba81bc8e2ca90e78` |
| EN catalogue `tui.session_move` | `local_operator/i18n/catalogues/en/tui.session_move.json` | `f4dd682b3b50d64a4009eac5ac78a34f5734b5dd0a5ef379951065da326a913e` |
| EN context `tui.session_move` | `local_operator/i18n/catalogues/en/tui.session_move.context.json` | `09ed30a0b11969671c41e24fe255072c961fff1f90791a7b9f05cd998cbee573` |
| Style guide | `i18n/style-guides/ru.md` | `f20e1cafaf70578c17ce072c71725628808d228e4be5604329f8da30386f1fb5` |
| Glossary | `i18n/glossary.md` | `0eb34437a8c3d6f564f278ef957a603b1d9f51bb07d5ddae5ea8f37deba962f8` |

The 22 EN values in `keys.json` are **byte-identical** to the committed EN catalogues (checked key-by-key, 0 mismatches; key sets equal, order equal). The audit therefore judges the real source bytes, not only a scratch extraction. The ru target has no catalogue file in the tree yet (`local_operator/i18n/catalogues/ru/` has `cli.config_help.json` but no `cli.config.json` / `tui.session_move.json`), so the target bytes audited are the translator's scratch file.

## Commands actually run (this session, in `~/local-operator/.worktrees/i18n-c1`)

1. `python3 scripts/i18n/check.py --catalogues-only` → `check.py: catalogue checks: 0 problem(s)` (repo validator, baseline clean at this head).
2. `.venv/bin/python …/c1a-ru/verify.py keys.json ru.json` using **the repo's own ICU runtime** (`local_operator.i18n.runtime`: `message_arguments`, `parse_message`, `plural_selectors`, `render_message`) → `problem_keys 0`. It checks, per key: key presence/order, ICU parse of EN and ru, argument tuples (name **and** kind) equal, declared `context.placeholders` equal to the EN arguments, `{name}` token sets, backtick spans byte-identical, brace/backtick counts, plural selector sets (none of these 22 keys is plural/select), and an actual `render_message(…, "ru")` + EN render (no leftover braces).
3. `shasum -a 256` on the five input files above (table).
4. `.venv/bin/python -c "catalogues.catalogue_sha256('en', ns)"` for both namespaces → equal to the EN catalogue file hashes above (confirms the ledger's `source_sha256` basis).
5. `python3 scripts/i18n/ledger.py history-sha --locale ru --namespace {cli.config,tui.session_move}` → `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` (empty history, both).
6. `python3 scripts/i18n/ledger.py append --receipt … --expected-history-sha 4f53cda1… --dry-run` then `--apply`, per namespace → wrote `i18n/audit-history/ru/cli.config.json` (`14876b63f9ffe437c0e52626126962ce6beb7671f98163a4fde26527083be094`, history digest `52f08b39…`) and `i18n/audit-history/ru/tui.session_move.json` (`dbf55a78e058c07effc48b22ab1c7c666cd7f77cc5bed7534cab6bf00ebd3964`, history digest `195e44d8…`); both rows `status: passed`.
7. `python3 scripts/i18n/ledger.py validate --base origin/main` → `audit history: valid and append-only`; `python3 scripts/i18n/check.py --catalogues-only` → `0 problem(s)` and `python3 scripts/i18n/generate.py --check` → `162 artifact(s) byte-identical`, both re-run **after** the append.
8. Shipped-catalogue terminology sweeps (grep over `local_operator/i18n/catalogues/ru/`): `Умные подсказки`, `вывод`, `резюм|суммир|сводк`, `журнал`, `пир`, `/approvals`, `--to`, `взаимо…`, `conversation_length|detail_length|auto_save_conversation`, `model_name`.

## Mechanical result — all 22 keys pass

Key set/order equal, no empties, no non-strings, **no value left identical to English**, placeholder names and kinds exact, no plural/select branches, every backticked command byte-identical (including `<id>`, `<to>`, `<seconds>` inside `` `lop sessions move <id> --to <to> --wait <seconds>` `` and `--keep`, `--queue`, `--drop-session-roles`), `local` kept English, `{flag}`/`{code}`/`{reason}`/`{path}`/`{key}`/`{value}`/`{session}`/`{to}`/`{description}` exact, all messages parse and render in the repo runtime.

**Stray English:** none beyond kept-literal tokens. The only Latin runs in the ru file are `{path}`, `{key}`, `{value}`, `{description}`, `{reason}`, `{flag}`, `{code}`, `{session}`, `{to}`, `CLI`, `shell`, `local`, `auto`, `approvals`, `lop sessions move`, `sessions`, `move`, `to`, `id`, `seconds`, `wait`, `/move`, `--to`, `--keep`, `--queue`, `--drop-session-roles`, `--wait`, `/approvals auto`, and the five provider names. Operator acceptance ("no English anywhere") holds for this group.

**Width:** no `maxCells` cap is stated for any of the 22 keys ("no cell cap; one terminal line, may wrap"). The one numeric hint, `cli.config.list_title` ("under about 30 cells"), is met: 22 cells (`Настройки конфигурации`). Lengths are otherwise reported for the register finding M4 only.

**Glossary/style compliance checked:** session → сессия ✓; approvals → одобрения (never подтверждения) ✓; peer → пир ✓; compaction → сжатие ✓; DEPRECATED tag upper-cased in brackets ✓; «ёлочки» quotes ✓; em dash with spaces ✓; вы-register ✓; no Title Case, no «!», no politeness padding ✓; digits/Latin only ✓.

## Findings

### MINOR

**M1 · `cli.config.cleared`** — `Значение {key} удалено: снова используется значение по умолчанию`
«значение» appears twice in a 64-cell line (EN 44), and the fronted participle makes a one-line confirmation heavy. The shipped word for a setting key is «ключ» (`cli.config_help.edit_key`: «Обновляемый ключ конфигурации»).
→ `Ключ {key} очищен: снова действует значение по умолчанию` (56 cells)

**M2 · `cli.config.list_desc_detail_length`** — `[УСТАРЕЛО — заменено сжатием] Число последних сообщений в истории разговора, оставляемых без резюмирования`
«резюмирование» is not the product's word for what compaction does: shipped ru says compaction turns history into «сводка» (`wire.settings.compaction.description`: «Когда беседа сжимается в сводку…»; `compaction_enabled.help`: «Сжимать старую историю в сводку…»). Two vocabularies for one mechanism in the same settings family. (The `[УСТАРЕЛО — заменено сжатием]` tag itself is correct and matches `wire.settings.*.help` «Устарело. Заменено движком сжатия.».)
→ `[УСТАРЕЛО — заменено сжатием] Число последних сообщений в истории беседы, которые не сжимаются в сводку` (103 cells)

**M3 · `cli.config.list_desc_model_name`** — `Модель для общения`
EN is "The specific model to use for interactions". «общение» narrows a model that also drives tool use and drops "specific"; the shipped settings wording for the neighbouring setting is `wire.settings.model_name.help` «Модель для новых бесед». At 18 cells it is also the shortest row in the group against a 42-cell source.
→ `Конкретная модель для взаимодействий` (36 cells)

**M4 · `cli.config.running_sessions_unchanged`** — `Запущенные сессии не изменились — запись в конфигурацию не может ослабить требования к одобрениям в них; введите /approvals auto в каждой сессии, где одобрения больше не нужны. Новые сессии запускаются в режиме auto.`
Safety meaning is **correct**: running sessions are unaffected, a config write cannot loosen them, `/approvals auto` must be typed in each session, new sessions start at `auto`; the command and the mode word are verbatim. The finding is register, not meaning: 216 cells against EN 148 (+46 %, past the guide's 20-30 % band), and the two heavy segments are «не может ослабить требования к одобрениям в них» and «где одобрения больше не нужны» for "ungated". Not a fit defect — no cap is set for this line.
→ `Запущенные сессии это не затрагивает — запись конфигурации не ослабит одобрения в них; введите /approvals auto в каждой сессии, где одобрения не нужны. Новые сессии стартуют в режиме auto.` (188 cells; same three safety claims)

**M5 · `tui.session_move.deadline_exceeded`** — `перемещение не завершилось за отведённое время; повторный запуск `lop sessions move {session} --to {to}` покажет, на каком этапе оно находится`
Required meaning: the timed-out move reports an **unknown** outcome and how to find out. The current copy neither claims failure nor success (good), but «на каком этапе оно находится» is a present-tense claim that the move *is* at some stage — it reads as "still in progress", which is a claim the copy must not make; EN "where it got to" leaves completion open.
→ `… повторный запуск `lop sessions move {session} --to {to}` покажет, до какого этапа оно дошло` (139 cells)

### NIT

**N1 · `cli.config.list_title`** — `Настройки конфигурации` is defensible, but the same command's help line renders "Configuration Options" as «параметров конфигурации» (`cli.config_help.list`); «Параметры конфигурации» (22 cells, same width) keeps one noun for the surface.

**N2 · `cli.config.running_sessions_unchanged`** — «запись в конфигурацию» vs the shipped «Запись конфигурации» for the same referent (`wire.settings.approvals.description`). If M4 is taken, use the shipped shape.

**N3 · `cli.config.list_desc_auto_save_conversation`** (+ the two history rows) — «разговор» where the setting's own shipped label is «Автосохранение беседы» (`wire.settings.auto_save_conversation.label`) and «Длина беседы» (`conversation_length.label`). Both nouns ship elsewhere in the catalogue (`разговор` in `cli.sessions_help.*`, `беседа` in `relay.chrome.*`), so either is defensible; pick one for these three rows rather than splitting inside one settings family.

**N4 · `cli.config.list_unrecognised`** — «программа его не использует» for "nothing reads it"; closer to the source and to the row it follows: «неизвестный ключ; его не читает ни одна настройка».

**N5 · `tui.session_move.to_needs_device`** — «Для --to нужно указать устройство: …» opens with a preposition in front of a Latin flag; «--to требует устройство: /move --to <пир|local>» reads cleaner and is 10 cells shorter. `local` stays English and `<пир|local>` keeps the bar either way.

## Judged and accepted (no finding)

- `cli.config.running_sessions_unchanged`: `/approvals auto` verbatim, `auto` kept as the mode word, «Новые сессии запускаются в режиме auto» — correct reading of "new sessions open at auto".
- `cli.config.list_desc_classification_notice`: «Умные подсказки» matches the shipped feature name (`wire.settings.classification.title` / `classification_notice.help`); «не выводят сообщений в чат» and «журнал сессии» are accurate; the `[УСТАРЕЛО]`-only tag matches the EN's missing "superseded" clause.
- `cli.config.list_desc_max_learnings_history`: «сохраняемых выводов» matches the shipped «Макс. история выводов» (`wire.settings.max_learnings_history.label`).
- `cli.config.updated`: dropping "Successfully" is correct ru practice for an impersonal receipt («Для {key} установлено значение {value}»); `{value}` stays untranslated as the note requires.
- `tui.session_move.*` refusal register: lowercase clauses match the shipped `tui.notices.*` voice; `<путь>`, `<сессия>`, `<пир>` translated inside kept angle brackets while `/move`, `--to`, `local` and the backticked shell command stay byte-identical — exactly what the context notes require.
- `tui.session_move.cannot_wait`: «выполните в shell» is a clarifying addition, not a meaning change; `shell` stays Latin per the ru guide.
- `cli.config.created` / `cli.config.opened`: «файл конфигурации» matches the shipped `cli.config_help.*` lines; the `: {path}` shape is idiomatic ru.
- No RTL, plural, markup/Rich-tag, command or flag issue exists in this group (ur is out of scope for this audit; nothing here carries rich markup).

## Ledger status — `passed`, both rows **applied**

Per locale + file group the result is **`passed`** (0 blocker, 0 major; unresolved blocker/major count 0), recorded with the sanctioned `ledger.py append` command:

| namespace | shard | shard SHA-256 | history digest |
|---|---|---|---|
| `cli.config` | `i18n/audit-history/ru/cli.config.json` | `14876b63f9ffe437c0e52626126962ce6beb7671f98163a4fde26527083be094` | `52f08b3981ba857c8091885e8421326e73ddd73cf182c219cb00b6f4b57781fc` |
| `tui.session_move` | `i18n/audit-history/ru/tui.session_move.json` | `dbf55a78e058c07effc48b22ab1c7c666cd7f77cc5bed7534cab6bf00ebd3964` | `195e44d8c63b95b934b4e45a9837b428afc900b6d82b77d39120737c2c0e5812` |

`source_sha256` in both rows is `catalogues.catalogue_sha256('en', ns)` (verified equal to the EN catalogue file hashes above); the history digest before each append was the empty-history value `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.

**Why appended though the ru catalogue is not in the tree.** `docs/I18N.md` orders catalogue-then-receipt (step 4 writes the locale files, step 6 appends the receipt), so my first decision was to hand the rows over unapplied. I changed it on evidence, not preference: the es lanes' rows for these **same two namespaces** already existed in the shared worktree with the same `source_sha256` basis, i.e. this wave's convention is to append at audit time and land the catalogue in the same PR. Applying is safe here and I checked it rather than assumed it — with the rows in place `check.py --catalogues-only` reports `0 problem(s)`, `generate.py --check` reports `162 artifact(s) byte-identical` (no generated drift), and `ledger.py validate --base origin/main` reports `audit history: valid and append-only`. What the rows do **not** yet bind is `target_sha256`; that is measured from the ru catalogue when it lands in this PR.

**Projected** target hashes from the audited values, using the generator's serialisation (`json.dumps(indent=2, ensure_ascii=False)` + trailing newline, order = EN order — verified by round-tripping `en/cli.config.json` to a byte-identical re-serialisation): `cli.config` → `1480bc908eb4bc56c943c8d79e295db33b0ca12f451e66b446f1d57d49bcaddd`, `tui.session_move` → `8bb32ae7520210d403853227f3da50dc689accc45632d68d80f850fb8afb9228`. **These are projections, not measurements** — the applier must recompute them from the written catalogue files.

## Limitations

- Independent LLM review, not native-speaker certification; the values were judged by reasoning about the code path and the shipped Russian catalogue, with no translation/MT service.
- Scalar text evidence only. Body text is CLI/TUI lines: no terminal or rendered frame was produced for this group, so this is not rendered-layout proof. There is no `maxCells` cap to violate and the only numeric hint (title ≤ ~30 cells) is measured at 22.
- The target bytes are the translator's scratch file; the committed ru catalogue bytes do not exist yet, so the ledger rows bind the EN source now and `target_sha256` is measured at landing (above).
- `check.py --audit-base origin/main` (the whole-tree AST ratchet) was not run locally: per the quality gates that scan is CI's, and my local run hit the 120 s budget on the ratchet walk with no output. `check.py --catalogues-only`, `generate.py --check` and `ledger.py validate --base origin/main` were run and are green.
- The mechanical verifier imported the worktree's own `local_operator.i18n.runtime` (confirmed by printing `runtime.__file__` under the worktree `.venv`), so ICU parity is checked against the tree's parser, not the installed `lop` generation.
