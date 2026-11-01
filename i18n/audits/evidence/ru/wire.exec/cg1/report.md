# ru / wire.exec: core catalogue gap audit (cg1)

17 keys, all judged by the round-1 auditor against the English source and context sidecar; 2 audit round(s). Final locale verdict: **PASS**.

## This namespace
- **RU-018** (round 1, MINOR) `wire.exec.waiting_on_you`
  - English: 'Waiting on you? lop sessions shows what a run needs'
  - Value at audit: 'Ждёте ответа? lop sessions показывает, что нужно запуску'
  - Problem: Reverses who is waiting: English asks whether the RUN is waiting on the user; «Ждёте ответа?» asks whether the user is waiting for an answer.
  - Suggested: 'Запуск ждёт вас? lop sessions показывает, что ему нужно'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)
- **RU-019** (round 1, NIT) `wire.exec.exec_epilog`
  - English: "Examples:\n  lop exec 'Review the change' --team release --background\n  printf 'Inspect this report' | lop exec --profile reviewer\n  lop exec --goal 'Finish the checklist' --loop 3 --name 'Night audit'\n  lop exec --resume SESSION_ID --loop-goal 'Verify every acceptance criterion'\n  lop exec --status JOB_ID\n\nResume restores the transcript, team, profile, goal and name first; explicit\nstartup flags override their own slots. Team and profile may coexist. A loop\nruns after the optional initial prompt; --loop N counts continuations only.\nSaved loop progress is visible on resume, but iterations never replay automatically.\nA goal alone does not run a model. Prompts are literal, not slash commands.\n--goal only SETS the objective; unlike the TUI's /goal it does not also\nsend that text as a message, because the prompt is exec's message channel.\n--agent/--agent-id select legacy agent data and are mutually exclusive.\nDefault non-TTY approvals deny. --control may wait for a supervisor; --yolo\nis an explicit override, never implied by --background or --team.\n--tools bounds what the run can reach: excluded tools are unreachable.\nForeground events/text use stdout; receipts use stderr. Detached runs log\nboth streams and print distinct job/session IDs. Use lop --resume SESSION_ID\nto view a live run or resume a finished one; exec refuses a live owner."
  - Value at audit: "Примеры:\n  lop exec 'Review the change' --team release --background\n  printf 'Inspect this report' | lop exec --profile reviewer\n  lop exec --goal 'Finish the checklist' --loop 3 --name 'Night audit'\n  lop exec --resume SESSION_ID --loop-goal 'Verify every acceptance criterion'\n  lop exec --status JOB_ID\n\nСначала resume восстанавливает транскрипт, команду, профиль, цель и имя;\nявные флаги запуска переопределяют свои значения. Команда и профиль могут\nсосуществовать. Цикл выполняется после необязательного начального промпта;\n--loop N считает только продолжения. Сохранённый прогресс цикла виден при\nвозобновлении, но итерации никогда не повторяются автоматически.\nОдна цель не запускает модель. Промпты буквальные, не слеш-команды.\n--goal только ЗАДАЁТ цель; в отличие от /goal в TUI он не отправляет этот\nтекст ещё и как сообщение, потому что промпт — канал сообщений exec.\n--agent/--agent-id выбирают устаревшие данные агента и взаимоисключающи.\nОдобрения без TTY по умолчанию отклоняются. --control может ждать супервизора;\n--yolo — явное переопределение, которое никогда не подразумевается\n--background или --team.\n--tools ограничивает доступное запуску: исключённые инструменты недостижимы.\nСобытия и текст переднего плана идут в stdout; квитанции — в stderr. Отсоединённые\nзапуски пишут оба потока в журнал и выводят различные ID задания и сессии.\nlop --resume SESSION_ID позволяет следить за живым запуском или возобновить\nзавершённый; exec отказывает, если есть живой владелец."
  - Problem: «Одна цель не запускает модель» reads as 'one goal'; 'alone' is «сама по себе». «взаимоисключающи» (short form) is non-standard.
  - Suggested: '«Сама по себе цель не запускает модель.» · «…и исключают друг друга.»'
  - Disposition: fixed (remediation core-fix, session ef4f00143d56)

## Auditor reports (locale-wide excerpts, verbatim)
### Round 1 (session 4f28a4bcc8e2)
# ru core catalogue patch audit — independent
Verdict: **CHANGES_REQUESTED** — BLOCKER 0 · MAJOR 3 · MINOR 17 · NIT 5 · 1311 values in 56 namespaces
Auditor: i18n-auditor subagent, model anthropic/claude-opus-5-5 (no fallback). Independent of translator session 3ef6295a5545. No MT/translation service used. Read-only on the repo; inputs read from origin/main via `git cat-file --batch`.
## Blocking findings (must fix before `passed`)
- RU-008 tui.hub_links.action.enable: Glossary 'Wave-2 kept-English set' keeps `push`/`pull` Latin for hub operations, and the existing ru neighbour wire.slash.hub already says «auto-push». The patch coins «автопубликация» throughout (also hint.auto_push, policy.auto_push.label

## Could not verify
- No rendered terminal frames. The no-builds/no-tests constraint ruled them out, so widths are cell counts (East-Asian-width rule), not rendered.
- No context sidecar exists for tui.import, tui.library, wire.errors.agents/auth/files/media, wire.exec or wire.slash.library, and one key in tui.hub_links has none. For those, fit was judged by reading only.
- The repo validator (`scripts/i18n/check.py`) and the ledger write were not run: read-only instruction, and the patches are scratch files, not catalogue files yet. **No ledger row was recorded.** One should be written as `changes_requested` once the patches land in a tree.

### Round 2 (session 4586ac33eefc)
# ru core catalogue patches: independent round-2 audit
**Verdict: PASS.** BLOCKER 0, MAJOR 0, MINOR 4, NIT 4.
- **Auditor:** i18n-auditor, model `anthropic/claude-opus-5-5` as reported by the runtime. No fallback was observed.
- **Independence:** I did not write the translation, the round-1 findings, or the remediation.
- **Method:** I used no machine translation or translation service, and I edited nothing in the repo or the patches. English source and context sidecars came from `origin/main` through `git show`, extracted to `core-audit2/ru/en/`.

## Not verified

- **No rendered terminal frames:** this was a read-only audit with no builds or tests. Widths are computed cell counts.
- **ICU arm widths:** only one width-limited value contains ICU, so arm widths were checked by reading.
- **Missing context sidecars:** `tui.import`, `tui.library`, `wire.errors.*` (except sdk_*), `wire.exec` and `wire.slash.library` have none, so fit there was judged by reading.
- **Repo tooling and ledger:** I did not run `scripts/i18n/check.py` or write a ledger row. That follows the read-only instruction, and the patches are not yet catalogue files. **No ledger row was recorded.**
- **Untouched values:** I read the ~1,200 untouched values only for the sweep in (C), not for full accuracy.
