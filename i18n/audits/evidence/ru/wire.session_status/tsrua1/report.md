# Russian terminal-status safety-copy audit — first independent pass

**Verdict: passed** — 17/17 keys, no BLOCKER/MAJOR/MINOR/NIT translation findings. Second independent audit remains pending.

Auditor: i18n-auditor / terminal-status-ru-audit; session `56bc997306a1`; actual model `openai/gpt-6.1-sol` from host model declaration; no fallback used. Independent from translator session `3f1cf5b56ba0`; no target edits. Audit date: 2026-11-19.

Frozen worktree HEAD: `2930492e05a6968a2e1adfe34a091b94802bacd0`.

## Input hashes

- en: `/Users/damian/local-operator/.worktrees/i18n-terminal-status/local_operator/i18n/catalogues/en/wire.session_status.json` — `52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326`
- context: `/Users/damian/local-operator/.worktrees/i18n-terminal-status/local_operator/i18n/catalogues/en/wire.session_status.context.json` — `5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525`
- ru: `/Users/damian/.local-operator/sessions/3f1cf5b56ba0/scratchpad/terminal-status-ru/wire.session_status.json` — `b15cc559652777694f2e252404862095bbdc902f90fe72fcee930a03b698f47c`

## Execution and limitations

Actual execution: `audit.py` invokes repository `scripts/i18n/check.py` through its normal `main(["--catalogues-only"])` entry point with the supported isolated catalogue-root seam. Normal candidate exits 0. Negative controls reject renamed placeholder, dropped/added key, malformed ICU, invalid Russian category and absent other (exit 1 each). Final restored candidate exits 0.

Measured validator gap: removing only Russian few is admitted (exit 0). This is a completeness gap in ordinary plural validation, not a defect in this target: independent AST inspection confirms one/few/many/other all present. No new blocker invented.

Rig correction disclosed: the first malformed-ICU mutation removed the final literal parenthesis, not the ICU closing brace, so admission correctly passed and our assertion failed. That execution is preserved in setup-control-mistake.log. Corrected mutation removes the outer ICU brace; repository admission rejects it. This was a rig mistake, not parser leniency.

No application-runtime, pseudo, visual, fit-in-layout, RTL or shaping proof claimed. ICU expansions below are mechanical branch/number samples only. No full suite/build/install/server/PR/history apply or repository write. No translation/MT service used. Initial invented guide URI was unavailable; authoritative guide://i18n plus current checked-out HOWTO/style/glossary were read.

## Mechanical and fit assessment

JSON valid; exactly 17 keys match source; placeholder names and kinds match; all command/cause select arms retain stable machine identifiers. No catalogue hardcoded commands/flags/backticks/Rich tags were introduced; command parameters survive exact interpolation and none omits attribution. Four CLDR categories manually verified. Legacy retired_label matches checkout bytes.

All strings are one logical line. Sidecar explicitly has no fixed cell cap and declares no maxCells/maxChars for these keys. Russian Cyrillic here is ordinary single-cell text; no combining controls or wide-glyph dependence. Long cause summaries may require existing wrapping/truncation but no unsupported overflow finding is asserted. Real terminal layout is untested. RTL is not applicable to ru.

## Full per-key and per-branch native audit

### `wire.session_status.retired_label`

EN: {base} — retired for an update

RU: {base} — из-за обновления

Assessment: PASS — Unchanged legacy suffix; cause is update, not user stop. Existing unused formatter scope is retained; no new English-base claim.

- `{"base": "Прервано"}` → Прервано — из-за обновления

### `wire.session_status.interrupted`

EN: Interrupted

RU: Прервано

Assessment: PASS — Impersonal neuter status is idiomatic: Прервано refers to execution, not a feminine session noun.

- `{}` → Прервано

### `wire.session_status.unseen_interruption`

EN: Unseen interruption

RU: Прервано (не просмотрено)

Assessment: PASS — Unread/view state is parenthetical, separate from interruption outcome; not described as a new failure.

- `{}` → Прервано (не просмотрено)

### `wire.session_status.retired`

EN: Interrupted — retired for an update

RU: Прервано для обновления

Assessment: PASS — Purpose для обновления explicitly distinguishes build retirement from operator stopping.

- `{}` → Прервано для обновления

### `wire.session_status.unseen_retirement`

EN: Unseen interruption — retired for an update

RU: Прервано для обновления (не просмотрено)

Assessment: PASS — Preserves update purpose and independently marks not yet viewed.

- `{}` → Прервано для обновления (не просмотрено)

### `wire.session_status.signal_stop`

EN: Interrupted — stopped with a signal{command, select, none {} other { by {command}}}

RU: Прервано — процесс остановлен сигналом{command, select, none {} other { по команде {command}}}

Assessment: PASS — Signal termination is explicit; no forced-kill claim. none has no connector; technical command preserved byte-for-byte.

- `{"command": "none"}` → Прервано — процесс остановлен сигналом
- `{"command": "/stop --all"}` → Прервано — процесс остановлен сигналом по команде /stop --all
- `{"command": "lop sessions stop audit-synthetic-id"}` → Прервано — процесс остановлен сигналом по команде lop sessions stop audit-synthetic-id

### `wire.session_status.unseen_signal_stop`

EN: Unseen interruption — stopped with a signal{command, select, none {} other { by {command}}}

RU: Прервано (не просмотрено) — процесс остановлен сигналом{command, select, none {} other { по команде {command}}}

Assessment: PASS — Same signal/command semantics; neuter не просмотрено agrees with impersonal execution outcome.

- `{"command": "none"}` → Прервано (не просмотрено) — процесс остановлен сигналом
- `{"command": "/stop --all"}` → Прервано (не просмотрено) — процесс остановлен сигналом по команде /stop --all
- `{"command": "lop sessions stop audit-synthetic-id"}` → Прервано (не просмотрено) — процесс остановлен сигналом по команде lop sessions stop audit-synthetic-id

### `wire.session_status.forced_kill`

EN: Interrupted — killed{command, select, none {} other { by {command}}}

RU: Прервано — процесс принудительно завершён{command, select, none {} other { по команде {command}}}

Assessment: PASS — Принудительно завершён names force, distinct from signal stop; masculine процесс agrees with завершён.

- `{"command": "none"}` → Прервано — процесс принудительно завершён
- `{"command": "/stop --all"}` → Прервано — процесс принудительно завершён по команде /stop --all
- `{"command": "lop sessions stop audit-synthetic-id"}` → Прервано — процесс принудительно завершён по команде lop sessions stop audit-synthetic-id

### `wire.session_status.unseen_forced_kill`

EN: Unseen interruption — killed{command, select, none {} other { by {command}}}

RU: Прервано (не просмотрено) — процесс принудительно завершён{command, select, none {} other { по команде {command}}}

Assessment: PASS — Force remains explicit; parenthetical unseen qualifier is separate; none attribution omitted.

- `{"command": "none"}` → Прервано (не просмотрено) — процесс принудительно завершён
- `{"command": "/stop --all"}` → Прервано (не просмотрено) — процесс принудительно завершён по команде /stop --all
- `{"command": "lop sessions stop audit-synthetic-id"}` → Прервано (не просмотрено) — процесс принудительно завершён по команде lop sessions stop audit-synthetic-id

### `wire.session_status.complete`

EN: Complete

RU: Завершено

Assessment: PASS — Завершено is idiomatic impersonal completion, not merely stopped.

- `{}` → Завершено

### `wire.session_status.unseen_completion`

EN: Unseen completion

RU: Завершено (не просмотрено)

Assessment: PASS — Completion is preserved and not-yet-viewed is separate; no new error implied.

- `{}` → Завершено (не просмотрено)

### `wire.session_status.recent`

EN: Recent

RU: Недавно

Assessment: PASS — Недавно is natural terse recency status rather than a success/error assertion.

- `{}` → Недавно

### `wire.session_status.dormant`

EN: Stopped ({count, plural, one {# wake dormant} other {# wakes dormant}})

RU: Остановлено ({count, plural, one {# пробуждение приостановлено} few {# пробуждения приостановлены} many {# пробуждений приостановлено} other {# пробуждения приостановлено}})

Assessment: PASS — Приостановлено conveys dormant/suspended wakes, not cancellation or a promise to fire. one/few/many/other all present. 1/21/101/1000001: пробуждение приостановлено; 2/22: пробуждения приостановлены; 0/5/11/55: пробуждений приостановлено. Fractions 0,5/1,5/2,5: пробуждения приостановлено, genitive singular with neuter singular predicate. Number formatting is mechanical ICU, not terminal proof.

- `{"count": 0}` → Остановлено (0 пробуждений приостановлено)
- `{"count": 1}` → Остановлено (1 пробуждение приостановлено)
- `{"count": 2}` → Остановлено (2 пробуждения приостановлены)
- `{"count": 5}` → Остановлено (5 пробуждений приостановлено)
- `{"count": 11}` → Остановлено (11 пробуждений приостановлено)
- `{"count": 21}` → Остановлено (21 пробуждение приостановлено)
- `{"count": 22}` → Остановлено (22 пробуждения приостановлены)
- `{"count": 55}` → Остановлено (55 пробуждений приостановлено)
- `{"count": 101}` → Остановлено (101 пробуждение приостановлено)
- `{"count": 1000001}` → Остановлено (1 000 001 пробуждение приостановлено)
- `{"count": 1.5}` → Остановлено (1,5 пробуждения приостановлено)
- `{"count": 2.5}` → Остановлено (2,5 пробуждения приостановлено)
- `{"count": 0.5}` → Остановлено (0,5 пробуждения приостановлено)

### `wire.session_status.error`

EN: Error

RU: Ошибка

Assessment: PASS — Ошибка is standard, glossary-compatible terse error label.

- `{}` → Ошибка

### `wire.session_status.unseen_error`

EN: Unseen error

RU: Ошибка (не просмотрена)

Assessment: PASS — Ошибка (не просмотрена) correctly uses feminine agreement and separates read state.

- `{}` → Ошибка (не просмотрена)

### `wire.session_status.error_cause`

EN: {cause, select, user_stop {Error — stopped by the user} runtime_stall_bound {Error — runtime ended itself after its stall bound} continuation_limit {Error — continuation limit reached; pending message dropped} output_limit {Error — model output limit reached; could not continue} runtime_retired {Error — retired for an update} runtime_shutdown {Error — runtime terminated during the turn} runtime_overdue {Error — runtime left for an update after its progress bound} runtime_update_failed {Error — update did not finish; runtime kept its loaded build} runtime_killed {Error — runtime disappeared without a clean exit; no stop requested} install_mid_update {Error — install replaced during the turn} owner_lost {Error — runtime stopped answering during the turn} disposed {Error — session disposed during the turn} other {Error}}

RU: {cause, select, user_stop {Ошибка — выполнение остановлено пользователем} runtime_stall_bound {Ошибка — рантайм сам завершил работу по достижении предела простоя} continuation_limit {Ошибка — достигнут лимит продолжений; ожидающее сообщение отброшено} output_limit {Ошибка — достигнут лимит вывода модели; продолжить не удалось} runtime_retired {Ошибка — работа прервана для обновления} runtime_shutdown {Ошибка — рантайм завершил работу во время хода} runtime_overdue {Ошибка — рантайм завершил работу для обновления, исчерпав лимит ожидания прогресса} runtime_update_failed {Ошибка — обновление не завершилось; рантайм сохранил загруженную сборку} runtime_killed {Ошибка — рантайм исчез без штатного завершения; остановка не запрашивалась} install_mid_update {Ошибка — установленная копия заменена во время хода} owner_lost {Ошибка — рантайм перестал отвечать во время хода} disposed {Ошибка — сессия окончательно закрыта во время хода} other {Ошибка}}

Assessment: PASS — All 13 stable arms including other reviewed individually below; no identifier translated.

- `{"cause": "user_stop"}` → Ошибка — выполнение остановлено пользователем
  - Native/meaning assessment: PASS — Explicit user agency: выполнение остановлено пользователем; distinct from update and unexplained disappearance.
- `{"cause": "runtime_stall_bound"}` → Ошибка — рантайм сам завершил работу по достижении предела простоя
  - Native/meaning assessment: PASS — Self-termination сам завершил работу and stall threshold предел простоя retained; no invented duration or operator blame.
- `{"cause": "continuation_limit"}` → Ошибка — достигнут лимит продолжений; ожидающее сообщение отброшено
  - Native/meaning assessment: PASS — Continuation cap plus pending-message loss both retained; отброшено is appropriately explicit, not softened to delayed.
- `{"cause": "output_limit"}` → Ошибка — достигнут лимит вывода модели; продолжить не удалось
  - Native/meaning assessment: PASS — Model output cap and inability to continue both retained.
- `{"cause": "runtime_retired"}` → Ошибка — работа прервана для обновления
  - Native/meaning assessment: PASS — Interrupted for update, not user stop or deleted history.
- `{"cause": "runtime_shutdown"}` → Ошибка — рантайм завершил работу во время хода
  - Native/meaning assessment: PASS — Runtime termination during turn; во время хода carries the turn context without claiming force or user action.
- `{"cause": "runtime_overdue"}` → Ошибка — рантайм завершил работу для обновления, исчерпав лимит ожидания прогресса
  - Native/meaning assessment: PASS — Update departure after progress bound: исчерпав лимит ожидания прогресса makes bound explicit without numeric duration; update reason retained.
- `{"cause": "runtime_update_failed"}` → Ошибка — обновление не завершилось; рантайм сохранил загруженную сборку
  - Native/meaning assessment: PASS — Update failure and continued loaded build both retained; сохранил загруженную сборку does not claim installation success.
- `{"cause": "runtime_killed"}` → Ошибка — рантайм исчез без штатного завершения; остановка не запрашивалась
  - Native/meaning assessment: PASS — Unclean disappearance plus absence of requested stop both explicit; no invented signal/user attribution.
- `{"cause": "install_mid_update"}` → Ошибка — установленная копия заменена во время хода
  - Native/meaning assessment: PASS — Установленная копия заменена during turn means installed copy replacement, not session/history deletion.
- `{"cause": "owner_lost"}` → Ошибка — рантайм перестал отвечать во время хода
  - Native/meaning assessment: PASS — Stopped answering during turn, not proof of killed/deleted process.
- `{"cause": "disposed"}` → Ошибка — сессия окончательно закрыта во время хода
  - Native/meaning assessment: PASS — Сессия окончательно закрыта describes lifecycle closure, not deletion of saved transcript/history/data; no удалить/данные claim.
- `{"cause": "other"}` → Ошибка
  - Native/meaning assessment: PASS — Generic error only; unseen counterpart keeps read-state qualifier.
- `{"cause": "unrecognized_synthetic_cause"}` → Ошибка
  - Native/meaning assessment: PASS — Unknown semantic cause selects other. This is not the intentionally unresolved external diagnostic path.

### `wire.session_status.unseen_error_cause`

EN: {cause, select, user_stop {Unseen error — stopped by the user} runtime_stall_bound {Unseen error — runtime ended itself after its stall bound} continuation_limit {Unseen error — continuation limit reached; pending message dropped} output_limit {Unseen error — model output limit reached; could not continue} runtime_retired {Unseen error — retired for an update} runtime_shutdown {Unseen error — runtime terminated during the turn} runtime_overdue {Unseen error — runtime left for an update after its progress bound} runtime_update_failed {Unseen error — update did not finish; runtime kept its loaded build} runtime_killed {Unseen error — runtime disappeared without a clean exit; no stop requested} install_mid_update {Unseen error — install replaced during the turn} owner_lost {Unseen error — runtime stopped answering during the turn} disposed {Unseen error — session disposed during the turn} other {Unseen error}}

RU: {cause, select, user_stop {Ошибка (не просмотрена) — выполнение остановлено пользователем} runtime_stall_bound {Ошибка (не просмотрена) — рантайм сам завершил работу по достижении предела простоя} continuation_limit {Ошибка (не просмотрена) — достигнут лимит продолжений; ожидающее сообщение отброшено} output_limit {Ошибка (не просмотрена) — достигнут лимит вывода модели; продолжить не удалось} runtime_retired {Ошибка (не просмотрена) — работа прервана для обновления} runtime_shutdown {Ошибка (не просмотрена) — рантайм завершил работу во время хода} runtime_overdue {Ошибка (не просмотрена) — рантайм завершил работу для обновления, исчерпав лимит ожидания прогресса} runtime_update_failed {Ошибка (не просмотрена) — обновление не завершилось; рантайм сохранил загруженную сборку} runtime_killed {Ошибка (не просмотрена) — рантайм исчез без штатного завершения; остановка не запрашивалась} install_mid_update {Ошибка (не просмотрена) — установленная копия заменена во время хода} owner_lost {Ошибка (не просмотрена) — рантайм перестал отвечать во время хода} disposed {Ошибка (не просмотрена) — сессия окончательно закрыта во время хода} other {Ошибка (не просмотрена)}}

Assessment: PASS — Every cause preserves the same meaning as the seen counterpart with feminine не просмотрена; no outcome/read-status conflation.

- `{"cause": "user_stop"}` → Ошибка (не просмотрена) — выполнение остановлено пользователем
  - Native/meaning assessment: PASS — Explicit user agency: выполнение остановлено пользователем; distinct from update and unexplained disappearance.
- `{"cause": "runtime_stall_bound"}` → Ошибка (не просмотрена) — рантайм сам завершил работу по достижении предела простоя
  - Native/meaning assessment: PASS — Self-termination сам завершил работу and stall threshold предел простоя retained; no invented duration or operator blame.
- `{"cause": "continuation_limit"}` → Ошибка (не просмотрена) — достигнут лимит продолжений; ожидающее сообщение отброшено
  - Native/meaning assessment: PASS — Continuation cap plus pending-message loss both retained; отброшено is appropriately explicit, not softened to delayed.
- `{"cause": "output_limit"}` → Ошибка (не просмотрена) — достигнут лимит вывода модели; продолжить не удалось
  - Native/meaning assessment: PASS — Model output cap and inability to continue both retained.
- `{"cause": "runtime_retired"}` → Ошибка (не просмотрена) — работа прервана для обновления
  - Native/meaning assessment: PASS — Interrupted for update, not user stop or deleted history.
- `{"cause": "runtime_shutdown"}` → Ошибка (не просмотрена) — рантайм завершил работу во время хода
  - Native/meaning assessment: PASS — Runtime termination during turn; во время хода carries the turn context without claiming force or user action.
- `{"cause": "runtime_overdue"}` → Ошибка (не просмотрена) — рантайм завершил работу для обновления, исчерпав лимит ожидания прогресса
  - Native/meaning assessment: PASS — Update departure after progress bound: исчерпав лимит ожидания прогресса makes bound explicit without numeric duration; update reason retained.
- `{"cause": "runtime_update_failed"}` → Ошибка (не просмотрена) — обновление не завершилось; рантайм сохранил загруженную сборку
  - Native/meaning assessment: PASS — Update failure and continued loaded build both retained; сохранил загруженную сборку does not claim installation success.
- `{"cause": "runtime_killed"}` → Ошибка (не просмотрена) — рантайм исчез без штатного завершения; остановка не запрашивалась
  - Native/meaning assessment: PASS — Unclean disappearance plus absence of requested stop both explicit; no invented signal/user attribution.
- `{"cause": "install_mid_update"}` → Ошибка (не просмотрена) — установленная копия заменена во время хода
  - Native/meaning assessment: PASS — Установленная копия заменена during turn means installed copy replacement, not session/history deletion.
- `{"cause": "owner_lost"}` → Ошибка (не просмотрена) — рантайм перестал отвечать во время хода
  - Native/meaning assessment: PASS — Stopped answering during turn, not proof of killed/deleted process.
- `{"cause": "disposed"}` → Ошибка (не просмотрена) — сессия окончательно закрыта во время хода
  - Native/meaning assessment: PASS — Сессия окончательно закрыта describes lifecycle closure, not deletion of saved transcript/history/data; no удалить/данные claim.
- `{"cause": "other"}` → Ошибка (не просмотрена)
  - Native/meaning assessment: PASS — Generic error only; unseen counterpart keeps read-state qualifier.
- `{"cause": "unrecognized_synthetic_cause"}` → Ошибка (не просмотрена)
  - Native/meaning assessment: PASS — Unknown semantic cause selects other. This is not the intentionally unresolved external diagnostic path.

## Findings and scope exclusions

No translation findings. Suggestions are not padded where current wording is native and safe. Stable machine arms and loaded-build/no-stop polarity are preserved. The unchanged retired_label and unknown external diagnostic legacy-English path are not newly localized; latter remains intentionally UNRESOLVED outside this scope. First pass is not two-audit completion.

## Retention

These are scratch handoff artifacts only, not tracked or merged evidence. Proposed canonical promotion path: `i18n/audits/evidence/ru/wire.session_status/tsrua1/`. Manager must promote the complete bundle and reconcile artifact paths/hashes before applying the receipt. No existing dated audit bundles or manifests were modified.
