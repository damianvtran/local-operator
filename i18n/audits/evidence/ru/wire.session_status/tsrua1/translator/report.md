# Russian terminal-status translation handoff

Status: **translated, NOT audited**. 17 keys delivered: 16 new whole messages and the existing retired_label value retained unchanged.

Actual translator: i18n-translator; session 3f1cf5b56ba0; model openai/gpt-6.1-sol (host session-state declaration; no fallback used).

The frozen English and context hashes match the task. Candidate SHA-256: `b15cc559652777694f2e252404862095bbdc902f90fe72fcee930a03b698f47c`.

## Mechanical execution

Existing repo catalogue checker on the isolated frozen EN/RU namespace fixture: `check.py: catalogue checks: 0 problem(s)`; exit 0. It checks the candidate, not the entire repository. Extra assertions verify exact key order, placeholder kinds and select-arm identifiers, all four Russian CLDR count arms, preserved legacy value, valid JSON and absent manual bidi controls. Branch interpolation covers both command arms (including literal `/stop --all`), all cause arms, and 10 plural samples. See mechanical-check.log and reproducible check_candidate.py. This is NOT visual/app-runtime clearance or an independent audit.

`ledger.py history-sha` plus `ledger.py append --receipt receipt.json --expected-history-sha <recorded> --dry-run` exited 0 and named `i18n/audit-history/ru/wire.session_status.json`. No history append was applied. The receipt status is translated, never passed. Hash is optimistic: re-read history before applying if another writer appends.

## Native decisions / ambiguities

- unseen: Use «(не просмотрено)» for neuter impersonal outcomes and «(не просмотрена)» with «Ошибка». It means the existing result has not been viewed, never a new interruption/error.
- retired: Use «для обновления» explicitly, not operator stop; legacy retired_label is preserved byte-for-byte as a value.
- signal vs forced kill: «процесс остановлен сигналом» versus «процесс принудительно завершён». The unknown-command sentinel none emits no attribution; real command bytes follow «по команде» unchanged.
- dormant wakes: «пробуждение приостановлено» and inflected Russian count arms distinguish dormancy from cancellation or an active firing schedule.
- disposed: «сессия окончательно закрыта» avoids asserting that stored session data was deleted; source says disposed, not deleted.
- install replaced: «установленная копия заменена» refers to the installation, not the update operation itself.

No fixed length caps occur in this context sidecar; all values remain one logical line. No string was truncated.

## Glossary proposals

- turn → ход
- dormant wake → приостановленное пробуждение
- stall bound → предел простоя
- progress bound → лимит ожидания прогресса
- session disposed → сессия окончательно закрыта

## Remaining gates / known programme gap

Two independent safety-copy audits are required. No self-audit performed. Unknown external diagnostic English legacy branch stays preserved outside the namespace: an unresolved programme gap, NOT an allowed exception. No rendered or runtime clearance claimed. No repository/generated/history writes, commits, PRs, builds, installs or servers.

All source, glossary/style, validator, runtime, target and companion hashes are in provenance.json. These are temporary translation handoff artifacts, not final tracked audit evidence.
