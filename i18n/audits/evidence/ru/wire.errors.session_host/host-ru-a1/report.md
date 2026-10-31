# Russian session-host error audit — first independent pass

Date: 2026-10-31. PR #3222; pinned source 9c979a03d551020ce6ff1b16c8066aa69ce7bb2a.
Auditor: i18n-auditor / af28bb46bcc2 / actual openai/gpt-6.1-sol (fallback from Sonnet via Opus following quota notices). Translator bd43602aa8db is a different session; shared model is disclosed, not shared authorship.

Verdict: passed, 3/3 keys, no findings at any severity. This does not supply the separately required second high-visibility audit.

## wire.errors.session_host.unsupported — PASS
Current: На этом компьютере нельзя запустить фоновую службу сессий, поэтому сессия не запущена. Используйте компьютер с поддерживаемым диспетчером фоновых служб.

Cannot start on this computer is terminal; explicitly says the session was not started. Supported background service manager is preserved, without suggesting reconnecting. No wording change requested.

## wire.errors.session_host.unavailable — PASS
Current: Фоновая служба сессий не ответила, поэтому сессия не запущена. Подождите немного и попробуйте снова.

Host did not answer; explicitly no session started. Wait then manually retry preserves temporary unavailability; no automatic retry or reconnect. No wording change requested.

## wire.errors.session_host.refused — PASS
Current: Фоновая служба сессий отклонила запрос на запуск этой сессии. Сессия не запущена. Прежде чем повторить попытку, проверьте logs/session-host.log в папке настроек Local Operator.

Host declined the startup request; explicit non-start. Log inspection precedes retry. logs/session-host.log and Local Operator remain exact literals. No wording change requested.

## Mechanical, idiom, terminology, fit

Validator executed against existing isolated translator tree, read-only; exit 0 and 0 problems (audit.log). Independently pinned check.py, source, context and candidate bytes; other tree files not independently verified. Earlier broader verification attempts timed out before validator execution; no pass claimed for them. Exact three-key set; no placeholders, ICU branches, rich tags or backtick code in either language. Required path/product literal unchanged. NFC and no Cf controls verified. No MT services used.

Impersonal clauses, formal imperatives, sentence case and native punctuation conform. «Фоновая служба сессий» idiomatically describes a local hosting service, not a network computer; aligns with glossary service → служба and session → сессия. «Диспетчер фоновых служб» accurately describes the supported service manager. No unsafe start/reconnect claim or permission inference.

Terminal cells en/ru: unsupported 145/152; unavailable 110/100; refused 174/176. No maxCells/maxChars hints in context; these explanatory errors are not pinned button/status labels. No copy-only overflow finding; no rendered terminal/wrapping proof claimed. RTL inapplicable.

## Retention and recording

This report is audit_notes_not_execution_log; audit.log contains actual validator execution output. Bundle is scratch-only pending manager retention at the canonical path in record.json; no tracked/main claim. Shared worktree and ledger were explicitly off limits. Inspected RFC §6/current scripts expose ledger schema but no audit-ledger recording CLI; ledger-proposal.json is a proposal, not a write receipt.
