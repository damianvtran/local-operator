# Russian session-host errors — second independent semantic audit

Date: 2026-10-31 UTC. Audit ID: host3222b.
Auditor: i18n-auditor (lopdev, host-ru-audit-b); session df0e9229eed3; actual model openai/gpt-6.1-sol (quota fallback, full provenance in record.json).
Translator: session bd43602aa8db; actual model openai/gpt-6.1-sol per sibling provenance.json. Same model, different independent session; auditor did not author/edit candidate.
Source revision: 9c979a03d551020ce6ff1b16c8066aa69ce7bb2a.
English SHA-256: b7d6561be15577c118ec333586f7c809e65471c5dfd0dd839abee700b2ed1ba0
Context SHA-256: d26acd5a6357a0c54bbd5ed409b4ad2c060ee52b9bd76670679397c4b464f724
Russian SHA-256: 232cdecfca4bd8018b47046cfc9ca0060980d476fb8fa01759ddb6b6d9192983

## Verdict: passed

All three keys independently reviewed. No BLOCKER, MAJOR, MINOR or NIT findings. Unresolved blocker/major count: 0. No suggested changes.
Blindness: no audit-a records or first-auditor findings read. Russian style guide and glossary read before committed English/context, then candidate and sibling translator provenance/validation.

## wire.errors.session_host.unsupported — PASS

Current: На этом компьютере нельзя запустить фоновую службу сессий, поэтому сессия не запущена. Используйте компьютер с поддерживаемым диспетчером фоновых служб.

All three propositions are preserved: this machine cannot run the background host, the session did not start, and the remedy is a machine with a supported service manager. «На этом компьютере нельзя запустить» describes capability in this explicit platform-support context, not a retry promise. «фоновая служба сессий» accurately describes a local host process rather than a network computer; «диспетчер фоновых служб» distinguishes the service manager. «сессия» matches the glossary. Professional, impersonal Russian; no defect or replacement wording.

## wire.errors.session_host.unavailable — PASS

Current: Фоновая служба сессий не ответила, поэтому сессия не запущена. Подождите немного и попробуйте снова.

«не ответила» accurately preserves the unanswered-host condition without inventing a shutdown or permission error. «сессия не запущена» explicitly denies startup. «Подождите немного и попробуйте снова» instructs a manual retry after waiting, never automatic progress or reconnecting. Natural professional register, consistent terminology and no politeness padding. No defect or replacement wording.

## wire.errors.session_host.refused — PASS

Current: Фоновая служба сессий отклонила запрос на запуск этой сессии. Сессия не запущена. Прежде чем повторить попытку, проверьте logs/session-host.log в папке настроек Local Operator.

«отклонила запрос на запуск этой сессии» preserves refusal to start this particular session without inventing a cause. The separate «Сессия не запущена» makes terminal failure explicit. «Прежде чем повторить попытку» preserves the requirement to inspect the log first. logs/session-host.log and Local Operator remain byte-identical; «папка настроек» accurately identifies the config folder. No false reconnecting or running-session implication. No defect or replacement wording.

## Mechanical validation — actual execution

The auditor freshly ran the repository validator in the existing translator validation-tree without modifying it. Committed checker, English/context and candidate snapshot bytes were verified first. audit.log captures the command, timestamp, exit code 0 and actual output: `check.py: catalogue checks: 0 problem(s)`.

Independent assertions: valid JSON, exactly three identical key IDs, no placeholders/ICU/select/plural structures in either language (CLDR category expansion not applicable), no markup/Rich tags/backticks, preserved literal path and product name, NFC and no Cf controls. No commands/flags appear. Russian lengths are 152, 100 and 176 characters respectively; these are static counts, not rendered dimensions. Context declares no maxCells/maxChars. No rendered-fit proof or layout sign-off is claimed. No RTL requirement for ru and no bidi controls found.

## Scope and retention

This is semantic/static copy auditing, not UI rendering, full-suite QA or release approval. report.md is audit notes, not an execution log. audit.log records actual execution. Translator validator.log was read as provenance, not substituted for the fresh validator run.

No MT services, translation edits, commits, shared ledger writes, installs, copied trees or full suites. A transient eval-kernel eviction interrupted artifact creation before any record was written; state was reconstructed from original inputs without inventing execution evidence.

Future tracked canonical root: i18n/audits/evidence/ru/wire.errors.session_host/host3222b/. Currently scratch only at explicit manager request, not tracked or on main. Integrator must retain bundle byte-identically. ledgerproposal.json is a proposal, not an applied ledger write. This new report is original, not a historical recreation or backdated replacement.
