# Russian remediation audit — cli.serve_readiness / sd1r

Date: 2026-11-16. Auditor: independent-ru-auditor, session `b8c027b19533`, actual model `openai/gpt-6.1-sol`, per session runtime declaration; no fallback observed. Independent from original translator `acf0a84cd992` (manager identifies the same translator as remediation author).

Provenance reference: `i18n/audits/evidence/ru/cli.serve_readiness/sd1r/record.json`.

## Verdict

**passed — 0 unresolved BLOCKER/MAJOR; all three sd1 findings fixed, none open.** No new out-of-scope findings. This is the original independent auditor's remediation follow-up, not a second independent auditor or new full semantic audit.

## Retained coverage and bounded scope

Retain the complete 11-key semantic coverage in `../sd1/report.md` and its original input hashes in `../sd1/record.json`. English/context hashes are unchanged. The reproduction in mechanical.py.txt reverses only the three exact accepted replacements and compares the resulting UTF-8 bytes with the original Russian SHA-256. Equality proves all eight remaining keys and formatting are unchanged; no translator assertion is substituted for this check.

Independently re-read the three changed messages and their direct semantic neighbors: unproven, not_this_daemon, unknown_state, recorded_gap, unknown and other_unit. The retained install_unknown/stale assessments are unaffected: neither their text nor the build-comparison contract changed. The glossary and style decisions already read for sd1 remain the basis; policy hashes are checked unchanged. Source/context/current target and original record/report hashes are recorded in record.json.

## Finding disposition

### RU-SD1-01 — FIXED (was MAJOR) — cli.serve_readiness.not_reporting

Current: «зарегистрирован, но не отвечает (сигналы активности больше не поступают)».

Exact sd1 suggestion applied. This is native, factual Russian for missing heartbeat signals. It does not assert that the process died, that identity is proved, or that a different process owns the PID. Its combination with unknown/unproven diagnostics preserves the conservative distinction.

### RU-SD1-02 — FIXED (was MAJOR) — cli.serve_readiness.cgroup_unknown

Current: «работает в cgroup, которую эта проверка не может распознать».

Exact sd1 suggestion applied. The checker cannot recognize/classify the supervision context; the revised prose no longer describes a spatial “position.” It remains an unknown result, not a claim of absent supervision or proof that the process belongs to another service. Literal cgroup remains intact. The neighboring unknown/other_unit distinction is preserved.

### RU-SD1-03 — FIXED (was MINOR) — cli.serve_readiness.identity_restart

Current: «на {device} выполните `lop services status`, чтобы просмотреть записи, затем `lop services restart`, чтобы записать идентификационные данные процессов; перед перезапуском команда проверяет каждый демон».

Exact sd1 suggestion applied. Native imperative in the required вы-register; technical identity data rather than an unnatural abstract identity noun. Inspection still precedes the restart command; the text explicitly retains the per-daemon check before restarting. No unguarded action or guaranteed success is introduced. Both commands and the device placeholder are unchanged.

## Mechanical, fit and safety results

The actual validator and reproduction output is retained in audit.log. All 11 keys keep valid JSON, exact source placeholder names/kinds, intact commands/tags and valid ICU. The unchanged three plural messages retain one/few/many/other, exercised at the original representative counts. No maxCells/maxChars exists for these diagnostic strings; text can wrap, and no live wrapping or rendered-frame evidence is claimed. Russian remains LTR, with no combining-mark/bidi-control hazard detected. The edits introduce no permission/destructive-action polarity error.

## Handoff and provenance limits

No translation edits, commits, history appends, installs, full tests, live process probes or translation services. Existing sd1/sd2 bundles and initial failed ledger rows remain untouched. ledger-receipt.json is the append-ready passed row; audit.log includes its actual CLI dry-run, not an applied append. Parent owns append/commit. receipt.json is the bundle hash manifest.

All prose in this report is **audit_notes_not_execution_log**. Only captured command responses in audit.log prove execution; parser rendering/cell measurement is not a live-app visual test.
