# ES — second independent terminal-status safety-copy audit

Date: 2026-11-19. Verdict: **passed**, 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT. Findings: none.

Auditor: i18n-auditor / session 191daf68e868 / openai/gpt-6.1-sol (actual session runtime Model header; no fallback observed). Independent of translator b39246e0d986 and first auditor f9eeb42530ec. Same model does not mean same agent/session.

Frozen worktree: /Users/damian/local-operator/.worktrees/i18n-terminal-status @ 2930492e05a6968a2e1adfe34a091b94802bacd0.

## Sequence and reasoning (audit_notes_not_execution_log)
Read locale guide and glossary, complete English source/context, then all target values before opening either first receipt. Pre-receipt conclusions preserved in independent-pre-receipt-review.json. Neutral terse Spanish; sin leer marks unreadness rather than causation. Retirada para actualizar describes update-purpose interruption, not operator action. Señal and finalizada por la fuerza remain distinct. Inactivo does not claim a wake was cancelled or will fire. Sesión descartada does not assert data deletion. Runtime update failure retains the loaded build. All cause arms, including fallback, preserve the source meaning.

## Every-key coverage
- `wire.session_status.retired_label`: PASS. Current: {base} — retirada por una actualización
- `wire.session_status.interrupted`: PASS. Current: Interrumpida
- `wire.session_status.unseen_interruption`: PASS. Current: Interrupción sin leer
- `wire.session_status.retired`: PASS. Current: Interrumpida — retirada para actualizar
- `wire.session_status.unseen_retirement`: PASS. Current: Interrupción sin leer — retirada para actualizar
- `wire.session_status.signal_stop`: PASS. Current: Interrumpida — detenida con una señal{command, select, none {} other { mediante {command}}}
- `wire.session_status.unseen_signal_stop`: PASS. Current: Interrupción sin leer — detenida con una señal{command, select, none {} other { mediante {command}}}
- `wire.session_status.forced_kill`: PASS. Current: Interrumpida — finalizada por la fuerza{command, select, none {} other { mediante {command}}}
- `wire.session_status.unseen_forced_kill`: PASS. Current: Interrupción sin leer — finalizada por la fuerza{command, select, none {} other { mediante {command}}}
- `wire.session_status.complete`: PASS. Current: Completada
- `wire.session_status.unseen_completion`: PASS. Current: Finalización sin leer
- `wire.session_status.recent`: PASS. Current: Reciente
- `wire.session_status.dormant`: PASS. Current: Detenida ({count, plural, one {# despertar inactivo} many {# de despertares inactivos} other {# despertares inactivos}})
- `wire.session_status.error`: PASS. Current: Error
- `wire.session_status.unseen_error`: PASS. Current: Error sin leer
- `wire.session_status.error_cause`: PASS. Current: {cause, select, user_stop {Error — detenida por el usuario} runtime_stall_bound {Error — el runtime se cerró al agotar su límite sin avances} continuation_limit {Error — límite de continuación alcanzado; mensaje pendiente descartado} output_limit {Error — límite de salida del modelo alcanzado; no se pudo continuar} runtime_retired {Error — retirada para actualizar} runtime_shutdown {Error — el runtime terminó durante el turno} runtime_overdue {Error — el runtime salió para actualizar al agotar su margen de progreso} runtime_update_failed {Error — la actualización no terminó; el runtime conservó la compilación cargada} runtime_killed {Error — el runtime desapareció sin un cierre limpio; no se solicitó detenerlo} install_mid_update {Error — instalación reemplazada durante el turno} owner_lost {Error — el runtime dejó de responder durante el turno} disposed {Error — sesión descartada durante el turno} other {Error}}
- `wire.session_status.unseen_error_cause`: PASS. Current: {cause, select, user_stop {Error sin leer — detenida por el usuario} runtime_stall_bound {Error sin leer — el runtime se cerró al agotar su límite sin avances} continuation_limit {Error sin leer — límite de continuación alcanzado; mensaje pendiente descartado} output_limit {Error sin leer — límite de salida del modelo alcanzado; no se pudo continuar} runtime_retired {Error sin leer — retirada para actualizar} runtime_shutdown {Error sin leer — el runtime terminó durante el turno} runtime_overdue {Error sin leer — el runtime salió para actualizar al agotar su margen de progreso} runtime_update_failed {Error sin leer — la actualización no terminó; el runtime conservó la compilación cargada} runtime_killed {Error sin leer — el runtime desapareció sin un cierre limpio; no se solicitó detenerlo} install_mid_update {Error sin leer — instalación reemplazada durante el turno} owner_lost {Error sin leer — el runtime dejó de responder durante el turno} disposed {Error sin leer — sesión descartada durante el turno} other {Error sin leer}}

## Execution and controls
Scoped existing repo validator functions `_arguments_problem` and `_check_plural_categories` used with all 17 source/target values; valid/duplicate-free JSON and exact keys checked. Independent parsed-AST guard compares every select arm to English and requires the complete locale CLDR set and a count in every plural arm. Missing-arm negative controls demonstrate the known repository admission gap and independent rejection. Placeholder rename, unsupported category, malformed ICU, dropped key and renamed cause controls reject. See audit.log and validate.py.txt for actual successful execution and per-output evidence.

64 formatter outputs per locale inspected: all 17 keys; both complete 13-arm cause selects and unknown fallback; command none, /stop --all, /kill in signal/forced seen/unseen; plural 0,1,2,4,5,11,21,22,25,1000000,1500000,1.5,2.5,1.0. Locale categories: one/many/other. Legacy retired_label bytes equal frozen catalogue; catalog._interruption_label uses the new complete keys, not legacy retired_label. No hard cell/character hint; all outputs one logical line. No CJK/RTL-specific requirement for this locale. No backtick/markup literals occur in this group; command values remain exact.

First receipt verified only after independent semantic verdict: `/Users/damian/.local-operator/sessions/f9eeb42530ec/scratchpad/es-first/receipt.json`, SHA256 `1c1fe357ffe269869169355b49b9127c4e9f1ec838df2ea930fea0f61e1e295c`. Its linked record SHA256 `0bfd56772aa519f899fdbe484f3a463e2c1e631a6bd62a6a8c473fbf24f9907d` binds exact English/context/target hashes and independent identities. Prior clean verdict did not establish this verdict.

## Limits and retention
Unknown external diagnostic legacy English remains unresolved. No visual/native/integrated-runtime/pseudo/activation/release clearance. No fixed cell cap in sidecar; downstream rendered fit not cleared.

Harness setup correction: the first negative-control run removed the final literal parenthesis, not an ICU delimiter, and correctly remained parseable. The harness assertion failed. Corrected to append an unmatched opening brace, reran the complete audit successfully. This paragraph is audit_notes_not_execution_log, not preserved raw failure output. Decimal(1.0) selects other but formats as 1 in the existing formatter (same-code source fmt._operands retains scale; format_number strips zeros). This yields a synthetic mismatched visible number/noun, not a target CLDR-arm defect. Nonintegral 1.5/2.5 render correctly; production wakes=len(schedules), catalog.py:1554, is integral. Not global numeric-formatter clearance.

Requested virtual guide guide://i18n/HOWTO was unavailable; authoritative AGENTS.md points to docs/I18N.md, whose audit/append procedure was read instead. No MT/API/service used. No target, repository, history, build, install, server or PR modifications. This bundle is a scratch handoff, NOT claimed tracked/on main. Coordinator must promote its bytes to `i18n/audits/evidence/es/wire.session_status/ts2er/` before applying the prepared append-compatible receipt; canonical artifact references below are intended destinations. No history apply authorized/performed.
