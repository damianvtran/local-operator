# Spanish independent audit — cli.serve_readiness / sd3

Date: 2026-11-16. Revision: 1fc11dab96954dcc4c31abcc695c696baf2ead8c plus uncommitted remediation, exact inputs pinned in record.json.

**Verdict: passed. No open findings; zero unresolved BLOCKER/MAJOR.** This is the first audit by this auditor of the newly changed source/target contract, not a reuse of the old-source approval.

Auditor: Audit daemon Spanish; session e68b64992930; actual model openai/gpt-6.1-sol (host session state); no fallback recorded or observed. Independent of original translator session acf0a84cd992. Requester identifies current translation job 170903662c6e as the same translator; original translator actual model openai/gpt-6.1-sol is retained as requester-reported provenance, not newly measured provider telemetry. No translation/MT service used. No product source, translations or history edited.

## Coverage and retained evidence

Read current Spanish style guide and glossary, English source/context, all target values, contributor audit instructions, and the changed readiness.py grouping/remedy selection. Reviewed 7 changed keys and 4 new keys independently for accuracy, native register, terminology, punctuation, safety and fit. Four unchanged keys retain initial semantic approval and were checked again mechanically and as direct context. The prior HEAD target file SHA-256 matches sd1r's target hash. Each of the four retained JSON lines is byte-identical to that audited baseline; per-line/value hashes are in record.json and the real check output is in audit.log.

## Changed/new-key decisions

| Key suffix | Result and reasoning |
|---|---|
| unproven | PASS. “hay registro” is a neutral diagnosis that can apply to a grouped PID list. Explicitly says process identity cannot be verified and retains the unavailable start-time comparison. It does not ascribe the record's build to a live process. |
| not_reporting | PASS. “no hay respuesta” and “ya no se reciben señales de actividad” preserve both missing response and heartbeat cessation in idiomatic Spanish. Neutral with singular/plural PID headings. Prior ES-SD1-01 remains resolved. |
| not_this_daemon | PASS. “hay registro” avoids a singular-adjective mismatch under plural PID grouping. Another process now has that PID; no unsupported identity or build assertion. The singular “ese PID” is distributive within the shared diagnosis, matching source. |
| unknown_state | PASS. Compact neutral nominal phrase, “registro con estado desconocido”, works as a shared diagnosis, without turning unknown into failure. |
| recorded_gap | PASS. “el registro indica” explicitly attributes the build to the record, and “está instalada” identifies the installed build. No duplicate identity parenthetical; both opaque build placeholders remain intact. |
| unknown | PASS. Device and PID diagnosis are joined by spaced em dashes, not nested colons. one/many/other correctly agree in daemon, coincide/coinciden and supervisado/supervisados. Currentness and supervision remain unknown; not upgraded to success or fault. |
| identity_restart | PASS. Tú imperative, command unchanged. “pasar el daemon registrado a la compilación instalada” expresses migration onto installed code, not obtaining proof of identity. Pre-action checks remain explicit. |
| pid_reason | PASS. “identificador(es) de proceso” naturally localizes PID/PIDs as the context requests; pids remains opaque bounded numeric text. Singular/plural headings are correct and shared diagnosis is emitted once. No double colon. |
| and_more | PASS. “y # más” is natural for all counts; one/many/other all present; # is runtime-formatted, not preformatted or dropped. |
| identity_no_action | PASS. The subject is explicitly the unverified process-identity record, not the live daemon. Record matching the installed build is all it asserts. “para esta comprobación no obligatoria” confines the no-action advice to the non-required check, not the machine's overall health or all unknown causes. |
| identity_restart_manual | PASS. Manual restart of the literal lop serve command targets the installed build; “no puede recargarse en el mismo proceso” preserves the non-reloadable limitation and does not promise identity verification. |

## Four unchanged keys

install_unknown, cgroup_unknown, stale, other_unit: exact prior JSON-line and decoded-value bytes proved unchanged, retaining sd1/sd1r semantic coverage. Arguments, commands, plural branches and rendering were rechecked across all 15 current keys. The cgroup wording N1 is explicitly deferred by the requesting manager and is not reopened or silently represented as remediated.

## Source-selection consistency (inspection, not executed service behavior)

The inspected code increments unproven_current only for identity == unproven with a comparable installation stamp and a matching recorded build. It leaves the capability UNKNOWN/ok=False. A nonmatching record sets unproven_gap; its record-label clause cannot establish the live PID's build. With no stale neighbor, restart advice is selected only for that gap; identity_restart is used only when all gap records are reloadable, otherwise identity_restart_manual. Proven-stale neighbors already supply the single restart step. With no stale/held/other conflict, an unproven current record can receive the scoped no-action note. The Spanish strings preserve these distinctions. This source inspection does not claim that a service restart was exercised or that every mixed-row behavior was runtime-tested.

Grouping keys on complete diagnoses, including recorded build labels, retains PID-to-reason association; PIDs are capped before grouping and overflow is keyed. Pure render samples cover singular/plural grouped diagnoses, overflow, each remedy and million-category selection. No repeated identity parenthetical or double-colon structure occurs in the composed samples.

## Mechanical, fit and limits

Actual check.py --catalogues-only output: 0 problems. Independent strict JSON/key/context checks cover 15/15 keys. All argument names/types and branch-level arguments match English; all six Spanish plural messages carry one/many/other. Backtick commands, flags and any markup tags match source literal branches. All target strings are NFC with no bidi controls. Opaque values containing braces, markup-like text, a flag, backticks and a colon/path fragment are byte-preserved through every applicable branch by the pure ICU renderer. This does not certify caller-side Rich escaping.

No maxCells/maxChars hints are declared. audit.log records per-key render cell/character widths for all branches. Long terminal diagnostics, especially the explicit record-scoped no-action sentence, need wrapping at narrow widths; no fixed-width button/tab cap or confirmed overflow defect is established here. No terminal screenshot, live layout, services, pytest, full tests, translation API or provider call was used. Spanish is LTR; Urdu-specific RTL/shaping checks do not apply. This audit is not the independently staffed second pass, a design review or release authorization.

## Evidence authority

Fresh sd3 bundle only. Existing sd1/sd1r/sd2 bundles, history and authority addenda remain untouched. audit.log and ledger-dry-run.log are actual captured output; this report is audit_notes_not_execution_log. ledger-receipt.json is append-ready and its <=280-character note links record.json. Dry-run only; parent owns append and commit. Files are presently uncommitted; no claim they are on main.
