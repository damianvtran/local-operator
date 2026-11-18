# Urdu first audit of #4112 source/target remediation — sd3

**Verdict: passed.** No BLOCKER, MAJOR, MINOR or NIT findings in the assigned delta.

- Date: 2026-11-16. Namespace: cli.serve_readiness, 15 keys.
- Auditor: i18n-auditor / Audit daemon Urdu, session c736f176af6f, actual openai/gpt-6.1-sol from the session model declaration; no fallback reported.
- Translation: job 170903662c6e, described by the manager as the same original translator (original task/session acf0a84cd992, actual openai/gpt-6.1-sol). That model provenance is supplied by delegation, not independently inferred from catalogue authorship. Auditor and translator are separate sessions.
- Worktree: 1fc11dab96954dcc4c31abcc695c696baf2ead8c plus uncommitted remediation; input hashes in record.json bind the reviewed bytes.
- Coverage: all seven updated and four new keys independently reviewed against current English/context; four unchanged values retain the previous sd1r/sd1 coverage with exact UTF-8 value-byte equality proven against HEAD. This is the first audit of the new source hash, not a rebranding of sd1r, and not a separate second auditor.

## Mechanical checks — actual execution in audit.log

Run repository check.py --catalogues-only, then an independent narrow parser/render probe retained as mechanical.py.txt. All current 15 keys have duplicate-free valid JSON, matching key/context sets, argument names/kinds/order, matching ICU structure, unchanged backtick commands and markup. The six plural messages each include Urdu one/other; raw render inputs 0/1/2/12 exercise both branches and numeric overflow text. Opaque simple parameters containing braces, punctuation, paths, brackets and Latin IDs remain literal, not reparsed. No inserted bidi control, Urdu/Arabic digits, changed flags or identifiers.

The parser script proves exactly seven updated/four added/four byte-identical values in each English/Urdu catalogue against HEAD. It also renders grouped unproven-plus-gap diagnostics through pid_reason and unknown, including capped-list overflow. No doubled colon; identity uncertainty appears once, not repeated in recorded_gap. Cell/character counts are logged, with no maxCells/maxChars hints present. Raw parser output/cell counts do not constitute a terminal or browser RTL render.

## Every changed/new key — semantic and native-copy assessment

This table is **audit_notes_not_execution_log**. It records the auditor's reasoning, not proof of service execution.

| Key suffix | Change | Assessment |
|---|---|---|
| unproven | Updated | Accept. Neutral record-exists wording works after a PID/group label; inability to prove process identity and compare a start-time stamp are preserved. No assertion about a live process build. |
| not_reporting | Updated | Accept. Record exists but no response arrives; stopped heartbeat is retained without claiming proven process death. The diagnosis can be shared by a group without a pronoun pretending the PID itself answers. |
| not_this_daemon | Updated | Accept. Record exists; that PID is now held by another process. It does not infer that other process's version or supervision. PID token unchanged. |
| unknown_state | Updated | Accept. Short nominal diagnosis describes an unknown-state record; natural in the labelled/grouped diagnostic, without a dangling copula or duplicated subject. |
| recorded_gap | Updated | Accept. Explicitly the record contains build {recorded}, whereas {install} is installed. Not a claim the present process runs the recorded build. Identity uncertainty is not repeated in a parenthesis. Placeholder roles/order preserved. |
| unknown | Updated | Accept both branches. Em dash separates daemon heading from already labelled PID reasons, avoiding colon nesting. Uncertainty about currency/supervision remains explicit; singular/plural verbs match count, including a capped displayed list. |
| identity_restart | Updated | Accept. Respectful چلائیں; moves the recorded daemon onto the installed build. The command checks each daemon before acting. No promise that restart establishes identity, and the previous unnecessary identity-recording rationale is gone. |
| pid_reason | New | Accept both branches. عمل کا شناختی نمبر / اعمال کے شناختی نمبر supplies singular/plural process-ID wording; numeric {pids} stays opaque. Em dash cleanly joins a shared complete diagnosis. No claim a numeric PID establishes identity. |
| and_more | New | Accept both branches. اور مزید # is natural clipped overflow wording; number is formatter-rendered Latin digits and remains visible in both branches. |
| identity_no_action | New | Accept both branches. The statement explicitly concerns the build entered in record(s) whose process identity is unproven, not a verified running process. No action is required only for this غیر لازمی (non-required) check; it does not waive some unrelated required check or assert healthy supervision. Singular/plural records/build agreement is intelligible and accurate. |
| identity_restart_manual | New | Accept. Respectful دوبارہ شروع کریں; explicitly manual restart of literal lop serve, to move the recorded daemon to the installed build. Inability to reload itself in the existing process retains the in-place limitation; no automatic-reload/identity guarantee is added. |

## Four unchanged keys — retained coverage and direct neighbors

install_unknown, cgroup_unknown, stale and other_unit are byte-identical values in both English and Urdu to HEAD; HEAD's complete prior Urdu catalogue hash equals the sd1r target hash. Retain sd1r/sd1's semantic/glossary/RTL decisions, and mechanically parse/render their current values again. In particular stale/other_unit keep both grammatical branches, install_unknown keeps build-comparison uncertainty, and service/unit terminology stays consistent. The manager's cgroup N1 deferral is not reopened or edited in this delta audit.

## Selection context and conservative meaning

Read the new readiness.py selection, especially unproven-record comparison (3024–3054), grouped reasons (3140–3158) and remedies (3181–3217). These are code-reading observations, not executed daemon/branch tests:

- Unproven identity remains in unknown_reasons and continues before any proven-process verdict. A current record/install match increments unproven_current only; the Urdu no-action wording remains record-scoped.
- A recorded build gap selects guarded automatic restart only when all relevant gap records are reloadable; otherwise it selects the new manual instruction. A proven stale neighbor already supplies its movable/manual action, so the new identity step is not duplicated.
- pid_reason receives numeric PID lists and a complete shared diagnosis; the keyed overflow suffix remains localizable. The tested raw compositions use neutral record language and em dashes rather than two colons.
- No service was invoked and no claim is made that live restart branches, safety guards or identity probes were exercised. This audit checks copy against its inspected selection contract.

## Idiom, terminology, fit and RTL

Natural respectful Urdu developer-register wording; سیشن، ریموٹ، سروس، یونٹ and زیرِ نگرانی follow the guide/glossary. Commands stay byte-identical Latin. Urdu commas/semicolons and the source's neutral em dash are appropriate; no arrows or direction-dependent glyph hint to mirror. There is no untranslated English PID/PIDs overflow prose in the new keys.

Long record-scoped no-action/unknown sentences wrap as diagnostics; there are no fixed-width hints or new compact controls. Combining marks and mixed Latin device/build/PID/path tokens remain an integration risk for shaping/isolation, not a reason to inject manual controls. The catalogue contains none. Actual terminal RTL shaping, per-token visual order and rendered fit are not certified by this raw render audit.

## Retention and history

New record.json binds current en/context/ur inputs, supporting source/style inputs, actual identities, this report and actual logs. receipt.json hashes companions; ledger-receipt.json is the append-ready passed row with a record-linked note under 280 characters, CLI dry-run validated only. Parent owns append/staging/commit. All historical bundles and failed/passed ledger rows remain intact; no claim this uncommitted sd3 bundle is on main.
