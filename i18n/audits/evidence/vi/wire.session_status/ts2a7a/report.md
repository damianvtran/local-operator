# Vietnamese R2 — first auditor delta review

**Verdict: passed on current target `91fdc2a9fb34b9bd23efce8daabd33a74a7511ad67a9b269e711dd90c96b277e`.** No unresolved BLOCKER/MAJOR, no additional translation findings. This is the first auditor's current-hash R2 approval, not the second independent auditor's approval.

Actual auditor: `i18n-auditor`, task `terminal-status-vi-audit`, session `7a2548e2c7b1`, model `openai/gpt-6.1-sol` from session host Model header, no fallback. Independent of translator session `56c3029571f6`. No MT/translation service used. Frozen worktree HEAD remains `2930492e05a6968a2e1adfe34a091b94802bacd0`.

This document is **audit_notes_not_execution_log**. `vi-r2-first-audit.log` is actual execution output; `vi-r2-first-cases.json` records all 26 actual ICU text-formatting cases. No integrated runtime/visual/pseudo/shaping proof claimed.

## Scope and independent semantic conclusion

Only `runtime_overdue` bodies in `wire.session_status.error_cause` and `wire.session_status.unseen_error_cause` changed:

- Old body: `runtime rời đi để cập nhật sau ngưỡng tiến triển`
- New body: `runtime đã dừng để cập nhật sau khi hết thời hạn chờ tiến triển`
- Seen output: `Lỗi — runtime đã dừng để cập nhật sau khi hết thời hạn chờ tiến triển`
- Unseen output: `Lỗi — chưa đọc — runtime đã dừng để cập nhật sau khi hết thời hạn chờ tiến triển`

I independently inspected the cited source rather than treating the second auditor's finding as authority:

1. `local_operator/incidents.py:469–475` explicitly defines the historical cause as leaving for the newer build after a period **of no movement reported from in-flight work**. Thus this is expiry of a wait window, not attainment of a quantity of progress.
2. `local_operator/session/runtime/types.py:1166–1184` explains the historical bounded handover was forced and involuntary, distinguishable from ordinary retirement and deliberate user stop.
3. `local_operator/session/runtime/process.py:4106–4145` explains this historical force-cut was removed; current code abandons the attempted move and continues serving the loaded build. The historical incident cause must still describe its historical event, not claim the current deferral behavior occurred or a successor successfully loaded.

**VI-S2-01 is correctly remediated in both arms.** `đã dừng` removes the anthropomorphic `rời đi`; `sau khi hết thời hạn chờ tiến triển` naturally conveys the expiry of the allowed wait for progress. It does not claim progress was achieved, invent a numeric duration, attribute a user request, claim a successful new-build load, or turn the separate update progress bound into the general stall watchdog. `để cập nhật` remains purpose, not proof the update succeeded. The terse `runtime đã dừng` describes a stopped runtime without introducing intentional agency. Unread status remains a separate `chưa đọc` marker.

My first audit accepted the old wording too literally against the short English label. The incident's explicit **no-movement duration** clarifies the ambiguity the first pass missed. That original audit remains preserved, not rewritten or represented as current approval of the old body. This current-hash delta judgment is based on the actual source above, not blind agreement with the second auditor.

## Reverse-byte retention proof

`audit_vi_r2.py` independently asserted all input SHA-256 values. Replacing the new UTF-8 phrase with the old phrase in exactly its two occurrences in the complete new file restored the **entire old candidate byte-for-byte**. Additionally, AST comparison established only `runtime_overdue` changed within each cause select.

Accordingly, retain the original first audit's complete 17-key review and **58 original ICU cases** for all unchanged content. Do not claim all 58 were rerun for R2. Original `vi-first-record.json` and `vi-first-receipt.json` hashes were asserted unchanged during this run. All other content, including dormant counts, disposed session wording, legacy retired_label, command selects, fallback arms and remaining causes, is byte-identical.

## Actual R2 checks

- Cheap all-17 mechanical checks: exact source/target key set, placeholder name/type parity, complete stable select-arm sets, Vietnamese `other`-only CLDR and NFC. PASS.
- Actual repository `scripts/i18n/check.py::check_catalogues()` on isolated scratch en/context/vi fixture: `[]`. PASS.
- Actual malformed controls: missing key, renamed cause argument, invalid Vietnamese `one` plural arm rejected; restored candidate admission returned `[]`. PASS.
- 18 cause cases: seen/unseen × `runtime_overdue`, `runtime_stall_bound`, `runtime_retired`, `runtime_shutdown`, `runtime_update_failed`, `runtime_killed`, `user_stop`, `other`, synthetic unknown selector. Only the two intended overdue outputs differ from original; all direct neighbors and generic fallbacks match. PASS.
- 8 command cases: four seen/unseen signal/forced-kill keys × `none` and `/stop --all synthetic-opaque-123`. `none` omits attribution; opaque command bytes remain exact; every output matches the old candidate. PASS.
- New seen/unseen overdue labels measure 69/80 terminal cells through `rich.cells.cell_len`. Source sidecar has no fixed cell/character cap. No new combining/wide-glyph issue or overflow defect inferred without a rendered surface.

The first audit's observed validator limitations (missing select-arm and missing non-other CLDR enforcement) remain pre-existing, not a new target blocker. Independent arm/category checks verify this candidate is complete.

## Inputs

- English SHA-256: `52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326`
- Context SHA-256: `5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525`
- Original target SHA-256: `963b8bf337981a328bb80b265cb49d033db10c7afb2838c4ec8d5f05df20c1d1`
- R2 target SHA-256: `91fdc2a9fb34b9bd23efce8daabd33a74a7511ad67a9b269e711dd90c96b277e`

No source/context/target/history/repository/generated/build/install/server/PR edits. Receipt is append-compatible, **dry-run only**. Coordinator must retain the new R2 bundle under `i18n/audits/evidence/vi/wire.session_status/ts2a7a/`, with the original first bundle preserved, before applying history. Scratch is temporary handoff, not tracked evidence. Second independent current-hash approval remains separately required before promotion. Unknown external diagnostic legacy English remains intentionally UNRESOLVED outside this delta.
