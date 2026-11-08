# Second independent safety/high-visibility i18n audit — tui.profile_actions

**Verdict: PASS / TERMINAL for copy safety and semantic accuracy.** All 11 keys × 8 locales independently audited; 0 unresolved BLOCKER/MAJOR. Three non-gating punctuation NITs are recorded together below; they do not require a new round alone.

## Scope and provenance

- Auditor: designer acting as second independent safety i18n auditor; session `78d9864ffaac`.
- Actual served model: `openai/gpt-6.1-sol`, confirmed by the resume's host session-state. No model/tool/default changes by auditor. Previous blocked attempt recorded an Opus-quota fallback; no additional fallback used during this resume.
- Independent of coder/translator `4049c4f3fd43` and the other audit agent. Same actual model as author is disclosed. No other auditor judgments read.
- Manager-owned worktree: `~/local-operator/.worktrees/agent-save-consent`, committed frozen-copy HEAD `00542c8b657ec7b72482b8fd7bf9932c193d3814`; engine WIP outside the namespace was not reviewed.
- Root manifest: `/Users/damian/.local-operator/sessions/4049c4f3fd43/scratchpad/r3/copy-audit-manifest.json`.
- English source SHA-256: `8179f383fc12faa9fd44fa815c1a61bebe7fa1d07f0283a874a1146216e849e1`.
- All nine files' worktree and committed HEAD SHA-256 values match the primary manifest; exact hashes in `record.json` and `audit.log`.
- All 88 product strings were read by keyed `read_public_copy` calls; successful complete tool responses preserved in `guarded-reads.json`. A transient git-show timeout during a Chinese keyed read recovered; no raw catalogue read substituted for it.
- `read_public_copy` rejects the context sidecar's object-valued metadata as non-string catalogue values. Role/tone/notes are non-rendered metadata, so those were read normally; no product strings were read through that path. Sidecar hash is independently checked against HEAD and manifest.

## Acceptance contract

The operator's requested authorization distinction remains the backdrop, not a second engine/code audit: user's own requested profile change authorizes save; unsolicited change uses non-blocking exact-profile/change ask; peers and Team Bridge do not independently authorize mutation; show truthful changed-profile receipt and one-click revert; preserve backup/revert and unchanged model/tools/class unless requested. Approved discipline gate does not claim sandboxing.

The eleven strings cover only Revert availability, checking/reverting, read-only status reconciliation, success, stale/later-edit protection, unavailable verification and incomplete-outcome safe recovery. They must not conflate undo with consent denial, encourage blind retry, promise a universal backup command, or assert no change when outcome is unknown. The technical shortcut remains Latin `R` via `{shortcut}`; wire IDs remain English. Urdu isolation belongs at paint, not stored copy.

## Actual execution

Executed the repository's primary `.venv/bin/python` with `PYTHONPATH` pointing to the manager worktree, isolated scratch HOME/config, every inherited `CMUX_*` removed, and git noninteractive/system-credential-helper disabled. The pure, bounded script is preserved as `recheck.py.txt`; actual JSON stdout is `audit.log`, stderr `audit.stderr.log` (empty). No full test suite, installs, captures, native/browser/hardware process or provider calls.

Actual results: exit 0; 9/9 manifest/worktree/HEAD hashes identical; 11/11 context and catalogue keys per locale; 88/88 messages parsed with repository `runtime.parse_message`; argument parity and `runtime.render_message` output identity passed; NFC passed; 0 Unicode format controls in every string. Both shortcut-bearing messages per locale preserve exactly `{shortcut}` and render `(R)`. No plurals/selects occur, so no unused CLDR branch matrix is claimed. Malformed-ICU negative control rejected; wrong-placeholder negative control detected.

Context budget: eight short labels per locale must fit **52 Rich cells at the stated 60-column card layout**. Three reason strings wrap with no fixed maximum. Actual short-label maxima:

| Locale | Keys | Max label cells / 52 | Safety/semantic verdict |
|---|---:|---:|---|
| en | 11/11 | 28/52 | PASS |
| fr | 11/11 | 36/52 | PASS |
| es | 11/11 | 32/52 | PASS |
| zh-CN | 11/11 | 21/52 | PASS |
| ru | 11/11 | 31/52 | PASS |
| vi | 11/11 | 28/52 | PASS |
| ur | 11/11 | 23/52 | PASS |
| hi | 11/11 | 18/52 | PASS |

## Key-by-key safety assessment (each independently checked in all eight locales)

- `available`: The action means restore/undo this saved change, not deny consent or cancel an unrelated request. Label offers the action without claiming the action is already complete; actual authorization/enablement remains an engine concern.
- `checking`: An in-progress check, distinguishable from in-progress mutation; no completion guarantee.
- `reverting`: An in-progress restore/undo, not an already-completed result.
- `check_status`: A status inspection/check, not retry/reapply/revert; retains the technical shortcut placeholder. Suitable for lost-reply read-only reconciliation.
- `reverted`: Completed restore/undo, not merely available or requested. French/Hindi specify previous-version restoration, consistent with inverse context.
- `unavailable`: Restore action is not available/cannot be used; not a denial of the underlying requested profile change. Read together with unavailable_reason.
- `stale`: Profile changed; conveys stale state rather than completion. Compact ru/vi/ur/hi labels omit repeating Revert without losing profile-change meaning in receipt-control context.
- `failed`: Restore not completed/incomplete, leaving possible partial/unknown outcome; does not assert that nothing changed.
- `unavailable_reason`: Session cannot verify the action. Review the current profile before a new change request; no blind retry, unconditional backup command, or unauthorized-mutation advice.
- `stale_reason`: Profile changed after this save. Inspect current version; this action will not undo later changes. Negation and temporal scope retained.
- `failed_reason`: Could not complete restore; inspect the current profile before a new change request. No false no-change guarantee or automatic retry.

## Locale assessments

- **en:** Source meanings are appropriately conservative: unavailable is verification-limited; stale protects later edits; failed leaves outcome uncertainty. Technical R is not localized.
- **fr:** Restaurer/restauration is idiomatic rollback language, distinct from denying consent. Vérifier l’état is read-only; n’annulera pas les modifications ultérieures preserves the safety negation. Formal vous-compatible imperatives.
- **es:** Revertir/reversión accurately denotes undo, not refusal. Comprobar estado is inspection only; no deshará cambios posteriores preserves later-edit protection. Revisa is correct tú register per the Spanish guide, not a formality violation.
- **zh-CN:** 还原/已还原 and 查看状态 distinguish action, success and read-only check. 当前会话无法验证 is verification-limited; 不会撤销后续更改 preserves exact protection. 配置 naturally names the profile/configuration object in this context.
- **ru:** Восстановить/Восстановлено denotes restoration, not refusal. Проверить статус is read-only; не отменит последующие изменения preserves negation and temporal scope. Impersonal language and polite Проверьте follow style.
- **vi:** Khôi phục/Đã khôi phục and Kiểm tra trạng thái distinguish restore, completion and read-only inspection. không hoàn tác các thay đổi sau đó protects later edits; không thể hoàn tất leaves uncertainty. Pronoun-free imperatives are idiomatic and guide-compliant; NFC verified.
- **ur:** بحال کریں/بحالی مکمل and حیثیت جانچیں distinguish restore, success and read-only check. بعد کی تبدیلیاں واپس نہیں کرے گا retains the prohibition on reverting later edits. Respectful کریں/جانچیں/دیکھیں, session term سیشن, logical storage and absence of injected bidi controls conform to the Urdu contract. Shaping and native RTL layout not claimed.
- **hi:** बहाल करें/पिछला संस्करण बहाल and स्थिति जाँचें distinguish restore, completion and read-only checking. बाद के बदलाव वापस नहीं करेगी preserves later-edit protection; पूरी नहीं हो सकी leaves uncertain failure. Polite imperatives, सत्र, प्रोफ़ाइल and danda prose punctuation match style.

## One batched findings list

### D1 — NIT: en punctuation convention

Keys: `tui.profile_actions.checking`, `tui.profile_actions.reverting`.

Both labels contain one ASCII three-dot ellipsis (U+002E ×3), rather than U+2026. Rule: i18n/style-guides/en.md: Punctuation/Mechanics uses … for copy ellipses; ... reserved for code-like enumerations. Optional punctuation alignment in the batched remediation; no safety-meaning defect.

### D2 — NIT: fr punctuation convention

Keys: `tui.profile_actions.stale_reason`.

The character immediately before the single semicolon is U+0020 (measured), not U+202F. Rule: i18n/style-guides/fr.md: Punctuation & typography requires U+202F before prose semicolons. Use the required narrow no-break space if folding nits into a remediation wave; meaning already correct.

### D3 — NIT: zh-CN punctuation convention

Keys: `tui.profile_actions.checking`, `tui.profile_actions.reverting`.

Each loading label contains one U+2026, not the two-mark …… specified by the locale style guide. Rule: i18n/style-guides/zh-CN.md: Punctuation & typography defines ellipsis as two marks. Optional two-mark punctuation alignment in the same batched remediation. Still fits the 52-cell budget (current maximum across all zh-CN labels is 21).

These are explicit local style-guide departures, not proposed stylistic synonym rewrites. They do not change action identity, negation, success/failure polarity, temporal scope or recovery advice. Defer in the PR thread or fold into a single already-needed remediation; no standalone nit cycle. If changed, re-audit only changed bytes/direct meanings, not unrelated namespaces.

## Boundaries and handoff

This is **not rendered design/QA/native RTL sign-off**. Rich cell measurements do not establish real terminal shaping, narrow-widget layout, focus, OS/phone behavior or authorization engine correctness. No screenshot or native evidence invented. Urdu logical order/no manual controls and preserved Latin binding are copy-level PASS; actual paint isolation remains its owning QA stream.

No catalogue/context/repository/ledger/generated files were changed. Prior `blocked-receipt.json` is retained as historical failed setup evidence; this completed audit does not erase it.

The existing bundle format is `i18n/audits/evidence/README.md`: `record.json` (locale, namespace, date, key count, auditor identity/independence/model source, input hashes, artifacts, current verdict), `audit.log`, `report.md`, `receipt.json` (companion hashes). Own per-locale records under this scratch root follow that shape and reference the shared complete actual log/guarded tool receipts. Root `receipt.json` seals all records and companions.

The owner, **only after collecting both independent complete audits**, may append honest per-locale rows to `i18n/ledger/<locale>.json` with `source_sha256`, translator/auditor agent+actual model+date, `result: passed`, unresolved blocker/major `findings: 0`, `status: passed`, and a note referencing both audit records plus non-gating nit dispositions. Exact row schema and command handoff are in root `record.json`; per docs/I18N.md and docs/design/i18n.md §6. Never hand-edit aggregate `i18n/ledger.json` or shipped eligibility. Commands after shard append: `python3 scripts/i18n/generate.py`, then `python3 scripts/i18n/generate.py --check`. Auditor ran neither, as ledger integration is outside this slice.

**TERMINAL: no gating copy finding.**
