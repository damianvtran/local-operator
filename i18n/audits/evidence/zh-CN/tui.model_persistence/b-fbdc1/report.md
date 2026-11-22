# Independent B audit — zh-CN / tui.model_persistence

Date: 2026-11-22. Source revision: 87d87ec8b02ef3c279fda673302ce775f4a0329e.
Auditor: i18n-auditor-B; session fbdc1a799cf0; actual model openai/gpt-6.1-sol. Runtime switched from anthropic/claude-opus-5-5 because of provider quota failure before the copy assessment. No MT/API used. Translator session 58dee179aac0, A auditor 26adce0a0cfa and source author a97c7f76d5ee are other sessions. A conclusions were not read.

**Verdict: passed**, all 4 keys audited; zero unresolved BLOCKER/MAJOR. One non-gating MINOR integration risk below. This is a new current B audit, not recreated historical evidence, and not an eligibility/native UI certificate.

## Native reasoning (audit notes, not execution log)

- `persist`: `/model default将当前模型保存为新会话的默认模型` accurately saves the current model for future new sessions. No current-session switch implied. 模型 / 会话 comply with the glossary. Command-first declarative wording describes behavior; it is not a curt imperative requiring 请. Natural concise developer-facing Chinese.
- `settings`: `/settings也可修改新会话的默认模型` preserves the additional route and boot-default meaning. It remains self-contained when alone: 也可 means this route can also edit that setting, not that an unusable /model action must be taken. No /model route added.
- `saved`: `/model saved将本会话切换至已保存的默认模型` explicitly distinguishes switching THIS session from saving the model. 已保存的默认模型 and 本会话 are precise and idiomatic.
- `with_model`: `模型：{label} — {routes}` correctly translates the human prefix, retains both opaque arguments exactly once, and uses full-width Chinese colon. Spaced em dash is the existing cross-language composition separator, not translated command text. No unnecessary pronoun, honorific or sentence-final punctuation on this fragment.

All four are native Simplified Chinese, consistent terminology/register, with Latin commands unchanged and no added Chinese/Latin spaces as the guide requires. No destructive/permission meaning or polarity error. No maxCells/maxChars cap exists in the sidecar; the 43-cell legacy footer limit does not apply to this wrapping notice.

## Mechanical execution

`script.py.txt` invokes actual `scripts.i18n.check.check_catalogues()` with only catalogue enumeration narrowed to en/zh-CN and this namespace; actual loaders, ICU parser, plural checks and context validation remain intact. It also independently checks order, argument names/types/count, zero plural/select arms, exact command bytes/count, NFC and absent markup/backticks. All module paths resolve inside this worktree. Primary checkout interpreter is used with PYTHONPATH pointing here, env -i, isolated HOME/config, no CMUX variables.

Actual final output: `SCOPED_REPO_VALIDATOR []`; 4 `KEY_PASS` rows; `WRAP_SUMMARY 108`; `RECEIPT_SCHEMA_PASS note_chars 227`; `PASS 4/4 keys; zero mechanical problems; no global/base checker, native rendering or ledger write`. Before/after frozen hashes agree. Full output is audit.log, not these notes.

Wrapping uses actual ICU runtime and Rich Text.wrap at 80/100/120, short/long opaque model IDs, ordinary/missing-model/settings-only routes, and English/zh-CN/synthetic pseudo controls. Insets 4/8 are expressly ASSUMED string-risk budgets for the two transcript families, NOT measured app geometry. Pseudo expansion touches human characters only and retains placeholder/command tokens. 108 probes stay within those cell budgets, but do not establish no-wrap or command atomicity. Raw source cells: persist 46, settings 33, saved 42, envelope template 24.

### B-F1 — MINOR: integration command-fold risk; not a translation change request

Key: `tui.model_persistence.saved` (also composition neighbors).
Current text: `/model saved将本会话切换至已保存的默认模型`.
At width 80 / assumed 4-cell inset / short label, the ordinary zh-CN composition ends row 1 with `/model` and starts row 2 with `saved将…`. Across 36 probes per locale, 9 Chinese probes split a two-word command; English same-code controls split none, pseudo split 20. Thus this limited matrix does NOT establish that this is an English-baseline-identical effect; it is a layout-sensitive Chinese risk, not evidence of a novel app regression. The context explicitly recognizes an existing narrow-width fold baseline. Do not certify real NoticeBlock behavior from this probe.
Suggested wording: retain current wording; no native-copy defect demonstrated. Suggested integration action: exercise actual styled NoticeBlock at the required widths and preserve readable two-word command boundaries using the existing presentation mechanism if needed. Root owns that later integration; do not alter stored command bytes or opaque IDs. Not a blocker for this independent translation audit.

## Execution provenance / limitations

The first actual attempt failed a rig assertion that searched `/model` in the whole envelope: the opaque provider/model label itself contained `/model`. Preserved verbatim in attempt1.log. Corrected assertion checks routes only; final audit.log is the successful second execution. This was a probe bug, not a catalogue defect. Earlier A blocked admission is outside this B run; nothing here supersedes or deletes it.

No styled NoticeBlock app, native shaping, browser, server, global checker, generation or shipped eligibility was executed or claimed. No target, A bundle, ledger/history, index or git commit was changed. receipt.json is schema-valid and append-ready; translator model is explicitly unknown in this session (null), not invented. Root must retain its actual known translator provenance on centralized append. The receipt has not been appended by B.
