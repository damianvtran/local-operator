# Hindi A independent audit — tui.model_persistence

Date: 2026-11-22. Frozen head: b1c5ca8be245dcc3f9cae4463d9a5553b3e0c01f.
Auditor: i18n-auditor, session 29fd3d454da2, actual model openai/gpt-6.1-sol.
Host explicitly switched from anthropic/claude-opus-5-5 for provider quota failure before substantive audit. Translator fdf96049990f used openai/gpt-6.1-sol per author-fdf9/record.json; source author a97c7f76d5ee. Separate session/agent, NOT model-independent. Native-fluent LLM reasoning only; no MT/API/service or source/target edits.

## Verdict

PASSED: 4/4 keys audited in source order; zero BLOCKER/MAJOR, one MINOR. This is A only, not the second high-visibility independent pass or root admission. Minor is non-gating and does not request its own cycle.

## Key-by-key assessment

1. `tui.model_persistence.persist`: `/model default से इस मॉडल को नए सत्रों के डिफ़ॉल्ट के रूप में सहेजें` — accurate: saves this model for new sessions, not a current-session switch. Polite imperative सहेजें, glossary सत्र/मॉडल, conventional डिफ़ॉल्ट, correct nukta. Command untouched. Natural concise instruction; no added persistence scope or destructive implication.
2. `tui.model_persistence.settings`: `/settings से भी नए सत्रों का डिफ़ॉल्ट बदलें` — boot default made explicit as new sessions' default. भी preserves source “too”. As a sole hosting-deadend route it remains a complete usable clause, not an instruction to use an unavailable /model command. Good everyday polite Hindi. Source includes “too” in the sole-route state as well: do not mistake the inherited additive particle for a new meaning defect.
3. `tui.model_persistence.saved`: `/model saved से इस सत्र में सहेजे गए डिफ़ॉल्ट पर जाएँ` — imperative पर जाएँ preserves switching, rather than saving. Current-session scope is present. One optional attachment clarification below; no unsafe polarity or reversal. Re-read with neighboring persistence/settings clauses before classifying: saved-default contrast and switching verb make intended behavior recoverable, so MINOR, not MAJOR.
4. `tui.model_persistence.with_model`: `मॉडल: {label} — {routes}` — correct localized human model prefix and exact opaque simple arguments, each once and in source order. Routes remain complete prelocalized clauses; no recursion/retranslation. Colon/em dash spacing acceptable for this envelope. These are joined instructional clauses, not standalone prose sentences requiring a terminal danda. No Devanagari digit policy violation; there are no literal digits to localize.

## Finding HI-A1 — MINOR (non-gating)

Key: `tui.model_persistence.saved`.
Current: `/model saved से इस सत्र में सहेजे गए डिफ़ॉल्ट पर जाएँ`.
Problem: “इस सत्र में सहेजे गए” can attach to “saved”, suggesting the default was saved within this session rather than that this session is being switched. The switch verb and adjacent clauses preserve the intended operation, so this is a clarity improvement, not a demonstrated wrong/unsafe action.
Suggested wording: `/model saved से इस सत्र को सहेजे गए डिफ़ॉल्ट मॉडल पर ले जाएँ`.
Disposition: optional, deferred in this passed A report; do not create a separate remediation cycle for it. No other findings and no NIT padding.

## Actual mechanical / render evidence

`audit.log` is the real stdout/stderr of `script.py.txt`, exit 0, not this narrative. The rig uses the primary checkout's .venv Python with worktree PYTHONPATH; module provenance is asserted and printed. HOME/config is session-scratch isolated via env -i (CMUX and XPC_FLAGS absent). No application startup, global base checker, tests, generation, installs or UI dispatch.

Official `scripts/i18n/check.py::check_catalogues()` executed unmodified with ONLY namespace/locale enumeration narrowed in-memory to en/hi and tui.model_persistence. Output: `OFFICIAL SCOPED CHECK []`. Official checker missing-key/missing-argument/renamed-argument negative controls all failed as expected and actual diagnostics are logged. Runtime missing label/routes each raised MessageFormatError. Valid JSON, exact 4-key source/context/target order, simple argument names/types/count, no plural/select nodes in all four (zero arms; Hindi CLDR arms must NOT be invented), no markup/backtick changes, commands protected, no unexpected bidi/ZWJ/ZWNJ, and NFC all passed.

Cell estimates: persist 50 / settings 31 / saved 41 / envelope-template 23; characters 68/43/53/24. Sidecars have no maxCells/maxChars, so the old fixed footer cap is NOT applied. These are wrapping notices. 54 Rich-only diagnostics: English/Hindi/pseudo-English × ordinary/missing-model/hosting-deadend × short/long opaque labels × 80/100/120 columns. Printed rows and cell lengths allow inspection; no row exceeded its selected width. Pseudo expands human text only, protecting commands and ICU. The long ID includes braces/brackets/flag-like bytes, and a separate opaque routes argument confirms no recursive interpolation. Hosting-deadend has only /settings. Zero-value instructions render with {}.

Limitations: plain Rich is NOT the styled NoticeBlock, pristine/non-empty transcript geometry, native Hindi shaping, font fallback or screenshot proof. Devanagari combining/conjunct metrics may be optimistic; no fixed-cap overflow or shaping certification inferred from cell_len. Long opaque IDs can fold under Rich, as can English; existing command-fold regressions remain integration baseline. No global UI/native/eligibility/release clearance. Root integration must exercise actual NoticeBlock families and native presentation separately. Before/after all three input hashes match the supplied frozen bytes exactly.

## Retention / authority

This new canonical bundle only is owned by this auditor. No ledger/history apply, stage, commit, push, catalogue, generation, baseline, eligibility or prior-evidence edits. `receipt.json` is append-ready for the root's serialized admission, NOT an applied ledger row. At handoff the bundle is untracked repo-local evidence; manager must commit it for tracked durable authority. `record.json` binds inputs, artifacts and provenance. This report is audit_notes_not_execution_log; `audit.log` alone supplies execution output. There was no executed harness failure; an early unbounded discovery command was guard-rejected before execution and narrowed to scoped reads.
