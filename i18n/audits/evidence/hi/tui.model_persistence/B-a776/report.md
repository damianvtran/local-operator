# Independent Hindi audit B — tui.model_persistence

Date: 2026-11-22. Auditor: i18n-auditor, session a776dd8a4364, actual model openai/gpt-6.1-sol. Host explicitly switched from anthropic/claude-opus-5-5 after provider quota failure before this assessment. This is the separately requested second high-visibility pass B, not a remediation round. No A artifacts, conclusions or shared judgment were read. Translator session fdf96049990f and English author a97 are different actors. Translator model is verified as openai/gpt-6.1-sol from author-fdf9/record.json; sharing a model is not sharing a session or an audit.

Verdict: **changes_requested**. One unresolved MAJOR, zero BLOCKER. All four keys audited, not a selective sample. Hindi assessment is independent LLM reasoning; no MT/API/service and no native-speaker or font qualification claimed.

## Findings

### B-HI-1 — MAJOR — tui.model_persistence.saved

Current: `/model saved से इस सत्र में सहेजे गए डिफ़ॉल्ट पर जाएँ`

The adjacent phrase `इस सत्र में सहेजे गए डिफ़ॉल्ट` naturally parses as “the default saved in this session.” That incorrectly scopes *saving* to this session instead of clearly scoping the *model switch* to this session. It leaves the distinction between the previously saved startup default and current-session choice unclear, precisely where `/model saved` must be distinguished from `/model default`. This is a semantic attachment problem, not an objection to the glossary term or politeness. A second grammatical reading check confirms that `इस सत्र में` immediately precedes and modifies `सहेजे गए`; this is not just an English back-translation preference.

Suggested: `/model saved से इस सत्र का मॉडल सहेजे गए डिफ़ॉल्ट पर बदलें`

Here `इस सत्र का मॉडल` is the explicit object being changed, and `सहेजे गए डिफ़ॉल्ट` remains the saved target. Preserve the command bytes.

## Every-key assessment, source order

1. **persist** — `/model default से इस मॉडल को नए सत्रों के डिफ़ॉल्ट के रूप में सहेजें`: preserves saving the current model for future new sessions, not switching the current session. Polite सहेजें; everyday loan डिफ़ॉल्ट and glossary मॉडल/सत्र are appropriate. No unsupported urgency, permission or destructive claim. Pass.
2. **settings** — `/settings से भी नए सत्रों का डिफ़ॉल्ट बदलें`: makes the new-session scope explicit, keeps the English “too,” and is usable as a standalone complete instruction with no `/model` route. Polite बदलें; no hosting dead-end introduced. Pass.
3. **saved** — finding B-HI-1 above. Command and mechanical structure pass; scope clarity does not.
4. **with_model** — `मॉडल: {label} — {routes}`: human prefix localized, opaque label and already-localized route clauses each inserted exactly once. Natural compact product label, appropriate punctuation and spacing. No retranslation or inferred model identity. Pass.

No plural/select constructs exist in any of the four English or Hindi messages; no Hindi plural arms were invented. No full prose sentence requires adding a danda to these concise notice fragments. No ZWJ/ZWNJ insertion or non-NFC string was found.

## Execution evidence and limits

`audit.log` is actual output from the retained `script.py.txt`, executed with primary checkout .venv Python, worktree PYTHONPATH, isolated HOME/config and env -i (CMUX absent). Module paths prove worktree source. The official `scripts/i18n/check.py:check_catalogues` was called with catalogue enumeration narrowed to exactly en/hi and tui.model_persistence; its actual context/ICU/parity/plural checks passed. Missing-key, missing-argument and renamed-argument counterfactual dictionaries each failed this validator. Missing render parameters were rejected by the official runtime. Neither catalogue files nor ledger/history were mutated.

Both locales' four messages were rendered through runtime.render_message, with short and long opaque labels for the envelope, and wrapped by plain Rich at 80/100/120. Log includes actual text rows and cell geometry. Exact commands, argument cardinality/type, NFC, absent markup/code spans and synthetic human-copy-only pseudo expansion were checked. No maxCells/maxChars exists in these sidecars: the 43-cell legacy footer cap does not apply. Devanagari combining marks make code-point lengths different from Rich cell geometry; logged geometry is not font shaping proof. Long-label wrapping can split opaque IDs and command words; it does not change the original runtime string. These are plain-Rich observations, not new all-width guarantees, not styled NoticeBlock/pristine/non-empty transcript-family evidence, and not a new translation overflow finding. English controls were run at the same widths.

All three input hashes match the assigned values before and after. Source HEAD b1c5ca8be245dcc3f9cae4463d9a5553b3e0c01f. No app startup, global checker, test suite, install, generation, browser/native evidence, history append or release claim.

Discovery-only command issues: one combined discovery command was refused because it contained unbounded find; it did not execute. Two subsequent read-only discovery commands exited nonzero on guessed nonexistent render.py/pseudo.py/audit_history.py paths. The retained audit itself ran once and exited zero; these discovery issues are not validator executions or evidence of catalogue failure. This report is audit notes, not an execution log.
