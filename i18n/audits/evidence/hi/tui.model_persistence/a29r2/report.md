# Hindi A — round 2 independent whole-namespace audit

2026-11-22 · four keys · scope 61b98804881432cfaca9a7f84ef78d81183e2083..912e165fd76e217b5cd5d212a59e7e1e906fa6fd.
Record reference: i18n/audits/evidence/hi/tui.model_persistence/a29r2/record.json.

## Verdict: passed

Zero unresolved BLOCKER, MAJOR, MINOR or NIT. HI-A1 is fixed. The same attachment problem identified in B-HI-1 is absent from the new bytes in this A assessment; this does not overwrite B's failed history or substitute for a fresh B pass.

## Semantic and style assessment — all four current keys, in order

1. **persist**: `/model default से इस मॉडल को नए सत्रों के डिफ़ॉल्ट के रूप में सहेजें`.
The object is this model, action is saving, destination is the default for NEW sessions. Explicit नए सत्रों prevents a current-session-switch interpretation. सहेजें is a normal polite product imperative. मॉडल/सत्र conform to the glossary; डिफ़ॉल्ट is idiomatic without treating the author's glossary proposal as binding. Exact command preserved. No new finding.
2. **settings**: `/settings से भी नए सत्रों का डिफ़ॉल्ट बदलें`.
The instruction edits new-session boot default, not this session's model. भी retains source “too”; the clause is grammatically complete when it is the sole hosting-deadend route. It names only /settings and does not advertise a refused /model route. Standard polite बदलें and clear everyday Hindi, no register or meaning defect.
3. **saved**: `/model saved से इस सत्र का मॉडल सहेजे गए डिफ़ॉल्ट पर बदलें`.
Independent grammatical assessment: the direct object is now **इस सत्र का मॉडल** (this session's model). **सहेजे गए** modifies the destination **डिफ़ॉल्ट**, and **पर बदलें** directs switching to that saved destination. There is no locative “इस सत्र में” adjacent to the saved participle to imply the default was saved within this session. बदलें is switching here, not saving; the new line cannot naturally be read as an instruction to persist the current choice. Thus HI-A1's attachment ambiguity is resolved. The neighboring persist/settings lines identify new-session defaults, while this line explicitly identifies the current session's model; the three routes remain distinct. Idiomatic concise polite Hindi; no replacement suggestion needed.
4. **with_model**: `मॉडल: {label} — {routes}`.
The human prefix is localized; both simple arguments remain exact, once each, source order. No extra grammar is forced onto already-complete route clauses. Opaque labels/IDs and route arguments are passed through without retranslation or recursive interpolation. Colon/em-dash spacing is suitable for an envelope; these joined fragments do not require appended danda punctuation. No finding.

All four use consistent मॉडल, सत्र and डिफ़ॉल्ट, standard Devanagari/nukta, polite instructional register, and no literal numerals inconsistent with the Latin-digits policy. No unsupported English human copy remains; literal commands and opaque runtime IDs are intentionally English/unchanged. None is destructive or permission-granting; no negation/polarity issue appears. Hindi is LTR, so no Urdu RTL certificate is applicable.

## Mechanical execution and fit

`audit.log` is actual stdout/stderr of `script.py.txt`, exit 0; this report is narrative audit notes, not execution evidence. Official check_catalogues was executed unchanged with only in-memory locale/namespace iteration restricted to en/hi and tui.model_persistence: `[]`. All 4 ordered keys validated against source/context, simple arguments (zero for the three routes, label/routes each once for envelope), **zero plural/select structures**, exact commands, JSON, NFC, markup/backtick parity, no manual bidi controls. No Hindi CLDR arms were invented.

Official negative controls reported missing-key, dropped argument and renamed argument errors. Runtime missing label/routes raised MessageFormatError. Direct unknown-ID literal input was separately verified to pass through unchanged, NOT reported as a missing-key error. Opaque braces, flags, brackets and long model IDs survive as argument bytes.

54 actual Rich wrap diagnostics cover English/Hindi/pseudo-English, three semantic route sets, short/122-character opaque label, and 80/100/120 columns. Pseudo expansion preserves commands/ICU. All wrapped rows fit their requested Rich width. Current raw template cell widths are 50, 31, 46, 23; these are diagnostics, NOT fixed budgets. The source/context provide no maxCells/maxChars cap, and the footer's 43-cell cap does not apply. Devanagari conjunct/combining mark rendering and cluster-safe splitting of long IDs remain font/terminal risks that plain Rich does not certify. No native shaping, styled NoticeBlock, transcript geometry, native font, all-width or whole-UI claim.

Primary .venv interpreter with worktree PYTHONPATH is printed and module provenance asserted. env -i uses scratch HOME/config and strips CMUX/XPC_FLAGS. No app startup, install, tests, global checker or generator. Original EN/context and new HI hashes were checked before/after; all 17 old evidence/history files were hashed against the preserved author's manifest before and after execution. Normal-HOME Git independently proved only saved's value changed versus the scope base; three values and key order remain unchanged.

## Provenance and history

Current A round 2 auditor is i18n-auditor / session 29fd3d454da2 / anthropic/claude-opus-5-5 per current host model declaration, with no current resumed-run fallback event. Prior A's recorded GPT fallback is historical, not silently overwritten. Translator is session fdf96049990f. Original three unchanged strings: original author-fdf9 record reports openai/gpt-6.1-sol quota fallback. Saved repair: author-fdf9-fix1 record self-reports anthropic/claude-opus-5-5 from resumed context; parent subsequently reports wait(job9b5a2af8589d) completion AND hub list both identify served model openai/gpt-6.1-sol for that repair. Claude is NOT verified actual. These sources conflict; both are retained, and the host snapshot is parent-reported, not independently read by this auditor. No fresh repair fallback reason is known or inferred. Parent will retain the snapshots for admission. This qualification changes no target bytes, semantic verdict or validator evidence. Auditor is separate from translator/source author; no model-independent claim.

Root's original A passed row and B changes_requested row remain untouched. This new whole-namespace record is not a retroactive pass of the original target. Fresh B and root serialized admission remain separate. No other locale was audited. No ledger/history apply, stage, commit, push, generation, catalogue, baseline, caps, consumer or eligibility changes. New canonical evidence is repo-local and untracked until manager commits it; no claim it is on main. No failures occurred in this round's executed validator.
