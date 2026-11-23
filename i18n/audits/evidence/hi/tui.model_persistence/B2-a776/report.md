# Hindi independent audit B — round 2

**Verdict: passed.** Whole four-key namespace audited against English/context at frozen HEAD `912e165fd76e217b5cd5d212a59e7e1e906fa6fd`; remediation scope `61b98804881432cfaca9a7f84ef78d81183e2083..912e165fd76e217b5cd5d212a59e7e1e906fa6fd`. No unresolved BLOCKER/MAJOR, no new findings. This is the second independent high-visibility stream B's same-history remediation round, not a replacement for its historical failed audit. No A-round-2 conclusions read or judgment coordinated. No translation service/API/MT used.

## Finding disposition

**B-HI-1 resolved** (the manager identifies HI-A1 as the same accepted ambiguity; no A-round-2 verdict used).

Old: `/model saved से इस सत्र में सहेजे गए डिफ़ॉल्ट पर जाएँ`

New: `/model saved से इस सत्र का मॉडल सहेजे गए डिफ़ॉल्ट पर बदलें`

Independent grammatical assessment: `इस सत्र का मॉडल` is now the explicit object of `बदलें`; `सहेजे गए डिफ़ॉल्ट पर` is the destination. There is no longer an adjacent `इस सत्र में सहेजे गए` phrase that locates the saving inside this session. The instruction changes this session's model to the already saved default; it does not save the current model. The polite imperative is idiomatic developer-product Hindi. This independently resolves the original semantic attachment defect, not just its wording symptom. The author's native explanation is consistent with this reading but is not treated as independent approval.

## Full four-key assessment, source order

1. `persist`: `/model default से इस मॉडल को नए सत्रों के डिफ़ॉल्ट के रूप में सहेजें` — PASS. Explicit current-model saving for future new sessions, with no suggestion of switching the active session. Polite सहेजें; मॉडल and सत्र match the glossary; डिफ़ॉल्ट is a natural technical loan.
2. `settings`: `/settings से भी नए सत्रों का डिफ़ॉल्ट बदलें` — PASS. Preserves “too,” makes boot/new-session scope clear and remains a usable standalone route when model commands refuse. No added /model command or hosting claim.
3. `saved`: new wording above — PASS; B-HI-1 resolved. Separates session-model switching from saving the default. Exact ASCII command preserved.
4. `with_model`: `मॉडल: {label} — {routes}` — PASS. Localizes the human prefix; simple-string label and already-translated route clause arguments remain exactly once, unparsed and unchanged. Appropriate short label typography and spaces.

Everyday polite Hindi, consistent nukta and settled loan usage, no heavy literal calques or added honorifics. Fragment-style notices do not require added danda punctuation. No digits requiring a different policy. All four messages have zero plural/select constructs: there are no Hindi CLDR arms to add. No safety/polarity/error/permission issue identified.

## Executed checks

Retained `script.py.txt` is the prior B's mechanical rig reused in a NEW bundle, with current expected input hashes pinned. Exact invocation:

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/hi-B2-home" LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/hi-B2-home/config" PATH="$PATH" TERM=xterm-256color PYTHONPATH=/Users/damian/local-operator/.worktrees/i18n-model-persist-notice /Users/damian/local-operator/.venv/bin/python i18n/audits/evidence/hi/tui.model_persistence/B2-a776/script.py.txt
```

Exit 0. Actual stdout/stderr is `audit.log`; this report is notes, not a substitute execution log. Primary .venv interpreter and imported worktree module path are printed; HOME/config isolated, CMUX stripped by env -i. Official `check.check_catalogues()` ran with catalogue enumeration scoped to en/hi and this namespace only: context schema, keys, ICU argument parity/type and plural checks passed. Missing-key, missing-argument and renamed-argument negative dictionaries each produced the expected validator failure. Missing runtime label/routes parameters were rejected. Exact commands, markup/code-span parity, simple-string arguments exactly once, NFC and absence of unneeded ZWJ/ZWNJ passed. All four source/target messages rendered; synthetic human-copy expansion retained commands and placeholders. Short/long opaque model labels and already-localized routes inserted unchanged.

Plain Rich wrapping ran at 80/100/120 with same-code English controls. Actual rows/cell widths are in the log. No maxCells/maxChars in the sidecar, no inherited 43-cell footer cap, no invented no-wrap guarantee. Devanagari combining marks and conjuncts make cell estimates/font shaping distinct; plain Rich is NOT styled NoticeBlock, pristine/non-empty transcript-family, native font/shaping or real UI proof. Long opaque IDs may wrap and some command words can split, including English controls: this changes row projection, not source/runtime bytes, and does not establish a new translation overflow defect or qualify the integration baseline.

`preservation.log` records a separate normal-HOME Git comparison: all 17 prior evidence/history files remain byte-identical to base, including original failed B; only the saved value changed and key order/other three values are identical. The old failed history row is not overwritten or retroactively passed. English/context and current Hindi hashes verified before/after; HEAD remains pinned. No shared-history append, target edit, staging/commit/push, generator, consumer, global checker, native/browser/app startup or eligibility/release claim.

## Provenance qualification

Auditor session a776dd8a4364 is independent from translator session fdf96049990f and the English author. Current resumed host declaration names `anthropic/claude-opus-5-5`; no fresh auditor model-switch event is supplied in this resumed run. This states current available provenance, not a new inference about the historical B run. Historical B/author GPT fallback records remain unchanged.

The repair author self-reports Claude in `author-fdf9-fix1/record.json` and `notes.md`, relying on resumed context. The requesting manager explicitly reports that parent wait `9b5` and Hub completion metadata both name GPT, and withdraws the earlier assumption that Claude is verified as actual. These are distinguished rather than collapsed: repair host-served model is reported as `openai/gpt-6.1-sol` on the manager's named metadata authority, not directly inspected here; the Claude self-report remains preserved as an unverified conflicting claim. No fresh fallback reason or transition is invented. This does not change input bytes or semantic findings; the manager retains its host snapshot addendum separately. This record is not native-speaker/native-font sign-off or global shipping clearance.
