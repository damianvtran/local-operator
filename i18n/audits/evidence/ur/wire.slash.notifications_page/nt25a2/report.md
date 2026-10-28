# Independent Urdu delta audit — wire.slash.notifications_page — nt25a2

Auditor: translation-auditor; session `89dbc1a173e9`; instance `89dbc1a173e9-nt25a2`; actual model `openai/gpt-6-astra` from resumed session-state. Same independent auditor session as nt25a1, no translator/coder authorship. Prior audit and its model provenance remain immutable.

## Verdict
**Clean / passed** — 1/1 key re-audited. No unresolved blocker, major, minor or nit in this namespace. **T-UR-CORE-1 fixed.**

## Exactly the remediation
Key: `wire.slash.notifications_page.help`
- English (unchanged): Asks and agent updates; read marks all read
- Previous Urdu: سوالات اور ایجنٹ اپ ڈیٹس؛ read سب پڑھا ہوا
- Current Urdu: سوالات و ایجنٹ اپ ڈیٹس؛ read سب کو پڑھا قرار دے

Current bytes exactly equal nt25a1's proposed repair. سوالات و ایجنٹ اپ ڈیٹس retains asks plus agent updates. سب کو پڑھا قرار دے supplies the missing state-changing predicate and all-item scope. دے predicates the read command rather than addressing the operator informally, so it introduces no T/V-register shift. Concise coordination with و fits terse help. Literal read is unchanged; the Urdu semicolon is outside the Latin token; zero manual bidi controls. No new semantic/idiomatic issue in this one-key delta.

## Actual results
- Real core parse_message: English and Urdu PASS; argument tuples (), plural selectors ().
- Real render_message(current, {}, 'ur'): exact current Urdu text returned.
- Installed rich.cells.cell_len: **47**, context maxCells **50**.
- In-memory malformed ICU negative: expected MessageSyntaxError observed.
- In-memory missing required argument negative: expected MessageFormatError observed.
- In-memory translated read negative: missing literal token correctly detected.
- In-memory over-budget negative: **53 > 50** correctly detected.
- No invalid catalogue written; all four negatives are in memory only.

Original stdout/stderr retained byte-for-byte in audit.log, including real runtime path and measured hashes. Command ran under env -i with isolated HOME/config and existing repository .venv. No dependencies installed; no browser or live sessions.

## Retention, not re-review
Help English/context unchanged from nt25a1. Core errors English/Urdu/context and UI notifications English/Urdu/context all match prior recorded hashes. Core errors retains clean (4 keys), UI retains clean-with-findings (130 keys, existing 2 minors). Only **1 key** re-audited; **134 keys retained**. Overall retained-plus-delta scope: 135 keys, zero unresolved major/blocker, existing UI minors and rendered-fit follow-up retained. No out-of-scope findings.

## Provenance
Observed core HEAD: 329b5c581784df1082dbd0630c0f814c7a3c0c8f. Target uncommitted: git status M; git diff numstat 1/1. Input hashes bind actual working-tree bytes, not a nonexistent remediation commit. Previous evidence: i18n/audits/evidence/ur/wire.slash.notifications_page/nt25a1/record.json. All four prior companion hashes verified unchanged; nt25a1 not overwritten.

## Input hashes
- en: `local_operator/i18n/catalogues/en/wire.slash.notifications_page.json` — `9440f547bbf4e982c949202fd5ddb78a784a3d6da0f7e88694eb9ffdc3bc6594`
- en_context_sidecar: `local_operator/i18n/catalogues/en/wire.slash.notifications_page.context.json` — `2974cbbc68f15d61f45c8a79d99185ddb8b13afeb2f1920d592f752accdf45e0`
- target: `local_operator/i18n/catalogues/ur/wire.slash.notifications_page.json` — `88c087d1e231da9b916472cb38b733b1fba07fa010ec3e9c8943a22fe7d1d378`

## Limits
Independent LLM text/terminal-cell validation, not native-speaker, rendered RTL/Nastaliq or release certification. Original coder owns ledger and unified remediation commit. Auditor edited no catalogue, ledger, code, generated file or git state.
