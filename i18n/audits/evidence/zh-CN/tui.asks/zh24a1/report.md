# Independent translation audit — zh-CN / tui.asks / zh24a1

- Round: 1
- Audited revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (branch `i18n/zh-cn-core-rebased`)
- Auditor: translation-auditor (lopdev team), session `0c7aebe462e8`, served model `deepseek/deepseek-flash`; no fallback observed. Independent of the translation: these catalogues were not authored in this session and were not edited by this audit.
- Keys read: 11 of 11 English keys (full namespace, every value read in context)
- Key parity: 0 missing, 0 extra; key order: identical
- Verdict: **clean-with-findings**
- Findings: 2 minor, 0 major, 0 blocker

## Method

String-level audit against the English source and the English context sidecar, using the repository's own ICU runtime (`local_operator.i18n.runtime`) as the renderer of record. Mechanical checks were executed, not eyeballed: key parity and order; NFC; control, bidi and zero-width code points; full-width Latin/digit homoglyphs; the style guide's no-space-between-CJK-and-Latin rule; ICU argument names and kinds; plural selector categories and `#` binding; verbatim survival of flags, commands, env vars, URLs, paths and product names; and display-cell widths against every sidecar `maxCells`.

Every value was then read semantically against its context note: what the control does, who the actor is, what the surrounding labels say, and whether a native speaker would write the sentence. Candidate findings were cross-checked against the five shipped locales and against the existing zh-CN catalogues before being recorded, so a divergence that is this locale's established practice is reported as an observation, not a defect.

## Mechanical results (executed)

- Key parity: 0 missing, 0 extra; key order: identical
- Non-NFC values: 0; control chars: 0; bidi controls: 0; zero-width: 0; full-width Latin/digits: 0; CJK↔Latin spaces: 0
- ICU argument mismatches: 0; plural-structure errors: 0; select arguments present: 0
- Technical tokens dropped: 0
- Width: 11 budgeted key(s), 0 over budget
- Render: 17 render sample(s), 0 failure(s)

## Findings

### ZH-05 — minor — `tui.asks.context.messages`

- Current: `{count, plural, other {提问后你发来#条消息（最后{time}）}}`
- Proposed: `{count, plural, other {此问题提出后你发来#条消息（最近{time}）}}`
- Reason: Same line as the phone card (ZH-02) and the same two defects: 最后 renders a bare clock time as "finally at" rather than "most recently", and 提问后 drops the demonstrative 此问题, leaving who asked ambiguous. The proposed form renders at 38 cells against this row's 78-cell budget and still puts the count and 你发来 ahead of the parenthetical time, so the tail-clip order the sidecar requires is kept.

### ZH-06 — minor — `tui.asks.dismiss.moot_note`

- Current: `不再等待此问题`
- Proposed: `智能体不再等待回答`
- Reason: Same note as the phone card (ZH-03): the sidecar says it names what happens to the waiting ("the agent stops expecting an answer"), and the subjectless Chinese reads as an instruction to the operator instead.

## Observations (checked, no action requested)

- Plural structure: other only, `#` bound to {count}; matches the phone card exactly, which is what the two sidecars require (both surfaces read the same line in English).
- dismiss.hint keeps the key names `enter`/`esc` lowercase and the middle-dot separator, and uses the same verb (忽略) as the chooser title and the phone control — the sidecar's cross-surface verb requirement is met.
- fleet_gone says 该会话 (that conversation), not "this" — the sidecar's explicit "must not claim the ask is absent from this session" constraint is honoured.
- Widths: all 11 budgeted values are inside maxCells; the widest rendered row is fleet_gone at 28/60 cells.

## Caveats

- This is a string-level audit. No native terminal, relay SPA, Electron or rendered-frame pass was performed here, so truncation in a real paint, font-cell width and interaction behaviour are not claimed.
- zh-CN remains in `SECURITY_HELD_LOCALES` (`scripts/i18n/generate.py`) and out of `data/shipped.json`; this record changes no eligibility, ledger row or other locale.

## Evidence files

- `record.json`
- `receipt.json`
- `report.md`
- `audit.log`
- `mechanical-raw.json`
- `runtime-mechanical.json`
- `pseudo-evidence.json`
- `tui.asks.pairs.txt`
- `tui.asks.compact.txt`
- `audit.py.txt`
