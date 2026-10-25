# Independent translation audit — zh-CN / tui.tags / zh24a1

- Round: 1
- Audited revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (branch `i18n/zh-cn-core-rebased`)
- Auditor: translation-auditor (lopdev team), session `0c7aebe462e8`, served model `deepseek/deepseek-flash`; no fallback observed. Independent of the translation: these catalogues were not authored in this session and were not edited by this audit.
- Keys read: 27 of 27 English keys (full namespace, every value read in context)
- Key parity: 0 missing, 0 extra; key order: identical
- Verdict: **clean**
- Findings: 0 minor, 0 major, 0 blocker

## Method

String-level audit against the English source and the English context sidecar, using the repository's own ICU runtime (`local_operator.i18n.runtime`) as the renderer of record. Mechanical checks were executed, not eyeballed: key parity and order; NFC; control, bidi and zero-width code points; full-width Latin/digit homoglyphs; the style guide's no-space-between-CJK-and-Latin rule; ICU argument names and kinds; plural selector categories and `#` binding; verbatim survival of flags, commands, env vars, URLs, paths and product names; and display-cell widths against every sidecar `maxCells`.

Every value was then read semantically against its context note: what the control does, who the actor is, what the surrounding labels say, and whether a native speaker would write the sentence. Candidate findings were cross-checked against the five shipped locales and against the existing zh-CN catalogues before being recorded, so a divergence that is this locale's established practice is reported as an observation, not a defect.

## Mechanical results (executed)

- Key parity: 0 missing, 0 extra; key order: identical
- Non-NFC values: 0; control chars: 0; bidi controls: 0; zero-width: 0; full-width Latin/digits: 0; CJK↔Latin spaces: 0
- ICU argument mismatches: 0; plural-structure errors: 0; select arguments present: 0
- Technical tokens dropped: 0
- Width: 0 budgeted key(s), 0 over budget
- Render: 27 render sample(s), 0 failure(s)

## Findings

None. All 27 values are accurate, idiomatic and register-consistent, and every mechanical check passed. Nothing is requested for this namespace.

## Observations (checked, no action requested)

- This namespace has no context sidecar (no maxCells budget); widths were still measured for the record — the widest value (`invalid`) is 54 cells, and the cap/status rows wrap in the TUI as the English does.
- Numbers (8, 32, 1–24) are source literals in the same positions; {count} stays a simple argument in header, and /8 stays literal.
- Quotes are the mainland “ ” pair, the reserved name 未标记 matches the untagged value byte-for-byte, and the on/off pair (开启/关闭) is consistent with tui.tags.slash_tags.
- tag/label: the team form's `label` field is also 标签 elsewhere in the delta, but the two never co-occur on one screen and no English term is shared; not a finding.

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
- `tui.tags.pairs.txt`
- `tui.tags.compact.txt`
- `audit.py.txt`
