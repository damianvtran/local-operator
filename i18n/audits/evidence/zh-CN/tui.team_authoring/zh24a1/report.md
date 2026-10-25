# Independent translation audit — zh-CN / tui.team_authoring / zh24a1

- Round: 1
- Audited revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (branch `i18n/zh-cn-core-rebased`)
- Auditor: translation-auditor (lopdev team), session `0c7aebe462e8`, served model `deepseek/deepseek-flash`; no fallback observed. Independent of the translation: these catalogues were not authored in this session and were not edited by this audit.
- Keys read: 36 of 36 English keys (full namespace, every value read in context)
- Key parity: 0 missing, 0 extra; key order: identical
- Verdict: **clean-with-findings**
- Findings: 1 minor, 0 major, 0 blocker

## Method

String-level audit against the English source and the English context sidecar, using the repository's own ICU runtime (`local_operator.i18n.runtime`) as the renderer of record. Mechanical checks were executed, not eyeballed: key parity and order; NFC; control, bidi and zero-width code points; full-width Latin/digit homoglyphs; the style guide's no-space-between-CJK-and-Latin rule; ICU argument names and kinds; plural selector categories and `#` binding; verbatim survival of flags, commands, env vars, URLs, paths and product names; and display-cell widths against every sidecar `maxCells`.

Every value was then read semantically against its context note: what the control does, who the actor is, what the surrounding labels say, and whether a native speaker would write the sentence. Candidate findings were cross-checked against the five shipped locales and against the existing zh-CN catalogues before being recorded, so a divergence that is this locale's established practice is reported as an observation, not a defect.

## Mechanical results (executed)

- Key parity: 0 missing, 0 extra; key order: identical
- Non-NFC values: 0; control chars: 0; bidi controls: 0; zero-width: 0; full-width Latin/digits: 0; CJK↔Latin spaces: 0
- ICU argument mismatches: 0; plural-structure errors: 0; select arguments present: 0
- Technical tokens dropped: 0
- Width: 0 budgeted key(s), 0 over budget
- Render: 36 render sample(s), 0 failure(s)

## Findings

### ZH-07 — minor — `tui.team_authoring.members_hint`

- Current: `逗号分隔：coder、coder:2、team:pod`
- Proposed: `逗号分隔：coder, coder:2, team:pod`
- Reason: The string's own instruction is "comma-separated" and the example is the literal syntax the operator types; replacing the ASCII commas with the ideographic comma 、 makes the example contradict the instruction (all five shipped locales keep "coder, coder:2, team:pod" byte-identical).

## Observations (checked, no action requested)

- Field labels follow the source's own distinctions (instructions → 指令, standing guidance → 常驻团队指引), and `manager` → 管理者 matches the shipped locales, which also translate it (es/fr responsable, vi người quản lý).
- Slash commands, `team:pod`/`role:2` tokens and key chords (`ctrl+s`, `tab`, `esc`, `y`, `n`) survive verbatim; the form's wrap target of 58 content cells is met by the longest value (notice_read_only, 58 cells).
- conversation → 会话 follows the TUI surface's established rendering (32 of 34 existing tui.* occurrences), unlike the relay key flagged in ZH-04.
- `saved team {name}` → 已保存团队{name} follows the existing locale pattern (tui.authoring.saved 已保存智能体{name}); the boundary is established locale practice, so no finding.

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
- `tui.team_authoring.pairs.txt`
- `tui.team_authoring.compact.txt`
- `audit.py.txt`
