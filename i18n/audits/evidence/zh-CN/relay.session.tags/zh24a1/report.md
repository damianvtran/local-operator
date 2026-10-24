# Independent translation audit — zh-CN / relay.session.tags / zh24a1

- Round: 1
- Audited revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (branch `i18n/zh-cn-core-rebased`)
- Auditor: translation-auditor (lopdev team), session `0c7aebe462e8`, served model `deepseek/deepseek-flash`; no fallback observed. Independent of the translation: these catalogues were not authored in this session and were not edited by this audit.
- Keys read: 19 of 19 English keys (full namespace, every value read in context)
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
- Render: 25 render sample(s), 0 failure(s)

## Findings

### ZH-04 — minor — `relay.session.tags.limit_hint`

- Current: `一个会话最多可有{count}个标签 — 移除一个才能再添加。`
- Proposed: `每段对话最多可有{count}个标签 — 移除一个才能再添加。`
- Reason: This is the relay surface's own vocabulary: relay.session.list.no_matches is "没有匹配的对话" and relay.ask.dismiss.answered_in_chat_note (same delta) is "你已在对话中回复", while 会话 is this locale's glossary term for "session" (session → 会话); using 会话 here collapses two distinct English terms and disagrees with the rest of the relay copy. All shipped locales keep "conversation" distinct (es conversación, vi cuộc trò chuyện).

## Observations (checked, no action requested)

- Plural structure: {count, plural, other {另有#个标签}} — other only, `#` bound to {count} — correct for zh-CN; the accessible name reads naturally for the +N mark.
- catalogue → 目录 is this locale's established rendering (cli.sessions_help.peer "其自身目录"), and load_error/missing use it consistently.
- pin → 固定 matches the existing locale copy (tui.small.session_sidebar.pin_suffix "ctrl+k固定").
- tag → 标签 is uniform across the namespace; 主标签 for primary and 未标记 for Untagged agree with tui.tags.untagged.

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
- `relay.session.tags.pairs.txt`
- `relay.session.tags.compact.txt`
- `audit.py.txt`
