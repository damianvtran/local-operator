# Independent translation audit — zh-CN / relay.ask / zh24a1

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

### ZH-02 — minor — `relay.ask.context.messages`

- Current: `{count, plural, other {提问后你发来#条消息（最后{time}）}}`
- Proposed: `{count, plural, other {此问题提出后你发来#条消息（最近{time}）}}`
- Reason: "(last {time})" is the most recent message's clock time: 最后 with a bare time reads as "finally at", where 最近 is the natural rendering (es "último {time}", vi "gần nhất {time}"); adding 此问题 also restores the demonstrative that makes it clear the agent's ask — not the operator's — was the thing asked.

### ZH-03 — minor — `relay.ask.dismiss.moot_note`

- Current: `不再等待此问题`
- Proposed: `智能体不再等待回答`
- Reason: The sidecar defines this note as what happens to the waiting — "the agent stops expecting an answer" — but subjectless 不再等待 reads in Chinese as an instruction to the operator ("stop waiting on this question"); every shipped locale names the actor (es "el agente dejará de esperar respuesta", fr "l’agent n’attendra plus de réponse", ru "агент больше не будет ждать ответа").

## Observations (checked, no action requested)

- Plural structure: English has one/other, the target has other only — correct for zh-CN, whose plural category set is {other}. `#` is present and bound to {count}, and {time} stays a simple argument that is inserted, never reformatted.
- `dismiss…` uses the two-mark Chinese ellipsis …… required by the zh-CN style guide rather than the sidecar's single glyph; the locale guide is binding on typography and the meaning (a choice opens) is unchanged. Recorded as an accepted deviation, not a finding.
- Width: every budgeted value is inside its sidecar maxCells, measured in display cells (CJK/fullwidth 2, ASCII 1). The largest, context.messages, is 58 static cells / 32 rendered cells against 60 — the commit's own "shortened to 58 cells" claim re-measured exactly.
- The spaced em dash in the three receipts mirrors the English source punctuation and this locale's existing practice (77 occurrences of " — " in the existing zh-CN catalogues); not a finding.

## Caveats

- This is a string-level audit. No native terminal, relay SPA, Electron or rendered-frame pass was performed here, so truncation in a real paint, font-cell width and interaction behaviour are not claimed.
- zh-CN remains in `SECURITY_HELD_LOCALES` (`scripts/i18n/generate.py`) and out of `data/shipped.json`; this record changes no eligibility, ledger row or other locale.
- Translator provenance: the tree records no zh-CN translator provenance file (only `i18n/provenance/ur-core.translator.json` exists), so `record.json` names the commit as the only provenance source and leaves the translator's agent/model/date null rather than inventing them.

## Evidence files

- `record.json`
- `receipt.json`
- `report.md`
- `audit.log`
- `mechanical-raw.json`
- `runtime-mechanical.json`
- `pseudo-evidence.json`
- `relay.ask.pairs.txt`
- `relay.ask.compact.txt`
- `audit.py.txt`
