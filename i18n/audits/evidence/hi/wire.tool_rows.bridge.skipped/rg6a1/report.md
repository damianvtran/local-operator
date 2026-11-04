# Independent hi audit — `wire.tool_rows.bridge.skipped` (rg6a1)

Date: 2026-11-04. Auditor: i18n-auditor (radientdev team) / `3dcc5dbc44d0` / actual model `anthropic/claude-opus-5-5` (session runtime header; no fallback).
Original translator: coder (PR #3417 implementer) / `62d5b457bcf6` / `anthropic/claude-opus-5-5` (from the PR #3417 body). Retranslation of `verb.channel_post`, `verb.dm`: i18n-translator (task subagent of this auditor) / `c9f77e05a801` / `anthropic/claude-opus-5-5`, commissioned by this auditor; the auditor drafted no value.
Same model family as both translators — disclosed; independent sessions, no cross-model second pass.

## Verdict

**passed** — 3 keys read in their composed sentences (collapsed row `{verb} {target} · {preview} · {outcome}` and expanded headline/outcome). Findings: 0 blocker / 1 major / 1 minor / 0 nit; unresolved blocker/major: 0.

## Actual checks

`audit.log` is the real command output of `recheck.py.txt` run against the post-remediation catalogue bytes (key parity, ICU parse, argument parity, render identity, NFC, no bidi/Cf controls, Rich cell widths, and composed rows through `bridge_row`/`bridge_row_line`/`bridge_expanded` with a real `Skipped(seconds_ago=2)` receipt, for a #channel, a named DM recipient and both fallback targets). The sidecar declares no `maxCells`; widths are recorded, not gated. Glossary and style-guide reads, and the skip-term comparison against existing catalogue usage (`tui.import`, `relay.asks.card.skip`, `cli.hub_shared_help`), are audit notes, not execution logs.

## Findings

### HI-RG6-1 — MAJOR — `wire.tool_rows.bridge.skipped.verb.channel_post + wire.tool_rows.bridge.skipped.verb.dm`

- Current: पोस्ट छोड़ दी / DM छोड़ दिया
- Proposed: दोबारा पोस्ट नहीं किया / दोबारा DM नहीं भेजा
- Reason: «DM/संदेश छोड़ना» is the 'leave a message' idiom, so the row can read as SENT; «पोस्ट छोड़ दी» is ambiguous (gave up / left a post). 'Did not post/send again' states a deliberate non-send, distinct from failed «नहीं कर सका».
- Disposition: fixed in this audit's commit (retranslated by i18n-translator c9f77e05a801)

### HI-RG6-2 — MINOR — `wire.tool_rows.bridge.skipped.verb.channel_post + wire.tool_rows.bridge.skipped.verb.dm`

- Current: दोबारा DM नहीं भेजा Maya
- Proposed: (code) a {target}-bearing template, e.g. «Maya को दोबारा DM नहीं भेजा»
- Reason: Measured: the row composes f'{verb} {target}', so hi renders verb BEFORE target — contrary to the PR body's claim. Same as audited «DM भेजा Maya». Natural SOV order needs a templated key, a code change.
- Disposition: not edited; inherited composition; non-blocking


## Side-by-side (audited bytes)

### `wire.tool_rows.bridge.skipped.expanded`

- EN: Not posted: an identical message had just been sent, so this copy was skipped.
- hi: पोस्ट नहीं हुआ: ठीक यही संदेश अभी-अभी भेजा गया था, इसलिए यह प्रति छोड़ दी गई।
- Context note: Outcome sentence in the expanded card for a skipped exact double send (see verb.channel_post): explains WHY nothing was posted. The first copy may still have be…

### `wire.tool_rows.bridge.skipped.verb.channel_post`

- EN: Skipped post in
- hi: दोबारा पोस्ट नहीं किया  (was: पोस्ट छोड़ दी)
- Context note: Leading verb phrase of the collapsed row when the reply guard deliberately did NOT send a 'channel_post' because it was an exact copy of a message this agent se…

### `wire.tool_rows.bridge.skipped.verb.dm`

- EN: Skipped message to
- hi: दोबारा DM नहीं भेजा  (was: DM छोड़ दिया)
- Context note: The 'dm' (direct message) twin of verb.channel_post: the reply guard deliberately did not send an exact copy of a direct message this agent had just sent. Follo…

## Limitations

- LLM linguistic audit, not a native-speaker attestation.
- No terminal bidi/shaping or rendered-frame claim; rendered frames belong to the PR's design round.
