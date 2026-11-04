# Independent fr audit — `wire.tool_rows.bridge.skipped` (rg6a1)

Date: 2026-11-04. Auditor: i18n-auditor (radientdev team) / `3dcc5dbc44d0` / actual model `anthropic/claude-opus-5-5` (session runtime header; no fallback).
Original translator: coder (PR #3417 implementer) / `62d5b457bcf6` / `anthropic/claude-opus-5-5` (from the PR #3417 body).
Same model family as both translators — disclosed; independent sessions, no cross-model second pass.

## Verdict

**passed** — 3 keys read in their composed sentences (collapsed row `{verb} {target} · {preview} · {outcome}` and expanded headline/outcome). Findings: 0 blocker / 0 major / 0 minor / 1 nit; unresolved blocker/major: 0.

## Actual checks

`audit.log` is the real command output of `recheck.py.txt` run against the post-remediation catalogue bytes (key parity, ICU parse, argument parity, render identity, NFC, no bidi/Cf controls, Rich cell widths, and composed rows through `bridge_row`/`bridge_row_line`/`bridge_expanded` with a real `Skipped(seconds_ago=2)` receipt, for a #channel, a named DM recipient and both fallback targets). The sidecar declares no `maxCells`; widths are recorded, not gated. Glossary and style-guide reads, and the skip-term comparison against existing catalogue usage (`tui.import`, `relay.asks.card.skip`, `cli.hub_shared_help`), are audit notes, not execution logs.

## Findings

### FR-RG6-1 — NIT — `wire.tool_rows.bridge.skipped.verb.dm`

- Current: Message ignoré pour
- Proposed: (keep)
- Reason: «Message ignoré pour Maya» is understandable; «pour» is looser than the audited «Envoyé à», but «ignoré à» is ungrammatical and target-last forbids «Envoi à … ignoré». Accepted as-is.
- Disposition: not edited; accepted


## Side-by-side (audited bytes)

### `wire.tool_rows.bridge.skipped.expanded`

- EN: Not posted: an identical message had just been sent, so this copy was skipped.
- fr: Non publié : un message identique venait d’être envoyé, cette copie a donc été ignorée.
- Context note: Outcome sentence in the expanded card for a skipped exact double send (see verb.channel_post): explains WHY nothing was posted. The first copy may still have be…

### `wire.tool_rows.bridge.skipped.verb.channel_post`

- EN: Skipped post in
- fr: Publication ignorée dans
- Context note: Leading verb phrase of the collapsed row when the reply guard deliberately did NOT send a 'channel_post' because it was an exact copy of a message this agent se…

### `wire.tool_rows.bridge.skipped.verb.dm`

- EN: Skipped message to
- fr: Message ignoré pour
- Context note: The 'dm' (direct message) twin of verb.channel_post: the reply guard deliberately did not send an exact copy of a direct message this agent had just sent. Follo…

## Limitations

- LLM linguistic audit, not a native-speaker attestation.
- No terminal bidi/shaping or rendered-frame claim; rendered frames belong to the PR's design round.
