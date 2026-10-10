# Style guide — TEMPLATE (copy to `<locale>.md` when the wave starts)

Fill every field before the locale's first translation run; the auditor checks
the translated catalogue against THIS file. A wave PR adds
`i18n/style-guides/<locale>.md`; nothing else may be assumed from this
template.

## Formality

- Address: (tu / vous | du / Sie | 您 / 你 | …). Default person and number:
- Tone: (plain / warm / formal). Match the English source's register; do not
  upgrade or downgrade it per string.

## Product-name policy

- Keep English: "Local Operator", "lop", tool names, `--flags`, identifiers,
  config keys, paths, code snippets (see `i18n/glossary.md`).
- Per-locale calls for nouns the glossary leaves open (session/wake/agent/…):

## Kept-English technical list

(Extend the glossary's list with locale-specific additions and the reason.)

## Digits

- v1 default: Latin digits (`hi`, `ur` included — §2.8). Override with the
  reason if this locale's wave decides otherwise:

## RTL (only for `ur`)

- TUI: best-effort/degraded — strings wrap in FSI/PDI isolation; storage order
  preserved (correct on bidi-capable terminals, legible-but-unordered
  elsewhere). Documented user-facing note on the picker row.
- Web/Electron/RN: full mirroring (`dir=rtl`, logical properties, mirrored
  directional glyphs; LTR islands for code/paths/diffs/terminal output).

## Width rules

- TUI strings carry `maxCells` in the en context sidecar; a translation must
  not exceed it (the three pinned budgets: `PERSIST_HINT` 42/43 cells, the
  tip pool at 59, `/help` wrap at ~55).
- When a field cannot fit, prefer a shorter synonym over truncation; note the
  change in the wave's ledger `note` if the budget itself moves.

## Punctuation & typography

- Quotes, dashes, ellipses: follow locale conventions inside prose; keep code
  spans/literals byte-identical to English.
- No non-breaking-space changes inside technical strings (flags, paths).
