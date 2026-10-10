# Style guide — Urdu (`ur`) — the RTL locale

Read with `i18n/glossary.md`; everything here is ur-specific and binding for
translators and the auditor. This locale carries the RTL pass — read the
RTL & bidi section before translating a single string.

## Formality & tone

- Address: **آپ** — recorded decision: the standard polite address in Urdu
  software; "تم" (intimate) and "تو" (blunt) never appear in product copy,
  however terse the string.
- Terse; natural Urdu — not Hindi-ised (use Urdu vocabulary: ترتیبات،
  استعمال، محفوظ کریں) and not over-Persianised (everyday educated register).
- Commands use the respectful imperative: "محفوظ کریں", "دوبارہ کوشش کریں",
  "منسوخ کریں".
- "براہ کرم" only where the English says "please"; no "!"; no padding
  honorifics.

## Product-name policy

- Keep Latin as-is, untransliterated: `lop`, commands, `--flags`, paths,
  config keys, product names, tool names (glossary §1). Never wrap them in
  Urdu punctuation or reorder them.
- Kept-Latin acronyms stay half-width and undeclined: API, CLI, TUI, JSON,
  YAML, HTTP, URL.
- Settled loanwords take Urdu script: سیشن، ٹوکن، فائل، ریموٹ، ورک اسپیس.
- Prefer the settled Urdu term where one exists: ترتیبات (settings), محفوظ
  کریں (save), استعمال (usage).

## Locale calls (ur)

- session → سیشن (software loanword; نشست means a meeting — do not use it)
- wake → بیداری for the noun label; verb forms inside sentences ("10 بجے
  جاگے گا"). The wave that first translates `wake` strings records the final
  call here if it diverges.
- tool trace → ٹریس · transcript → ٹرانسکرپٹ
- approvals → منظوریاں (singular منظوری) · schedule → شیڈول
- relay → ریلے · workspace → ورک اسپیس

## Plurals

- Categories: `one` / `other`. Example: `{count, plural, one {# فائل} other {# فائلیں}}`
- Some English loanwords stay invariant in the plural (سیشن) — acceptable
  where it reads naturally; otherwise inflect (فائل → فائلیں).

## Numbers, dates, digits

- Latin digits in v1 (§2.8) — Urdu digits (۱۲۳) are deferred and recorded
  here as revisitable without code.
- Never hand-written — the formatter renders "1,234.5" and dates
  "10 اکتوبر، 2026"; am/pm suffix per the formatter.

## RTL & bidi

- Storage is logical order. The render layer wraps each translated string in
  FSI/PDI isolation (zero width) — **do not insert manual bidi control
  characters (RLM/LRM/FSI/PDI) into translations**; they are double-applied.
- LTR islands: code spans, paths, commands, diffs and terminal output stay
  byte-identical Latin; a string that is mostly an LTR token keeps the
  source's order.
- TUI stance: layout stays LTR, `ur` ships best-effort — correct on
  bidi-capable terminals (iTerm2 3.7+), legible-but-unordered elsewhere;
  documented on the locale picker row. This is a known degraded stance, not a
  translation defect — never "fix" a string to compensate.
- Web/Electron/RN mirror fully (`dir=rtl`, logical properties); the strings
  are the same ones.
- Fonts: Nastaliq (Noto Nastaliq / DecoType) — the TUI inherits the user's
  terminal font; glyph shaping is the terminal's job, not the copy's.

## Width rules

- Arabic-script text measures ≈ one cell per character (measured: 14→14);
  FSI/PDI cost 0 cells. Keep within `maxCells`; prefer short.
- Budgets: `maxCells` from the en sidecar; pinned: PERSIST_HINT 42/43 cells,
  tip pool 59, `/help` ≈ 55.

## Punctuation & typography

- Urdu punctuation: ۔ full stop, ، comma, ؟ question mark, ؛ semicolon,
  quotes “ ”. Do not use Hindi (।) or Arabic-only signs.
- Latin punctuation stays Latin inside code spans, flags and numbers
  ("1,234.5"); code spans/flags/paths stay byte-identical.
