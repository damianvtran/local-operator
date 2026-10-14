# Style guide — English (`en`) — the source locale

The `en` catalogue is the source every other locale renders from. This guide
fixes its voice and its mechanics so translations inherit intent, not
accidents. Translators: read this to understand what the English means — you
do not translate into `en`.

## Voice & tone

- Terse, direct, problem-first. A short sentence beats a smooth one.
- No marketing adjectives ("seamless", "powerful"), no exclamation marks, no
  "please" outside dialogs that genuinely block the user.
- Sentence case for sentences and labels; deliberate lowercase for fragments
  and status/choice text ("save on request", "show it", "auto"). Never Title
  Case. Do not "fix" a deliberate lowercase fragment into a sentence — the
  lowercase-leaning is a stylistic device translators are asked to *adapt*,
  not to copy literally into locales that capitalise sentences.
- Second person only when the user must act ("Edit it here"); otherwise state
  the fact ("A config write tightens every running session").
- Say what happened and what to do; no "successfully" ("Saved", not
  "Successfully saved").

## Mechanics (contract, not style)

- One message = one complete unit. Never concatenate fragments at a call site;
  if parts must join, make one message with placeholders.
- ICU subset only: `{name}`, `{n, plural, …}`, `{n, select, …}`, number/date
  formatting via the formatter layer — never hand-format numbers, dates or
  times into a string (glossary §3).
- Placeholders lower_snake (`{count}`, `{model}`). Editing a message without
  needing to? Don't — any edit changes the source hash and voids every
  locale's ledger freshness for that namespace. Queue copy edits; don't churn
  messages mid-wave.
- Width-sensitive strings record `maxCells` in the en context sidecar
  (`catalogues/en/<ns>.context.json`). The three pinned budgets (PERSIST_HINT
  42/43 cells, tip pool 59, `/help` ~55) are hard.
- Straight quotes and apostrophes in en copy; `…` for prose ellipsis (`...`
  only inside code-ish enumerations like `(a, b, ...)`); em dash " — " with
  spaces.

## Product-name policy

- Source locale: kept-English is everything, by definition — en copy is the
  reference spelling for every term another guide records as kept-English.
- The classes that never translate anywhere are glossary §1's ("Local
  Operator", `lop`, tool names, `--flags`, config keys, paths, identifiers).

## Plurals

- Categories: `one` (1) / `other` (0, 2, …). Example:
  `{count, plural, one {# session} other {# sessions}}`
- Never hand-roll `count == 1 else` at a call site — every count goes
  through a plural message; `#` renders the formatted number.

## Numbers, dates, digits

- Never hand-written — the formatter renders "1,234.5", and dates
  "Oct 10, 2026"; 12-hour clock with AM/PM ("2:30 PM").
- Digits: Latin.

## Punctuation

- Full stops end sentences; no full stop on labels or fragments.
- Backticked literals (commands, flags, keys, paths) stay byte-identical in
  every locale and are never translated.
- Keep terminology to this repo's glossary — a new concept needs a glossary
  row before it ships in copy.
