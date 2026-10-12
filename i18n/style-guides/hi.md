# Style guide — Hindi (`hi`)

Read with `i18n/glossary.md`; everything here is hi-specific and binding for
translators and the auditor.

## Formality & tone

- Address: **आप** — the standard polite register; तुम / तू never appear in
  product copy.
- Terse; everyday standard Hindi. Prefer common Hindi words over heavy
  Sanskritisation; where an English loan is the living term (टोकन, कैश,
  फ़ाइल), keep it rather than inventing a stilted native form.
- Commands use the polite imperative: "सहेजें", "फिर से शुरू करें",
  "हटाएँ", "रद्द करें".
- "कृपया" only where the English says "please"; no "!"; no padding
  honorifics ("माननीय उपयोगकर्ता" never).
- Keep nukta consistently (फ़, ज़, ड़) — it is part of the spelling, not
  decoration.

## Product-name policy

- Keep Latin as-is: `lop`, commands, `--flags`, paths, config keys, product
  names, tool names (glossary §1) — never transliterated into Devanagari.
- Kept-Latin: API, CLI, TUI, JSON, YAML, HTTP, URL.
- Settled loans in Devanagari: टोकन, कैश, फ़ाइल, फ़ोल्डर, प्रॉम्प्ट,
  डिवाइस, नेटवर्क.
- Prefer the settled Hindi term where one exists: सर्वर, उपयोग (usage),
  लागत (cost), कार्यक्षेत्र (workspace), टीम (team), कौशल (skill).

## Locale calls (hi)

- session → सत्र (standard; "सेशन" rejected as colloquial inconsistency)
- wake → जागृति for the noun label; verb forms inside sentences ("10 बजे
  जागेगा"). The wave that first translates `wake` strings records the final
  call here if it diverges.
- tool trace → ट्रेस · transcript → ट्रांसक्रिप्ट
- approvals → स्वीकृतियाँ (singular स्वीकृति) · schedule → शेड्यूल
- relay → रिले · compaction → संपीडन (संदर्भ संपीडन)
- Wave-3 endorsed calls: runtime → रनटाइम · catch-up → कैच-अप · job → जॉब
- credentials → क्रेडेंशियल (plural क्रेडेंशियल्स) · embedded → अंतर्निहित
- mesh → मेश · surface → सरफ़ेस · manager → मैनेजर · hints → संकेत ·
  hotkeys → हॉटकीज़
- child (agent) and subagent → उप-एजेंट consistently; the earlier child-only
  transliteration was abandoned during settings polish, not kept as an exception.
- monitor as a stored object → मॉनिटर; watching/monitoring as an activity →
  निगरानी. This deliberate noun/activity register split does not change the
  Latin tool name `monitor` (glossary §1).

## Plurals

- Categories: `one` / `other` — and `one` covers **0 and 1**: "0 फ़ाइल" is
  singular. Example: `{count, plural, one {# फ़ाइल} other {# फ़ाइलें}}`
- Use the natural plural (फ़ाइल → फ़ाइलें, सत्र → सत्रों in oblique cases).

## Numbers, dates, digits

- Latin digits in v1 (§2.8) — Devanagari digits (०१२…) are deferred and
  recorded here as revisitable without code.
- Never hand-written — the formatter renders "12,34,567.5" (Indian grouping)
  and dates "10 अक्टू॰ 2026"; am/pm suffix per the formatter.

## Width rules

- Devanagari measures optimistically (conjuncts: क्षत्रिय 8→5 cells); budget
  with `cell_len`, not code points, and stay within `maxCells`.
- Budgets: `maxCells` from the en sidecar; pinned: PERSIST_HINT 42/43 cells,
  tip pool 59, `/help` ≈ 55.
- Prefer a shorter idiomatic wording over truncation.

## Punctuation & typography

- Danda । for full stops in prose; ? and ! as-is; comma ,; quotes “ ”.
- Code spans, flags, paths stay byte-identical — Latin, half-width.
- Use standard conjunct spellings; do not insert ZWJ/ZWNJ where the
  orthography does not require them.
