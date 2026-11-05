# Translation receipt — PR #3433 translation delta

- Translator: independent i18n-translator subagent, session `1d36d618f249`.
- Actual model: `openai/gpt-6.1-sol` (session runtime metadata); no model fallback used.
- Date: 2026-11-04.
- Requested head: `bd9ad6f041ca`.
- Scope: 7 requested tui.sound keys × 7 locales = 49 entries. No repo files edited; no review/audit verdict claimed.
- Output: `translations.json`, locale → fully qualified key → value.
- Sources: worktree `local_operator/i18n/catalogues/en/tui.sound.json` and `tui.sound.context.json`; `i18n/glossary.md`; `i18n/style-guides/{es,fr,hi,ru,ur,vi,zh-CN}.md`.
- Prior round terminology source: `/Users/damian/.local-operator/sessions/7c5c172b0e89/scratchpad/notif-sounds/c4-i18n-audit-r1/overlay/local_operator/i18n/catalogues/<locale>/tui.sound.json`. Used for sound nouns and user-sound headings, not as authority over the changed English.

## Style decisions

- All id_help values start with the arrow-navigation playback fact before add/doctor/system. Full help intentionally exceeds 96 cells, matching the explicit assignment; truncation must not hide that first fact.
- Bounded headings/hints are terse; list_yours_hint consists solely of the unchanged command pair. Russian list_hint uses compact noun glosses (выбор / звук); Hindi and Urdu use short polite verbs. All bounded values except the intentionally long id_help fit.
- Spanish tú; French vous with curly apostrophe/required narrow spaces; Hindi आप register and danda; Russian lowercase вы/impersonal; Urdu respectful register, Urdu punctuation, logical storage without manual bidi controls; Vietnamese compact pronoun-light NFC; Chinese concise 你 register, full-width prose punctuation, no added Chinese/Latin spacing.
- Literal decimal `1.5` retained as required by this sidecar, rather than introducing locale decimal formatting into an untranslated numeric literal. ASCII `0-100` and the new `[--select]` preserved.
- Commands, subcommand words and placeholder forms retained exactly. Usage prose alone is translated.

## Verification derived in this session

Existing repository Python + `rich.cells.cell_len` measured serialized JSON. Assertions passed for exact locale key sets, NFC, no manual bidi controls, ordered placeholder/backtick spans, ordered /sound command forms, all usage literal forms including `[--select]` and ASCII range, and every bounded heading/hint. No dependency installs or visual rigs.

| Locale | id_help (exception, cap 96) | list_title (40) | list_yours (30) | list_hint (54) | list_yours_hint (54) |
|---|---:|---:|---:|---:|---:|
| es | 149 | 25 | 5 | 49 | 46 |
| fr | 165 | 22 | 10 | 50 | 46 |
| hi | 132 | 10 | 3 | 43 | 46 |
| ru | 155 | 17 | 4 | 48 | 46 |
| ur | 186 | 15 | 5 | 47 | 46 |
| vi | 170 | 18 | 7 | 47 | 46 |
| zh-CN | 119 | 10 | 8 | 45 | 46 |

Unverified: independent linguistic audit and rendered integration; those belong to the delegator’s audit/integration stream.

## Source digests (SHA-256)

- `local_operator/i18n/catalogues/en/tui.sound.json`: `ba467980099545040c0860e47d673d50b5f2f60cd261426538ccf64c3a13e888`
- `local_operator/i18n/catalogues/en/tui.sound.context.json`: `900c1a700da76e20651b1d976ac9c1cc6add638d5b61ce10fffae08869924fad`
- `i18n/glossary.md`: `0eb34437a8c3d6f564f278ef957a603b1d9f51bb07d5ddae5ea8f37deba962f8`
- `i18n/style-guides/es.md`: `4a6f6270f028e5f5fbca7156a8f5099077f92585db94c14653482e9b188248a6`
- `i18n/style-guides/fr.md`: `5315e1ac10c2ce57aac631ba653d08de680ee70c9a022ed3a8345ce2f9902ee7`
- `i18n/style-guides/hi.md`: `82292dc8d13fcc93dc752457719264aed3cc20bb65ecfaba0f0e5a31f4cb40d5`
- `i18n/style-guides/ru.md`: `f20e1cafaf70578c17ce072c71725628808d228e4be5604329f8da30386f1fb5`
- `i18n/style-guides/ur.md`: `c0eecf7f7b5e89956b7de9ab7a9eb647ff1a7ef701ccf2a7f0264d5eea15ca60`
- `i18n/style-guides/vi.md`: `e4901497ed553e2b2ae4a2660bf776adb787b89e11d7eb0f8d08f3da4b227888`
- `i18n/style-guides/zh-CN.md`: `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c`
