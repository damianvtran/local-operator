# fr audit - `relay.library` (round frbk1)

- locale `fr` / namespace `relay.library` / **1 key read**
- audit date 2026-10-29 / auditor translation-auditor (lopdev team), session `ab2610514b4d`, served model `radient/auto`
- fallback disclosed: session opened on anthropic/claude-sonnet-5-5; provider quota failure failed over openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, which served this audit
- translator: translator (lopdev team), model `radient/auto`; this audit is independent of it and edited nothing

## Inputs (sha256)

- `en/relay.library.json` - `ad478c433a4db3e725994766c724a2e8e92176f015f8283e71dc13b87185466a`
- `en/relay.library.context.json` - `877b88290cf14602e63b2bb1c58719fac96a20ae53ff04b147e21de23df93c54`
- `fr/relay.library.json` - `ab64d2fb171c78e79ac689d49da261474e4eaf687cbfbf113d28345234302ea2`

## Method

Key parity and order against `en`; placeholder, argument-kind and ICU-selector integrity through the repo parser (`message_arguments` / `plural_selectors` / `parse_message`), not regex; French plurals rendered through `render_message` at 0, 1, 2, 1.5e6, 1e6 and 2e6; typography by codepoint enumeration against `i18n/style-guides/fr.md`; glossary consistency against the tracked fr corpus and the peer backfill files on this branch; `scripts/i18n/check.py --catalogues-only` run read-only. Command output is in `audit.log`.

## Findings

None. This section is clean.

The single key `relay.library.count` reads `{count, plural, one {# élément} many {# d’éléments} other {# éléments}}`.

Both English arms survive (`one`, `other`) and the added `many` is a real French CLDR category — the repo's `plural_rules.json` lists `[many, one, other]` for `fr`, and `_plural_fr` yields `many` only for exact multiples of 10^6 with no fraction digits.

Rendered through the repo runtime the arm is correct at every boundary that matters: 0 -> « 0 élément », 1 -> « 1 élément », 2 -> « 2 éléments », 1 500 000 -> `other`, « 1 500 000 éléments », 1 000 000 -> `many`, « 1 000 000 d’éléments », 2 000 000 -> « 2 000 000 d’éléments ». The elided `de` arrives as `d’` with U+2019, which is the correct form before a vowel.

This is the right shape for the category: `#` still carries the number, so the count is formatted once by the locale formatter (U+202F group separator), and the wording matches the style guide's own example `many {# de sessions}`.

## Overflow

No `maxCells` is declared in this namespace's English sidecar, so nothing pins the width; this is the growth picture for a rendered check.

| key | en | fr | ratio |
|---|---|---|---|
| `relay.library.count` | 47 | 71 | 1.51x |

## Verdict - `relay.library`

**clean** - 1 key read, no finding.

