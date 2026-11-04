# fr audit - `relay.login` (round frbk1)

- locale `fr` / namespace `relay.login` / **4 keys read**
- audit date 2026-10-29 / auditor translation-auditor (lopdev team), session `ab2610514b4d`, served model `radient/auto`
- fallback disclosed: session opened on anthropic/claude-sonnet-5-5; provider quota failure failed over openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, which served this audit
- translator: translator (lopdev team), model `radient/auto`; this audit is independent of it and edited nothing

## Inputs (sha256)

- `en/relay.login.json` - `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5`
- `en/relay.login.context.json` - `9fcd5a1669e6831b10b7041d82f77c69e858183c001ba9faa949532b4d71980a`
- `fr/relay.login.json` - `5f6cc95bffee5f11984b4a083af057ba96a5e332a623c5b5c3659a9e61530d24`

## Method

Key parity and order against `en`; placeholder, argument-kind and ICU-selector integrity through the repo parser (`message_arguments` / `plural_selectors` / `parse_message`), not regex; French plurals rendered through `render_message` at 0, 1, 2, 1.5e6, 1e6 and 2e6; typography by codepoint enumeration against `i18n/style-guides/fr.md`; glossary consistency against the tracked fr corpus and the peer backfill files on this branch; `scripts/i18n/check.py --catalogues-only` run read-only. Command output is in `audit.log`.

## Findings

None. This section is clean.

Four keys, all matching the sidecar's « preserve current login copy and casing » brief.

`page.title` keeps the brand lowercase and the em dash: « local operator — connexion »; « connexion » is the standard French rendering of a sign-in page title.

`password.label` « mot de passe » and `submit.label` « se connecter » keep English's lowercase label casing; the infinitive is the correct French button form, so the control still reads as an action.

`error.password` « Mot de passe incorrect. » keeps sentence case and the closing period, and states the failure as plainly as the English does.

## Overflow

No `maxCells` is declared in this namespace's English sidecar, so nothing pins the width; this is the growth picture for a rendered check.

| key | en | fr | ratio |
|---|---|---|---|
| `relay.login.submit.label` | 7 | 12 | 1.71x |
| `relay.login.error.password` | 15 | 23 | 1.53x |
| `relay.login.password.label` | 8 | 12 | 1.50x |
| `relay.login.page.title` | 24 | 26 | 1.08x |

## Verdict - `relay.login`

**clean** - 4 keys read, no finding.

