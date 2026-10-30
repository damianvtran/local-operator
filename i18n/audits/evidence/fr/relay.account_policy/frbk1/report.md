# fr audit - `relay.account_policy` (round frbk1)

- locale `fr` / namespace `relay.account_policy` / **13 keys read**
- audit date 2026-10-29 / auditor translation-auditor (lopdev team), session `ab2610514b4d`, served model `radient/auto`
- fallback disclosed: session opened on anthropic/claude-sonnet-5-5; provider quota failure failed over openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, which served this audit
- translator: translator (lopdev team), model `radient/auto`; this audit is independent of it and edited nothing

## Inputs (sha256)

- `en/relay.account_policy.json` - `0019b08106e6f7762194b164b9b9b37054a8cfade759aa2f6490787a73dea890`
- `en/relay.account_policy.context.json` - `7e55e4fadebcde0a6b93d545a62b2743fbb16009498cdf4e2122cb309ecc07f4`
- `fr/relay.account_policy.json` - `61d4a6c52e2afd6afb4a83bc403995439da198d416507073832aaa26ce71874d`

## Method

Key parity and order against `en`; placeholder, argument-kind and ICU-selector integrity through the repo parser (`message_arguments` / `plural_selectors` / `parse_message`), not regex; French plurals rendered through `render_message` at 0, 1, 2, 1.5e6, 1e6 and 2e6; typography by codepoint enumeration against `i18n/style-guides/fr.md`; glossary consistency against the tracked fr corpus and the peer backfill files on this branch; `scripts/i18n/check.py --catalogues-only` run read-only. Command output is in `audit.log`.

## Findings

### FRBK1-ACP-01 - minor - « routage de compte » vs « routage des comptes » for the same English term

Keys: `relay.account_policy.loading`, `relay.account_policy.unavailable`, `relay.account_policy.load_error`

- current: `Chargement du routage de compte…`
- proposed: `Chargement du routage des comptes…`
- current: `Les paramètres de routage de compte ne sont pas disponibles sur ce relais. Mettez à jour Local Operator sur cet ordinateur.`
- proposed: `Les paramètres de routage des comptes ne sont pas disponibles sur ce relais. Mettez à jour Local Operator sur cet ordinateur.`
- current: `Impossible de charger le routage de compte. Vérifiez la connexion et réessayez.`
- proposed: `Impossible de charger le routage des comptes. Vérifiez la connexion et réessayez.`

**Why:** One English term, two local terms: this namespace writes the bare compound « routage de compte », while the fr wire copy for the same feature writes « Le routage des comptes est indisponible. » Glossary consistency requires one.

« routage des comptes » also reads as the definite plural the screen actually describes — routing among accounts — rather than an unattached compound.

### FRBK1-ACP-02 - nit - « Enregistrer la politique » is 2.18x its English button label

Keys: `relay.account_policy.save`

- current: `Enregistrer la politique`
- proposed: `Enregistrer`

**Why:** A submit control carries the tightest width budget in the set; with no declared `maxCells` this is a rendered-check flag, and the screen's own context supplies the object if the shorter label is preferred.

English source « Save policy » is 11 chars; the French is 24.

## Overflow

No `maxCells` is declared in this namespace's English sidecar, so nothing pins the width; this is the growth picture for a rendered check.

| key | en | fr | ratio |
|---|---|---|---|
| `relay.account_policy.save` | 11 | 24 | 2.18x |
| `relay.account_policy.saving` | 7 | 15 | 2.14x |
| `relay.account_policy.title` | 17 | 26 | 1.53x |
| `relay.account_policy.scope` | 36 | 54 | 1.50x |
| `relay.account_policy.back` | 4 | 6 | 1.50x |

## Verdict - `relay.account_policy`

**findings** - 13 keys read; 0 blocker, 0 major, 1 minor, 1 nit.

