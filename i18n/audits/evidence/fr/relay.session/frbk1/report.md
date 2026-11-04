# fr audit - `relay.session` (round frbk1)

- locale `fr` / namespace `relay.session` / **17 keys read**
- audit date 2026-10-29 / auditor translation-auditor (lopdev team), session `ab2610514b4d`, served model `radient/auto`
- fallback disclosed: session opened on anthropic/claude-sonnet-5-5; provider quota failure failed over openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, which served this audit
- translator: translator (lopdev team), model `radient/auto`; this audit is independent of it and edited nothing

## Inputs (sha256)

- `en/relay.session.json` - `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129`
- `en/relay.session.context.json` - `8777c48c750d982f3f795c81d0a543e92000a3898223cfa60637f8a2edc1e2db`
- `fr/relay.session.json` - `100b953cb117cd08812fd8ad27487067b3757b69d1e20da21ac54e9ba813de04`

## Method

Key parity and order against `en`; placeholder, argument-kind and ICU-selector integrity through the repo parser (`message_arguments` / `plural_selectors` / `parse_message`), not regex; French plurals rendered through `render_message` at 0, 1, 2, 1.5e6, 1e6 and 2e6; typography by codepoint enumeration against `i18n/style-guides/fr.md`; glossary consistency against the tracked fr corpus and the peer backfill files on this branch; `scripts/i18n/check.py --catalogues-only` run read-only. Command output is in `audit.log`.

## Findings

None. This section is clean.

Seventeen keys. The two search placeholders deliberately follow their sources' casing — « Rechercher une conversation… » capitalised where English capitalises, « rechercher des noms et des conversations… » lowercase where English is lowercase — so this is fidelity, not drift.

« démarrez-en une ci-dessous » correctly carries the partitive pronoun for « start one », which is the idiomatic French for the elliptical English; `TUI` stays English per the kept-English list.

All four loading/status keys use the same noun form (« chargement… », « démarrage… »), matching the shipped `relay.chrome` and `tui.status` renderings of « connecting… » as « connexion… ». No imperative became a noun or vice versa: the two hints that instruct (« envoyez un message… », « ou saisissez un autre chemin… ») stay second-person imperative.

« dossier de travail » for `working directory` uses the glossary's preferred « dossier ».

## Overflow

No `maxCells` is declared in this namespace's English sidecar, so nothing pins the width; this is the growth picture for a rendered check.

| key | en | fr | ratio |
|---|---|---|---|
| `relay.session.view.empty_title` | 15 | 28 | 1.87x |
| `relay.session.past.back` | 4 | 6 | 1.50x |
| `relay.session.model.filter_placeholder` | 13 | 19 | 1.46x |
| `relay.session.list.start_idle` | 11 | 16 | 1.45x |
| `relay.session.directory.path_placeholder` | 21 | 29 | 1.38x |

## Verdict - `relay.session`

**clean** - 17 keys read, no finding.

