# fr audit - `relay.shell` (round frbk1)

- locale `fr` / namespace `relay.shell` / **44 keys read**
- audit date 2026-10-29 / auditor translation-auditor (lopdev team), session `ab2610514b4d`, served model `radient/auto`
- fallback disclosed: session opened on anthropic/claude-sonnet-5-5; provider quota failure failed over openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, which served this audit
- translator: translator (lopdev team), model `radient/auto`; this audit is independent of it and edited nothing

## Inputs (sha256)

- `en/relay.shell.json` - `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560`
- `en/relay.shell.context.json` - `16746ecd1537006992935bb2aa88fe39c46700d9e0341e7aacd565e2366349ea`
- `fr/relay.shell.json` - `d6aa26b8e8dc15fb27feed031a5595470aeda0f15374ec665312b0bfbfe4c496`

## Method

Key parity and order against `en`; placeholder, argument-kind and ICU-selector integrity through the repo parser (`message_arguments` / `plural_selectors` / `parse_message`), not regex; French plurals rendered through `render_message` at 0, 1, 2, 1.5e6, 1e6 and 2e6; typography by codepoint enumeration against `i18n/style-guides/fr.md`; glossary consistency against the tracked fr corpus and the peer backfill files on this branch; `scripts/i18n/check.py --catalogues-only` run read-only. Command output is in `audit.log`.

## Findings

### FRBK1-SHELL-01 - major - the `many` arm of the three usage-unit plurals omits the required `de`

Keys: `relay.shell.usage.unit_credits`, `relay.shell.usage.unit_tokens`, `relay.shell.usage.unit_requests`

- current: `{count, plural, one {crédit} many {crédits} other {crédits}}`
- proposed: `{count, plural, one {crédit} many {de crédits} other {crédits}}`
- current: `{count, plural, one {token} many {tokens} other {tokens}}`
- proposed: `{count, plural, one {token} many {de tokens} other {tokens}}`
- current: `{count, plural, one {requête} many {requêtes} other {requêtes}}`
- proposed: `{count, plural, one {requête} many {de requêtes} other {requêtes}}`

**Why:** French `many` folds every exact multiple of 10^6, and a following full noun takes `de`. As shipped the composed row renders « restant : 1 000 000 tokens », which no French speaker writes; it also contradicts this locale's binding style-guide example and the shipped fr corpus, both of which write `de`: tracked fr catalogues with `many` arms over a full noun — relay.asks (4), relay.ask, relay.chrome, relay.notifications, relay.session.tags, relay.transcript, tui.asks, tui.notifications (7), tui.review, tui.schedule (4), tui.status (2), tui.transcript (2) — all use `de`.

Adding `de` inside the unit branch is safe for the outer messages: `remaining`/`used` put the number before `{unit}`, so the corrected unit yields « restant : 1 000 000 de tokens » with `de` appearing exactly once. The fix is local to the three `many` arms; `one` and `other` stay as they are.

### FRBK1-SHELL-02 - minor - « éligibilité de dépense » drifts from the established « éligibilité à la dépense »

Keys: `relay.shell.usage.eligibility_unknown`

- current: `Éligibilité de dépense inconnue`
- proposed: `Éligibilité à la dépense inconnue`

**Why:** `éligible`/`éligibilité` takes `à`; the same English string is already rendered « Éligibilité à la dépense inconnue » in the fr wave, so two spellings of one term are in flight.

Costs 2 characters over the current text (31 -> 33); no `maxCells` is declared for this namespace.

### FRBK1-SHELL-03 - nit - « usage.stale » runs 2.16x its English length because the translation supplies a subject

Keys: `relay.shell.usage.stale`

- current: `Ces chiffres peuvent ne plus être à jour.`
- proposed: `Peut ne plus être à jour.`

**Why:** No `maxCells` pins this namespace, so this is a rendered-check flag rather than a violation: the shipped line is 41 characters against the 25-character fragment that carries the same warning, and it renders as a one-line caption under the usage rows, where the extra width buys nothing the reader needs.

English source is 19 chars; the shipped French is 41, the shorter variant 25.

## Overflow

No `maxCells` is declared in this namespace's English sidecar, so nothing pins the width; this is the growth picture for a rendered check.

| key | en | fr | ratio |
|---|---|---|---|
| `relay.shell.usage.title` | 5 | 11 | 2.20x |
| `relay.shell.usage.stale` | 19 | 41 | 2.16x |
| `relay.shell.usage.family_caps` | 30 | 58 | 1.93x |
| `relay.shell.usage.command` | 26 | 46 | 1.77x |
| `relay.shell.usage.loading` | 21 | 37 | 1.76x |

## Verdict - `relay.shell`

**findings** - 44 keys read; 0 blocker, 1 major, 1 minor, 1 nit.

