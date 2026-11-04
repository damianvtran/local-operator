# fr audit - `mobile.setup` (round frbk1)

- locale `fr` / namespace `mobile.setup` / **76 keys read**
- audit date 2026-10-29 / auditor translation-auditor (lopdev team), session `ab2610514b4d`, served model `radient/auto`
- fallback disclosed: session opened on anthropic/claude-sonnet-5-5; provider quota failure failed over openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto, which served this audit
- translator: translator (lopdev team), model `radient/auto`; this audit is independent of it and edited nothing

## Inputs (sha256)

- `en/mobile.setup.json` - `6249b10379beb9b2433746416fa7baff061265a1b9315d47465140cd5fcf61e9`
- `en/mobile.setup.context.json` - `be33c74a329196fdbe78f11f015848ff4f9593add57eba414163d4042807c811`
- `fr/mobile.setup.json` - `4c0bd0d04f4946e04e4eff23fab925f674df6568b9d3adfbd092dcb3df86894b`

## Method

Key parity and order against `en`; placeholder, argument-kind and ICU-selector integrity through the repo parser (`message_arguments` / `plural_selectors` / `parse_message`), not regex; French plurals rendered through `render_message` at 0, 1, 2, 1.5e6, 1e6 and 2e6; typography by codepoint enumeration against `i18n/style-guides/fr.md`; glossary consistency against the tracked fr corpus and the peer backfill files on this branch; `scripts/i18n/check.py --catalogues-only` run read-only. Command output is in `audit.log`.

## Findings

### FRBK1-MOB-01 - major - « ne présumez pas d'un retour arrière » is ungrammatical and muddles a billing-safety instruction

Keys: `mobile.setup.billing_unconfirmed`

- current: `Facturation : l'activation a été demandée mais son résultat n'est pas confirmé. Vérifiez le compte avant de réessayer ; ne présumez pas d'un retour arrière.`
- proposed: `Facturation : l'activation a été demandée mais son résultat n'est pas confirmé. Vérifiez le compte avant de réessayer ; ne partez pas du principe qu'un retour arrière a eu lieu.`

**Why:** `présumer de` means to overestimate (« présumer de ses forces »), not to assume a state; the intended warning — do not assume a rollback happened — needs `ne présumez pas que…` or `ne partez pas du principe que…`. The same English clause is already rendered with the correct construction elsewhere in the fr wave.

Severity is major rather than minor because this is the billing branch telling the operator not to assume a rollback: a native reader stumbles exactly where the warning has to be unambiguous. The proposed text reuses the construction already used for the identical English sentence in the peer backfill `wire.tunnel_setup` (untracked on this branch, so a parallel rendering rather than shipped precedent).

### FRBK1-MOB-02 - minor - « phone address » is rendered three different ways inside one namespace

Keys: `mobile.setup.stage_reservation`, `mobile.setup.reserving`, `mobile.setup.reservation_saved`, `mobile.setup.reservation_existing`, `mobile.setup.reservation_unconfirmed`

- current: `réservation de l'adresse téléphone`
- proposed: `réservation de l'adresse du téléphone`
- current: `Application en cours — réservation de l'adresse de votre téléphone…`
- proposed: `unchanged (English says « your »)`
- current: `Adresse du téléphone : réservation enregistrée sur cet ordinateur ; une nouvelle tentative la réutilise.`
- proposed: `unchanged`
- current: `Adresse du téléphone : réservation existante conservée.`
- proposed: `unchanged`
- current: `Adresse du téléphone : résultat de la réservation non confirmé ; une nouvelle tentative réutilise la clé de demande enregistrée.`
- proposed: `unchanged`

**Why:** Three forms name the same object: « adresse téléphone » (compound, no preposition), « adresse de votre téléphone » (possessive, mirroring English « your ») and « adresse du téléphone » (definite). The bare compound is the outlier; the other two are defensible as grammar, not as different terms.

Unifying on « adresse du téléphone » costs 3 characters on the `stage_reservation` label (34 -> 37). If that label's budget is tight, keep the compound there deliberately and record it — the finding is the unrecorded drift, not the length.

### FRBK1-MOB-03 - nit - « ID des identifiants » is tautological

Keys: `mobile.setup.credential_ids`

- current: `ID des identifiants : {ids}`
- proposed: `Identifiants : {ids}`

**Why:** `identifiant` is this corpus's established rendering of `credential`, so the label reads « ID of the identifiers »; the list itself already shows IDs, and English « Credential IDs » is carried by naming the field plainly.

Placeholder `{ids}` is unaffected either way.

### FRBK1-MOB-04 - nit - « Radient » attaches only to the route in a coordinated phrase

Keys: `mobile.setup.account_failed`

- current: `La nouvelle vérification du compte et de la route Radient a échoué. Vérifiez la connectivité et lop login-status ; aucune écriture de configuration n'a été faite.`
- proposed: `La nouvelle vérification du compte Radient et de la route a échoué. Vérifiez la connectivité et lop login-status ; aucune écriture de configuration n'a été faite.`

**Why:** English says « Radient account/route verification » — both are Radient's. Placed after the second noun only, the modifier reads as qualifying the route alone; a status message, so low risk, but the fix is free.

No placeholder involved; `lop login-status` stays byte-identical.

## Overflow

No `maxCells` is declared in this namespace's English sidecar, so nothing pins the width; this is the growth picture for a rendered check.

| key | en | fr | ratio |
|---|---|---|---|
| `mobile.setup.reserving` | 40 | 67 | 1.68x |
| `mobile.setup.reservation_unconfirmed` | 81 | 128 | 1.58x |
| `mobile.setup.reservation_saved` | 67 | 104 | 1.55x |
| `mobile.setup.title_balance` | 33 | 51 | 1.55x |
| `mobile.setup.title_failed` | 35 | 54 | 1.54x |

## Verdict - `mobile.setup`

**findings** - 76 keys read; 0 blocker, 1 major, 1 minor, 2 nit.

