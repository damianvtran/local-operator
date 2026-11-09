# Audit report — fr / mobile.setup — repoint family (3 keys), round 1

- Verdict: **PASSED** — 0 blocker / 0 major / 1 minor / 2 nit.
- Auditor: independent translation auditor (lopdev team, `task` subagent), session e7b6613db529; did not translate or edit any value it judged
- Auditor model: deepseek/deepseek-flash — harness-disclosed cross-vendor fallback. The session was pinned to anthropic/claude-opus-5-5 and descended on provider quota (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash) before any string was read or judged. Recorded as the fallback, not the pinned id; no native-speaker pass is claimed.
- Translator (from its receipt): i18n translator lane, `task` teammate of the lopdev team, session 86c6d01615c6 — deepseek/deepseek-flash — quota-driven cross-vendor fallback; the lane was pinned to anthropic/claude-opus-5-5 and descended on provider quota (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash). Stated by the translator's own receipt in this directory, which says the text was drafted and written while running the fallback; recorded as such per docs/I18N.md step 3.
- Source: `local_operator/i18n/catalogues/en/mobile.setup.json` sha256 `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4`
- Target: `local_operator/i18n/catalogues/fr/mobile.setup.json` sha256 `57819ccf28a53236e45408481b379b84035f46b31d63fb9c55f1f7586a1b724b`
- Revision under audit: `0df4ddea13324dc8dd5feb1967caa29a178017c6` (branch `fix/tunnel-setup-repoint-successor`)

## Strings audited

| Key | English (cells) | Locale (cells) |
|---|---|---|
| `mobile.setup.title_repoint` | Phone access needs reconnecting to this device’s newer sign-in (62) | L’accès téléphone doit être reconnecté à la connexion la plus récente de cet appareil (85) |
| `mobile.setup.repoint` | A newer sign-in to the same Radient account is already on this device. Reconnect phone access to that sign-in — same account, same phone address. (145) | Une connexion plus récente au même compte Radient se trouve déjà sur cet appareil. Reconnectez l’accès téléphone à cette connexion — même compte, même adresse de téléphone. (172) |
| `mobile.setup.repointed` | Reconnected phone access to this device’s newer sign-in for the same account (credential {id}). (95) | Accès téléphone reconnecté à la connexion la plus récente de cet appareil pour le même compte (identifiant {id}). (113) |

## Rendered through the product path

- `mobile.setup.title_repoint` → 'L’accès téléphone doit être reconnecté à la connexion la plus récente de cet appareil'
- `mobile.setup.repoint` → 'Une connexion plus récente au même compte Radient se trouve déjà sur cet appareil. Reconnectez l’accès téléphone à cette connexion — même compte, même adresse de téléphone.'
- `mobile.setup.repointed` → 'Accès téléphone reconnecté à la connexion la plus récente de cet appareil pour le même compte (identifiant 42).'

Layer 1 (`messages.render`, resolver exactly as the tree stands) returns English in this tree because the generated shipped set still lists `en` alone; layer 3 runs the full `render_record → _lookup → runtime.render_message` chain with the shipped gate widened in process only, and shows this locale as the provenance. Details and negative controls: `rendered.txt`.

## Findings

### FR-01 — minor — keys: `mobile.setup.title_repoint`, `mobile.setup.repointed`

- String: L’accès téléphone doit être reconnecté à la connexion la plus récente de cet appareil
- Issue: Two things compound in the bare title: the connect- root doubles («reconnecté à la connexion»), and because «connexion» also carries the network sense in this namespace, a reader who takes that sense reads «la connexion … de cet appareil» as a link/connection rather than as a stored sign-in. The same doubling is in the receipt («Accès téléphone reconnecté à la connexion la plus récente»).
- Assessment: Meaning is not lost: the title is always painted with the remedy body («Une connexion plus récente au même compte Radient …»), which names the account and settles the sense, and the desktop renders it beside the run button. Severity minor, not major, because no reachable state produces the wrong action from it. If a later wave touches fr, «L’accès téléphone doit être rattaché à la connexion la plus récente de cet appareil» (or «… à la connexion Radient la plus récente …») removes the doubled root without changing the reference.
- Disposition: accepted for this lane — recommend a wording pass in the next fr wave (does not gate: 0 blocker / 0 major)

### FR-02 — nit — keys: `mobile.setup.title_repoint`, `mobile.setup.repointed`

- String: … à la connexion la plus récente de cet appareil …
- Issue: Superlative where the English is a comparative; recorded deliberately by the translator.
- Assessment: Referent is unique (single live same-account successor), so the superlative is true wherever the string can render. Keeping the comparative in the body sentence is the right split.
- Disposition: accepted — no change (adjudication fr-b)

### FR-03 — nit — keys: `mobile.setup.repointed`

- String: … pour le même compte (identifiant {id}).
- Issue: English '(credential {id})' becomes '(identifiant {id})'. French «identifiant» prototypically means an identifier / login name, so the parenthetical can be read as naming the account handle rather than the local credential row. `mobile.setup.account` ships the identical parenthetical, so this is the namespace's recorded call, not a new choice, and the glossary's fr rendering for credentials is «identifiants» (glossary §2, note 6).
- Assessment: Glossary-compliant and internally consistent; leaving it alone is correct here. Worth a glossary-side note only if a future fr wave distinguishes a stored credential from an identifier.
- Disposition: accepted — inherited, not this lane's defect

## Adjudications of translator-flagged departures

### fr-a — PASS

- Question: fr uses «connexion» for a sign-in while the same namespace uses it for a network connection (`mobile.setup.verification`: «La connexion n’est pas encore confirmée»). Acceptable or ambiguous?
- Rationale: Acceptable, and the alternative is worse. The polysemy is pre-existing and namespace-wide, not introduced here: `mobile.setup.title_login` ships «L’accès téléphone nécessite une connexion Radient» for the sign-in sense and `mobile.setup.owner_missing` ships «rétablissez la connexion du compte propriétaire», also the sign-in sense. «Connexion» is the ordinary French UI word for sign-in (Se connecter / Connexion); «identification» would be a calque that gives one concept two names in the same CLI, which the fr guide forbids in spirit and the glossary invariant forbids outright. In these three strings the account phrase («au même compte Radient», «à cette connexion») disambiguates. Residual risk is confined to the bare title and is logged as FR-01.

### fr-b — PASS

- Question: fr narrowed 'newer' to the superlative «la plus récente» in the title and the receipt, while es kept a comparative. Acceptable?
- Rationale: Acceptable. (1) The referent is unique: `providers/account_successor.py` returns a successor only when exactly one enabled, non-dead same-account row exists, so 'the most recent sign-in of this device' and 'this device's newer sign-in' denote the same row in every reachable state. (2) French cannot mirror a bare comparative here without a comparison anchor: «une connexion plus récente de cet appareil» reads as an unfinished comparison, which is why the translator kept the comparative only in the body, where «au même compte Radient» supplies the anchor. (3) The es comparison in the flag is not a clean contrast — es's title and receipt are superlatives too (ES-01). Recorded as FR-02, no change requested.

## Checks performed

- key set: 79 keys, identical and order-identical to the English catalogue (tests/unit/i18n/test_catalogues.py asserts order-sensitive parity); the three new keys are the trailing group, exactly where the English source puts them after mobile.setup.corrupt_intent
- append-only: every one of the 76 pre-existing values is byte-identical to `git show 0df4ddea133^:<path>`; the only other diff line is the trailing comma the append needs
- placeholder integrity: `{id}` appears exactly once, in `mobile.setup.repointed` only, in single-brace ICU-subset form; runtime.message_arguments() reports exactly one argument ('id', 'simple'); no brace anywhere in the title or the remedy body
- render gate: the product engine raises MessageFormatError when `id` is omitted, so the placeholder is live ICU, not escaped braces
- no bidi/format control characters (U+200E/F, U+202A-E, U+2066-9) in any of the three values; every value is NFC-stable; no decomposed sequences (vi checked explicitly)
- typography: no ASCII apostrophe, no double space, no space before punctuation, no leading/trailing space, em dash spaced ' — ', no ASCII '...'
- cell widths measured with rich.cells.cell_len against the English source; no `maxCells` is declared for these keys in the context sidecar, so nothing was judged against a hard budget
- does not imply a NEW sign-in (no 'sign in again' reading in any of the three strings; the title obliges a re-point of the existing access)
- does not imply a NEW tunnel, a NEW account or an address change — the remedy names the same account and the same phone address explicitly and the receipt repeats 'same account'
- does not imply a subscription/activation or any service mutation
- the {id} parenthetical is presented as a local credential row reference, not as an account id, email or phone address

## Limitations

- Independent LLM review on a provider-fallback model (deepseek/deepseek-flash after two quota descents), not native-speaker certification; the judgments are reasoned, not native intuition.
- Scalar text evidence only: no terminal frame, desktop frame or device screenshot was rendered. Widths are rich.cells.cell_len measurements against the English source; no maxCells is declared for these keys.
- This lane did not run scripts/i18n/generate.py or scripts/i18n/check.py: the shipped-set snapshot is only correct after every auditor's rows exist, so the shared gates run last under the second auditor.
- Desktop reachability is asserted from the route contract (desktop_tunnel_setup.py ships plan['remedy'] as remedy.text with kind 'run') and from tunnels/setup.py; no desktop build was exercised.

## Ledger row

- Storage: `i18n/audit-history/fr/mobile.setup.json` (ordered suffix), appended with `scripts/i18n/ledger.py append`; the last row per (locale, namespace) is the effective one.
- `i18n/ledger.json` and `i18n/ledger/<locale>.json` are baseline snapshots pinned by `i18n/audit-history.json`; docs/I18N.md forbids rewriting them after migration, so a new pass is an appended row, not an edit of the stale row.
- `source_sha256` = `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4` (current English hash of the namespace).
- Suffix `history_sha256` after this append: `7c5068e961d825c47ea2d7d4b9fec29ad700f35d0cd8114f35f0c2ce6702e9f5`.
- Row: status `passed`, auditor model recorded exactly as disclosed (provider fallback), translator model taken from the translator's own receipt.
