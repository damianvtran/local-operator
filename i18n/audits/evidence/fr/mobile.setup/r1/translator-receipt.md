# Translator receipt — fr / mobile.setup — repoint family (3 keys)

- Translator: `task` teammate (i18n translator lane), session `86c6d01615c6`, team `lopdev`.
- Actual model: `deepseek/deepseek-flash` — a quota-driven cross-vendor fallback. The session was pinned to `anthropic/claude-opus-5-5` and descended twice on provider quota (`anthropic/claude-opus-5-5` → `openai/gpt-6.1-sol` → `deepseek/deepseek-flash`). The text below was drafted, measured and written while running `deepseek/deepseek-flash`. This is not a model choice of the translator's; the ledger should record the fallback rather than the pinned id.
- Date: 2026-11-08. Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch
  `fix/tunnel-setup-repoint-successor`. Authored against catalogue content as of `d7c64e9a1fa`; that content is
  unchanged at `6200cf9bf49` (the parent of this commit, which touches only tunnel Python — no i18n path).
- Scope: 3 English keys × this one locale. Evidence slug: `r1`.
- Sources read: `docs/I18N.md` (“Add a string”, fallback/coverage rules), `i18n/glossary.md`,
  `i18n/style-guides/fr.md`, the English catalogue and its context sidecar, and the shipped
  `local_operator/i18n/catalogues/fr/mobile.setup.json` for established register.

## English source (frozen — not reworded, not edited)

| Key | English | en cells |
|---|---|---:|
| `mobile.setup.title_repoint` | Phone access needs reconnecting to this device’s newer sign-in | 62 |
| `mobile.setup.repoint` | A newer sign-in to the same Radient account is already on this device. Reconnect phone access to that sign-in — same account, same phone address. | 145 |
| `mobile.setup.repointed` | Reconnected phone access to this device’s newer sign-in for the same account (credential {id}). | 95 |

Source hash, as given and re-derived here:
`local_operator/i18n/catalogues/en/mobile.setup.json` sha256 = `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4` ✓ (matches).

Other policy inputs (sha256, this tree):

- `local_operator/i18n/catalogues/en/mobile.setup.context.json` = `2cf80ced83e19efddf0414fac6daed2c424fdcb13ca70f303d74471caf03c22a`
- `i18n/glossary.md` = `0eb34437a8c3d6f564f278ef957a603b1d9f51bb07d5ddae5ea8f37deba962f8`
- `i18n/style-guides/fr.md` = `5315e1ac10c2ce57aac631ba653d08de680ee70c9a022ed3a8345ce2f9902ee7`

## Catalogue state and ordering (verified, not assumed)

This locale **already carried all 76 earlier `mobile.setup` keys**, so this wave appends three keys to a
complete catalogue rather than creating a partial one — a partial catalogue would have been sanctioned by
`docs/I18N.md`, but it is not the state of the tree at `d7c64e9a1fa`. Verified: 76 keys before, key order
identical to the English order minus the three new keys, no duplicates, every pre-existing value equal to
`git show HEAD:…`.

The three keys are appended after `mobile.setup.corrupt_intent` because the **English source itself** places
the repoint family there, and catalogue key order mirrors English source order
(`tests/unit/i18n/test_catalogues.py:256` asserts order-sensitive key equality). The family therefore sits as
its own trailing group; it is not interleaved into the `title_*` block, and no existing entry was reordered or
dropped.

- Target before: `4d9454e44e389dae5bb75f205e8feb6be9f9938e85e79d6181424a07f6426e81` (76 keys)
- Target after: `57819ccf28a53236e45408481b379b84035f46b31d63fb9c55f1f7586a1b724b` (79 keys)
- Diff shape: one line gains a trailing comma, three lines added — nothing else.

## Translated values (verbatim)

| Key | fr | cells |
|---|---|---:|
| `mobile.setup.title_repoint` | L’accès téléphone doit être reconnecté à la connexion la plus récente de cet appareil | 85 |
| `mobile.setup.repoint` | Une connexion plus récente au même compte Radient se trouve déjà sur cet appareil. Reconnectez l’accès téléphone à cette connexion — même compte, même adresse de téléphone. | 172 |
| `mobile.setup.repointed` | Accès téléphone reconnecté à la connexion la plus récente de cet appareil pour le même compte (identifiant {id}). | 113 |

## Locale decisions (`fr`)

Glossary/style inputs: `i18n/glossary.md` §2 (credentials → `identifiants`, device → `appareil`) and
`i18n/style-guides/fr.md` (vous, sentence case, U+2019 apostrophe, « — » with spaces, U+202F before `: ; ! ?`).

- **phone access → « accès téléphone »**, the shipped rendering in this very namespace
  (`L’accès téléphone nécessite une connexion Radient`, `L’accès téléphone est connecté`) and in
  `cli.mobile_help.mobile` (`Accès téléphone`). Not « accès mobile », not « accès au téléphone ».
- **sign-in → « connexion »**, matching `relay.login.page.title` (`sign in` → `connexion`) and
  `mobile.setup.title_login` (`une connexion Radient`). Verb form in the remedy is « reconnectez »
  (vous imperative), the same register as `mobile.setup.login`'s « connectez-vous ».
- **credential → « identifiant »** (singular for one row), matching the shipped `(identifiant {id})` in
  `mobile.setup.account` and `Identifiants : {ids}` in `mobile.setup.credential_ids`. The glossary's fr cell
  says `identifiants`; the singular is the same term, inflected — not a second rendering.
- **account → « compte »**, **device → « appareil » / « cet appareil »**, **phone address → « adresse de
  téléphone »** (the shipped `reservation_saved` says `Adresse du téléphone`; the possessive form is used here
  because the source says “same phone address”).
- **“newer” → « la plus récente »** in the title and the receipt, « plus récente » in the remedy's first
  sentence. A bare comparative (« une connexion plus récente ») leaves “newer than what?” unanswered in a
  title; the English means the newer of the two rows on this device, and « la plus récente » states exactly the
  row being adopted. Flagged below.
- Typography: U+2019 in `L’accès` / `l’accès` (never ASCII `'`); the single em dash is in the remedy only,
  spaced « — » per the fr guide; no U+202F was needed because no new string contains `: ; ! ?` in prose.
  Sentence case, no Title Case, no « Veuillez ».

### Flags for the auditor

1. **« connexion » is polysemous.** It renders both *sign-in* and *connection*, and this namespace uses it for
   the network sense too (`mobile.setup.verification`: `La connexion n’est pas encore confirmée`). I kept
   « connexion » because `title_login` and `relay.login` already ship it for sign-in and inventing
   « identification » would give one concept two names; the account phrase (`au même compte Radient`,
   `à cette connexion`) is what disambiguates inside these three strings. If the auditor wants a distinct term,
   that is a namespace-wide decision, not a translation of these three keys.
2. **“newer” → « la plus récente »** (title/receipt) is a deliberate narrowing from comparative to superlative.
   The alternative « une connexion plus récente » is faithful but reads as an unfinished comparison in a
   title. Auditor's call.
3. **Length**: the title is 85 cells against English 62 (+37%) — the longest title in this namespace for
   `fr` (next longest: `title_quote_changed`, 61). No `maxCells` is declared for it and the TUI wraps. If a
   title budget is introduced later, the shortening candidate is « L’accès téléphone à reconnecter à la
   connexion la plus récente de cet appareil » (a nominal fragment, ~74 cells) — not used here because every
   shipped title in this namespace is a full predicate.

## Mechanical checks derived in this session

Measured on the written files with the repository interpreter and `rich.cells.cell_len`:

- Key order equals the 79-key English order; 76 pre-existing values byte-identical to `HEAD`; no duplicates.
- `{id}` appears exactly once, byte-identical ICU-subset single-brace form, in `repointed` only; no braces
  anywhere in `title_repoint` or `repoint`; no other placeholder introduced; no plural/select construct.
- No manual bidi control characters in any string (U+200E/200F, U+2066-2069, U+202A-202E).
- NFC-stable: every string equals its `unicodedata.normalize('NFC', …)` form.
- Length (cells, en 62 / 145 / 95 for reference): 85 / 172 / 113 — reported because the desktop shows these
  in a settings panel and the TUI wraps the title/receipt on an 80-column line. No `maxCells` is declared for
  these three keys in the context sidecar, so nothing was treated as a hard budget.
- `scripts/i18n/check.py --catalogues-only` → `0 problem(s)` across all catalogues.
- `pytest tests/unit/i18n/test_language_manifest.py` → 35 passed (the shrink-only coverage-allowance census:
  no `mobile.setup` allowance entries exist, so all seven locales must carry all 79 keys — they now do).
- `pytest tests/unit/i18n/test_catalogues.py` → 22 passed (includes the order-sensitive carried-namespace
  parity assertion). `pytest tests/unit/i18n/test_es_relay_mobile_state_pins.py` → 8 passed.

- Same-code control for the two other i18n test files that are red on this branch
  (`tests/unit/i18n/test_hold_split.py`, `tests/unit/i18n/test_catalogue_sizes.py`): with the HEAD catalogue bytes
  restored (control), the same two files give **13 failed / 6 passed**; with this append they give **12 failed /
  7 passed**, and the failing set is otherwise byte-identical. The 12 remaining failures assert shipped/ledger-driven
  locale activation (`active_locale() == "zh-CN"`, Hindi team-bridge copy, Chinese wire copy) and need the
  generation step and ledger rows the auditors own. None of them reads `mobile.setup`, and none is introduced or
  worsened by this wave.

## Files written

- `local_operator/i18n/catalogues/fr/mobile.setup.json` (three keys appended)
- `i18n/audits/evidence/fr/mobile.setup/r1/translator-receipt.md` (this file)

## Scope and claims

- No audit verdict is claimed. Two independent auditors follow this lane.
- The full `scripts/i18n/check.py` gate was **not** run and no pass is claimed: it needs the ledger rows and
  the generation step the auditors own. What was run is the catalogue slice only
  (`scripts/i18n/check.py --catalogues-only` → `0 problem(s)`), which covers key parity, ICU argument parity
  and placeholder preservation across the catalogues.
- Not run, deliberately: `scripts/i18n/generate.py` (it flips `data/shipped.json` before the ledger rows exist).
- Not touched: `i18n/ledger.json`, `i18n/coverage-allowance.json`, `i18n/glossary.md`, any style guide, the
  English catalogue and its context sidecar, any file outside
  `local_operator/i18n/catalogues/<locale>/` and this evidence directory.
- No rendered-frame, TUI or desktop evidence is claimed for these strings.
