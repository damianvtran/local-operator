# Translator receipt — es / mobile.setup — repoint family (3 keys)

- Translator: `task` teammate (i18n translator lane), session `86c6d01615c6`, team `lopdev`.
- Actual model: `deepseek/deepseek-flash` — a quota-driven cross-vendor fallback. The session was pinned to `anthropic/claude-opus-5-5` and descended twice on provider quota (`anthropic/claude-opus-5-5` → `openai/gpt-6.1-sol` → `deepseek/deepseek-flash`). The text below was drafted, measured and written while running `deepseek/deepseek-flash`. This is not a model choice of the translator's; the ledger should record the fallback rather than the pinned id.
- Date: 2026-11-08. Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch
  `fix/tunnel-setup-repoint-successor`. Authored against catalogue content as of `d7c64e9a1fa`; that content is
  unchanged at `6200cf9bf49` (the parent of this commit, which touches only tunnel Python — no i18n path).
- Scope: 3 English keys × this one locale. Evidence slug: `r1`.
- Sources read: `docs/I18N.md` (“Add a string”, fallback/coverage rules), `i18n/glossary.md`,
  `i18n/style-guides/es.md`, the English catalogue and its context sidecar, and the shipped
  `local_operator/i18n/catalogues/es/mobile.setup.json` for established register.

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
- `i18n/style-guides/es.md` = `4a6f6270f028e5f5fbca7156a8f5099077f92585db94c14653482e9b188248a6`

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

- Target before: `a488fa214a660045363b56d6edd5a6d6f2d0a2feb1d35c689cc47ccaf51cd535` (76 keys)
- Target after: `7c278ff03e2bb4e2a2cd04c7b005cf4c35fc7114055b141a5681b11f19adedb1` (79 keys)
- Diff shape: one line gains a trailing comma, three lines added — nothing else.

## Translated values (verbatim)

| Key | es | cells |
|---|---|---:|
| `mobile.setup.title_repoint` | El acceso telefónico debe reconectarse al inicio de sesión más reciente de este dispositivo | 91 |
| `mobile.setup.repoint` | Este dispositivo ya tiene un inicio de sesión más reciente de la misma cuenta de Radient. Reconecta el acceso telefónico a ese inicio de sesión — misma cuenta, misma dirección de teléfono. | 188 |
| `mobile.setup.repointed` | Acceso telefónico reconectado al inicio de sesión más reciente de este dispositivo para la misma cuenta (credencial {id}). | 122 |

## Locale decisions (`es`)

Glossary/style inputs: `i18n/glossary.md` §2 (credentials → `credenciales`, device → `dispositivo`) and
`i18n/style-guides/es.md` (tú, neutral Spanish, infinitive/imperative labels, “ ” quotes, no space before
`:` `;` `?` `!`, em dash with spaces).

- **phone access → « acceso telefónico »**, the shipped rendering in this namespace
  (`El acceso telefónico necesita un inicio de sesión de Radient`, `El acceso telefónico está conectado`).
  Note this differs from `cli.mobile_help.mobile`, which ships `Acceso desde el teléfono`; the namespace's own
  term wins for namespace copy, and the two are not in the same file. Flagged below.
- **sign-in → « inicio de sesión »**, matching `mobile.setup.title_login` and
  `wire.errors.media.speech_no_credential` (`Inicia sesión en Radient`); verb form « reconecta » (tú).
- **credential → « credencial »** (singular for one row), matching the shipped `(credencial {id})` in
  `mobile.setup.account`; the plural `credenciales` (glossary) is used when more than one is meant.
- **account → « cuenta »**, **device → « dispositivo » / « este dispositivo »**, **phone address → « dirección
  de teléfono »** (shipped `reservation_saved`: `Dirección del teléfono`).
- Neutral-Spanish word choices: `dispositivo` over `móvil`/`celular`; `teléfono` (never `móvil`) so the string
  reads identically in Madrid and Mexico City.
- Remedy sentence 1 is restructured to « Este dispositivo ya tiene un inicio de sesión más reciente de la misma
  cuenta de Radient. » so the source’s “on this device” does not stack two `en`-phrases; the meaning (already
  present, this device, same account) is unchanged and sentence 2 still names the sign-in as « ese inicio de
  sesión ».
- Typography: no space before `:`/`;`/`?`/`!`; one em dash with spaces in the remedy; no quotes needed; sentence
  case; imperative `Reconecta` (tú) per the es guide.

### Flags for the auditor

1. **Two “phone access” renderings ship in the catalogue**: « acceso telefónico » (this namespace,
   `wire.tunnel_setup`) and « acceso desde el teléfono » (`cli.mobile_help.mobile`). I matched the namespace,
   because the three new keys sit beside sixteen titles that already say « acceso telefónico ». If the wave
   standardises on one, it is a wider change than these three keys.
2. **Length**: 91 / 188 / 122 cells against English 62 / 145 / 95. The title is +47%, the longest in absolute
   terms across the seven locales. Nothing is truncated (the TUI wraps), but if a title budget arrives, the
   candidates are « …más reciente del dispositivo » (−3 cells) and, failing that, « …al inicio de sesión nuevo
   de este dispositivo » (−8, at the cost of the comparative). « equipo » was rejected outright: glossary
   team = `equipo`.
3. « más reciente » is the comparative, as in the English; unlike fr I did not narrow it to a superlative.

## Mechanical checks derived in this session

Measured on the written files with the repository interpreter and `rich.cells.cell_len`:

- Key order equals the 79-key English order; 76 pre-existing values byte-identical to `HEAD`; no duplicates.
- `{id}` appears exactly once, byte-identical ICU-subset single-brace form, in `repointed` only; no braces
  anywhere in `title_repoint` or `repoint`; no other placeholder introduced; no plural/select construct.
- No manual bidi control characters in any string (U+200E/200F, U+2066-2069, U+202A-202E).
- NFC-stable: every string equals its `unicodedata.normalize('NFC', …)` form.
- Length (cells, en 62 / 145 / 95 for reference): 91 / 188 / 122 — reported because the desktop shows these
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

- `local_operator/i18n/catalogues/es/mobile.setup.json` (three keys appended)
- `i18n/audits/evidence/es/mobile.setup/r1/translator-receipt.md` (this file)

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
