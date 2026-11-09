# Translator receipt — ur / mobile.setup — repoint family (3 keys)

- Translator: `task` teammate (i18n translator lane), session `86c6d01615c6`, team `lopdev`.
- Actual model: `deepseek/deepseek-flash` — a quota-driven cross-vendor fallback. The session was pinned to `anthropic/claude-opus-5-5` and descended twice on provider quota (`anthropic/claude-opus-5-5` → `openai/gpt-6.1-sol` → `deepseek/deepseek-flash`). The text below was drafted, measured and written while running `deepseek/deepseek-flash`. This is not a model choice of the translator's; the ledger should record the fallback rather than the pinned id.
- Date: 2026-11-08. Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch
  `fix/tunnel-setup-repoint-successor`. Authored against catalogue content as of `d7c64e9a1fa`; that content is
  unchanged at `6200cf9bf49` (the parent of this commit, which touches only tunnel Python — no i18n path).
- Scope: 3 English keys × this one locale. Evidence slug: `r1`.
- Sources read: `docs/I18N.md` (“Add a string”, fallback/coverage rules), `i18n/glossary.md`,
  `i18n/style-guides/ur.md`, the English catalogue and its context sidecar, and the shipped
  `local_operator/i18n/catalogues/ur/mobile.setup.json` for established register.

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
- `i18n/style-guides/ur.md` = `c0eecf7f7b5e89956b7de9ab7a9eb647ff1a7ef701ccf2a7f0264d5eea15ca60`

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

- Target before: `165acd974a6639c1d26fa9fcb667074423eec8c06484144ae8afe25b82a76215` (76 keys)
- Target after: `5defff535aa2238039f810c84d623519163119e0015be0bbca9065ee3fca2c15` (79 keys)
- Diff shape: one line gains a trailing comma, three lines added — nothing else.

## Translated values (verbatim)

| Key | ur | cells |
|---|---|---:|
| `mobile.setup.title_repoint` | فون رسائی کو اس ڈیوائس کے نئے سائن اِن سے دوبارہ منسلک کرنا ضروری ہے | 67 |
| `mobile.setup.repoint` | اسی Radient اکاؤنٹ کا ایک نیا سائن اِن پہلے سے اس ڈیوائس پر موجود ہے۔ فون رسائی کو اسی سائن اِن سے دوبارہ منسلک کریں — وہی اکاؤنٹ، وہی فون ایڈریس۔ | 144 |
| `mobile.setup.repointed` | فون رسائی کو اسی اکاؤنٹ کے لیے اس ڈیوائس کے نئے سائن اِن سے دوبارہ منسلک کر دیا گیا (کریڈینشل {id})۔ | 99 |

## Locale decisions (`ur`)

This is the RTL locale; the `ur` style guide's RTL & bidi section was read before any string was written.

Glossary/style inputs: `i18n/glossary.md` §2 (device → `ڈیوائس`), `i18n/style-guides/ur.md` (آپ register,
respectful imperatives, Urdu punctuation ۔ ، ؟ ؛, “ ” quotes, no manual bidi controls, em dash with spaces).

- **phone access → « فون رسائی »**, the namespace-wide rendering (`فون رسائی کے لیے Radient سائن اِن درکار ہے`,
  `فون رسائی منسلک ہے`) and `cli.mobile_help.mobile` / `wire.slash.mobile`.
- **sign-in → « سائن اِن »**, matching `mobile.setup.title_login` and `relay.login` (`سائن ان`). The catalogue
  also ships `لاگ اِن` where “login” is the borrowed noun in a command context (`owner_missing`: `مالک لاگ اِن
  بحال کریں`, and the grant-TTL setting label). For the title family that these three keys join, `سائن اِن` is
  the shipped term, so it is used here. Flagged below.
- **credential → « کریڈینشل »**, the shipped rendering in this namespace (`owner_mismatch`, `activated`,
  `credential_ids`, `account`) and in 29 other strings across the catalogue, including `(کریڈینشل {id})` in
  `mobile.setup.account`. The glossary's ur cell is still `—` (unendorsed); this wave
  follows the shipped catalogue rather than inventing a term, and records کریڈینشل here for the wave that fills
  that cell.
- **account → « اکاؤنٹ »**, **device → « اس ڈیوائس »** (glossary device = ڈیوائس), **phone address → « فون
  ایڈریس »** (shipped `reservation_saved`).
- **“newer” → نیا / نئے.** The namespace's word for *fresh* is تازہ and it is used for prices, quotes and
  re-verification (`تازہ اکاؤنٹ اور قیمت`, `تازہ توثیق`). A sign-in row is an object, not a fresh reading, so the
  new-object adjective is used: `ایک نیا سائن اِن` in the remedy and the oblique `نئے` in the possessive
  `اس ڈیوائس کے نئے سائن اِن`. Flagged below.
- **RTL discipline**: storage is logical order; **no** manual bidi control characters were inserted (asserted:
  no U+200E/200F, no U+2066-2069, no U+202A-202E) — the render layer applies FSI/PDI isolation and double
  application is a known defect. `Radient` stays a bare Latin run in source order, unquoted, never wrapped in
  Urdu punctuation. The TUI stays LTR and `ur` is best-effort there, per the guide; nothing was reworded to
  compensate for a terminal that shapes badly.
- Typography: Urdu `۔` for full stops (2 in the remedy's two sentences) and `،` for the comma; no Hindi `।`; no
  Arabic-only signs; one spaced em dash in the remedy, as the shipped `configuring` string does.

### Flags for the auditor

1. **سائن اِن vs لاگ اِن inside one namespace.** Both spellings ship in `mobile.setup` today: `سائن اِن` where
   the sign-in is the product noun (`title_login`, this new title, `relay.login`), `لاگ اِن` where “login” is a
   borrowed technical noun in command prose (`owner_missing`, the grant-TTL setting label). These three keys
   belong to the title family, so they take سائن اِن. If the wave wants a single word for both senses, that is a
   namespace-wide decision.
2. **نیا vs تازہ for “newer”.** I am fairly confident نیا is right here, but this is the kind of choice an
   Urdu reader should confirm: تازہ would match the words already used for “fresh” in this file, at the cost of
   sounding like a freshly obtained reading rather than a newer, still-live sign-in.
3. **Length**: 67 / 144 / 99 cells (Arabic script measures ≈1 cell per character). The remedy is one cell
   *shorter* than English; nothing is near a wrap. No width exception is claimed.

## Mechanical checks derived in this session

Measured on the written files with the repository interpreter and `rich.cells.cell_len`:

- Key order equals the 79-key English order; 76 pre-existing values byte-identical to `HEAD`; no duplicates.
- `{id}` appears exactly once, byte-identical ICU-subset single-brace form, in `repointed` only; no braces
  anywhere in `title_repoint` or `repoint`; no other placeholder introduced; no plural/select construct.
- No manual bidi control characters in any string (U+200E/200F, U+2066-2069, U+202A-202E).
- NFC-stable: every string equals its `unicodedata.normalize('NFC', …)` form.
- Length (cells, en 62 / 145 / 95 for reference): 67 / 144 / 99 — reported because the desktop shows these
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

- `local_operator/i18n/catalogues/ur/mobile.setup.json` (three keys appended)
- `i18n/audits/evidence/ur/mobile.setup/r1/translator-receipt.md` (this file)

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
