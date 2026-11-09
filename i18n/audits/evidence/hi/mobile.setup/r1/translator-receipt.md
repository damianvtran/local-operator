# Translator receipt — hi / mobile.setup — repoint family (3 keys)

- Translator: `task` teammate (i18n translator lane), session `86c6d01615c6`, team `lopdev`.
- Actual model: `deepseek/deepseek-flash` — a quota-driven cross-vendor fallback. The session was pinned to `anthropic/claude-opus-5-5` and descended twice on provider quota (`anthropic/claude-opus-5-5` → `openai/gpt-6.1-sol` → `deepseek/deepseek-flash`). The text below was drafted, measured and written while running `deepseek/deepseek-flash`. This is not a model choice of the translator's; the ledger should record the fallback rather than the pinned id.
- Date: 2026-11-08. Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch
  `fix/tunnel-setup-repoint-successor`. Authored against catalogue content as of `d7c64e9a1fa`; that content is
  unchanged at `6200cf9bf49` (the parent of this commit, which touches only tunnel Python — no i18n path).
- Scope: 3 English keys × this one locale. Evidence slug: `r1`.
- Sources read: `docs/I18N.md` (“Add a string”, fallback/coverage rules), `i18n/glossary.md`,
  `i18n/style-guides/hi.md`, the English catalogue and its context sidecar, and the shipped
  `local_operator/i18n/catalogues/hi/mobile.setup.json` for established register.

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
- `i18n/style-guides/hi.md` = `82292dc8d13fcc93dc752457719264aed3cc20bb65ecfaba0f0e5a31f4cb40d5`

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

- Target before: `ad319c7b6d07c6a83dd01431983af2dc462230f4ed2cefaceb85aa02816181a7` (76 keys)
- Target after: `633a90641eaf428d35716b2aa2af0803574d4588b2e39cc3f3cfb087968ed797` (79 keys)
- Diff shape: one line gains a trailing comma, three lines added — nothing else.

## Translated values (verbatim)

| Key | hi | cells |
|---|---|---:|
| `mobile.setup.title_repoint` | फ़ोन एक्सेस को इस डिवाइस के नए साइन-इन से फिर से जोड़ना होगा | 43 |
| `mobile.setup.repoint` | इसी Radient खाते का एक नया साइन-इन पहले से इस डिवाइस पर मौजूद है। फ़ोन एक्सेस को उसी साइन-इन से फिर से जोड़ें — वही खाता, वही फ़ोन पता। | 101 |
| `mobile.setup.repointed` | फ़ोन एक्सेस को उसी खाते के लिए इस डिवाइस के नए साइन-इन से फिर से जोड़ दिया गया (क्रेडेंशियल {id})। | 71 |

## Locale decisions (`hi`)

Glossary/style inputs: `i18n/glossary.md` §2 (credentials → `क्रेडेंशियल`, device → `डिवाइस`) and
`i18n/style-guides/hi.md` (आप register, polite imperatives, danda `।`, nukta kept, settled Devanagari loans,
Latin product/SDK tokens never transliterated).

- **phone access → « फ़ोन एक्सेस »**, the namespace-wide rendering (`फ़ोन एक्सेस के लिए Radient साइन-इन चाहिए`,
  `फ़ोन एक्सेस जुड़ गया है`) and the `cli.mobile_help.mobile` / `wire.slash.mobile` term.
- **sign-in → « साइन-इन »** with the hyphen, matching `title_login`'s `Radient साइन-इन चाहिए` and
  `identity_unavailable`'s `अपने साइन-इन किए गए खाते`. The verb form elsewhere is `साइन इन करें` (`mobile.setup.login`);
  the attributive noun keeps the hyphen, which is what the shipped titles do.
- **credential → « क्रेडेंशियल »**, per glossary (singular for one row; plural क्रेडेंशियल्स) and the shipped
  `(क्रेडेंशियल {id})` in `mobile.setup.account`.
- **account → « खाता »**, **device → « इस डिवाइस »** (settled loan per the guide), **phone address → « फ़ोन
  पता »** (shipped `reservation_saved`).
- **“newer sign-in” → नया साइन-इन / नए साइन-इन.** The catalogue's word for fresh/new is नया
  (`account_failed`: `नया सत्यापन`, `cloudflared`: `या नया संस्करण`), so the same adjective carries “newer” here;
  the oblique `नए` is the correct form inside the possessive `इस डिवाइस के नए साइन-इन से`. Hindi has no
  comparative inflection that would improve this, and `नएतर` is not a word.
- आप-register respectful imperatives (`जोड़ें`), no तुम/तू, no padding honorifics; danda `।` for both full
  stops; nukta kept where the spelling has one (`फ़ोन` carries it; `पता` carries none, so nothing was added);
  `Radient` stays a bare Latin run.
- No ZWJ/ZWNJ inserted anywhere; conjunct spellings are standard (जोड़, फ़ोन, क्रेडेंशियल).

### Flags for the auditor

1. **नया carries both “new” and “newer”** here. I judged that safe because the sentence states the row is
   already present on this device and the receipt says which account it belongs to; a stricter comparative
   (`इससे नया`, “newer than it”) would need an antecedent the string does not have. Worth a confirm.
2. **Length**: 43 / 101 / 71 cells against English 62 / 145 / 95. Devanagari measures optimistically in
   `cell_len` (conjuncts collapse), so the visual line is longer than the number; even so nothing approaches an
   80-cell line. No width exception claimed.

## Mechanical checks derived in this session

Measured on the written files with the repository interpreter and `rich.cells.cell_len`:

- Key order equals the 79-key English order; 76 pre-existing values byte-identical to `HEAD`; no duplicates.
- `{id}` appears exactly once, byte-identical ICU-subset single-brace form, in `repointed` only; no braces
  anywhere in `title_repoint` or `repoint`; no other placeholder introduced; no plural/select construct.
- No manual bidi control characters in any string (U+200E/200F, U+2066-2069, U+202A-202E).
- NFC-stable: every string equals its `unicodedata.normalize('NFC', …)` form.
- Length (cells, en 62 / 145 / 95 for reference): 43 / 101 / 71 — reported because the desktop shows these
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

- `local_operator/i18n/catalogues/hi/mobile.setup.json` (three keys appended)
- `i18n/audits/evidence/hi/mobile.setup/r1/translator-receipt.md` (this file)

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
