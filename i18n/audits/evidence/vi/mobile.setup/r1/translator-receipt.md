# Translator receipt — vi / mobile.setup — repoint family (3 keys)

- Translator: `task` teammate (i18n translator lane), session `86c6d01615c6`, team `lopdev`.
- Actual model: `deepseek/deepseek-flash` — a quota-driven cross-vendor fallback. The session was pinned to `anthropic/claude-opus-5-5` and descended twice on provider quota (`anthropic/claude-opus-5-5` → `openai/gpt-6.1-sol` → `deepseek/deepseek-flash`). The text below was drafted, measured and written while running `deepseek/deepseek-flash`. This is not a model choice of the translator's; the ledger should record the fallback rather than the pinned id.
- Date: 2026-11-08. Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch
  `fix/tunnel-setup-repoint-successor`. Authored against catalogue content as of `d7c64e9a1fa`; that content is
  unchanged at `6200cf9bf49` (the parent of this commit, which touches only tunnel Python — no i18n path).
- Scope: 3 English keys × this one locale. Evidence slug: `r1`.
- Sources read: `docs/I18N.md` (“Add a string”, fallback/coverage rules), `i18n/glossary.md`,
  `i18n/style-guides/vi.md`, the English catalogue and its context sidecar, and the shipped
  `local_operator/i18n/catalogues/vi/mobile.setup.json` for established register.

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
- `i18n/style-guides/vi.md` = `e4901497ed553e2b2ae4a2660bf776adb787b89e11d7eb0f8d08f3da4b227888`

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

- Target before: `8578a9567ee326af42d3eca11b68a43f4380df27faae8182e386c47ac61687e2` (76 keys)
- Target after: `6fd0509a69ba62a6127b9525aedd6e55555b58a7015cfff741c8b822e4dfe9af` (79 keys)
- Diff shape: one line gains a trailing comma, three lines added — nothing else.

## Translated values (verbatim)

| Key | vi | cells |
|---|---|---:|
| `mobile.setup.title_repoint` | Truy cập điện thoại cần kết nối lại với lần đăng nhập mới hơn trên thiết bị này | 79 |
| `mobile.setup.repoint` | Thiết bị này đã có lần đăng nhập mới hơn vào cùng tài khoản Radient. Hãy kết nối lại truy cập điện thoại với lần đăng nhập đó — cùng tài khoản, cùng địa chỉ điện thoại. | 168 |
| `mobile.setup.repointed` | Đã kết nối lại truy cập điện thoại với lần đăng nhập mới hơn trên thiết bị này cho cùng tài khoản (thông tin xác thực {id}). | 124 |

## Locale decisions (`vi`)

Glossary/style inputs: `i18n/glossary.md` §2 (credentials → `thông tin xác thực`, device → `thiết bị`) and
`i18n/style-guides/vi.md` (bạn used sparingly, no pronoun where a verb carries the sentence, NFC always, no
space before punctuation, em dash with spaces, `setup` → `thiết lập`).

- **phone access → « Truy cập điện thoại »**, the namespace-wide rendering (`Truy cập điện thoại cần đăng nhập
  Radient`, `Truy cập điện thoại đã được kết nối`) and `cli.mobile_help.mobile`. Not « truy cập từ xa »
  (that is *remote access*, a different object in `mobile.setup.complete`).
- **sign-in → « đăng nhập »**; for one countable *instance* « lần đăng nhập ». This is the one place I diverge
  from the bare verb-with-object form the catalogue uses in `title_login` (`cần đăng nhập Radient`): there the
  verb takes Radient as its object, whereas here the sign-in is a noun carrying a modifier and a demonstrative
  (« lần đăng nhập mới hơn », « lần đăng nhập đó »). Vietnamese needs the classifier for that reading; bare
  « đăng nhập mới hơn » reads as a malformed verb phrase.
- **credential → « thông tin xác thực »**, per glossary and the shipped `(thông tin xác thực {id})` in
  `mobile.setup.account`.
- **account → « tài khoản »**, **device → « thiết bị này »**, **phone address → « địa chỉ điện thoại »** (shipped
  `reservation_saved`).
- **« Hãy »** opens the remedy, matching `quote_unread`'s « Hãy kiểm tra … »; no `bạn` appears anywhere in the
  three strings, per the guide's “Vietnamese drops pronouns”.
- Orthography: every string verified NFC; open-syllable `oa/uy` words follow the tone-first rule where they
  occur (`khóa`/`hóa` are not in these strings). No space before punctuation; one spaced em dash in the remedy.

### Flags for the auditor

1. **« lần đăng nhập » vs bare « đăng nhập ».** I am confident a classifier is required in these three
   sentences, but it is a visible departure from how `title_login` names the same concept, so it deserves a
   second read. If the auditor prefers consistency over grammaticality, the fallback is « đăng nhập mới hơn trên
   thiết bị này » in the title and « với đăng nhập đó » in the remedy — grammatical but clipped.
2. **Length**: 79 / 168 / 124 cells against English 62 / 145 / 95 (+27% on the title). The title is 79 cells,
   so it just fits an unwrapped 80-column line; the remedy is allowed to run two sentences.

## Mechanical checks derived in this session

Measured on the written files with the repository interpreter and `rich.cells.cell_len`:

- Key order equals the 79-key English order; 76 pre-existing values byte-identical to `HEAD`; no duplicates.
- `{id}` appears exactly once, byte-identical ICU-subset single-brace form, in `repointed` only; no braces
  anywhere in `title_repoint` or `repoint`; no other placeholder introduced; no plural/select construct.
- No manual bidi control characters in any string (U+200E/200F, U+2066-2069, U+202A-202E).
- NFC-stable: every string equals its `unicodedata.normalize('NFC', …)` form.
- Length (cells, en 62 / 145 / 95 for reference): 79 / 168 / 124 — reported because the desktop shows these
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

- `local_operator/i18n/catalogues/vi/mobile.setup.json` (three keys appended)
- `i18n/audits/evidence/vi/mobile.setup/r1/translator-receipt.md` (this file)

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
