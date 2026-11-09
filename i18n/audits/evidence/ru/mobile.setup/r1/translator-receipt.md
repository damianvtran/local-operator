# Translator receipt — ru / mobile.setup — repoint family (3 keys)

- Translator: `task` teammate (i18n translator lane), session `86c6d01615c6`, team `lopdev`.
- Actual model: `deepseek/deepseek-flash` — a quota-driven cross-vendor fallback. The session was pinned to `anthropic/claude-opus-5-5` and descended twice on provider quota (`anthropic/claude-opus-5-5` → `openai/gpt-6.1-sol` → `deepseek/deepseek-flash`). The text below was drafted, measured and written while running `deepseek/deepseek-flash`. This is not a model choice of the translator's; the ledger should record the fallback rather than the pinned id.
- Date: 2026-11-08. Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch
  `fix/tunnel-setup-repoint-successor`. Authored against catalogue content as of `d7c64e9a1fa`; that content is
  unchanged at `6200cf9bf49` (the parent of this commit, which touches only tunnel Python — no i18n path).
- Scope: 3 English keys × this one locale. Evidence slug: `r1`.
- Sources read: `docs/I18N.md` (“Add a string”, fallback/coverage rules), `i18n/glossary.md`,
  `i18n/style-guides/ru.md`, the English catalogue and its context sidecar, and the shipped
  `local_operator/i18n/catalogues/ru/mobile.setup.json` for established register.

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
- `i18n/style-guides/ru.md` = `f20e1cafaf70578c17ce072c71725628808d228e4be5604329f8da30386f1fb5`

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

- Target before: `1cb725c9e457cba6d17f6ec4baf6c482e1282fe20bfd3e31b5f71d4fb2556ba5` (76 keys)
- Target after: `22a3c2528d1c5e95197d1d94800a3f8b6a3c80c84c57f5307778c3fb0b9e58e3` (79 keys)
- Diff shape: one line gains a trailing comma, three lines added — nothing else.

## Translated values (verbatim)

| Key | ru | cells |
|---|---|---:|
| `mobile.setup.title_repoint` | Доступ с телефона нужно переподключить к более новому входу на этом устройстве | 78 |
| `mobile.setup.repoint` | На этом устройстве уже есть более новый вход в ту же учётную запись Radient. Переподключите доступ с телефона к этому входу — та же учётная запись, тот же адрес телефона. | 170 |
| `mobile.setup.repointed` | Доступ с телефона переподключён к более новому входу на этом устройстве для той же учётной записи (учётные данные {id}). | 120 |

## Locale decisions (`ru`)

Glossary/style inputs: `i18n/glossary.md` §2 (credentials → `учётные данные`, device → `устройство`) and
`i18n/style-guides/ru.md` (lowercase `вы`, impersonal constructions, infinitive labels, «ёлочки» quotes, no
typographic space before `: ; ! ?`, em dash with spaces, `ё` retained).

- **phone access → « Доступ с телефона »**, the namespace-wide rendering (`Для доступа с телефона нужен вход в
  Radient`, `Доступ с телефона подключён`) and `cli.mobile_help.mobile`. Not « мобильный доступ ».
- **sign-in → « вход »** (noun), matching `mobile.setup.title_login` and `relay.login.page.title` (`sign in` →
  `вход`); verb « войти ». The remedy's imperative is « Переподключите » (lowercase `вы`).
- **credential → « учётные данные »**, per glossary and the shipped `(учётные данные {id})` in
  `mobile.setup.account`. Plural-form noun used for a single row, as the shipped strings already do — not
  « идентификатор » (that is an identifier, which the glossary explicitly separates).
- **account → « учётная запись »** throughout, never « аккаунт »: the namespace ships `учётная запись` in
  eighteen earlier strings (twenty counting these keys), including `Учётная запись Radient: {identity}`.
- **“newer sign-in” → « более новый вход »**. The namespace's word for *fresh* is different and is not reused
  here: `quote_unread`/`account_failed` say « актуальная учётная запись », « заново проверить ». For a newer
  object, `cloudflared` already ships the comparative « 2025.4.0 или новее ».
- **device → « этом устройстве »**, **phone address → « адрес телефона »** (shipped `reservation_saved`).
- Typography: « — » with spaces, one occurrence, in the remedy only; no «ёлочки» needed; `ё` kept in
  « учётные », « переподключён »; no space before punctuation; impersonal phrasing where it reads naturally
  (« На этом устройстве уже есть … ») rather than « Мы … ».

### Flags for the auditor

1. **« более новый вход » is the one phrase I would most want a second opinion on.** It is grammatical and
   unambiguous (a newer sign-in than the dead row), but Russian UI copy more often says « новый вход » (the
   newly available one) and leaves the comparison implicit. I kept the explicit comparative because the whole
   point of the blocked state is that a *newer* row exists while an older one is pinned — dropping the
   comparison would make the remedy read as “sign in again”, which is exactly what this plan does not ask for.
2. **Length**: 78 / 170 / 120 cells against English 62 / 145 / 95 (+26% on the title). The title still fits one
   unwrapped 80-column TUI line at 78 cells; nothing is near a wrap that would hide a word.

## Mechanical checks derived in this session

Measured on the written files with the repository interpreter and `rich.cells.cell_len`:

- Key order equals the 79-key English order; 76 pre-existing values byte-identical to `HEAD`; no duplicates.
- `{id}` appears exactly once, byte-identical ICU-subset single-brace form, in `repointed` only; no braces
  anywhere in `title_repoint` or `repoint`; no other placeholder introduced; no plural/select construct.
- No manual bidi control characters in any string (U+200E/200F, U+2066-2069, U+202A-202E).
- NFC-stable: every string equals its `unicodedata.normalize('NFC', …)` form.
- Length (cells, en 62 / 145 / 95 for reference): 78 / 170 / 120 — reported because the desktop shows these
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

- `local_operator/i18n/catalogues/ru/mobile.setup.json` (three keys appended)
- `i18n/audits/evidence/ru/mobile.setup/r1/translator-receipt.md` (this file)

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
