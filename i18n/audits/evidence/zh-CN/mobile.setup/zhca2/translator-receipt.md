# Translator receipt — zh-CN / mobile.setup — repoint family (3 keys)

- Translator: `task` teammate (i18n translator lane), session `86c6d01615c6`, team `lopdev`.
- Actual model: `deepseek/deepseek-flash` — a quota-driven cross-vendor fallback. The session was pinned to `anthropic/claude-opus-5-5` and descended twice on provider quota (`anthropic/claude-opus-5-5` → `openai/gpt-6.1-sol` → `deepseek/deepseek-flash`). The text below was drafted, measured and written while running `deepseek/deepseek-flash`. This is not a model choice of the translator's; the ledger should record the fallback rather than the pinned id.
- Date: 2026-11-08. Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch
  `fix/tunnel-setup-repoint-successor`. Authored against catalogue content as of `d7c64e9a1fa`; that content is
  unchanged at `6200cf9bf49` (the parent of this commit, which touches only tunnel Python — no i18n path).
- Scope: 3 English keys × this one locale. Evidence slug: `zhca2`.
- Sources read: `docs/I18N.md` (“Add a string”, fallback/coverage rules), `i18n/glossary.md`,
  `i18n/style-guides/zh-CN.md`, the English catalogue and its context sidecar, and the shipped
  `local_operator/i18n/catalogues/zh-CN/mobile.setup.json` for established register.

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
- `i18n/style-guides/zh-CN.md` = `8f95d89debbd0af848e72e4805e256ef9566632e520a5c6d9b37928cd617142c`

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

- Target before: `15333fe4a0bc853904649906e60d2335036e898d15b361845b97d67b99bea47a` (76 keys)
- Target after: `da9d130ede023c303b20549dedebe9da8dbd60dead8e2016dcd8da8974150222` (79 keys)
- Diff shape: one line gains a trailing comma, three lines added — nothing else.

## Translated values (verbatim)

| Key | zh-CN | cells |
|---|---|---:|
| `mobile.setup.title_repoint` | 手机访问需要重新连接到此设备较新的登录 | 38 |
| `mobile.setup.repoint` | 此设备上已有同一Radient账户的较新登录。将手机访问重新连接到该登录——同一账户、同一手机地址。 | 91 |
| `mobile.setup.repointed` | 已为同一账户将手机访问重新连接到此设备较新的登录（凭据{id}）。 | 62 |

## Locale decisions (`zh-CN`)

This locale is the one that already ships a full translation of the namespace, so it is the reference for
register: 你-free verb-forward phrasing, full-width prose punctuation, no space inserted between Chinese and
Latin/digit runs, no `。` on short labels.

- **phone access → 手机访问**, the namespace-wide term (`手机访问需要登录Radient`, `手机访问已连接`) and the
  term in `cli.mobile_help.mobile` / `wire.slash.mobile`. Not 手机接入, not 移动访问.
- **sign-in → 登录**, matching `mobile.setup.title_login` (`手机访问需要登录Radient`) and `relay.login`
  (`sign in` → 登录). See flag 1: 登录 names the act/instance of signing in; in these strings it stands for the
  local sign-in *row* the tunnel is being re-pointed at.
- **credential → 凭据**, the shipped rendering in four `mobile.setup` strings already
  (`选择现有隧道所属的凭据`, `凭据ID：{ids}`, `（凭据{id}）`). The glossary's zh-CN credential cell is still `—`
  (unendorsed); this wave follows the shipped catalogue rather than inventing a term, and records 凭据 here for
  the wave that fills that cell.
- **account → 账户** (the namespace ships 账户 throughout, and it is the catalogue's dominant rendering —
  64 strings carry 账户 against 3 carrying 账号, in `cli.hub_org.ineligible` and two `cli.team_bridge` strings
  about a different object); **device →
  此设备** (shipped `reserved`: `已保存在此设备上`; `reservation_saved` uses 此电脑 for “this computer”, which is a
  different object); **phone address → 手机地址** (shipped `reservation_saved`, `consent`).
- **“newer” → 较新**, the comparative, distinct from the namespace's 最新 for “fresh/latest”
  (`quote_unread`: `最新的账户信息和报价`). Chinese has no comparative inflection, so 较新的登录 is the newer of
  the two local sign-ins named by the context.
- **“for the same account”** in the receipt is rendered by fronting it: 已为同一账户将手机访问重新连接到…
  instead of appending a clause, because a trailing 同一账户 would read as a second, separate assertion.
- Typography: full-width `，。、（）` and the doubled dash `——` (the shipped convention in this namespace:
  `正在准备手机访问——更改前先检查…`); no `。` on the blocked-state title; no space between 登录/凭据/账户 and the
  Latin `Radient` or the `{id}` placeholder; `「」` not used (mainland “ ” convention, and no quotes are needed
  here).

### Flags for the auditor

1. **登录 vs 登录凭据.** The English “sign-in” is a local row whose credential id the receipt prints. 登录 is the
   namespace's established word and reads correctly, but a reader who pairs 「凭据{id}」 in the receipt with
   「登录」 in the sentence above may wonder whether the row or the act is meant. 登录凭据（sign-in credential）
   would bind the two explicitly; I did not use it because it over-specifies what the English says. Auditor's
   call, and if it changes it should change in all three keys together.
2. **Length**: 38 / 91 / 62 cells against English 62 / 145 / 95 — the shortest of the seven, as expected at
   2 cells per CJK character. Nothing approaches an 80-cell line.
3. **zhca1 is untouched and no longer describes this file.** `zhca1` is a byte-carry attestation of the 76-key
   file (`target 15333fe4a0bc853904649906e60d2335036e898d15b361845b97d67b99bea47a`); this wave moves that target
   to `da9d130ede023c303b20549dedebe9da8dbd60dead8e2016dcd8da8974150222` (79 keys). `zhca1` stays history;
   `zhca2` is the attestation scope for the current state.

## Mechanical checks derived in this session

Measured on the written files with the repository interpreter and `rich.cells.cell_len`:

- Key order equals the 79-key English order; 76 pre-existing values byte-identical to `HEAD`; no duplicates.
- `{id}` appears exactly once, byte-identical ICU-subset single-brace form, in `repointed` only; no braces
  anywhere in `title_repoint` or `repoint`; no other placeholder introduced; no plural/select construct.
- No manual bidi control characters in any string (U+200E/200F, U+2066-2069, U+202A-202E).
- NFC-stable: every string equals its `unicodedata.normalize('NFC', …)` form.
- Length (cells, en 62 / 145 / 95 for reference): 38 / 91 / 62 — reported because the desktop shows these
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

- The one test that flipped red → green is
  `tests/unit/i18n/test_hold_split.py::test_chinese_carries_its_own_copy_in_every_namespace`: at `d7c64e9a1fa`
  the English `mobile.setup` carried 79 keys and zh-CN 76, so the namespace had no zh-CN counterpart for the three
  new keys. This append is what makes it pass — the en-only source commit left it red.
## Files written

- `local_operator/i18n/catalogues/zh-CN/mobile.setup.json` (three keys appended)
- `i18n/audits/evidence/zh-CN/mobile.setup/zhca2/translator-receipt.md` (this file)

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
