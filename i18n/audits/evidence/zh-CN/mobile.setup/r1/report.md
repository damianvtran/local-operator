# Independent translation audit — zh-CN / `mobile.setup` repoint family — round 1

**Verdict: PASSED** — 0 blocker, 0 major, 0 minor, 1 nit.

- Auditor: second independent translation auditor (lopdev team, `task` subagent), session 84b112fab6ca; did not translate or edit any value it judged, and audited none of the four locales the first auditor covered (es, fr, ru, vi)
- Actual model: deepseek/deepseek-flash — harness-disclosed cross-vendor fallback, the same descent as the translator lane (pinned anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash) before any string was read or judged. Recorded as the fallback, never the pinned id; no native-speaker pass is claimed for zh-CN, ur or hi.
- Translator: i18n translator lane, `task` teammate of the lopdev team, session 86c6d01615c6; wrote all three values and did not audit them
- Translator actual model: deepseek/deepseek-flash — quota-driven cross-vendor fallback. The lane was pinned to anthropic/claude-opus-5-5 and descended twice (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash); the text was drafted and written while running the fallback, as its own receipt in this directory states. Recorded as the fallback, not the pinned id, per docs/I18N.md step 3.
- English source: `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4` (`local_operator/i18n/catalogues/en/mobile.setup.json`, 79 keys)
- zh-CN catalogue: `da9d130ede023c303b20549dedebe9da8dbd60dead8e2016dcd8da8974150222` (79 keys)
- Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch `fix/tunnel-setup-repoint-successor`, revision audited `0df4ddea133`

## The three keys, rendered

| Key | English | en cells | zh-CN | cells | rendered cells |
|---|---|---:|---|---|---:|
| `mobile.setup.title_repoint` | Phone access needs reconnecting to this device’s newer sign-in | 62 | 手机访问需要重新连接到此设备较新的登录 | 38 | 38 |
| `mobile.setup.repoint` | A newer sign-in to the same Radient account is already on this device. Reconnect phone access to that sign-in — same account, same phone address. | 145 | 此设备上已有同一Radient账户的较新登录。将手机访问重新连接到该登录——同一账户、同一手机地址。 | 91 | 91 |
| `mobile.setup.repointed` | Reconnected phone access to this device’s newer sign-in for the same account (credential {id}). | 95 | 已为同一账户将手机访问重新连接到此设备较新的登录（凭据{id}）。 | 62 | 60 |

## Meaning preserved

- does not imply a NEW sign-in: none of the three strings contains the locale's verb for 'sign in' in an imperative or prospective frame. zh-CN writes 重新连接 (reconnect) and never 重新登录; ur writes دوبارہ منسلک (reconnect again) and never دوبارہ سائن اِن; hi writes फिर से जोड़ें (join again) and never फिर से साइन इन करें. The sign-in appears only as the OBJECT the existing access is re-pointed at
- does not imply a NEW tunnel or a service mutation: the action verb in all three locales is the locale's word for 'reconnect/re-join', with no create/start/enable verb
- does not imply an account change: every locale names the same account explicitly (zh-CN 同一账户 / ur وہی اکاؤنٹ · اسی Radient اکاؤنٹ / hi वही खाता · इसी Radient खाते) and the receipt repeats it
- does not imply the phone address changes: every locale says the address is the same one (zh-CN 同一手机地址 / ur وہی فون ایڈریس / hi वही फ़ोन पता), matching the shipped `reservation_saved` form of that object in each locale
- the {id} parenthetical is presented as a local credential ROW reference, not an account id, email or phone address — and it reuses the exact shipped parenthetical form of `mobile.setup.account` in all three locales
- the comparative 'newer' is decidable in every state this plan can be in: `local_operator/providers/account_successor.py:35` refuses to return a successor unless EXACTLY ONE enabled, non-dead same-account row exists on the same hub, so 'newer' and 'the newer one' name one row — verified by reading that function, not by trusting a translator's note

## Mechanical checks

- key set: 79 keys, identical and order-identical to the English catalogue (tests/unit/i18n/test_catalogues.py asserts order-sensitive parity); the three new keys are the trailing group, exactly where the English source puts them after mobile.setup.corrupt_intent
- append-only: all 76 pre-existing values are byte-identical to `git show 0df4ddea133^:<path>` (compared as parsed JSON maps over the whole namespace, not by eye); the only other diff line is the trailing comma the append needs (4 insertions / 1 deletion per file)
- placeholder integrity: `{id}` appears exactly once, in `mobile.setup.repointed` only, in single-brace ICU-subset form; `runtime.message_arguments()` reports exactly one argument (`('id', 'simple')`); brace counts are (0,0) in the title and the remedy body
- render gate: the product engine raises `MessageFormatError` when `id` is omitted, so the placeholder is live ICU, not escaped braces
- no manual bidi/format control characters (U+200E/F, U+202A-E, U+2066-9) in any of the three values; every value is NFC-stable
- typography: no ASCII apostrophe, no double space, no space before punctuation, no leading/trailing space, no ASCII '...' in any of the three values
- cell widths measured with `rich.cells.cell_len` — the same measure the TUI wraps and the desktop lays out with; no `maxCells` is declared for these keys in the context sidecar (read, not assumed: the sidecar carries only role/tone/note), so nothing was judged against a hard budget

## Glossary compliance

- phone access → 手机访问 ✓ — the namespace-wide term (title_login, and the shipped 手机访问已连接); not 手机接入, not 移动访问
- device → 设备 / 此设备 ✓ (glossary §2 设备); the title says 此设备, matching the shipped `reserved` (已保存在此设备上)
- account → 账户 ✓. The glossary carries no zh-CN row, so the catalogue is the authority: across all zh-CN catalogues this tree carries 账户 75 times against 账号 3 times (recounted here — which corrects the translator receipt's '64 against 3' figure while agreeing with its direction; the discrepancy is in the receipt's arithmetic, not in the copy)
- credentials → 凭据 ✓. The glossary's zh-CN cell is still `—` (unendorsed, and the glossary forbids inventing one there), so the shipped catalogue is the authority: 凭据 appears in 选择现有隧道所属的凭据, 凭据ID：{ids} and （凭据{id}）; the new receipt reuses that last form
- sign-in → 登录 ✓ (title_login, login, login_tui)
- 'newer' → 较新 ✓, the comparative, distinct from the namespace's 最新 for the fresh quote (quote_unread: 最新的账户信息和报价) — a real and correctly used split
- phone address → 手机地址 ✓ (shipped reservation_saved, consent)

## Style-guide compliance

- Address: the strings carry no pronoun at all (verb-forward 需要 / 将 … 重新连接到), the zh-CN guide's preferred shape; no 您 anywhere
- Full-width prose punctuation, verified by codepoint census rather than by eye: U+3002 IDEOGRAPHIC FULL STOP and U+3001 IDEOGRAPHIC COMMA in the remedy, U+FF08/U+FF09 FULLWIDTH PARENTHESES around the receipt's credential id. No ，(U+FF0C) is needed and none was substituted for the ideographic comma
- The em dash is the doubled —— (two U+2014), the convention this namespace already ships (正在准备手机访问——更改前先检查…)
- No space between Chinese and Latin runs — 同一Radient账户 and 凭据{id}. That is the zh-CN guide's explicit width rule ('No space inserted between Chinese text and Latin/digit runs'), the opposite of the more common pangu convention, so it is right as written and would be wrong if 'corrected'
- No sentence-final 。 on the blocked-state title (guide: no 。 on short labels and statuses); the two full sentences keep theirs
- Digits Latin; {id} renders as bare Latin digits with no grouping (the product passes str(row_id)), and Chinese has only the `other` plural category, which these strings do not invoke
- No quotes needed or invented (mainland “ ” convention; 「」 not used)

## Idiomaticity

- Reads written, not translated. 此设备上已有同一Radient账户的较新登录 is how a Chinese product states 'there is already a newer one of this on the device', and 将手机访问重新连接到该登录 is the standard 将 disposal imperative.
- The one thing a native editor might raise is register, not meaning: 已为同一账户将… is a slightly formal fronting of 'for the same account', chosen (per the translator's receipt) because a trailing 同一账户 would read as a second, separate assertion. I agree with that reasoning and did not ask for a change.
- 该登录 is the correct anaphor for the sign-in just named (as opposed to 此设备 for 'this device'); the two demonstratives do not collide.

## Adjudications the brief named

### (c) 登录 for a sign-in *instance/row* — right, or does it need 登录凭据 to bind to the receipt's 凭据{id}?

**PASS — 登录 is right; no change**

English itself separates the two objects: 'sign-in' in the prose and 'credential {id}' only in the parenthetical. The translation keeps exactly that split — 登录 in all three sentences, 凭据 only inside （凭据{id}）, which is the shipped parenthetical of mobile.setup.account. Widening to 登录凭据 would diverge from English by naming the row a credential in prose, and would then collide with the receipt's own 凭据{id}, reading as 'sign-in credential (credential {id})'. 登录 is also the namespace's established word for this object (title_login) and is understood as the instance in 较新的登录 / 较新登录 because 较新 needs a comparable noun, not a verb. Nothing in the copy is wrong.

## Findings

### ZH-CN-01 — nit — accepted — no change

- Keys: `mobile.setup.title_repoint`, `mobile.setup.repoint`, `mobile.setup.repointed`
- String: …此设备较新的登录 / …（凭据{id}）。
- Issue: 登录 names the sign-in *instance* in all three sentences while the receipt's parenthetical names the row 凭据. A reader who pairs （凭据{id}） with 登录 in the sentence above it may wonder whether the sentence means the row or the act; 登录凭据 would bind the two explicitly.
- Assessment: No change, and this is the correct split rather than an oversight. English makes the same distinction the translation does: 'sign-in' in the prose, 'credential {id}' only inside the parenthetical. Rendering it as 登录凭据 would (a) name in prose something English names only parenthetically, and (b) then read as 'sign-in credential (credential {id})' beside the receipt's own 凭据{id}. 登录 is also this namespace's established word for the object (mobile.setup.title_login 手机访问需要登录Radient, login, login_tui), and （凭据{id}） is byte-for-byte the shipped parenthetical of mobile.setup.account. Recorded as the adjudication the brief asked for; no edit requested. If a future wave wants to bind the two it must move all three keys together, as the translator noted.

## Observations outside this delta

- At the TUI notice's real body width (74 cells at 80 columns) the remedy wraps to 2 rows and the break falls inside the compound 同一账户 (…同一账 | 户、同一手机地址。). That is ordinary CJK per-character line breaking, not a defect: Chinese wraps between characters, and no punctuation rule is violated at that break.
- The receipt (60 rendered cells) fits one row where the English source (93) takes two, and the title (38) fits where English (62) fits — so this locale never paints more rows than the source.
- `mobile.setup` is not a `wire.*` namespace, so these strings never reach the phone client's bundled chunks; the surfaces are the CLI/TUI notice and the desktop remedy paragraph.

## Translator receipt: claims checked

- Source hash f38364c1… — re-derived, matches. Context 2cf80ced…, glossary 0eb34437…, zh-CN guide 8f95d89d…, target da9d130e… — all re-derived, all match the receipt.
- Cell figures 38 / 91 / 62 — re-measured with rich.cells.cell_len: 38 / 91 / 62. Confirmed exactly.
- 'the shortest of the seven' — consistent with the measurements (91 against en 145 for the remedy).
- '账户 64 against 账号 3' — direction confirmed, figure corrected here to 75 against 3 (my count spans every zh-CN catalogue and counts occurrences, not strings). No copy consequence.
- Key-order and append-only claims — re-verified against `git show 0df4ddea133^:…`: 76 values byte-identical, 79 keys in English order. Confirmed.

## Rendered evidence

- `rendered.txt` (this directory): returns the ENGLISH source for this locale in this tree, because the generated shipped set still listed only ['en'] when the evidence was emitted (the regeneration is this auditor's own shared gate and runs after the ledger rows). Shown verbatim rather than hidden: rendered.txt Layer 1.
- Layer 2: `runtime.render_message(catalogues.load_catalogue(locale, 'mobile.setup')[code], params, locale)` — the same ICU engine Layer 1 uses, over this locale's own loaded bytes; with {id} substituted as 42 and again as the product's own str(row_id) '1042'.
- Layer 3: `messages.render_record(...)` with `resolve.shipped_locales` / `resolve.is_shipped` widened IN PROCESS for the measurement only, then restored; it exercises render_record -> _lookup -> runtime.render_message end to end and reports this catalogue as `source_locale`. No file is modified: data/shipped.json kept its checked-in content at emission time.
- Negative control: omitting `id` raises `runtime.MessageFormatError: message argument 'id' was not supplied (have: none)` — the placeholder is live ICU, not escaped braces. The public resolver instead degrades to the code with `source_locale=None`, the documented additive contract.

## Tracked wrap rows (product `wrap_cells`, `NoticeBlock.body_budget`)

```
terminal 80 -> body 74 cells
  mobile.setup.title_repoint: zh-CN=1 row(s)  en=1 row(s)
  mobile.setup.repoint: zh-CN=2 row(s)  en=2 row(s)
  mobile.setup.repointed: zh-CN=1 row(s)  en=2 row(s)
terminal 100 -> body 94 cells
  mobile.setup.title_repoint: zh-CN=1 row(s)  en=1 row(s)
  mobile.setup.repoint: zh-CN=1 row(s)  en=2 row(s)
  mobile.setup.repointed: zh-CN=1 row(s)  en=1 row(s)
terminal 120 -> body 114 cells
  mobile.setup.title_repoint: zh-CN=1 row(s)  en=1 row(s)
  mobile.setup.repoint: zh-CN=1 row(s)  en=2 row(s)
  mobile.setup.repointed: zh-CN=1 row(s)  en=1 row(s)
```

## Gate results (this commit)

### `scripts/i18n/ledger.py append --receipt <row> --expected-history-sha <history-sha> --apply  (x3: zh-CN, ur, hi)`

exit `0`

```
i18n/audit-history/zh-CN/mobile.setup.json
i18n/audit-history/ur/mobile.setup.json
i18n/audit-history/hi/mobile.setup.json
```

### `scripts/i18n/ledger.py validate --base HEAD`

exit `0`

```
audit history: valid and append-only
```

### `scripts/i18n/generate.py`

exit `0`

```
wrote local_operator/i18n/keys/__init__.py
wrote local_operator/mobile/web/src/i18n/keys.gen.ts
wrote local_operator/mobile/web/src/i18n/bundled.gen.ts
wrote local_operator/mobile/web/src/i18n/bundled.es.gen.ts
wrote local_operator/mobile/web/src/i18n/bundled.fr.gen.ts
wrote local_operator/mobile/web/src/i18n/bundled.hi.gen.ts
wrote local_operator/mobile/web/src/i18n/bundled.ru.gen.ts
wrote local_operator/mobile/web/src/i18n/bundled.ur.gen.ts
wrote local_operator/mobile/web/src/i18n/bundled.vi.gen.ts
wrote local_operator/mobile/web/src/i18n/bundled.zh-CN.gen.ts   (tail; every generated artifact was written)
```

### `scripts/i18n/generate.py --check`

exit `0`

```
generate.py --check: 157 artifact(s) byte-identical
```

### `scripts/i18n/check.py --catalogues-only`

exit `0`

```
check.py: catalogue checks: 0 problem(s)
```

### `scripts/i18n/check.py`

exit `0`

```
check.py: ok — 215 file(s) scanned, 2528 literal(s) at or below the baseline, catalogues consistent; 81 advisory finding(s) (report-only until activated)
```

### `.venv/bin/python -m pytest tests/unit/i18n/test_language_manifest.py tests/unit/i18n/test_catalogue_sizes.py tests/unit/i18n/test_ledger_shards.py -q`

exit `0`

```
.........................................................                [100%]
57 passed in 19.42s
```

## What was not verified

- Independent LLM review on a provider-fallback model (deepseek/deepseek-flash after two quota descents), not native-speaker certification; the judgments are reasoned, not native intuition.
- Scalar text evidence only: no terminal frame, desktop frame or device screenshot was captured. Widths are rich.cells.cell_len measurements plus the product's own wrap_cells rows, not pixels, and the desktop panel's pixel width lives in local-operator-ui and was not driven.
- The shipped-gate render required widening resolve.shipped_locales in process; that patch is disclosed, measurement-only and restored. The real eligibility proof is the post-append generation, whose output is recorded in gate_results and shipped_after_gate.
- No native RTL shaping was verified (the `ur` TUI stance is documented best-effort: layout stays LTR, per-string isolation only). Glyph shaping in a specific terminal is not something this evidence can settle.

## What this audit refused to pass

- Nothing was refused. All three keys pass; the single nit is adjudicated and asks for no change.
