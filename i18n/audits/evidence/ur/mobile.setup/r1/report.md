# Independent translation audit — ur / `mobile.setup` repoint family — round 1

**Verdict: PASSED** — 0 blocker, 0 major, 0 minor, 3 nit.

- Auditor: second independent translation auditor (lopdev team, `task` subagent), session 84b112fab6ca; did not translate or edit any value it judged, and audited none of the four locales the first auditor covered (es, fr, ru, vi)
- Actual model: deepseek/deepseek-flash — harness-disclosed cross-vendor fallback, the same descent as the translator lane (pinned anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash) before any string was read or judged. Recorded as the fallback, never the pinned id; no native-speaker pass is claimed for zh-CN, ur or hi.
- Translator: i18n translator lane, `task` teammate of the lopdev team, session 86c6d01615c6; wrote all three values and did not audit them
- Translator actual model: deepseek/deepseek-flash — quota-driven cross-vendor fallback. The lane was pinned to anthropic/claude-opus-5-5 and descended twice (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash); the text was drafted and written while running the fallback, as its own receipt in this directory states. Recorded as the fallback, not the pinned id, per docs/I18N.md step 3.
- English source: `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4` (`local_operator/i18n/catalogues/en/mobile.setup.json`, 79 keys)
- ur catalogue: `5defff535aa2238039f810c84d623519163119e0015be0bbca9065ee3fca2c15` (79 keys)
- Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch `fix/tunnel-setup-repoint-successor`, revision audited `0df4ddea133`

## The three keys, rendered

| Key | English | en cells | ur | cells | rendered cells |
|---|---|---:|---|---|---:|
| `mobile.setup.title_repoint` | Phone access needs reconnecting to this device’s newer sign-in | 62 | فون رسائی کو اس ڈیوائس کے نئے سائن اِن سے دوبارہ منسلک کرنا ضروری ہے | 67 | 67 |
| `mobile.setup.repoint` | A newer sign-in to the same Radient account is already on this device. Reconnect phone access to that sign-in — same account, same phone address. | 145 | اسی Radient اکاؤنٹ کا ایک نیا سائن اِن پہلے سے اس ڈیوائس پر موجود ہے۔ فون رسائی کو اسی سائن اِن سے دوبارہ منسلک کریں — وہی اکاؤنٹ، وہی فون ایڈریس۔ | 144 | 144 |
| `mobile.setup.repointed` | Reconnected phone access to this device’s newer sign-in for the same account (credential {id}). | 95 | فون رسائی کو اسی اکاؤنٹ کے لیے اس ڈیوائس کے نئے سائن اِن سے دوبارہ منسلک کر دیا گیا (کریڈینشل {id})۔ | 99 | 97 |

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

- device → ڈیوائس ✓ (glossary §2); the title says اس ڈیوائس, matching the namespace's shipped use
- account → اکاؤنٹ ✓ (Radient اکاؤنٹ, the shipped form in mobile.setup.account)
- phone access → فون رسائی ✓ — the namespace-wide term (title_login, and cli.mobile_help.mobile)
- phone address → فون ایڈریس ✓ (shipped reservation_saved)
- credentials → کریڈینشل ✓. The glossary's ur cell is still `—` (unendorsed), so the shipped catalogue is the authority; the new receipt reuses (کریڈینشل {id}) exactly, ASCII parentheses included, as mobile.setup.account ships it
- sign-in → سائن اِن ✓ — see UR-01 and adjudication (a)

## Style-guide compliance

- آپ register. The title is impersonal (ضروری ہے) and the remedy uses the respectful imperative منسلک کریں; no تم / تو anywhere
- Urdu punctuation: ۔ full stop in the remedy and the receipt, ، comma in وہی اکاؤنٹ، وہی فون ایڈریس۔ (the bidi census shows exactly one CS class character = U+060C); no Hindi । and no Arabic-only signs
- Latin stays Latin, unquoted and in source order: Radient is a bare 7-character L run in the remedy and the title, never wrapped in Urdu punctuation; {id} renders as bare Latin digits (the guide defers Urdu digits ۱۲۳ in v1)
- No manual bidi controls inserted — asserted over the three values. The TUI paint resolver wraps the value in exactly one balanced RLI/PDI pair with the inner bytes identical to the raw value (executed in rendered.txt, not asserted)
- One spaced em dash (U+2014) in the remedy, the convention this namespace ships (identity_unavailable, configuring); the bidi census shows it as the single ON character
- Parentheses are ASCII ( ) as shipped in mobile.setup.account, not Urdu-specific brackets — consistent with the namespace and with the style guide, which lists ۔ ، ؟ ؛ and quotes but no bracket form

## Idiomaticity

- Reads written: اسی Radient اکاؤنٹ کا ایک نیا سائن اِن پہلے سے اس ڈیوائس پر موجود ہے is the natural Urdu way to say 'a newer one is already present on this device', and کر دیا گیا is the right completed-aspect passive for a receipt — the same choice the namespace's other receipts make.
- The two اسی in the remedy are distinguishable by their nouns (اسی Radient اکاؤنٹ … / اسی سائن اِن سے) and are the deliberate anaphora the English 'the same … that sign-in' calls for.
- پہلے سے ('already') carries the contrast that makes 'newer' intelligible without a comparative form, which is the load the English 'already on this device' carries.

## Adjudications the brief named

### (a) The namespace ships both «سائن اِن» and «لاگ اِن»; the title family chose «سائن اِن» — consistent or confusing?

**PASS — consistent; سائن اِن is the term this family requires**

Counted, not assumed: in mobile.setup, 6 pre-existing keys use سائن اِن (title_login, login, complete, complete_tui, login_tui, identity_unavailable) and exactly 1 uses لاگ اِن (owner_missing). With these three keys the namespace is 9:1 for سائن اِن, and every key that names the sign-in as a product noun — including every title — is سائن اِن. A title that used لاگ اِن would be the odd one out, so choosing سائن اِن is what consistency demands. The single لاگ اِن is not arbitrary either: it names the borrowed command-noun domain of `lop login-status`, the same register cli.login_help uses for the `lop login` command's own help. Both are ordinary Urdu; the split follows sense, not accident. Confusing? No — a reader meets سائن اِن ten times and لاگ اِن once, in a string about a command literally called `login`.

### (b) «نیا/نئے» for 'newer' while the same file uses «تازہ» for 'fresh' (quotes, prices) — is that split right?

**PASS — the split is right, and it is the file's own established one**

تازہ in this namespace means a FRESH READING of remote state: quote_unread تازہ اکاؤنٹ اور قیمت (a re-read account and price) and account_failed تازہ توثیق (a fresh re-verification); catalogue-wide it is also 'latest' (تازہ ترین). نیا means a new OBJECT or build: cloudflared … یا نیا ورژن, relay.chrome.list.new نیا. The referent of 'newer' here is a local credential row — an object that exists on disk — so نیا/نئے is the right member of the pair. Using تازہ would say 'a freshly obtained sign-in', which drifts toward the exact misreading the copy must avoid (that the user should sign in again to get one). The translator flagged this as the one call worth an Urdu reader's confirmation; I confirm it on the sense split rather than on idiom alone.

## Findings

### UR-01 — nit — accepted — no change

- Keys: `mobile.setup.title_repoint`, `mobile.setup.repoint`, `mobile.setup.repointed`
- String: …نئے سائن اِن سے… (all three values)
- Issue: This namespace ships two words for 'sign-in': سائن اِن and لاگ اِن. Both are correct Urdu; two spellings in one namespace is the kind of drift a reader notices.
- Assessment: Not drift, and the choice is the consistent one. Counted in this tree: 6 pre-existing mobile.setup keys use سائن اِن (title_login, login, complete, complete_tui, login_tui, identity_unavailable) and exactly 1 uses لاگ اِن (owner_missing) — with these three keys the namespace is 9:1 for سائن اِن. Every title and every login key is سائن اِن, and these three join the title family, so سائن اِن is required by consistency rather than merely permitted. The lone لاگ اِن sits in the one string about `lop login-status`, where the borrowed command-name register applies (the same pattern as cli.login_help, which uses لاگ اِن for the `lop login` command's own help). It is pre-existing and outside this delta. See adjudication (a).

### UR-02 — nit — accepted — no change

- Keys: `mobile.setup.title_repoint`, `mobile.setup.repoint`, `mobile.setup.repointed`
- String: نئے سائن اِن / ایک نیا سائن اِن
- Issue: The same file uses تازہ for 'fresh' (`quote_unread`: تازہ اکاؤنٹ اور قیمت, `account_failed`: تازہ توثیق) while the new keys use نیا / نئے for 'newer'. A reader could ask why the same domain uses two adjectives for near-neighbours.
- Assessment: The split is deliberate and correct, and it is the file's own. تازہ is this namespace's word for a FRESH READING of remote state — a re-read price, a re-verified account — and across the catalogue for 'latest' (تازہ ترین). نیا is its word for a new OBJECT or build (cloudflared … یا نیا ورژن). The referent here is a local credential row: an object, not a reading. Using تازہ would say 'a freshly obtained sign-in', which is uncomfortably close to the one reading the copy must avoid ('sign in again'). See adjudication (b).

### UR-03 — nit — accepted — no change; recorded as a wrap measurement

- Keys: `mobile.setup.repointed`
- String: …(کریڈینشل {id})۔
- Issue: The receipt is 97 rendered cells against the English source's 93. At the TUI notice's real body width for a 100-column terminal (94 cells) the English receipt fits one row while this one wraps to two, and the break falls between «(کریڈینشل» and «42)۔».
- Assessment: Measured with the product's own `wrap_cells` at `NoticeBlock.body_budget`, not guessed. Nothing is truncated or lost, the {id} digits stay whole and in the right order, and at the default 80-column terminal both English and Urdu take two rows — so this is a difference in a narrow band of terminal widths, not a defect. Arabic script measures ≈1 cell per character and the style guide's instruction is 'keep within maxCells; prefer short'; no maxCells is declared for these keys, and the receipt is deliberately 4 cells longer than English because Urdu needs postpositions (اسی اکاؤنٹ کے لیے) that English leaves implicit. Not worth a remediation round; recorded so the next wave can see the consequence if it shortens the receipt.

## Observations outside this delta

- «سائن اِن» is written as two space-separated tokens (سائن + U+0020 + اِن, with the kasra inside اِن), so a hard wrap can in principle split the compound. Measured: it does not split in any of these three values at any tested terminal width (80 / 100 / 120 columns), and the same form ships in 6 pre-existing keys of this namespace, so this is the namespace's existing typography rather than something this delta introduces.
- At 80 columns the repoint notice takes the same 2 rows as English and the title 1; at 120 columns Urdu's receipt (97 cells) fits one row like English's (93).
- The `/mobile status|enable` notice path builds a `NoticeBlock` from an already-rendered string and passes no source provenance, so the TUI's per-string RLI/PDI projection is not applied there even though the copy is Urdu. This is pre-existing paint plumbing (`OperatorApp._system_notice` → `NoticeBlock._rows` → `wrap_cells`), affects all 79 keys of this namespace, is not introduced by this delta, and must NOT be compensated for in the copy (the ur guide forbids 'fixing' a string for a terminal). Recorded so a future bidi work item starts from the right code path.
- `mobile.setup` is not a `wire.*` namespace, so these strings never reach the phone client's bundled chunks; the surfaces are the CLI/TUI notice and the desktop remedy paragraph (the desktop mirrors RTL at the app root, which is a UI-repo concern).

## Translator receipt: claims checked

- Source hash f38364c1… — re-derived, matches. Context 2cf80ced…, glossary 0eb34437…, ur guide c0eecf7f…, target 5defff53… — all re-derived, all match.
- Cell figures 67 / 144 / 99 — re-measured with rich.cells.cell_len: 67 / 144 / 99. Confirmed exactly.
- 'The remedy is one cell shorter than English' — confirmed: 144 against 145.
- Key-order and append-only claims — re-verified against `git show 0df4ddea133^:…`: 76 values byte-identical, 79 keys in English order. Confirmed.

## Rendered evidence

- `rendered.txt` (this directory): returns the ENGLISH source for this locale in this tree, because the generated shipped set still listed only ['en'] when the evidence was emitted (the regeneration is this auditor's own shared gate and runs after the ledger rows). Shown verbatim rather than hidden: rendered.txt Layer 1.
- Layer 2: `runtime.render_message(catalogues.load_catalogue(locale, 'mobile.setup')[code], params, locale)` — the same ICU engine Layer 1 uses, over this locale's own loaded bytes; with {id} substituted as 42 and again as the product's own str(row_id) '1042'.
- Layer 3: `messages.render_record(...)` with `resolve.shipped_locales` / `resolve.is_shipped` widened IN PROCESS for the measurement only, then restored; it exercises render_record -> _lookup -> runtime.render_message end to end and reports this catalogue as `source_locale`. No file is modified: data/shipped.json kept its checked-in content at emission time.
- Negative control: omitting `id` raises `runtime.MessageFormatError: message argument 'id' was not supplied (have: none)` — the placeholder is live ICU, not escaped braces. The public resolver instead degrades to the code with `source_locale=None`, the documented additive contract.

## Tracked wrap rows (product `wrap_cells`, `NoticeBlock.body_budget`)

```
terminal 80 -> body 74 cells
  mobile.setup.title_repoint: ur=1 row(s)  en=1 row(s)
  mobile.setup.repoint: ur=2 row(s)  en=2 row(s)
  mobile.setup.repointed: ur=2 row(s)  en=2 row(s)
terminal 100 -> body 94 cells
  mobile.setup.title_repoint: ur=1 row(s)  en=1 row(s)
  mobile.setup.repoint: ur=2 row(s)  en=2 row(s)
  mobile.setup.repointed: ur=2 row(s)  en=1 row(s)
terminal 120 -> body 114 cells
  mobile.setup.title_repoint: ur=1 row(s)  en=1 row(s)
  mobile.setup.repoint: ur=2 row(s)  en=2 row(s)
  mobile.setup.repointed: ur=1 row(s)  en=1 row(s)
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

- Nothing was refused. All three keys pass; the three nits are adjudicated and none asks for a change.
