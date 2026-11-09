# Independent translation audit — hi / `mobile.setup` repoint family — round 1

**Verdict: PASSED** — 0 blocker, 0 major, 0 minor, 2 nit.

- Auditor: second independent translation auditor (lopdev team, `task` subagent), session 84b112fab6ca; did not translate or edit any value it judged, and audited none of the four locales the first auditor covered (es, fr, ru, vi)
- Actual model: deepseek/deepseek-flash — harness-disclosed cross-vendor fallback, the same descent as the translator lane (pinned anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash) before any string was read or judged. Recorded as the fallback, never the pinned id; no native-speaker pass is claimed for zh-CN, ur or hi.
- Translator: i18n translator lane, `task` teammate of the lopdev team, session 86c6d01615c6; wrote all three values and did not audit them
- Translator actual model: deepseek/deepseek-flash — quota-driven cross-vendor fallback. The lane was pinned to anthropic/claude-opus-5-5 and descended twice (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash); the text was drafted and written while running the fallback, as its own receipt in this directory states. Recorded as the fallback, not the pinned id, per docs/I18N.md step 3.
- English source: `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4` (`local_operator/i18n/catalogues/en/mobile.setup.json`, 79 keys)
- hi catalogue: `633a90641eaf428d35716b2aa2af0803574d4588b2e39cc3f3cfb087968ed797` (79 keys)
- Worktree `/Users/damian/local-operator/.worktrees/fix-setup-repoint`, branch `fix/tunnel-setup-repoint-successor`, revision audited `0df4ddea133`

## The three keys, rendered

| Key | English | en cells | hi | cells | rendered cells |
|---|---|---:|---|---|---:|
| `mobile.setup.title_repoint` | Phone access needs reconnecting to this device’s newer sign-in | 62 | फ़ोन एक्सेस को इस डिवाइस के नए साइन-इन से फिर से जोड़ना होगा | 43 | 43 |
| `mobile.setup.repoint` | A newer sign-in to the same Radient account is already on this device. Reconnect phone access to that sign-in — same account, same phone address. | 145 | इसी Radient खाते का एक नया साइन-इन पहले से इस डिवाइस पर मौजूद है। फ़ोन एक्सेस को उसी साइन-इन से फिर से जोड़ें — वही खाता, वही फ़ोन पता। | 101 | 101 |
| `mobile.setup.repointed` | Reconnected phone access to this device’s newer sign-in for the same account (credential {id}). | 95 | फ़ोन एक्सेस को उसी खाते के लिए इस डिवाइस के नए साइन-इन से फिर से जोड़ दिया गया (क्रेडेंशियल {id})। | 71 | 69 |

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

- credentials → क्रेडेंशियल ✓ singular for one row (glossary §2 and note 6; plural क्रेडेंशियल्स), matching the shipped (क्रेडेंशियल {id}) in mobile.setup.account
- device → डिवाइस ✓ (settled loan per the hi guide's list); account → खाता ✓; phone access → फ़ोन एक्सेस ✓ (title_login)
- phone address → फ़ोन पता ✓ (shipped reservation_saved)
- sign-in → साइन-इन ✓ with the hyphen, matching title_login and identity_unavailable — see HI-02

## Style-guide compliance

- आप register; respectful imperative जोड़ें; no तुम / तू; no padding honorifics
- Danda । closes both full stops in the remedy and the receipt (guide: danda for full stops in prose); the blocked-state title carries none, like the shipped title_login
- Nukta kept where the spelling has one (फ़ोन) and NOT added where it does not (पता stays पता) — checked character by character, which is the guide's 'keep nukta consistently' rule applied in both directions
- ASCII comma in वही खाता, वही फ़ोन पता। is the guide's sanctioned 'comma ,', and the spaced em dash matches the namespace's identity_unavailable
- No ZWJ/ZWNJ anywhere (asserted); standard conjunct spellings (जोड़, फ़ोन, क्रेडेंशियल)
- Digits Latin; {id} renders as bare Latin digits with no grouping and no Devanagari numerals (the guide defers ०१२… in v1)
- ASCII parentheses ( ) around क्रेडेंशियल {id}, as mobile.setup.account ships them

## Idiomaticity

- Reads written: इसी Radient खाते का एक नया साइन-इन पहले से इस डिवाइस पर मौजूद है is idiomatic (पहले से … मौजूद है for 'already on this device'), and जोड़ दिया गया is the right completed passive for a receipt — the same aspect choice the namespace's other receipts make.
- Oblique forms are all correct: इसी … खाते का (genitive), इस डिवाइस के नए साइन-इन से (oblique + postposition), उसी खाते के लिए (oblique after के लिए).
- फ़ोन एक्सेस is kept as the namespace's term even though it is a loan: the guide prefers the living loan over an invented native form, and the alternative (मोबाइल पहुँच) would fork the concept.

## Adjudications the brief named

### (d) «नया» carries both 'new' and 'newer' — acceptable without a comparative?

**PASS — acceptable; no comparative is available and none is needed here**

Four reasons, in order of weight. (1) Hindi has no usable comparative of this adjective: नएतर is not a word, and इससे नया ('newer than it') requires an antecedent that these strings deliberately do not have. (2) The catalogue already uses नया for the fresh re-read (account_failed नया सत्यापन, quote_unread नया खाता और कोट), so one carrier is this locale's established register rather than a compression. (3) The wrong reading is blocked structurally, not lexically: 'this device's new sign-in' is the possessive इस डिवाइस के, which makes the sign-in an artifact the device already holds (a sign-in the user would perform reads इस डिवाइस पर नया साइन-इन), and the verb throughout is जोड़ें ('join/reconnect'), never साइन इन करें. (4) The referent is provably unique: local_operator/providers/account_successor.py requires exactly one enabled, non-dead same-account row on the same hub before it re-points, so 'the new one' and 'the newer one' are the same row in every state where this plan is offered. The residual risk is that the title read alone loses 'newer'; that is a nuance, not a wrong instruction, and the remedy restores it immediately with पहले से … मौजूद है.

## Findings

### HI-01 — nit — accepted — no change

- Keys: `mobile.setup.title_repoint`, `mobile.setup.repoint`, `mobile.setup.repointed`
- String: …इस डिवाइस के नए साइन-इन से… / एक नया साइन-इन…
- Issue: «नया» carries 'new', 'newer' and 'fresh' in this catalogue, so the English comparative 'newer' loses its comparative form. Read alone, the title could be taken as 'this device's new sign-in' rather than 'the newer of the two local sign-ins'.
- Assessment: Acceptable, and no better form exists. Hindi has no usable comparative of this adjective (नएतर is not a word; इससे नया needs an antecedent the strings do not have), the catalogue already uses नया for the fresh re-read, the possessive इस डिवाइस के plus पहले से … मौजूद है fix the referent, the verb is जोड़ें ('join/reconnect') and never साइन इन करें, and account_successor.py guarantees exactly one candidate row, so 'the new one' and 'the newer one' are the same row. The residual loss is a nuance in the title, not a wrong instruction. See adjudication (d).

### HI-02 — nit — accepted — no change; note for a future wave

- Keys: `mobile.setup.title_repoint`, `mobile.setup.repoint`, `mobile.setup.repointed`
- String: साइन-इन (hyphenated) vs साइन इन (spaced) elsewhere in the namespace
- Issue: The namespace ships «साइन-इन» in title_login, identity_unavailable and these three keys, and «साइन इन» in login, complete, complete_tui and login_tui. Two spellings in one namespace can read as drift.
- Assessment: The split is grammatical, not accidental: the hyphenated form is the compound NOUN (Radient साइन-इन चाहिए, साइन-इन किए गए खाते), and the spaced form is the verb phrase (साइन इन करें / साइन इन करने). These three keys use the noun, so they belong on the hyphenated side — the same side as the title ledger they extend. No change. A future hi wave should record the noun/verb split in the style guide so the next translator does not read it as drift and 'unify' it.

## Observations outside this delta

- At the TUI notice's real body width (74 cells at 80 columns) the remedy wraps to 2 rows and the receipt fits one; Devanagari is measured with the product's own `cell_len`, the same measure Textual lays out with, so these are the rows that paint. Glyph shaping (conjunct rendering in the user's terminal) is the terminal's job and is not claimed here.
- `mobile.setup` is not a `wire.*` namespace, so these strings never reach the phone client's bundled chunks; the surfaces are the CLI/TUI notice and the desktop remedy paragraph.

## Translator receipt: claims checked

- Source hash f38364c1… — re-derived, matches. Context 2cf80ced…, glossary 0eb34437…, hi guide 82292dc8…, target 633a9064… — all re-derived, all match.
- Cell figures 43 / 101 / 71 — re-measured with rich.cells.cell_len: 43 / 101 / 71. Confirmed exactly.
- 'नएतर is not a word' — agreed, and no catalogue in this tree uses a comparative of नया.
- Key-order and append-only claims — re-verified against `git show 0df4ddea133^:…`: 76 values byte-identical, 79 keys in English order. Confirmed.

## Rendered evidence

- `rendered.txt` (this directory): returns the ENGLISH source for this locale in this tree, because the generated shipped set still listed only ['en'] when the evidence was emitted (the regeneration is this auditor's own shared gate and runs after the ledger rows). Shown verbatim rather than hidden: rendered.txt Layer 1.
- Layer 2: `runtime.render_message(catalogues.load_catalogue(locale, 'mobile.setup')[code], params, locale)` — the same ICU engine Layer 1 uses, over this locale's own loaded bytes; with {id} substituted as 42 and again as the product's own str(row_id) '1042'.
- Layer 3: `messages.render_record(...)` with `resolve.shipped_locales` / `resolve.is_shipped` widened IN PROCESS for the measurement only, then restored; it exercises render_record -> _lookup -> runtime.render_message end to end and reports this catalogue as `source_locale`. No file is modified: data/shipped.json kept its checked-in content at emission time.
- Negative control: omitting `id` raises `runtime.MessageFormatError: message argument 'id' was not supplied (have: none)` — the placeholder is live ICU, not escaped braces. The public resolver instead degrades to the code with `source_locale=None`, the documented additive contract.

## Tracked wrap rows (product `wrap_cells`, `NoticeBlock.body_budget`)

```
terminal 80 -> body 74 cells
  mobile.setup.title_repoint: hi=1 row(s)  en=1 row(s)
  mobile.setup.repoint: hi=2 row(s)  en=2 row(s)
  mobile.setup.repointed: hi=1 row(s)  en=2 row(s)
terminal 100 -> body 94 cells
  mobile.setup.title_repoint: hi=1 row(s)  en=1 row(s)
  mobile.setup.repoint: hi=2 row(s)  en=2 row(s)
  mobile.setup.repointed: hi=1 row(s)  en=1 row(s)
terminal 120 -> body 114 cells
  mobile.setup.title_repoint: hi=1 row(s)  en=1 row(s)
  mobile.setup.repoint: hi=1 row(s)  en=2 row(s)
  mobile.setup.repointed: hi=1 row(s)  en=1 row(s)
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

- Nothing was refused. All three keys pass; both nits are adjudicated and neither asks for a change.
