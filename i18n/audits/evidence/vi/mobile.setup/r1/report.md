# Audit report — vi / mobile.setup — repoint family (3 keys), round 1

- Verdict: **PASSED** — 0 blocker / 0 major / 0 minor / 2 nit.
- Auditor: independent translation auditor (lopdev team, `task` subagent), session e7b6613db529; did not translate or edit any value it judged
- Auditor model: deepseek/deepseek-flash — harness-disclosed cross-vendor fallback. The session was pinned to anthropic/claude-opus-5-5 and descended on provider quota (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash) before any string was read or judged. Recorded as the fallback, not the pinned id; no native-speaker pass is claimed.
- Translator (from its receipt): i18n translator lane, `task` teammate of the lopdev team, session 86c6d01615c6 — deepseek/deepseek-flash — quota-driven cross-vendor fallback; the lane was pinned to anthropic/claude-opus-5-5 and descended on provider quota (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash). Stated by the translator's own receipt in this directory, which says the text was drafted and written while running the fallback; recorded as such per docs/I18N.md step 3.
- Source: `local_operator/i18n/catalogues/en/mobile.setup.json` sha256 `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4`
- Target: `local_operator/i18n/catalogues/vi/mobile.setup.json` sha256 `6fd0509a69ba62a6127b9525aedd6e55555b58a7015cfff741c8b822e4dfe9af`
- Revision under audit: `0df4ddea13324dc8dd5feb1967caa29a178017c6` (branch `fix/tunnel-setup-repoint-successor`)

## Strings audited

| Key | English (cells) | Locale (cells) |
|---|---|---|
| `mobile.setup.title_repoint` | Phone access needs reconnecting to this device’s newer sign-in (62) | Truy cập điện thoại cần kết nối lại với lần đăng nhập mới hơn trên thiết bị này (79) |
| `mobile.setup.repoint` | A newer sign-in to the same Radient account is already on this device. Reconnect phone access to that sign-in — same account, same phone address. (145) | Thiết bị này đã có lần đăng nhập mới hơn vào cùng tài khoản Radient. Hãy kết nối lại truy cập điện thoại với lần đăng nhập đó — cùng tài khoản, cùng địa chỉ điện thoại. (168) |
| `mobile.setup.repointed` | Reconnected phone access to this device’s newer sign-in for the same account (credential {id}). (95) | Đã kết nối lại truy cập điện thoại với lần đăng nhập mới hơn trên thiết bị này cho cùng tài khoản (thông tin xác thực {id}). (124) |

## Rendered through the product path

- `mobile.setup.title_repoint` → 'Truy cập điện thoại cần kết nối lại với lần đăng nhập mới hơn trên thiết bị này'
- `mobile.setup.repoint` → 'Thiết bị này đã có lần đăng nhập mới hơn vào cùng tài khoản Radient. Hãy kết nối lại truy cập điện thoại với lần đăng nhập đó — cùng tài khoản, cùng địa chỉ điện thoại.'
- `mobile.setup.repointed` → 'Đã kết nối lại truy cập điện thoại với lần đăng nhập mới hơn trên thiết bị này cho cùng tài khoản (thông tin xác thực 42).'

Layer 1 (`messages.render`, resolver exactly as the tree stands) returns English in this tree because the generated shipped set still lists `en` alone; layer 3 runs the full `render_record → _lookup → runtime.render_message` chain with the shipped gate widened in process only, and shows this locale as the provenance. Details and negative controls: `rendered.txt`.

## Findings

### VI-01 — nit — keys: `mobile.setup.title_repoint`, `mobile.setup.repoint`, `mobile.setup.repointed`

- String: … với lần đăng nhập mới hơn …
- Issue: Classifier «lần» is absent from `title_login`, so the namespace now names the same concept in a verb form in one key and a classified noun form in three.
- Assessment: Not a drift: the two are different grammatical roles and the noun form is the shipped convention in three other namespaces. Keeping the verb form in `title_login` is also correct. No change.
- Disposition: accepted — no change (adjudication vi-c)

### VI-02 — nit — keys: `mobile.setup.repointed`

- String: Đã kết nối lại truy cập điện thoại với lần đăng nhập mới hơn trên thiết bị này cho cùng tài khoản (thông tin xác thực {id}).
- Issue: Three prepositional phrases stack at the end («với … trên … cho …»); the English has the same shape, so this is faithful, just heavier than the rest of the string.
- Assessment: Understandable on one read and not the kind of heaviness a vi editor would reject. If a later wave touches it, «… trên thiết bị này, cho cùng tài khoản …» only adds a comma. No remediation round warranted.
- Disposition: accepted — optional smooth, no change requested

## Adjudications of translator-flagged departures

### vi-c — PASS

- Question: vi added a classifier («lần») that the sibling `title_login` does not use. Is the departure right?
- Rationale: Right, and the sibling is not a counter-example. `title_login` is «Truy cập điện thoại cần đăng nhập Radient», where «đăng nhập» is the VERB ('needs to sign in to Radient') and correctly takes no classifier; these three strings need the NOUN ('a sign-in'), and Vietnamese requires a classifier for a counted/identified instance. The nominal form already ships elsewhere in the same style: «lần đăng nhập» in `wire.remote_compute.unknown_run` («cho lần đăng nhập đã chọn»), `cli.team_bridge.outcome.not_addressable` («vào lần đăng nhập tiếp theo») and `tui.welcome.tip.setup` («một lần đăng nhập, không cần dán khóa»). The translator's proposed fallback («với đăng nhập đó») is the clipped form and would have been the defect. Consistency here means matching the noun convention, which the shipped catalogue already sets.

## Checks performed

- key set: 79 keys, identical and order-identical to the English catalogue (tests/unit/i18n/test_catalogues.py asserts order-sensitive parity); the three new keys are the trailing group, exactly where the English source puts them after mobile.setup.corrupt_intent
- append-only: every one of the 76 pre-existing values is byte-identical to `git show 0df4ddea133^:<path>`; the only other diff line is the trailing comma the append needs
- placeholder integrity: `{id}` appears exactly once, in `mobile.setup.repointed` only, in single-brace ICU-subset form; runtime.message_arguments() reports exactly one argument ('id', 'simple'); no brace anywhere in the title or the remedy body
- render gate: the product engine raises MessageFormatError when `id` is omitted, so the placeholder is live ICU, not escaped braces
- no bidi/format control characters (U+200E/F, U+202A-E, U+2066-9) in any of the three values; every value is NFC-stable; no decomposed sequences (vi checked explicitly)
- typography: no ASCII apostrophe, no double space, no space before punctuation, no leading/trailing space, em dash spaced ' — ', no ASCII '...'
- cell widths measured with rich.cells.cell_len against the English source; no `maxCells` is declared for these keys in the context sidecar, so nothing was judged against a hard budget
- does not imply a NEW sign-in (no 'sign in again' reading in any of the three strings; the title obliges a re-point of the existing access)
- does not imply a NEW tunnel, a NEW account or an address change — the remedy names the same account and the same phone address explicitly and the receipt repeats 'same account'
- does not imply a subscription/activation or any service mutation
- the {id} parenthetical is presented as a local credential row reference, not as an account id, email or phone address

## Limitations

- Independent LLM review on a provider-fallback model (deepseek/deepseek-flash after two quota descents), not native-speaker certification; the judgments are reasoned, not native intuition.
- Scalar text evidence only: no terminal frame, desktop frame or device screenshot was rendered. Widths are rich.cells.cell_len measurements against the English source; no maxCells is declared for these keys.
- This lane did not run scripts/i18n/generate.py or scripts/i18n/check.py: the shipped-set snapshot is only correct after every auditor's rows exist, so the shared gates run last under the second auditor.
- Desktop reachability is asserted from the route contract (desktop_tunnel_setup.py ships plan['remedy'] as remedy.text with kind 'run') and from tunnels/setup.py; no desktop build was exercised.

## Ledger row

- Storage: `i18n/audit-history/vi/mobile.setup.json` (ordered suffix), appended with `scripts/i18n/ledger.py append`; the last row per (locale, namespace) is the effective one.
- `i18n/ledger.json` and `i18n/ledger/<locale>.json` are baseline snapshots pinned by `i18n/audit-history.json`; docs/I18N.md forbids rewriting them after migration, so a new pass is an appended row, not an edit of the stale row.
- `source_sha256` = `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4` (current English hash of the namespace).
- Suffix `history_sha256` after this append: `83dcbad3fd53e560443aaba7c34cb89cf9666a43bd717ba3efc8ebf6ba5250c2`.
- Row: status `passed`, auditor model recorded exactly as disclosed (provider fallback), translator model taken from the translator's own receipt.
