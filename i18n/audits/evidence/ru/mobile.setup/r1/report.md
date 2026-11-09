# Audit report — ru / mobile.setup — repoint family (3 keys), round 1

- Verdict: **PASSED** — 0 blocker / 0 major / 0 minor / 2 nit.
- Auditor: independent translation auditor (lopdev team, `task` subagent), session e7b6613db529; did not translate or edit any value it judged
- Auditor model: deepseek/deepseek-flash — harness-disclosed cross-vendor fallback. The session was pinned to anthropic/claude-opus-5-5 and descended on provider quota (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash) before any string was read or judged. Recorded as the fallback, not the pinned id; no native-speaker pass is claimed.
- Translator (from its receipt): i18n translator lane, `task` teammate of the lopdev team, session 86c6d01615c6 — deepseek/deepseek-flash — quota-driven cross-vendor fallback; the lane was pinned to anthropic/claude-opus-5-5 and descended on provider quota (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash). Stated by the translator's own receipt in this directory, which says the text was drafted and written while running the fallback; recorded as such per docs/I18N.md step 3.
- Source: `local_operator/i18n/catalogues/en/mobile.setup.json` sha256 `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4`
- Target: `local_operator/i18n/catalogues/ru/mobile.setup.json` sha256 `22a3c2528d1c5e95197d1d94800a3f8b6a3c80c84c57f5307778c3fb0b9e58e3`
- Revision under audit: `0df4ddea13324dc8dd5feb1967caa29a178017c6` (branch `fix/tunnel-setup-repoint-successor`)

## Strings audited

| Key | English (cells) | Locale (cells) |
|---|---|---|
| `mobile.setup.title_repoint` | Phone access needs reconnecting to this device’s newer sign-in (62) | Доступ с телефона нужно переподключить к более новому входу на этом устройстве (78) |
| `mobile.setup.repoint` | A newer sign-in to the same Radient account is already on this device. Reconnect phone access to that sign-in — same account, same phone address. (145) | На этом устройстве уже есть более новый вход в ту же учётную запись Radient. Переподключите доступ с телефона к этому входу — та же учётная запись, тот же адрес телефона. (170) |
| `mobile.setup.repointed` | Reconnected phone access to this device’s newer sign-in for the same account (credential {id}). (95) | Доступ с телефона переподключён к более новому входу на этом устройстве для той же учётной записи (учётные данные {id}). (120) |

## Rendered through the product path

- `mobile.setup.title_repoint` → 'Доступ с телефона нужно переподключить к более новому входу на этом устройстве'
- `mobile.setup.repoint` → 'На этом устройстве уже есть более новый вход в ту же учётную запись Radient. Переподключите доступ с телефона к этому входу — та же учётная запись, тот же адрес телефона.'
- `mobile.setup.repointed` → 'Доступ с телефона переподключён к более новому входу на этом устройстве для той же учётной записи (учётные данные 42).'

Layer 1 (`messages.render`, resolver exactly as the tree stands) returns English in this tree because the generated shipped set still lists `en` alone; layer 3 runs the full `render_record → _lookup → runtime.render_message` chain with the shipped gate widened in process only, and shows this locale as the provenance. Details and negative controls: `rendered.txt`.

## Findings

### RU-01 — nit — keys: `mobile.setup.title_repoint`, `mobile.setup.repoint`, `mobile.setup.repointed`

- String: … к более новому входу …
- Issue: Analytic comparative where colloquial ru UI copy often writes «новый вход» and leaves the comparison implicit.
- Assessment: Correct as shipped: it preserves the comparison the state depends on and mirrors the English instead of flattening it.
- Disposition: accepted — no change (adjudication ru-flag)

### RU-02 — nit — keys: `mobile.setup.repointed`

- String: … (учётные данные {id}).
- Issue: English '(credential {id})' is singular; «учётные данные» is a plural-only noun, so the parenthetical names one row in the plural.
- Assessment: Russian has no natural singular for the credential set, and `mobile.setup.account` ships the identical parenthetical. Consistent and glossary-mandated; no change.
- Disposition: accepted — inherited, not this lane's defect

## Adjudications of translator-flagged departures

### ru-flag — PASS

- Question: The translator asks for a second opinion on «более новый вход»: grammatical and unambiguous, but Russian UI copy more often says «новый вход».
- Rationale: Keep it. The blocked state's whole point is that a newer row exists while an older one is pinned; «новый вход» alone reads as 'there is a new sign-in' and, in a remedy, invites the 'sign in again' reading this plan explicitly refuses. «Более новый вход» is grammatical, declension-correct in the dative («к более новому входу») and genitive («более новый вход» as subject), and Russian product copy does use analytic «более» when the comparison is the message. No edit requested; logged as RU-01.

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

- Storage: `i18n/audit-history/ru/mobile.setup.json` (ordered suffix), appended with `scripts/i18n/ledger.py append`; the last row per (locale, namespace) is the effective one.
- `i18n/ledger.json` and `i18n/ledger/<locale>.json` are baseline snapshots pinned by `i18n/audit-history.json`; docs/I18N.md forbids rewriting them after migration, so a new pass is an appended row, not an edit of the stale row.
- `source_sha256` = `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4` (current English hash of the namespace).
- Suffix `history_sha256` after this append: `4025381b33f4c552c7c8feae515cffeb91bff9ac227240a4d5f8a21c8e71669c`.
- Row: status `passed`, auditor model recorded exactly as disclosed (provider fallback), translator model taken from the translator's own receipt.
