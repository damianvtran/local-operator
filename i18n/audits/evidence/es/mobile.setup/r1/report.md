# Audit report — es / mobile.setup — repoint family (3 keys), round 1

- Verdict: **PASSED** — 0 blocker / 0 major / 0 minor / 2 nit.
- Auditor: independent translation auditor (lopdev team, `task` subagent), session e7b6613db529; did not translate or edit any value it judged
- Auditor model: deepseek/deepseek-flash — harness-disclosed cross-vendor fallback. The session was pinned to anthropic/claude-opus-5-5 and descended on provider quota (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash) before any string was read or judged. Recorded as the fallback, not the pinned id; no native-speaker pass is claimed.
- Translator (from its receipt): i18n translator lane, `task` teammate of the lopdev team, session 86c6d01615c6 — deepseek/deepseek-flash — quota-driven cross-vendor fallback; the lane was pinned to anthropic/claude-opus-5-5 and descended on provider quota (anthropic/claude-opus-5-5 → openai/gpt-6.1-sol → deepseek/deepseek-flash). Stated by the translator's own receipt in this directory, which says the text was drafted and written while running the fallback; recorded as such per docs/I18N.md step 3.
- Source: `local_operator/i18n/catalogues/en/mobile.setup.json` sha256 `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4`
- Target: `local_operator/i18n/catalogues/es/mobile.setup.json` sha256 `7c278ff03e2bb4e2a2cd04c7b005cf4c35fc7114055b141a5681b11f19adedb1`
- Revision under audit: `0df4ddea13324dc8dd5feb1967caa29a178017c6` (branch `fix/tunnel-setup-repoint-successor`)

## Strings audited

| Key | English (cells) | Locale (cells) |
|---|---|---|
| `mobile.setup.title_repoint` | Phone access needs reconnecting to this device’s newer sign-in (62) | El acceso telefónico debe reconectarse al inicio de sesión más reciente de este dispositivo (91) |
| `mobile.setup.repoint` | A newer sign-in to the same Radient account is already on this device. Reconnect phone access to that sign-in — same account, same phone address. (145) | Este dispositivo ya tiene un inicio de sesión más reciente de la misma cuenta de Radient. Reconecta el acceso telefónico a ese inicio de sesión — misma cuenta, misma dirección de teléfono. (188) |
| `mobile.setup.repointed` | Reconnected phone access to this device’s newer sign-in for the same account (credential {id}). (95) | Acceso telefónico reconectado al inicio de sesión más reciente de este dispositivo para la misma cuenta (credencial {id}). (122) |

## Rendered through the product path

- `mobile.setup.title_repoint` → 'El acceso telefónico debe reconectarse al inicio de sesión más reciente de este dispositivo'
- `mobile.setup.repoint` → 'Este dispositivo ya tiene un inicio de sesión más reciente de la misma cuenta de Radient. Reconecta el acceso telefónico a ese inicio de sesión — misma cuenta, misma dirección de teléfono.'
- `mobile.setup.repointed` → 'Acceso telefónico reconectado al inicio de sesión más reciente de este dispositivo para la misma cuenta (credencial 42).'

Layer 1 (`messages.render`, resolver exactly as the tree stands) returns English in this tree because the generated shipped set still lists `en` alone; layer 3 runs the full `render_record → _lookup → runtime.render_message` chain with the shipped gate widened in process only, and shows this locale as the provenance. Details and negative controls: `rendered.txt`.

## Findings

### ES-01 — nit — keys: `mobile.setup.title_repoint`, `mobile.setup.repointed`

- String: El acceso telefónico debe reconectarse al inicio de sesión más reciente de este dispositivo
- Issue: The translator receipt's flag 3 says es 'kept a comparative, as in the English; unlike fr I did not narrow it to a superlative'. That is true of the remedy sentence («un inicio de sesión más reciente de la misma cuenta», indefinite, comparative) but not of the title or the receipt: with the definite article plus the «de este dispositivo» complement, «al inicio de sesión más reciente de este dispositivo» is the Spanish superlative, i.e. the same construction fr was flagged for.
- Assessment: Not a copy defect: exactly one live same-account row can be the referent (`local_operator/providers/account_successor.py` requires `len(candidates) == 1`), so 'the most recent one on this device' and 'this device's newer one' pick out the same row in every state this plan can be in. Recorded so the next reader does not rely on the receipt's claim when comparing es with fr. No edit requested.
- Disposition: accepted — no change (receipt-accuracy note)

### ES-02 — nit — keys: `mobile.setup.repoint`

- String: … — misma cuenta, misma dirección de teléfono.
- Issue: Term variance inside the namespace: the same object ships as «Dirección del teléfono» (four keys: reservation_saved, reservation_existing, reservation_unconfirmed and the stage_reservation fragment) and as «la dirección del teléfono»/«la dirección de tu teléfono» inline (stage_reservation, consent). The new tail uses the article-less compound «dirección de teléfono».
- Assessment: Reads acceptably in the elliptical «misma cuenta, misma X» frame and the referent is unmistakable; a future es wave may align it to «la misma dirección del teléfono», which is the form the namespace ships for the object. Not worth a remediation round on its own.
- Disposition: accepted — no change

## Adjudications of translator-flagged departures

### es-d — PASS

- Question: es ships two renderings of 'phone access': «acceso telefónico» (mobile.setup, wire.tunnel_setup) and «acceso desde el teléfono» (`cli.mobile_help.mobile`). Did matching the namespace make the right call?
- Rationale: Yes. «acceso telefónico» is the feature's name in this namespace (15 shipped titles plus six non-title keys — `applying`, `busy`, `in_progress`, `quote_unread`, `unexpected_failure`, `corrupt_intent`) and in `wire.tunnel_setup` (3 keys); counted in this tree, not estimated; `cli.mobile_help.mobile` is a one-line help blurb about accessing the sessions *from* the phone, i.e. a different predicate, and it is untouched by this lane. Matching the namespace keeps these three keys inside one vocabulary; changing them here would have created the split inside mobile.setup instead of at the edge of it. Measured: fr/ru/vi are single-term across mobile.setup, wire.tunnel_setup and cli.mobile_help, so es is the only one of these four locales whose tree still carries the older variant — a pre-existing, out-of-scope es cleanup, not a defect of these keys.

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

- Storage: `i18n/audit-history/es/mobile.setup.json` (ordered suffix), appended with `scripts/i18n/ledger.py append`; the last row per (locale, namespace) is the effective one.
- `i18n/ledger.json` and `i18n/ledger/<locale>.json` are baseline snapshots pinned by `i18n/audit-history.json`; docs/I18N.md forbids rewriting them after migration, so a new pass is an appended row, not an edit of the stale row.
- `source_sha256` = `f38364c1895f79f93c5565d5d5f3047065cb68ebb0b8954e429c2ea9ad6800b4` (current English hash of the namespace).
- Suffix `history_sha256` after this append: `8381edf41799df78a9efe36dfde629f9654167617b5f45a90e35b82a30d3293c`.
- Row: status `passed`, auditor model recorded exactly as disclosed (provider fallback), translator model taken from the translator's own receipt.
