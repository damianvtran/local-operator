# Independent Spanish final65 audit

**Verdict: changes_requested — ES65-1 MAJOR unresolved.** No ledger, shipped, or publication approval is granted. Rendered/fit/pseudo evidence is **BLOCKED**, not passed.

Auditor: independent task agent, session `6fbcfaf3acb1`; actual model `openai/gpt-6.1-sol`; no provider/model fallback used. This is LLM Spanish semantic evaluation, not a human-native-speaker credential. Translator is the separate `relay-es-final65-translation` agent identified by HANDOFF.md (same model family, different agent/session).

Source revision: `de9b9a490c839d6de2c2e41ea03ec819f66be489`. All **65** rows audited anew (44 shell / 17 session / 4 login), including unchanged session/login; the prior55 changes_requested slice is not treated as approved. English and all context sidecars read from the specified checkout.

## Namespace results and exact byte hashes

| Namespace | Keys | Native result | EN SHA256 | ES SHA256 | Rendered |
|---|---:|---|---|---|---|
| relay.shell | 44 | changes_requested | `8b17439e9f31f159bc60a9e2940eb2d88400dacdcda9854ae1b3c8eb0ca00586` | `79b38a869a17d43911401ab4f5fa8ff5dcce98ffe588dde031ee8d757a002482` | BLOCKED |
| relay.session | 17 | passed | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` | `17a2bf64b2a6f5ffeb10d7d8d54a06d76e6a3102c477681fdf59815a57b463f2` | BLOCKED |
| relay.login | 4 | passed | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | `b6e676acdc5cbb9a85423c458a19a5d6e5bc35594eb1dd21b23569af32f29658` | BLOCKED |

Native passed for session/login is bounded semantic sign-off only; it is not whole-programme approval or permission to mutate eligibility metadata. Context hashes are also recorded in AUDIT.json. Source/target hashes were recomputed here, and candidates rechecked unchanged after probes.

## Findings for unified remediation

### ES65-1 — MAJOR: Complete amount messages force plural unit labels and a plural verb at value 1.

- Location: `historical-scratch/5ad0af2dd59e/es-final65/relay.shell.json:29` — `relay.shell.usage.remaining`
- Location: `historical-scratch/5ad0af2dd59e/es-final65/relay.shell.json:30` — `relay.shell.usage.used`
- Location: `historical-scratch/5ad0af2dd59e/es-final65/relay.shell.json:44` — `relay.shell.usage.remaining_unknown_unit`

Affected keys: `relay.shell.usage.remaining`, `relay.shell.usage.used`, `relay.shell.usage.remaining_unknown_unit`.

Actual runtime strings:
- `Quedan 1 créditos`
- `Uso: 1 créditos`
- `Quedan 1 tókenes`
- `Uso: 1 solicitudes`
- `Quedan 1`

UsageSheet selects invariant localized unit labels and passes numeric 1 into complete messages; no unit inflection path exists. Context explicitly permits rephrasing to avoid invariant-label agreement.
local_operator/mobile/web/src/components/usage-sheet.tsx:48-59 -> createMessageRuntime -> IntlMessageFormat

**Remediation:** Use a count-independent label structure, e.g. Restante ({unit}): {value, number}, Uso ({unit}): {value, number}, Restante: {value, number}. Keep existing named parameters and numeric ICU types. Alternatively introduce correctly inflected unit handling through a separately reviewed source change.

**Classification:** Repeated incorrect native complete messages at ordinary value 1 across three changed amount variants fail the explicitly requested agreement audit; this is a native-copy defect, not a wrong numeric amount or spend-permission bug.

### ES65-2 — MINOR: Fixed a las is not grammatical when the formatted time is 1:00.

- Location: `historical-scratch/5ad0af2dd59e/es-final65/relay.shell.json:32` — `relay.shell.usage.reset`
- Location: `historical-scratch/5ad0af2dd59e/es-final65/relay.shell.json:34` — `relay.shell.usage.as_of`

Affected keys: `relay.shell.usage.reset`, `relay.shell.usage.as_of`.

Actual runtime strings:
- `Se restablece el 1 ene 2026 a las 1:00`
- `Datos del 1 ene 2026 a las 1:00.`




**Remediation:** Avoid the time-dependent article with neutral timestamp punctuation, while preserving date medium/time short placeholders, e.g. Restablecimiento: {value, date, medium}, {value, time, short}.

**Classification:** Timestamp value stays correct and understandable; this is a narrow grammatical follow-up, not an independent blocker/major.

### ES65-3 — MINOR: Native caché/tókenes choices are sound, but the source Spanish style guide still lists cache/token as kept-English.

- Location: `source-checkout/i18n/style-guides/es.md:24` — `kept-English technical list`

Affected keys: `relay.shell.usage.cached`, `relay.shell.usage.unavailable`, `relay.shell.usage.loading`, `relay.shell.usage.empty`, `relay.shell.usage.command`, `relay.shell.usage.unit_tokens`.

Actual primary-source evidence: i18n/style-guides/es.md:24-27 includes token and cache in the binding kept-English list; translator HANDOFF.md records adapted prose/measurement choices only in scratch.




**Remediation:** Record the narrowly scoped Spanish prose caché and measurement-label tókenes decisions in the durable Spanish style guide when landing translations. Keep API tokens/identifiers byte-identical. No need to replace sound native wording with raw English.

**Classification:** Documentation consistency follow-up; no change to meaning, numeric rendering, or eligibility. This alone does not force another native round.

## Independent execution evidence

Existing shared repo Python interpreter and existing worktree Node dependency tree only; no installation, dependency copies, browser, native window, live session or heavy suite. Each process used `env -i` with scratch HOME/config and no inherited CMUX/LOP/XPC flags; phone probe explicitly `TZ=UTC`.

Commands actually executed successfully (exit 0):
```sh
S=historical-scratch/6fbcfaf3acb1/es-final65-audit
R=source-checkout
T=historical-scratch/5ad0af2dd59e/es-final65
env -i HOME="$S/home" LOCAL_OPERATOR_CONFIG_DIR="$Soperator-home" PATH="$PATH" TERM=xterm-256color source-checkout/.venv/bin/python "$S/validate_audit.py" "$R" "$T" "$S"
env -i HOME="$S/home" LOCAL_OPERATOR_CONFIG_DIR="$Soperator-home" PATH="$PATH" TZ=UTC TERM=xterm-256color node "$S/native_probe.mjs" "$R" "$T" "$S"
```

| Surface | Actual response | Result |
|---|---|---|
| Repository check_catalogues() on own isolated en/es fixtures | `[]` | PASS |
| Exact EN/ES keys and parsed argument types | shell44/session17/login4, all equal | PASS |
| Negative: percent ICU changed to simple placeholder | `arguments (('value', 'simple'),) != en's (('value', 'number'),)` | PASS (rejected) |
| Negative: row_family removed | `missing key 'relay.shell.usage.row_family'` | PASS (rejected) |
| Restored isolated fixture | `[]` | PASS |
| Python catalogue formatter | 65 actual string probes in mechanical.json | PASS mechanically; ES65-1/2 copy defects |
| Actual phone createMessageRuntime / IntlMessageFormat | Node v26.5.0; UTC; 63 probes in native-probes.json | PASS mechanically; same ES65-1/2 reproduced |
| Fractions .29/.07 | `29 % usado` / `7 % usado`; one NBSP percent | PASS |
| Fractions 0/1/2 | `0 % usado` / `100 % usado` / `200 % usado` | PASS; input1 means fraction100%, not percent-point1% |
| Numeric123.5, all four units | `123,5` plus créditos/USD/tókenes/solicitudes | PASS |
| Amount0/1/2, all units and unknown-unit variants | all 40 strings recorded; 1 fails count agreement | FAIL ES65-1 |
| Complete named-family row | `Cuota por familia de modelos (opus) · 7 días — Uso: 1 créditos` | Separator PASS; amount FAIL ES65-1 |
| Complete no-family row | `Cuota por familia de modelos · 7 días — Uso: 1 créditos` | No empty parentheses; separator PASS; amount FAIL ES65-1 |
| Timestamp1791983100000 / 2026-10-14T13:05Z | `Se restablece el 14 oct 2026 a las 13:05` | PASS |
| Timestamp1767229200000 / 2026-01-01T01:00Z | `Datos del 1 ene 2026 a las 1:00.` | MINOR ES65-2 |

No candidate has ICU plural/select branches. CLDR branch validation is therefore vacuous here, not evidence of singular grammar correctness; the explicit amount0/1/2 matrix exposes the defect. Used/fraction percent wording is grammatically sound with understood porcentaje. Date medium/time short are both typed on the same value; native values/timezone remain intact.

## Terminology and tone decisions

- **Tókenes is appropriate**, not an invented accent/plural. Independently fetched [FundéuRAE, «token» (en redonda) o «toquen»](https://www.fundeu.es/recomendacion/token-alternativas-en-espanol/), which explicitly states token/toquen plurals are tókenes/tóquenes. Its invariant plural should be a unit label, not a direct singular count noun.
- All five cache-prose occurrences use **caché** consistently; no live-provider refresh claim. Quota vocabulary uses **cuota** throughout paid/included/model-family/unlimited variants. Durable technical-list exceptions need recording (ES65-3), not a native wording reversal.
- **Esperando la vista de la sesión…** accurately represents a pending live state snapshot after connection, matching explicit EN context. It does not mean projection/prediction or establish a second connection.
- Session imperatives use neutral tú; action labels use infinitives; brand, TUI, paths and machine family identifier remain unchanged. All65 rows reviewed for source meaning, concise tone, punctuation and placeholders.

## Rendered, pseudo and RTL boundary

- **Rendered: BLOCKED / HOLD.** No browser or pixel/width/fit capture was dispatched, as explicitly scoped. Runtime output is not screenshot evidence. Separate approved rendered/browser/design/UX evidence is still required; do not turn this into a passed rendered ledger row.
- **Pseudo: applicable but BLOCKED.** Expansion, clipping, loading/empty/error/populated fit and consecutive-frame layout require actual rendering. String punctuation/type tests are not pseudo UI proof.
- **RTL: Spanish is LTR.** Actual `directionFor("es")` returned `ltr`. Spanish native RTL mirroring is not applicable; mixed-script/path/family isolation and generic RTL regression render coverage are outside this no-browser audit, not reported as passed.

## All65 per-key assessments

| Key | Native result | Assessment |
|---|---|---|
| `relay.shell.list.empty` | passed | Faithful running-session empty state; concise lowercase. |
| `relay.shell.list.connecting` | passed | Connection in progress, distinct from connected empty state; U+2026 preserved. |
| `relay.shell.list.section.active` | passed | Correct active-session heading and source title case. |
| `relay.shell.list.section.previous` | passed | Correct previous-session heading and source title case. |
| `relay.shell.list.theme` | passed | Infinitive is natural accessible control name. |
| `relay.shell.session.waiting` | passed | Vista de la sesión conveys delivered live view/state snapshot, not prediction, graphics or reconnection. |
| `relay.shell.session.connecting` | passed | Conectando con la sesión preserves pre-connection state distinct from waiting for snapshot. |
| `relay.shell.usage.title` | passed | Uso is the ordinary compact usage-sheet title. |
| `relay.shell.usage.cached` | passed_with_minor | Caché spelling is correct; the second sentence preserves no provider refresh request, not a live refresh promise. Durable style-guide exception needed; ES65-3. |
| `relay.shell.usage.unavailable` | passed_with_minor | Correct unavailable cached usage, not zero usage. Durable style-guide exception needed; ES65-3. |
| `relay.shell.usage.loading` | passed_with_minor | Correct in-flight cached data state; U+2026 preserved. Durable style-guide exception needed; ES65-3. |
| `relay.shell.usage.empty` | passed_with_minor | No reported cached data, not an exhausted or zero balance. Durable style-guide exception needed; ES65-3. |
| `relay.shell.usage.last_known` | passed | Preserves last-known versus current-unavailable contrast. |
| `relay.shell.usage.continue` | passed | Agotada expresses included quota exhaustion; credit continuation remains paired with family-cap warning. |
| `relay.shell.usage.family_caps` | passed | Preserves model-family caps still applying; no blanket unlimited-spend promise. |
| `relay.shell.usage.paid` | passed | Cuota de pago coheres with included/family/unlimited allowance vocabulary. |
| `relay.shell.usage.family` | passed | Cuota por familia de modelos is semantically accurate. |
| `relay.shell.usage.included` | passed | Cuota incluida coheres with exhausted included-quota disclosure. |
| `relay.shell.usage.amount_unknown` | passed | No reported amount remains unknown, not zero. |
| `relay.shell.usage.blocked` | passed | Gasto no disponible preserves spend-unavailable state. |
| `relay.shell.usage.eligibility_unknown` | passed | Bureaucratic but unambiguous and concise; explicitly unknown spending eligibility, not permission. |
| `relay.shell.usage.unlimited` | passed | Ilimitada qualifies only paid allowance; separate family caps retained. |
| `relay.shell.usage.window_five` | passed | 5 horas is natural compact interval label. |
| `relay.shell.usage.window_week` | passed | 7 días is natural compact interval label. |
| `relay.shell.usage.window_month` | passed | Mensual preserves monthly window. |
| `relay.shell.usage.window_lifetime` | passed | Total histórico conveys lifetime cumulative window, not a future lifetime guarantee. |
| `relay.shell.usage.window_unknown` | passed | Unknown window remains explicitly unknown. |
| `relay.shell.usage.remaining` | changes_requested | Numeric format/type correct but invariant plural unit and verb fail singular; ES65-1. |
| `relay.shell.usage.used` | changes_requested | Numeric format/type correct but invariant plural noun fails direct singular count; ES65-1. |
| `relay.shell.usage.fraction` | passed | Typed numeric fraction produces 29/7/0/100/200 NBSP percent; singular usado refers to porcentaje, not a unit noun. |
| `relay.shell.usage.reset` | passed_with_minor | Typed medium date/short time preserves timestamp semantics; a las fails at 1:00, ES65-2. |
| `relay.shell.usage.command` | passed_with_minor | Muestra is a natural command description; cached/provider meaning retained. Durable style-guide exception needed; ES65-3. |
| `relay.shell.usage.as_of` | passed_with_minor | Datos del preserves as-of cache freshness, not a refresh time; a las fails at 1:00, ES65-2. |
| `relay.shell.usage.stale` | passed | Desactualizado agrees with uso and preserves may-be-stale uncertainty. |
| `relay.shell.usage.row` | passed | Whole row owns order and em-dash separator; composed use amount has one colon, no doubled separator. |
| `relay.shell.usage.row_family` | passed | Family machine identifier remains literal in parentheses; translated labels and whole row punctuation correct. |
| `relay.shell.usage.unit_credits` | passed | Correct invariant measurement label; complete amount syntax must not force this plural after 1. |
| `relay.shell.usage.unit_usd` | passed | Correct untranslated currency acronym; no currency conversion. |
| `relay.shell.usage.unit_tokens` | passed_with_minor | Tókenes is the accepted adapted plural of token; FundéuRAE explicitly verifies accent and plural. Durable style-guide exception needed; ES65-3. |
| `relay.shell.usage.unit_requests` | passed | Solicitudes is correct provider request-count label; complete amount syntax must avoid fixed plural after 1. |
| `relay.shell.usage.remaining_percent` | passed | Queda agrees with understood porcentaje; fraction formatting not prescaled or duplicated. |
| `relay.shell.usage.used_percent` | passed | Usado agrees with understood porcentaje; correct percent scalar and spacing. |
| `relay.shell.usage.remaining_unknown_unit` | changes_requested | Unknown unit omitted correctly but Quedan 1 has singular verb defect; ES65-1. |
| `relay.shell.usage.used_unknown_unit` | passed | Uso: value is grammatical, neutral and omits unknown unit without inventing currency. |
| `relay.session.list.search_placeholder` | passed | Search conversation bodies; source capitalization and ellipsis retained. |
| `relay.session.list.empty_hint` | passed | Informal tú imperative and una refer to sesión; below/TUI/on-machine routes preserved. |
| `relay.session.list.no_matches` | passed | Accurate no-matching-conversation verdict distinct from no active sessions. |
| `relay.session.list.start_idle` | passed | Natural lowercase new-session control. |
| `relay.session.list.start_busy` | passed | In-flight session creation distinct from idle control; ellipsis correct. |
| `relay.session.view.back` | passed | Natural accessible back-to-session-list label. |
| `relay.session.view.empty_title` | passed | Accurate no-messages-yet state, preserves aún. |
| `relay.session.view.empty_hint` | passed | Informal tú imperative; below/start behavior preserved. |
| `relay.session.past.back` | passed | Natural compact backward navigation label. |
| `relay.session.past.heading` | passed | Natural past-session heading, consistent with shell previous-session noun. |
| `relay.session.past.search_placeholder` | passed | Preserves searching both session names and conversation bodies. |
| `relay.session.past.loading` | passed | Natural in-flight loading placeholder; ellipsis correct. |
| `relay.session.directory.sheet_title` | passed | Correct working-directory surface title, not current path value. |
| `relay.session.directory.path_placeholder` | passed | Preserves alternate typed path option; ruta idiomatic. |
| `relay.session.model.sheet_title` | passed | Correct compact model-sheet title. |
| `relay.session.model.filter_placeholder` | passed | Natural infinitive filter instruction for models. |
| `relay.session.model.loading` | passed | Correct in-flight model loading placeholder; ellipsis correct. |
| `relay.login.page.title` | passed | Brand remains untranslated; sign-in page title natural. |
| `relay.login.password.label` | passed | Correct password field label, lowercase. |
| `relay.login.submit.label` | passed | Natural sign-in submit action, consistent with page title. |
| `relay.login.error.password` | passed | Wrong-password alert is accurate and sentence-cased; literal independently checked without relying on tool redaction. |

## Durable evidence files

- `AUDIT.json`: exact hashes, identities, namespace verdicts, precise finding locations/severity, all65 source/target assessments.
- `AUDIT.md`: this report.
- `mechanical.json`, `validation.log`, `validate_audit.py`: independent repository checks/negatives/Python actual outputs.
- `native-probes.json`, `native-validation.log`, `native_probe.mjs`: actual phone runtime outputs and reusable rig.

**Terminal disposition:** changes_requested, not approved. One unresolved MAJOR (ES65-1); two MINOR follow-ups (ES65-2/3), which alone do not force an extra round. Unified remediation should fix the major plus accepted minors together. No repo, translator target, ledger, shipped metadata or release claims/edits.
