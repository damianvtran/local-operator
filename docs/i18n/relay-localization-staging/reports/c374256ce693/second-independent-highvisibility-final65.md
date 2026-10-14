# PR2329 final65 — second independent high-visibility audit

**Auditor/model:** task / openai/gpt-6.1-sol; no fallback. LLM linguistic review, not human native-speaker certification. Independent of target translation and first-audit reports; French warning independently reproduced.

**Verdict: BLOCKED** — FR and ES each have one MAJOR. RU, VI and HI are clean for this sample only. No claim about unsampled strings.

Pinned source: `de9b9a490c839d6de2c2e41ea03ec819f66be489` at `source-checkout`.

## Method and limitations
Read target strings against pinned EN + contexts; run existing intl-messageformat dependency directly (same formatter as relay runtime), UTC isolated HOME; invoke actual Python message_arguments for sampled argument order. No app/browser/heavy tests, dependencies installed or target edits.

BLOCKED — customer resource restriction; no pixels, width/clipping, real browser flow, full app runtime, shipped eligibility or release claim.

29 keys per locale (plus included/window_five used as supporting row labels); 0/1/1.5/2 × credits/USD/tokens/requests × remaining/used, unknown units, .29/.07 fractions, row/family row, reset/as-of. Python argument encounter order: all 29 × 5 PASS; do not relax it to sets. All 15 target hashes rechecked unchanged.

## Findings
### HV2-FR-01 — MAJOR — FR
The invariant plural unit is still attached as an ordinary counted noun. Removing adjective agreement did not fix singular unit grammar.

Locations: `relay.shell.usage.remaining` line 29; `relay.shell.usage.used` line 30 in `historical-scratch/c4131ebe3f64/relay-final65/fr/relay.shell.json`.
Actual formatter output: `reste 1 crédits` / `utilisation de 1 crédits` / `reste 1 jetons` / `utilisation de 1 requêtes`.
Caption the invariant unit explicitly, e.g. Restant : {value, number} (unité : {unit}) / Utilisation : {value, number} (unité : {unit}). Keep value before unit; do not introduce plural/select arguments or reorder source arguments.

### HV2-ES-01 — MAJOR — ES
The ordinary counted-noun phrase retains invariant plural labels; at 1 both Quedan and créditos/solicitudes/tókenes disagree. The used caption fixes verb agreement but not the counted plural noun.

Locations: `relay.shell.usage.remaining` line 29; `relay.shell.usage.used` line 30; `relay.shell.usage.remaining_unknown_unit` line 44 in `historical-scratch/5ad0af2dd59e/es-final65/relay.shell.json`.
Actual formatter output: `Quedan 1 créditos` / `Uso: 1 créditos` / `Quedan 1 solicitudes` / `Uso: 1 tókenes`.
Caption the invariant unit, e.g. Disponible: {value, number} (unidad: {unit}) / Uso: {value, number} (unidad: {unit}). Caption the unknown-unit remaining quantity too (Disponible: {value, number}) rather than Quedan 1. Keep source argument encounter order.

## Per-locale results
### FR — BLOCKED
- **auth:** PASS: brand unchanged; connexion / mot de passe / se connecter / incorrect error are idiomatic and faithful.
- **navigation:** PASS: list/start/back/empty convey expected action and state.
- **waiting:** PASS: en attente des données… is meaningful nontechnical data-wait wording in connected/no-projection state, though less explicit than session snapshot.
- **numeric:** FAIL: invariant plural ordinary noun fails at 1; French 0 and 1.5 also normally take singular counted units. Caption invariant labels instead.
- **percent_dates_rows:** PASS: 0.29/0.07 become 29 %/7 % once, dates and field identities intact. Row itself is faithful but embeds defective singular amount.

### ES — BLOCKED
- **auth:** PASS: sign-in/password/wrong-password wording faithful and idiomatic.
- **navigation:** PASS: start/back/empty action and state preserved.
- **waiting:** PASS: esperando la vista de la sesión… communicates waiting for session view, not user approval or message.
- **numeric:** FAIL: Quedan 1 créditos / Uso: 1 créditos; bare Quedan 1 also wrong singular verb.
- **percent_dates_rows:** PASS: 29 %/7 % once, reset/as-of distinct; complete row preserves fields but embeds defective singular amount.

### RU — CLEAN
- **auth:** PASS: title, пароль, войти and wrong-password alert faithful.
- **navigation:** PASS: session/start/back/no-messages wording meaningful.
- **waiting:** PASS: ожидание данных сессии… describes waiting for session data.
- **numeric:** PASS: Остаток: N (кредиты) / Использовано: N (кредиты) make invariant nominative-plural units parenthetical captions, not a counted noun; 0/1/1.5/2 all acceptable.
- **percent_dates_rows:** PASS: 29 %/7 %, fields, Сброс and По состоянию на preserve distinctions. Nested row colon is readable, not a defect.

### VI — CLEAN
- **auth:** PASS: đăng nhập / mật khẩu / Mật khẩu không đúng. natural.
- **navigation:** PASS: clear create/start/back/no-message actions.
- **waiting:** PASS: đang chờ trạng thái hiện tại của phiên… explicitly describes current-session-state wait.
- **numeric:** PASS: Vietnamese units do not inflect with 0/1/1.5/2; Còn lại / Đã dùng work with all sampled units. tín dụng is understandable financial credit vocabulary; no demonstrated mistranslation.
- **percent_dates_rows:** PASS: 29%/7% once, reset vs snapshot timestamp clear, fields retained.

### HI — CLEAN
- **auth:** PASS: लॉग इन / पासवर्ड / पासवर्ड गलत है। faithful.
- **navigation:** PASS: session/start/back/no-messages wording clear.
- **waiting:** PASS: सत्र की मौजूदा स्थिति मिलने की प्रतीक्षा है… explicitly waits for current state.
- **numeric:** PASS: क्रेडिट/टोकन/अनुरोध are usable direct-count invariant forms at all tested amounts; शेष and का उपयोग हुआ avoid inflected plural agreement.
- **percent_dates_rows:** PASS: 29%/7% once; reset/as-of distinct; row retains fields. Intl Hindi Latin digits and pm are runtime locale output, not untranslated catalogue copy.

## Exact sampled keys
### relay.login
- `relay.login.page.title`
- `relay.login.password.label`
- `relay.login.submit.label`
- `relay.login.error.password`
### relay.session
- `relay.session.list.empty_hint`
- `relay.session.list.start_idle`
- `relay.session.list.start_busy`
- `relay.session.view.back`
- `relay.session.view.empty_title`
- `relay.session.view.empty_hint`
### relay.shell
- `relay.shell.list.empty`
- `relay.shell.list.section.active`
- `relay.shell.list.section.previous`
- `relay.shell.session.waiting`
- `relay.shell.usage.row`
- `relay.shell.usage.row_family`
- `relay.shell.usage.remaining`
- `relay.shell.usage.used`
- `relay.shell.usage.fraction`
- `relay.shell.usage.remaining_percent`
- `relay.shell.usage.used_percent`
- `relay.shell.usage.remaining_unknown_unit`
- `relay.shell.usage.used_unknown_unit`
- `relay.shell.usage.unit_credits`
- `relay.shell.usage.unit_usd`
- `relay.shell.usage.unit_tokens`
- `relay.shell.usage.unit_requests`
- `relay.shell.usage.reset`
- `relay.shell.usage.as_of`

## SHA-256 — English source and context
- relay.login: `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5`; context `9fcd5a1669e6831b10b7041d82f77c69e858183c001ba9faa949532b4d71980a`
- relay.session: `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129`; context `8777c48c750d982f3f795c81d0a543e92000a3898223cfa60637f8a2edc1e2db`
- relay.shell: `8b17439e9f31f159bc60a9e2940eb2d88400dacdcda9854ae1b3c8eb0ca00586`; context `cb317b9989bcb49ad2fb1cc4c843faf350b67e6633f508a4d00825b36f756de0`

## SHA-256 — delivered targets
### FR — `historical-scratch/c4131ebe3f64/relay-final65/fr`
- relay.login: `5f6cc95bffee5f11984b4a083af057ba96a5e332a623c5b5c3659a9e61530d24`
- relay.session: `ff308557eaaac62556f20625039d9e70df97cf55f5b150580f7511c16ecd2018`
- relay.shell: `9c8b4ee29b5da192235ae6722d44e51e49e59e4d4829ebb03035e88ed88580b9`
### ES — `historical-scratch/5ad0af2dd59e/es-final65`
- relay.login: `b6e676acdc5cbb9a85423c458a19a5d6e5bc35594eb1dd21b23569af32f29658`
- relay.session: `17a2bf64b2a6f5ffeb10d7d8d54a06d76e6a3102c477681fdf59815a57b463f2`
- relay.shell: `79b38a869a17d43911401ab4f5fa8ff5dcce98ffe588dde031ee8d757a002482`
### RU — `historical-scratch/a753be6f0d01/relay-final65/ru`
- relay.login: `6235f9f3f058fe859194f37879d485199449345c1692beeda2dc767f691dfaa1`
- relay.session: `1d3618c0928e956da89e70b9eaf0945c15062493a043a6e17a8364fc912efdec`
- relay.shell: `076b295ad1f4b9a90ef02424063f716c84af3cbf85ae41f1c53ce724e1d066ff`
### VI — `historical-scratch/590f5d5140c6/vi-final65/vi`
- relay.login: `a87e8ed3381143ee8d61dada52b0df604c72191cdde1747b9ea8fd6c46537020`
- relay.session: `7743c7b5d018b84395478e99231714a46cd998e8c78ce465333be8df62fd517a`
- relay.shell: `6f196e2535134079dfec349f645b27a75a132c95bbd40bd98a3f3cd68f18d22f`
### HI — `historical-scratch/a862db3d9624/relay-final65/hi`
- relay.login: `589c538128272c5cd7c54468f9e126dda52a004ee21b9eab1949590b5156c4ff`
- relay.session: `0fb4a5abfa0fb1ca26c2683a16a6db2f84f0f44fbf55522a7a21ec33deacb43e`
- relay.shell: `ee6c90d66c844549caa867ed81645b3cb642efad970a4537f9985b0f9b100ab0`

## Durable evidence
- Full structured report: `reports/c374256ce693/second-independent-highvisibility-final65.json`
- input: `reports/c374256ce693/native-sample-input.json`
- formatter_probe: `historical-scratch/c374256ce693/native-sample-probe.cjs`
- formatter_output: `reports/c374256ce693/native-sample-output.json`
- argument_probe: `historical-scratch/c374256ce693/native-sample-arguments.py`
- argument_output: `reports/c374256ce693/native-sample-arguments.json`
