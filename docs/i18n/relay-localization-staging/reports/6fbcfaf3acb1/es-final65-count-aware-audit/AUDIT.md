# Independent Spanish count-aware delta audit

**Native verdict: clean with follow-ups; terminal native round, zero unresolved BLOCKER/MAJOR. Rendered remains BLOCKED/HOLD.**

Actual auditor: independent task `6fbcfaf3acb1` / `openai/gpt-6.1-sol`; no fallback. No publication or committed-head freshness claim. Working EN/context is uncommitted on base `de9b9a490c839d6de2c2e41ea03ec819f66be489`. Original final65 audit and findings preserved.

## Scope and hashes

Seven target changes, three EN unit-source changes, direct amount/row neighbors. Retained58 have identical serialized target lines and unchanged English/context per-key values; session/login whole bytes unchanged.

### relay.shell (44)
- EN `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560`
- ES `852545691f324684e456b67d9a1e6b3193b346a82bc544bb340f3f6a4e5e32d2`
- Context `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`

### relay.session (17)
- EN `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129`
- ES `17a2bf64b2a6f5ffeb10d7d8d54a06d76e6a3102c477681fdf59815a57b463f2`
- Context `8777c48c750d982f3f795c81d0a543e92000a3898223cfa60637f8a2edc1e2db`

### relay.login (4)
- EN `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5`
- ES `b6e676acdc5cbb9a85423c458a19a5d6e5bc35594eb1dd21b23569af32f29658`
- Context `9fcd5a1669e6831b10b7041d82f77c69e858183c001ba9faa949532b4d71980a`

## Finding dispositions

- **ES65-1 resolved in this candidate:** count-aware nouns plus natural Cantidad restante fix all earlier count defects. Same numeric count propagation inspected in real UsageSheet lines48-63.
- **ES65-2 resolved in this candidate:** actual1:00 timestamps use comma punctuation, no fixed a las.
- **ES65-3 MINOR remains parent-owned durable documentation follow-up:** scoped rationale is in translator REMEDIATION.md22, not yet source style guide. Record the corrected singular token and plural tókenes.

### ES65-4 — MINOR: incorrect accent on singular token (historical-scratch/5ad0af2dd59e/es-final65-count-aware/relay.shell.json:40)

Actual `Cantidad restante: 1 tóken` and `Uso: -1 tóken`; expected token without accent. [FundéuRAE](https://www.fundeu.es/recomendacion/token-alternativas-en-espanol/) explicitly specifies singular token/toquen and plural tókenes/tóquenes (plural accent because esdrújula). Full source independently fetched this round. **Exact target delta:** `one {tóken}` → `one {token}`; retain `many {de tókenes}` and `other {tókenes}`. This is orthographic, not a new major, and does not force another native round.

## Critical exact-million lead

Preserved second-sample lead: French OQLF says full digit millions omit de. **Rejected for Spanish, not generalized across languages.** Primary [RAE DPD cardinales §6](https://www.rae.es/dpd/cardinales) explicitly states:

> Lo mismo sucede si se escriben con números: 1 000 000 de personas, 1 200 000 personas.

Thus current Spanish many branches de créditos/de tókenes/de solicitudes are supported even for fully digit-written numbers. [Wikilengua §3](https://www.wikilengua.org/index.php/mill%C3%B3n) independently reports the same RAE rule and notes technical-symbol contexts sometimes omit de; no forced change follows from that optional register. **Required ES many-branch delta: none.** Actual ±million rows have exactly one de, not doubled de. French finding requires its own language-specific disposition.

## Actual independent execution

Existing Python interpreter and actual phone createMessageRuntime/IntlMessageFormat dependency only. Scratch HOME/config, env-i, TZ=UTC; inherited CMUX/XPC/LOP flags stripped. No installs, dependency copies, browser, live sessions, source/target/ledger edits or heavy suites.

Both scripts in this report directory executed exit0:
```sh
S=historical-scratch/6fbcfaf3acb1/es-final65-count-aware-audit
R=source-checkout
T=historical-scratch/5ad0af2dd59e/es-final65-count-aware
P=historical-scratch/5ad0af2dd59e/es-final65
env -i HOME="$S/home" LOCAL_OPERATOR_CONFIG_DIR="$Soperator-home" PATH="$PATH" TERM=xterm-256color TZ=UTC GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0 source-checkout/.venv/bin/python "$S/validate.py" "$R" "$T" "$P" "$S"
env -i HOME="$S/home" LOCAL_OPERATOR_CONFIG_DIR="$Soperator-home" PATH="$PATH" TERM=xterm-256color TZ=UTC node "$S/phone.mjs" "$T" "$S"
```

- Exact keys, argument encounter order/types and formatter AST skeletons: PASS all65.
- Isolated repository check_catalogues: PASS (`[]`). Missing-count negative rejects `arguments () != en's (('count', 'plural'),)`; invalid few negative rejects category absent from `[many,one,other]`; restored fixtures PASS. Independent AST style negative distinguishes percent from plain number.
- Eight counts0/1/2/1.5/-1/-1.5/+million/-million × credits/tokens/requests/USD/unknown × remaining/used =80 composed cases per runtime. All80 Python/phone amount, no-family and named-family row strings identical. CLDRone selects±1, other0/2/±1.5, many±million. No duplicated de or number; row exactly one colon, family opus preserved.
- Actual `Cantidad restante: 1 crédito`, `Uso: 1 solicitud`, `Cantidad restante: -1,5 créditos`, `Uso: -1.000.000 de solicitudes`; token cases have ES65-4 spelling only.
- Actual `Restablecimiento: 14 oct 2026, 1:00`, `Datos del 14 oct 2026, 1:00.`; .29/.07 retain29/7 NBSP percent with one percent sign.

## Narrow native assessments

- `relay.shell.usage.remaining`: PASS: Cantidad restante is natural, neutral and count-independent; correctly inflected unit follows numeric value, no parentheses workaround or spend-permission implication.
- `relay.shell.usage.remaining_unknown_unit`: PASS: count-independent unknown-unit caption retains amount and omits untrusted API token.
- `relay.shell.usage.used`: PASS direct neighbor: unchanged Uso caption now agrees with inflected units at 0/1/2/fractions/negative/millions.
- `relay.shell.usage.unit_credits`: PASS: crédito/créditos/de créditos one/other/many; numeric exact-million de explicitly supported by RAE DPD cardinales §6.
- `relay.shell.usage.unit_tokens`: PASS agreement with MINOR ES65-4 orthography: singular tóken needs token, plural tókenes valid; many de correct for numeric exact million per RAE.
- `relay.shell.usage.unit_requests`: PASS: solicitud/solicitudes/de solicitudes one/other/many, including negative absolute-value plural selection.
- `relay.shell.usage.reset`: PASS: Restablecimiento neutral noun caption retains reset semantics; typed medium date/short time and 1:00 no article agreement defect.
- `relay.shell.usage.as_of`: PASS: Datos del with comma preserves timestamp freshness semantics; typed medium date/short time; no fixed a las.
- `relay.shell.usage.row`: PASS direct neighbor: whole-row punctuation uses em dash plus exactly one amount colon; no duplicated de or number.
- `relay.shell.usage.row_family`: PASS direct neighbor: family opus remains machine identifier in parentheses; unknown-window label and complete punctuation retained.

## Limits and durable evidence

- Native status shell passed with minor follow-ups; session/login native passes retained under proved byte compatibility.
- Rendered/fit/pseudo **BLOCKED/HOLD**; no waiver. Spanish LTR, native RTL N/A; mixed-script visual proof unclaimed.
- `AUDIT.json` contains exact metadata, all finding IDs/dispositions and the preserved/rejected million lead. `mechanical.json`/`validation.log` and `phone.json`/`phone.log` contain actual outputs; scripts adjacent.
- Original final65 failed audit untouched. No commit SHA supplied for candidate corrections; these are audited-candidate resolutions, not claims that fixes landed.
