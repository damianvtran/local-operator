# PR2329 — FINAL five-locale second independent highvisibility sample

**Verdict: CLEAN — terminal for the reviewed linguistic sample across FR/ES/RU/VI/HI.** No blocker/major remains in scope.

**Auditor actual model:** openai/gpt-6.1-sol for every locale; no fallback. Independent LLM linguistic audit, not human native-speaker certification. No source/target edits.

HOLD — customer resource restriction. No rendered pixel/width/flow validation or activation/shipped/release claim.

## Source pin
Working head `de9b9a490c839d6de2c2e41ea03ec819f66be489` plus uncommitted count-aware source correction at `source-checkout`.
- relay.shell: source `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560`; context `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`
- relay.session: source `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129`; context `8777c48c750d982f3f795c81d0a543e92000a3898223cfa60637f8a2edc1e2db`
- relay.login: source `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5`; context `9fcd5a1669e6831b10b7041d82f77c69e858183c001ba9faa949532b4d71980a`

## Final per-locale disposition
### FR — CLEAN — openai/gpt-6.1-sol
HV2-FR-02 fixed: many now bare plurals crédits/jetons/requêtes. Actual ±million used/remaining correct; singular 1 and 1.5 retained. HV2-FR-01 remains fixed.
Final directory: `historical-scratch/c4131ebe3f64/relay-final65-numeric-million/fr`
- relay.shell SHA-256 `7c33c08b1632baec788b8e84f018ae3c861a2d3ccfb180c5029de214341a1e15`
- relay.session SHA-256 `ff308557eaaac62556f20625039d9e70df97cf55f5b150580f7511c16ecd2018`
- relay.login SHA-256 `5f6cc95bffee5f11984b4a083af057ba96a5e332a623c5b5c3659a9e61530d24`

### ES — CLEAN — openai/gpt-6.1-sol
Only token singular orthography changed tóken→token; 64 other keys unchanged. Actual ±1 token correct, 1.5 tókenes correct; exact million de tókenes retained, 1.2 million tókenes lacks de as RAE DPD §6 requires. ES01 fixed, ES02 withdrawn false positive.
Final directory: `historical-scratch/5ad0af2dd59e/es-final65-final-token`
- relay.shell SHA-256 `9e6691cdd0fab8dfbf09c314d8acd353a2d109a135d487493c652cb02a712438`
- relay.session SHA-256 `17a2bf64b2a6f5ffeb10d7d8d54a06d76e6a3102c477681fdf59815a57b463f2`
- relay.login SHA-256 `b6e676acdc5cbb9a85423c458a19a5d6e5bc35594eb1dd21b23569af32f29658`

### RU — CLEAN — openai/gpt-6.1-sol
one/few/many/other: 1 кредит; 2 кредита; 0/5/1000000 кредитов; 1.5 кредита; 21 кредит. Negative magnitudes correct. Parentheses retained unchanged, readable units. All three files byte-identical to delta-audited target, coverage retained without rerun.
Final directory: `historical-scratch/a753be6f0d01/relay-minimal3noun/ru`
- relay.shell SHA-256 `b96e8923fe82824db2434720f722d2cf323d5aa733d78ad1c75285391c16245b`
- relay.session SHA-256 `1d3618c0928e956da89e70b9eaf0945c15062493a043a6e17a8364fc912efdec`
- relay.login SHA-256 `6235f9f3f058fe859194f37879d485199449345c1692beeda2dc767f691dfaa1`

### VI — CLEAN — openai/gpt-6.1-sol
Other-only invariant tín dụng/token/yêu cầu natural at all sampled counts including fractions/negatives/million. All three files byte-identical to delta-audited target, coverage retained without rerun.
Final directory: `historical-scratch/590f5d5140c6/vi-final65-countplural/vi`
- relay.shell SHA-256 `f475dbefccd75d713fdea93fd8c5a12b541f94d00ec00c23b00fe72e2e942319`
- relay.session SHA-256 `7743c7b5d018b84395478e99231714a46cd998e8c78ce465333be8df62fd517a`
- relay.login SHA-256 `a87e8ed3381143ee8d61dada52b0df604c72191cdde1747b9ea8fd6c46537020`

### HI — CLEAN — openai/gpt-6.1-sol
one/other identical native direct forms क्रेडिट/टोकन/अनुरोध valid for direct counted nouns across all sampled counts, including 10,00,000 grouping. No numeric agreement defect. All three files byte-identical to delta-audited target, coverage retained without rerun.
Final directory: `historical-scratch/a862db3d9624/relay-countnoun3/hi`
- relay.shell SHA-256 `feab9e023be86d7841bd1e08cb273a48bbaaa118c41ade6040d7a29a701305eb`
- relay.session SHA-256 `0fb4a5abfa0fb1ca26c2683a16a6db2f84f0f44fbf55522a7a21ec33deacb43e`
- relay.login SHA-256 `589c538128272c5cd7c54468f9e126dda52a004ee21b9eab1949590b5156c4ff`

## Actual final discriminating outputs
- FR `reste 1 crédit`
- FR `utilisation de 1 crédit`
- FR `reste 1,5 crédit`
- FR `utilisation de 1,5 crédit`
- FR `reste 1 000 000 crédits`
- FR `utilisation de 1 000 000 crédits`
- FR `reste -1 000 000 crédits`
- FR `utilisation de -1 000 000 crédits`
- ES `Cantidad restante: 1 token`
- ES `Uso: 1 token`
- ES `Cantidad restante: 1,5 tókenes`
- ES `Uso: 1,5 tókenes`
- ES `Cantidad restante: -1 token`
- ES `Uso: -1 token`
- ES `Cantidad restante: 1.000.000 de tókenes`
- ES `Uso: 1.000.000 de tókenes`
- ES `Cantidad restante: 1.200.000 tókenes`
- ES `Uso: 1.200.000 tókenes`

## Finding resolutions
- **HV2-FR-01: fixed** — Count-aware singular nouns independently reproduced in prior delta and final French rerun.
- **HV2-ES-01: fixed** — Count-aware nouns and Cantidad restante wording independently reproduced.
- **HV2-FR-02: fixed** — Final many forms remove de; actual fully numeric ±million amounts now grammatical under cited OQLF exact-number rule.
- **HV2-ES-02: withdrawn — false positive** — RAE DPD cardinales §6 explicitly: “Lo mismo sucede si se escriben con números: 1 000 000 de personas, 1 200 000 personas.” https://www.rae.es/dpd/cardinales

FR source rule retained: https://vitrinelinguistique.oqlf.gouv.qc.ca/21647/la-typographie/nombres/ecriture-de-million-et-de-milliard

## Scope and compatibility
Final reruns limited to French 3-noun remediation and Spanish 1-noun orthography plus direct compositions. Existing intl-messageformat dependency, env -i isolated HOME/TZ=UTC. Final FR/ES probes include ±million, 1/1.5, other same direct neighbors; no all65 re-audit. RU/VI/HI prior actual probe results retained by whole-file byte equality. 60 current Python message_arguments checks retain encounter order.

FR shell differs from prior countdelta only in three noun-value lines. ES shell differs only in unit_tokens singular spelling. Their remaining bytes are identical after excluding those exact changed-key lines. All 13 other files are whole-file byte-identical. Earlier sample coverage is retained, not expanded to an all65 audit. All final hashes rechecked unchanged.

Original sample keys (coverage retained):
- login: `relay.login.page.title`, `relay.login.password.label`, `relay.login.submit.label`, `relay.login.error.password`
- session: `relay.session.list.empty_hint`, `relay.session.list.start_idle`, `relay.session.list.start_busy`, `relay.session.view.back`, `relay.session.view.empty_title`, `relay.session.view.empty_hint`
- shell: `relay.shell.list.empty`, `relay.shell.list.section.active`, `relay.shell.list.section.previous`, `relay.shell.session.waiting`, `relay.shell.usage.row`, `relay.shell.usage.row_family`, `relay.shell.usage.remaining`, `relay.shell.usage.used`, `relay.shell.usage.fraction`, `relay.shell.usage.remaining_percent`, `relay.shell.usage.used_percent`, `relay.shell.usage.remaining_unknown_unit`, `relay.shell.usage.used_unknown_unit`, `relay.shell.usage.unit_credits`, `relay.shell.usage.unit_usd`, `relay.shell.usage.unit_tokens`, `relay.shell.usage.unit_requests`, `relay.shell.usage.reset`, `relay.shell.usage.as_of`

Current argument-signature keys (all 12 × 5 PASS): `relay.shell.usage.unit_credits`, `relay.shell.usage.unit_tokens`, `relay.shell.usage.unit_requests`, `relay.shell.usage.remaining`, `relay.shell.usage.used`, `relay.shell.usage.remaining_unknown_unit`, `relay.shell.usage.used_unknown_unit`, `relay.shell.usage.reset`, `relay.shell.usage.as_of`, `relay.shell.usage.row`, `relay.shell.usage.row_family`, `relay.shell.usage.unit_usd`

## Preserved evidence chain
- `reports/c374256ce693/second-independent-highvisibility-final65.json`
- `reports/c374256ce693/second-independent-highvisibility-countdelta.json`
- `reports/c374256ce693/second-independent-highvisibility-es02-reconciliation.json`
- Final FR raw probe: `reports/c374256ce693/fr-final-million-output.json`
- Final ES raw probe: `reports/c374256ce693/es-final-token-output.json`
- Current argument results: `reports/c374256ce693/final-five-arguments.json`
- Consolidated structured final audit: `reports/c374256ce693/second-independent-highvisibility-FINAL-five-locales.json`

Earlier reports remain unchanged. This final report supersedes their blocked dispositions only through the explicit fixes and ES false-positive withdrawal documented above.
