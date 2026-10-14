# PR2329 — second independent highvisibility count-noun delta audit

**Actual model:** openai/gpt-6.1-sol; no fallback. No target/source edits.

**Verdict: BLOCKED.** Original FR/ES singular majors are fixed, but both introduce an exact-numeric-million MAJOR. RU/VI/HI delta is clean.

Three EN noun changes; 3 target noun changes FR/RU/VI/HI and 7 ES changes plus direct composition neighbors. Untouched byte compatibility proven; no full65 re-audit.

Source `de9b9a490c839d6de2c2e41ea03ec819f66be489` + uncommitted 3-noun correction. Shell SHA-256 `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560`; context `c1eb5190cc1c0737bc53320e08a8ef28dd1ec01d8ccf9ba9d5a7b315d2c9a8a1`.

## Actual execution
Isolated HOME, env -i, TZ=UTC, existing intl-messageformat from relay node_modules; no installs/browser/heavy gates. Values: 0,1,2,1.5,-1,-2,-1.5,1000000,-1000000,5,21 × credits/USD/tokens/requests × used/remaining. Unknown-unit statuses, direct rows/dates/percentage regression neighbors also exercised. Actual Python message_arguments: all 12 × 5 PASS.

HOLD — customer resource restriction; text-only existing-dependency formatter probe, no pixels/app-flow/shipped assertion

## Findings
### HV2-FR-02 — MAJOR
The new many branches prefix de to full decimal-digit million amounts. De is appropriate after spelled million(s)/millón/millones, not after the exact-number formatting used here. Common million-token totals therefore render ungrammatically.
Locations: `historical-scratch/c4131ebe3f64/relay-final65-counted/fr/relay.shell.json`: relay.shell.usage.unit_credits line 37, relay.shell.usage.unit_tokens line 39, relay.shell.usage.unit_requests line 40
Actual: `reste 1 000 000 de crédits` / `utilisation de 1 000 000 de crédits` / `reste 1 000 000 de jetons` / `utilisation de 1 000 000 de jetons` / `reste 1 000 000 de requêtes` / `utilisation de 1 000 000 de requêtes`
OQLF Nombres exacts: when million/milliard are written in figures, La préposition de est absente.
https://vitrinelinguistique.oqlf.gouv.qc.ca/21647/la-typographie/nombres/ecriture-de-million-et-de-milliard
Keep the required many category but render the bare plural noun in it (same noun as other); do not alter remaining/used wrappers or introduce captions. Current number formatting is exact decimal digits, not compact/spelled million.

### HV2-ES-02 — MAJOR
The new many branches prefix de to full decimal-digit million amounts. De is appropriate after spelled million(s)/millón/millones, not after the exact-number formatting used here. Common million-token totals therefore render ungrammatically.
Locations: `historical-scratch/5ad0af2dd59e/es-final65-count-aware/relay.shell.json`: relay.shell.usage.unit_credits line 38, relay.shell.usage.unit_tokens line 40, relay.shell.usage.unit_requests line 41
Actual: `Cantidad restante: 1.000.000 de créditos` / `Uso: 1.000.000 de créditos` / `Cantidad restante: 1.000.000 de tókenes` / `Uso: 1.000.000 de tókenes` / `Cantidad restante: 1.000.000 de solicitudes` / `Uso: 1.000.000 de solicitudes`
RAE §4.3 ties de to explicit numeral nouns (15 millares DE ejemplares; 327 millones DE habitantes), unlike all-digit cardinal expressions. Fully numeric 1.000.000 créditos has no numeral noun licensing de.
https://www.rae.es/ortografía/uso-combinado-de-cifras-y-palabras
Keep the required many category but render the bare plural noun in it (same noun as other); do not alter remaining/used wrappers or introduce captions. Current number formatting is exact decimal digits, not compact/spelled million.

## Per-locale disposition
### FR — BLOCKED
HV2-FR-01 fixed for singular/fraction/negative ordinary amounts
0/1/1.5 use singular; 2/5/21 plural; negative magnitudes select correctly. No adjacent doubled de, but 1 000 000 de crédits is a new many-branch defect.

### ES — BLOCKED
HV2-ES-01 fixed: Cantidad restante avoids singular verb issue, singular nouns correct, unknown-unit Cantidad restante: 1 correct.
0/2/1.5 plural, ±1 singular; reset/as_of neutral dates correct. New million de is defective.

### RU — CLEAN
one/few/many/other: 1 кредит; 2 кредита; 0/5/1000000 кредитов; 1.5 кредита; 21 кредит. Negative magnitudes correct. Parentheses retained unchanged, readable units.

### VI — CLEAN
Other-only invariant tín dụng/token/yêu cầu natural at all sampled counts including fractions/negatives/million.

### HI — CLEAN
one/other identical native direct forms क्रेडिट/टोकन/अनुरोध valid for direct counted nouns across all sampled counts, including 10,00,000 grouping. No numeric agreement defect.

## Exact changed keys and target hashes
### FR — `historical-scratch/c4131ebe3f64/relay-final65-counted/fr`
Changed: `relay.shell.usage.unit_credits`, `relay.shell.usage.unit_tokens`, `relay.shell.usage.unit_requests`
- relay.shell: `b2ddca1751c5a650fa00f29035944701c17dc1039480f80abe74ae1a322bc84c`
- relay.session: `ff308557eaaac62556f20625039d9e70df97cf55f5b150580f7511c16ecd2018`
- relay.login: `5f6cc95bffee5f11984b4a083af057ba96a5e332a623c5b5c3659a9e61530d24`
### ES — `historical-scratch/5ad0af2dd59e/es-final65-count-aware`
Changed: `relay.shell.usage.remaining`, `relay.shell.usage.reset`, `relay.shell.usage.as_of`, `relay.shell.usage.unit_credits`, `relay.shell.usage.unit_tokens`, `relay.shell.usage.unit_requests`, `relay.shell.usage.remaining_unknown_unit`
- relay.shell: `852545691f324684e456b67d9a1e6b3193b346a82bc544bb340f3f6a4e5e32d2`
- relay.session: `17a2bf64b2a6f5ffeb10d7d8d54a06d76e6a3102c477681fdf59815a57b463f2`
- relay.login: `b6e676acdc5cbb9a85423c458a19a5d6e5bc35594eb1dd21b23569af32f29658`
### RU — `historical-scratch/a753be6f0d01/relay-minimal3noun/ru`
Changed: `relay.shell.usage.unit_credits`, `relay.shell.usage.unit_tokens`, `relay.shell.usage.unit_requests`
- relay.shell: `b96e8923fe82824db2434720f722d2cf323d5aa733d78ad1c75285391c16245b`
- relay.session: `1d3618c0928e956da89e70b9eaf0945c15062493a043a6e17a8364fc912efdec`
- relay.login: `6235f9f3f058fe859194f37879d485199449345c1692beeda2dc767f691dfaa1`
### VI — `historical-scratch/590f5d5140c6/vi-final65-countplural/vi`
Changed: `relay.shell.usage.unit_credits`, `relay.shell.usage.unit_tokens`, `relay.shell.usage.unit_requests`
- relay.shell: `f475dbefccd75d713fdea93fd8c5a12b541f94d00ec00c23b00fe72e2e942319`
- relay.session: `7743c7b5d018b84395478e99231714a46cd998e8c78ce465333be8df62fd517a`
- relay.login: `a87e8ed3381143ee8d61dada52b0df604c72191cdde1747b9ea8fd6c46537020`
### HI — `historical-scratch/a862db3d9624/relay-countnoun3/hi`
Changed: `relay.shell.usage.unit_credits`, `relay.shell.usage.unit_tokens`, `relay.shell.usage.unit_requests`
- relay.shell: `feab9e023be86d7841bd1e08cb273a48bbaaa118c41ade6040d7a29a701305eb`
- relay.session: `0fb4a5abfa0fb1ca26c2683a16a6db2f84f0f44fbf55522a7a21ec33deacb43e`
- relay.login: `589c538128272c5cd7c54468f9e126dda52a004ee21b9eab1949590b5156c4ff`

All bytes outside changed-key lines match old reviewed targets in all 15 files; no new unsampled approval implied. Target hashes rechecked unchanged.

Full JSON: `reports/c374256ce693/second-independent-highvisibility-countdelta.json`
Probe: `historical-scratch/c374256ce693/delta-count-probe.cjs`
Input: `reports/c374256ce693/delta-count-input.json`
Output: `reports/c374256ce693/delta-count-output.json`
Argument results: `reports/c374256ce693/delta-count-arguments.json`
