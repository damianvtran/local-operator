# PR2329 — HV2-ES-02 reconciliation

**Reviewer/model:** independent task / openai/gpt-6.1-sol; no fallback.

## Corrected verdict
**ES sampled linguistic audit: CLEAN — terminal. HV2-ES-02 withdrawn as a false positive.** Original HV2-ES-01 remains fixed. No ES blocker/major remains in this reviewed sample.

## Primary authority defeats the earlier inference
RAE DPD, cardinales §6: https://www.rae.es/dpd/cardinales
> Lo mismo sucede si se escriben con números: 1 000 000 de personas, 1 200 000 personas.

My previous conclusion from Ortografía §4.3 was an overbroad inference. The DPD explicitly covers all-digit exact-million notation, and the existing ES many branches are correct. Do **not** remove their de based on my previous report.

## Actual current-target contrast
Existing relay intl-messageformat, isolated HOME/env -i/TZ=UTC; no dependency installation or browser.
- `Cantidad restante: 1.000.000 de créditos`
- `Uso: 1.000.000 de créditos`
- `Cantidad restante: 1.200.000 créditos`
- `Uso: 1.200.000 créditos`

The same contrast passes for tókenes/solicitudes; exact million selects many with de, complex 1.2 million selects other without de.

## Exact target pin
Directory: `historical-scratch/5ad0af2dd59e/es-final65-count-aware`
- relay.shell SHA-256: `852545691f324684e456b67d9a1e6b3193b346a82bc544bb340f3f6a4e5e32d2`
- relay.session SHA-256: `17a2bf64b2a6f5ffeb10d7d8d54a06d76e6a3102c477681fdf59815a57b463f2`
- relay.login SHA-256: `b6e676acdc5cbb9a85423c458a19a5d6e5bc35594eb1dd21b23569af32f29658`

All three hashes equal the previous delta-audited files. Unchanged prior native/text sample coverage retained, not re-audited. No source or target edits.

## History and remaining hold
Prior evidence preserved: `reports/c374256ce693/second-independent-highvisibility-countdelta.json`. This addendum supersedes only ES02 and the ES blocked verdict. FR02 remains open on the distinct OQLF rule; no revision to other locale findings.
HOLD — customer resource restriction; no pixel, full flow, shipped or release claim

Structured result: `reports/c374256ce693/second-independent-highvisibility-es02-reconciliation.json`
Raw formatter result: `reports/c374256ce693/es-million-reconciliation-output.json`
