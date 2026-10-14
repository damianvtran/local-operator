# Render-held relay localization archive

This branch is a durable archive, NOT a locale PR or publication approval. It is
cut from code-only commit `57bcb939881066c7e645cdec604fd72c3e667471` and adds only
this documentation directory. Never merge/activate these assets before the actual
rendered/full-publication audits pass. Nothing here is a generator input.

- `candidates/<locale>/`: 15 exact audited JSON files, 325 entries. Their raw
  hashes agree with independent first/second native audits; no reformatting.
- `pending-ledger-rows.json`: 15 **translated**, not passed, pending provenance
  rows. Do NOT copy them into the active ledger while rendered evidence is held.
- `source-contracts/en/`: final English/context snapshots for exact hash identity.
- `reports/<session>/`: final native and second highvisibility reports, context
  compatibility addenda, and explicitly referenced prior coverage/failure/
  withdrawn-finding receipts. No claim that all old reports are current clean.
- `manifest.json`: identity roster, exact source/target pins, original raw report
  digests versus archive-copy digests, dispositions, excluded references/limits.

Only personalized absolute path references were sanitized in report copies;
original scratch reports remain untouched. Paths in prose are logical paths from
this archive root. `source-checkout` means the original audited Git revision, not
the mutable final source snapshot; `historical-scratch` records old provenance.
Historical source/target hashes and failed/withdrawn findings are preserved, not
relabelled as current evidence. Native/DOM-text approval does not certify pixels,
fit, geometry, pseudo/nativeRTL, daemon authentication or release eligibility.

The generator checks every namespace a locale carries, requiring current-source
passed rows for all of them. A translated/render-held carried namespace removes
that locale from SHIPPED globally; per-namespace fallback does NOT waive that
eligibility rule. Staging outside the active roots avoids the intermediate209d
regression without a mechanism change. Future minimum path: finish actual render
and publication audits, then propose a separate reviewed locale PR bound to these
exact pins and truthful complete rows. Do not filter pending rows or weaken guards.

The active PR head and working checkout remain unchanged by archival. No PR,
merge, tag, release or runtime promotion is authorized by this archive.
