# Attestation — zh-CN / tui.welcome / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 30 (en) / 30 (zh)

## Hash equality proof

- Source: prior row `cd5bb5b95c8e71d4bf89fdc7f26bede91dee8fadedb5ef786a7f3e7731c57a50` = held `cd5bb5b95c8e71d4bf89fdc7f26bede91dee8fadedb5ef786a7f3e7731c57a50` = current `cd5bb5b95c8e71d4bf89fdc7f26bede91dee8fadedb5ef786a7f3e7731c57a50` — **equal**
- Target: held `c9ba858be971494e1a12947b8d76ceed5b3a6cb68bbed7cebfa8f1338faf7423` = current `c9ba858be971494e1a12947b8d76ceed5b3a6cb68bbed7cebfa8f1338faf7423` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`bfd46d9dbcc5c4dfbee6a3fd93e47264f17f123e01ddf84a7aae99869c41debd`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 30 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 18 declared `maxCells` key(s), 0 over budget

## Stratified sample

5/30 keys read (16.7%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
