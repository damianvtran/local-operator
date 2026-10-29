# Attestation — zh-CN / wire.settings / zhca1

- **Verdict: not_carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 818 (en) / 818 (zh)

## Hash equality proof

- Source: prior row `d1bc117e26fd79c55663c7ba1972304509a42370213f90924b0b9dc6ec81f63a` = held `d1bc117e26fd79c55663c7ba1972304509a42370213f90924b0b9dc6ec81f63a` = current `c1e237c4dd3c2ea236f9e39d15cf4d0b34a231841357aee5a389f34201a73582` — **DIFFERS**
- Target: held `b5bcb8ebfaa674d31b155ed64130f4a7a256c40b2c7833cfa1f901d12c5c8138` = current `d0df5e3e6975719b39d3e81171b2d9416844a1df8a5498f81f398476eb284b55` — **DIFFERS** (the prior ledger row does not record a target hash)
- Context sidecar: DIFFERS (`67662db2835ed0f6db3a65844efc318c79706735457eb974bb7c64cf34f861a9`)
- Whole target file byte-identical to held candidate: **false**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 818 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 6 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 7 declared `maxCells` key(s), 0 over budget

## Stratified sample

Not performed: the namespace is not attestable.

## Change since the held candidate

- en keys: 800 -> 818; zh keys: 800 -> 818
- added in en: wire.settings.remote_compute.description, wire.settings.remote_compute.title, wire.settings.remote_compute_enabled.choice.false.description, wire.settings.remote_compute_enabled.choice.true.description, wire.settings.remote_compute_enabled.help, wire.settings.remote_compute_enabled.label, wire.settings.session_compression.description, wire.settings.session_compression.title, wire.settings.session_compression_after_days.help, wire.settings.session_compression_after_days.label, wire.settings.session_compression_enabled.choice.false.description, wire.settings.session_compression_enabled.choice.true.description, wire.settings.session_compression_enabled.help, wire.settings.session_compression_enabled.label, wire.settings.session_scratchpad.description, wire.settings.session_scratchpad.title, wire.settings.session_scratchpad_max_bytes.help, wire.settings.session_scratchpad_max_bytes.label
- added in zh: wire.settings.remote_compute.description, wire.settings.remote_compute.title, wire.settings.remote_compute_enabled.choice.false.description, wire.settings.remote_compute_enabled.choice.true.description, wire.settings.remote_compute_enabled.help, wire.settings.remote_compute_enabled.label, wire.settings.session_compression.description, wire.settings.session_compression.title, wire.settings.session_compression_after_days.help, wire.settings.session_compression_after_days.label, wire.settings.session_compression_enabled.choice.false.description, wire.settings.session_compression_enabled.choice.true.description, wire.settings.session_compression_enabled.help, wire.settings.session_compression_enabled.label, wire.settings.session_scratchpad.description, wire.settings.session_scratchpad.title, wire.settings.session_scratchpad_max_bytes.help, wire.settings.session_scratchpad_max_bytes.label
- removed in zh: none
- changed in zh: none

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. No attestation or semantic read is made: the hashes differ and a fresh audit is required.
