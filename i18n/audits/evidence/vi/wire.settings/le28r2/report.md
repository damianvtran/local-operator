# vi / wire.settings — lost-evidence re-audit (le28r2)

Date: 2026-10-28. Verdict: **FINDINGS**; publication ledger **blocked**; counts {'blocker': 0, 'major': 3, 'minor': 3, 'nit': 0}.

## Provenance and scope

820/820 current keys reviewed. Linguistic source revision `aeb9bb9dc34e3988c78ebbb6921e79e049143cbf`; packaging base `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`. Input SHA-256 values and phase-specific actual models are in `record.json`.

DeepSeek performed the initial repository survey. Sonnet performed the TUI source/target/context comparison and mechanical probes. Five independent child auditors read all wire.settings keys under Sonnet after their DeepSeek quota fallbacks. GPT completed context coverage, reconciled findings/severity, and packaged evidence after a further provider-quota fallback. No model was a human/native speaker.

No native-speaker sign-off is claimed or required by the operator; uncertainty remains disclosed.

## Semantic judgement

All 820 keys read by the named independent locale subauditor against English and its guide/glossary; full sidecar (32 entries) read. Coordinator verified current-text quotes, gating findings, numerals flagged by mechanical comparison, and selected minors. Unselected child observations remain in the immutable raw report as non-gating/unendorsed leads, not silently counted as accepted. No plurals/selects. All 8 explicit sidecar maxCells limits passed, but inferred 29-cell row-label overflows are recorded separately. Numeric token differences for French/Russian cache TTL and Russian rolling-day budget are word-form equivalents (une heure/часовой, doublé/вдвое, суточный), not missing thresholds.

## Findings and disposition

`findings.json` contains exact keys, current/source text, proposed fixes and final severity. PASS means no accepted textual defect in this bounded LLM audit; FINDINGS may still be non-gating when every finding is minor. Blocker/major findings block linguistic approval. Separately, missing sanctioned-reader-route verification holds every new ledger row blocked.

No catalogue correction was applied. A separate translator must remediate gating findings, followed by an independent delta audit. Original failed reports remain immutable.

### F1 — major — `wire.settings.web_fetch_render_backend.help`

Glossary §2 wave-2 kept-English set fixes 'extra' (pip extras) as English, undeclined; es/fr/ru all keep 'extra'. 'gói phụ trợ' translates the term and obscures the pip-extra name.

Proposed: auto dùng markdownify khi extra [fetch] được cài.

### F2 — major — `wire.settings.speech_transcription_cleanup.help`

'âm đệm' is the phonology term for the medial glide (the /w/ in 'hoa'), not filler words (um/uh). Filler words are 'từ đệm' / 'từ thừa'. Same wrong term in ...choice.off.description. LOW-MEDIUM confidence (I am not a native speaker) but the phonological sense is standard.

Proposed: Cách nhập bằng giọng nói loại bỏ từ đệm và âm thanh phi ngôn ngữ. auto dùng tính năng làm sạch tích hợp sẵn của nhà cung cấp khi có và quy tắc thận trọng phía máy khách trong các trường hợp còn lại; provider và client mỗi bên chỉ dùng một trong hai cơ chế; off giữ nguyên bản ghi. Trường `cleanup` trong từng yêu cầu vẫn được ưu tiên.

### F2b — major — `wire.settings.speech_transcription_cleanup.choice.off.description`

Same 'âm đệm' mis-term as F2; 'tín hiệu' for 'cue' is also vaguer than the help's 'âm thanh phi ngôn ngữ'.

Proposed: Giữ nguyên; không loại bỏ từ đệm hay dấu hiệu phi ngôn ngữ.

### F3 — minor — `wire.settings.web_search_timeout_seconds.help`

'kẹp' is a calque of 'clamp'; the same catalogue renders 'Clamped 60 s..24 h' as 'Được giới hạn trong khoảng' (proactive_patience_default_ms.help). Same calque in speech_voice_pace.help ('tự kẹp về giới hạn của nó').

Proposed: Hết giờ cho mỗi yêu cầu nhà cung cấp. Được giới hạn trong khoảng 1-120 khi đọc.  /  speech_voice_pace.help: '... Nhà cung cấp có dải hẹp hơn sẽ tự giới hạn theo dải của nó.'

### F6 — minor — `wire.settings.providers_openrouter_sort.choice.unset.description`

'không tùy chọn' reads as 'no options', not 'no preference set'. Same in providers_openrouter_allow_fallbacks.choice.unset.description and providers_openrouter_require_parameters.choice.unset.description.

Proposed: không đặt tùy chọn — định tuyến cố định được giữ (cache ấm nhất)

### F7 — minor — `wire.settings.speech_transcription_cleanup.choice.auto.description`

Elliptical: drops 'rules' and reads as 'built-in; otherwise, the client' (the 'client' choice description says 'Quy tắc phía máy khách').

Proposed: Bản của nhà cung cấp; nếu không, quy tắc phía máy khách.

## Subaudit reconciliation

The original JSON is retained byte-for-byte. Final findings are bounded to the substantive defects and highest-value minors; other original observations are advisory/not endorsed rather than silently fixed. The Hindi raw report has no accompanying markdown file because none was produced. The Spanish original markdown has a stale last-line summary acknowledged by its author; use original JSON plus final adjudication, not that line.

## What was and was not re-derived

- No lost scratchpad, screenshot, geometry-report.json, prior report hash, or old audit trajectory was reconstructed. The deletion date/count in the task are supplied incident context, not independently proven deletion events.

- No live OperatorApp, real terminal/font, browser, native app, 60x24/100x24 frame or settled-frame equality was exercised. Those previous visual/geometry claims are UNVERIFIABLE in this re-audit. Rich cell_len and ICU output are scalar/text evidence only, not layout or shaping proof.

- Style/semantic decisions are independent LLM review, not native-speaker certification. Hindi idiom would need a native speaker for that certification; none of these auditors is one. The operator waived human/native-speaker sign-off for all locales; it is not a merge gate here.

- No catalogue text was changed. Proposed fixes are recommendations to a separate translator, not translations authored by this auditor and self-approved.

- The full check.py result is from a SPARSE tree; its AST scanned-file count is not whole-repository coverage. --catalogues-only checked the complete carried catalogues. No full test suite or UI QA was run.

- Prior palette target hashes, French 48/51-cell alternatives, scratchpad cap label widths, and queued strict-prefix claims were re-derived where applicable. This does not recover their original reports or prove their original model identities.

- `audit.log` contains actual command stdout/stderr and exit codes. `mechanical.json` contains heuristic leads as well as hard checks: config tokens/ICU quoting and literal member grammar cause known false positives; a lead is not a semantic finding.

- `receipt.json` hashes all bundle companions. No screenshot, app geometry, native shaping or old scratchpad artifact is included.

## Sanctioned reader-route addendum (packaging, 9495f8a6c3bf)

The linguistic judgement above came from **direct git reads** and stays labelled that way. Sanctioned `read_public_copy` receipts were then obtained through `lop exec --run-in` at this base: `local_operator/i18n/catalogues/en/wire.settings.json` (session `f3b2ff8fdc0d`) and `local_operator/i18n/catalogues/vi/wire.settings.json` (session `3e7920a93342`). Each receipt's sha256 was recomputed from the base blob and matches, so the route read the same bytes this audit judged; it does not certify the judgement. The en context sidecar is **not routable**: `read_public_copy` refuses it with `refused: catalogue keys and values must be strings` because its values are objects, not strings, so it remains a direct read. The pair is therefore not fully routed and that is disclosed rather than waived.

Linguistic status: **blocked** — 3 unresolved blocker/major finding(s) require translator remediation and an independent delta audit.