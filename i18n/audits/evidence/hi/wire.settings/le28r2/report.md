# hi / wire.settings — lost-evidence re-audit (le28r2)

Date: 2026-10-28. Verdict: **FINDINGS**; publication ledger **blocked**; counts {'blocker': 0, 'major': 1, 'minor': 3, 'nit': 0}.

## Provenance and scope

820/820 current keys reviewed. Linguistic source revision `aeb9bb9dc34e3988c78ebbb6921e79e049143cbf`; packaging base `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`. Input SHA-256 values and phase-specific actual models are in `record.json`.

DeepSeek performed the initial repository survey. Sonnet performed the TUI source/target/context comparison and mechanical probes. Five independent child auditors read all wire.settings keys under Sonnet after their DeepSeek quota fallbacks. GPT completed context coverage, reconciled findings/severity, and packaged evidence after a further provider-quota fallback. No model was a human/native speaker.

No native-speaker sign-off is claimed or required by the operator; uncertainty remains disclosed.

## Semantic judgement

All 820 keys read by the named independent locale subauditor against English and its guide/glossary; full sidecar (32 entries) read. Coordinator verified current-text quotes, gating findings, numerals flagged by mechanical comparison, and selected minors. Unselected child observations remain in the immutable raw report as non-gating/unendorsed leads, not silently counted as accepted. No plurals/selects. All 8 explicit sidecar maxCells limits passed, but inferred 29-cell row-label overflows are recorded separately. Numeric token differences for French/Russian cache TTL and Russian rolling-day budget are word-form equivalents (une heure/часовой, doublé/вдвое, суточный), not missing thresholds.

## Findings and disposition

`findings.json` contains exact keys, current/source text, proposed fixes and final severity. PASS means no accepted textual defect in this bounded LLM audit; FINDINGS may still be non-gating when every finding is minor. Blocker/major findings block linguistic approval. Separately, missing sanctioned-reader-route verification holds every new ledger row blocked.

No catalogue correction was applied. A separate translator must remediate gating findings, followed by an independent delta audit. Original failed reports remain immutable.

### F1 — major — `wire.settings.proactive_patience_episode_ttl_ms.help`

‘wake/fire’ is rendered ‘जागरण’ (vigil/awakening), but glossary §2 and the hi guide fix wake → जागृति (the rest of this namespace uses जागृति: aida_cadence_min_gap_minutes.help, wakes.*). A different term for the same concept breaks the one-term invariant.

Proposed: एक चक्र की कड़ी सीमा, इंतज़ार चाहे कैसे भी तय हुए हों (मिलीसेकंड; 7200000 = 2 h)। इसके बाद आने वाली जागृति चुपचाप बंद हो जाती है।

### F2 — minor — `wire.settings.shell_environment_exclude.help`

‘children’ here means child processes of the bash/eval tools, not agents. ‘चिल्ड्रन’ (literally ‘kids’) is a bare transliteration a native reader will find odd/unclear, and the hi guide records that the child-only transliteration was abandoned. Use ‘child process’ wording.

Proposed: खाली = कुछ नहीं हटाया जाता। ऐसे नाम जो एजेंट के चलाए कमांड को किसी भी मोड में कभी नहीं दिए जाते — सत्र क्रेडेंशियल स्टोर के अपने इंजेक्शन समेत, यानी किसी वेरिएबल को सीधे मना करने का यही तरीक़ा है। सिर्फ़ bash और eval की चाइल्ड प्रोसेस पर लागू; MCP सर्वर और हार्नेस की अपनी प्रक्रियाएँ इसके दायरे में नहीं आतीं।

### F3 — minor — `wire.settings.speech.description`

‘cannot express’ rendered with ‘दिखा सकता’ (show) and ‘degrade with a note’ as ‘कम हो जाते हैं’ (get reduced) — the sense (unsupported fields are downgraded/approximated with a note rather than erroring) is not conveyed.

Proposed: जिन फ़ील्ड को कोई प्रोवाइडर व्यक्त नहीं कर सकता, वे विफल होने के बजाय नोट के साथ सरल रूप में लागू होते हैं।

### F5 — minor — `wire.settings.supplements_denyprefixes.help`

en says ‘Comma-separated folders’; the input-format cue is dropped (every sibling list help — providers_openrouter_ignore/only/order, static_roots, web_search_providers — says ‘कॉमा से अलग’).

Proposed: हाइलाइट्स चालू होने पर ही लागू। कॉमा से अलग फ़ोल्डर, जिनकी फ़ाइलें कभी सूचीबद्ध नहीं होतीं और निर्णय मॉडल को नहीं भेजी जातीं; ग्राहक डेटा वाली मशीन पर इस्तेमाल करें।

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

The linguistic judgement above came from **direct git reads** and stays labelled that way. Sanctioned `read_public_copy` receipts were then obtained through `lop exec --run-in` at this base: `local_operator/i18n/catalogues/en/wire.settings.json` (session `f3b2ff8fdc0d`) and `local_operator/i18n/catalogues/hi/wire.settings.json` (session `3ac46147b332`). Each receipt's sha256 was recomputed from the base blob and matches, so the route read the same bytes this audit judged; it does not certify the judgement. The en context sidecar is **not routable**: `read_public_copy` refuses it with `refused: catalogue keys and values must be strings` because its values are objects, not strings, so it remains a direct read. The pair is therefore not fully routed and that is disclosed rather than waived.

Linguistic status: **blocked** — 1 unresolved blocker/major finding(s) require translator remediation and an independent delta audit.