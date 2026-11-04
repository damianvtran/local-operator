# fr / wire.settings — lost-evidence re-audit (le28r2)

Date: 2026-10-28. Verdict: **FINDINGS**; linguistic status **passed**, publication ledger **blocked** for missing reader-route verification; counts {'blocker': 0, 'major': 0, 'minor': 5, 'nit': 0}.

## Provenance and scope

820/820 current keys reviewed. Linguistic source revision `aeb9bb9dc34e3988c78ebbb6921e79e049143cbf`; packaging base `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`. Input SHA-256 values and phase-specific actual models are in `record.json`.

DeepSeek performed the initial repository survey. Sonnet performed the TUI source/target/context comparison and mechanical probes. Five independent child auditors read all wire.settings keys under Sonnet after their DeepSeek quota fallbacks. GPT completed context coverage, reconciled findings/severity, and packaged evidence after a further provider-quota fallback. No model was a human/native speaker.

No native-speaker sign-off is claimed or required by the operator; uncertainty remains disclosed.

## Semantic judgement

All 820 keys read by the named independent locale subauditor against English and its guide/glossary; full sidecar (32 entries) read. Coordinator verified current-text quotes, gating findings, numerals flagged by mechanical comparison, and selected minors. Unselected child observations remain in the immutable raw report as non-gating/unendorsed leads, not silently counted as accepted. No plurals/selects. All 8 explicit sidecar maxCells limits passed, but inferred 29-cell row-label overflows are recorded separately. Numeric token differences for French/Russian cache TTL and Russian rolling-day budget are word-form equivalents (une heure/часовой, doublé/вдвое, суточный), not missing thresholds.

## Findings and disposition

`findings.json` contains exact keys, current/source text, proposed fixes and final severity. PASS means no accepted textual defect in this bounded LLM audit; FINDINGS may still be non-gating when every finding is minor. Blocker/major findings block linguistic approval. Separately, missing sanctioned-reader-route verification holds every new ledger row blocked.

No catalogue correction was applied. A separate translator must remediate gating findings, followed by an independent delta audit. Original failed reports remain immutable.

### F1 — minor — `wire.settings.local_providers.base_url.label`

{name} is a LOCAL_PRESETS display name: 'LM Studio','Ollama','vLLM','llama.cpp','OpenAI-compatible' (providers/local.py:16-26; consumed at settings_io.py:4701). 'endpoint de {name}' renders 'endpoint de Ollama' and 'endpoint de OpenAI-compatible' — in French 'de' must elide before a vowel (d’Ollama, d’OpenAI-compatible) and ICU cannot elide, so the phrasing must avoid 'de {name}'.

Proposed: endpoint {name}

### F2 — minor — `wire.settings.local_providers.models.label`

'surcharges de modèle de {name}' renders 'de Ollama' / 'de OpenAI-compatible' (same elision defect as F1; settings_io.py:4711).

Proposed: surcharges de modèle pour {name}

### F3 — minor — `wire.settings.librarian.description`

U+0020 (ASCII space) before ':' — the style guide requires U+202F before : ; ! ? in prose; every other key in the namespace uses U+202F (177 occurrences). Same defect in librarian_cadence_min_gap_minutes.help (before ';'), librarian_cadence_paused.help (before ';'), librarian_enabled.help (before ';'): replace the U+0020 with U+202F in each.

Proposed: Le Bibliothécaire garde la Bibliothèque globale organisée : son nom affiché, la passe quotidienne et la pause. La Bibliothèque est durable et partagée par toutes les conversations.

### F4 — minor — `wire.settings.remote_compute.description`

Contains U+0020 followed by U+202F before ':' ('Aperçu' + ' ' + U+202F + ':'), i.e. a visible double space. Sibling remote_compute_enabled.help is correct.

Proposed: Aperçu : exécuter le travail de l’agent sur le calcul hébergé par Radient, rattaché à l’un de vos maillages.

### F6 — minor — `wire.settings.librarian_cadence_max_extra_per_day.help`

The Librarian is masculine in this section ('Le Bibliothécaire', 'Bibliothécaire activé', 'désactivé/activé' choices) but this help says 'elle peut demander' (en: 'she'). Aida is consistently feminine (Cheffe de cabinet activée). Choose one gender for the Librarian and keep it throughout the section.

Proposed: Combien de passes supplémentaires il peut demander par jour. 0 les désactive. (or feminise the section: La Bibliothécaire, Bibliothécaire activée, activée/désactivée)

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

The linguistic judgement above came from **direct git reads** and stays labelled that way. Sanctioned `read_public_copy` receipts were then obtained through `lop exec --run-in` at this base: `local_operator/i18n/catalogues/en/wire.settings.json` (session `f3b2ff8fdc0d`) and `local_operator/i18n/catalogues/fr/wire.settings.json` (session `60a1513c4603`). Each receipt's sha256 was recomputed from the base blob and matches, so the route read the same bytes this audit judged; it does not certify the judgement. The en context sidecar is **not routable**: `read_public_copy` refuses it with `refused: catalogue keys and values must be strings` because its values are objects, not strings, so it remains a direct read. The pair is therefore not fully routed and that is disclosed rather than waived.

Linguistic status: **passed** — no unresolved blocker/major finding; minors above are non-gating follow-ups.