# es / tui.team_authoring — lost-evidence re-audit (le28r2)

Date: 2026-10-28. Verdict: **PASS**; linguistic status **passed**, publication ledger **blocked** for missing reader-route verification; counts {'blocker': 0, 'major': 0, 'minor': 0, 'nit': 0}.

## Provenance and scope

36/36 current keys reviewed. Linguistic source revision `aeb9bb9dc34e3988c78ebbb6921e79e049143cbf`; packaging base `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`. Input SHA-256 values and phase-specific actual models are in `record.json`.

DeepSeek performed the initial repository survey. Sonnet performed the TUI source/target/context comparison and mechanical probes. Five independent child auditors read all wire.settings keys under Sonnet after their DeepSeek quota fallbacks. GPT completed context coverage, reconciled findings/severity, and packaged evidence after a further provider-quota fallback. No model was a human/native speaker.

No native-speaker sign-off is claimed or required by the operator; uncertainty remains disclosed.

## Semantic judgement

All 36 keys compared (not the historical 35-key snapshot), including promote_hint. Manager/instructions requirements, member role:2/team:pod grammar, alias clearing, desktop-only overrides, conflict/read-only notices, y/n direction, saved receipt and conversation-to-account promotion meaning checked. No plurals/selects. French colons within role:2/team:pod are literal grammar, not missing French punctuation spaces. Command footer measures 50 cells; wrap/label-column behaviour remains unverified.

## Findings and disposition

`findings.json` contains exact keys, current/source text, proposed fixes and final severity. PASS means no accepted textual defect in this bounded LLM audit; FINDINGS may still be non-gating when every finding is minor. Blocker/major findings block linguistic approval. Separately, missing sanctioned-reader-route verification holds every new ledger row blocked.

No catalogue correction was applied. A separate translator must remediate gating findings, followed by an independent delta audit. Original failed reports remain immutable.

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

The linguistic judgement above came from **direct git reads** and stays labelled that way. Sanctioned `read_public_copy` receipts were then obtained through `lop exec --run-in` at this base: `local_operator/i18n/catalogues/en/tui.team_authoring.json` (session `3bd250c4986d`) and `local_operator/i18n/catalogues/es/tui.team_authoring.json` (session `3bd250c4986d`). Each receipt's sha256 was recomputed from the base blob and matches, so the route read the same bytes this audit judged; it does not certify the judgement. The en context sidecar is **not routable**: `read_public_copy` refuses it with `refused: catalogue keys and values must be strings` because its values are objects, not strings, so it remains a direct read. The pair is therefore not fully routed and that is disclosed rather than waived.

Linguistic status: **passed** — no unresolved blocker/major finding; minors above are non-gating follow-ups.