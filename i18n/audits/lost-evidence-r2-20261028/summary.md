# Lost i18n evidence — independent re-audit and remediation, 2026-10-28

**Audit ids:** `le28r2` (24 pairs) and `le28r2-remed` (delta re-audit of the 11 corrected keys).
**State at this head:** all 24 current rows `passed`; `generate.py --check` green; eligible set restored to `en, fr, es, zh-CN, ru, vi, ur, hi`.

24 bundles cover 4,628 locale-key comparisons (`tui.transcript` 17, `tui.authoring` 49, `tui.team_authoring` 36, `tui.palette` 7, `wire.settings` 820 keys × the locales that carry them). Nothing from the deleted scratchpads was reconstructed.

## Two phases, both evidenced

**Phase 1 — judgement, on the pre-remediation bytes** (`9495f8a6c3bf`): **13 PASS / 11 FINDINGS — 0 blocker, 8 major, 26 minor**. Each pair holds the en source, locale catalogue, en context sidecar, glossary and locale guide; findings carry exact key, current text, proposal and disposition. Two child-graded majors were reconciled to minor with the reason recorded (fr `local_providers.*.label` elision; hi `shell_environment_exclude` wording) and one proposed transliteration (`endpoint`/`merge`/`pull`/`push`) was rejected against the glossary's all-locale kept-English set.

**Phase 2 — translator remediation + delta re-audit** (`4d38a1d4b8e`): 11 value replacements in 4 files (fr 2, ru 4, vi 3, hi 2), no key added or removed, no English source, context sidecar or generated artifact touched. The delta audit re-read only those keys and their direct sentences: **PASS, 0 unresolved blocker/major**; 8 majors raised → 8 resolved. The unchanged keys keep their phase-1 judgement, by design (round 2 is delta-scoped).

Result after phase 2: **17 PASS-equivalent pairs** (13 clean + 4 remediated) and **7 pairs carrying minor findings only** — non-gating follow-ups, listed in their `findings.json`.

## Provenance, stated exactly

1. **The linguistic judgement came from direct git/python reads** in an owned sparse worktree. Those reads are **not** `read_public_copy` and are never labelled as such.
2. **Sanctioned `read_public_copy` receipts were obtained afterwards** via `lop exec --run-in <worktree> --tools read_public_copy` — 29 pre-remediation files (5 en sources + 24 locale catalogues) plus the 4 corrected catalogues at `4d38a1d4b8e`. Each receipt is the tool's own audit-log line and each digest was independently recomputed from the base/remediation blob and matches. Receipts confirm the bytes, not the judgement.
3. **The en context sidecars are not routable at this tool version**: `read_public_copy` requires a flat key→string catalogue, so all five `.context.json` files are refused (`refused: catalogue keys and values must be strings`, verbatim in the child session transcript). Every record marks `sanctioned_route_verified: "partial"` and keeps the refusal; this is a disclosed limitation, not a waiver.

Human/native-speaker certification was waived by the operator for all supported locales; that is **not** a waiver of independent LLM audit, which this is. No auditor here is a native speaker, Hindi included.

## Ledger handling

All 401 pre-existing ledger objects keep every field, source hash and prior verdict. The 24 rows that cite deleted scratchpads each keep their note verbatim and gain a dated supersession sentence. 24 current-source rows are appended and are the rows the generator reads.

## Measured eligibility

- Pre-remediation state: `_shipped_locales()` = `[en, fr, es, zh-CN, ur]` (ru/vi/hi held by the three blocked `wire.settings` pairs) and `generate.py --check` exited 1.
- At this head: `_shipped_locales()` = `[en, fr, es, zh-CN, ru, vi, ur, hi]`, matching the committed `shipped.json`; `generate.py --check` exits 0 with **100 byte-identical artifacts**. Nothing needed regenerating, which is itself the check that the ledger rows agree with the committed bytes.

## Validation (actual output)

- `scripts/i18n/check.py --catalogues-only` → **exit 0**, 0 problems.
- `scripts/i18n/check.py` → **exit 0**, 7 advisory findings, **sparse-tree AST only** (34 files) — not whole-repo ratchet coverage.
- `scripts/i18n/generate.py --check` → **exit 0**, 100 artifacts byte-identical.
- `verify_evidence.py` → **PASS**: 401 historical objects preserved, 24 dated notes, 24 appended rows, **345 companion hashes, 120 base input hashes, 48 routed receipts, 11 remediated findings verified**, byte-corruption negative control detected.
- ICU key/argument/category parity passes; plural branches rendered through the repo runtime at 0/1/2/5/21/22/10⁶/1.5·10⁶. French/Russian cache-TTL and rolling-day help use worded equivalents (`une heure`, `doublé`, `часовой`, `вдвое`, `суточный`); all limits present.
- Declared sidecar `maxCells` audits pass. Separately measured, inferred 29-cell label budget overruns: en 6 / es 57 / fr 48 / ru 39 / vi 23 / hi 0 (fr unchanged by the remediation; its two corrected labels went 18→15 and 30→32 cells, the latter already over budget).

## Not verified (stated, not implied)

Deleted audit reports and their original model histories; the lost OperatorApp 60×24 / 100×24 previews and `geometry-report.json`; top/settled frame identity; native terminal/font shaping (Devanagari included); native-speaker idiom certification. No screenshot, geometry file or old scratchpad artefact exists in these bundles. No full suite, browser, live app or real UI flow ran. The 2026-10-25 deletion date/count are the request's incident account, not a reconstructed filesystem event.

## Remaining follow-ups (non-gating)

The 26 minors and the advisory leads in each `findings.json` are recorded and dispositioned but not fixed: cross-namespace consistency (es/fr/ru label widths, ru «название»/«метка», vi «lớp», es «subagente» vs bare «hijo»), terminology drift inside `wire.settings` help text, and typography nits. They do not block eligibility and are left for the i18n programme's own backlog rather than this evidence PR.
