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

All 401 pre-existing ledger rows survive. 24 of them carried an audit whose evidence was the deleted scratchpad; each keeps its note verbatim and gains a dated supersession sentence, and its `auditor` field is replaced by the live recovery auditor. **No rows are appended:** `i18n/ledger.json` admits one row per `(locale, namespace)` at a given English source hash (`tests/unit/i18n/test_hold_split.py`: "a second row at the same hash is a duplicate, not history"), and this recovery re-judged the SAME English bytes, so the legal shape is a refresh in place. The ledger stays at 408 entries and each pair keeps exactly one row; the prior row verbatim is kept in each bundle's `prior-ledger.json`.

## Measured eligibility

- Pre-remediation state: `_shipped_locales()` = `[en, fr, es, zh-CN, ur]` (ru/vi/hi held by the three blocked `wire.settings` pairs) and `generate.py --check` exited 1.
- At this head: `_shipped_locales()` = `[en, fr, es, zh-CN, ru, vi, ur, hi]`, matching the committed `shipped.json`; `generate.py --check` exits 0 with **100 byte-identical artifacts**. Nothing needed regenerating, which is itself the check that the ledger rows agree with the committed bytes.

## Validation (actual output)

- `scripts/i18n/check.py --catalogues-only` → **exit 0**, 0 problems.
- `scripts/i18n/check.py` → **exit 0**, 7 advisory findings, **sparse-tree AST only** (34 files) — not whole-repo ratchet coverage.
- `scripts/i18n/generate.py --check` → **exit 0**, 100 artifacts byte-identical.
- `verify_evidence.py` → **PASS**: 401 historical rows preserved, 24 refreshed in place (33 notes extended), 0 rows appended, **345 companion hashes, 120 base input hashes, 48 routed receipts, 11 remediated findings verified**, byte-corruption negative control detected. Negative controls: a duplicated refreshed row, a reverted refresh (23 of 24), a corrupted companion and a dropped base row are each detected.
- ICU key/argument/category parity passes; plural branches rendered through the repo runtime at 0/1/2/5/21/22/10⁶/1.5·10⁶. French/Russian cache-TTL and rolling-day help use worded equivalents (`une heure`, `doublé`, `часовой`, `вдвое`, `суточный`); all limits present.
- Declared sidecar `maxCells` audits pass. Separately measured, inferred 29-cell label budget overruns: en 6 / es 57 / fr 48 / ru 39 / vi 23 / hi 0 (fr unchanged by the remediation; its two corrected labels went 18→15 and 30→32 cells, the latter already over budget).

## Not verified (stated, not implied)

Deleted audit reports and their original model histories; the lost OperatorApp 60×24 / 100×24 previews and `geometry-report.json`; top/settled frame identity; native terminal/font shaping (Devanagari included); native-speaker idiom certification. No screenshot, geometry file or old scratchpad artefact exists in these bundles. No full suite, browser, live app or real UI flow ran. The 2026-10-25 deletion date/count are the request's incident account, not a reconstructed filesystem event.

## Remaining follow-ups (non-gating)

The 26 minors and the advisory leads in each `findings.json` are recorded and dispositioned but not fixed: cross-namespace consistency (es/fr/ru label widths, ru «название»/«метка», vi «lớp», es «subagente» vs bare «hijo»), terminology drift inside `wire.settings` help text, and typography nits. They do not block eligibility and are left for the i18n programme's own backlog rather than this evidence PR.

## Review round 1 remediation (#3123)

- **F1:** ru `wire.settings.librarian_enabled.help` now reads «его сессия» — the ru catalogue at the reviewed head said `её сессия`, against every other Russian `wire.settings` mention of `librarian` (`его`/`его`/`он`, masculine `Библиотекарь`). Fixed in `bd47fce698d3d498c503f047216e4e442f7d0d67`, exactly one word in one value; `i18n/audits/evidence/ru/wire.settings/le28r2-remed/` re-pins `target_sha256` to `9156360167e72197d97bfb8caeaa0c7e2795b03d0dceeb7195d24b8f9d1f40bd`. The delta was judged by an independent `translation-auditor` (`anthropic/claude-sonnet-5-5`, job `772c338410e1`, not the translator, not a native speaker): PASS, 0 findings. Its `read_public_copy` receipt was compared to the file **by eye and content, not by byte diff**; the byte claim rests on the implementer's recomputed sha256.
- **F2:** the records keep `packaged_at_utc` (first packaging) and gain a truthful `repackaged_at_utc` for the later edit, rather than a back-dated freeze.
- **F3:** left alone, but the reason is that they are verbatim transcripts, not that resealing is impossible (this PR reseals receipts whenever it changes a hashed file, so it plainly is). The `audit.log` and `checks-*.log` lines are the tool's own captured output — the `$ /Users/damian/local-operator/.venv/bin/python …` command lines that say what actually ran where — and the `route-receipts.json` / `routed-read-index.json` entries are the reader's own audit-log records, whose path, session, timestamp and digest are the evidence. Rewriting the home path inside them would make a true transcript say the run happened somewhere it did not, and resealing the receipt afterwards would present the edited transcript as attested. Earlier bundles (zh-CN, ur) carry the same paths. No new home path is introduced beyond the files this PR adds.
- **F4:** `verify_evidence.py.txt` no longer pins the base by total length or by "the last 24": each of the 401 base rows must reappear in order with every field except note/auditor identical, the 24 refreshed rows are located by content (their own bundle citation), and each is asserted to be its pair's last row with no second row at the same English hash. Re-run at this head: `result: PASS`, 408 ledger rows, 24 refreshed, 0 appended, 345 companions, 120 input hashes; controls: corrupted companion detected, dropped base row detected, an unrelated row appended after the tail is accepted (merge drift), a duplicated refreshed row detected, a reverted refresh detected (23 of 24), altered `source_sha256` detected, pre-F1 catalogue rejected.
- **F6:** `lint` at the reviewed head failed `isort` on 72 files this PR adds, none inherited; imports sorted with isort 5.13.2 under the repo's own config, and an AST comparison proves body and imported names are unchanged. The 24 bundle receipts are resealed over those bytes.

## Review round 2 (#3123)

- **F7:** CI's `i18n` shard was red on this branch, first at `assert len(hindi) == 48`, then — once that pin was moved in a scratch run - at the same test's duplicate rule for `hi` `wire.settings`. The pin was the symptom, not the cause: the 24 re-audit rows had been **appended**, and `tests/unit/i18n/test_hold_split.py` rejects a second row for the same `(locale, namespace)` at the same English source hash (`an earlier row is at the current hash: a duplicate, not history`). Moving the pin to 52 would have made the suite green while breaking the invariant it exists to protect, so the rows are folded into the rows they supersede instead — see Ledger handling. The test file is unchanged; `pytest tests/unit/i18n/test_hold_split.py` → 14 passed at this head. That also restores every locale's cardinality pin to its main value (`hi` 48, `es` 48, `fr` 82, `ru` 48, `vi` 48, `zh-CN` 93), measured here, not assumed.
- **F8:** `ru/wire.settings/le28r2-remed/receipt.json` now pins `9156360167e72197…`; the pre-F1 `5185479b62b9…` is kept beside it as `target_sha256_original_remediation_4d38a1d4b8e`. Nothing sealed that receipt (verified: its sha256 appears in no tracked file), so the edit invalidated no digest; the receipt's own `hashes` block still seals the four bundle files.
- **F9:** `verify_evidence.py` no longer infers the re-audited rows from their position at the tail of the ledger (`new[len(old):]`); it identifies each one by content and checks it explicitly.
