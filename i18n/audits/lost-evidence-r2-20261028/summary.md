# Lost i18n evidence — independent re-audit, 2026-10-28

**Audit id:** `le28r2`. **Disposition:** OPEN / HOLD. Do not merge or release this tree.

24 fresh bundles cover 4,628 locale-key comparisons (17 + 49 + 36 + 7 + 820 keys × the locales that carry them). Nothing from the deleted scratchpads was reconstructed. Semantic result: **13 PASS, 11 FINDINGS; 8 major, 26 minor, 0 blocker**. The five original `wire.settings` subaudit reports (77 observations, before coordinator reconciliation) are preserved byte-for-byte in the bundles; final findings and their dispositions are a separate file.

## Provenance, stated exactly

Two distinct things happened, in this order:

1. **The linguistic judgement came from direct git/python reads** in this session's sparse worktree (`aeb9bb9dc34e3988c78ebbb6921e79e049143cbf`). Those reads are **not** `read_public_copy` and are never labelled as such.
2. **Sanctioned `read_public_copy` receipts were then obtained** through `lop exec --run-in <worktree> --tools read_public_copy` at packaging base `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`, for **29 files: the 5 en sources and 24 locale catalogues**. Each receipt is the tool's own audit-log line (`~/.local-operator/audit/public-copy-reads.jsonl`); each digest was independently recomputed from the base blob and matches. The receipts confirm the *bytes*; they do not certify the judgement.

**The en context sidecars are not routable at this tool version.** `read_public_copy` validates a catalogue as a flat key→string map, so all five `.context.json` files are refused (`refused: catalogue keys and values must be strings`, verbatim in the child session transcript). They remain direct reads. Every pair therefore records `sanctioned_route_verified: "partial"` plus the refusal — not a silent waiver, and not a claim of full routing. `routed-read-index.json`, the driver log and the child-model banner file are in this directory.

The operator waived human/native-speaker certification for all supported locales. That is not a waiver of independent LLM audit. None of these auditors is a native speaker; Hindi idiom would need one for native certification, which is neither claimed nor required here.

## Per-pair semantic verdicts

| Locale | tui.transcript | tui.authoring | tui.team_authoring | tui.palette | wire.settings |
|---|---|---|---|---|---|
| es | PASS | FINDINGS (1 minor) | PASS | PASS | FINDINGS (5 minor) |
| fr | PASS | FINDINGS (1 minor) | PASS | PASS | FINDINGS (5 minor) |
| ru | PASS | FINDINGS (1 minor) | PASS | PASS | **FINDINGS — 4 major**, 3 minor |
| vi | PASS | FINDINGS (1 minor) | FINDINGS (1 minor) | PASS | **FINDINGS — 3 major**, 3 minor |
| hi | out of affected scope | FINDINGS (2 minor) | PASS | PASS | **FINDINGS — 1 major**, 3 minor |

Ledger: **21 rows `passed`** (clean, or minor-only findings which are non-gating) and **3 rows `blocked`** (`ru`, `vi`, `hi` `wire.settings`, each with unresolved majors). The brief's "no hi transcript row" means no *affected* scratchpad-citing row: the Hindi transcript catalogue and its ledger rows exist.

## Eligibility effect — measured, not inferred

`scripts/i18n/generate.py --check` at this state exits **1** and names `local_operator/i18n/data/shipped.json`; read-only `_shipped_locales()` returns **[en, fr, es, zh-CN, ur]**, against the eight locales in the committed generated file. The three blocked pairs each carry `wire.settings`, so `ru`, `vi` and `hi` un-ship until the translator remediation lands and is independently re-audited. **No generation write was performed in this PR**, and the committed generated data is deliberately left stale: publishing a reduced set is a decision for the integration PR, not for an evidence commit. That is the honest cost of not leaving falsely-passed rows in place.

## Gating findings for the translator batch

Exact current text, exact proposals and original subaudit IDs are in each bundle's `findings.json`.

- **ru `wire.settings` (4 major)**: `subagents_max_running.label`, `supplements_maxcostusd.help`, `supplements_timeouts.help` use «задача» for a background job where the ru guide fixes job → «задание» (and reserves «задача» for todos); `librarian_enabled.help` uses «сеанс» where session = «сессия», plus an awkward disabled-install calque.
- **vi `wire.settings` (3 major):** `web_fetch_render_backend.help` translates pip `extra`, which the glossary's all-locale kept-English set keeps as `extra`; `speech_transcription_cleanup.help` and `.choice.off.description` use `âm đệm` (the medial-glide term) for filler words — `từ đệm` is the correct term.
- **hi `wire.settings` (1 major):** `proactive_patience_episode_ttl_ms.help` renders a wake as `जागरण` while the hi guide and the rest of the namespace use `जागृति`, breaking the one-term rule.

Two findings a child auditor graded major were **reconciled to minor with the reason recorded**, not silently dropped: `fr local_providers.*.label` (`endpoint de {name}` fails French elision for `Ollama`/`OpenAI-compatible`; meaning survives, grammar defect) and `hi shell_environment_exclude.help` (child-process wording). The `hi` finding that proposed transliterating `endpoint`/`merge`/`pull`/`push` was **rejected** — the glossary applies its kept-English set to all locales.

## History preserved

All 401 pre-existing ledger objects are otherwise byte-identical: no field, provenance, source hash or prior verdict was rewritten. The 24 rows that cite deleted scratchpads each keep their note verbatim and gain a dated supersession sentence pointing at the new record. 24 current-source rows are appended (last-row-wins in the generator). The verifier re-checks all of this from the base commit rather than trusting this summary.

## Validation (actual output)

- Existing repo interpreter (Python 3.12.13) and Node 24.22.1; no venv created, nothing installed.
- `scripts/i18n/check.py --catalogues-only`: **exit 0**, 0 problems (all carried catalogues; the CLI has no namespace filter).
- `scripts/i18n/check.py`: **exit 0**, 7 advisory findings, **sparse-tree AST only** (34 files) — not whole-repository ratchet coverage.
- `scripts/i18n/generate.py --check`: pre-ledger **exit 0** / 100 byte-identical; post-ledger **exit 1**, stale `shipped.json` as described above.
- `verification.log` (reproducible `verify_evidence.py`): 24 bundles, **345 companion hashes, 120 input hashes, 48 routed receipts**, 401 preserved historical objects, 24 appended notes, 24 appended rows, byte-corruption negative control detected.
- All explicit sidecar `maxCells` checks pass. Separately measured, the inferred 29-cell settings label budget is exceeded by en 6 / es 57 / fr 48 / ru 39 / vi 23 / hi 0 labels — a text-width risk, not a rendered-frame claim.
- ICU key/argument/category parity passes; plural branches rendered through the repo runtime at 0/1/2/5/21/22/10⁶/1.5·10⁶. French and Russian cache-TTL/rolling-day help carry worded equivalents (`une heure`, `doublé`, `часовой`, `вдвое`, `суточный`) rather than literal digits; the limits are present.

## UNVERIFIABLE — not fabricated

Deleted audit reports and their original model histories; the old 60×24 / 100×24 OperatorApp previews and `geometry-report.json`; top/settled frame identity; native terminal/font shaping (including Devanagari); native-speaker idiom certification; any claim that the lost reports' original auditors used the models their notes name. No screenshot, geometry file or old scratchpad artefact exists in these bundles. No full test suite, browser, live app or real UI flow was exercised. The 2026-10-25 deletion date/count are the request's incident account, not a reconstructed filesystem event.
