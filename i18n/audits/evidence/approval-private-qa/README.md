# Private settings QA — preserved evidence index

## Scope and provenance

This directory preserves QA session `ff0e39d33743` results at audited source
`fad3ab5703fc5bb159f48cb43755c8d9e1323b6c`. Preparation session `4930e1cd3312`
used **openai/gpt-6.1-sol**, a quota fallback from anthropic/claude-opus-5-5.
These are attributed prior executions, **not newly executed tests or formal PR QA**.
The explicit preparation request authorizes this evidence preservation under
`i18n/audits/evidence/`; it is an exception to AGENTS.md's general rule to keep
PR proof artifacts off product branches. No executable harness was introduced;
actual drivers are copied byte-for-byte as inert `.txt` files for reproduction.
Absolute paths in immutable raw receipts describe the original environment.

## Read in this order

1. [Original QA report](private-settings-qa-report.md) and
   [digest](private-qa-digest.json): historical P1 **BLOCKED** remains immutable.
2. [P1 follow-up](p1-followup-report.md) and [digest](p1-followup-digest.json):
   valid nine-case lexical comparisons resolve that specific baseline-fixture block.
3. [Source manifest](source-manifest.json): exact original path, SHA-256 and size
   for each copied file; driver filename mapping is explicit (`.py.txt` / `.sh.txt`).
4. [Retention proof](retention-proof.json): independently rechecked all 1,669
   catalogue/context candidate hashes, six consumer blobs, nine retained after-SVG
   hashes and seven 895-key raw catalogue hashes; English source hash and original
   diff classification included. All existing tracked files were unchanged.
5. [Current capture records](after-capture-result.json),
   [actual API requests/results](private-api-result.json),
   [counterfactual/bidi controls](private-controls-result.json),
   [changed values](catalogue-changes.json),
   [fixture manifest](p1-fixture-manifest.json),
   [corrected baseline records](p1-before-result.json).

## Executed commands and actual results — attributed to QA

Original commands use the launcher's allowlisted `env -i`, isolated HOME/config,
synthetic FakeSession and in-process FastAPI ASGI capability fixture. The real App
loads `local_operator.tcss`. No authentication coverage is claimed.

```sh
bash "$R/run-private-case.sh" after-complete private-settings-capture.py after
bash "$R/run-private-case.sh" api-complete private-settings-api.py
bash "$R/run-private-case.sh" controls private-controls-diagnostic.py
bash "$R/run-private-case.sh" p1-fixed-before p1-before-only.py
```

`R` is the source directory in source-manifest.json. For reconstruction, restore
inert driver names under an isolated owned scratch directory, provide the source
catalogues/old-data overlay as described in p1-before-only.py.txt and the fixture
manifest, and preserve all launch prerequisites. The data fixture itself is not
copied again. `candidate-import-prereq.log`, `after-complete.log`, `after-final.log`,
`api-complete.log`, `controls.log`, `p1-fixed-before.log` and process logs are actual
outputs, not synthesized command transcripts. Original report/logs retain their
historical scope and paths.

- 21 populated/current setting flows plus six compact flows: 27 stable
  first/settled pairs, all config-byte assertions unchanged.
- Eight GETs: HTTP 200, 278 settings each, typed machine values/defaults unchanged.
- Five PATCHes: HTTP 200 and persisted/read back; two invalid PATCHes: HTTP 422
  (`demander` as a machine approval value; boolean True for a string enum),
  rejected writes leave config bytes unchanged.
- 362 changed live registry slots; three resolver-only slots explicitly listed
  in the original digest. Old-copy predicate rejects all 365 changed slots;
  `LOP_I18N=0` deliberately selects English and rejects 34 French native slots.
- Corrected baseline: all 1,496 namespaces accepted; nine no-write cases.
  Eight baseline first/settled pairs equal; French shell startup status changes
  `connexion…` to `test/model` without geometry change. Current pairs are stable.

## Valid before/current image pairs

All nine images were personally viewed by QA, not newly captured by preparation.
They show current App/CSS with old native catalogue data versus retained current
data, **not a historical shipped application binary**.

| Case | Surface | Before / current |
|---|---|---|
| 1 | French shell choices, 120×36 | [PNG](p1-valid-comparison-1.png) |
| 2 | Chinese shell choices, 120×36 | [PNG](p1-valid-comparison-2.png) |
| 3 | Hindi shell choices, 120×36 | [PNG](p1-valid-comparison-3.png) |
| 4 | Urdu shell choices, 120×36 | [PNG](p1-valid-comparison-4.png) |
| 5 | Spanish shell choices, 120×36 | [PNG](p1-valid-comparison-5.png) |
| 6 | Russian shell choices, 120×36 | [PNG](p1-valid-comparison-6.png) |
| 7 | Vietnamese shell choices, 120×36 | [PNG](p1-valid-comparison-7.png) |
| 8 | Hindi OpenRouter true description, 120×36 | [PNG](p1-valid-comparison-8.png) |
| 9 | French role label, 100×30 | [PNG](p1-valid-comparison-9.png) |

First/settled SVGs and geometry sidecars are adjacent. At 120×36: list80,
detail114; at 100×30: list60, detail94. French label31 cells is visibly
ellipsized with complete help; Hindi description46 cells yields row57≤60.
These are bounded compositor/geometry observations, not native Urdu glyph proof.

## Deliberate limits and residuals

- Only `wire.settings` across seven locales (895 keys each), not the whole UI.
- Raw summary/default localization depends on open PR #4428. Existing English
  read-only footer and role/classification badges remain separate programme work.
- Urdu checks prove limited technical-token preservation and balanced isolation
  controls only; not native shaping or all-code-island correctness.
- No new asynchronous component or loading/empty/error-frame matrix; existing
  populated/compact interaction captures and API 422 paths have the stated scope.
- Original invalid `comparison-1..8.png`, their old baseline SVGs and
  `other-three-locales.png` remain in source QA scratch only. They are historical,
  non-load-bearing references in the immutable initial report; use ONLY the nine
  `p1-valid-comparison-*` images above as before/current proof.
- Root's generator result (200 artifacts byte-identical) is retained, not rerun.
  Strict actual-base validation is **BLOCKED**, not waived by fork-scope PASS.
- No translations, schema, code, dependencies, baseline/status override, version
  bump, native dispatch, full CI run, convergence, merge or release in preparation.
