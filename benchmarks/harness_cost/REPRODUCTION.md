# Reproduction evidence — EXPLORATORY v1

Date: 2026-11-03. Scope: this analysis/publication kit only; **no model/API
calls, harness runs, office checker runs, or paid inference**. Package/license
downloads are not inference. No source-workspace logs were written. The original
frozen source's 13 SHA256SUMS entries were independently verified before final
publication. This document is implementation evidence, not a new independent
review of the kit, nor a repair of the confounded study.

## PR 3345 unified round-1 remediation — C1, C2, D1, D2

**Core remediation is complete; global manifest/ZIP packaging is provisional
until the parallel experimental-port lane freezes its own folder.** The
coordinator will regenerate the global inventory and final archive afterwards.
The sealed standalone-copy checks below cover the captured snapshot, not a claim
that concurrently edited source files are immutable or a privacy failure.

Scope is the requested code/design delta, not a new audit of previously cleared
runtime/experimental-driver code. No commit, push, real-index staging, model/API
call or root README edit was performed by this lane. The coordinator owns the
single unified remediation commit and review-thread responses.

- **C1:** graph labels now derive Codex n from the actual panel rows: **n=16**
  for clean and **n=4** for full. A contextual-label regression test covers both.
- **C2:** metering publication uses a recursively closed schema, including
  `usage.cache_creation` and scalar leaf types. Unknown nested keys, private
  dictionaries/lists hidden in token counters, null nested metadata and boolean
  token counts fail closed. Eight synthetic nested/leaf corruptions exercise
  the actual publication scanner with benign text that does not match any
  secret pattern. Valid null top-level usage for discovery/incomplete responses
  remains allowed; the clean frozen snapshot still passes.
- **D1:** `generated/cost-per-task-table.png` and `.md` are produced by the same
  analysis command from the same derived rows, with full-precision `.json`/`.csv`
  companions. There are **10 rows in three explicitly separate populations**:
  five clean-16, four full-24, one Codex-full-4. Columns are observed USD/task,
  two labelled counterfactual estimates and logged passed/n counts. No historical
  adjusted Codex cost/ratio, current ranking or quality-parity claim is included.
  Regeneration parses the exported Markdown numeric cells and checks them against
  exported JSON at the six-place USD rounding tolerance; deliberate wrong costs,
  pass/sample counts and population headings are rejected by regression tests.
  The kit README displays graph plus table graphic with the accessible Markdown
  link immediately next to the graph. Root README integration remains external.
- **D2:** graphs use vertically stacked panels, larger type, no tiny rotated
  numeric labels, distinct hatches and dark bar/stack boundaries. Graph numbers
  are readable in the companion table instead of tiny bar-label glyphs. Observed,
  historical-cold and cross-run-cold fills have **9.11 / 4.76 / 5.02** contrast
  ratios against white. Stack boundaries have **at least 12.48:1** contrast with
  each adjacent fill. Every estimate/component has a distinct solid/pattern
  encoding; hue is not the sole distinguishing signal.

Actual PNGs were opened using the image reader, then resized to **exactly 861 px
wide** and opened again: graph 861×963, components 861×786, table 861×792.
No titles, legends, counts, numbers or footnotes clip; all remain readable.
`figure_geometry.json` records **37 / 33 / 73** text artists respectively, minimum
font sizes at 861 CSS px of **13.916 / 13.916 / 13.884 px**, no canvas overflow,
no tick/legend overlap, and no table text overlap. The producer gates every
visible text artist at **≥12 CSS px** at this embed width. Smaller screens should
use the selectable Markdown table. Numeric parity, not cross-platform pixel
identity, remains the reproducibility acceptance criterion.

The targeted suite now has **16 passing tests** (the prior 12 plus four direct
regression tests); all historical numeric fields and reconciliation totals still
pass. Seven CSV exports, including the new table, use LF only. Manifests and the
privacy scan were refreshed. The versioned archive is
**`harness-cost-kit-review-r1-remediation.zip`**; older archives were left untouched
and are superseded. Because the new table files intentionally remain unstaged,
packaging uses a fresh standalone manifest-only copy, then extraction verifies
all members, regenerated numerical outputs, the table, and all 16 tests with
network connections disabled for the analysis. The checkout builder's normal
tracked-file gate remains unchanged; the coordinator can rebuild there after
staging the unified remediation. This implementation evidence is not a claim
that the independent design re-review has already signed off.

## Publication hardening follow-up (integration round 2)

Integration found two defects before any successful push: an offline dummy
LaunchDarkly invocation value matched hosted push-protection credential shapes,
and CSV's default CRLF appeared as trailing whitespace in Git. Neither issue
was bypassed or suppressed.

- Capture now constructs the **same previously accepted offline dummy** only in
  process memory; `published_command()` records the non-credential marker
  `<offline-dummy-generated-at-runtime>` instead. Offline comparison against the
  local integration commit proves that the generated runtime invocation is
  identical and that the snapshot differs in exactly **one command metadata
  element**. All **493 tool definitions across 9 servers**, package pins,
  licenses, source URLs, schema byte counts and schema SHA256s are unchanged.
  No live recapture, MCP process, API request, real key or inference was used.
- Privacy controls now reject credential-shaped all-zero dummies too, including
  LaunchDarkly UUID/compact forms, provider-style keys and known access-token
  prefixes. Synthetic negative cases are assembled only at test runtime, never
  committed as complete credential-shaped literals. Existing public commit
  hashes are not treated as credentials solely because they contain 40 hex
  characters. There is no dummy/entropy exemption for key-shaped values.
- CSV output now specifies `DictWriter(lineterminator='\\n')` at the producer.
  All **six** regenerated CSVs contain **no CR bytes**; no after-the-fact
  whitespace stripping is used. `git diff --check -- benchmarks/harness_cost`
  passes. Numerical parity and all **12 tests** pass after the source fixes;
  the tests also exercise LF export, published-command sanitization and all
  per-server schema hashes. Test count remains 12 with stronger assertions.
- Manifests/checksums were rebuilt and the bounded privacy scan rerun. The
  versioned replacement is **`harness-cost-kit-publication-r2.zip`**. The older
  `harness-cost-kit.zip` is obsolete and **must not be distributed**; it was
  intentionally not overwritten or deleted. A clean extraction of the new
  archive passes manifest verification, all 12 tests and the actual regeneration
  entry point with sockets/DNS blocked. ZIP members are individually checked
  against their manifest hashes and contain LF-only CSVs and sanitized metadata.

The coordinator owns amending the **local-only unpushed commit** and retrying
publication; this lane did not push, amend, commit, change branches or touch the
root README. Local readiness is not a claim that a remote push was accepted.

## Isolation and installation actually exercised

1. Built the initial inventory/checksums and bounded secret/PII scan.
2. Copied **only inventory-listed kit files** into a fresh session scratch
   directory. No checkout metadata, source rig, private fixture, auth database,
   cached MCP schema, raw body, or pre-existing environment was copied.
3. From that copy, executed the README commands:

   ```sh
   uv venv --python python3 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   .venv/bin/python publication.py verify
   .venv/bin/python regenerate.py
   TMPDIR="$OWN_SCRATCH" .venv/bin/python test_kit.py
   ```

   `uv 0.10.8` selected CPython **3.14.3**, Darwin arm64. Eleven pinned analysis
   packages installed through uv's existing shared cache in 95 ms after cache
   preparation/download (~16 MB compressed for the newly needed large wheels).
   No dependency tree or virtual environment is part of the publication.

   Installed: contourpy 1.4.0, cycler 0.12.1, fonttools 4.68.0, kiwisolver 1.5.1,
   matplotlib 3.10.7, numpy 2.3.4, packaging 26.3, pillow 13.0.0, pyparsing 3.3.3,
   python-dateutil 4.0.0, tzdata 2026.6. Exact portable pins: requirements.txt.

4. A final inventory-only copy also ran the actual `regenerate.py` entry point
   with socket connect, connect_ex, create_connection and DNS resolution patched
   to raise immediately. **PASS**: all numeric outputs and both plots completed
   with networking blocked. The dedicated analysis environment was outside this
   second fresh copy. No dependency or hidden source-workspace fallback exists.
5. Final source scripts compiled with `compile()` without writing bytecode;
   public shell helpers passed `bash -n`. Public fetch/capture/replay/home-build
   helpers were **not** run: they are optional methods assets, not analysis
   prerequisites. No browser engine, native harness or MCP tool call was started.

## Numerical acceptance results

Every historical key in all four reference JSON files was checked recursively,
not just selected summary means. Frozen references are loaded **after** derivation
for validation, never as computational inputs. Monetary tolerance: absolute
$1e-10 or relative 1e-11; labels, populations and integer values also checked.

| Check | Actual result |
|---|---:|
| Metering records | 1,575 |
| Priced messages | 1,533 |
| Discovery requests excluded from pricing | 41 |
| Missing-usage message (known infrastructure failure; HTTP 200) | 1 |
| Run records, all unique IDs and cells | 186 |
| Analysis rep-0 records | 180 |
| Smoke records excluded from analysis | 6 |
| Sum model cost, all priced snapshot calls | $36.909711435 |
| Sum OpenRouter usage.cost, same calls | $36.909711435 |
| Rep-0 model cost | $32.766771485 |
| Smoke model cost | $4.142939950 |
| Maximum absolute per-run model/usage.cost residual | $2.7755575615628914e-17 |
| Per-run component sums / raw observed cost | all 186 agree |
| Historical cold-cost reconstruction | all 180 agree |
| Independent review regrouping / observed cost | all 180 agree |
| Independent timestamp-ordered cross-run cold estimate | all 180 agree |
| Historical Codex sensitivity | all 4 agree; 3 change, 2 increase |

The classifier is absent by v1 configuration and is separately labelled, not
claimed to have a zero-cost metering record. v1 total equals model cost.
OpenRouter reconciliation is a rate-card cross-check from the same usage block,
**not** independent proof of token counts or an invoice. Earlier aborted logs and
the broader spend counter are intentionally outside this snapshot reproduction.

CSV row counts: results **180**, summary_tables **22**, figure_data **5**,
cost_components_long **5**, call_sequences **1,635**, reconciliation **186**.
Nested structures are JSON-encoded cells. All new CSVs carry the exploratory
classification and corrections reference. Full score matrix: **14 all-pass,
6 all-fail, 4 mixed** tasks among the four complete full-arm harnesses.

## Negative controls actually exercised

`test_kit.py`: **12/12 pass**, both normal and `python -O` execution. This proves
checks remain active with Python assertions optimized away.

- All frozen numerical fields, population sizes and known absent-classifier total.
- Duplicate run ID/cell rejected.
- Duplicate request rejected.
- Missing score rejected.
- Missing priced usage rejected.
- OpenRouter cost discrepancy rejected.
- Inconsistent cache TTL subtotals rejected.
- Exact signed sensitivity protected against clamping/one-run special casing.
- Synthetic Haiku >100k / 1h branches tested separately; no false claim that v1
  observations exercise those branches.
- Unexpected private request-body field rejected at the projection boundary.
- Checksum mutation rejected.
- Synthetic secret marker and symlink rejected (dummy marker built at runtime,
  never a real credential).

No original dataset is mutated by these controls; disposable copies only.

## Independent historical review scripts

From the isolated kit, `reviews/recompute.py` and `reviews/cold.py` ran without
additional dependencies. Their regrouped costs and alternative cold estimates
agree with all 180 main results. Optional `reviews/requirements.txt` was then
installed via uv shared cache (only scipy 1.16.3 added; ~20 MB compressed), and
`reviews/stats.py` executed successfully. Its task-bootstrap outputs reproduce
the historical warm/cold reversal and McNemar/Wilson caveats in the review; these
are **task-sampling intervals only**, not run-to-run uncertainty or causal claims.

Historical `feasibility.py` / `rescore.py` are included with explicit external
fixture guards. **Their original 180/180 checker-rescore claim was not rerun in
this public kit**: full private artifacts, evaluator specs and office fixtures
are not shipped. Likewise no public-loadout/v2 cost or quality claim was tested.

## Rendered figures inspected

The actual pinned-environment PNGs were opened with the image-reading tool,
not merely checked for file existence. Both show **EXPLORATORY v1**, corrections,
units, n labels and separate Codex scale. Observed and two counterfactual cold
estimates are distinct; historical Codex sensitivity is not plotted. All uncached
input costs are included in figure 2 (some are subpixel-small at this scale).

An initial figure 1 tick-label/legend crowding was fixed by reserving a larger
bottom margin before final inspection. Final text-bound gates inspected **81**
text artists in figure 1 (1500×700 measurement canvas) and **38** in figure 2
(1400×600); all are within the canvas, with no tick-label/legend intersections.
PNG export is 150 dpi with tight padding. Geometry is saved in
`generated/figure_geometry.json`. Final rendered frames show no clipped labels,
legends, titles or footnotes. No pixel-identical cross-font/platform promise is
made; numeric parity, explicit scope and no clipping are the acceptance criteria.

## Privacy, source inventory and ZIP checks

The publication scan combines explicit metering/task field allowlists, blocked
file classes, symlink refusal, credential/private-key/home-path patterns and email
screening. Only two exact generic upstream schema email examples are allowed;
full license attributions are public upstream notices. The scan never prints a
suspected credential. Projection/source hashes are in PROVENANCE.json; SHA256SUMS
and MANIFEST.json bind the actual shareable files. No raw bodies, internal MCP
schemas, auth stores, personal instructions/home paths, tokens, fetched skills,
virtual environments, .git or source-workspace caches are included.

Standalone ZIP creation succeeded, and extraction into another empty scratch
directory passed checksums and the documented regeneration command. Archive
membership was checked for the forbidden classes above. The builder uses only
manifest members; normalized member timestamps and no user metadata. A Git
checkout additionally refuses inventory files not yet tracked. Both paths were
exercised: the ordinary checkout refused the new untracked folder and created
no archive; the positive Git path used a **disposable GIT_INDEX_FILE and
GIT_OBJECT_DIRECTORY**, staging only kit members into scratch, never the real
index/object store. That ZIP contained **66 members** (64 inventoried payload
files plus manifest/checksums), extracted successfully, passed checksums, and
regenerated all outputs. A final archive is about **0.63 MB**, not an environment
bundle. The normal checkout remains unstaged for coordinator integration.
No commit, branch switch, PR, runtime-code edit or root README edit was
performed by this lane.

The ZIP/checksum process is deliberately strict about PNG bytes: if a different
renderer changes them, numerical regeneration can pass while the original
publication checksum no longer does. Verify the original kit **before** rendering;
review and refresh `publication.py manifest` before packaging your new rendering.
This is not treated as a numeric failure or hidden by disabling checksum checks.

Disk checks before installs were above the **15 GiB** floor; later free-space
observation was ~16.4 GiB amid other concurrent lanes. No heavy build, duplicate
runtime dependency tree or task-image download occurred. Nothing belonging to
other sessions was removed. Final integration/review/staging and any future
experimental v2 driver remain the main coordinator's responsibility.
