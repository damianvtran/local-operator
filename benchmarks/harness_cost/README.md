# Harness-cost analysis kit — EXPLORATORY v1

**This is a reproducibility and methods artifact, not a promotional benchmark or
an estimate of current harness performance.** The historical v1 study is
materially confounded. The author develops Local Operator (conflict of interest).
Read [CORRECTIONS.md](CORRECTIONS.md) and the
[independent rigor review](reviews/REVIEW-RIGOR-R1.md) before using any number.

## Zero-spend regeneration

Requires Python **3.12–3.14**, [uv](https://docs.astral.sh/uv/), and about 250 MB
for a small environment (shared cache reduces additional physical disk use).
No harness, API key, authenticated service, office suite, model call, or private
workspace is required. Dependency download is the only necessary network access;
with wheels cached, installation can use `uv pip install --offline`.

From this directory, whether in a public checkout or an extracted ZIP:

```sh
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python publication.py verify
.venv/bin/python regenerate.py
```

**The last line is the ONE analysis command.** It reads only
`v1-snapshot/requests.jsonl`, `runs.jsonl` and the task identity/tier manifest to
compute results. The four historical JSON outputs are subsequently read **only
as validation references**, never as the producing algorithm's inputs.
It fails on population, duplicate, missing-value, per-call/per-run reconciliation,
or numerical-parity discrepancies; checks are active under `python -O` too.
Use `--out /chosen/output` or `--snapshot /chosen/snapshot` if needed.

`generated/` contains:

- `results.json` / `.csv`: 180 rep-0 rows, raw model costs, scores, cold estimates,
  classifier status, and separately labelled Codex historical sensitivity;
- `summary_tables.json` / `.csv`: clean/full, short/long/all, and matched-four
  slices (no unmatched Codex all-task population);
- `figure_data.json` / `.csv`: historical figure values retained for parity;
- `cost_components_long.json` / `.csv`: all cost components, including uncached
  input and side calls, on 8 long tasks (Codex: 4);
- `call_sequences.json` / `.csv`: per-main-call context and all-call cost series;
- `reconciliation.json` / `.csv`: checks for all 186 runs, including 6 smoke runs;
- `fig1-cost-per-task.png`: observed and both counterfactual cold estimates,
  clean and full arms, separate Codex panel;
- `fig2-why-costs-differ.png`: complete observed cost-component stacks;
- `figure_geometry.json`: rendered text-versus-canvas bounds, clipping gate.

All machine-readable money is **USD**, not cents. All token counts are tokens,
not bytes. Float comparisons use absolute tolerance $1e-10 / relative 1e-11;
no intermediate monetary rounding. CSV nested fields are JSON-encoded cells.
The historical compatibility fields `figure_data.codex.long4`, `.cpp` and
`.ratio4` are **sensitivity estimates**, not observations. The explicit
`long4_measured` field is the observed cost. Neither the old ratio nor the
adjusted cost is used in the new figures.

## Frozen population and caveats

There are 24 selected OSWorld tasks (16 short, 8 long), 180 rep-0 run records,
6 smoke run records, 1,575 metering records, and 1,533 priced responses. Clean
has 16 short tasks × 5 harnesses; full has 24 × 4 plus Codex's first 4 long
tasks. No run is silently dropped. Score ≥ 1 counts as a pass. Infrastructure
failures are flagged, not retroactively reclassified or rerun. The upstream
selection universe has **368** tasks (multi_apps: **101**), correcting the
historical review's 369/102 count; this kit does not establish an eligible v2
sample or promise 60 feasible tasks.

- One repeat per cell: no run-to-run uncertainty estimate, no pass-rate ranking.
- Cache warmth depends on order/timing and a disk pause; observed is not a
  controlled steady-state treatment. Counterfactual cold is not measured cold.
  The full-arm arithmetic ordering reverses versus OpenCode under cold pricing.
- Codex's fallback model metadata triggered repeated compaction, with different
  thinking behavior and proxy cache injection; n=4 and a post-hoc sensitivity
  cannot support a current-performance ratio or schema-only causal explanation.
- Unequal private instructions/skills/plugins, effort, output budget and side
  calls; task selection, checker feasibility, instruction ambiguity and crashes
  confound both quality and cost. Historical pass counts are not quality parity.
- OpenRouter `usage.cost` confirms a rate-card calculation from the **same usage
  block**, not an independent invoice or independent token measurement.
- v1 did **not** enable the Radient classifier. Its explicitly absent component
  is recorded separately as zero marginal cost by configuration, not falsely
  described as metered. Total equals model cost for v1. No unobserved v2
  classifier cost is inferred, and v1/v2 observations are not pooled.
- Logged scores are reproducible as data; full checker/artifact re-execution is
  **not** possible from this public kit. Private result artifacts and office
  fixtures are intentionally excluded. See `reviews/README.md`.

Any future v2 preregistration/selection is a **draft/pending evidence** process,
not validated by reproducing these v1 numbers. A new public loadout is included
for transparent methods, not as a byte-matched v1 replacement or a measured v2
result. No v2 harness runs are performed by this kit.

## Public loadout (optional; not used for analysis)

`public_loadout/` carries 493 OSS MCP tool schemas, neutral instructions, pinned
skill hashes, fetch/capture/replay scripts and a portable base-home builder.
See [its README](public_loadout/README.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The default regeneration
command does not fetch skills, launch a schema server, run capture, or make any
requests to a model. Package capture and skill fetch require network access but
no paid model calls. No skill sources or proprietary document skills are
redistributed. Their licenses apply when fetched.

## Privacy, provenance and packaging

`PROVENANCE.json` records original frozen file hashes and exact minimization
rules. Metering has usage and size metadata only; no raw request bodies,
internal MCP schemas, personal instructions, auth databases, tokens, or home
paths. Task specs are projected to ID/domain/tier. Original promotional figures
and REPORT.md are deliberately **not** shipped. Historical output JSON remains
under `v1-snapshot/` solely for numerical parity; it is not an endorsed report.

```sh
python3 publication.py scan       # verify checksums + bounded secret/PII scan
python3 publication.py zip --output /chosen/path/harness-cost-kit-publication-r2.zip
```

In Git, the ZIP builder refuses inventory members not yet tracked. The
coordinator must stage/review this new folder first; it never stages files,
commits, or opens PRs. Outside Git, the verified manifest is the package
allowlist. ZIP members have normalized timestamps, contain only this kit,
and never include `.git`, environments, caches, fetched skills or extra files.
An existing ZIP is not overwritten. The current checked archive is
`harness-cost-kit-publication-r2.zip`; any older `harness-cost-kit.zip` is
obsolete and must not be distributed (it predates the invocation-metadata
privacy and CSV-LF fixes). Maintainers who intentionally change files
run `python3 publication.py manifest` after scanning/review and regenerate it
again after updating evidence. The manifest excludes its own two control files
to avoid circular hashes; SHA256SUMS includes MANIFEST.json. Regeneration on a
different platform/font stack may change rendered PNGs and their geometry
metadata without changing numerical results. Verify the distributed manifest
**before** regeneration; to publish your newly rendered kit, review it and run
`python3 publication.py manifest` again before `zip`.

Run `TMPDIR=/chosen/scratch .venv/bin/python test_kit.py` for the bounded
regression/negative controls (12 tests, no additional dependencies).

See [REPRODUCTION.md](REPRODUCTION.md) for the kit-only extraction test,
negative controls, actual reconciliations and rendered-frame checks. Numerical
parity is the acceptance criterion; **pixel-identical plots across platforms or
fonts are not promised**. This folder inherits the repository MIT license;
third-party schema notices and full upstream license texts are included.
