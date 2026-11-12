# Benchmark catalog

Keep executable kits, their input snapshots and their licensing/provenance
records together. Generated files in a published, checksummed kit may be
reproduction references, not disposable local output.

## Self-contained kits

- [Harness-cost analysis](harness_cost/README.md): exploratory historical study,
  corrections, independent review and zero-spend regeneration. Read its caveats
  before citing a number. Its manifests, snapshots, licenses, checksums and
  validation outputs are part of the reproducible artifact.
- [OSWorld adapter](osworld_v2_adapter/README.md): adapter implementation and
  local validation. The [OSWorld apparatus documentation](../docs/benchmarks/osworld_2/README.md)
  explains the external environment and what has actually been measured.

## Runtime performance harnesses

The [performance and cost guide](../docs/BENCHMARKS.md) describes harness
measurements. Runnable `bench_*.py` entrypoints remain in [`scripts/`](../scripts/)
so their imports and established commands do not change just to tidy the tree.
Additional methodology and results live under
[`docs/benchmarks/`](../docs/benchmarks/), including
[session efficiency](../docs/benchmarks/session-efficiency/).

Some scripts call live providers or require external services. Read the script's
help and its associated protocol first; do not equate a successful offline check
with a live benchmark result. Put new run output in scratch storage (or the
ignored root `bench/` measurement store where a script expects it), then attach
PR-specific evidence to the PR rather than committing it.

For behavioral evaluations, cases and labels, see the
[evaluation catalog](../evals/README.md). The
[documentation map](../docs/README.md) indexes design and historical records.
