# Documentation map

Start with the [user guide](guide/overview.md) or
[contributor setup](../CONTRIBUTING.md). This index separates current guidance,
reusable evaluation tools and historical engineering records without moving
files that are consumed by tools or linked from other repositories.

## Using and building Local Operator

- [Getting started](guide/getting-started.md), [configuration](guide/configuration.md)
  and [safety](guide/safety.md).
- [Architecture](REWRITE.md), [SDK](SDK.md), [desktop API](DESKTOP_API.md)
  and [internationalization](I18N.md).
- [Agent development instructions](../AGENTS.md): isolation, validation and
  contribution gates.

## Evaluations and benchmarks

- [Evaluation catalog](../evals/README.md): moderation datasets, protocols,
  model-run entrypoints and preserved baseline results.
- [Benchmark catalog](../benchmarks/README.md): reproducible analysis kits,
  performance harnesses and external benchmark apparatus.
- [Performance and cost guide](BENCHMARKS.md): commands and interpretation.

## Design and historical records

- [Design records](design/) and [proposals](proposals/) describe decisions and
  constraints; a proposal is not proof that a feature shipped.
- [Verification records](verification/) and [verification ledger](VERIFICATION.md)
  contain historical validation. Results apply to the revisions they identify,
  not automatically to the current tree.
- [Reusable reproduction tools](../scripts/repro/README.md) contain retained
  capture/diagnostic sources and the inputs they need. Historical proof captures
  formerly under `docs/evidence/` are available in Git history, not the tree.
- [Legacy PR records](pr-evidence/) and [performance records](performance/)
  retain existing audit-linked measurements; their presence does not authorize
  committing new PR-only proof artifacts.
- [Extension store assets](store/) are publication inputs, not disposable
  screenshots.

## Where new material belongs

Keep reusable evaluation corpora and protocols under `evals/`, self-contained
benchmark kits under `benchmarks/`, and runnable development harnesses under
`scripts/`. Shared test data belongs in `tests/fixtures/`; fixture data bundled
with a self-contained evaluation kit stays with that kit.

Put new PR-only screenshots, logs, command captures and temporary reports in the
session scratchpad and attach them to the PR rather than committing them. See
[the evidence policy](../AGENTS.md#7-evidence-goes-on-the-pr-never-into-the-repository).
Do not treat a historical record as unused configuration merely because a
string search finds no caller; tools may discover configuration by filename.
