# Reproduction tools

Reusable capture and diagnostic sources live here, grouped by the surface they
exercise. Historical screenshots, geometry dumps and execution reports do not.
The original run context remains available through immutable links in the topic
READMEs and Git history. A `docs/evidence/<topic>/...` path cited in a source
comment resolves at commit `60a908c247c4c5067488356da9c935a505759b87`
([browse](https://github.com/damianvtran/local-operator/tree/60a908c247c4c5067488356da9c935a505759b87/docs/evidence));
older sweeps' paths are in the commit table under AGENTS.md "Evidence goes on
the PR"; see [docs/README.md](../../docs/README.md).

- [Job IDs](job-id-coercion/README.md)
- [Cold session facade](l3c-lazy-cold-facade/README.md)
- [Ledger row layout](ledger-row-left-inset/README.md)
- [Media providers](media-wave2-providers/README.md)
- [Model receipts](model-receipt/README.md)
- [Output attachments](output-attachments/README.md)
- [Search bar](search-bar-rung/README.md)
- [Build retirement](serve-build-retire/README.md)
- [Session load](session-load-central-cache/README.md)
- [Subagent activity](subagent-activity/README.md)
- [Team Bridge](team-bridge-h1/README.md)
- [Usage grant followups](usage-dead-grant-followups/README.md)
- [Usage freshness source](usage-freshness/usage_shot.py)

These are developer tools, not a batch test suite. Some require external
services, a prepared session or native rendering. Read their arguments and
original invocation notes before running them; retain normal HOME/config
isolation and process cleanup. Generated output belongs in the session
scratchpad and the PR/project evidence channel, not in this directory.

The type-check discovery boundary remains the one these historical tools had
under `docs/`: `scripts/repro` is excluded in `[tool.pyright]`. Relocation does
not certify compatibility with today's APIs or turn opt-in live-provider/native
rigs into CI fixtures. Maintained scripts outside this directory, tests and
shipping source remain type-checked; `tests/unit/test_pyright_scope.py` exercises
that boundary with the real analyzer. New maintained tooling belongs outside
this historical directory.

The Team Bridge replay JSON is an executable rig input and stays adjacent to
its consumer. The [hub evaluation runbook](../../evals/team_bridge_moderation/HUB_ARM.md)
lives with the evaluation kit. Shared product contract fixtures live in
[`tests/fixtures/`](../../tests/fixtures/).
