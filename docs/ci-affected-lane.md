# CI affected lane (#4074)

The PR merge gate (`full-suite.yml`) may run only the edited test files instead
of the 6-shard x 2-Python unit suite, for ONE narrow shape. `main`, the nightly
and every dispatch always run the full suite.

Eligible (all must hold, else FULL): `pull_request` event; 1..10 files, each a
plain in-place `M` of a regular file at `tests/unit/**/test_*.py`; not on
ci_scope's WINDOWS sets or affected_tests' INFRA set; no `tui`/`e2e`/
`integration`/`perf`/`bench`/`bulk`/`native`/`layout`/`windows` in the path; the
module's imports unchanged (AST); module level limited to imports, literal
assignments, `re.compile`, pytest marks/params, function/module/class-scope
fixtures and baseless or `unittest.TestCase` classes; nothing else under
`tests/`, `scripts/`, `.github/` or a root `conftest.py` names the module.

Always FULL in v1: new test files, data fixtures, helper modules, renames,
deletions, mode changes, conftest, product, workflow or dependency changes, and
any planner error or missing input. The planner and the two modules it imports
are read from the tested base, so the PR that introduces them runs FULL.

Known trade-off: this isolates selection; it does not prove a test body cannot
leak global state or that nothing imports the module dynamically. The full run
on `main` and the nightly are that backstop.

Wiring: the `Plan the affected lane` step in `changes`; shard 0 runs the targets
through the normal pytest step on both Pythons, shards 1-5 are idle,
`coverage-report` skips, and `Merge gate (full suite)` (name unchanged) writes
the lane and targets to its summary and fails an `affected` pass with no
successful `test` job.
