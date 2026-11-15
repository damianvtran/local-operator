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

Whole-tree guards (round 2): two unit tests scan every test file's text, so any
test-body edit can flip them although no changed file names them
(`test_ask_tool.py::test_no_test_still_pins_the_retired_refusal_sentence`,
`test_notification_isolation.py::test_no_test_nests_a_fixture_scoped_patch_under_the_opt_in`).
The planner always appends those two nodes (`GLOBAL_READERS`) to the targets, so
the run is "the changed files PLUS two named guard nodes", not "only the changed
files". A catalog entry that no longer resolves, or a PR that edits a reader's
own file, is FULL. Limits: the catalog is not a proof that no other whole-tree
reader exists (the `main`/nightly full run is the backstop), and `skipif`
condition strings are evaluated at runtime, not by the module-level check.
The six serial controls do not run on an `affected` lane, and an empty or
malformed target list fails the extract step rather than letting the test
runner collect everything. In a 12-PR sample about 4/12 were eligible, so this
is not a claim about the fleet's savings.

Wiring: the `Plan the affected lane` step in `changes`; shard 0 runs the targets
through the normal pytest step on both Pythons, shards 1-5 are idle,
`coverage-report` skips, and `Merge gate (full suite)` (name unchanged) writes
the lane and targets to its summary and fails an `affected` pass with no
successful `test` job.
