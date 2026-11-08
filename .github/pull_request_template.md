# PR Description

## Summary of Changes

Provide a brief overview of the changes introduced in this PR.

- What does this change address (e.g., bug fix, new feature, refactoring)?
- What are the key improvements or modifications?

## Related Issue(s)

Link any related issue or feature request (e.g., closes #123):

- Related: #...

## Impact

Describe the impact of these changes on the codebase:

- Does this change introduce any breaking changes?
- Are there any dependency updates?
- Are there any performance or security implications?

## Testing Details

- How were these changes tested?
- What specific scenarios were validated?
- Mention any automated tests that were added or updated.

## Targeted test evidence

PR CI runs only the fast guard tests; this block is the test gate. Re-run on the final head.
If `make test-affected` prints `WIDE:`, dispatch `gh workflow run full-suite.yml --ref <branch>` and cite it.
The whole suite is not run per review: it runs once at the merge gate, so add `ready-to-merge`
only once the head is final and every review round is clean, and re-arm it with `make merge-gate PR=<n>`
after any push. A red `Merge gate (full suite)` blocks the merge.

- Scope: `<base-sha>..<head-sha>`
- Command(s): `make test-affected` (or `scripts/affected_tests.py --since <base> --run --evidence`) plus additions by judgement
- Result: `<N passed, M failed>` in `<wall time>`; other interpreter (3.12/3.13) if version/OS-sensitive
- WIDE dispatch run (if any): <url>
- Not run, and why:

## Checklist

- [ ] My code follows the style guidelines of this project.
- [ ] I have performed a self-review of my code.
- [ ] I have added tests that prove my changes work as intended.
- [ ] Targeted tests for this diff pass on the final head, recorded above (never the full suite).
- [ ] Merge gate: `ready-to-merge` added only on the final head with all rounds clean, and
      `Merge gate (full suite)` green (re-arm after any push with `make merge-gate PR=<n>`).
- [ ] I have run formatting (black/isort), linting (flake8), and type checks (pyright), and they pass.
- [ ] I have updated the documentation when required.
- [ ] Security considerations are addressed (especially for code execution features).
- [ ] I have linked all related issues.

## Screenshots (Optional)

If applicable, please attach screenshots that illustrate the changes or new features.
