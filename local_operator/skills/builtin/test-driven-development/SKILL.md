---
name: test-driven-development
description: "Writing code with tests first: RED-GREEN-REFACTOR, reproduce bugs as failing tests before fixing, and avoid testing anti-patterns. Use when implementing behaviour, fixing a defect, or asked to add tests."
---

# Test-driven development

Iron law: no production code without a failing test that demands it.

## The cycle

1. **RED**: write the smallest test for the next bit of behaviour. Run it. Read the failure and confirm it fails for the reason you intended (your assertion), not an import error, a typo, or a missing fixture. A test that fails for the wrong reason proves nothing.
2. **GREEN**: write the least production code that passes. Run the test again. Nothing more than the test asks for.
3. **REFACTOR**: improve names and structure while all tests stay green. Run them again.

Then the next slice. Cycles are minutes, not hours; commit-sized.

## Bugs: reproduce first

- Turn the bug into a failing test before touching the fix. Watch it fail, fix, watch it pass.
- If you cannot express the bug as a failing test, you do not understand it yet; switch to `skill://systematic-debugging`.
- Keep the repro test after the fix as the regression test. A bug fix without a test that dies when the bug returns is incomplete.

## Already wrote the code? Delete it

Code written before its tests is untrusted: nothing was ever observed failing, and its behaviour was never pinned. Delete and re-derive from the tests. Adapting pre-test code to pass tests keeps the untested behaviour hidden inside.

Exception: throwaway exploration deleted before the work starts. Spikes inform; they do not ship.

## Anti-patterns

| Anti-pattern | Why it fails | Instead |
|---|---|---|
| Mock-heavy tests that assert on the mock | Tests the mock; passes while the real component is broken | Prefer the real object; fake only at system boundaries; assert the outcome |
| Assertion-free tests ("it runs") | Passes always; proves nothing | Assert the observable outcome |
| Implementation-detail tests (private calls, call order) | Break on any refactor; resist change | Test behaviour through the public seam |
| Coverage chasing | Volume without discrimination | One test per behaviour and per edge |
| Tests written after, never seen red | Cannot distinguish a test from a tautology | Run new tests before the code; confirm red |
| Skipped or xfailed tests left behind | Silent coverage hole | Fix or delete; never skip quietly |

## Red flags

- "I'll add tests after."
- A new test that passed the first time: did it exercise the change at all?
- Testing private internals because the public seam is inconvenient; fix the seam.
- A fix verified only by "the code looks right now".
