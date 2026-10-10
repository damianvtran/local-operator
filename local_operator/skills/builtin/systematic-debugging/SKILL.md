---
name: systematic-debugging
description: "Root-causing defects instead of guessing: reproduce, isolate, instrument component boundaries, kill hypotheses, and recognise when to question the design. Use on any bug, flaky test, or unexplained behaviour."
---

# Systematic debugging

Debugging is hypothesis testing against a running system, not code reading followed by a guess.

## Phase 1: Reproduce

- Get a reliable, minimal reproduction: exact command, input, environment.
- Confirm you can make it fail on demand. Cannot reproduce it? Say so, collect frequency and conditions before theorising.
- Capture the failure itself: full error text, stack, exit code. Not a paraphrase.

## Phase 2: Isolate

- Shrink the reproduction until the smallest failing unit is visible (scope, input size, config).
- Binary-search the surface: return early or disable at the midpoint of the suspect path. The failure is on one side of the cut; halve again.
- Instrument component boundaries. At each boundary, log what goes IN and what comes OUT. Find the FIRST boundary whose output is wrong given a correct input. The fault lives between the last correct boundary and that one.
- Let the system tell you where it breaks. You do not need to read all the code.

## Phase 3: Fix the cause, or question the design

- **Fixed-cause**: you can say why it broke and why the fix prevents exactly that. **Guessed-cause** ("this might help") is not done.
- Change one variable at a time. Stacking two fixes teaches nothing about either.
- Tripwire: three or more fixes for one defect means the model is wrong or the design is. Stop patching; question the architecture or the assumption, and write down what you now know.
- Prove the fix is connected: revert it and watch the bug return, then reapply.

## Phase 4: Verify and lock in

- Rerun the original reproduction; it must pass.
- Add the regression test (see `skill://test-driven-development`).
- Check the neighbours: the same pattern elsewhere, callers of the changed code. Evidence standards: `skill://verification-before-completion`.

## Red flags

- Editing before reproducing; "I think it's X" followed by a fix.
- Treating a flaky test as noise; flakiness is a real defect with real conditions.
- Suppressing the error (wider catch, retry loop, muted log) instead of finding it.
- Reading code for an hour without running anything; the running system has the answer.
- No one-sentence theory of the failure ("X fails because Y reaches it before Z").
