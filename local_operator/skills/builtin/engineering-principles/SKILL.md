---
name: engineering-principles
description: "Practical engineering judgement: smallest change, subtract before adding, data structures before logic, idempotent operations, migrate callers then delete. Use when designing, building, or reviewing changes."
---

# Engineering principles

Judgement for building changes. One rule per line; the third column is what breaking it looks like.

| Principle | In practice | Violated by |
|---|---|---|
| Laziness protocol | Do not build what nothing needs yet. | Speculative abstraction, config for a case nobody has |
| Smallest change | The best change leaves the least to review, test, and revert. | Drive-by refactors mixed into a fix |
| Subtract before adding | Delete a concept before you add one; every new layer is paid for forever. | A new wrapper around an old wrapper |
| Data structures before logic | Get the shape of the data right and the code collapses into simple reads. | Nested conditionals over a badly shaped object |
| Boundary discipline | Validate at the edges (user input, network, files); trust the interior. | Re-validating everywhere, or validating nowhere |
| Type system discipline | Make illegal states unrepresentable where the language allows it. | Runtime flags that contradict each other |
| Idempotency | Running it twice is as safe as running it once. | Operations that double-charge, double-post, or duplicate |
| Migrate then delete | Expand, migrate callers, contract; the old path goes in the same change. | "Deprecated" code left for a ticket that never comes |
| Foundations before features | Verify the facts the design rests on before building on them. | A feature built on an unmeasured assumption |
| Encode lessons in structure | After the same bug twice, change the code or tooling so it cannot recur. | A comment saying "be careful" |
| Highest enforceable level | Fix at the strongest level available: lint, then test, then prose. | A wiki page about the rule being broken |
| Experience first | The user-visible result decides; internal elegance is a cost, not a product. | Refactors with no observable change |
| Delete dead weight | Unused, unreachable, and commented-out code is a claim nobody checks. | "We might need it"; version control remembers |

## How to apply

- Name the rule you are applying and to what. Tension between rules is normal; say which one wins and why.
- These are judgement calls, not laws. Breaking one on purpose is fine; breaking one silently is not.
