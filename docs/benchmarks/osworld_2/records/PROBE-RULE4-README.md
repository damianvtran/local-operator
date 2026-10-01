# probe_rule4.py — prefix-cache append/resend probe (arm 1830)

Referenced by `~/local-operator/docs/benchmarks/osworld_2/README.md`
(§8 "Measured scoring-side defects", cache-re-billing finding).

Preserved here 2026-10-01 as the arm's owner: the script previously lived only
in a session scratchpad, so the README cited a source that would not travel with
the records. Copied unmodified.

## What it measures

Nine cases (R1–R9) that isolate whether a provider re-extends the prompt cache or
re-bills the whole prefix, by varying the outgoing message list between calls:
resend-identical, append-one, append-many, small/large prefixes.

## The rows the README quotes

| case | prompt | cached | write |
|---|---|---|---|
| R8 resend big | 18,572 | **18,566** | 0 |
| R9 append [..,x4] | 18,575 | **0** | 18,569 |

Every case was served by upstream **Alibaba** (each row carries
`"provider": "Alibaba"`), so the observation is provider-specific and the README
says so.

## Caveat recorded with the numbers

The rows are reproduced from the probe's own output as captured in the arm's
transcript; they were **not re-measured** after the fact (a live paid probe, run
under a disk/memory hold). Treat them as recorded observations, not as a
re-runnable result unless the probe is executed again.

## Related records in this directory

- `COST-FORENSICS-1830-task_003.md` — the $14.32 / 53.6%-cached outlier analysis
- `EVAL-JUDGE-FINDING-1830-task_003.md` — the silent judged-zero finding
- `ZERO-CLASSIFICATION-1830-009-013.md` — apparatus-vs-capability zero classification
