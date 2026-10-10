---
name: data-analysis
description: "Analysing data rigorously: profile before trust (keys, missingness), reproducible queries, provenance per number, units and sanity checks. Use when exploring datasets, computing metrics, or answering data questions."
---

# Data analysis

Numbers that survive scrutiny: profile first, keep queries reproducible, carry provenance, sanity-check the result.

## Profile before trust

| Check | Why it matters |
|---|---|
| Shape: rows, columns, time range | Know what you have before asking questions of it |
| Types and units per column | A text date, or cents read as dollars, corrupts every downstream number |
| Keys and duplicates | Duplicate keys silently change every aggregate |
| Missingness per column | Null-heavy columns change what a rate means |
| Distributions and ranges | Outliers and impossible values (negative ages, future dates) surface here |
| Timezone and calendar of timestamps | Day boundaries move with zones |

Keep the profile as a small set of queries you can rerun, not ad-hoc typing.

## Reproducible queries

- Keep the exact query behind every number in the report. A number nobody can re-derive is a rumour.
- Deterministic: pin the snapshot or time window, make ordering explicit before any limit, and avoid "latest row" ambiguity.
- Work in steps; save intermediates. After changing an input, rerun the step rather than mentally adjusting its output.

## Guard silent joins and filters

- Count rows before and after every join. An inner join on an incomplete key drops rows without an error.
- Check filter semantics: inclusive or exclusive bounds, null membership, timezone edges. Apply each filter once.
- State the population: "of the 4,210 rows matching X" beats "of the data".

## Provenance and checks

- Every reported number carries its source (table, snapshot), filter, and derivation. Derive it, or say where it came from.
- Reconcile: totals equal the sum of their parts; an independent cut of the same data reproduces the number.
- Units on everything: counts, rates, currency. Percentages name their denominator. Magnitudes are checked against expectation; a 1000x jump is a units or join error until proven otherwise.
- Small n: below your stated threshold, report the count, not a rate. "60% (n=5)" misleads; "3 of 5" does not.

## Output

- Separate raw from summary: show the summary and keep the raw one step away so the reader can drill down.
- State assumptions in the report: time window, definitions, exclusions, null handling.

## Red flags

- Answering from the first query without looking at the data's shape.
- A join with no row count before and after.
- A percentage with no denominator; a total with no source.
- Dropping rows to "clean" data without recording the rule.
- A distribution treated as stable when it trends over time.
