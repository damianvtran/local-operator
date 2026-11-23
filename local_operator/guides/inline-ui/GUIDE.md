---
name: inline-ui
description: Choose an inline chart, graph, pie, scatter plot, histogram, heatmap, table, or spreadsheet grid for an answer. Read for the component catalog and JSON shapes.
---

# Inline UI catalog

Use this only when the current turn advertises inline UI. Choose the view that answers the question, not the most elaborate one. Keep the main finding in normal prose so it survives in viewers without widgets. Use real data only; retain units, use a consistent scale, and label any aggregation. Don't use HTML, executable formulas, or a separate rendering call.

Place a single JSON object inside a top-level fenced block tagged `lop-ui`; put the closing backticks on their own line. Use `type` and a nonempty `title` on every component. Use no more than 32 rows or points and 8 columns per component; keep labels short. Use a CSV or workbook deliverable for a full dataset or editable spreadsheet, and put a small read-only preview inline. Do not silently truncate data.

| Choose | Best for | Fields beyond `type` and `title` |
| --- | --- | --- |
| `bar` | Rank or compare discrete categories | `data`: `[label, value]` pairs; optional `unit` |
| `series` | Compare multiple series across common categories or periods | See modes below |
| `line` | Trend over an ordered sequence with evenly spaced observations | Same as `bar`; keep records in chronological order |
| `pie` | Parts of one whole; use a bar for precise comparisons | Same as `bar`, with positive, mutually exclusive parts; use few categories |
| `scatter` | Relationship, clustering or outliers between two numeric variables | `xLabel`, `yLabel`, `data`: `[x, y]` numeric pairs |
| `histogram` | Distribution of one numeric variable | Same as `bar`, but use ordered, nonoverlapping, equal-width bin labels and nonnegative counts |
| `heatmap` | Two-dimensional patterns, such as weekday by hour or a correlation matrix | `columns`, `rowLabels`, `rows`: rectangular numeric matrix, `null` if missing; optional `unit` |
| `table` | Exact values or mixed details for comparison | `columns`: labels; `rows`: rectangular strings/numbers |
| `spreadsheet` | A read-only grid preview for tabular data | Like `table`, but `null` for blank cells; formula-like strings are literal text |

For `series`, specify `mode`, `labels` (category labels), and `series` (up to four objects, each with a `label` and `values` aligned to `labels`). Add `unit` when applicable. Choose:

- `grouped`: side-by-side comparisons in the same unit.
- `stacked`: show both total and contributions. Negative and positive contributions accumulate separately. Use only additive measures in a common unit.
- `percent`: show changing composition. Use nonnegative values with a positive total in every category; values are normalized within each category.
- `dual`: compare measurements with distinct units (some viewers show an exact, unit-labeled table). Set `unit` for the left axis and `rightUnit` for the right axis; set each series' `axis` to `left` or `right` and use both. Prefer separate charts if relative magnitudes or visual crossings could mislead.

Example:

```lop-ui
{"type":"heatmap","title":"Requests by shift","columns":["Early","Late"],"rowLabels":["Mon","Tue"],"rows":[[18,50],[12,null]],"unit":"requests"}
```

Plot numbers must be finite and within the safe numeric range (absolute value at most 9,007,199,254,740,991). Put larger exact counts or decimals needing extra precision in a table as quoted values. The schema is data-only; there is no custom CSS, arbitrary resource URL or action. If no component fits, use prose or a file deliverable instead of inventing a type.
