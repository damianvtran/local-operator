# EXPLORATORY v1 — cost per task

Materially confounded; one run per cell. No current ranking or quality-parity claim.
All costs are mean **USD per task** (six decimal places). Passed / n are logged counts.
**Cold columns are counterfactual estimates, not measured cold executions.**
Classifier absent in v1; total equals observed model cost. See [corrections](../CORRECTIONS.md).

## Clean — 16 short tasks

| Harness | Passed / n | Observed USD/task | Historical cold estimate USD/task | Cross-run cold estimate USD/task |
|---|---:|---:|---:|---:|
| Local Operator | 13 / 16 | 0.023969 | 0.079421 | 0.079421 |
| omp | 13 / 16 | 0.020373 | 0.047095 | 0.047130 |
| OpenCode | 13 / 16 | 0.023885 | 0.036403 | 0.036493 |
| Claude Code | 12 / 16 | 0.059221 | 0.124177 | 0.121081 |
| Codex | 12 / 16 | 0.057542 | 0.060010 | 0.060010 |

## Full — 24 tasks (excluding Codex)

| Harness | Passed / n | Observed USD/task | Historical cold estimate USD/task | Cross-run cold estimate USD/task |
|---|---:|---:|---:|---:|
| Local Operator | 16 / 24 | 0.043441 | 0.110046 | 0.110046 |
| omp | 17 / 24 | 0.068227 | 0.158845 | 0.158845 |
| OpenCode | 17 / 24 | 0.079233 | 0.090708 | 0.090790 |
| Claude Code | 14 / 24 | 0.092382 | 0.125941 | 0.122846 |

## Codex full — 4 selected long tasks

| Harness | Passed / n | Observed USD/task | Historical cold estimate USD/task | Cross-run cold estimate USD/task |
|---|---:|---:|---:|---:|
| Codex | 3 / 4 | 5.752039 | 5.752039 | 5.752039 |

Codex full is a separate n=4 population with a metadata/compaction confound.
No historical cache-adjusted Codex cost or ratio is shown. Pass counts do not establish quality parity.
