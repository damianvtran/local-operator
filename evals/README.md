# Evaluation catalog

Evaluations preserve their cases, labels, protocols and baseline results together.
They are reusable inputs and comparison records, not local command output to
remove during a scratch-file cleanup.

## Team Bridge moderation

- [End-to-end moderation evaluation](team_bridge_moderation/README.md): what is
  real, what is substituted, leak detection and reproduction commands.
- [Baseline](team_bridge_moderation/BASELINE.md): historical results and limits.
- [Screen-only evaluation](team_bridge_moderation/screen/README.md): protocols,
  sealed corpora, candidate templates and dated results.

Entrypoints live in [`scripts/`](../scripts/):

- [`eval_bridge_moderation.py`](../scripts/eval_bridge_moderation.py) exercises
  moderation against a fixture world. Its `--dry-run` validates the corpus and
  detector controls without a provider call; see the evaluation README for the
  full command and the distinction between validation and model evidence.
- [`eval_bridge_screen.py`](../scripts/eval_bridge_screen.py) evaluates inbound
  screening. Live evaluations require the credentials and spend controls
  documented by the kit.
- [`eval_research_disposition.py`](../scripts/eval_research_disposition.py) checks
  research-answer disposition; consult its command help before running it.

Use an explicit scratch output location for new exploratory runs. Preserve sealed
baseline results and provenance rather than overwriting them with a local run.
For performance and cost measurements, see the
[benchmark catalog](../benchmarks/README.md); for the wider documentation map,
see [`docs/README.md`](../docs/README.md).
