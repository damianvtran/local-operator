# Historical independent rigor review and reproduction scripts

`REVIEW-RIGOR-R1.md` is the independent data-science review of **exploratory
v1**, not a review of this new kit's implementation. Original reviewer/model
attribution is retained. Publication edits are declared in its opening note and
source hashes are in `../PROVENANCE.json`. Findings remain material; reproducing
arithmetic does not fix the experiment. See `../CORRECTIONS.md` for the precise
Codex sensitivity, upstream population-count correction and score split.

From the kit root, after the normal minimal analysis installation:

```sh
.venv/bin/python reviews/recompute.py
.venv/bin/python reviews/cold.py
# Optional historical hypothesis tests (additional scipy dependency):
uv pip install --python .venv/bin/python -r reviews/requirements.txt
.venv/bin/python reviews/stats.py
```

`recompute.py` prices independently grouped calls and sums usage.cost, calling
the documented rate-card function (not independent token measurement).
`cold.py` independently estimates cross-run reads per model using timestamp
order. The scripts write under `generated/reviews/`; `stats.py` reads that
cold result. `stats.out` and `feasibility.out` are scrubbed historical review
transcripts. No model or paid API calls occur in these scripts.

`feasibility.py` and `rescore.py` retain the historical reproduction method but
**cannot reproduce checker outcomes with this kit alone**. They refuse to run
without explicit `BENCH_REVIEW_FIXTURES` pointing to an external rig containing
`evaluate.py`, full task/evaluator specs, office fixtures, metric dependencies,
and retained run artifacts. Those are not shipped because this lane's public
boundary is usage/score metadata, not private bodies or result files. Their
original historical review reported 180/180 rescore agreement; that assertion
is a review result, **not a checker validation rerun performed by this kit**.

Optional statistics resample tasks, not run repetitions. Their intervals do
not estimate run-to-run variance and cannot establish quality parity. The
reproduction entry point is `../regenerate.py`; the old sensitivity ratio in
historical data and transcripts must not be quoted as current performance.
