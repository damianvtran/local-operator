# OSWorld 2.0 adapter for local-operator

A separately-distributed evaluation adapter that runs OSWorld 2.0
(`osworld-v2-2026.08.08`) episodes behind local-operator's verified adapter
boundary. **PR 1 is the cloud-free slice**: the complete adapter plus a
`FakeProvider`, proved end to end through the real `EpisodeRunner` with zero
AWS spend. The AWS provider (`providers/aws.py`) is PR 2.

## Why it is its own distribution

The adapter imports `local_operator.evaluation.adapters.api`, so it is coupled
to the harness protocol version and lives in-tree. But
`discovery.distribution_digest` pins the adapter by verifying every RECORD row
of the installed wheel — so it must be a *separate* distribution, or every
harness release would invalidate the adapter pin. Isolation comes from the
wheel + digest + isolated worker, not from the source's location.

## Prerequisite: the gated task corpus

OSWorld 2.0's 108 task classes are a **gated** Hugging Face dataset
(`xlangai/osworld_v2_tasks`, `gated: "auto"`). A human must accept the terms
once, and an `HF_TOKEN` must exist, before the workspace can be materialised.
This is a **build-time** prerequisite, never an episode-time one: the tasks are
downloaded once into the workspace, whose `workspace_digest` then pins the
exact task bytes every episode runs.

## Build, lock, install

```sh
cd benchmarks/osworld_v2_adapter

# 1. lock (in the source tree)
uv lock                                     # writes uv.lock (committed)

# 2. build the wheel
uv build --wheel --out-dir dist/

# 3. create the dedicated interpreter + install the locked set.
#    --copies is REQUIRED: python_executable must be a real file, not a
#    symlink (discovery._symlink_free).
uv venv --python 3.12 --copies /opt/lop-adapters/osworld-v2/0.1.0/venv
uv pip install --python /opt/lop-adapters/osworld-v2/0.1.0/venv/bin/python \
    --no-deps dist/lop_osworld_v2_adapter-0.1.0-py3-none-any.whl
uv pip install --python /opt/lop-adapters/osworld-v2/0.1.0/venv/bin/python \
    -r <(uv export --frozen --no-emit-project)
uv pip install --python /opt/lop-adapters/osworld-v2/0.1.0/venv/bin/python \
    local-operator==<harness version>

# 4. materialise the workspace (needs HF_TOKEN once; writes
#    adapter-release.json, tasks/, benchmark_release.json, task_hashes.json,
#    then chmod -R a-w). See scripts/build_osworld_adapter.py.
python ~/local-operator/scripts/build_osworld_adapter.py \
    --benchmark-release osworld-v2-2026.08.08 \
    --out /opt/lop-adapters/osworld-v2/0.1.0/workspace

# 5. compute the three digests the AdapterSelector needs
python - <<'PY'
from importlib.metadata import PathDistribution
from local_operator.evaluation.adapters.discovery import (
    distribution_digest, workspace_digest,
)
print("package_digest  ", distribution_digest(PathDistribution(<dist-info path>)))
print("workspace_digest", workspace_digest("/opt/lop-adapters/osworld-v2/0.1.0/workspace"))
PY
```

`release_digest` is our attestation of the build:
`sha256("lop-osworld-v2-adapter" || version || package_digest ||
benchmark_release_name || task_hash_manifest_sha256)`, written into both
`adapter-release.json` and the selector. It ties the harness build to the
benchmark release — the claim a leaderboard number must carry.

## Leak audit (operator command)

Every instance this adapter creates carries the `lop:adapter=osworld-v2` tag,
so a complete leak detector is one query:

```sh
aws ec2 describe-instances \
  --filters Name=tag:lop:adapter,Values=osworld-v2 Name=instance-state-name,Values=running
```

Run it before and after any paid episode; it must return empty.

## Known scope limitations (PR 1)

- **Infeasible tasks are refused, not merely undocumented.** The runner returns
  on a `finish` batch without calling `execute` (episode.py:531-534), so the
  adapter never sees the terminal action and cannot push `DONE`/`FAIL` into
  OSWorld's `action_history`, which `evaluate()` reads to score `infeasible`
  tasks. An agent correctly declaring such a task infeasible would score 0. We
  will not fabricate a `FAIL` the agent never sent — that would be score fraud.
  `reset_start` therefore raises `InfeasibleTaskExcluded` **before allocating
  anything** when a task's evaluator declares `func: "infeasible"`, so an
  operator-built workspace containing one fails loudly instead of silently
  grading an honest refusal as a failure. Support lands when the terminal-claim
  gap is resolved.
- **Screenshot-only observations.** The a11y tree is not shipped as a frame
  (a geometry for an XML document is a fiction). Its presence is recorded in
  observation metadata; shipping it is a protocol addition, not a fake frame.
- **`user_simulator` is one-sided.** The harness's own responder supplies the
  answer the model sees; the benchmark's simulator is notified for the record.
  Faithful two-sided wiring is a later PR. Pilot tasks declare no simulator.
- **No AWS provider.** `providers/aws.py`, the `rescue_root` sweep, and the
  paid single-episode proof are PR 2.

## Out of scope (permanently)

Multi-task run orchestration, leaderboard reporting/aggregation, and any CLI
or TUI surface.
