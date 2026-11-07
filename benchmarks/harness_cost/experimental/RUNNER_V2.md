# Experimental runner_v2 — immutable-source port, not qualified live runs

The adapter owner handed off `runner_v2.py` at source SHA256
`3f36b8aba2454bd496b42726790863dd8ec2cdfa2d35342781d32ef4992b9e22`.
That hash was checked before copying. This public copy has intentional portability
and guard changes, and therefore **does not inherit the source review as proof
of its own runtime qualification**. Its current bytes are in the distinct
experimental manifest. The original development output/logs are not shipped.

## What changed for the public folder

- Canonical `PUBLIC_LOADOUT` resolves `../public_loadout`, optionally overridden
  with `BENCH_PUBLIC_LOADOUT`. No snapshot/schema is duplicated here.
- `BENCH_TASKS_DIR` selects external generated data; `BENCH_TASKS_FILE` and
  `BENCH_TASK_CACHE` override its tasks.json and cache/ paths. `BENCH_EVALUATOR`
  defaults to this folder's evaluator. `BENCH_OSWORLD_ROOT` selects the pinned
  external upstream metric checkout. See `TASKS.md` for executable opt-in fetch.
- `BENCH_SCRATCH` holds mutable outputs, locks/current-run routing and logs, not
  this source folder. It must be outside the account HOME so ancestor configs
  cannot be discovered. Use an explicitly chosen filesystem with sufficient free
  space. The safe temporary-root default is not an approved benchmark location.
- `BENCH_TASK_PATH` and `BENCH_TASK_PY` select task tools/interpreter. Defaults
  are system command-search paths/current Python, not a supplied task toolchain.
- `LOP_BENCH_SOURCE_ROOT` plus `BENCH_LOP_PY` are passed explicitly to the LOP
  launcher. It requires the reviewed source install, not installed 0.94.22.
- External evaluator preflight must check source pin, task/fixture hashes and
  metric import before configuration/execution; unfetched planning output fails.
- Paid LOP is **unconditionally blocked** pending a validated classifier
  reservation/accounting implementation. The main-model meter cannot bound a
  separate classifier service. Do not disable Radient or fabricate a boolean
  attestation to bypass this gate. The user's own `lop login radient` path may
  be used during separately authorized qualification, but no auth DB is copied.
- CoW is preferred: APFS uses `cp -Rc`; Linux first tries
  `cp --reflink=always -a`. Cross-filesystem copies remain refused.
  `BENCH_CP_BIN` can select the copy executable. If Linux CoW is unavailable,
  only dependency-free profile data bounded to 8 MiB and 4,096 entries may use
  ordinary copying. Symlinks, special files, runtime/dependency/cache directories
  and oversized profiles are refused; actual reads/writes are bounded and partial
  cleanup is limited to the destination this call created. Large/runtime profiles
  (including prepared Hermes environments) still need CoW; dependencies are never
  silently copied. This preparation fallback is not native runtime qualification.
  The Linux path is unverified. Hermes' non-Darwin lifecycle path is explicitly
  rejected rather than risking a detached, unmetered gateway.

## Explicit local setup and invocation

The following commands are **instructions only**, not a live verification record.
Use `TASKS.md` first for the pinned public development task and its opt-in fetch.
Supply optional Python dependencies and seven isolated harness source installs
separately; no script installs them automatically. Record versions and hashes.

```sh
# From this experimental directory; choose actual existing programs/paths.
export BENCH_KIT="$PWD"
export BENCH_PUBLIC_LOADOUT="$BENCH_KIT/../public_loadout"
export BENCH_SCRATCH='<fresh-outside-HOME-root-on-clone-capable-filesystem>'
export BENCH_PUBLIC_BASEHOME="$BENCH_SCRATCH/basehome-public"
export BENCH_CLEAN_BASEHOME="$BENCH_SCRATCH/basehome"
export BENCH_TASKS_DIR='<external-fetch-output-from-TASKS.md>'
export BENCH_TASKS_FILE="$BENCH_TASKS_DIR/tasks.json"
export BENCH_TASK_CACHE="$BENCH_TASKS_DIR/cache"
export BENCH_OSWORLD_ROOT='<clean-pinned-OSWorld-checkout>'
export BENCH_EVALUATOR="$BENCH_KIT/evaluate.py"
export BENCH_EVAL_PY='<existing-metric-dependency-python>'
export BENCH_BRIDGE_PY='<existing-bridge-dependency-python>'
export BENCH_TASK_PY='<prepared-task-python>'
export BENCH_TASK_PATH='<prepared-task-bin-and-system-PATH>'
export BENCH_CLAUDE_BIN='<isolated-native-claude>'
export BENCH_CODEX_BIN='<isolated-native-codex>'
export BENCH_OMP_BIN='<isolated-native-omp>'
export BENCH_OPENCODE_BIN='<isolated-native-opencode>'
export BENCH_LOP_PY='<reviewed-source-install-python>'
export LOP_BENCH_SOURCE_ROOT='<clean-reviewed-07c324e9-source-root>'
export BENCH_HERMES_BIN='<isolated-native-hermes>'
export BENCH_HERMES_SOURCE='<pinned-hermes-source>'
export BENCH_HERMES_INSTALLS='<prepared-relocatable-Anthropic-runtime-installs>'
export BENCH_NODE_BIN='<prepared-node>'
export BENCH_DSH_ENTRY='<installed-dsh-lib/bin.js>'
export BENCH_CODEX_CATALOG="$BENCH_KIT/codex-model-catalog.json"
export BENCH_MODEL_METADATA='<independently-verified-model-metadata.json>'
export BENCH_PROXY=http://127.0.0.1:8787
export BENCH_MCP_URL=http://127.0.0.1:8790
export BENCH_LITELLM_URL=http://127.0.0.1:4000/v1
export BENCH_CURRENT_RUN_FILE="$BENCH_SCRATCH/current_run"
export BENCH_REQUEST_LOG="$BENCH_SCRATCH/logs/requests.jsonl"
export BENCH_BODY_DIR="$BENCH_SCRATCH/logs/bodies"
export BENCH_WIRE_ATTESTATIONS='<independent-runtime-bound-wire-record-directory>'
export BENCH_PREREG_FILE='<coordinator-frozen-preregistration.json>'

# After preparing clean/public homes via the shared public-loadout builder:
python3 "$BENCH_KIT/runner_v2.py" --augment-public "$BENCH_PUBLIC_BASEHOME"
# clean needs runtime-only Hermes state too; do not copy public skills/auth into it.
# The clean/public home checks document required native discovery paths in code.

# Config-only: no model call/auth resolution, but validates external input assets.
python3 "$BENCH_KIT/runner_v2.py" omp 37608790 40 public-full --config-only
# Read-only service/wire preflight; requires a genuine public replay + attestations.
python3 "$BENCH_KIT/runner_v2.py" omp 37608790 41 public-full --preflight-only

# PAID, NOT RUN/AUTHORIZED BY THIS PUBLICATION; frozen gates must all pass first:
# python3 "$BENCH_KIT/runner_v2.py" omp 37608790 42 public-full --execute
```

The example task is **development-only**, not a heldout headline score. Reps are
40–49 only, arms `clean`/`public-full`, IDs start `v2dev__`; previous trial IDs
are never replaced. Results go to scratch `logs/runs_v2dev.jsonl`, never the
historical rep-0 ledger. Config-only removes its temporary HOME afterwards.

## Translation, budgets and wire gates

`ADAPTERS.md` documents meter bounds/auth and Codex bridge; OpenRouter auth is
`OPENROUTER_API_KEY` in the **meter** environment, never a task config. The
Responses translator is separately provided (LiteLLM), no private litellm.yaml
or key is shipped. `BENCH_LITELLM_KEY` is a local translator credential supplied
in memory/env for Codex. Configure the translator's Anthropic api_base to
`http://127.0.0.1:8787/r/@current` and serialize trials sharing that routing file.
A complete explicit translator setup is in `TRANSLATOR.md`; no install/start was
performed here. Other harnesses address their `/r/<run_id>` meter route directly.

The meter must use the same `BENCH_CURRENT_RUN_FILE` and request-log path. Native
wire attestation needs final Messages bodies; default capture is off for privacy.
Enable `BENCH_CAPTURE_BODIES=1`/`BENCH_BODIES` only in an isolated qualification
session, retain evidence privately, and do not add transcripts to this kit.
Attestations bind runtime fingerprints, integration input hashes, actual public
schema/instruction exposure and raw deferred search/replay exchanges. Fabricated
summaries cannot stand in for these records. Model metadata must verify served
model/context/output limits; a handwritten fixture used in tests is not evidence.

The draft prose protocol is **not** the JSON freeze record accepted by
`frozen_prereg()`. The latter must bind this exact adapted runner hash, public
inputs, task manifest, evaluator, model metadata/catalog, runtime fingerprints,
arm and reviewed LOP head. No ready-to-sign authorization record is bundled.
Meter caps, rates and maximum exposures must be independently qualified first.
Classifier costs remain a separate blocker even if main-model tests are green.
Billing aggregation includes every recorded Messages attempt, including billed
HTTP errors. Missing/invalid billing makes `cost_complete` false and complete
model/total amounts null; `measured_model_cost_usd` preserves the partial known
sum. Status/unknown-attempt counts remain visible. Discovery/count-token rows
are not model attempts and cannot by themselves establish a measured model run.
This checks recorded rows, not attempts entirely absent from the meter ledger.

## Actually verified versus pending

The round-2 remediated port passed **20 stdlib-only runner tests**, including
mixed-success/missing/error billing negative controls and seven config paths,
synthetic-skill filesystem cloning/instruction deduplication, external-preflight
refusal, missing wire/freeze refusal, explicit LOP source-launch environment and
paid-LOP refusal before auth/preparation. These tests launch no harness/model.
They do not load real skill bodies or establish actual runtime wire behavior.

Unverified: real seven-harness launch/discovery, Linux copy/runtime behavior,
Hermes completion, downloaded inputs/oracle qualification, served provider
cache/cost bounds, LiteLLM/Codex live translation and classifier reservations.
This source port is useful executable setup, **not a ready benchmark**.
