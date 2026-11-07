# EXPERIMENTAL live-method setup — draft, not ready

This directory adds inspectable setup code and a **draft pending** preregistration
to the historical zero-spend analysis in the parent kit. It is **not validated v2
results**, a live extracted reproduction, a quality-parity claim, or evidence of
wins. Historical exploratory source artifacts remain private and are not shipped.
Nothing here changes the core analysis, its immutable snapshot or its loadout.

## Current gates and limitations

- The original OSWorld 60-task quota is infeasible under current screening.
  Selection metadata identifies 83 candidates, **not** 83 feasible/oracle-cleared
  tasks. The old quota/budget arithmetic remains a draft proposal; amend before
  collection, not after observing outcomes.
- Private development reports 21/30 Terminal-Bench oracles passing. Only six
  Linux runtime toolchains were verified; Hermes remains incomplete. There have
  been **no paid coding pilots**. The source reports are not independently
  revalidated or redistributed here.
- Docker disk policy is an admission gate of at least **21 GiB free**, estimated
  post-pull floor **17 GiB**, and emergency hard floor **15 GiB**. Containers and
  task images/toolchains are not supplied. Do not pull/build anything to test
  this port.
- LOP's reviewed source head is
  `07c324e9f613b378c437936ede1978c9fc0b49db`. An installed `0.94.22` binary is
  distinct and does not prove this source fix. Use an isolated source install of
  that head, record the full source and executable hashes and dependency lock;
  do not update the operator's global runtime.
- Paid coding dispatch is unconditionally hard-blocked pending real executable
  and retained wire-capture admission verification; matching hashes/self-asserted
  JSON are not sufficient. All paid LOP paths remain separately blocked pending
  classifier reservations. Other paid driver paths require a frozen
  preregistration, immutable runtime hashes and tool-wire attestations. A file existing or unit test passing is not wire
  attestation. Cache candidate qualification is not blanket cold-cache approval.
- Public-loadout exposure, all seven runtime routes, streamed usage, classifier
  costs/reservations, oracle validity, independent review and a revised feasible
  selection/schedule are still gates. No headline collection is authorized.

## Layout and shared inputs

- `PREREGISTRATION.md`: draft protocol with explicit current-status corrections.
- `CACHE-QUALIFICATION.md`, `cache_namespace.py`, `cache_forwarder.py`, cache
  tests: final-Messages namespace intervention, with limitations and controls.
- `evidence/`: minimal allowlisted historical counters and hash/marker summaries;
  **no private prompt/profile bodies**. `verify_cache_evidence.py` reproduces the
  stated counter arithmetic, not the historical live responses.
- `codex_catalog.py`, catalogs, `codex_toolsearch_bridge.py`: Codex-specific
  experimental model metadata and tool-search translation. Read `ADAPTERS.md`
  and third-party attribution before use; catalogs are version-specific inputs.
- `lop_launch.py`, `lop_shim/sitecustomize.py`: process-local LOP routing shim;
  do not export this directory as global `PYTHONPATH` to task subprocesses.
- `meter_proxy.py`: explicitly configured, fail-closed model-cost admission and
  metering. See `ADAPTERS.md`; no billing tier/rate/reservation is inferred.
- `runner_v2.py`, `RUNNER_V2.md`: adapted after the author's immutable handoff;
  source/config checks only, with paid LOP blocked pending classifier accounting.
- `fetch_task.py`, `evaluate.py`, `TASKS.md`: explicit pinned upstream public-task
  fetching and upstream-metric evaluation. Downloads are opt-in and stay external.
- `TRANSLATOR.md`, `translator_config.example.json`: credential-free local
  translator setup instructions; no live translator qualification is claimed.
- `coding/`: Harbor adapter/driver/oracle source and subset metadata only. No
  benchmark task payloads, fixtures, Docker toolchains, job logs or credentials.
- `EXPERIMENTAL-MANIFEST.json`: a separate SHA256 draft input inventory, not the
  core kit manifest or a runtime freeze. `SOURCE-AUDIT.json` maps source inputs.

The only public loadout is **`../public_loadout/`**. Its snapshot, instructions
and skill hashes are loaded directly. Do not copy a schema snapshot into this
folder. Downloaded skill bodies are not redistributed here; upstream licenses
(including source-available office skills) still apply.

## Setup sequence (instructions, not actions executed for this port)

1. Read `../public_loadout/README.md` and this directory's notices. Provide your
   own pinned harness binaries/interpreters, optional dependencies and isolated
   task environment. No install command is run automatically. Use explicit
   executable variables/flags documented by each adapter rather than a personal
   wrapper, daemon or unpinned global executable. Never inherit personal HOME
   configs. On macOS, scratch-HOME subprocesses that use keychains require an
   explicit safe isolation strategy; do not provoke credential dialogs.
2. If you choose to fetch skills, run the **existing** public-loadout fetch and
   base-home builder from its directory, after reviewing its network/license
   effects. Use a fresh, user-owned scratch root:

   ```sh
   cd ../public_loadout
   # Optional network operation, NOT run as part of this port:
   bash fetch_public_skills.sh
   BENCH_SCRATCH=/chosen/fresh/scratch bash build_basehomes_public.sh
   # Optional local replay server (frozen schemas, tool calls refused):
   python3 mcp_replay_public.py
   ```

   The builder does not validate native harness discovery. Perform and archive
   runtime-bound public-only wire checks separately. Never copy an auth database
   or token file into any base home.
3. Select an existing Python interpreter with the required optional libraries
   (meter/bridge: Starlette, HTTPX, uvicorn; transformation integration test:
   LiteLLM and jsonschema). Pin and record the resolved versions before freeze;
   these dependencies are not part of the historical analysis requirements.
   Catalog generation requires your separately pinned Codex source prompt:

   ```sh
   "$BENCH_PYTHON" codex_catalog.py --help
   # Use the --prompt/--out and context/search options in ADAPTERS.md.
   ```

4. Configure public model auth using **`OPENROUTER_API_KEY` in the environment**.
   Obtain it securely outside this directory; never echo it, put it on a command
   line, save it in a YAML/JSON/config file, or commit a populated environment.
   Meter/translator credentials remain host-side; do not pass provider secrets
   into task containers. LOP classifier access may use the user's own `lop login
   radient` path in a separate installation; no token or auth database is shipped
   or copied. Classifier costs require separate defensible accounting and bounds;
   a model-only meter is not proof of total-cost coverage.
5. Follow `ADAPTERS.md` to configure an isolated local meter and translator.
   Explicitly set total and per-trial USD limits, conservative maximum request
   exposure, model allowlist, rates and output/input bounds. Use only loopback
   services unless you deliberately secure the network boundary. Avoid occupied
   ports; own/reap exact service PIDs. No default tier or nominal cap authorizes
   payment. A missing price/usage/classifier charge must stop admission, not be
   treated as zero cost.
6. Use the exact opt-in clone/fetch/dependency commands in `TASKS.md` to prepare
   a pinned OSWorld development smoke task outside this distribution; then follow
   `RUNNER_V2.md` for exact environment/config/preflight/execute interfaces.
   Supply your own licensed, pinned coding data/toolchains **outside this
   distribution**, amended and frozen preregistration, schedule, runtime manifest,
   independent oracle/negative-control evidence and actual tool-wire attestations.
   Follow `coding/README.md` for Harbor invocation and gates. The subset is
   development metadata, not a clean headline holdout. Config/preflight modes
   cannot substitute for actual wire validation.
   The public OSWorld fetch/evaluator path is executable setup, not downloaded
   fixtures or oracle proof. Paid LOP remains blocked until classifier reservation
   and accounting are implemented/validated; do not disable Radient to bypass it.
7. Only after every gate is reviewed and frozen may an explicitly authorized
   driver execute paid trials. Preserve run IDs, retries, failures, maximum-
   exposure reservations and attributable side calls. Reconcile all usage and
   side effects before analysis. Keep raw logs private until a privacy review;
   do not add runtime outputs to this distribution by default.

## Cheap local checks

Run from this directory using an existing environment; no dependency installation,
model API, harness process or Docker operation is necessary:

Set `BENCH_PYTHON` to your chosen test-environment interpreter. The full ASGI
controls additionally use `httpx==0.28.1`, `starlette==1.10.0` and
`jsonschema==4.26.0`; install them explicitly in that environment if needed.
A Local Operator session already supplies `LOCAL_OPERATOR_SCRATCHPAD`. In an
ordinary terminal, create an isolated test directory first—no login is needed:

```sh
if [ -z "${LOCAL_OPERATOR_SCRATCHPAD:-}" ]; then
  mkdir -p "${XDG_CACHE_HOME:-$HOME/.cache}"
  export LOCAL_OPERATOR_SCRATCHPAD="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/harness-cost-offline.XXXXXX")"
fi
export TMPDIR="$LOCAL_OPERATOR_SCRATCHPAD"
python3 -B -m unittest -v test_cache_namespace
python3 -B verify_cache_evidence.py
python3 -B -m unittest -v test_runner_v2
# Optional; fails import if LiteLLM/jsonschema are unavailable (do not auto-install):
"$BENCH_PYTHON" -B -m unittest -v test_cache_bridge
# Verify the immutable distributed bytes, including shared public-loadout inputs:
python3 -B freeze_inputs.py
```

`VERIFICATION.md` separates checks actually executed for this port from unverified
live gates. When editing this folder deliberately, inspect changes first and
explicitly regenerate **only** its inventory with `freeze_inputs.py --write`.
The coordinator owns core manifests, review, commits and ZIP assembly. This port
makes none of those actions and must not be described as frozen merely because
its draft content hashes match.
