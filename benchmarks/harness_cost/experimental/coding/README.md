# Experimental coding lane — portable, not validated

This is a sanitized port of the local `harness-cost-bench/coding` lane. It is
**not part of the published comparison** and contains no paid coding results.
An **unverified private report** claims **21/30** Terminal-Bench 2.1 oracle
passes, six baseline Linux runtime builds and incomplete Hermes setup. Public
subset metadata contains task identifiers, not evidence of oracle success. Neither that
snapshot nor these offline checks establishes a working public coding run.

## Included and deliberately excluded

Included: Python adapters/dispatcher/Responses bridge/Local Operator
launcher/oracle selector, the incomplete subset metadata and the Codex model
catalog. Task text, reference solutions, fixtures, runtime binaries, toolchains,
logs, jobs, results, private profiles and credentials are **not included**.

The shared public loadout is **`../../public_loadout`** relative to this directory.
Its `public_mcp_schemas.json` is the only schema snapshot; do not duplicate it here.
The current coding adapters do not themselves inject that MCP loadout into task
containers. Attestation of an actual tool wire is still required, not inferred
from the presence of the snapshot.

Original adapters retain Harbor 0.25.0 integration points and pinned harness
versions. Harbor's implementations are imported, not copied. Codex's catalog and
`tool_search` translation are a separately labelled **codex-best** configuration,
not Codex as shipped. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Cheap checks (no installation, network, Docker or model)

```sh
python3 -B -m unittest -v test_guard test_driver_safety test_adapters test_lop_container
python3 -B -c 'from pathlib import Path; [compile(p.read_text(), str(p), "exec") for p in Path(".").glob("*.py")]'
python3 -B driver.py --help
```

The stdlib-only tests exercise preflight refusals, disk thresholds, frozen-file
hashes, CLI refusal and the container contracts (egress override, pull guard,
gold/quiescence/egress probe, gate relay, verifier record). They do **not** import Harbor, run ASGI,
start Docker, execute Linux runtimes or verify model-tool translation.

## External dependencies and paths

For a separately authorized future run, provision Python >=3.11 with Harbor
0.25.0, httpx, starlette and uvicorn. Hermes' adapter also uses the toolchain's
PyYAML. These packages/toolchains are not bundled or installed by these scripts.
Keep the dependency environment shared; do not copy dependency trees per task.
Obtain Terminal-Bench task payloads separately, review their licenses, and use an
external scratch copy: the dispatcher temporarily pins `task.toml`'s image then
restores it. Supply an independently built Linux/amd64 toolchain mounted read-only
at `/opt/bench`, with the layout expected by `adapters.py` (node, claude, codex,
opencode, lop, omp, dsh, hermes). All host paths are CLI/env configurable.

Local Operator is pinned to released v0.94.22,
`500883ef56ca504325ffd80c5b90bf2435d00e3f`, run unmodified. It is metered only
through the per-run TLS edge (`../lop_edge.py`) on the host, reached from the
task container through the bench gate (`egress.py`, see STATUS.md); no endpoint
shim exists. A matching source/runtime attestation is still mandatory, and paid
dispatch needs a container S4 record bound to `driver.CONTAINER_CLOSURE`.

A separately operated meter must accept `/r/<run_id>/v1/messages`, authenticate
upstream using **`OPENROUTER_API_KEY` from its environment**, enforce the explicitly
chosen billing tier, and append exact-run usage/cost rows to the external request
log. There is no `runner.py`, private auth database, token refresh, recommendation
credential mount, or auth-file fallback. The driver requires the environment
key but never copies it into Harbor's environment options. The external
Responses translator is also separately provisioned; optional
`CODING_TRANSLATOR_API_KEY` is read only from its host-side bridge environment.
Never put a credential in URLs, CLI arguments, JSON attestations or logs.

The lane's Codex bridge listens on loopback :8789 (must be free); containers
reach it, the meter and the LoP edge only through the bench gate, and only a
Codex trial's gate forwards the bridge port (R7 F21).
`--meter-current-run-file` must be the meter's absolute `BENCH_CURRENT_RUN_FILE`:
Codex rows are tagged from it, and a mismatch leaves the trial with no rows
(recorded unverified, lane stops). The driver writes the run id there only for
the Codex trial and restores the prior contents (or removes the file) afterwards,
whatever the outcome. The driver only owns/reaps the processes it
starts (bridge, per-trial LoP edge); it does not restart shared services.

## Paid coding dispatch hard-blocked — evidence admission incomplete

`driver.py --help` lists all required arguments. There are **no implicit** model,
billing-tier, budget or price defaults and no automatically generated approval.
**Paid coding dispatch is unconditionally disabled**, including for non-LOP
harnesses and even with matching hashes/self-asserted manifests. The current
runtime/toolwire checks do not verify actual executables and retained captures;
they cannot license a paid run. Both preflight admission and `run_trial()` refuse
before services, authentication or Harbor/model dispatch. `--help` remains safe;
with no arguments the CLI exits 2. Current `subset.json` is also incomplete. A future owner must complete the original seeded oracle
order before freezing a 30-task subset, not relabel 21 tasks as 30.

The following describes the **legacy draft metadata contract**, not a working
admission gate. Supplying these four JSON files and hashes cannot unblock paid
coding. A future, separately reviewed implementation must bind actual executable
components and independently interpret retained wire captures before lifting the
hard hold; boolean assertions and arbitrary hashed text are insufficient:

* **Oracle attestation** (`--oracle-attestation`, `--oracle-sha256`):
  `frozen: true`, `verified: true`, `subset_sha256`, and a `tasks` mapping of
  each selected name to `reward: 1`, matching `image_digest` and `task_checksum`.
  Obtain this from separately retained oracle evidence; the distributed subset
  contains no measured rewards, host Python version or disk measurements.
* **Runtime attestation** (`--runtime-attestation`, `--runtime-sha256`):
  `frozen: true`; `runtimes` maps each selected harness to `version`,
  `linux_verified: true`, and a nonempty `files` mapping of toolchain-relative
  paths to SHA-256. Local Operator additionally requires the full
  `source_revision` above and version `500883ef-v0.94.22`. All mapped files are
  hash-checked and may not escape the supplied toolchain root.
* **Toolwire attestation** (`--toolwire-attestation`, `--toolwire-sha256`):
  `verified: true`, `preregistered_tools_observed: true`, `text_only: true`,
  `runtime_sha256`, exact `model`, `output_limit_enforced: true`,
  `billing_tier_enforced: true`, `guard_route_verified: true`, the exact
  `billing_tier`, `models: [<explicit model>]`, and `schema_sha256` matching
  the shared public schema snapshot. These booleans are **unverified assertions**
  in the legacy contract: the port does not derive them from retained captures.
  Consequently, even matching assertions cannot unblock paid coding dispatch.
* **Preregistration** (`--preregistration`, `--preregistration-sha256`):
  `frozen: true`; exact `billing_tier`, `model`, ordered `harnesses` and `tasks`
  lists, `rep`, `total_budget_usd`, `trial_budget_usd`, `input_usd_per_m`,
  `output_usd_per_m`, `runtime_sha256`, `toolwire_sha256`, `oracle_sha256`, `subset_sha256`,
  `catalog_sha256`; `source_sha256` maps every local `*.py` basename to its
  SHA-256. `task_files` maps each selected task name to its **complete** recursive
  mapping of external payload-relative filenames to SHA-256. Escaping symlinks
  and payload changes fail. Freeze these only after independent review.

Price bounds are **USD per million tokens**, not claims about current provider
pricing; they are preregistration bindings. **The external meter
(`../meter_proxy.py`) is the single budget authority** for this lane, as for the
desktop lane (R6 F19): every container route -- meter port, LoP edge, and Codex
bridge -> translator -> meter -- ends at its ledger, which enforces the trial and
campaign caps on durable reservations (`../ADAPTERS.md`). Start it with
`BENCH_BUDGET_USD`/`BENCH_TRIAL_BUDGET_USD` equal to `--total-budget-usd` and
`--trial-budget-usd`: preflight reads the meter's caps from `/health` and refuses
unless both match exactly (R7 F22). The lane-local reservation guard that used to sit on
:8788 was removed: after the bench gate it was on no container route, so it was
a documented layer that bounded nothing. Measured `usage.cost` is reported
separately; missing costs are never treated as
free. Disk space is checked before pulls: >=21 GiB free and estimated >=17 GiB
remaining (compressed bytes ×3 +512 MiB). Active trials stop below 17 GiB,
protecting the **15 GiB hard floor**. Shared writers mean this is a guard, not a
filesystem quota. Only the current exact task image is removed; never global prune.

## Oracle continuation (not run for this port)

`oracle_sweep.py` requires `--execute`, external `CODING_TASK_ROOT`, external
`CODING_ORACLE_OUTPUT`, and optional `CODING_DISK_VOLUME` (defaults to cwd).
It checks >=21 GiB before any network/Docker work, enforces estimated >=17 GiB
post-pull, stops active jobs below 17 GiB, and resumes its external JSONL log.
It does not write over this published incomplete subset. Reference solutions
make no model calls but still consume Docker/disk resources: do not run them
without a separate resource lease. This publication task authorized no such run.

## Local Operator classifier gate remains pending

The model-only budget path does **not** bound Radient classifier spend or
establish a classifier-envelope guarantee. The dispatcher **unconditionally
rejects paid Local Operator trials**, even if arbitrary frozen JSON claims
verification. Classifier reservation and complete accounting need executable
implementation and independent validation before lifting that gate. Radient is
not disabled or replaced to bypass this missing coverage. The reviewed source
identity remains distinct from the installed 0.94.22 baseline.
