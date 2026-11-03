# Experimental coding lane — portable, not validated

This is a sanitized port of the local `harness-cost-bench/coding` lane. It is
**not part of the published comparison** and contains no paid coding results.
An **unverified private report** claims **21/30** Terminal-Bench 2.1 oracle
passes, six baseline Linux runtime builds and incomplete Hermes setup. Public
subset metadata contains task identifiers, not evidence of oracle success. Neither that
snapshot nor these offline checks establishes a working public coding run.

## Included and deliberately excluded

Included: Python adapters/dispatcher/budget guard/Responses bridge/Local Operator
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
python3 -B -m unittest -v test_guard
python3 -B -c 'from pathlib import Path; [compile(p.read_text(), str(p), "exec") for p in Path(".").glob("*.py")]'
python3 -B driver.py --help
```

The stdlib-only tests exercise conservative budgets, corrupt ledgers, wrong
routes at the accounting boundary, output-limit/model refusal, disk thresholds,
frozen-file hashes and CLI refusal. They do **not** import Harbor, run ASGI,
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

Local Operator's required reviewed source revision is
`07c324e9f613b378c437936ede1978c9fc0b49db`. Its source installation is **not**
the existing `0.94.22` installed baseline. This port does not build either one;
a matching source/runtime attestation is mandatory before dispatch.

A separately operated meter must accept `/r/<run_id>/v1/messages`, authenticate
upstream using **`OPENROUTER_API_KEY` from its environment**, enforce the explicitly
chosen billing tier, and append exact-run usage/cost rows to the external request
log. There is no `runner.py`, private auth database, token refresh, recommendation
credential mount, or auth-file fallback. The driver requires the environment
key but never copies it into Harbor's environment options. The external
Responses translator is also separately provisioned; optional
`CODING_TRANSLATOR_API_KEY` is read only from its host-side bridge environment.
Never put a credential in URLs, CLI arguments, JSON attestations or logs.

The host guard listens on loopback :8788; the bridge on loopback :8789. Ports
must be free. `--container-guard-url` names the Docker-reachable mapping to that
same guard (for example a Docker host gateway), not a direct model endpoint.
Do not expose a bearerless guard beyond an isolated task network. The driver
only owns/reaps the two processes it starts; it does not restart shared services.

## Fail-closed frozen gate

`driver.py --help` lists all required arguments. There are **no implicit** model,
billing-tier, budget or price defaults and no automatically generated approval.
Without `--execute`, only local preflight runs; with no arguments it exits 2
before dispatch. Current `subset.json` is incomplete and intentionally cannot
pass full preflight. A future owner must complete the original seeded oracle
order before freezing a 30-task subset, not relabel 21 tasks as 30.

Supply four external JSON artifacts and each artifact's explicit SHA-256:

* **Oracle attestation** (`--oracle-attestation`, `--oracle-sha256`):
  `frozen: true`, `verified: true`, `subset_sha256`, and a `tasks` mapping of
  each selected name to `reward: 1`, matching `image_digest` and `task_checksum`.
  Obtain this from separately retained oracle evidence; the distributed subset
  contains no measured rewards, host Python version or disk measurements.
* **Runtime attestation** (`--runtime-attestation`, `--runtime-sha256`):
  `frozen: true`; `runtimes` maps each selected harness to `version`,
  `linux_verified: true`, and a nonempty `files` mapping of toolchain-relative
  paths to SHA-256. Local Operator additionally requires the full
  `source_revision` above and version `07c324e9-source`. All mapped files are
  hash-checked and may not escape the supplied toolchain root.
* **Toolwire attestation** (`--toolwire-attestation`, `--toolwire-sha256`):
  `verified: true`, `preregistered_tools_observed: true`, `text_only: true`,
  `runtime_sha256`, exact `model`, `output_limit_enforced: true`,
  `billing_tier_enforced: true`, `guard_route_verified: true`, the exact
  `billing_tier`, `models: [<explicit model>]`, and `schema_sha256` matching
  the shared public schema snapshot. These booleans are attestations of actual
  external evidence, not values this port has measured. Retain that evidence
  outside this source tree; do not assert them just to bypass the gate.
* **Preregistration** (`--preregistration`, `--preregistration-sha256`):
  `frozen: true`; exact `billing_tier`, `model`, ordered `harnesses` and `tasks`
  lists, `rep`, `total_budget_usd`, `trial_budget_usd`, `input_usd_per_m`,
  `output_usd_per_m`, `runtime_sha256`, `toolwire_sha256`, `oracle_sha256`, `subset_sha256`,
  `catalog_sha256`; `source_sha256` maps every local `*.py` basename to its
  SHA-256. `task_files` maps each selected task name to its **complete** recursive
  mapping of external payload-relative filenames to SHA-256. Escaping symlinks
  and payload changes fail. Freeze these only after independent review.

Price bounds are **USD per million tokens**, not claims about current provider
pricing. They must upper-bound the explicit tier's input/cache-write/read and
output charges. The guard uses request UTF-8 bytes as a conservative text-token
bound and requires a positive integer output cap on every forwarded POST. It
rejects unknown models, multimodal inputs, unpriced routes and mismatched
request service tiers. Helpers using an unpriced model are refused, not silently
billed at the main model's rate. These restrictions may prevent a native harness
from running; toolwire QA must demonstrate compatibility before freezing.

Every outbound request reserves against **both trial and total USD caps** before
forwarding, under an exclusive process lock. Reservations are atomic, durable,
and **never automatically refunded** after a stream/error: this sacrifices
capacity to avoid late-metering and restart gaps. The external ledger is the
reservation source of truth, not a zeroed per-process counter. Corrupt or
interrupted ledgers fail closed; reconcile deliberately outside a paid run.
Measured `usage.cost` is reported separately; missing costs are never treated as
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

The model-only budget guard does **not** bound Radient classifier spend or
establish a classifier-envelope guarantee. The dispatcher **unconditionally
rejects paid Local Operator trials**, even if arbitrary frozen JSON claims
verification. Classifier reservation and complete accounting need executable
implementation and independent validation before lifting that gate. Radient is
not disabled or replaced to bypass this missing coverage. The reviewed source
identity remains distinct from the installed 0.94.22 baseline.
