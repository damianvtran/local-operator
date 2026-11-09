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

**No arm gets a server-side tool (R14 M2).** An Anthropic-executed tool
(web search/fetch, code execution, computer use, ...) runs outside the network
fence, a contamination route, and is unpriced by our rules. Client-side web
tools stay available to every arm and fail identically: there is no route. Each
arm's server tools are denied through its own native mechanism
(`adapters.SERVER_TOOL_AUDIT`, tested in `test_adapters`), and the meter still
refuses every server tool (402) as the backstop; no price is added.

Codex 0.147 would otherwise send its Responses `web_search` tool, which LiteLLM
1.106.6 turns into `web_search_20250305` (d0b2b32c DEV capture: 6 refusals per
pass, 12 across both passes). Claude Code 2.1.274 lists `WebSearch` as a
deferred tool and implements it as a sub-request carrying `web_search_20250305`
(the d0b2b32c fixture passes never chose it). `codex_bridge.py` also refuses
(400, nothing forwarded) any `web_search*` tool, so a codex config regression
fails at $0 instead of being silently stripped.

**Adapter finding: a refusal costs codex-best the trial (S11, docs only).** After
the agent refuses, Codex's adapter can re-send the conversation ending on an
assistant turn, and the provider rejects that outright: HTTP 400
`invalid_request_error`, "This model does not support assistant message
prefilling. The conversation must end with a user message.", `usage: {}`, no
generation. Measured on the DEV pilot (`PILOT-dd0cb631.md`, attempt 2), one
refusal became a **dead trial** — the agent process exits on the 400 — so the
headline run must either avoid this path or account for it explicitly, and the
results must describe it as **harness behaviour, not a model-quality signal**.
The trial still ends as a graded native failure (Harbor's reward stands), and the
rejected call costs **$0**: the meter settles a 4xx that states no bill at zero
as a rejected request (`settled_no_charge`; [ADAPTERS.md](../ADAPTERS.md)), so the
trial is dead but its cost is not in doubt. The adapter's behaviour is unchanged
here — this note records the finding, it does not fix it.

Read-only **byte-level** audit (R15): every file under each pinned toolchain
root, `node_modules` and single-file binaries included, no ignore rules, for
the server tool type families `web_search_*`, `web_fetch_*`, `code_execution_*`,
`computer_*`, `bash_*`, `text_editor_*`, `memory_*`, `tool_search_tool_regex_*`,
`tool_search_tool_bm25_*`, `advisor_*`, `agent_toolset_*`, `browser_toolset_*`,
`computer_toolset_*` and `mcp_toolset_*` (`adapters.SERVER_TOOL_FAMILIES`). The
hits are frozen in [server-tool-scan.json](server-tool-scan.json), and
`test_adapters` requires every hit to map to a row here
(`adapters.SERVER_TOOL_AUDIT` / `SERVER_TOOL_SCAN_ROWS`). Context-management
edits (`clear_thinking_*`, `compact_*`, `clear_tool_uses_*`) are not tools.

| arm | tool | side | action |
|---|---|---|---|
| claude-code | WebSearch | server (`web_search_20250305`, `max_uses: 8`) | denied: `--disallowedTools WebSearch` (`FrozenClaudeCode`; Harbor also writes it to its settings `permissions.deny` on the ACP path) |
| claude-code | advisor | server (`advisor_20260301`) | inactive: added to a request only when an `advisorModel` setting is present; the bench sets none, so no action |
| claude-code | WebFetch | client (no `web_fetch_*` type in the binary) | kept, fenced |
| codex-best | web_search | server once translated (no versioned type in the binary) | denied: `web_search = "disabled"`; bridge refuses |
| omp | web_search | server (searches through a model role, default `anthropic/claude-haiku-4-5` + `web_search_20250305`) | denied: `--config` overlay `web_search.enabled: false` (`adapters.Omp`) |
| opencode | websearch, webfetch | client (websearch needs a configured provider; none is) | kept, fenced; the scan found no server tool type |
| local-operator | web_search, `deepseek` provider | server, at api.deepseek.com (`web_search/providers.py`) | unreachable: not the meter; needs a DeepSeek credential (none), and `deepseek.com` is a fence model host |
| local-operator | web_search, other providers | client (HTTP from the tool process) | kept, fenced |
| hermes | server-tool refusal check | types only (`side_question_request.has_server_side_tools` names them to refuse them) | kept |
| hermes | vendored `anthropic` SDK | types only (definitions, never sent) | kept |
| hermes | web tools | client | kept, fenced |
| dsh | web search, `deepseek` provider | server: `@deepseek-ai/dsh-web-search-deepseek` (via dsh-base) sends `web_search_20250305` to `https://api.deepseek.com/anthropic/v1` (`lib/index.js:119`; `$DEEPSEEK_SEARCH_BASE_URL` unset) | unreachable: not the meter; needs a DeepSeek credential (none), and `deepseek.com` is a fence model host |
| dsh | session format/persistence | types only (validates recorded tool types) | kept |
| dsh | vendored `@anthropic-ai/sdk` | types only (definitions, never sent) | kept |
| dsh | web tools | client | kept, fenced |

## Cheap checks (no installation, network, Docker or model)

```sh
python3 -B -m unittest -v test_guard test_driver_safety test_adapters test_lop_container
BENCH_TASK_ROOT=<terminal-bench-2-1 task root> python3 -B -m unittest -v test_public_provenance
<Harbor 0.25.0 interpreter> -B -m unittest -v test_harbor_integration
python3 -B -c 'from pathlib import Path; [compile(p.read_text(), str(p), "exec") for p in Path(".").glob("*.py")]'
python3 -B driver.py --help
```

The stdlib-only tests exercise preflight refusals, disk thresholds, frozen-file
hashes, CLI refusal and the container contracts (egress override, pull guard,
gold/quiescence/egress probe, gate relay, verifier record). They do **not** import Harbor, run ASGI,
start Docker, execute Linux runtimes or verify model-tool translation.
`test_harbor_integration` is the exception that needs Harbor itself (it fails,
never skips, without it): it builds a trial through Harbor's own `Trial.create`
(no container is started) to prove `QualificationOracle` gets Harbor's oracle
values under stock Harbor 0.25, which hands `task_dir`/`trial_paths` only to an
agent *named* `oracle` (S9e; no rig entry point such as `harbor_qual.py` is
needed), and renders codex-best's config through Harbor's Codex builder.
`test_public_provenance` re-derives the F27 policy from real task bytes, so it
FAILS, not skips, when `BENCH_TASK_ROOT` is unset or lacks the tasks (R12 F33).
`BENCH_TASK_ROOT_OPTIONAL=1` downgrades that to a skip for local iteration only;
the publication gate never sets it.

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

Hermes is pinned to **v0.22.6 = source commit
`f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab`**. Its CLI reports `vunknown` and
its package metadata 0.0.0, so the version is never read from the binary:
`driver.hermes_version` derives it from the host source check (every tracked
file of that commit, 22066 files, hashed against the toolchain copy) and the
frozen `git ls-remote --tags` capture in `hermes-tag-evidence.json`, where both
`rc.3-v0.22.6` and the final `v0.22.6` (an automated promotion of rc.3) peel to
that commit. Preflight also requires the Hermes runtime record's
`source_revision` to be that commit. Every other harness keeps its probed
version check.

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
  `catalog_sha256`, `package_egress_sha256` (`{policy, allowlist}` hashes of the
  agent-phase package proxy, `driver.package_egress_hashes()`); `source_sha256` maps every local `*.py` basename to its
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

## Agent-phase package egress

Paid trials have one package route in the agent phase, identical for all seven
harnesses: an exact-host forward proxy (`../package_egress.py`) served by the
bench gate on `bench-gate:3128` and, for Local Operator, by its metering edge
under the same policy (LoP's proxy variables must name the edge). Allowlist:
`package-egress-allowlist.json` (exact hosts, ports 80/443); model/LLM hosts are
always denied, even if listed. TLS is tunnelled, not intercepted, so package
bytes are unmetered; each connection is logged and reported per trial
(`runs.jsonl` → `package_egress`). Changing the allowlist or policy changes the
container closure and the preregistration input, voiding earlier records. The
verifier phase keeps Harbor's own uplink and does not use this proxy.

## Gold isolation: strict before inference, reviewed disclosure after (S9b/S9c)

Identical for all seven arms; native Harbor grading unchanged; raw strict
matches are always retained in `bench-isolation.json`.

* **Pre-inference (unchanged, strict):** any whole-file match of a hidden
  `solution/`/`tests/` digest outside the frozen public pairs
  (`public-provenance.json`) refuses the trial before the first model request.
  On success or failure `adapters._preinference_checks` writes a host-side
  receipt, `<trial>/bench-preflight-receipt.json` (trial dir, not a container
  mount), bound to trial id, image digest and manifest hash.
* **Pre-grading, default REFUSE (R11 F27):** a hidden match after the agent
  phase refuses grading unless it is a reviewed **expected-output fixture** --
  a tests/ file the verifier compares the agent's own output against -- listed
  in `post-agent-classes.json` with citations of the exact verifier lines that
  read and compare it (re-validated against the task bytes and bound to its
  payload hash). No heuristic proposes entries; today only sanitize-git-repo's
  `process.py`, `ray_cluster.yaml` and its tokenized json are listed. Executed
  hidden tests (e.g. schemelike `shadow_test/*.scm`), all other tests/ files,
  and any solution-tree file (`solve.sh`, `headless_terminal.py`; R11 F30)
  refuse.
* **A disclosed match is graded only when** (R11 F28) this trial's preflight
  receipt exists, matches trial/image/manifest and is `ok`, AND the fence is
  intact: no denied model-host row, no allowed off-allowlist connection, no
  unparsed or **missing** egress log, and a fresh egress/canary probe passes.
  Missing logs fail closed (R11 F31): every arm launches the gate's package
  proxy (LoP also its edge policy) and the driver creates each listed log
  empty before the trial; `BENCH_PACKAGE_PROXY_LAUNCHED=none` (no arm today)
  is the only case where no log is expected. Premature `/tests` or verifier
  logs, scan truncation and probe failure still refuse.
* **Published:** `runs.jsonl` carries `task` and `post_agent_flags` (rows,
  counts, graded, fence and receipt results); `post-agent-flags.json` is
  `task -> harness -> {trials, graded, refused}`, counting disclosures of
  graded and refused trials separately (R11 F29).
* **Oracle rows** (`rig/requalify.py`) all run `oracle_qualification`'s
  scan-first oracle, so they meet the same receipt precondition; its
  authorized-source mode still permits the oracle's own `/solution` upload.

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
