# Coding lane status — 2026-11-04

## Container metering path (implemented; container S4 pending)

* **Egress:** `main` joins only a run-owned `internal: true` Docker network (no
  route, no external DNS). A `bench-gate` container (same digest-pinned task
  image, frozen-toolchain Python, `gate_relay.py`) on that network plus an uplink
  relays a fixed port map to the host meter (`/r/<run_id>`), Codex bridge and
  LoP edge. Rejected: host-gateway + host firewall (Docker Desktop's VM traffic
  never crosses host `pf`). Compose services use `pull_policy: never`; every
  driver/verifier/oracle docker call goes through `egress.run_docker`, which
  parses global options and refuses run/create/compose up|run|create without a
  leading `--pull never`, and any pull/build except the driver's one explicit,
  disk-gated `docker pull` (R6 F17). The egress probe dials a run-owned OPEN
  host listener the gate never forwards and passes only on no-route/timeout,
  never on "refused" (R6 F16).
* **LoP:** `lop_launch.py` (read-only mount) requires `HTTPS_PROXY` =
  `bench-gate:<edge port>` with run identity, the read-only run CA, the edge's
  dummy `RADIENT_API_KEY`, and the `lop-baseline-500883ef` overlay (direct_url
  commit, RECORD hashes, import origins) before importing `local_operator.cli`.
* **Gold isolation:** before inference `gold_probe.py isolation` checks that no
  container file has the bytes of any host `solution/`/`tests/` file (hashes
  only), `/solution` and `/tests` are absent, and `/logs/verifier` is empty;
  `egress` checks the fence. `bench_verifier.BenchVerifier` repeats isolation,
  records quiescence (survivors' pid, command-line hash and listening ports, as
  a disclosed `agent-processes-survived-into-grading` flag; never killed, never
  a void), attaches the uplink, then runs Harbor's stock verifier unchanged
  (R6 F15: nginx-request-logging and kv-store-grpc grade the agent's running
  server). Only a gold-isolation failure refuses grading; since S9b a hidden
  digest first seen after the agent phase is disclosed and graded unless it is
  verifier code or the trial shows a fence anomaly (README "Gold isolation").
  The isolation scan
  hashes every regular file streaming, with no size cut-off (R6 F20);
  partial-content matches (archives, concatenations) are out of scope.
* **Budget:** the external meter is the single budget authority for both lanes
  (R6 F19); the lane-local reservation guard was removed.
* **Agent-phase package egress (S8, 2026-11-04):** the paid fence left no
  package route in the agent phase, so `apt-get`/`pip` in the oracle's own
  `solve.sh` failed (nginx-request-logging, kv-store-grpc: 0). The gate now
  also runs `../package_egress.py`, an HTTP CONNECT + plain-HTTP forward proxy
  on `bench-gate:3128`: exact hosts from `package-egress-allowlist.json` (apt
  mirrors, PyPI, four solve.sh download hosts), ports 80/443, no IP literals,
  upstream addresses must be public. Model/LLM hosts and their subdomains
  (anthropic, openai, openrouter, radienthq, deepseek, Gemini/Vertex, Bedrock,
  Azure OpenAI, Mistral, xAI, Groq, ...) are denied even if listed. TLS is
  tunnelled, not intercepted: package bytes are unmetered and disclosed. Every
  connection (host, port, bytes, allowed/denied + reason) is logged and copied
  into the trial's `runs.jsonl` row as `package_egress`. Same route for all 7:
  the six non-LoP harnesses get `HTTP(S)_PROXY=http://bench-gate:3128` with
  `NO_PROXY=bench-gate,...` (their model URLs all name `bench-gate`, so model
  traffic never enters the proxy); LoP keeps `HTTPS_PROXY` = its edge (the
  launcher refuses anything else), and the edge applies the SAME policy to
  tool-child requests, tunnelling allowlisted hosts and logging them in the
  same row shape, while model routes stay MITM-metered. Policy and allowlist
  are `CONTAINER_CLOSURE` members and the prereg `package_egress_sha256` input.
  Limits: static-scan derivation (redirect/CDN hosts unconfirmed until a logged
  oracle sweep through the proxy); a tool that ignores proxy env still has no
  route; the canary denial is unchanged.
* **REQUIRED before any pilot:** re-qualify the oracle for the subset under
  `BenchVerifier` (Docker; not run). The earlier oracle acceptance used Harbor's
  stock verifier and does not carry over to this grading path.


## Historical source status, not new validation

An **unverified private report** claims 21/30 accepted Terminal-Bench 2.1
oracles and six completed baseline Linux runtime builds, with Hermes incomplete.
No raw minimal oracle/runtime evidence is redistributed, so those statements
are not independently reproducible publication results. `subset.json` contains
only public selection/image/checksum metadata, not oracle rewards, measured
host disk figures or host Python version. External frozen oracle evidence is
mandatory before dispatch; the public task list does not establish passes.

Local Operator is pinned to released v0.94.22
(`500883ef56ca504325ffd80c5b90bf2435d00e3f`) and runs here only behind the
host-side per-run edge, reached through the bench gate (`egress.py`). Paid
dispatch stays HARDBLOCKED until a zero-paid container S4 record
(`--coding-edge-evidence`) matches the exact container metering closure
(`driver.CONTAINER_CLOSURE`); that S4 has not run yet (disk gate). No source rebuild,
paid task-container pilot, finalized preregistration or final toolwire validation
was completed by this port.

## What this port changes

* External task/toolchain/output/meter/translator paths; portable disk checks.
* Public auth via `OPENROUTER_API_KEY` environment, with no private `runner.py`,
  auth DB, recommendation-key mount or persisted real API credential.
* Explicit USD trial + total caps, explicit tier and pricing bounds, durable
  conservative reservations, missing usage treated as incomplete evidence.
* Fail-closed dispatch on incomplete subset, absent frozen artifacts, runtime,
  code/catalog/task hash mismatches or missing toolwire/tier attestations.
* Shared root public-loadout reference, no duplicate MCP schema or task payloads.
* Process-local Local Operator endpoint shim; bridge remains a labelled
  non-native Codex compatibility treatment, not an ordinary upstream arm.

## Validation performed here

Only stdlib offline unit tests, Python compilation/AST/module-docstring checks,
JSON parsing and CLI negative controls. No dependencies were installed.
The current Python has no httpx, starlette or uvicorn; ASGI/HTTP transport was
**not** exercised. Harbor/Docker/Linux native binaries/model translation remain
unvalidated. No network, model, harness, Docker, heavy or paid pilot ran.

Reproduction commands and artifact contracts are in README.md. Completion needs
an independently approved runtime build, remaining free oracles, actual toolwire
QA, a reviewed frozen preregistration and a new resource/budget authorization.
Do not include this lane in the public cost comparison or derive a full-run
budget estimate from it.

## Local Operator classifier gate remains pending

The model-only budget path does **not** bound Radient classifier spend or
establish a classifier-envelope guarantee. The dispatcher **unconditionally
rejects paid Local Operator trials**, even if arbitrary frozen JSON claims
verification. Classifier reservation and complete accounting need executable
implementation and independent validation before lifting that gate. Radient is
not disabled or replaced to bypass this missing coverage. The reviewed source
identity remains distinct from the installed 0.94.22 baseline.
