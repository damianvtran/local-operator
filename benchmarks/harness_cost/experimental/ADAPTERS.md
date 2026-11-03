# Experimental harness adapters

These are sanitized copies of the matching files in the private
`harness-cost-bench` working rig, not evidence of a public end-to-end run.
No model call, network request, installation, runtime update, or harness launch
was performed when preparing this copy. `runner_v2.py` is coordinated separately.
The public loadout remains `../public_loadout` relative to this directory;
these adapters neither copy its schemas nor change that loadout.

## Dependencies and runtime identity

Supply the Python interpreter explicitly (`BENCH_PYTHON` below). The meter and
bridge require existing `httpx`, `starlette`, and `uvicorn` installations; no
installation or binary lookup is performed. Codex/LiteLLM executables, their
configuration, and their interpreters must be selected explicitly by the runner.
LiteLLM is optional and was absent in the inspected environments, so translator
integration was not exercised. The catalog targets Codex `rust-v0.147.0` only.

`lop_launch.py` requires `LOP_BENCH_PYTHON` (absolute path to the selected source
install's Python), `LOP_BENCH_SOURCE_ROOT`, and the per-run edge environment
(`HTTPS_PROXY` naming a loopback edge with a run identity, `SSL_CERT_FILE` =
`REQUESTS_CA_BUNDLE` = the run CA, `NO_PROXY` = loopback only). It checks Git
HEAD against released v0.94.22 `500883ef56ca504325ffd80c5b90bf2435d00e3f` (full
SHA), refuses tracked modifications and untracked files, and checks the actual
`local_operator` package origin. It patches nothing: `lop_edge.py` meters LoP's
real first-party URLs at the network boundary (see `LOP-EDGE-VERIFICATION.md`).
It does not update, rebuild, or otherwise mutate any install.
The shim is in-process, not added to inherited `PYTHONPATH`.

## Explicit budget/auth setup (illustrative, not executed)

Inject `OPENROUTER_API_KEY` into the environment using your secret manager.
Never put it into a command argument, config file, report, or checked-in `.env`.
There is no private auth-store fallback, credential ID, subscription impersonation,
or identity injection. The following placeholders are deliberate: choose real
rates and caps from a reviewed provider price schedule before any paid run.

```sh
export BENCH_PYTHON='<absolute-path-to-existing-python>'
export BENCH_MODEL='claude-sonnet-5-5'
export BENCH_BUDGET_USD='<explicit-total-USD-cap>'
export BENCH_TRIAL_BUDGET_USD='<explicit-per-trial-USD-cap>'
export BENCH_REQUEST_RESERVE_USD='<conservative-per-request-USD-reserve>'
export BENCH_INPUT_USD_PER_MILLION='<worst-case-input-rate-including-cache-write-premiums>'
export BENCH_OUTPUT_USD_PER_MILLION='<worst-case-output-rate>'
export BENCH_MAX_INPUT_BYTES='<positive-integer>'
export BENCH_MAX_OUTPUT_TOKENS='<positive-integer>'
export BENCH_SERVICE_TIER=default
export BENCH_SPEND_FILE='<absolute-path-to-durable-shared-ledger.json>'
export BENCH_LOG='<absolute-path-to-usage-only.jsonl>'
export BENCH_AUTO_CACHE=0
export BENCH_CAPTURE_BODIES=0
"$BENCH_PYTHON" meter_proxy.py
```

The meter binds only `127.0.0.1`; `BENCH_PORT` defaults to `8787`.
`BENCH_UPSTREAM` defaults to `https://openrouter.ai/api`. Do not point it at an
untrusted destination: it receives the environment credential.
`BENCH_PIN_MODEL=1` is an explicit optional mutation, but does not permit
unpriced side models. Only the configured model is accepted either way.
`BENCH_AUTO_CACHE=0` preserves incoming cache markers exactly. Opt-in `=1`
adds top-level ephemeral caching only when no marker is present and records
`auto_cache: true`; freeze and label that changed arm separately.
`BENCH_CAPTURE_BODIES=1` is a separate opt-in that writes full transcripts to
`BENCH_BODIES`: treat them as sensitive; they are not sanitized public artifacts.

Before dispatch, the request must fit the configured model, default tier,
input-byte limit, output-token limit, and text/custom-tool policy. Extensions,
server tools, and non-text modalities are rejected rather than guessed/priced.
The minimum reservation bound is:

```
reserve_USD >= ((4 * max_input_bytes + 65536) * input_USD_per_million
                + max_output_tokens * output_USD_per_million) / 1_000_000
```

The byte multiplier and framing slack intentionally over-reserve text input;
this is not a tokenizer estimate. Explicit provider rate ceilings are also sent
in `provider.max_price`, with Anthropic routing and fallbacks disabled. There
is no implicit billing tier or price table. **The operator must verify that the
configured rates cover cache writes and all applicable provider surcharges.**
This is client-side enforcement, not a provider-side account spending limit.
It covers only model requests routed through this meter. Local Operator's
Radient classifier path is outside this accounting; the coordinating runner
must reject paid Local Operator trials until that separate path has a validated
bounded reservation. Do not disable Radient to produce a cheaper comparison.

The ledger is protected by an OS file lock, atomically replaced, and shared
across trials. A reservation must fit both total and trial headroom. Requests
are serialized: an outstanding reservation blocks further dispatch, including
from another proxy sharing the ledger. Missing/nonfinite/negative cost,
transport failure, interruption, or malformed state cannot reset spend to zero.
Unknown billing retains the reservation and blocks every later trial until the
operator reconciles against provider billing. Do not delete/reset the ledger
between trials. Reuse each logical trial's ID for all retries and side calls;
changing IDs would defeat the per-trial grouping (the total cap still applies).
`@current` reads the explicitly configured `BENCH_CURRENT_RUN_FILE`, needed by
a strictly sequential translator whose upstream URL cannot vary per trial.
Only POST `v1/messages` is priced/forwarded; other billable endpoints are denied.

For the bridge, set `BRIDGE_UPSTREAM` to the selected local translator and
`BRIDGE_LOG` to a usage/count-only log, then invoke:

```sh
"$BENCH_PYTHON" codex_toolsearch_bridge.py '<local-port>'
```

For an already installed LiteLLM, supply an explicit interpreter and isolated
config file with this structure. The placeholder key is replaced by the meter:

```yaml
model_list:
  - model_name: claude-sonnet-5-5
    litellm_params:
      model: anthropic/claude-sonnet-5-5
      api_base: http://127.0.0.1:8787/r/@current
      api_key: benchmark-dummy
```

```sh
export BENCH_CURRENT_RUN_FILE='<absolute-path-to-current-trial-id-file>'
# Set this before starting the meter. Write one stable trial ID before each
# sequential trial, never while a translated request remains outstanding.
export LITELLM_PYTHON='<absolute-path-to-existing-litellm-python>'
export LITELLM_CONFIG='<absolute-path-to-isolated-config.yaml>'
"$LITELLM_PYTHON" -m litellm --config "$LITELLM_CONFIG" --port 4000
# In a separate process, with the explicitly chosen bridge interpreter:
BRIDGE_UPSTREAM=http://127.0.0.1:4000 "$BENCH_PYTHON" codex_toolsearch_bridge.py 4001
```

Point an isolated Codex config at the bridge, not the public provider directly:

```toml
model = "claude-sonnet-5-5"
model_provider = "benchmark"
model_catalog_json = "<absolute-path-to-experimental/codex-model-catalog.json>"
[model_providers.benchmark]
name = "Benchmark translator"
base_url = "http://127.0.0.1:4001/v1"
wire_api = "responses"
env_key = "BENCH_DUMMY_API_KEY"
```

Select `CODEX_HOME` and the Codex executable explicitly in the runner, and set
`BENCH_DUMMY_API_KEY=benchmark-dummy`. The runner owns MCP/public-loadout config;
this example deliberately does not duplicate those schemas. These translator
examples describe the intended wiring, not verified installed CLI compatibility:
LiteLLM was absent and no process above was started.

This rewrites client-side tool-search requests and streamed search results;
it is a separately labelled translator-bridge arm, not stock Codex. The
non-streaming response pass-through remains as in the private source and is
not a validated tool-search response path. No translator or Codex process was
started here.

## Cheap checks and provenance

Run only the bounded offline tests with existing dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 "$BENCH_PYTHON" -m unittest discover \
  -s . -p test_adapters.py -v
```

`LOCAL_OPERATOR_SCRATCHPAD` must identify an existing scratch directory for test
fixtures. Tests use ASGI and mock transports: no listening socket or upstream
network connection. They cover route forwarding, JSON/SSE usage settlement,
trial and total rejection, unknown cost, malformed/inflight ledgers, missing
configuration/reserve, conservative-bound rejection, unsupported model/tier/
modality/server-tool rejection, no auth/body content in default logs, and
catalog/bridge transformations. These checks are source-only, not native
harness, provider-price, provider-billing, or end-to-end integration proof.

Source-to-destination audit:

| Source basename | Destination | Intentional delta |
| --- | --- | --- |
| `codex_toolsearch_bridge.py` | same | Explicit interpreter usage example; transformation retained |
| `codex_catalog.py` | same | Unchanged |
| `codex-model-catalog*.json` | same | Unchanged, including embedded prompt bytes |
| `lop_launch.py` | same | Explicit interpreter/root/edge env, released v0.94.22 identity and clean-tree checks; shim removed |
| `lop_edge.py` | new | Per-run CONNECT/TLS edge: exact routes, host-held credentials, shared ledger, sandbox profile |
| `radient_meter/` | public-port | Byte-identical reviewed decisions-billing qualification (not modified) |
| `meter_proxy.py` | same | Env-only key; durable dual budgets/rate bounds; default-off cache/body mutations; sanitized error logs |

The catalogs retain the upstream Codex prompt verbatim, verified against tagged
upstream source. Its SHA-256 is recorded in `THIRD_PARTY_NOTICES.md`; Apache-2.0
license and upstream NOTICE are preserved in `licenses/`. No private home paths,
private config pointers, authdb imports, or private runner imports are needed.
