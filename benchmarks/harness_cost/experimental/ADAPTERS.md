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
It does not update, rebuild, or otherwise mutate any install. There is no
endpoint shim: lowercase/uppercase proxy variables must agree, `NO_PROXY` may
exempt only loopback, and `ALL_PROXY` is refused.

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
export BENCH_TRIAL_BUDGET_USD='5'   # default; >= 2 for a pilot spot-check, recorded per run
export BENCH_RATE_CARD='<absolute-path-to-rate-card.json>'  # built at run start, see below
export BENCH_MAX_INPUT_BYTES='<positive-integer>'
export BENCH_MAX_OUTPUT_TOKENS='128000'   # must admit LoP's native main cap
export BENCH_ADMISSION_WAIT_S='<bounded-queue-seconds>'
export BENCH_MAX_REQUEST_S='<longest-legitimate-request-seconds>'
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
unpriced side models.

**Rate card.** The meter prices exactly `NATIVE_MODELS` — every model a
benchmarked harness natively requests in the wire captures
(`anthropic/claude-sonnet-5.5` main turns for all seven;
`anthropic/claude-haiku-5.5` helpers for omp and OpenCode; Claude Code's
captured rows are all Sonnet). At run
start fetch the live `https://openrouter.ai/api/v1/models` listing and build
the card with `meter_proxy.rate_card_from_models_listing(raw, source_url=…,
fetched_at=…)`; it keeps the response SHA-256 and fetch time as provenance and
is refused when stale (`RATE_CARD_MAX_AGE_S`), partial or non-finite. Any other
model (e.g. a native failover to Opus) is refused before dispatch **and voids
the run** with that reason; nothing is silently measured on a demoted path.
`BENCH_AUTO_CACHE=0` preserves incoming cache markers exactly. Opt-in `=1`
adds top-level ephemeral caching only when no marker is present and records
`auto_cache: true`; freeze and label that changed arm separately.
`BENCH_CAPTURE_BODIES=1` is a separate opt-in that writes full transcripts to
`BENCH_BODIES`: treat them as sensitive; they are not sanitized public artifacts.

Before dispatch, the request must fit the configured model, default tier,
input-byte limit, output-token limit, and text/custom-tool policy. Extensions,
server tools, and non-text modalities are rejected rather than guessed/priced.
Each request reserves its OWN upper bound, computed from that request:

```
reserve_USD = ((body_bytes + FRAMING_TOKENS) * max(prompt, cache_write)_USD_per_million
               + max_tokens * completion_USD_per_million) / 1_000_000
```

UTF-8 bytes overbound text tokens and `FRAMING_TOKENS` (8192) covers framing;
it is not a tokenizer estimate, and any bill above the reservation latches at
settle. `max_tokens` is never lowered. Reservations gate only the **total**
campaign budget (settled spend plus every pending reservation), so the total
allowance is never overspent.

**The per-trial cap is decided on settled actual spend**, not on reservations:
a request is admitted whenever the trial's settled spend is below the cap,
whatever its native `max_tokens`. A reservation-gated trial cap would bind at
`cap − reserve(next request)` — about $3.7 of $5 for the six 128k-output
harnesses but $4.6 for Claude Code (32k) — which is not the same limit for all
harnesses (PREREGISTRATION §9.1). The last admitted request may overshoot the
cap; the overshoot is recorded (`cost_caps[<run_id>]`: settled USD and cap) and
the trial ends as a **primary cost-cap failure**: cost counted, Y=0, never a
void or rerun (§9.2). The cap is a planning/abort control, not a price, and is
identical for all seven harnesses and recorded with each run: default $5 (long
Terminal-Bench tasks make many calls), $2 minimum for a pilot spot-check.

**The preregistered caps are infeasible for long tasks.** §9.1's $0.30
(OSWorld) and $0.45 (Terminal-Bench) per trial are below what a single native
128k main turn can bill and below observed long-task spend, so most long trials
would end as cost-capped truncations (§9.1 itself says to replan then). The
cap for the 28-task cohort must be set prospectively in the amendment draft
(proposal recorded there: $5/trial, not frozen), before any holdout outcome.
Explicit provider rate ceilings
are also sent in `provider.max_price`, with Anthropic routing and fallbacks
disabled. This is client-side enforcement, not a provider-side account
spending limit. Local Operator's Radient classifier spend is metered by
`lop_edge.py` on the same ledger (`classifier` channel).

The ledger is protected by an OS file lock, atomically replaced, and shared
across trials and harnesses. Admission sums settled spend plus every pending
reservation and waits (bounded by `BENCH_ADMISSION_WAIT_S`) when only in-flight
exposure stands in the way. Two kinds of failure are recorded:

* `voids[<run_id>]` — an infrastructure or metering defect: the attempt is
  invalid and may be rerun at most once (§9.2). Causes: a provider 5xx/429
  (incl. a 529 overload, even when billed), transport failure, a request the
  meter cannot price (unpriced model or extension), a campaign-budget or
  admission-deadline refusal. A void blocks only that run. A harness-caused
  4xx (e.g. "prompt is too long") is **not** a void: its stated bill (often 0)
  settles, the trial continues, and the outcome is the harness's own.
* `cost_caps[<run_id>]` — the trial's settled spend reached the per-trial cap:
  a primary cost-cap failure with its full cost, not a void.
* `blocked` — spend may be unaccounted for; every later admission on the
  shared ledger refuses until the operator reconciles: missing/nonfinite/
  negative cost, a bill above its reservation, transport failure, and any
  reservation that never settles. Settle-or-latch runs on every exit path
  including cancellation; the guard latches its own unsettled reservations at
  teardown; and admission latches when a pending reservation's owner process
  is dead or the reservation is older than `BENCH_MAX_REQUEST_S`.

**Unbilled responses settle from their own tokens (S10).** A response that states
no `usage.cost` is not automatically unknown spend. When it carries the counts
the bill is made of — `input_tokens` and `output_tokens` both present as
nonnegative integers, plus any `cache_read_input_tokens` or
`cache_creation_input_tokens` it reports — the meter settles the charge at the
rate card frozen for that run and records `settled_from_tokens: true` with
`cost_derivation: {tokens, rates_usd_per_m, usd}` in its request row, the rate
map keyed by the same token fields it priced (`input_tokens`, `output_tokens`,
…). The card
carries every rate such a settlement can be made of — `prompt`, `completion`,
`input_cache_write` and `input_cache_read` — mapped from the same live listing; a
listing that omits any of them is a startup configuration error, never a silent
zero (an omitted cache-write rate priced the pilot's refusal at $0.000004).

WHY: the provider bills the input and the cache write whatever the output
turns out to be, so a response with zero output is charged. The DEV pilot
(255f6035) refused on its first call with 27166 input tokens (27164 of them a
cache write) and 0 output; at that run's card the charge is $0.067914. Treating
it as unknown latched the whole campaign on that one call and blocked every
later admission. Why this is arithmetic and not a guess:

* Every term comes from the card the run was frozen under, and the row records
  both the counts and the rates used, so the amount is recomputable from the row.
* A count the card cannot price does **not** settle: a 1-hour cache write, which
  bills at double the 5-minute rate the card carries. A card missing any rate is
  refused when it is loaded, so a card frozen before S10 must be refetched. A
  breakdown that disagrees with its top-level count does not settle either.
* A missing `input_tokens`/`output_tokens` count does not settle: partial usage
  is not a bill. Neither does a response carrying any `unpriced_usage` field
  (F23). Both keep the pre-S10 latch exactly.
* A response that ended before completion never settles: a truncated stream or
  a cancelled request has partial usage, which is a floor rather than the
  charge, so that row is written `incomplete: true` and the ledger latches as it
  did before S10. A `usage.cost` the provider printed beside that partial usage
  is a floor too: the row keeps it as evidence and the ledger does not settle it
  (R18 m4 — settling it booked $0.075 for a stream no reader accepted).
* Nothing reads `stop_reason`: every unbilled, complete response follows the same
  rule, so a refusal, an ordinary stop and a harness-caused 4xx whose body
  happens to carry counts are treated alike.

A refusal remains a primary outcome, never an infrastructure void: the attempt
is not rerun, its native reward (0) stands, and its cost and wall time are
retained (PREREGISTRATION §9.2). A stated `usage.cost` is still used unchanged
wherever a complete response states one — derivation only fills a gap, it never
overrides a bill. An `incomplete` row is the exception in every reader (the
OSWorld lane, the coding driver and the publishing rig refuse it identically):
it is never a number, never zero and never `cost_complete: true`, whatever it
states. The Local Operator classifier channel is unchanged:
`lop_edge.py` settles it only from the reviewed top-level billed `cost` of a
classifier response.

**A request the provider rejects at validation costs $0, it is not an unknown
charge (S11, narrowed in S12).** A provider that refuses to infer on a request
answers 4xx and states no bill: the DEV pilot measured HTTP 400
`invalid_request_error` ("This model does not support assistant message
prefilling") with `usage: {}` in 0.20 s — no generation ran, so no charge exists.
The meter settles such a response at **exactly $0**, records it as
`settled_no_charge: true` with `upstream_status` and the provider's own
`upstream_error_type` (and `upstream_error_code` where it sends one) copied
verbatim — publication replaces both with a reviewed classifier or an
`unclassified:<sha256 prefix>` digest, so no provider string is published as
text and the verbatim value stays in this private row — and does **not** latch
the ledger: the attempt is metered at $0 and the
trial continues. The three money readers (`runner_v2.stated_amount`,
`coding.driver.row_cost`, and the publishing rig's `coding/billing.stated_amount`)
read that marker as a measured zero and refuse a row that claims it while stating
a bill basis, naming a status outside the settled set or omitting the upstream
error.

**The settled set is 400 and 422, and nothing wider (S12, R19 M1).** S11 accepted
any 4xx carrying a typed error body, which also settled a 401
`authentication_error`, a 402 `billing_error` and a 403 `permission_error` at $0
*without* latching, and a 429 at $0 *and* as a rerun void. Those are the
campaign's environment failing, not a response to what a harness sent: a key
revoked or a credit balance exhausted mid-run would have driven every remaining
trial to a $0 ledger published as `cost_complete: true`, where pre-S11 the first
one latched it. So 401, 402, 403 and 429 are infrastructure — they latch and
void for one rerun exactly as a 5xx does, and never settle. Only a 400 or a 422,
which is a request the provider rejected at validation, earns the marker;
everything else in the 4xx family keeps the pre-S11 latch.

The rule is shaped by the *response*, never by a message string or a harness, so
every arm gets it identically. Its bounds, all of which keep the unknown-spend
latch or the rerun void:

* any 2xx that states no cost and no priceable token counts (S10's rule, above);
* any 400/422 that **does** state a cost or token counts — it is priced, not
  zeroed (a stated cost settles as stated; counts settle from their derivation);
* a 401, 402, 403 or 429 **that states no bill**: infrastructure, so the attempt
  is voided for one rerun and latched, and it never settles at $0 (S12; pre-S11
  behaviour restored). One that **does** state a cost or token counts is priced
  as stated and voided for one rerun, but **not** latched: the classification
  decides the void and the amount decides the latch, because a bill the meter can
  read is not unknown spend and blocking on it would end the campaign over a
  charge that was measured (S13, R20 MINOR 2);
* a 400/422 whose usage carries an `unpriced_usage` field, or a token count the
  frozen card cannot price (e.g. a 1-hour cache write): the bill is not whole;
* a 5xx (including 529): infrastructure, so the attempt is voided for one rerun —
  latched like the four statuses above when it states no bill, priced at the
  stated cost without latching when it states one it can read;
* any other 4xx (404, 413, …): neither a validation rejection nor an
  infrastructure class, so it keeps the latch;
* a response whose error body the meter cannot read — not JSON, no `error`
  object, or no error `type` to copy — is not evidence of a zero bill;
* a 4xx delivered as an event stream carries no JSON body for the row to copy,
  so it keeps the latch.

The rule is applied in `write_row`, which this meter reaches only for a
`POST …/v1/messages` (every other path is refused before any dispatch), so no
discovery-route row is produced here at all (S12, R19 N2; figures corrected in
S13 from R20 MINOR 1: the 37 typed-404 rows for `version`, `props`, `v1/props`,
`models`, `api/v1/models`, `api/tags` and `api/show` in `logs/requests.jsonl`
carry **none** of the fields `write_row` always writes — `channel`,
`priced_model`, `reserve_usd`, `in_flight_at_admission`, `n_messages` — across
five distinct run ids, only 7 of which are `probe`, so they were logged outside
the meter rather than by a proxied request, and under this rule a 404 latches
rather than settling).

Publication-wise this is honest by construction: a rejected request contributes
**$0** to its trial's settled cost, and the trial's quality outcome is whatever
native grading says — a model that refuses, or a harness whose adapter re-sends
what the provider will not accept, is a graded failure, not an infrastructure
void and not free work. The pilot's codex-best cell is that shape (see
[coding/README.md](coding/README.md)).

**A request is never forwarded with its conversation ending on an assistant turn
(S12).** After a refusal, Codex's adapter re-sent the transcript ending on the
assistant turn and the provider rejected the whole request — the 400 above — so
that trial died while the other arms went on and recovered. That truncated one
arm's quality ceiling by its retry shape, and because refusal rate is
task-dependent the penalty landed unevenly across the task set; identical
treatment across arms cannot absorb an asymmetry that is itself task-dependent.
`_patch_body` — the shaping path every arm's `v1/messages` body passes through —
therefore appends one minimal user continuation turn (`PREFILL_CONTINUATION`,
the same string for every harness: `"Continue."`) whenever the outgoing
`messages` would end on an assistant turn, and records `prefill_continuation:
true` on the row. Nothing else is shaped: tools, system blocks, prior turns, the
model pin and the cache markers are untouched, and no arm has a branch of its
own. A body already ending on a user turn is forwarded byte-for-byte unchanged.

Every configuration error fails at startup (one guard per meter/edge process),
never per request. A ledger written before `models`/`voids` existed reads as
invalid (fail closed); start a fresh ledger file for this schema. Do not
delete/reset the ledger between trials. Reuse each logical trial's ID for all
retries and side calls; changing IDs would defeat the per-trial grouping (the
total cap still applies). `@current` reads the explicitly configured
`BENCH_CURRENT_RUN_FILE`, needed by a strictly sequential translator whose
upstream URL cannot vary per trial.
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
