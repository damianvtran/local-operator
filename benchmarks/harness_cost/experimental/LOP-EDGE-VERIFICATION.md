# LoP edge metering — verification (S1–S5), 2026-11-03

Spec: `METERING-ARCHITECTURE.md` (radient-dev handoff), slices S1–S5, against
released Local Operator v0.94.22 = `500883ef56ca504325ffd80c5b90bf2435d00e3f`.

**No paid or external model calls were made.** All credentials in tests are
fake literals; the operator's Radient token and OpenRouter key were not
resolved. No Docker, no browsers, no auth-store reads.

## What is NOT verified here (read this first)

* **No paid run and no real upstream.** S4/S5 below ran the real released
  product (`lop exec`, clean detached `v0.94.22` checkout, own `.venv`) through
  the real edge and meter, but every upstream was a loopback fixture with
  synthetic billing. Live pricing (`/api/v1/models` rate card) and real
  Radient/OpenRouter billing are first exercised by the paid pilot.
* **S6 (runner integration) is out of scope**: `execute()` still refuses paid
  LoP, and the edge is not wired into `runner_v2.execute`. S6 TODO (D4): the
  runner must enforce a wall-clock limit on the sandboxed `lop exec` and reap
  its process group by exact pid — with egress blocked, `lop exec` keeps
  retrying (observed > 600 s) instead of exiting.
* The unpriced `GET` metadata allowlist stays **empty**: the only reads seen
  (10× each across S4 runs) were `GET api.anthropic.com/v1/models?limit=1000`
  and the tiktoken encodings fetch; both are denied without latching and LoP
  degrades natively (chars/4 estimate, built-in model metadata).
* The Harbor coding path (`coding/`) has no edge, so its LoP launcher refuses.

## S4/S5 real-path summary (2026-11-03)

Full commands, outputs and the S1 child probe are in the scratch evidence file
`LOP-EDGE-S4-EVIDENCE.md` (kept out of the repo so the reviewed head stayed
fixed): sha256 `e0f5f532…c55e0ac` (`e0f5f53201d2ca43fea53d0d0490ba50b6a9b37a6e0eb3398e0d8a160c55e0ac`).
Rig hashes: `rig.py` (round 0) `a946dde1…`, `oracle_child.py` `6d6a86dc…`,
`check.py` `9b813885…`. S1 child probe: `local_operator.__file__` under
`~/workspace/repos/lo-harness-cost/.worktrees/lop-v0.94.22`, HEAD `500883ef…`
clean, `ANTHROPIC_API_URL == "https://api.anthropic.com"`, no `sitecustomize`.

| run | rc | oracle | edge rows | oracle∖edge | edge∖oracle | fixture hits | sandbox denials | ledger blocked | replay after finish |
|---|---|---|---|---|---|---|---|---|---|
| native | 0 | 7 | 7 | 0 | 0 | 5 | 0 | no | 407 |
| failover (round 0) | 1 | 8 | 8 | 0 | 0 | 4 | 0 | yes | 407 |
| failover-billed | 1 | 17 | 17 | 0 | 0 | 14 | 0 | no | 407 |
| title-demote | 0 | 8 | 8 | 0 | 0 | 5 | 0 | no | 407 |
| fork | 0 | 8 | 8 | 0 | 0 | 6 | 0 | no | 407 |
| tool-bypass (curl `--noproxy`, urllib) | 0 | 7 | 7 | 0 | 0 | 5 | 2 (EPERM) | no | 407 |
| typesafe leg forced | 1 | 7 | 7 | 0 | 0 | 0 | 0 | yes | 407 |
| diagnostics 400 | 1 | 7 | 7 | 0 | 0 | 4 | 0 | yes | 407 |
| decisions without `cost` | 1 | 7 | 7 | 0 | 0 | 2 | 0 | yes | 407 |
| decisions 401 | 1 | 8 | 8 | 0 | 0 | 2 | 0 | yes | 407 |
| no proxy env (launcher) | 1 | 0 | 0 | 0 | 0 | 0 | 0 | — | 407 |
| wrong nonce | 1 | 15 | 0 (407) | 15 | 0 | 0 | 0 | — | 407 |
| no proxy env, launcher bypassed | killed | 29 | 0 | 29 | 0 | 0 | 21 (EPERM) | — | 407 |

Direct TLS with the run-only CA bundle (LoP's httpx and `requests`, loopback
host not signed by the run CA): `CERTIFICATE_VERIFY_FAILED` for both. Token
hygiene: fake credentials and their hash prefixes in no artifact; `ps -E`
shows only the dummy `RADIENT_API_KEY`.

Defects found by S4 and fixed in remediation round 1: D1 flat worst-case
reservation ($4.07/request) admitted nothing under a $1 trial cap; D2 a native
failover to an unpriced model (Opus) and an unbilled 529 left the run looking
valid; D3 refusing an unpriced model (Haiku 4.5 title tier) did not void the
run. **Round 1 re-run** (rig `d47b7449…`, R1 meter contract, $5 trial cap):
`native` rc 0, oracle 7 = edge 7, fixture 5/5, 0 denials, ledger clean, no
voids, per-request reserves title $0.0325 / main $1.4675 / $1.4685;
`failover` rc 1, oracle 9 = edge 9, Opus refused before dispatch, run voided
with `upstream status 529`, `unbilled or unknown charge`, `refused request:
unpriced model anthropic/claude-opus-5.5`. Reserve repro (`repro_reserve_r1.py`
`2bf900aa…`, LoP's captured S4 bodies, Sonnet 5.5 list rates): trial cap $1 →
title admitted, main ($1.4611) refused and the run voided; $1.50 (lowest tested
cap that admits both) / $2 / $5 → title + main + a second main admitted.

## Commands

Isolated runner used for every test (scratch `HOME`, no keychain-capable git
config, no inherited credentials):

```sh
env -i HOME="$ISO" PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="$SCRATCH" \
  LOCAL_OPERATOR_SCRATCHPAD="$SCRATCH" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" \
  GIT_CONFIG_SYSTEM=/dev/null GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 \
  GIT_ASKPASS=/bin/false PYTHONDONTWRITEBYTECODE=1 \
  "$BENCH_PYTHON" -m unittest -v test_adapters test_lop_edge test_runner_v2 \
    test_cache_namespace test_cache_forwarder test_bridge_sse test_clone_path
(cd coding && ... -m unittest test_guard test_driver_safety)   # same isolation
(cd .. && ... -m unittest test_kit)                            # same isolation
python3 -B freeze_inputs.py; (cd .. && python3 -B publication.py verify)
```

Results at the committed head: experimental `Ran 95 tests in 5.050s — OK`
(25 new in `test_lop_edge`; 12 new/rewritten in `test_adapters` and
`test_runner_v2`); coding `Ran 16 tests — OK`; retained core `test_kit`
`Ran 16 tests — OK`; `freeze_inputs.py` "Experimental and shared input hashes
match"; `publication.py verify`/`scan` `"status": "PASS", "files_scanned": 133`.
Every new behaviour was observed RED first (S2: 3 failures + 3 errors — 402 on
`diagnostics`, cap not enforced, concurrent request refused, no
`try_acquire`/`channel`; S3: `lop_edge.py` missing; S1: old pin, shim present,
launcher demanding `BENCH_ANTHROPIC_URL`; runner: old `LOP_HEAD`,
`BENCH_ANTHROPIC_URL` in env).

Lint/types on all changed files: `flake8` → exit 0; `pyright` 1.1.414 →
`0 errors, 0 warnings, 0 informations`.

Launcher refusals on REAL git trees (no CLI import reached):

```text
== root HEAD 07c324e9f61   (lo-thinking-replay worktree)
benchmark source is not reviewed revision 500883ef56ca504325ffd80c5b90bf2435d00e3f   rc=1
== root HEAD 5906b5564fd   (bench/harness-cost checkout)
benchmark source is not reviewed revision 500883ef56ca504325ffd80c5b90bf2435d00e3f   rc=1
== no proxy env
HTTPS_PROXY must name this run's loopback metering edge                             rc=1
```

## Results by slice

| Slice | Item | Evidence | Result |
|---|---|---|---|
| S1 | pin → `500883ef…` (full SHA) in launcher + runner | `test_launcher_refuses_unreleased_head_by_full_sha`, `test_lop_is_pinned_to_released_v0_94_22`, real-tree run above | pass |
| S1 | refuses 07c324e9 | real `lo-thinking-replay` tree, rc=1 | pass |
| S1 | refuses dirty tracked AND untracked files (stray `sitecustomize.py`) | `test_launcher_refuses_dirty_or_untracked_tree_in_a_real_repository` (real `git init` repo) | pass |
| S1 | `lop_shim` + `BENCH_ANTHROPIC_URL` removed; CLI run without shim | `test_launcher_runs_the_unmodified_cli_without_any_shim`, `test_source_launch_environment_is_explicit` | pass |
| S1 | launcher refuses without loopback edge proxy/CA/NO_PROXY env | `test_launcher_requires_the_edge_proxy_environment` | pass |
| S1 | clean v0.94.22 env; child-probe `__file__`/`ANTHROPIC_API_URL` | S4 evidence file | pass |
| S2a | `diagnostics` forwarded unchanged, price-neutral | `test_diagnostics_is_price_neutral_and_forwarded_unchanged`, edge `test_model_route_…` | pass |
| S2b | two concurrent admissions both dispatch & settle; third waits → 402 at deadline | `test_concurrent_admissions_queue_against_pending_exposure` | pass |
| S2b | a waiting admission dispatches once exposure settles | `test_waiting_admission_dispatches_once_exposure_settles` | pass |
| S2c | `BENCH_MAX_OUTPUT_TOKENS` < 128000 refused; 128000 admitted | `test_main_output_cap_below_native_128000_is_refused` | pass |
| S2d | one ledger, `channel` model/classifier | `test_model_rows_carry_the_model_channel`, edge decisions test (`channels.classifier`) | pass |
| S2 | missing / NaN / negative / string / bool cost latches | `test_unknown_costs_latch_the_ledger` | pass |
| S2 | classifier with only informational `usage.cost` latches | `test_classifier_informational_usage_cost_alone_latches` | pass |
| S2 | unqualified contract: bill recorded, ledger latched | `test_classifier_unqualified_contract_records_bill_and_latches` | pass |
| S3 | CONNECT MITM, exact model route → shared meter ASGI app; inbound auth stripped; OpenRouter key injected by meter | `test_model_route_reaches_meter_without_child_credentials` | pass |
| S3 | decisions route: host token injected, exact URL, settles classifier channel, usage id hashed | `test_decisions_route_injects_host_token_and_settles_classifier_channel` | pass |
| S3 | only the CA certificate on disk (no key) | `test_only_the_ca_certificate_reaches_disk`, entrypoint test | pass |
| S3 | child env: proxy, CA bundle, `NO_PROXY`, dummy `RADIENT_API_KEY`, no real credential | `test_child_environment_carries_proxy_ca_and_dummy_key_only` | pass |
| S3 | logs never contain headers/credentials/nonce | `test_logs_never_contain_headers_or_credentials` | pass |
| S3 | default httpx + `requests` clients in a SANDBOXED CHILD process route via edge | `test_default_clients_in_a_sandboxed_child_route_through_the_edge` | pass |
| S3 | host process entrypoint (token on stdin, env announced, exits on EOF) | `test_stdin_entrypoint_announces_child_env_and_exits_on_close` | pass |
| S5 | no `HTTPS_PROXY` → EPERM, 0 hits | `test_child_without_proxy_environment_gets_eperm_and_zero_hits` (real child, sandbox) | pass (edge/sandbox level) |
| S5 | `trust_env=False` client inside sandbox → EPERM | `test_trust_env_false_client_gets_eperm_even_with_proxy_env` | pass |
| S5 | wrong nonce / wrong run id → 407, 0 hits; missing auth → 407 | `test_wrong_nonce_is_407_with_zero_fixture_hits`, `test_missing_proxy_authorization_is_407` | pass |
| S5 | replayed nonce of finished run → 407 | `test_replayed_nonce_of_finished_run_is_407` | pass |
| S5 | direct TLS with run-only CA bundle → certificate failure | `test_run_ca_bundle_refuses_a_server_it_did_not_sign`, `test_leaf_is_valid_only_for_its_host` | pass |
| S5 | POST to unlisted host (typesafe leg) → 403, run latched, later valid routes 402 | `test_unlisted_post_is_denied_and_latches_without_dispatch` | pass |
| S5 | unlisted path/method/query on a routed host → 403 | `test_unlisted_path_and_method_on_routed_host_are_denied` | pass |
| S5 | diagnostics 400 without billing → native retry refused (402), fixture sees exactly 1 | `test_diagnostics_400_without_billing_latches_and_native_retry_is_refused` | pass |
| S5 | decisions missing `cost` → latch | `test_classifier_informational_usage_cost_alone_latches` | pass |
| S5 | decisions 401 → latch; re-resolve retry reaches edge and is refused, 1 upstream hit | `test_classifier_401_latches_and_reresolve_retry_is_refused` | pass |
| S5 | plain-HTTP proxying / CONNECT to non-443 → 403 + latch | `test_plain_http_proxying_is_denied_and_post_latches`, `test_connect_to_non_tls_port_is_denied_and_latches` | pass |
| S5 | sandbox allows only the listed loopback port | `test_only_allowed_loopback_port_connects` | pass |
| S4 | full-path `lop exec` gate, all §4 S4 items | S4/S5 summary above | pass (failover completes only to its void, by design) |

## Remediation round 1 (review `LOP-EDGE-REVIEW-R1.md`)

Each fix observed RED first (68 errors/failures before implementation: missing
rate card, `make_app`, unix-socket allowlist, launcher/coding pins), then:
experimental `Ran 114 tests — OK`; coding `Ran 23 tests — OK`; core `test_kit`
`Ran 16 tests — OK`; `flake8` exit 0; `pyright` `0 errors` on all changed files.

| Finding | Behaviour | Test |
|---|---|---|
| F1 | settle-or-latch on every exit incl. cancellation (meter and edge); stale/orphan backstop; teardown latch | `test_cancelled_request_latches`, `test_non_stream_body_reset_after_dispatch_latches`, `test_orphaned_reservation_of_a_dead_process_refuses_the_next_request`, `test_live_concurrent_owner_is_not_an_orphan`, `test_stale_reservation_latches_and_teardown_latches_unsettled`, `test_edge_close_mid_decision_latches_the_reservation` |
| F2 | lowercase/`ALL_PROXY`/`HTTP_PROXY` conflicts refused; edge env accepted verbatim | `test_launcher_requires_the_edge_proxy_environment`, `test_launcher_accepts_the_edge_child_environment_verbatim` |
| F3 | unix sockets denied unless an exact path is listed | `test_unix_sockets_are_denied_unless_explicitly_allowed` (real `sandbox-exec`: listed → connected, other → errno 1, default → errno 1) |
| F4 | `count_tokens` is an unlisted POST: 403 + latch | `test_count_tokens_is_an_unlisted_post_that_latches` |
| F5 | any admission refusal voids the run with its reason | `test_decisions_admission_refusal_voids_the_run`, `test_concurrent_admissions_queue_against_pending_exposure`, `test_total_cap_independent_of_trial_ids` |
| F6 | one guard built at startup; bad env fails construction/startup | `test_configuration_errors_fail_at_startup`, `test_misconfigured_meter_environment_fails_at_construction`, entrypoint test |
| D1 | per-request reservation from the request's own bytes and `max_tokens` | `test_reservation_is_per_request_not_a_flat_worst_case` + reserve repro above |
| D2/D3 | rate card = exact native allowlist with provenance; other models refused + void; provider 5xx/429 voids (R2: harness 4xx counted); unbilled latches | `test_rate_card_is_the_exact_native_allowlist_with_provenance`, `test_haiku_is_priced_from_the_card_and_recorded_per_model`, `test_unpriced_model_is_refused_and_voids_the_run`, `test_upstream_overload_voids_the_run_even_when_billed`, `test_unbilled_overload_latches_with_reason` |
| coding | Harbor LoP path repinned to `500883ef`; shim deleted; launcher refuses | `test_coding_lop_path_has_no_endpoint_shim_and_refuses_the_candidate` |

## Remediation round 2 (review `LOP-EDGE-REVIEW-R2.md`)

| Finding | Behaviour | Test |
|---|---|---|
| F7 | per-trial cap on SETTLED spend (identical for every native `max_tokens`); reservations gate only the total budget; cap stop recorded in `cost_caps` as a primary cost-cap failure, never a void | `test_trial_cap_is_enforced_on_settled_spend_not_native_max_tokens` (reviewer P7: $5 cap, $3.70 settled, 60 kB; 32k reserve $0.736 and 128k reserve $2.176 both admitted), `test_cost_capped_trial_is_refused_without_voiding_and_others_continue`, `test_real_asgi_route_settles_and_stops_trial` |
| F8 | only provider 429/5xx (and transport) void; a harness-caused 4xx settles its stated bill and the trial continues; an unbilled 4xx still latches, without a rerun void | `test_provider_429_and_5xx_void_for_rerun`, `test_harness_caused_4xx_is_counted_and_the_trial_continues`, `test_harness_caused_4xx_without_a_bill_latches_but_is_not_a_rerun` |
| N4 | Haiku provenance: omp and OpenCode only (Claude Code 203/203 Sonnet) | docs/comment |

## Provenance of vendored code

`radient_meter/{classifier,json_types,model,sse_stream}.py` are byte-identical
copies of the reviewed public-port sources (sha256 equal to its `SHA256SUMS`):
`classifier.py e8aebf5e…`, `json_types.py 5e833ea6…`, `model.py d35963fd…`,
`sse_stream.py 6be4440a…`. They are not modified here.
