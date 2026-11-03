# LoP edge metering — verification (S1–S5), 2026-11-03

Spec: `METERING-ARCHITECTURE.md` (radient-dev handoff), slices S1–S5, against
released Local Operator v0.94.22 = `500883ef56ca504325ffd80c5b90bf2435d00e3f`.

**No paid or external model calls were made.** All credentials in tests are
fake literals; the operator's Radient token and OpenRouter key were not
resolved. No Docker, no browsers, no auth-store reads.

## What is NOT verified here (read this first)

* **S4 (zero-paid full-path `lop exec` gate): NOT RUN — blocked on disk.**
  `df -m /System/Volumes/Data` showed 13,782 MB free at start (17,220 MB at the
  end), below the coordinator's 15 GiB hard floor and the brief's 17 GiB
  build floor, so no clean v0.94.22 checkout/venv
  (`<repo>/.worktrees/lop-v0.94.22`) was built. None of the S4 acceptance
  items (oracle-vs-edge 1:1 parity, title/fit/resource decision rows,
  `max_tokens` 1024/128000 rows, `ps -E` hygiene, failover/fork/title-demote
  variants) has evidence yet.
* **S1 child probe** (`local_operator.__file__` under the pinned root;
  `clients.ANTHROPIC_API_URL == "https://api.anthropic.com"` with no
  `sitecustomize`) needs that same environment: not run.
* **S5 real-path half** (each control through a real `lop exec` under the
  sandbox, with fixture hit counts) is S4-dependent: not run. The S5 controls
  below are exercised against the real edge over real loopback sockets, TLS
  and `sandbox-exec`, with in-memory upstream fixtures — they prove the edge
  and sandbox fail closed, not that the product reaches them on every path.
* The unpriced `GET` metadata allowlist is intentionally empty (every read is
  a non-latching 403) until the S4 capture lists the reviewed set.
* S6 (runner integration) is out of scope: `execute()` still refuses paid
  LoP, and the edge is not wired into `runner_v2.execute`.

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
| S1 | clean v0.94.22 env; child-probe `__file__`/`ANTHROPIC_API_URL` | — | **blocked (disk)** |
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
| S4 | full-path `lop exec` gate, all §4 S4 items | — | **blocked (disk)** |

## Provenance of vendored code

`radient_meter/{classifier,json_types,model,sse_stream}.py` are byte-identical
copies of the reviewed public-port sources (sha256 equal to its `SHA256SUMS`):
`classifier.py e8aebf5e…`, `json_types.py 5e833ea6…`, `model.py d35963fd…`,
`sse_stream.py 6be4440a…`. They are not modified here.
