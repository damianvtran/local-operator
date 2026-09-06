# Adapter-owned public answers — offline verification

## Contract

Adapter RPC 1.6 adds `AdapterCapabilities.ask_user_answer_owner` (`host` by
default, or explicitly `adapter`) and a request-bound public answer in
`AskUserExchangeResult`. Ownership is selected from the validated handshake,
never the benchmark name, task identity, credential presence, or `ask_user`
boolean. Adapter ownership without ask support is invalid.

The adapter-owned path registers the outstanding question locally, sends one
existing `ask_user_exchange` RPC with `answer=null`, validates the returned
`ask_id`, canonical request digest, acceptance and bounded nonempty answer,
and completes locally. There is no host-finish RPC or second generation.
OSWorld calls its provider once and returns only the provider's public answer.
Missing simulators and provider refusals cancel unscored. Explicit responder
overrides fail before prepare. Host-owned exchanges preserve `UserResponder`
but cannot publish an answer the adapter refused.

Successful answers use the existing response artifact → `ask_answer` → next
model request path. The exchange event still precedes execution of the ask
batch. Empty/oversized replies and identity mismatches fail closed. An unanswered
mutating call poisons the session and requires rescue; neither automatic nor
explicit retry may regenerate an answer on that session.

## Reproduction

Baseline: `481d4fbbef94baf27fc0601784ba71e6d615c356`.

A temporary pytest plugin extracted `EpisodeRunner._run_ask` verbatim from that
commit using Python AST and rebound only that method in the current runner.
This isolates the proven choreography defect without changing the worktree,
reverting wire models, or touching an existing benchmark environment. The real
installed-worker regression was then run with that plugin:

```text
test_public_answer_crosses_worker_into_next_model_request[adapter]
FAILED: assert 'cancelled' == 'completed'
diagnostic: ask-user was not answered
1 failed in 8.02s
```

The installed adapter's `tiny_ask_calls` marker was absent: **zero adapter
exchange/simulator invocations**. This is a replay of the old method, not a claim
that the complete historical wheel was rebuilt or exercised.

## Post-fix verification

All commands used explicit worktree imports with the shared interpreter
read-only; no cloud or external model calls were made.

```sh
PYTHONPATH="$PWD" ~/local-operator/.venv/bin/python -m pytest -n 2 -q \
  tests/unit/evaluation/adapters/test_api.py \
  tests/unit/evaluation/adapters/test_supervisor.py \
  tests/unit/evaluation/adapters/test_worker.py \
  tests/unit/evaluation/adapters/test_discovery.py \
  tests/unit/evaluation/adapters/test_launch.py \
  tests/unit/evaluation/adapters/test_rpc.py \
  tests/unit/evaluation/runner/test_episode.py \
  tests/unit/evaluation/runner/test_episode_subprocess.py \
  tests/unit/evaluation/adapters/osworld/test_fake_end_to_end.py \
  tests/unit/evaluation/adapters/osworld/test_spawn.py \
  tests/unit/evaluation/adapters/osworld/test_build_and_scripts.py \
  tests/unit/evaluation/adapters/osworld/test_secrets_and_rescue.py \
  tests/unit/evaluation/test_action_surface.py \
  tests/unit/evaluation/runner/test_provider_client.py::test_ask_answer_reaches_the_model_with_the_next_observation
```

Result: **245 passed in 150.90s**. The final direct-retry and provider-refusal
assertions were then included in this focused delta/real-path run:

```sh
PYTHONPATH="$PWD" ~/local-operator/.venv/bin/python -m pytest -n 2 -q \
  tests/unit/evaluation/adapters/test_supervisor.py::test_a_timeout_never_claims_the_observation_phase \
  tests/unit/evaluation/adapters/osworld/test_fake_end_to_end.py::test_simulator_is_called_once_and_host_finish_is_refused \
  tests/unit/evaluation/runner/test_episode_subprocess.py::test_public_answer_crosses_worker_into_next_model_request \
  tests/unit/evaluation/adapters/osworld/test_spawn.py::test_real_wheel_simulator_answer_enters_public_artifact_and_next_request
```

Result: **7 passed in 18.44s**. These exercise real `ProviderModelClient` request
construction through a local `RecordingStream`, not a paid model. A copied,
installed interpreter runs the real supervisor/worker boundary. The OSWorld
case builds and installs the actual adapter wheel with the existing fixture
helper and selects its no-cloud fake provider using a pinned synthetic
workspace. No gated task/profile/evaluator or private fixture data was read.

The reopened sealed bundles independently verified; their public response bytes
and SHA-256 values were:

| Case | Public answer | SHA-256 |
| --- | --- | --- |
| Adapter-owned subprocess | `public simulator answer 1` | `d6f51f4a8e995dd62eb2fd257caa19951339c77159bb8e5306413147bc44b61f` |
| Host-owned subprocess | `host public answer` | `e770afcf8a3cc200763802a360d29ff6a05fedb64604872aa9d0fabe0c4ddfb4` |
| Installed OSWorld wheel | `simulated user answer` | `c7ace92bfbe173154b7186db8059238b3bc768e9cad01e321601a4dfb9b5fdfd` |

The adapter-owned invocation log was exactly `["None"]`; the host-owned log
was exactly `["host public answer"]`. Each next `ChatRequest` contained the
matching `Answer from the user: ...` string, and the exchange event immediately
preceded the ask batch and environment step. The simulator's counted answer
changes on another invocation, so a hidden regeneration cannot pass as the
same reply. Adversarial subprocess cases cover refusal, missing/empty/oversized
answers, wrong IDs/prompts/episodes, and a timeout **after** generation; none
publishes a substitute or requests another model decision. Protocol 1.5 and
incompatible capability declarations fail before allocation.

Final static gates: `git diff --check`, flake8 on every changed Python file,
and pyright on the modified runtime and behavioral-test files all passed
(`0 errors, 0 warnings, 0 informations`).

## Integration with proxy/scoped infrastructure

Rebased onto PR #683's merged `origin/main`,
`da293e177009e5135afc9b853ad136fd61ea19c1`. Both implementation/documentation
commits replayed without conflicts; `git range-diff` confirmed unchanged patches.
The compatibility census included the new proxy/scoped-infrastructure tests:
remaining `1.5` literals in executable tests are exclusively intentional
old-protocol rejection fixtures. Shared fixture builders use RPC 1.6.

Pinned isort 5.13.2 required only mechanical nested-import formatting in
`test_episode_subprocess.py`, `test_spawn.py`, and `test_fake_end_to_end.py`.
No proxy/scoped-infrastructure implementation or protocol behavior needed
remediation. Black 26.1.0, isort 5.13.2, and flake8 7.3.0 were installed into an
isolated temporary target and run with the shared interpreter; the shared
virtual environment remained unchanged. Pyright used pinned 1.1.411.

The bounded matrix above, plus the following proxy/scoped-infrastructure files,
passed together with explicit `PYTHONPATH="$PWD"` imports and `-n 2`:
**415 passed in 186.15s**.

```text
tests/unit/evaluation/adapters/osworld/test_proxy_policy.py
tests/unit/evaluation/adapters/osworld/test_aws_provider.py
tests/unit/evaluation/adapters/osworld/test_requirements.py
tests/unit/evaluation/adapters/osworld/test_run_episode_script.py
tests/unit/evaluation/adapters/osworld/test_provisioning.py
tests/unit/evaluation/runner/test_run_episode_infra.py
```

This includes botocore Stubber/loopback-only provider tests and the actual
`run_episode.py` subprocess against the installed fake-provider wheel, not
cloud or paid-model calls. After import remediation, all whole-tree gates
passed: `black --check --workers 2 .` (940 unchanged files), `isort --check .`,
`flake8 --jobs 2 .`, and `pyright --pythonpath ~/local-operator/.venv/bin/python`
(`0 errors, 0 warnings, 0 informations`).

After formatting, the three affected test modules were rerun together with
`PYTHONPATH="$PWD"` and `-n 2`: **47 passed in 119.95s**. This re-exercised the
real supervisor/worker exchange, installed OSWorld wheel, response artifacts,
next model requests, refusal paths, and no-regeneration checks on the final
Python tree.

A separate actual `scripts/run_episode.py` subprocess with
`--infra unknown:NAME=synthetic-marker` returned exit 2 before reading its
nonexistent selector: stdout was empty, stderr was
`--infra expects [purpose:]NAME=VALUE with a valid name, purpose and value`,
and the requested run root was not created. The value was not echoed.

## Limits and release boundary

This is offline synthetic evidence, not task 098 benchmark performance. No
private profile, benchmark model call, cloud allocation, existing environment,
or existing benchmark evidence was touched. Root and adapter distribution
versions were deliberately not bumped; a release must build a new artifact
and pin its selector together with RPC 1.6. Frozen environments/rescue pins
must remain intact. No full repository suite, push, PR, or merge was performed
as part of this bounded slice.
