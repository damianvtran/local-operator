# Experimental Responses translator setup — unverified live integration

Codex's tool-search bridge expects a separately provisioned Responses-to-Messages
translator. The original private `litellm.yaml` is **not** distributed: it may
contain credentials and local settings. `translator_config.example.json` is a
new portable JSON/YAML-subset configuration with only a dummy downstream bearer
and an environment reference for the local translator's master key. It contains
no OpenRouter key, private loadout or schema snapshot.

Use an existing interpreter with **LiteLLM 1.106.6** and its proxy dependencies,
matching the transformation integration test's intended version. This version
is a proposed setup pin, not a validated public install/lock. No dependency was
installed or translator launched for this port. Review the version's own license
and CLI configuration behavior before freezing. If your install cannot load this
configuration or transform the Responses route, stop and qualify it separately;
do not silently switch translators or drop parameters.

From this directory, after starting the guarded meter per `ADAPTERS.md`:

```sh
export BENCH_LITELLM_PY='<existing-LiteLLM-1.106.6-proxy-python>'
export BENCH_LITELLM_BIN='<existing-LiteLLM-1.106.6-litellm-CLI>'
export BENCH_CURRENT_RUN_FILE='<same-absolute-scratch-current_run-as-meter>'
export BENCH_LITELLM_URL=http://127.0.0.1:4000/v1
# Inject BENCH_LITELLM_KEY via a secret manager into the environment only.
# It is a LOCAL proxy credential, not the upstream OpenRouter credential.
"$BENCH_LITELLM_BIN" \
  --config "$PWD/translator_config.example.json" --host 127.0.0.1 --port 4000
```

The downstream `anthropic/claude-sonnet-5-5` route terminates at the loopback
meter's `/r/@current` route. The meter alone holds `OPENROUTER_API_KEY` and uses
explicit provider/model/price bounds. The dummy bearer in this example is not a
secret or a provider credential; the meter never forwards it as upstream auth.
Change the local meter port in a reviewed external copy of the example if needed.
Do not put a real key in the example/config or change it to a direct provider URL.

The runner writes the current run ID under a lock before starting its per-run
bridge. All processes sharing this translator must honor that sequential routing
contract; otherwise cost attribution is unsafe. Do not run unrelated clients or
legacy runners concurrently. Matching `BENCH_CURRENT_RUN_FILE` is mandatory.

Optional zero-spend transformation check with the same existing interpreter:

```sh
PYTHONDONTWRITEBYTECODE=1 "$BENCH_LITELLM_PY" -m unittest -v test_cache_bridge
```

This test imports the actual transformation classes, not the CLI server; it
cannot establish native Codex tool choice, provider response conversion, served
model identity, cache qualification, billing or bounded exposure. Those require
separately authorized runtime-bound wire controls before a frozen live run.

Public upstream source: https://github.com/BerriAI/litellm
No LiteLLM implementation code is vendored here.
