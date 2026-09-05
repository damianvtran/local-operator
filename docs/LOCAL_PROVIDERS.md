# Local and self-hosted model servers

Local Operator connects to LM Studio, Ollama, vLLM, llama.cpp, and other
OpenAI-compatible servers. It does not install runtimes, download or load
models, or change server settings. Start your server and load a chat model in
its own application first. In LM Studio, enable the server in the Developer tab.

## Set up in the app

1. Enter `/login` and choose the server, for example `lmstudio`.
2. Confirm its HTTP(S) server URL. A server root or an API root ending in `/v1`
   works; reverse-proxy prefixes are retained.
3. Enter an optional API token. Input is masked and never becomes composer
   history or transcript text. Enter keeps an existing token only for the exact
   same endpoint; `-` clears it. A changed endpoint never inherits the old token.
4. The app checks the server and shows its chat model IDs. Enter the exact ID
   to use, or accept the displayed default from the live list. If the server
   disables model listing with HTTP 404/405, enter an exact served ID manually.
5. Confirm **Activate and save as default**. Escape at any prompt, a rejected
   token, an invalid URL, or a failed/empty listing leaves the existing
   configuration unchanged.

After setup, `/model` opens the existing model picker. `/model provider/exact-id`
selects an exact model for this session; `/model default` persists the selection.
Model IDs retain their slashes, colons, and case. There is no fabricated model
default and no inference request merely to test the connection.

| Preset | Default API root | Metadata beyond `/v1/models` |
| --- | --- | --- |
| `lmstudio` | `http://localhost:1234/v1` | `/api/v1/models`: loaded instance IDs, active context, tool/vision/reasoning support |
| `ollama` | `http://localhost:11434/v1` | `/api/ps`: active context; bounded `/api/show`: trained context and capabilities |
| `vllm` | `http://localhost:8000/v1` | `max_model_len` from the compatible list |
| `llamacpp` | `http://localhost:8080/v1` | Optional `/props` generation context |
| `openai-compatible` | Explicit URL required | Compatible list; manual metadata for MLX, LocalAI, proxies, and similar servers |

A stopped server stays visible in the provider list so it can be configured.
A cached model list is not proof a server is currently running. Optional native
metadata endpoints may be absent; the compatible listing still works. Known
embedding-only entries are excluded, but a server that does not identify their
type may require you to select the correct chat model yourself.

## Endpoints and model budgets in Settings

The **Local servers** section of `/settings` exposes `providers.<id>.base_url`
and `providers.<id>.models`. Model overrides are JSON keyed by exact model ID:

```json
{
  "my-model/with:tag": {
    "context_window": 8192,
    "max_output_tokens": 2048,
    "supports_tools": true,
    "supports_images": false,
    "reasoning": true,
    "supports_sampling_params": false
  }
}
```

Hand-written YAML mappings are accepted too. Invalid fields and non-positive
budgets are rejected before persistence. Overrides affect future model
activation; reselect with `/model saved` after changing them. Changing an
endpoint does not silently retarget an active conversation. Reconfigure or
reselect explicitly before continuing.

**A client override does not resize the server.** An observed active context
limit caps the requested budget. A model's training maximum is not proof that
the server loaded it with that context size. Where the server states no active
limit, the client uses a conservative 4,096-token budget and 1,024-token output
reservation unless explicitly overridden. Tool/vision/reasoning metadata keeps
unknown separate from false; unknown image support defaults to off, while
unknown tool support permits the server to handle tool definitions. Use the
overrides when your model/template configuration is known.

Local model names do not inherit cloud-provider sampling or effort settings.
Reasoning support does not imply a standard effort control. Structured
`reasoning_content`/`reasoning` responses are retained as protocol state and
replayed only to the same provider, model, endpoint, credential, and unchanged
message. They are not inserted into visible answer text.

Ollama does not support `tool_choice`. Automatic tool use sends definitions
without that field; `none` omits definitions entirely; `required` fails explicitly
rather than silently becoming automatic. Tool use still requires a suitable
model and a correctly configured server template/parser, particularly with vLLM.

Local presets show zero API-token price, not zero hardware or electricity cost.
Generic gateways have unknown prices unless the server explicitly quotes them;
a keyless gateway is not assumed free.

## Desktop application / HTTP API

`GET /v1/models/providers` includes these provider IDs with no required
credentials. `GET /v1/models?provider=lmstudio` uses the same endpoint resolver
and catalogue as the TUI. The separate desktop application consumes those lists
dynamically, so a running default-port keyless LM Studio server is selectable
without command-line configuration. This does not add an endpoint/token editor
to that separate application; advanced configuration is available in this
runtime's TUI Settings/setup flow.

## Comparison and boundaries

Like OMP and OpenCode, Local Operator uses the compatible Chat Completions wire
for user-operated servers rather than maintaining a second inference engine.
This integration adds in-app endpoint/token setup, endpoint-scoped discovery,
native local context metadata, exact-model overrides, and scoped reasoning
replay. It is not a claim of complete configuration or runtime-management parity.

Not included: automatic downloads/load/unload, GPU management, multiple named
instances of one preset, native Responses routing by default, or universal tool
parser compatibility. A proxy exposing only `/v1` can use the generic preset.

References: [LM Studio API](https://lmstudio.ai/docs/developer/rest),
[Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility),
[vLLM compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html),
[llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server).
