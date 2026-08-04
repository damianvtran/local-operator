# Scout Report — ~/local-operator Deep Audit (read-only)

> **Delivery note:** `local://scout-lo-audit.md` could not be created — this session provisions no write/edit tool, and the node_repl sandbox returns `EPERM` for every path tried (`/Users/damian/tmp`, tmpDir, and the `local://` root). The full report follows verbatim; write it to that path from an agent that has filesystem access.

Audited tree: `/Users/damian/local-operator`. Package `local_operator` v0.15.10, Python >= 3.12. ~53k LOC across `local_operator/` + `tests/`. Entry point: `local-operator = local_operator.cli:main`.

## 0. Size map

| File | LOC | Bytes |
|---|---:|---:|
| `local_operator/executor.py` | 3238 | 133 KB |
| `local_operator/prompts.py` | 2584 | 176 KB |
| `local_operator/tools/general.py` | 2169 | — |
| `local_operator/agents.py` | 1838 | 74 KB |
| `local_operator/server/routes/agents.py` | 1607 | — |
| `local_operator/model/registry.py` | 1481 | — |
| `local_operator/operator.py` | 1266 | — |
| `local_operator/clients/radient.py` | 1158 | — |
| `local_operator/helpers.py` | 1058 | — |
| `local_operator/server/models/schemas.py` | 962 | — |
| `local_operator/scheduler_service.py` | 924 | — |
| `local_operator/cli.py` | 822 | — |

Three files (`executor.py`, `prompts.py`, `tools/general.py`) are ~8k LOC = the entire agent harness.

---

## 1. Current harness

### 1.1 `local_operator/operator.py` — the main loop (1266 LOC)

Key symbols:
- `OperatorType(Enum)` — `CLI` | `SERVER` (L52)
- `process_classification_response(response_content) -> RequestClassification` (L57) — hand-rolled XML tag scraping for `<type>`, `<planning_required>`, `<relative_effort>`, `<subject_change>`.
- `class Operator` (L107) — holds `executor: LocalCodeExecutor`, `credential_manager`, `model_configuration`, `config_manager`, `agent_registry`, `current_agent: AgentData | None`, `env_config`, `verbosity_level`, `auto_save_conversation`, `persist_agent_conversation`.

**Turn flow — `Operator.handle_user_input(user_input, user_message_id=None, attachments=[], additional_instructions=None)` (L862):**

1. `executor.update_ephemeral_messages()` — refresh env details / HUD.
2. `apply_attachments_to_prompt(user_input, attachments)` (from `prompts.py`).
3. Append a `CodeExecutionResult` with `execution_type=USER_INPUT` to code history.
4. `classify_request(user_input)` (L261) — **an extra full LLM round trip per user turn**. Builds a throwaway message list with `RequestClassificationSystemPrompt`, truncated to `max_conversation_depth=8`, retries up to 3x.
5. If `classification.subject_change`: clear plan, `reset_learnings()`, inject a `<system>` "possible subject change" nudge message.
6. If type != `CONTINUE`: `add_task_instructions(classification)` (L432) -> ephemeral `TaskInstructionsPrompt` + per-type instruction blob.
7. Append the user message to `executor.agent_state.conversation`.
8. If `classification.planning_required`: `generate_plan()` (L372) — **another full LLM round trip** producing a free-text plan stored via `executor.set_current_plan()`.
9. **Agent loop** `while not done and not final_response and not needs_user_input and not executor.interrupted`:
   - `invoke_and_process_response(conversation, classification)` (L541)
   - if `response_json is None` -> `process_text_response()` terminates the loop
   - else `executor.update_ephemeral_messages()` (HUD refresh)
   - optional autosave per step (`handle_autosave`, L1199)
   - break on `CANCELLED`/`INTERRUPTED`
10. Returns `(ResponseJsonSchema | None, final_response: str)`.

**`invoke_and_process_response` (L541)** is the streaming core:
- creates an in-progress streamable `CodeExecutionResult` in code history;
- retry loop (`max_attempts=3`) around `async for chunk in executor.stream_model(messages)`;
- each chunk: `accumulated_text += chunk`, then `finished, result = stream_action_buffer(accumulated_text)` — **re-parses the entire accumulated buffer from scratch on every chunk (O(n^2))**;
- mutates `new_message` fields and `await executor.broadcast_message_update(...)` per chunk (WebSocket fan-out per token);
- on parse failure, injects a `<system>Your previous action response was not parsable...` message and retries;
- finally `executor.process_response(response_json, classification)`.

Other notable methods: `interpret_action_response` (L460, uses `helpers.parse_agent_action_xml`), `_has_action_tag` (L537, regex `<action>([^<]+)</action>`), `delegate_to_agent` (L1040), `chat()` (L1140, CLI REPL with readline history), `execute_single_command()` (L1116, for the `exec` subcommand), `process_message_for_agent()` (L1239).

**Legacy/dead paths:** `_agent_is_done`/`_agent_requires_user_input`/`_agent_should_exit` compare `response.action` against `"DONE"`/`"ASK"`/`"BYE"` — `ActionType` marks these as "Kept in for backwards compatibility, but not used anymore" (`types.py` L41-45). The loop's exit condition therefore rests on legacy enum values plus "did the model emit a non-action text response".

**Pain points:**
- 3 LLM calls minimum per user turn (classify -> plan -> act), all serial, none cached.
- Signal handler installed globally at construction time (`_setup_interrupt_handler`, L194) — a process-global `signal.signal(SIGINT, ...)` from a library class.
- CLI concerns (readline history, ANSI box drawing, `print`) hard-wired into the same class the server uses; `verbosity_level >= VerbosityLevel.VERBOSE` checks scattered through business logic.
- Streaming display logic (splitting on `<action_response>`, printing partial text) lives inline in `invoke_and_process_response`.
- Error recovery is string-splicing prompts back into history rather than a typed retry protocol.

### 1.2 `local_operator/executor.py` — code execution (3238 LOC)

`class LocalCodeExecutor` (L311). Constructor (L361) takes `model_configuration`, `max_conversation_history=100`, `detail_conversation_length=10`, `can_prompt_user`, `agent`, `agent_state: AgentState`, `agent_registry`, `max_learnings_history=50`, `verbosity_level`, `persist_conversation`, `job_id`.

**Execution model — the single most important thing to understand:**
`self.context = {"__builtins__": builtins}` is a **plain dict used as the globals namespace of `exec()` in the host process**. There is no sandbox, no subprocess, no container.
- `_run_code(code)` (L1543): redirects `sys.stdin` to `/dev/null`; if the code contains `"async def"` or `"await"`, it **textually indents the whole block into a generated `async def __exec_async_code_wrapper__(__context_dict_to_update__)`** and copies `locals()` back into the context dict; otherwise `compile(code, "<agent_generated_code>", "exec")` and `await asyncio.to_thread(exec, ...)` — except when the code text contains `matplotlib`/`tkinter`/`PIL`, in which case `exec` runs on the event-loop thread.
- `_execute_with_output` (L1311) captures stdout/stderr via `io.StringIO` + a logging `StreamHandler`, streaming partial output.
- Errors wrapped in `CodeExecutionError` (L243) whose `agent_info_str()` (L264) walks the traceback for frames named `<agent_generated_code>` and emits line-annotated code inside XML tags for the model (`annotate_code`, L187).
- Agent `context` is persisted with **`dill`** (see `agents.py`) — arbitrary pickled Python objects reloaded at startup (`__init__` L420: `agent_registry.load_agent_context(agent.id)`).

**Action dispatch — `perform_action(response, classification)` (L1835):**
`DONE`/`BYE`/`ASK` -> no-op success. Otherwise dispatch on `ActionType`:
- `DELEGATE` -> `delegate_to_agent(agent, message)` (L2182) via `self.delegate_callback` (wired by `Operator.__init__`).
- `WRITE` -> safety check -> `write_file(file_path, content)` (L2397)
- `EDIT` -> safety check -> `edit_file(file_path, replacements)` (L2513, uses `difflib`)
- `READ` -> `read_file(file_path, max_text_file_size_bytes=MAX_FILE_READ_SIZE_BYTES)` (L2252). Caps at `MAX_FILE_READ_TOKENS = 50000` (`CHARS_PER_TOKEN = 4` -> 200 KB).
- `CODE` -> `execute_code(response, max_retries=1)` (L1207)

**Safety subsystem:** `check_response_safety` (L1005) makes **yet another LLM call** with `SafetyCheckSystemPrompt`/`SafetyCheckConversationPrompt`, expecting the literal markers `[SAFE]`/`[UNSAFE]`/`[OVERRIDE]` parsed by `get_confirm_safety_result` (L97) via naive substring search. `ConfirmSafetyResult` enum (L86). `prompt_for_safety` (L1282) blocks on terminal input when `can_prompt_user`.

**Model I/O:** `invoke_model` (L885), `stream_model` (L952), `_convert_and_stream` (L697) — builds LangChain-shaped message dicts with multimodal parts: base64 data URLs for png/jpg/gif/bmp/webp/heic/heif (HEIC via `convert_heic_to_png_data_url`) and `{"type":"file","file":{...}}` for PDFs; applies **manual Anthropic cache-control markers** when `"anthropic" in model_name`. Token metrics via `ExecutorTokenMetrics` (L229) + `langchain_community.callbacks.manager.get_openai_callback` and `tiktoken.encoding_for_model` (falls back to `gpt-4o`).

**Context management:** `_limit_conversation_history` (L2649), `_summarize_old_steps` (L460), `_summarize_conversation_step` (L2667) — **summarization is another LLM call** using `MessageSummarySystemPrompt`. `update_ephemeral_messages` (L2988), `create_hud_message` (L3019) / `add_ephemeral_hud_message` (L3080) build the "agent heads up display"; `get_environment_details` (L2835) + `_get_git_status` (L2867) + `format_directory_tree` (L2748).

**Server bridge (leaks server concerns into the executor):** `update_code_history` (L3157), `broadcast_message_update` (L3170), `update_job_execution_state` (L3184), `tool_execution_callback` (L3216), plus `self.status_queue` (a `multiprocessing.Queue`).

**Pain points:** one class owns model I/O, sandboxing, safety, file CRUD, summarization, HUD generation, websocket broadcasting, job state, and token accounting. `exec` into a shared mutable dict makes concurrency and isolation impossible. `dill`-pickled agent context is an RCE-shaped persistence format.

### 1.3 `local_operator/stream.py` — incremental action parser (262 LOC)

Single public function `stream_action_buffer(accumulated_text, lookahead_length=32) -> Tuple[bool, CodeExecutionResult]`.
- `_extract_thinking_content` — strips leading `<think>`/`<thinking>` blocks.
- Locates `<action_response>` / `</action_response>`, tolerating a preceding triple-backtick `xml` fence (`XML_FENCE`).
- `_parse_action_content(content, result, partial)` — scans for the fixed tag list `["action","content","code","replacements","mentioned_files","learnings","file_path","agent"]` with `str.find`, supporting unterminated tags in partial mode.
- `_assign_tag_content` **appends** (`result.code += content`) — safe only because the caller re-parses the whole buffer each chunk with a fresh `CodeExecutionResult`.
- `_handle_partial_mentioned_files` splits on newlines.
- 32-char lookahead prevents flushing a half-written tag to the UI.

**Pain points:** O(n^2) re-parse per chunk; no nesting support (a `<code>` block containing the literal string `</code>` breaks it); duplicated tag knowledge with `helpers.parse_agent_action_xml` and `prompts.ActionResponseFormatPrompt` — three places must agree. This module is small, well tested (`tests/unit/test_stream.py`, 472 LOC), and the cleanest candidate for straight reuse.

### 1.4 `local_operator/helpers.py` — parsing + platform glue (1058 LOC)

- Response cleanup: `remove_think_tags` (L51), `clean_plain_text_response` (L76), `clean_json_response` (L134), `process_json_response` (L321), `is_marker_inside_json` (L344), `_extract_initial_think_tags` (L617).
- XML action parsing: `_extract_tag_content` (L368), `parse_replacements` (L391 — SEARCH/REPLACE diff-notation blocks, supports nesting), `parse_agent_action_xml` (L453) — the non-streaming counterpart to `stream.py`.
- Image: `convert_heic_to_png_data_url` (L669), `convert_heic_to_png_file` (L707) — optional `PIL`/`pillow_heif` guarded by try/except.
- Platform PATH repair: `get_windows_registry_path` (L749), `get_posix_shell_path` (L839), `setup_cross_platform_environment` (L929) — called at CLI and server startup so subprocesses inherit a login-shell PATH. **Genuinely valuable, hard-won, non-obvious code.**

**Pain points:** a grab-bag module mixing LLM-output parsing with OS environment repair; the JSON-cleaning helpers are vestigial (the protocol moved to XML but the `process_json_response` / `ResponseJsonSchema` JSON path survives).

### 1.5 `local_operator/prompts.py` — 176 KB / 2584 lines

**Contents summary** (~95% string constants):

*Prompt-construction helpers (~340 lines of code at the top):* `get_installed_packages_str` (L18), `get_tools_str(tool_registry)` (L60) + `_should_skip_tool`, `_format_tool_documentation`, `_format_function_args`, `_get_return_type_info`, `_is_custom_type`, `_generate_type_documentation`, `_generate_pydantic_model_docs`, `_generate_example_values`, `_get_property_type` — an ad-hoc reflection-based tool-schema generator that renders Python signatures + Pydantic models into prose for the prompt (no JSON-schema / function-calling).

*Core prompt constants:*
- `LocalOperatorPrompt` (L342) — persona preamble.
- `BaseSystemPrompt` (L358) — core principles, safety, code-reuse-across-steps semantics, scheduling examples.
- `ActionResponseFormatPrompt` (L754) — the XML action protocol spec + worked examples.
- `PlanSystemPrompt` (L1026), `PlanUserPrompt` (L1036), `ReflectionUserPrompt` (L1048).
- `ActionInterpreterSystemPrompt` (L1061), `JsonResponseFormatSchema` (L1182) — legacy JSON path.
- `SafetyCheckSystemPrompt` (L1207), `SafetyCheckConversationPrompt` (L1322), `SafetyCheckUserPrompt` (L1455).
- `RequestClassificationSystemPrompt` (L1469), `RequestClassificationUserPrompt` (L1555).
- `MessageSummarySystemPrompt` (L1567).
- `AgentHeadsUpDisplayPrompt` (L2228), `TaskInstructionsPrompt` (L2290), `ScheduleInstructionsPrompt` (L2306), `EditFileInstructionsPrompt` (L2317), `FinalResponseInstructions` (L2196).

*Per-request-type instruction blobs (the bulk):* `RequestType(str, Enum)` (L1590) with 19 categories, each with a dedicated multi-hundred-line constant: `ConversationInstructions`, `CreativeWritingInstructions`, `DataScienceInstructions`, `MathematicsInstructions`, `AccountingInstructions`, `LegalInstructions`, `MedicalInstructions`, `ResearchInstructions`, `DeepResearchInstructions`, `MediaInstructions`, `CompetitiveCodingInstructions`, `SoftwareDevelopmentInstructions`, `FinanceInstructions`, `NewsReportInstructions`, `ConsoleCommandInstructions`, `PersonalAssistanceInstructions`, `ContinueInstructions`, `TranslationInstructions`, `OtherInstructions` — mapped by `REQUEST_TYPE_INSTRUCTIONS: Dict[RequestType, str]` (L2174), fetched via `get_request_type_instructions` (L2411).

*Assembly functions (bottom):* `get_system_details_str` (L2416, uses `psutil` for CPU/RAM/GPU), `apply_attachments_to_prompt` (L2494), `create_action_interpreter_prompt` (L2534), `create_system_prompt(tool_registry, response_format, agent_system_prompt, ...)` (L2542).

**Pain points:** prompt text is compiled Python source — no hot reload, no per-model variants, no diffing, no A/B, no i18n, and every edit is a package release. The 19 vertical instruction blobs are the main reason the classification LLM call exists at all. Externalising these to markdown/templates is the single highest-leverage change available.

---

## 2. CLI surface — `local_operator/cli.py` (822 LOC) — **must stay backward compatible**

Built with `argparse` in `build_cli_parser()` (L57). A `parent_parser` supplies these flags to **every** subcommand:

| Global flag | Notes |
|---|---|
| `--debug` | sets `LOCAL_OPERATOR_DEBUG=true`, verbosity -> `DEBUG` |
| `--agent` / `--agent-name` (dest `agent_name`) | creates the agent if it does not exist |
| `--train` | persist conversation to the agent dir after each task |

Root-parser-only flags:

| Flag | Notes |
|---|---|
| `--version` | `v{version('local-operator')}` |
| `--hosting` | choices: `radient, deepseek, openai, anthropic, ollama, kimi, alibaba, google, mistral, openrouter, xai, test` |
| `--model` | free-form model id |
| `--run-in` (dest `run_in`) | `os.chdir` to a validated directory |

Subcommands:

| Command | Args | Handler |
|---|---|---|
| *(none)* | — | interactive `operator.chat()` |
| `exec <command>` | positional `command` | `operator.execute_single_command` |
| `serve` | `--host` (default `0.0.0.0`), `--port` (default **1111**), `--reload` | `serve_command` -> `uvicorn.run("local_operator.server.app:app", ...)` |
| `credential update <key>` | `key` | `credential_update_command` -> `prompt_for_credential` |
| `credential delete <key>` | `key` | `credential_delete_command` -> sets value to `""` |
| `config create` | — | writes `~/.local-operator/config.yml` |
| `config open` | — | `open`/`xdg-open`/`start` on the config file |
| `config edit <key> <value>` | auto-coerces int/float/bool/null | `config_edit_command` |
| `config list` | — | prints keys + descriptions |
| `agents list` | `--page` (1), `--perpage` (10) | `agents_list_command` |
| `agents create <name>` | positional `name` (prompts if empty) | `agents_create_command` |
| `agents delete` | mutually exclusive **required**: `--name` (local) or `--id` (Radient) | `agents_delete_command` |
| `agents push` | mutually exclusive **required**: `--name` or `--id` | Radient upload (zip export) |
| `agents pull` | `--id` (**required**) | Radient download |

Documented credential keys: `RADIENT_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `KIMI_API_KEY`, `ALIBABA_CLOUD_API_KEY`, `GOOGLE_AI_STUDIO_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `XAI_API_KEY`.

`main()` (L539) order of operations: parse -> `setup_cross_platform_environment()` -> set `LOCAL_OPERATOR_DEBUG` -> `get_env_config()` -> fixed dirs `~/.local-operator` (config) and `~/local-operator-home` (created if missing) -> dispatch credential/config/agents/serve subcommands (which **return before** model config) -> otherwise build `ConfigManager`/`CredentialManager`/`AgentRegistry`, `update_config_from_args`, resolve/create agent, build `JobManager` + `WebSocketManager` + `SchedulerService`, `initialize_operator(...)`, then `asyncio.run(async_main_cli())` which starts the scheduler, runs `chat()` or `execute_single_command()`, then shuts the scheduler down. All exceptions print a red banner + `traceback.print_exc()` and return `-1`.

**Backward-compat contract to preserve:** exact subcommand/flag names (including the `--agent-name` alias and `dest` names), default port 1111, exit codes `0` / `-1`, and the two on-disk directories.

---

## 3. Model / provider layer

### 3.1 `local_operator/model/registry.py` (1481 LOC)

- `class ProviderDetail(BaseModel)` — `id, name, description, url, requiredCredentials: List[str], recommended: bool`.
- `SupportedHostingProviders: List[ProviderDetail]` (L28) — **11 providers**: `radient` (recommended, `RADIENT_API_KEY`), `openai`, `anthropic`, `google` (`GOOGLE_AI_STUDIO_API_KEY`), `mistral`, `ollama` (no creds), `openrouter`, `deepseek`, `kimi`, `alibaba` (`ALIBABA_CLOUD_API_KEY`), `xai`. The CLI additionally accepts `test` (-> `ChatMock`) and code paths reference `noop` (-> `ChatNoop`).
- `RecommendedOpenRouterModelIds` / `RecommendedRadientModelIds` (= the former + `"auto"`).
- `class ModelInfo(BaseModel)` — `input_price`, `output_price` (per **million** tokens), `max_tokens`, `context_window`, `supports_images`, `supports_prompt_cache`, `cache_writes_price`, `cache_reads_price`, `description`, `recommended`.
- The remaining ~1200 lines are **hardcoded per-model `ModelInfo` tables** per provider plus `get_model_info(hosting, model)` and the `openrouter_default_model_info` / `radient_default_model_info` fallbacks. A stale-by-construction pricing catalogue.

### 3.2 `local_operator/model/configure.py` (720 LOC)

- `ModelType = Union[ChatOpenAI, ChatOllama, ChatAnthropic, ChatGoogleGenerativeAI, ChatMock, ChatNoop]` — **every provider other than Anthropic/Google/Ollama is a `ChatOpenAI` pointed at a different `base_url`**.
- `class ModelConfiguration` (L37) — `hosting, name, instance, info, api_key, temperature (0.2), top_p (0.9), top_k, max_tokens, frequency_penalty, presence_penalty, stop, seed`. Plain class, not a Pydantic model.
- `validate_model(hosting, model, api_key) -> bool` (L139) — a hand-written `if/elif` chain of model-list endpoints: deepseek `api.deepseek.com/v1/models`, openai, openrouter, radient `api.radienthq.com/v1/models`, anthropic (`x-api-key` + `anthropic-version: 2023-06-01`), kimi `api.moonshot.cn`, alibaba `dashscope-intl.aliyuncs.com/compatible-mode/v1`, google `generativelanguage.googleapis.com/v1` (`x-goog-api-key`), mistral, ollama `http://localhost:11434/api/tags`, xai `api.x.ai`. Unknown hosting -> `True`. Called only for `OperatorType.CLI` (from `bootstrap.py`).
- `_check_model_exists_payload` (L100) — per-provider response-shape normalisation (`models` vs `data`, `name` vs `id`, Anthropic `-latest` prefix matching).
- `get_model_info_from_openrouter` / `get_model_info_from_radient` — live pricing lookup, per-token -> per-million.
- `configure_model(hosting, model_name, credential_manager, model_info_client=None, env_config=None, **sampling)` — the factory; `calculate_cost(...)` also lives here (imported by `executor.py`).

### 3.3 `local_operator/clients/`

| File | LOC | Purpose |
|---|---:|---|
| `radient.py` | 1158 | Radient gateway: image gen, web search, model list/pricing, send-email-to-user, transcription, **OAuth token refresh** (`RadientTokenResponse`, `RadientTokenRefreshRequest`), agent marketplace push/pull/delete |
| `google_client.py` | 975 | Raw REST client for Gmail / Calendar / Drive + `refresh_google_access_token(client_id, client_secret, refresh_token)` |
| `serpapi.py` | 522 | SERP API search; ~15 Pydantic result models |
| `fal.py` | 348 | FAL image generation (`ImageSize`, `GenerationType`, polling via `FalRequestStatus`) |
| `openrouter.py` | ~110 | model list + pricing |
| `ollama.py` | ~90 | health check + `/api/tags` |
| `tavily.py` | ~90 | Tavily search |

All are thin `requests`-based sync clients with Pydantic response models and `SecretStr` keys. No shared base class, no retry/backoff, no async.

### 3.4 `local_operator/config.py` — `config.yml` schema (265 LOC)

Location: `~/.local-operator/config.yml`. `ConfigManager(config_dir)` reads/writes YAML via `yaml.safe_load`/`yaml.dump`.

Document shape:

```yaml
version: "0.15.10"            # compared against importlib.metadata version; warns if newer
metadata:
  created_at: <iso8601>
  last_modified: <iso8601>    # rewritten on every _write_config
  description: "Local Operator configuration file"
values:
  conversation_length: 100
  detail_length: 15
  max_learnings_history: 50
  hosting: ""
  model_name: ""
  auto_save_conversation: false
```

Missing `values` keys are backfilled from `DEFAULT_CONFIG` on load. Keys read elsewhere but **not** in the defaults: `radient_base_url` (cli.py — with two different in-tree fallbacks, `https://api.radienthq.com` and `https://api.radientlabs.ai`) and `max_conversation_history` (bootstrap.py reads this key while the default dict defines `conversation_length` — a live inconsistency, so the configured value never reaches the executor).
API: `get_config`, `get_config_value(key, default)`, `set_config_value`, `update_config(updates, write=True)`, `update_config_from_args(args)`, `reset_to_defaults`, `_write_config`.

### 3.5 `local_operator/credentials.py` — **NOT encrypted** (167 LOC)

`CredentialManager(config_dir)` stores credentials in **`~/.local-operator/credentials.env` as plaintext `KEY=VALUE` lines**, with `chmod 0600` on creation. The class docstring claims a "local encrypted configuration file" — this is false; there is no crypto in the module (`cryptography>=43.0.1` is pinned purely as a transitive-CVE floor).

API: `load_from_file()`, `write_to_file()`, `_ensure_config_exists()`, `get_credentials()`, `get_credential(key) -> SecretStr` (falls back to `os.environ`, caching into memory), `list_credential_keys(non_empty=True)`, `set_credential(key, value, write=True)`, `prompt_for_credential(key, reason)` (`getpass` + ANSI box).
Values are wrapped in `pydantic.SecretStr` in memory. `write_to_file` rewrites the whole file with no escaping — a value containing a newline corrupts the store.

Also stored here (written by the scheduler/OAuth flow): `GOOGLE_ACCESS_TOKEN`, `GOOGLE_REFRESH_TOKEN`, `GOOGLE_TOKEN_EXPIRY_TIMESTAMP`, plus tool keys `SERP_API_KEY`, `TAVILY_API_KEY`, `FAL_API_KEY`.

### 3.6 `local_operator/env.py` + `.env.template`

`EnvConfig` is a frozen dataclass that misuses `pydantic.Field` as a default value. `get_env_config()` reads `RADIENT_API_BASE_URL` (default `https://api.radienthq.com/v1`) and `RADIENT_CLIENT_ID` (**hardcoded default UUID `b0fd1aa8-05a2-4ca2-bac2-82db293e7584`**). Module import sets `ANONYMIZED_TELEMETRY=false` (for browser-use) and loads `.env` from the package parent.
`.env.template` contains only `RADIENT_API_BASE_URL` and `RADIENT_CLIENT_ID`.

---

## 4. Scheduler, jobs, agents

### 4.1 `local_operator/scheduler_service.py` (924 LOC)

Backed by **APScheduler** (`AsyncIOScheduler`, `CronTrigger`, `DateTrigger`; interval jobs are expressed as cron).

- `_execute_scheduled_task_logic(job_id, agent_id_str, schedule_id_str, prompt, agent_registry_config_dir, env_config, operator_type_str, verbosity_level_str, target_agent_hosting, target_agent_model, status_queue)` (L40) — a **module-level picklable function run in a separate `multiprocessing.Process`**. Creates a fresh event loop, reconstructs `AgentRegistry`/`ConfigManager`/`CredentialManager` from the config dir, calls `initialize_operator(...)`, then `handle_user_input(prompt, additional_instructions=ScheduleInstructionsPrompt)`. Afterwards it reloads agent state, stamps `last_run_at`, pops one-time schedules, deactivates schedules past `end_time_utc`, and pushes `("status_update", job_id, JobStatus.*, payload)` tuples onto `status_queue`. Note `scheduler_service_for_tools = None` — **scheduling tools are unavailable inside a scheduled run**.
- `class SchedulerService` (L204): `__init__(agent_registry, config_manager, credential_manager, env_config, operator_type, verbosity_level, job_manager, websocket_manager)`; `start()` (L669), `shutdown()` (L918), `load_all_agent_schedules()` (L706, replays past-due one-time jobs immediately), `add_or_update_job(schedule: Schedule)` (L472), `remove_job(schedule_id: UUID)` (L655), `_trigger_agent_task(agent_id, schedule_id, prompt)` (L346) -> `create_and_start_job_process_with_queue(...)`.
- **Second responsibility: OAuth token refresh.** `_schedule_radient_token_refresh` (L313) / `_execute_radient_token_refresh_task` (L230) / `add_radient_token_refresh_job_if_needed` (L696) run a cron job (`RADIENT_TOKEN_REFRESH_JOB_ID`, `TOKEN_REFRESH_CRON_MINUTES = "*/15"`) refreshing `GOOGLE_ACCESS_TOKEN` via Radient and rewriting `credentials.env`.

**Schedule model** (`types.py`): `Schedule(id: UUID, agent_id: UUID, prompt: str, interval: int, unit: ScheduleUnit(MINUTES|HOURS|DAYS), start_time_utc, end_time_utc, last_run_at, created_at, is_active: bool, next_run_at, one_time: bool)` with pydantic-v1-style `@validator`s coercing all datetimes to UTC-aware and asserting `end > start`. Persisted per agent in `schedules.jsonl`.

**Pain points:** the state of record is per-agent JSONL files, so the scheduler must reload from disk to observe changes; APScheduler has no job store (schedules are re-registered at every `start()`); one process is forked per scheduled run; pydantic v1 `@validator` is deprecated.

### 4.2 `local_operator/jobs.py` (463 LOC)

- `JobStatus(str, Enum)` — `PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED`.
- `JobContextRecord`, `JobResult`, `Job(BaseModel)` — `id`, `prompt`, `model`, `hosting`, status, timestamps, `task: asyncio.Task | None` (validated by `@field_validator`), `process: Process | None`, `current_execution: CodeExecutionResult`.
- `JobContext` — a context manager that only saves/restores `os.getcwd()` (`change_directory`). "Isolation" is cwd-scoped, nothing more.
- `JobManager` — **in-memory `Dict[str, Job]` guarded by an `asyncio.Lock`; nothing is persisted**. `create_job`, `get_job`, `update_job_status`, `register_task`, `register_process`, `update_job_execution_state`, `cancel_job`, `list_jobs(agent_id, status)`.
- Bridged to child processes by `server/utils/job_processor_queue.py` (`run_job_in_process_with_queue`, `create_and_start_job_process_with_queue`) and the older `server/utils/job_processor.py`.

### 4.3 `local_operator/agents.py` (1838 LOC)

- `AgentData(BaseModel)` (L33) — `id, name, created_date, version, security_prompt, hosting, model, description, last_message, temperature, top_p, top_k, max_tokens, stop, frequency_penalty, presence_penalty, seed, current_working_directory, tags, categories`.
- `AgentEditFields(BaseModel)` (L101) — the mutable subset used by CLI/API/tools.
- `AgentRegistry(config_dir, refresh_interval=5.0)` (L161) — filesystem registry rooted at `~/.local-operator/agents/<agent_id>/`. Metadata in `agent.yml`; a time-based `_refresh_if_needed`/`_refresh_agents_metadata` (L579/L588) re-reads from disk so forked job processes' writes become visible (server uses 3.0s, tests 1.0s).

**On-disk agent layout** (`load_agent_state` L672 / `save_agent_state` L781):

```
~/.local-operator/agents/<agent_id>/
  agent.yml                 # AgentData metadata
  conversation.jsonl        # ConversationRecord[]
  execution_history.jsonl   # CodeExecutionResult[]
  learnings.jsonl           # str[] or {"learning": str}[]
  schedules.jsonl           # Schedule[]
  current_plan.txt
  instruction_details.txt
  system_prompt.md          # per-agent prompt (get_agent_system_prompt)
  context.pkl               # dill-pickled execution context  <-- security hazard
```

Other API: `create_agent`, `save_agent`, `update_agent`, `delete_agent`, `clone_agent`, `get_agent`, `get_agent_by_name`, `list_agents`, `create_autosave_agent`, `export_agent` (zip) / import, `upload_agent_to_radient`, `download_agent_from_radient`, `load_agent_context`/`save_agent_context` (dill).

### 4.4 `local_operator/admin.py` (733 LOC) + `bootstrap.py` (308 LOC)

`admin.py` exposes agent/config self-management to the agent as tools (closure factories): `create_agent_from_conversation_tool`, `save_agent_training_tool`, `list_agent_info_tool`, `create_agent_tool`, `edit_agent_tool`, `delete_agent_tool`, `get_agent_info_tool`, `save_conversation_raw_json_tool`, `get_config_tool`, `update_config_tool`, wired by `add_admin_tools(...)`.

`bootstrap.py` is the **single composition root**, used identically by CLI, server and scheduler:
- `build_tool_registry(executor, agent_registry, config_manager, credential_manager, env_config, model_configuration, scheduler_service=None, status_queue=None)` — conditionally attaches `SerpApiClient`/`TavilyClient`/`FalClient`/`RadientClient` based on which credentials exist, then `tool_registry.init_tools()` + `add_admin_tools(...)`.
- `initialize_operator(operator_type, config_manager, credential_manager, agent_registry, env_config, scheduler_service=None, status_queue=None, request_hosting=None, request_model=None, current_agent=None, persist_conversation=False, auto_save_conversation=False, job_id=None, verbosity_level=VERBOSE) -> Operator` — resolves hosting/model precedence (**agent override > request > config**), loads agent state, maps agent sampling params into `chat_args`, calls `configure_model`, `validate_model` (CLI only), constructs `LocalCodeExecutor` then `Operator`, then the tool registry, then `executor.set_tool_registry(...)`.

**This file is the cleanest thing in the repo and should survive the rewrite nearly intact in shape.**

### 4.5 Tools — `local_operator/tools/`

`ToolRegistry` (`tools/general.py` L1860) is a callable registry (`add_tool(name, fn)`); the agent calls tools as `tools.<name>(...)` **from inside executed Python code**, not via function-calling. `init_tools()` (L1970) registers:
- always: `get_page_html_content`, `get_page_text_content` (Playwright chromium headless), `list_working_directory`, `start_recording`, `stop_recording` (`tools/screen_recorder.py`);
- if any search client: `search_web`;
- if FAL or Radient: `generate_image`, `generate_altered_image`;
- if credential manager: `get_credential`, `list_credentials`, and — only when `GOOGLE_ACCESS_TOKEN` exists — 16 Google tools (`list_gmail_messages`, `get_gmail_message`, `create_gmail_draft`, `send_gmail_message`, `send_gmail_draft`, `update_gmail_draft`, `delete_gmail_draft`, `list_calendar_events`, `create_calendar_event`, `update_calendar_event`, `delete_calendar_event`, `list_drive_files`, `download_drive_file`, `upload_drive_file`, `update_drive_file_metadata`, `update_drive_file_content`);
- if model configuration: `run_browser_task` (**browser-use**, via `_BrowserUseLangChainAdapter` at L1101 — ~250 lines of shim translating LangChain chat models into browser-use's native LLM interface, with `importlib` lazy loading for version drift);
- if agent registry: `schedule_task`, `stop_schedule`, `list_schedules`;
- if Radient key: `send_email_to_user`, `create_audio_transcription`, `create_speech`.

---

## 5. Server — `local_operator/server/` (**integration surface; must stay backward compatible**)

`server/app.py` — `FastAPI(title="Local Operator API", version=<pkg version>, docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")`, permissive CORS (`allow_origins=["*"]`, `allow_credentials=True`). The `lifespan` handler builds `CredentialManager`, `ConfigManager`, `AgentRegistry(refresh_interval=3.0)`, `JobManager`, `WebSocketManager`, `EnvConfig`, `SchedulerService` onto `app.state` and starts/stops the scheduler. `server/dependencies.py` exposes them as FastAPI `Depends` providers. `server/generate_openapi.py` + `make openapi` emit `docs/openapi.json`.

Routers carry no prefix except websockets (`APIRouter(prefix="/v1/ws")`); paths are absolute in the decorators.

**Full endpoint inventory:**

| Method | Path | Handler (file:symbol) |
|---|---|---|
| GET | `/health` | `health.py:health_check` |
| POST | `/v1/chat` | `chat.py:chat_endpoint` |
| POST | `/v1/chat/agents/{agent_id}` | `chat.py:chat_with_agent` |
| POST | `/v1/chat/async` | `chat.py:chat_async_endpoint` -> `JobResultSchema` |
| POST | `/v1/chat/agents/{agent_id}/async` | `chat.py:chat_with_agent_async` |
| POST | `/v1/chat/agents/{agent_id}/edit` | `chat.py:edit_file_with_agent` |
| GET | `/v1/agents` | `agents.py:list_agents` (`page`, `per_page`) |
| POST | `/v1/agents` | `agents.py:create_agent` |
| GET | `/v1/agents/{agent_id}` | `agents.py:get_agent` |
| PATCH | `/v1/agents/{agent_id}` | `agents.py:update_agent` |
| DELETE | `/v1/agents/{agent_id}` | `agents.py:delete_agent` |
| POST | `/v1/agents/{agent_id}/upload` | `agents.py:upload_agent_to_radient` |
| GET | `/v1/agents/{agent_id}/download` | `agents.py:download_agent_from_radient` |
| GET | `/v1/agents/{agent_id}/conversation` | `agents.py:get_agent_conversation` |
| DELETE | `/v1/agents/{agent_id}/conversation` | `agents.py:clear_agent_conversation` |
| POST | `/v1/agents/import` | `agents.py:import_agent` (multipart ZIP) |
| GET | `/v1/jobs` | `jobs.py:list_jobs` (`agent_id`, `status`) |
| GET | `/v1/jobs/{job_id}` | `jobs.py:get_job_status` |
| DELETE | `/v1/jobs/{job_id}` | `jobs.py:cancel_job` |
| POST | `/v1/jobs/cleanup` | `jobs.py:cleanup_jobs` (`max_age_hours=24`) |
| GET | `/v1/config` | `config.py:get_config` |
| PATCH | `/v1/config` | `config.py:update_config` |
| GET | `/v1/config/system-prompt` | `config.py:get_system_prompt` |
| PATCH | `/v1/config/system-prompt` | `config.py:update_system_prompt` |
| GET | `/v1/credentials` | `credentials.py:list_credentials` (keys only) |
| PATCH | `/v1/credentials` | `credentials.py:update_credential` |
| GET | `/v1/models/providers` | `models.py:list_providers` |
| GET | `/v1/models` | `models.py:list_models` (`ModelListQueryParams`) |
| POST | `/v1/agents/{agent_id}/schedules` | `schedules.py:create_schedule_for_agent` |
| GET | `/v1/schedules` | `schedules.py:list_all_schedules` (`page`, `per_page<=100`) |
| GET | `/v1/agents/{agent_id}/schedules` | `schedules.py:list_schedules_for_agent` |
| GET | `/v1/schedules/{schedule_id}` | `schedules.py:get_schedule_by_id` |
| PATCH | `/v1/schedules/{schedule_id}` | `schedules.py:edit_schedule` |
| DELETE | `/v1/schedules/{schedule_id}` | `schedules.py:remove_schedule` |
| POST | `/v1/transcriptions` | `transcription.py:create_transcription_endpoint` (multipart, `model` default `gpt-4o-transcribe`) |
| POST | `/v1/tools/speech` | `speech.py:create_speech` |
| POST | `/v1/agents/{agent_id}/speech` | `speech.py:create_agent_speech` |
| GET | `/v1/static/images` | `static.py:get_image` (`?path=`) |
| GET | `/v1/static/videos` | `static.py:get_video` |
| GET | `/v1/static/audio` | `static.py:get_audio` |
| GET | `/v1/static/html` | `static.py:get_html` |
| WS | `/v1/ws/messages/{message_id}` | `websockets.py:websocket_message_endpoint` |
| WS | `/v1/ws/health` | `websockets.py:websocket_health_endpoint` |

**Response envelope:** everything is wrapped in `CRUDResponse[T]` (`server/models/schemas.py` L110) — a generic `{status, message, result}`. Preserve this shape verbatim.

**Key schemas** (`server/models/schemas.py`, 962 LOC): `ChatOptions`, `ChatRequest`, `ChatStats`, `ChatResponse`, `AgentChatRequest`, `AgentEditFileRequest/Response`, `Agent`, `AgentCreate`, `AgentUpdate`, `AgentListResult`, `AgentGetConversationResult`, `AgentExecutionHistoryResult`, `AgentImportResponse`, `JobResultSchema`, `ConfigUpdate`, `ConfigResponse`, `SystemPromptResponse/Update`, `CredentialUpdate`, `CredentialKey`, `CredentialListResult`, `ModelEntry`, `ModelListResponse`, `ProviderListResponse`, `ModelListQuerySort`, `ModelListQueryParams`, `HealthCheckResponse`, `WebsocketConnectionType`, `ScheduleResponse/CreateRequest/UpdateRequest/ListResponse`, `ExecutionVariable(s)Response`, `SpeechRequest`, `AgentSpeechRequest`.

**Server utils:** `utils/operator.py` (`create_operator` wrapper over `bootstrap.initialize_operator`), `utils/job_processor.py` (305 LOC, older asyncio-task path), `utils/job_processor_queue.py` (415 LOC, current multiprocess + `Queue` path), `utils/websocket_manager.py` (457 LOC), `utils/attachment_utils.py`, `utils/speech_utils.py`.

**Pain points:** `/v1/static/*` serves **arbitrary absolute filesystem paths by query parameter** with only extension-type filtering, combined with `allow_origins=["*"]` — a local-file-read surface. Streaming is push-only via a websocket keyed on `message_id` rather than SSE on the request itself. Chat handlers recompute token stats with `tiktoken` independently of the executor's own accounting.

---

## 6. Tests

**527 test functions** across `tests/unit/` (23 files use `@pytest.mark.asyncio`). `tests/conftest.py` is fully commented out (a disabled `event_loop` fixture); the real fixtures live in `tests/unit/server/conftest.py` (316 LOC).

Layout: `tests/unit/{test_*.py, clients/, model/, tools/, server/{utils/}}`.

| Test file | tests | Covers |
|---|---:|---|
| `test_executor.py` | 45 | code exec, safety, file ops, summarization |
| `server/test_server_agents.py` | 40 | agent CRUD API |
| `test_scheduler_service.py` | 36 | APScheduler wiring, one-time/end-time semantics |
| `test_agents.py` | 34 | registry, on-disk state, import/export |
| `test_console.py` | 31 | ANSI formatting/wrapping |
| `model/test_configure.py` | 30 | provider factory + `validate_model` |
| `test_helpers.py` | 22 | XML/JSON parsing, replacements |
| `clients/test_radient.py` | 21 | Radient client |
| `clients/test_fal.py` | 17 | FAL client |
| `test_cli.py` | 15 | arg parsing + command handlers |
| `test_admin.py` | 15 | admin tools |
| `server/test_server_schedules.py` | 15 | schedules API |
| `server/utils/test_attachment_utils.py` | 14 | attachments |
| `server/test_server_chat.py` | 14 | chat endpoints |
| `test_jobs.py` / `server/test_server_jobs.py` | 13 / 13 | job manager + jobs API |
| `test_config.py` | 12 | config.yml round-trip |
| `test_credentials.py` | 11 | credentials.env round-trip |
| `server/test_server_models.py` / `test_server_config.py` | 10 / 10 | models + config APIs |
| `server/test_server_static.py` | 9 | static file serving |
| `test_operator.py` | 8 | classification + loop (**thin — the main loop is undertested relative to its size**) |
| `server/test_openapi.py` | 8 | spec generation |
| others | — | `test_stream.py` (472 LOC), `test_prompts.py`, `test_notebook.py`, `test_tools.py`, `test_browser_use_compat.py`, `test_job_multiprocessing.py`, `clients/test_serpapi.py`, `clients/test_ollama.py`, `server/test_server_websockets.py`, `server/test_server_transcription.py`, `server/test_server_agent_import_export.py` |

**How they run** (`Makefile`, `pyproject.toml`):
- `make install` -> `./scripts/install_pyenv.sh`, creates `.venv` with Python 3.12, `pip install -e ".[dev]"`
- `make test` -> `pytest` (`[tool.pytest.ini_options] addopts = "-vv -s"`, `testpaths = ["tests"]`; **no `asyncio_mode`, so every async test needs an explicit `@pytest.mark.asyncio`**)
- `make coverage` -> `pytest --cov=local_operator --cov-report=html` -> `htmlcov/` (note: `pytest-cov` is **not** in the dev extras — this target only works if it is installed separately)
- `make format` (black+isort, line-length 100), `make lint` (flake8, max-line-length 100, `extend-ignore=E203`), `make type-check` (pyright), `make security` (pip-audit), `make clean`
- `make server` / `make dev-server` -> `local-operator serve [--reload]`; `make cli`; `make openapi` -> `docs/openapi.json`

**Mocks/fixtures:**
- `local_operator/mocks.py` (190 LOC) — **ships in the production package, not in tests**. `USER_MOCK_RESPONSES: dict` maps lowercase prompt substrings to canned responses (`"hello"`, `"please proceed according to your plan"`, `"print hello world"`, `"think aloud about what you will need to do"`, `"please summarize"`, `"determine a status for the following agent generated json response" -> "[SAFE]"`, ...). `ChatMock` (L61) and `ChatNoop` (L150) implement `ainvoke/invoke/stream/astream` returning `langchain_core.messages.BaseMessage`. Reached in production via `--hosting test` / `noop`.
- `tests/unit/server/conftest.py` — `DummyResponse`, `DummyExecutor`, `DummyOperator`, and fixtures `dummy_executor`, `temp_dir` (tmp_path), `test_app_client` (swaps `app.state.*` for tmp-dir-backed managers and yields an `httpx.AsyncClient` over `ASGITransport`, restoring state after), `dummy_registry`, `mock_create_operator` (patches `local_operator.server.routes.chat.create_operator`), `mock_credential_manager`, `mock_config_manager`, `mock_job_manager`.
- Note: `test_app_client` calls `mock_scheduler_service.start()` without awaiting it (`_ = ...`) — a never-awaited coroutine.

---

## 7. Keep / Adapt / Rewrite

| Module | Verdict | Reason |
|---|---|---|
| `local_operator/stream.py` | **KEEP** | Small, pure, well-tested incremental XML parser; only fix the O(n^2) full re-parse. |
| `local_operator/bootstrap.py` | **KEEP** | Already the clean composition root shared by CLI/server/scheduler; port its shape verbatim. |
| `local_operator/types.py` | **ADAPT** | Good Pydantic core (`ConversationRecord`, `CodeExecutionResult`, `AgentState`, `Schedule`); drop the legacy `DONE/ASK/BYE` `ActionType`s and the triplicated `dict`/`to_dict`/`model_dump`. |
| `local_operator/helpers.py` (PATH half) | **KEEP** | `setup_cross_platform_environment`/`get_windows_registry_path`/`get_posix_shell_path` are hard-won cross-platform fixes. |
| `local_operator/helpers.py` (parse half) | **ADAPT** | Merge `parse_agent_action_xml` + `parse_replacements` with `stream.py` into one protocol module; delete the dead JSON path (`clean_json_response`, `process_json_response`). |
| `local_operator/cli.py` | **ADAPT** | The argparse surface is the public contract and must be preserved verbatim; handler bodies (ANSI box drawing, inline Radient push/pull) should be re-implemented behind a command table. |
| `local_operator/config.py` | **ADAPT** | `config.yml` schema is a contract; replace the ad-hoc `Config` class with a Pydantic model and fix the `conversation_length` vs `max_conversation_history` key mismatch. |
| `local_operator/credentials.py` | **ADAPT** | `credentials.env` location/format is a contract for existing installs; add real at-rest encryption or OS keychain, value escaping, and correct the false "encrypted" docstring. |
| `local_operator/agents.py` | **ADAPT** | On-disk layout (`agent.yml` + `*.jsonl`) must stay readable; drop the `dill` `context.pkl`, replace polling `_refresh_if_needed` with an explicit store. |
| `local_operator/model/registry.py` | **ADAPT** | Keep `ProviderDetail`/`ModelInfo` and the provider list; replace ~1200 lines of hardcoded pricing tables with a fetched/cached catalogue. |
| `local_operator/model/configure.py` | **ADAPT** | Keep `ModelConfiguration` + the base-url-per-provider insight; collapse the `if/elif` chains into a provider descriptor table. |
| `local_operator/clients/*` | **KEEP** | Working REST wrappers with Pydantic models (radient, google, serpapi, fal, tavily, openrouter, ollama); only add a shared base (retry/timeout/async). |
| `local_operator/server/models/schemas.py` | **KEEP** | The published API contract (`CRUDResponse[T]` + all request/response models); changing it breaks integrators. |
| `local_operator/server/routes/*` | **ADAPT** | Paths, verbs and envelope must be byte-identical; handler internals should call the new harness. |
| `local_operator/server/routes/static.py` | **REWRITE** | Serves arbitrary absolute paths by query param under wildcard CORS — needs a scoped, authorised file surface. |
| `local_operator/server/utils/job_processor*.py` | **REWRITE** | Two overlapping implementations (task-based + process+Queue); consolidate to one. |
| `local_operator/jobs.py` | **ADAPT** | `Job`/`JobStatus` shape is exposed via the API; the in-memory `Dict` store and cwd-only `JobContext` need real persistence and isolation. |
| `local_operator/scheduler_service.py` | **ADAPT** | APScheduler + `Schedule` semantics (one-time, end-time, UTC) are worth keeping; split out the Radient/Google OAuth refresh, which does not belong in a scheduler. |
| `local_operator/operator.py` | **REWRITE** | The classify->plan->act triple round trip, CLI printing, signal handling and streaming display are fused into one class; this is the core of the rewrite. |
| `local_operator/executor.py` | **REWRITE** | 3238 LOC doing model I/O + `exec` into a shared dict + safety + file CRUD + summarization + HUD + websocket + job state. Salvage `CodeExecutionError.agent_info_str`, `annotate_code`, and the multimodal `_convert_and_stream` logic. |
| `local_operator/prompts.py` | **REWRITE** | 176 KB of Python string literals; externalise to versioned templates and delete the 19 per-request-type blobs along with the classification round trip that exists to select them. |
| `local_operator/tools/general.py` | **REWRITE** | 2169 LOC mixing a registry, playwright scraping, image gen, credentials and a 250-line browser-use shim; split registry from tool impls and adopt real tool schemas instead of `get_tools_str` docstring reflection. |
| `local_operator/tools/google.py` | **KEEP** | 16 thin closure-factory tools over `google_client.py`; port as-is. |
| `local_operator/tools/screen_recorder.py` | **KEEP** | Self-contained, no coupling. |
| `local_operator/admin.py` | **ADAPT** | Useful self-management tools; re-target at the new registry API. |
| `local_operator/console.py` | **REWRITE** | 553 LOC of hand-rolled ANSI box drawing + `VerbosityLevel` checks leaking into business logic; replace with a renderer behind an interface. |
| `local_operator/mocks.py` | **ADAPT** | Keep `ChatMock`/`ChatNoop` (used by `--hosting test` **and** the server suite) but move substring-matched canned responses out of the shipped package. |
| `local_operator/env.py` | **ADAPT** | Fix the `@dataclass` + `pydantic.Field` default misuse and the hardcoded `RADIENT_CLIENT_ID` UUID. |
| `local_operator/notebook.py`, `logger.py` | **KEEP** | Small, single-purpose (Jupyter export of code history; log-level config). |
| `tests/unit/server/conftest.py` | **ADAPT** | The `test_app_client` app.state-swap pattern is reusable; fix the un-awaited `scheduler_service.start()`. |

---

## 8. Dependencies

Declared in `pyproject.toml` (`[project].dependencies`) and duplicated — **with different pins** — in `setup.py` and `requirements.txt`. `pyproject` pins `browser-use==0.4.0` while `setup.py` pins `browser-use==0.1.45`. **Consolidating to one manifest is a prerequisite for any rewrite.**

### Keep

| Dep | Used at | Why |
|---|---|---|
| `pydantic` | everywhere | The whole data model + API contract. |
| `fastapi`, `uvicorn`, `python-multipart`, `websockets` | `server/` | The integration surface. |
| `requests` | `clients/*` | All 7 provider clients (could migrate to `httpx`, which arrives anyway via fastapi/tests). |
| `pyyaml` (transitive) + `jsonlines` | `config.py`, `agents.py` | On-disk formats are a compat contract. |
| `apscheduler` | `scheduler_service.py` | Real cron/date trigger semantics; not worth reimplementing. |
| `tiktoken` | `executor.py`, `routes/chat.py` | Token accounting/limits. |
| `python-dotenv` | `env.py` | `.env` loading. |
| `psutil` | `prompts.py:get_system_details_str` | System details in the prompt (single use — droppable for `platform`+`os`). |
| `pillow`, `pillow-heif` | `helpers.py` | HEIC->PNG for multimodal attachments; already optional-guarded. |
| `pyreadline3` | `operator.py` (Windows only) | CLI history parity on Windows. |
| `python-dateutil`, `six`, `certifi`, `idna`, `urllib3`, `jinja2`, `cryptography`, `setuptools`, `configobj`, `twisted` | — | Transitive CVE floors, not direct imports. Prune once the tree above them changes. |

### LangChain — reduce, don't remove

Currently: `langchain`, `langchain-community`, `langchain-core`, `langchain-openai`, `langchain-ollama`, `langchain-anthropic`, `langchain-google-genai`.
Actual usage is narrow:
- `langchain_core.messages.BaseMessage` — the streaming chunk type (`executor.py`, `mocks.py`).
- `langchain_openai.ChatOpenAI`, `langchain_ollama.ChatOllama`, `langchain_anthropic.ChatAnthropic`, `langchain_google_genai.ChatGoogleGenerativeAI` — the 4 concrete clients in `model/configure.py`.
- `langchain_community.callbacks.manager.get_openai_callback` — **one import**, for token accounting in `executor.py`.
- The meta-package **`langchain` is never imported** (only `logger.py` sets its log level by name).

Recommendation: **drop the `langchain` meta-package and `langchain-community`** (replace `get_openai_callback` with usage metadata already present on response chunks); keep the four provider adapters, or replace all of them with the official `openai`/`anthropic`/`google-genai` SDKs — 8 of 11 providers are already just `ChatOpenAI` with a custom `base_url`, so the LangChain abstraction is earning very little.

### Drop

| Dep | Verdict |
|---|---|
| **`faiss-cpu==1.10.0`** | **Dead. Zero imports anywhere in `local_operator/` or `tests/`** — it appears only in the three manifests. A ~30 MB wheel with no call site. Delete. |
| `browser-use` | Drop or make an optional extra. Pinned inconsistently (`0.4.0` vs `0.1.45`), imported under try/except, and requires a ~250-line `_BrowserUseLangChainAdapter` compat shim plus a dedicated `tests/unit/test_browser_use_compat.py` to survive its API churn. It powers exactly one tool (`run_browser_task`). |
| `dill` | Drop with `context.pkl`. Pickled arbitrary Python objects loaded from disk at agent startup is an unnecessary code-execution surface, and it is why execution context cannot be safely shared or migrated. |
| `langchain` (meta), `langchain-community` | See above — one function's worth of value. |
| `playwright` | Keep **only** if `get_page_html_content`/`get_page_text_content` stay in the core harness; it is a heavy install (browser download) for two scraping tools and should be an optional extra. |

### Missing / broken

- `pytest-cov` is required by `make coverage` but absent from `[project.optional-dependencies].dev`.
- `pyyaml` is imported directly (`config.py`, `agents.py`) but never declared — it arrives transitively via langchain, so dropping langchain breaks config loading unless it is declared.
- `google-api-python-client` is **not** used; `clients/google_client.py` hand-rolls the REST calls with `requests` (deliberate, and worth keeping).
- Three manifests (`pyproject.toml`, `setup.py`, `requirements.txt`) disagree. `pyproject.toml` is authoritative for the build (`setuptools.build_meta`); `setup.py` is a stale shadow.

