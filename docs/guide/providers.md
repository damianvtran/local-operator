<!-- Moved verbatim from the project README (2026-11-08 README refresh); the README now links here. -->

# Providers

Every provider below runs on the same harness and the same model picker.
OAuth providers sign in through the browser and use your existing
subscription; API-key providers prompt once and store the key locally. Local
servers can be configured in the app with `/login`, including endpoints and
optional masked API tokens. See
[Local and self-hosted providers](../../docs/LOCAL_PROVIDERS.md) for setup,
metadata overrides, desktop-app support, and server-specific limitations.

| Provider | Access |
| --- | --- |
| OpenAI / ChatGPT | `openai` / `openai-device`, or `OPENAI_API_KEY` |
| Anthropic / Claude | `anthropic`, or `ANTHROPIC_API_KEY` |
| Kimi (Moonshot) | `kimi`, or `KIMI_API_KEY` |
| Meta Muse Code (unofficial subscription) | `meta-muse` device approval; no PAYG-key fallback |
| xAI / Grok | `xai-oauth`, or `xai` / `XAI_API_KEY` |
| Z.AI (GLM) | `zai-oauth`, or `zai` / `ZAI_API_KEY` |
| Qwen (Alibaba) | `alibaba-token-plan-oauth`, or API key |
| Google Gemini | `GOOGLE_AI_STUDIO_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY`: one key, many models |
| Radient | `RADIENT_API_KEY`: automatic per-step model selection |
| LM Studio, Ollama, vLLM, llama.cpp | User-operated server; optional token |
| OpenAI-compatible | Explicit server URL; optional token; MLX/LocalAI/proxy escape hatch |

```bash
lop login              # list login-capable providers
lop login openai       # OAuth flow; for OpenAI/Anthropic/Z.AI/Qwen, repeat to add a second account
lop login-status       # what's signed in
lop logout kimi
```

Legacy `--hosting <name> --model <name>` flags keep working, and API keys can
be stored with `lop credential update <KEY_NAME>` (a masked prompt); the
`lop secret` store, and the `/credential` hand-over that keeps a secret out of
the prompt text, are covered in
[Credentials and secrets](configuration.md#credentials-and-secrets).
