"""OAuth flows: shared machinery (PKCE, loopback callback server, device-code
poller) plus per-provider implementations (Anthropic, OpenAI/ChatGPT, Kimi,
xAI). Ported 1:1 from omp ``packages/ai/src/registry/oauth`` — endpoint URLs,
client ids, ports, and pitfalls are documented per file."""
