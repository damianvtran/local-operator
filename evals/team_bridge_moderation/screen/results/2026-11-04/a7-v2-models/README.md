# A7 (2026-11-04): production v2 across seven screening models (MEASUREMENT ONLY)

No template is judged here. Every corpus file is NON-HELD-OUT: ho2 was spent in A5,
ho3 in A6, and `relay.jsonl` is the B4 targeted set. The numbers describe production
`bridge-screen-v2` per screening model; they adopt nothing.

- `trials.jsonl` / `plan.json`: the new columns, `openai/gpt-6-astra`, `openai/gpt-6.1-sol`,
  `anthropic/claude-haiku-5-5`, `anthropic/claude-haiku-4-5`, v2 only, N=5, 2,760 trials,
  0 invalid. The Opus, Sonnet and radient/auto v2 columns are the v2 arm of
  `../v5-b5/trials.jsonl` (same template bytes, same corpus bytes, same day).
- `tables.md`: accuracy, cost/latency and unsafe-bypass tables (copied into PROTOCOL A7).
- `anthropic_prompt_tokens.json`: Anthropic usage reports the screen's prompt as a
  5-minute CACHE WRITE (`input_tokens: 4`, `cache_write_tokens: ~730`), so the bench's
  `input_tokens` column reads 4 for Opus/Sonnet/Haiku 5.5. The full prompt size was
  measured on 12 sampled items per model through the same `ScreenLeg` path and is
  costed at the 5m cache-write rate; no cache read occurred (each screen is a fresh
  nonce, so the prompt never repeats).
- Model ids are the harness's: `build_model_spec("openai", "gpt-6-astra")` etc. Effort:
  the screener does not carry a per-target effort (`model_boundary.resolve_screening_chain`);
  each spec's own default applies (Opus `medium`, Haiku 5.5 `high`, GPT none set).
