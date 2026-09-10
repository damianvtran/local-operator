# OpenRouter provider preferences — `/settings` frames

Captured from the real `OperatorApp` (`run_test`, `scripts.visual_capture.
save_capture`, `isolate_capture` before app imports) on the feature branch,
120x44. The rows render straight from the `settings_io` registry; nothing in
the settings page needed a change for them to appear.

- `sort.svg` — the routing-policy ENUM row at rest, showing `—` (unset).
- `sort-open.svg` — the row expanded: `default` ("no preference — sticky
  routing stays on (warmest cache)") beside the three real policies.
- `order-warning.svg` — the `order` row's detail line carrying the HARD
  warning: disables sticky routing, cold DeepSeek prompt cache.
- `ignore-note.svg` — the `ignore` row's detail line with the soft
  cold-start note.

Wire-shape proof (not pixels): with defaults the emitted chat-completions
body has NO `provider` key; with `sort=throughput` + `ignore=[groq,together]`
set through the settings write path the body carries exactly
`{"sort": "throughput", "ignore": ["groq", "together"]}` beside the untouched
`model`/`messages` keys. Reproduced in
`tests/unit/providers/test_clients.py` (`test_openrouter_provider_preferences_
reach_the_request_body`, `test_openrouter_body_omits_provider_key_when_
unconfigured`).
