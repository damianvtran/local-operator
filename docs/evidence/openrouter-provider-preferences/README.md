# OpenRouter provider preferences — `/settings` frames

Captured from the real `OperatorApp` (`run_test`, `scripts.visual_capture.
save_capture`, `isolate_capture` before app imports) on the feature branch,
120x44. Round-1 recapture after the dedicated "OpenRouter routing" section,
the pinned danger-ink `order` warning, the empty-LIST placeholder ghost, and
the shortened `max_price` help.

- `sort-open.svg` — the routing-policy ENUM expanded under the new section:
  `default` ("no preference — sticky routing stays on") beside price /
  throughput / latency.
- `order-default.svg` — the `order` row at rest. The HARD warning
  (`disables sticky routing — prompt cache goes cold`) leads the detail in
  danger ink.
- `order-offdefault.svg` — the same row after committing `deepseek, groq`.
  The warning remains beside `default: —` — it is not shed when the
  dangerous value is stored.
- `ignore-editor.svg` — empty LIST editor with the common-slug ghost
  (`deepseek, groq, mistral, …`) and `empty = no opinion` on the detail.
- `max-price-rest.svg` — shortened help
  `{"prompt": 1, "completion": 2} — USD / million tokens` fully visible at
  120 columns.

Wire-shape proof (not pixels): with defaults the emitted chat-completions
body has NO `provider` key; with `sort=throughput` + `ignore=[groq,together]`
set through the settings write path the body carries exactly
`{"sort": "throughput", "ignore": ["groq", "together"]}` beside the untouched
`model`/`messages` keys. A cache-capable model with both preferences and
`prompt_cache_key` set carries BOTH keys on one body. Reproduced in
`tests/unit/providers/test_clients.py`.
