# Third-party material

The `model_messages.instructions_template` embedded in both
`codex-model-catalog.json` and `codex-model-catalog.eager.json` is verbatim
OpenAI Codex `codex-rs/models-manager/prompt.md`, tag `rust-v0.147.0`:
https://github.com/openai/codex/blob/rust-v0.147.0/codex-rs/models-manager/prompt.md

Copyright 2025 OpenAI. Licensed under Apache License 2.0; see
`licenses/CODEX-APACHE-2.0.txt` and preserved `licenses/CODEX-NOTICE.txt`.
The prompt SHA-256 is
`ac8ae107a0d72fe3476b430afb161ea4e67da2e446d778aefc44828160559807`.
No modifications were made to that prompt. The catalog surrounding it is a
benchmark configuration derived from Codex fallback metadata, not an OpenAI
endorsement of the tested model or translator bridge. `codex_catalog.py`
reproduces that version's fallback field layout with benchmark overrides.
