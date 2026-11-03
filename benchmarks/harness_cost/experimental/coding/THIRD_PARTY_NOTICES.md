# Coding lane attribution and license examination

The original benchmark glue was ported from `harness-cost-bench/coding`, with
portability/authentication/safety changes documented in README.md. It is covered
by the enclosing benchmark kit's MIT license (`../../LICENSE`, Copyright (c)
2024 Radient Inc.). No separate upstream license header was present in those
source glue modules; no attribution header was removed.

`codex-model-catalog.json` is the source lane's configured Codex catalog. It
includes the upstream OpenAI Codex `model_messages` template, not just numeric
metadata. That external template is retained, not claimed as original kit text.
Codex source is Apache-2.0, OpenAI copyright 2025. The full upstream license and
notice are preserved beside it as **CODEX-LICENSE** and **CODEX-NOTICE**, copied
from the locally available Codex source checkout's LICENSE/NOTICE without
network access. Its NOTICE also credits Ratatui; retain the complete notice even
though this kit redistributes no Ratatui binary/source. Catalog overrides are a
benchmark treatment (model identity, context/output/search capabilities), not an
upstream endorsement. The catalog itself is copied byte-for-byte from the source
lane; provenance of that source's generated metadata is not fresh model validation.

`codex_bridge.py` preserves the source lane's original translator implementation
and explanatory module docstring. The original source identifies its parent as
`codex_toolsearch_bridge.py` in the benchmark rig. It is not copied OpenAI or
LiteLLM implementation code. The rewrite routes are a benchmark compatibility
layer and are explicitly labelled codex-best.

`adapters.py` imports Harbor classes at runtime (Harbor 0.25.0); it does not vendor
Harbor source. Harness packages, Linux toolchains, containers, dependency trees,
Terminal-Bench task/reference-solution payloads, private fixtures, MCP servers
and skills are not redistributed here. Obtain them separately and examine their
own licenses before executing. The external Terminal-Bench dataset identifier,
task names and image hashes in subset.json are provenance metadata, not copied
task text or reference implementations.

The common public MCP schema snapshot and its third-party notices remain at
`../../public_loadout` and `../../THIRD_PARTY_NOTICES.md`; no schema copy is added
here. This records what was examined locally, not legal advice or a statement
that unbundled payloads are relicensable under the kit's MIT license.
