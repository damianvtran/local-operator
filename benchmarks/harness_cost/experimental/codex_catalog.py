"""Write a Codex `model_catalog_json` entry for a non-OpenAI model (claude-sonnet-5-5).

WHY: Codex 0.147.0 resolves per-model behaviour from its model catalog. A slug
it does not know (our `claude-sonnet-5-5` on a custom `model_provider`) gets
FALLBACK metadata (codex-rs/models-manager/src/model_info.rs,
`model_info_from_slug`), and Codex warns at startup: "Model metadata for
`claude-sonnet-5-5` not found. Defaulting to fallback metadata; this can
degrade performance and cause issues." Two fallback values matter here:

  * `supports_search_tool: false` -> MCP tools are never deferred behind
    Codex's `tool_search` (core/src/tools/spec_plan.rs,
    `search_tool_enabled = model_info.supports_search_tool &&
    provider.capabilities().namespace_tools`); all 480+ MCP schemas ship on
    every call.
  * `context_window: 272000` (and `max_context_window: 272000`, which caps
    the `model_context_window` config override) -> auto-compaction fires at
    90% = 244.8k tokens, below the ~376k-token full MCP loadout, so Codex
    compacts after EVERY turn.

The supported, user-facing fix is the top-level config key
`model_catalog_json = "<path>"` (config.schema.json: "Optional path to a JSON
model catalog (applied on startup only)"), which replaces the catalog with a
static one (model-provider/src/provider.rs, `StaticModelsManager`).

WHAT this writes: one ModelInfo that is byte-for-byte the fallback metadata
Codex already used for this slug, EXCEPT for the explicitly listed overrides,
so a comparison isolates exactly those knobs:
  --search-tool        supports_search_tool = true   (MCP deferral)
  --context-window N   context_window = max_context_window = N
                       (Sonnet 5.5 serves 1,000,000 tokens per the
                       provider's model listing)
  --reasoning LEVEL    default_reasoning_level = LEVEL plus the
                       supported_reasoning_levels list. Fallback metadata has
                       NO default level, so Codex sends no `reasoning.effort`
                       and the translator sends no `thinking`: Codex ran
                       Sonnet with thinking off while the other harnesses ran
                       adaptive thinking.
The instructions template is Codex's own `models-manager/prompt.md` at the
matching tag (the text fallback metadata uses), read from --prompt (a path to
that file in a checkout of openai/codex at rust-v0.147.0).

Constraints: the catalog must be valid for the exact Codex version being
benchmarked (field set from protocol/src/openai_models.rs at rust-v0.147.0);
validate with `codex debug models -c model_catalog_json=<path>`, which loads
the catalog without any model call.

Usage: codex_catalog.py --prompt <codex>/codex-rs/models-manager/prompt.md \
           --out codex-model-catalog.json [--search-tool] [--context-window N] [--reasoning high]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SLUG = "claude-sonnet-5-5"


def fallback_entry(slug: str, instructions: str) -> dict:
    """Mirror of `model_info_from_slug` (rust-v0.147.0), field for field."""
    return {
        "slug": slug,
        "display_name": slug,
        "description": None,
        "default_reasoning_level": None,
        "supported_reasoning_levels": [],
        "shell_type": "default",
        "visibility": "none",
        "supported_in_api": True,
        "priority": 99,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "default_service_tier": None,
        "availability_nux": None,
        "upgrade": None,
        "model_messages": {
            "instructions_template": instructions,
            "instructions_variables": None,
            "approvals": None,
            "collaboration_modes": None,
            "auto_review": None,
            "permissions": None,
        },
        "include_skills_usage_instructions": False,
        "include_plugin_usage_instructions": False,
        "include_apps_usage_instructions": False,
        "supports_reasoning_summary_parameter": True,
        "default_reasoning_summary": "auto",
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_parallel_tool_calls": False,
        "supports_image_detail_original": False,
        "context_window": 272000,
        "max_context_window": 272000,
        "auto_compact_token_limit": None,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"],
        "supports_search_tool": False,
        "use_responses_lite": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--slug", default=SLUG)
    ap.add_argument("--search-tool", action="store_true")
    ap.add_argument("--context-window", type=int)
    ap.add_argument("--reasoning", choices=["low", "medium", "high", "xhigh"])
    a = ap.parse_args()
    m = fallback_entry(a.slug, a.prompt.read_text())
    if a.search_tool:
        m["supports_search_tool"] = True
    if a.context_window:
        m["context_window"] = m["max_context_window"] = a.context_window
    if a.reasoning:
        m["default_reasoning_level"] = a.reasoning
        m["supported_reasoning_levels"] = [
            {"effort": e, "description": e} for e in ("low", "medium", "high", "xhigh")]
    a.out.write_text(json.dumps({"models": [m]}, indent=1) + "\n")
    print(json.dumps({k: m[k] for k in ("slug", "supports_search_tool", "context_window", "max_context_window", "default_reasoning_level")}))


if __name__ == "__main__":
    main()
