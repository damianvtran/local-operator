#!/usr/bin/env python3
"""Measure a fresh session's start context, component by component.

Why chars are the primary unit
------------------------------
The analytics ledger (``~/.local-operator/analytics.db``, ``calls`` table,
``c_*`` columns) apportions a call's BILLED prompt tokens across components in
proportion to each component's CHARACTER count. The ratios are therefore exact
by construction, but the chars->tokens rate is whatever the provider's
tokenizer produced — measured at ~2.78 chars/token on this surface, not
cl100k's ~4.22. Projecting a saving with cl100k understates it by ~45%.

So: this script reports CHARACTERS as the ground truth and converts at the
measured ledger rate. A cl100k column is printed alongside only because it is
the number a reader can independently reproduce; it is not the estimate to
quote for a billed-token saving.

Usage:
    .venv/bin/python scripts/measure_start_context.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.harness.types import AgentTool  # noqa: E402
from local_operator.prompts_api import build_system_blocks  # noqa: E402
from scripts.real_tool_surface import build_real_tools  # noqa: E402

#: Measured billed-token rate for this prompt surface. See the module
#: docstring: this is the ledger's own rate, not a generic tokenizer's.
CHARS_PER_BILLED_TOKEN = 2.78


def tool_schema_chars(tools: list[AgentTool]) -> int:
    """Characters the provider tools array costs.

    Matches what ``Session.context_breakdown`` counts: name, description and
    the JSON-serialized parameter schema, per tool.
    """
    return sum(
        len(tool.name) + len(tool.description or "") + len(json.dumps(tool.parameters or {}))
        for tool in tools
    )


def measure(*, with_browser: bool = True) -> dict[str, int]:
    tools = build_real_tools(str(REPO))
    if not with_browser:
        tools = [t for t in tools if t.name != "browser"]
    blocks = build_system_blocks(
        tools,
        skills_block="<skills/>",
        env_details="cwd: /tmp\nplatform: Darwin 25.6.0",
        date_str="2026-09-07",
    )
    return {
        "instructions": len(blocks[0]),
        "tool_inventory": len(blocks[1]),
        "environment": len(blocks[2]),
        "knowledge": len(blocks[3]),
        "tool_schemas": tool_schema_chars(tools),
        "n_tools": len(tools),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = {"with_browser": measure(with_browser=True), "no_browser": measure(with_browser=False)}
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    for label, data in rows.items():
        total = sum(v for k, v in data.items() if k != "n_tools")
        print(f"\n=== {label} ({data['n_tools']} tools) ===")
        for key, chars in data.items():
            if key == "n_tools":
                continue
            print(f"  {key:16} {chars:>8,} chars  ({chars / CHARS_PER_BILLED_TOKEN:>8,.0f} billed)")
        print(f"  {'TOTAL':16} {total:>8,} chars  ({total / CHARS_PER_BILLED_TOKEN:>8,.0f} billed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
