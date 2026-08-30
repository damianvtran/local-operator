#!/usr/bin/env python3
"""Always-on prompt cost, on the basis ``scripts/bench_role_overhead.py`` fixes.

Run in a worktree of each ref and diff the two outputs; that is how the
before/after table in ``token-cost.txt`` is reproduced. The basis matters more
than the absolute numbers: a system prompt costs the tokens of its rendered
markdown, and a tool costs the tokens of ``{name, description, parameters}`` as
JSON, which is what rides the provider's ``tools`` array. Measuring a tool by
its schema alone understates it, because ``_render_tool_inventory`` also emits
the full description into system block ``[1]`` — so description text is billed
twice, which is why this change put its prose in the prompt rather than in the
two web tools' descriptions.

    .venv/bin/python docs/evidence/web-research-disposition/token_probe.py
"""

from __future__ import annotations

import json

import tiktoken

from local_operator.harness.types import ToolContext
from local_operator.prompts_api import build_system_blocks
from local_operator.tools.registry import create_tools


def main() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")

    def count(text: str) -> int:
        return len(encoding.encode(text))

    tools = create_tools(ToolContext(cwd="."))
    blocks = build_system_blocks(tools, "", "Platform: x", "2026-01-01")
    tools_array = sum(
        count(
            json.dumps(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )
        )
        for tool in tools
    )
    rows = {
        "system_prompt": count(blocks[0]),
        "tool_inventory": count(blocks[1]),
        "tools_array": tools_array,
    }
    rows["always_on_total"] = sum(rows.values())
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
