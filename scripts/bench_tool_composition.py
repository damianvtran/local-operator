"""Verify a prescribed three-read pipeline through each checkout's real loop.

This is an orchestration benchmark, not an LLM quality/latency measurement.
The scripted provider chooses three native calls on the baseline and one eval
composition when the bridge exists. Both execute the same three reads and must
produce the same answer. Count model boundaries and retained result bytes;
do not infer that an unprompted model will always choose the optimal plan.
Run with each checkout's own interpreter and --repo, as with bench_tool_io.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))

from local_operator.harness.loop import AgentLoop, LoopContext  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    AgentTool,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolResult,
)
from tests.unit.harness.test_loop import ScriptedStream  # noqa: E402


def call(name: str, arguments: dict[str, Any], index: int) -> list[StreamEvent]:
    return [
        StreamToolCallDelta(
            index=0, id=str(index), name=name, argument_delta=json.dumps(arguments)
        ),
        StreamEndEvent(stop_reason="toolUse"),
    ]


async def measure() -> dict[str, Any]:
    executed = []

    async def read(call_id, params, signal, update, context):
        executed.append(params["index"])
        return ToolResult(
            tool_call_id=call_id,
            tool_name="records",
            content=[
                TextContent(
                    text=json.dumps(
                        {
                            "value": params["index"] * 2,
                            "unused": "x" * 50000,
                        }
                    )
                )
            ],
        )

    remote = AgentTool(
        name="records", description="Read a record page", parameters={}, execute=read
    )
    bridge = "dispatch_tool" in ToolContext.model_fields
    with tempfile.TemporaryDirectory(prefix="lo-composition-bench-") as scratch:
        context = ToolContext(cwd=scratch, session_id=scratch)
        if bridge:
            from local_operator.tools.eval import build_eval_tool

            tools = [build_eval_tool()]
            code = (
                "import json\n"
                'sum(json.loads(tool("records", index=i)["content"][0]["text"])["value"] '
                "for i in range(3))"
            )
            turns = [call("eval", {"code": code}, 0)]
        else:
            tools = [remote]
            turns = [call("records", {"index": index}, index) for index in range(3)]
        turns.append([StreamTextDelta(delta="complete"), StreamEndEvent(stop_reason="stop")])
        stream = ScriptedStream(turns)
        state = LoopContext(tools=tools, tool_context=context)
        config = LoopConfig(
            model=ModelSpec(provider="test", model_id="composition"),
            stream_fn=stream,
            resolve_fallback_tool={remote.name: remote}.get,
            convert_to_llm=lambda rows: [row for row in rows if isinstance(row, Message)],
        )
        try:
            async for _ in AgentLoop().run(
                [Message.user("Sum values from pages 0, 1, 2")], state, config
            ):
                pass
            results = [m for m in state.messages if isinstance(m, Message) and m.role == "tool"]
            assert executed == [0, 1, 2]
            assert all(not result.is_error for result in results)
            if bridge:
                assert "result: 6" in results[0].text
            else:
                assert sum(json.loads(result.text)["value"] for result in results) == 6
            return {
                "source": str(args.repo.resolve()),
                "python": sys.executable,
                "loop_sha256": hashlib.sha256(
                    (args.repo / "local_operator/harness/loop.py").read_bytes()
                ).hexdigest(),
                "plan": "composed" if bridge else "native",
                "accepted": True,
                "actual_reads": len(executed),
                "model_requests": len(stream.requests),
                "retained_tool_result_bytes": sum(len(m.text.encode()) for m in results),
            }
        finally:
            if bridge:
                from local_operator.tools.eval import close_session_kernel

                await close_session_kernel(scratch)


report = asyncio.run(measure())
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
