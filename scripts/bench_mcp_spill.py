#!/usr/bin/env python3
"""Deterministic footprint benchmark for oversized MCP tool results.

The baseline formats the same synthetic server result without a ``ToolContext``,
which is the pre-spill behavior. The compact arm uses the production MCP bridge,
spill store, and production message token estimator, then reads the saved text
back to prove the advertised expansion path is real.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.cutpoint import _message_tokens  # noqa: E402
from local_operator.harness.types import Message, ToolContext  # noqa: E402
from local_operator.mcp.tool_bridge import format_mcp_result  # noqa: E402
from local_operator.tools.spill import get_store  # noqa: E402

MIN_MODEL_TOKEN_REDUCTION = 0.50
MIN_METADATA_BYTE_REDUCTION = 0.50


@dataclass(frozen=True)
class BenchmarkResult:
    baseline_model_tokens: int
    compact_model_tokens: int
    model_token_reduction_percent: float
    baseline_metadata_bytes: int
    compact_metadata_bytes: int
    metadata_reduction_percent: float
    spill_bytes: int
    spill_complete: bool
    recovery_matches: bool


def _tool_message(text: str) -> Message:
    message = Message(role="tool", tool_call_id="call", tool_name="mcp__bench_rows")
    message.content = Message.user(text).content
    return message


def _metadata_bytes(details: dict[str, Any] | None) -> int:
    return len(json.dumps(details, sort_keys=True, separators=(",", ":")).encode())


def run_benchmark(
    config_dir: Path, *, rows: int = 1_000, value_chars: int = 256
) -> BenchmarkResult:
    """Measure one under-cap payload so spill recovery can be asserted exactly."""
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
    lines = [f"row {index:04d}: " + (f"value-{index % 10} " * value_chars) for index in range(rows)]
    text = "\n".join(lines)
    server_result = {
        "content": [{"type": "text", "text": text}],
        "structuredContent": {"rowCount": rows, "columns": ["id", "value"]},
        "isError": False,
        "meta": {"cursor": "next-page", "protocolVersion": "2025-06-18"},
    }

    baseline = format_mcp_result(server_result, "call", "mcp__bench_rows")
    compact = format_mcp_result(
        server_result,
        "call",
        "mcp__bench_rows",
        ToolContext(session_id="mcp-benchmark"),
    )
    assert compact.details is not None
    spill = compact.details.get("spill")
    assert isinstance(spill, dict)
    handle = str(spill["handle"])
    recovered = get_store().read_lines(handle, 1, rows)
    recovered_text = "\n".join(recovered[0]) if recovered is not None else ""

    baseline_tokens = _message_tokens(_tool_message(baseline.text))
    compact_tokens = _message_tokens(_tool_message(compact.text))
    baseline_bytes = _metadata_bytes(baseline.details)
    compact_bytes = _metadata_bytes(compact.details)
    return BenchmarkResult(
        baseline_model_tokens=baseline_tokens,
        compact_model_tokens=compact_tokens,
        model_token_reduction_percent=round((1 - compact_tokens / baseline_tokens) * 100, 2),
        baseline_metadata_bytes=baseline_bytes,
        compact_metadata_bytes=compact_bytes,
        metadata_reduction_percent=round((1 - compact_bytes / baseline_bytes) * 100, 2),
        spill_bytes=int(spill["bytes"]),
        spill_complete=bool(spill["complete"]),
        recovery_matches=recovered_text == text,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--value-chars", type=int, default=256)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lop-mcp-bench-") as tmp:
        result = run_benchmark(Path(tmp), rows=args.rows, value_chars=args.value_chars)

    report = asdict(result)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"model-visible tokens: {result.baseline_model_tokens:,} -> {result.compact_model_tokens:,}"
        )
        print(f"model-token reduction: {result.model_token_reduction_percent:.2f}%")
        print(
            f"transcript metadata: {result.baseline_metadata_bytes:,} -> {result.compact_metadata_bytes:,} bytes"
        )
        print(f"metadata reduction: {result.metadata_reduction_percent:.2f}%")
        print(f"spill: {result.spill_bytes:,} bytes, complete={result.spill_complete}")
        print(f"recovery matches: {result.recovery_matches}")

    passed = (
        result.model_token_reduction_percent >= MIN_MODEL_TOKEN_REDUCTION * 100
        and result.metadata_reduction_percent >= MIN_METADATA_BYTE_REDUCTION * 100
        and result.spill_complete
        and result.recovery_matches
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
