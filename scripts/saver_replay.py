#!/usr/bin/env python
"""Replay REAL request snapshots through the reference reducer and this client's gate.

WHY THIS IS COMMITTED. Round 2 shipped a check-6 verifier whose evidence was a
synthetic conformance corpus with no `i` argument and no heredoc, so it accepted
31/31 there while rejecting 108 of 133 honest epochs from a real session. The
compatibility claim has to be measured on real traffic, not on the shapes we
happened to write down, so the driver itself is part of the repository.

WHAT IT DOES. For every `*.json` request snapshot in a directory: project the
request's messages exactly as `saver_projection` uploads them, ask the REFERENCE
reducer (`radient_context_reduction.saver.prepare`, a separate checkout, never a
stub) to prepare them, and hand its epoch bytes to `validate_response`. Nothing
here contacts Radient, a provider, or the network.

USAGE (needs the reference checkout importable, which is why it is a script and
not a pytest module):

    PYTHONPATH=~/tools/radient-context-reduction:. .venv/bin/python \\
        scripts/saver_replay.py ~/tools/radient-context-reduction/results/fixtures

Exit code is 0 when every epoch the reference produced was accepted; any
rejection prints its reason and the offending record so it can be classified as
an honest shape this verifier must learn or a real contract violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from local_operator.harness.types import Message, TextContent, ToolCall
from local_operator.saver.canonical import canonical_bytes
from local_operator.saver.projection import saver_projection, verbatim_wire
from local_operator.saver.validate import SaverRejected, validate_response


def _arguments(value: Any) -> dict[str, Any]:
    """A snapshot's tool arguments as a dict; anything else is no arguments.

    `messages_from_blocks` reads untyped JSON, so `isinstance` at the call site
    left the value `str | Unknown | dict` to pyright. Narrowed here instead.
    """
    return value if isinstance(value, dict) else {}


def messages_from_blocks(body: dict[str, Any]) -> list[Message]:
    """Anthropic-shaped request -> lop rows, the way a session would hold them."""
    out: list[Message] = []
    for row in body.get("messages") or []:
        role = row.get("role")
        content = row.get("content")
        blocks = (
            [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
        )
        texts = [b for b in blocks if isinstance(b, dict) and isinstance(b.get("text"), str)]
        results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
        uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        if role in ("system", "developer"):
            continue
        if role == "user" and results:
            for block in results:
                out.append(
                    Message(
                        role="tool",
                        tool_call_id=str(block.get("tool_use_id") or ""),
                        tool_name="tool",
                        is_error=bool(block.get("is_error")),
                        content=[TextContent(text=_flat(block.get("content")))],
                    )
                )
            if texts:
                out.append(
                    Message(role="user", content=[TextContent(text=b["text"]) for b in texts])
                )
            continue
        call = [
            ToolCall(
                id=str(b.get("id") or ""),
                name=str(b.get("name") or "tool"),
                arguments=_arguments(b.get("input")),
            )
            for b in uses
        ]
        out.append(
            Message(
                role=role,
                content=[TextContent(text=b["text"]) for b in texts],
                tool_calls=call,
            )
        )
    return out


def _flat(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(str(p.get("text", "")) for p in content or [] if isinstance(p, dict))


def epoch_text(epoch: dict[str, Any]) -> str:
    """The packed bytes a client receives for one epoch."""
    parts = []
    for message in epoch.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path, help="directory of request JSON snapshots")
    parser.add_argument("--method", default="reduce-v2")
    parser.add_argument("--show", type=int, default=3, help="rejected records to print")
    parser.add_argument("--tokens", action="store_true", help="count o200k tokens per epoch")
    parser.add_argument(
        "--explain", action="store_true", help="name the record that failed, not the epoch's first"
    )
    parser.add_argument("--list", action="store_true", help="print each epoch's verdict and digest")
    args = parser.parse_args()

    # Out of tree by design (see the module docstring): the client repo does not
    # depend on the reference reducer, so pyright cannot resolve these. One line
    # each with a bare ignore: isort moves a bracketed code onto the imported
    # name, where pyright stops applying it.
    from radient_context_reduction.saver import SaverConfig, prepare  # type: ignore
    from radient_context_reduction.tokenize import token_count_text  # type: ignore

    files = sorted(args.snapshots.glob("*.json"))
    if not files:
        print(f"no snapshots under {args.snapshots}", file=sys.stderr)
        return 2
    cfg = SaverConfig(method=args.method)
    seen: set[str] = set()
    accepted = rejected = 0
    shown = 0
    span_tokens = epoch_tokens = 0
    worst_ratio = 0.0
    worst_case = ""
    for path in files:
        body = json.loads(path.read_text())
        rows = saver_projection(messages_from_blocks(body))
        encoded = [canonical_bytes(row) for row in rows]
        payload = prepare(
            {"contract": "saver-v1", "method": args.method, "messages": rows, "prior": []}, cfg
        )
        offset = 0
        for position, epoch in enumerate(payload.get("epochs") or []):
            count = int(epoch.get("source_count") or 0)
            window_rows = rows[offset : offset + count]
            window_bytes = encoded[offset : offset + count]
            offset += count
            text = epoch_text(epoch)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                validate_response(
                    {
                        "contract": "saver-v1",
                        "epochs": [epoch],
                        # One epoch at a time: `advanced` is the sum of the
                        # source counts the response claims (check 2).
                        "advanced": epoch["source_count"],
                    },
                    uploaded_rows=window_rows,
                    uploaded_bytes=window_bytes,
                    verbatim=verbatim_wire,
                    id_prefix="replay",
                    first=position == 0,
                )
            except SaverRejected as exc:
                rejected += 1
                if args.list:
                    print(f"REJECT {digest} {exc.reason}")
                if shown < args.show:
                    shown += 1
                    detail = (
                        _explain(text, window_rows, window_bytes)
                        if args.explain
                        else _first_offence(text)
                    )
                    print(f"REJECT {path.name} {exc.reason}: {detail}")
                continue
            accepted += 1
            if args.list:
                print(f"ACCEPT {digest}")
            if args.tokens:
                window_tokens = token_count_text(_all_text(window_rows))
                packed_tokens = token_count_text(text)
                span_tokens += window_tokens
                epoch_tokens += packed_tokens
                # The manager's gate is about EVERY accepted epoch, so the
                # worst per-epoch ratio is the number that matters; a mean can
                # hide one inflating epoch among many shrinking ones.
                ratio = packed_tokens / window_tokens if window_tokens else 0.0
                if ratio > worst_ratio:
                    worst_ratio, worst_case = ratio, path.name
    total = accepted + rejected
    print(f"distinct epochs {total}: accepted {accepted}, rejected {rejected}")
    if args.tokens and accepted:
        ratio = epoch_tokens / span_tokens if span_tokens else 0.0
        print(f"tokens o200k: span {span_tokens} -> packed {epoch_tokens} ({ratio:.3f}x)")
        print(f"worst accepted epoch: {worst_ratio:.3f}x ({worst_case})")
    return 0 if rejected == 0 else 1


def _explain(text: str, rows: list[dict[str, Any]], encoded: list[bytes]) -> str:
    """Bisect the epoch so a rejection names the record that actually failed."""
    from local_operator.saver.provenance import _elements, _pieces, check_records
    from local_operator.saver.validate import parse_records

    lines = text.split("\n")
    records = parse_records(text)
    starts = [
        index
        for index, line in enumerate(lines)
        if line and not line.startswith("\t") and not line.startswith("[radient:epoch")
    ]
    for position in range(len(starts)):
        cut = starts[position + 1] if position + 1 < len(starts) else len(lines)
        prefix = "\n".join(lines[:cut])
        try:
            check_records(prefix, rows)
        except SaverRejected as exc:
            label, body = records[position]
            pieces = _pieces(_elements(body))[:2]
            return f"{label} [{exc.reason}] pieces={[p[:60] for p in pieces]} body={body[:90]!r}"
    return _first_offence(text)


def _all_text(rows: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        content = row.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        for call in row.get("tool_calls") or []:
            parts.append(str((call.get("function") or {}).get("arguments", "")))
    return "\n".join(parts)


def _first_offence(text: str) -> str:
    """The first record of an epoch, so a rejection can be classified."""
    for line in text.split("\n"):
        if line and not line.startswith("\t") and not line.startswith("[radient:"):
            return line[:160]
    return text[:160]


if __name__ == "__main__":
    sys.exit(main())
