"""The PROVIDER's own floor, against the real API, through local_operator's wire client.

WHY THIS EXISTS AND WHY IT IS NOT A BENCH
=========================================
``scripts/bench_ttft.py`` measures what this repository can move, with the provider
stubbed to a known cost. This script measures the part it CANNOT move: how long the
provider itself takes to put the first token on the wire. Its numbers are the reason
``bench_ttft.py`` asserts a budget on the first event the RUNTIME emits and never on
the first provider token — see ``scripts/ttft/metrics.py``, ``BUDGET_MS``.

Measured on this machine against real deepseek, with two independent clients
(``OpenAICompatClient`` here, and a raw ``httpx`` probe): TTFB 355-514 ms on a
102-token prompt, and 1.55 s on the operator's 227k-token p50 prompt even at a
99%+ prefix-cache hit. So "first token in under 300 ms" is not a property any code
change can deliver, and a harness that asserted it would fail on every run forever.

WHAT IT REPORTS, per run, all monotonic ms from the instant the call begins:

* ``ttfb``      — response headers (the provider accepted the request)
* ``first_any`` — first decoded SSE chunk, i.e. ``StreamStartEvent``
* ``reasoning`` — first ``reasoning_content`` delta: the phase the runtime DROPS
  today, so on the current tree a real turn shows nothing for this whole interval
* ``text``      — first ``content`` delta: what the user actually sees
* ``tool``      — first ``tool_calls`` delta
* ``in_tokens`` / ``cache_read`` / ``reasoning_tokens`` — what the provider billed

HOW TO RUN IT (NOT IN CI — IT SPENDS MONEY)
===========================================
    .venv/bin/python scripts/probe_provider_ttfb.py --runs 7
    .venv/bin/python scripts/probe_provider_ttfb.py --runs 3 --input-tokens 150000
    MAXTOK=1200 SMALL_PROMPT="<a question that makes the model think>" \\
        .venv/bin/python scripts/probe_provider_ttfb.py --runs 7

A campaign sweeps the input size, because a cached prefix is NOT free: the slope of
a fully cached prefix is ~4.5 ms per 1k tokens on this machine, so the sizes matter.

The script REFUSES to run under GitHub Actions: it hits a paid provider and must
never be wired into a pipeline. It is not collected by pytest either (it is not a
test module), which is the second layer of that guard.

CREDENTIALS
===========
Resolved in-process from the operator's own encrypted store with
``AuthStore().get_api_key(<provider>, read_only=True)`` and never printed, logged,
written to a file, or passed on a command line. ``read_only=True`` matters: it makes
the resolve decide nothing about routing, so a probe cannot consume a quota
rotation or demote a credential. Nothing else in the operator's state is touched —
this script starts no session, writes no transcript, and spawns no child.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.harness.types import (  # noqa: E402
    ChatRequest,
    Message,
    ModelSpec,
    TextContent,
)
from local_operator.providers.clients import client_for_spec  # noqa: E402

#: Prose of a known shape, repeated to reach a target input size. ~4 chars/token.
FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the river runs past the "
    "willow. Local operator maintains a transcript of every turn. "
)

#: Events worth a first-of-kind mark, taken from the event's own ``type``.
MARKED_EVENTS: tuple[tuple[str, str], ...] = (
    ("reasoning_delta", "reasoning"),
    ("text_delta", "text"),
    ("tool_call_delta", "tool"),
)


async def resolve_key(provider: str) -> str:
    """The provider's API key, from the operator's store, read-only and never printed.

    ``async`` because the caller is already inside an event loop: ``asyncio.run``
    here raised ``RuntimeError: asyncio.run() cannot be called from a running event
    loop`` on the first real invocation, and a silently un-awaited coroutine would
    have been the alternative.
    """
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore()
    try:
        key = await store.get_api_key(provider, read_only=True)
    finally:
        store.close()
    if not key:
        raise SystemExit(
            f"no {provider} API key in the credential store; run `lop login {provider}` first"
        )
    return key


def big_prompt(target_tokens: int) -> str:
    """A prompt of roughly ``target_tokens`` tokens, by repeating known prose."""
    repeats = max(1, (target_tokens * 4) // len(FILLER_SENTENCE))
    return FILLER_SENTENCE * repeats


def make_request(spec: ModelSpec, prompt: str, max_tokens: int) -> ChatRequest:
    return ChatRequest(
        model=spec,
        system_blocks=["You are a terse assistant. Answer in one word."],
        messages=[Message(role="user", content=[TextContent(text=prompt)])],
        tools=[],
        max_tokens=max_tokens,
        temperature=0.0,
        purpose="ttft-probe",
    )


class _TimedCM:
    """Records the instant the response headers arrive (the provider's first byte)."""

    def __init__(self, inner: Any, marks: dict[str, float], base: float) -> None:
        self._inner = inner
        self._marks = marks
        self._base = base

    async def __aenter__(self) -> Any:
        response = await self._inner.__aenter__()
        self._marks.setdefault("ttfb", (time.monotonic() - self._base) * 1000)
        return response

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._inner.__aexit__(*exc)


class _TimedHTTP:
    """Wraps the client's httpx object so ``.stream()`` can be timed."""

    def __init__(self, inner: Any, marks: dict[str, float], base: float) -> None:
        self._inner = inner
        self._marks = marks
        self._base = base

    def stream(self, *args: Any, **kwargs: Any) -> _TimedCM:
        return _TimedCM(self._inner.stream(*args, **kwargs), self._marks, self._base)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


async def one_run(
    spec: ModelSpec, key: str, prompt: str, *, label: str, index: int, max_tokens: int
) -> dict[str, Any]:
    """One streamed call, with a first-of-kind stamp for every event that matters."""
    client = client_for_spec(spec)
    request = make_request(spec, prompt, max_tokens)
    marks: dict[str, Any] = {}
    base = time.monotonic()
    try:
        client._http = _TimedHTTP(client._http, marks, base)  # type: ignore[assignment]
        stream = client.stream(request, key)
        async for event in stream:
            now = (time.monotonic() - base) * 1000
            kind = str(getattr(event, "type", ""))
            if "first_any" not in marks:
                marks["first_any"] = now
                marks["first_event_type"] = kind
            for event_type, mark in MARKED_EVENTS:
                if kind == event_type:
                    marks.setdefault(mark, now)
            usage = getattr(event, "usage", None)
            if usage is not None:
                marks["in_tokens"] = getattr(usage, "input_tokens", 0)
                marks["cache_read"] = getattr(usage, "cache_read_tokens", 0)
                marks["reasoning_tokens"] = getattr(usage, "reasoning_tokens", 0)
                marks["out_tokens"] = getattr(usage, "output_tokens", 0)
    finally:
        # ``aclose`` is on the concrete clients, not on the ``WireClient`` protocol
        # the factory is typed as; the cast keeps the analyzer honest without
        # inventing a second close path.
        await cast(Any, client).aclose()
    marks["total"] = (time.monotonic() - base) * 1000
    marks["label"] = label
    marks["run"] = index
    return marks


def _median(rows: Iterable[dict[str, Any]], key: str) -> str:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return f"{statistics.median(values):.0f}" if values else "-"


def _summarize(rows: list[dict[str, Any]], label: str) -> None:
    """Per label, the median over every good run and over the warm ones.

    Run 1 is reported separately because it is the one that pays a COLD prefix: a
    median that folds it in understates a cached deployment and overstates an
    uncached one, and the two want different conclusions.
    """
    good = [row for row in rows if row.get("label") == label and "error" not in row]
    for condition, selection in (
        ("all", good),
        ("warm(run>=2)", [row for row in good if int(row.get("run", 0)) >= 2]),
    ):
        if not selection:
            continue
        print(
            f"{label:14s} {condition:12s} n={len(selection):2d} "
            f"ttfb={_median(selection, 'ttfb')} first_any={_median(selection, 'first_any')} "
            f"reasoning={_median(selection, 'reasoning')} text={_median(selection, 'text')} "
            f"total={_median(selection, 'total')} "
            f"in={_median(selection, 'in_tokens')} cached={_median(selection, 'cache_read')}"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=7, help="runs per input size")
    parser.add_argument("--provider", type=str, default="deepseek")
    parser.add_argument("--model", type=str, default="deepseek-flash")
    parser.add_argument(
        "--input-tokens",
        type=str,
        default="100",
        help="comma-separated approximate input sizes to sweep (one label each)",
    )
    parser.add_argument("--max-tokens", type=int, default=32, help="output cap per call")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Reply with the single word: ok",
        help="the prompt used for the smallest size; larger sizes repeat filler prose",
    )
    parser.add_argument("--json", type=str, default="", help="write every run here")
    args = parser.parse_args()

    # The paid-provider guard. `CI` alone is not usable as the marker: this
    # machine's agent shell sets it for every command to keep CLIs
    # non-interactive (AGENTS.md), so it would refuse to run anywhere at all.
    if os.environ.get("GITHUB_ACTIONS"):
        print(
            "refusing to run under GitHub Actions: this probe calls a PAID provider. "
            "See the module docstring for how it is meant to be invoked.",
            file=sys.stderr,
        )
        return 2

    spec = ModelSpec(
        provider=args.provider,
        model_id=args.model,
        context_window=1_000_000,
        max_output_tokens=393_216,
        supports_prompt_cache=True,
    )
    key = await resolve_key(args.provider)
    sizes = [int(value) for value in args.input_tokens.split(",")]
    rows: list[dict[str, Any]] = []
    print(f"provider={args.provider}/{args.model} runs={args.runs} sizes={sizes}")
    for size in sizes:
        label = args.prompt if size <= 200 else f"{size // 1000}k"
        prompt = args.prompt if size <= 200 else big_prompt(size)
        for index in range(1, args.runs + 1):
            try:
                row = await one_run(
                    spec, key, prompt, label=label, index=index, max_tokens=args.max_tokens
                )
            except Exception as exc:  # noqa: BLE001 — a provider error is a result
                row = {"label": label, "run": index, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            print(json.dumps(row), flush=True)
            await asyncio.sleep(0.3)
    print("\n=== SUMMARY (ms) ===")
    for size in sizes:
        label = args.prompt if size <= 200 else f"{size // 1000}k"
        _summarize(rows, label)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
