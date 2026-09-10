#!/usr/bin/env python3
"""Measure xAI prompt-cache hit rate with and without ``x-grok-conv-id``.

xAI stores cache entries per server; without a routing header a conversation's
turns spread across servers and roughly every other request lands cold
(measured 48.5% cached-input share over 14 days of local analytics, vs 97.0%
on the affinity-routed Anthropic path). This benchmark runs the SAME adapter
twice over a small multi-turn loop -- once with the affinity header stripped
before the wire (baseline), once with it intact (affinity) -- so the only
difference between arms is the routing key. See docs/XAI_CACHING.md.

The strip happens in a transport wrapper, not a flag in the product: the
baseline arm measures today's shipped request path byte-for-byte minus one
header. Each arm gets its own synthetic prefix namespace AND cache lineage, so
arms can neither share nor poison each other's cache groups.

xAI cached tokens are a subset of ``prompt_tokens``: rate = cached / prompt.
Provider usage is authoritative. A zero cache result is evidence, not a reason
to discard a run. Credentials are read from the harness auth store read-only
(the xAI OAuth row's ``access``, used as the wire bearer exactly as
``xai-oauth`` does at runtime) or ``XAI_API_KEY``; the key is never printed.

Budget: 2 arms x --turns calls (default 7 + 7 = 14 small calls).

Examples:
    .venv/bin/python scripts/bench_xai_cache_rate.py --dry-run
    .venv/bin/python scripts/bench_xai_cache_rate.py --live
    .venv/bin/python scripts/bench_xai_cache_rate.py --live --arm baseline
    LOP_XAI_BENCH_STRIP_HEADER=1 .venv/bin/python scripts/bench_xai_cache_rate.py \
        --live --arm affinity   # env forces the baseline arm (A/B toggle)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.harness.types import (  # noqa: E402
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamUsageEvent,
    TextContent,
)
from local_operator.model.configure import build_model_spec  # noqa: E402
from local_operator.providers.clients import OpenAICompatClient  # noqa: E402

XAI_BASE = "https://api.x.ai/v1"
MODEL_ID = "grok-4.6"
TERSE = " Answer in 10 words or fewer, no code."
TURN_PROMPTS = [
    "Name a primary color.",
    "Name a secondary color.",
    "Name a warm color.",
    "Name a cool color.",
    "Name a color of the sky.",
    "Name a color of grass.",
    "Name a color of night.",
    "Name any color at all.",
]


@dataclass
class TurnUsage:
    """Provider counters; cached is a subset of input, not an extra bucket."""

    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    native_replay_items: int = 0
    conv_id_on_wire: str | None = None
    body_chars: int = 0
    message_count: int = 0

    @property
    def cache_share(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class ArmResult:
    name: str
    turns: list[TurnUsage] = field(default_factory=list)
    error: str | None = None

    @property
    def total_prompt(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def total_cached(self) -> int:
        return sum(t.cached_tokens for t in self.turns)

    @property
    def cache_rate(self) -> float:
        return self.total_cached / self.total_prompt if self.total_prompt else 0.0


def _xai_api_key(path: Path) -> str:
    """Read the xAI bearer without store migration, refresh, or printing it.

    The OAuth row's ``access`` token is what ``xai-oauth`` presents on the
    wire at runtime (``_oauth_api_key``), so using it here exercises the real
    credential path; an unexpired row is required, never refreshed.
    """
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT id, data FROM auth_credentials WHERE provider='xai' "
            "AND disabled_cause IS NULL ORDER BY id"
        )
        for _row_id, data in rows:
            credential = json.loads(data)
            if (
                credential.get("access")
                and credential.get("type") == "oauth"
                and credential.get("expires", 0) > time.time() * 1000 + 120_000
            ):
                return str(credential["access"])
    raise ValueError(
        "No unexpired xAI credential in the auth store; log in outside this benchmark."
    )


class _ArmTransport(httpx.AsyncBaseTransport):
    """Record the routing header; optionally strip it BEFORE the wire.

    The affinity arm is a passive observer. The baseline arm proves the client
    added the header (it must have been present to be stripped) and then
    removes it, so the A/B pair differs by exactly one header.
    """

    def __init__(self, strip: bool) -> None:
        self.inner = httpx.AsyncHTTPTransport(retries=0)
        self.strip = strip
        self.saw_conv_id: str | None = None
        self.last_body_chars = 0
        self.last_message_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if "x-grok-conv-id" in request.headers:
            self.saw_conv_id = request.headers["x-grok-conv-id"]
        if self.strip:
            request.headers.pop("x-grok-conv-id", None)
        try:
            body = json.loads(request.content)
            self.last_body_chars = len(request.content)
            self.last_message_count = len(body.get("messages") or [])
        except (ValueError, UnicodeDecodeError):
            pass
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


def _system_blocks(namespace: str, rows: int) -> list[str]:
    """A stable synthetic prefix per arm; never private context.

    The unique namespace must FILL THE FIRST ~512-TOKEN BLOCK, not just name
    itself: xAI's cache matches content prefixes at block granularity, so a
    short namespace line followed by shared synthetic records lets unrelated
    invocations hit each other's cache (measured: a fresh conversation's
    first turn cached 1152/1235 tokens it could only have inherited from an
    earlier invocation's identical records). Padding the first block with
    seed-unique text keeps arms and runs genuinely cold-start isolated.
    """
    padding = " ".join(f"namespace-{namespace}-{i}" for i in range(rows * 4))
    return [
        f"Synthetic xAI cache benchmark, namespace {namespace}. {padding}",
        " ".join(
            f"Synthetic record {i}: approved color is blue, revision {i % 7}." for i in range(rows)
        ),
    ]


def _source_identity() -> dict[str, Any]:
    from local_operator.providers import clients

    try:
        revision = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "source_revision": revision,
        "adapter_sha256": hashlib.sha256(Path(clients.__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


async def _run_arm(args: argparse.Namespace, name: str, api_key: str, output: Any) -> ArmResult:
    result = ArmResult(name)
    namespace = f"{args.seed}:{name}"
    spec: ModelSpec = build_model_spec("xai", args.model)
    lineage = f"bench-xai:{args.seed}:{name}"  # stable per arm, distinct across arms
    blocks = _system_blocks(namespace, args.prefix_rows)
    history: list[Message] = []
    transport = _ArmTransport(strip=name == "baseline")
    async with httpx.AsyncClient(transport=transport, timeout=90) as http:
        client = OpenAICompatClient(XAI_BASE, http_client=http)
        for index in range(args.turns):
            prompt = TURN_PROMPTS[index % len(TURN_PROMPTS)] + TERSE
            request = ChatRequest(
                model=spec,
                system_blocks=blocks,
                messages=[*history, Message.user(prompt)],
                prompt_cache_key=lineage,
            )
            turn = TurnUsage(conv_id_on_wire=transport.saw_conv_id)
            text = ""
            usage = None
            terminal: StreamEndEvent | None = None
            try:
                async with asyncio.timeout(90):
                    async for event in client.stream(request, api_key):
                        if isinstance(event, StreamTextDelta):
                            text += event.delta
                        elif isinstance(event, StreamUsageEvent):
                            usage = event.usage
                        elif isinstance(event, StreamEndEvent):
                            terminal = event
                            usage = event.usage or usage
                if terminal is None or terminal.stop_reason not in ("stop", "toolUse"):
                    raise ValueError(f"Incomplete response: {terminal and terminal.stop_reason}")
                if usage is None:
                    raise ValueError("Provider completed without usage; cache rate is unknown.")
                turn.prompt_tokens = usage.input_tokens
                turn.cached_tokens = usage.cache_read_tokens
                turn.output_tokens = usage.output_tokens
                turn.reasoning_tokens = usage.reasoning_tokens or 0
                replay = (terminal.provider_payload or {}).get("native_replay") or {}
                turn.native_replay_items = len(replay.get("items") or [])
                turn.body_chars = transport.last_body_chars
                turn.message_count = transport.last_message_count
                history = [
                    *history,
                    Message.user(prompt),
                    Message(
                        role="assistant",
                        content=[TextContent(text=text or "(empty)")],
                        provider_payload=terminal.provider_payload,
                    ),
                ]
                result.turns.append(turn)
                print(
                    json.dumps(
                        {
                            "arm": name,
                            "turn": index,
                            "prompt_tokens": turn.prompt_tokens,
                            "cached_tokens": turn.cached_tokens,
                            "cache_share": round(turn.cache_share, 4),
                            "output_tokens": turn.output_tokens,
                            "reasoning_tokens": turn.reasoning_tokens,
                            "native_replay_items": turn.native_replay_items,
                            "body_chars": turn.body_chars,
                            "message_count": turn.message_count,
                            "conv_id_sent": (
                                (transport.saw_conv_id or None) if name == "affinity" else None
                            ),
                        }
                    ),
                    file=output,
                )
            except Exception as exc:  # noqa: BLE001 - one failed turn fails the arm, loudly
                result.error = f"turn {index}: {exc}"
                break
            if index + 1 < args.turns:
                await asyncio.sleep(args.gap)
    if name == "affinity" and transport.saw_conv_id is None:
        result.error = result.error or "affinity arm never saw x-grok-conv-id on the wire"
    if name == "baseline" and result.turns and transport.saw_conv_id is None:
        # The strip is only meaningful if the client actually added the header.
        result.error = result.error or "baseline arm stripped nothing: header was never added"
    return result


def _print_table(results: list[ArmResult]) -> None:
    print("arm          turns    prompt    cached   hit rate", file=sys.stderr)
    for result in results:
        rate = f"{result.cache_rate:.1%}" if result.total_prompt else "unknown"
        print(
            f"{result.name:<12} {len(result.turns):>5} {result.total_prompt:>9} "
            f"{result.total_cached:>9}   {rate:>7}",
            file=sys.stderr,
        )
        if result.error:
            print(f"ERROR: {result.error}", file=sys.stderr)
    print(
        (
            "cached/prompt per turn: "
            + ", ".join(f"{t.cached_tokens}/{t.prompt_tokens}" for t in results[-1].turns)
            if results
            else ""
        ),
        file=sys.stderr,
    )


async def _run(args: argparse.Namespace, api_key: str, output: Any) -> int:
    arms = [args.arm] if args.arm != "both" else ["baseline", "affinity"]
    results = []
    for name in arms:
        result = await _run_arm(args, name, api_key, output)
        results.append(result)
        if result.error:
            break
    _print_table(results)
    return int(any(r.error for r in results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print plan; no requests, no key.")
    mode.add_argument("--live", action="store_true", help="Spend the bounded POST budget.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--arm", choices=("baseline", "affinity", "both"), default="both")
    parser.add_argument("--turns", type=int, choices=range(2, 9), default=7)
    parser.add_argument("--prefix-rows", type=int, default=40)
    parser.add_argument("--gap", type=float, default=2.0, help="Seconds between turns.")
    parser.add_argument("--seed", default=uuid.uuid4().hex)
    parser.add_argument(
        "--auth-db",
        type=Path,
        default=Path(
            os.environ.get("LOCAL_OPERATOR_CONFIG_DIR", str(Path.home() / ".local-operator"))
        )
        / "auth.db",
    )
    parser.add_argument(
        "--output", type=Path, help="New JSONL file; existing evidence is never overwritten."
    )
    args = parser.parse_args(argv)
    # Env A/B toggle: forces the baseline arm even when --arm affinity is asked
    # for, so a shell can flip one variable between two invocations.
    if os.environ.get("LOP_XAI_BENCH_STRIP_HEADER") and args.arm == "affinity":
        args.arm = "baseline"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "plan": [f"{args.arm}:{args.turns} turns"],
                    "model": args.model,
                    "seed": args.seed,
                    **_source_identity(),
                },
                indent=2,
            )
        )
        return 0
    api_key = os.environ.get("XAI_API_KEY") or _xai_api_key(args.auth_db)
    output = args.output.open("x") if args.output else sys.stdout
    saved = {key: os.environ.get(key) for key in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR")}
    try:
        # Redirect config the same way the oauth cache bench does: the client
        # import path must not touch the operator's live sessions or caches.
        with tempfile.TemporaryDirectory(prefix="lop-xai-cache-") as home:
            os.environ["HOME"] = home
            os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(Path(home) / ".local-operator")
            print(json.dumps(_source_identity()), file=output)
            return asyncio.run(_run(args, api_key, output))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        output.flush()
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
