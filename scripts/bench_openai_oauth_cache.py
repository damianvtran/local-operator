#!/usr/bin/env python3
"""Measure the current OpenAI OAuth wire path, not a promised cache improvement.

Run the SAME script on the baseline and candidate checkout. The adapter owns
all request shaping; this benchmark does not enable experimental API fields.
Every live invocation uses a fresh synthetic prefix namespace, redirects HOME
and config, and reads one existing OAuth access token without refreshing it.
No real user conversations or external tool execution enter the experiment.

OpenAI cached/write tokens are subsets of input: rate = sum(cached)/sum(input).
Provider usage is authoritative; list-price-equivalent input units are only an
estimate, NOT an OAuth subscription charge. A zero cache result is evidence,
not a reason to discard a run or claim a speedup. See docs/OPENAI_CACHING.md.

Examples (live calls are opt-in; --turns is the POST budget per scenario):
    .venv/bin/python scripts/bench_openai_oauth_cache.py --dry-run
    .venv/bin/python scripts/bench_openai_oauth_cache.py --live --model gpt-6-astra \
        --scenario long_session --turns 3 --output /tmp/cache-run.jsonl

The Codex backend removes max_output_tokens. Terse synthetic prompts and a
request timeout bound exposure, but are NOT a provider-enforced output cap.
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
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    ToolContext,
    ToolResult,
)
from local_operator.model.configure import build_model_spec  # noqa: E402
from local_operator.prompts_api import build_system_blocks  # noqa: E402
from local_operator.providers.auth_store import OAuthAccess  # noqa: E402
from local_operator.providers.clients import (  # noqa: E402
    CODEX_RESPONSES_URL,
    OpenAICompatClient,
)
from local_operator.tools.registry import create_tools  # noqa: E402

MODEL_ID = "gpt-5.6-sol"
SCENARIOS = ("long_session", "stable_prefix", "tool_heavy", "reasoning_on", "repeat_then_append")
TERSE = " Answer in 15 words or fewer, no code or tool calls."


@dataclass
class TurnUsage:
    """Provider counters: cached and written are subsets of input, not extras."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    @property
    def input_price_equivalent_tokens(self) -> float:
        # Public GPT-5.6+ list-rate multipliers, not subscription billing. Never
        # infer cache writes from misses: these are the reported buckets only.
        plain = self.input_tokens - self.cache_read_tokens - self.cache_write_tokens
        return plain + 0.1 * self.cache_read_tokens + 1.25 * self.cache_write_tokens


@dataclass
class ScenarioResult:
    name: str
    turns: list[TurnUsage] = field(default_factory=list)
    error: str | None = None

    @property
    def total_input(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_cache_read(self) -> int:
        return sum(t.cache_read_tokens for t in self.turns)

    @property
    def total_cache_write(self) -> int:
        return sum(t.cache_write_tokens for t in self.turns)

    @property
    def denom(self) -> int:
        return self.total_input

    @property
    def cache_rate(self) -> float:
        return self.total_cache_read / self.denom if self.denom else 0.0


def _build_prefix(namespace: str, rows: int) -> tuple[list[AgentTool], list[str]]:
    """Real tool schemas plus synthetic reference text, never private context."""
    tools = create_tools(ToolContext(cwd=str(Path.home()), session_id="bench-oauth-cache"))
    if tools:
        # Tools precede developer instructions on the wire. Isolate that earliest
        # prefix too, otherwise independent arms can share cached schema tokens.
        tools[0] = tools[0].model_copy(
            update={"description": f"Synthetic namespace {namespace}. " + tools[0].description}
        )
    tools.append(
        AgentTool(
            name="synthetic_lookup",
            description="Return the benchmark's synthetic color without reading files.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            execute=_lookup,
        )
    )
    blocks = build_system_blocks(tools, "", "Synthetic isolated benchmark workspace.", "2026-09-04")
    # Keys only influence routing. Independent leading CONTENT, not keys alone,
    # keeps an earlier arm from warming the reusable developer prefix of this arm.
    blocks[0] = f"Synthetic cache namespace: {namespace}\n" + blocks[0]
    blocks.append(
        " ".join(
            f"Synthetic record {i}: approved color is blue, revision {i % 7}." for i in range(rows)
        )
        + "\nFor a lookup, call ONLY synthetic_lookup once, then acknowledge its result with OK."
    )
    return tools, blocks


async def _lookup(call_id: str, *_args: Any, **_kwargs: Any) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id, tool_name="synthetic_lookup", content=[TextContent(text="blue")]
    )


def _resolve_spec(model: str, reasoning_effort: str) -> ModelSpec:
    spec = build_model_spec("openai", model)
    if reasoning_effort not in spec.reasoning_efforts:
        raise ValueError(f"Unsupported benchmark reasoning effort: {reasoning_effort}")
    return spec.model_copy(update={"reasoning_effort": reasoning_effort})


def _make_request(
    spec: ModelSpec,
    system_blocks: list[str],
    tools: list[AgentTool],
    messages: list[Message],
    session_id: str,
) -> ChatRequest:
    return ChatRequest(
        model=spec,
        system_blocks=system_blocks,
        messages=list(messages),
        tools=tools,
        prompt_cache_key=session_id,
    )


def _source_identity() -> dict[str, Any]:
    """A dirty worktree needs a source hash as well as its last committed SHA."""
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


def _oauth_access(path: Path) -> OAuthAccess:
    """Read an unexpired access token without store migration or token rotation."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT id, data FROM auth_credentials WHERE provider='openai' "
            "AND credential_type='oauth' AND disabled_cause IS NULL ORDER BY id"
        )
        for row_id, data in rows:
            credential = json.loads(data)
            if (
                credential.get("access")
                and credential.get("org_id")
                and credential.get("expires", 0) > time.time() * 1000 + 120_000
            ):
                return OAuthAccess(
                    access_token=credential["access"],
                    credential_id=row_id,
                    org_id=credential["org_id"],
                )
    raise ValueError("No unexpired OpenAI OAuth access token; log in outside this benchmark.")


class _CaptureStream(httpx.AsyncByteStream):
    """Observe usage only; forward original bytes without buffering the response."""

    def __init__(self, stream: httpx.AsyncByteStream, record: dict[str, Any]) -> None:
        self.stream, self.record = stream, record

    async def __aiter__(self):
        pending = b""
        async for data in self.stream:
            pending += data
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if not line.startswith(b"data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except (ValueError, UnicodeDecodeError):
                    continue
                if event.get("type") in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    response = event.get("response") or {}
                    self.record["raw_usage"] = response.get("usage")
                    self.record["returned_model"] = response.get("model")
                    self.record["terminal"] = event["type"]
            yield data

    async def aclose(self) -> None:
        await self.stream.aclose()


class _CaptureTransport(httpx.AsyncBaseTransport):
    """Observe the real adapter's body; never rewrite it into a candidate API."""

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self.inner = inner or httpx.AsyncHTTPTransport(retries=0)
        self.record: dict[str, Any] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.record.update(
            endpoint=str(request.url),
            requested_model=body["model"],
            body_sha256=hashlib.sha256(request.content).hexdigest(),
            body_shape=_redact_body(body),
            affinity_headers={
                name: request.headers[name]
                for name in ("session-id", "thread-id")
                if name in request.headers
            },
        )
        response = await self.inner.handle_async_request(request)
        self.record["http_status"] = response.status_code
        assert isinstance(response.stream, httpx.AsyncByteStream)
        response.stream = _CaptureStream(response.stream, self.record)
        return response

    async def aclose(self) -> None:
        await self.inner.aclose()


def _redact_body(body: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value for key, value in body.items() if key not in ("instructions", "input", "tools")
    }
    out["instructions_chars"] = len(body.get("instructions", ""))
    out["input_types"] = [item.get("type") or item.get("role") for item in body.get("input", [])]
    out["tool_names"] = [tool.get("name") for tool in body.get("tools", [])]
    return out


async def _stream_turn(
    client: OpenAICompatClient, request: ChatRequest, access: OAuthAccess, record: dict[str, Any]
) -> tuple[TurnUsage, Message]:
    start = time.monotonic()
    text = ""
    calls: dict[int, dict[str, str]] = {}
    terminal: StreamEndEvent | None = None
    usage = None
    try:
        # A read timeout alone never stops an active reasoning stream.
        async with asyncio.timeout(90):
            async for event in client.stream(request, None, oauth_access=access):
                if isinstance(event, StreamTextDelta):
                    record.setdefault("ttft_s", time.monotonic() - start)
                    text += event.delta
                elif isinstance(event, StreamToolCallDelta):
                    call = calls.setdefault(event.index, {"id": "", "name": "", "arguments": ""})
                    if event.id:
                        call["id"] = event.id
                    if event.name:
                        call["name"] = event.name
                    call["arguments"] += event.argument_delta
                elif isinstance(event, StreamUsageEvent):
                    usage = event.usage
                elif isinstance(event, StreamEndEvent):
                    terminal = event
                    usage = event.usage or usage
        record["actual_response_text"] = text
        if terminal is None or terminal.stop_reason not in ("stop", "toolUse"):
            stop_reason = terminal.stop_reason if terminal else "no terminal"
            raise ValueError(f"Incomplete benchmark response: {stop_reason}")
        record["stop_reason"] = terminal.stop_reason
        if usage is None:
            raise ValueError("Provider completed without usage; cache rate is unknown.")
        record["normalized_usage"] = usage.model_dump()
        raw = record.get("raw_usage") or {}
        if not raw or "input_tokens" not in raw:
            raise ValueError("Raw provider usage is missing; cache rate is unknown.")
        if record.get("returned_model") != request.model.model_id:
            raise ValueError("Provider returned a different or unidentified model.")
        details = raw.get("input_tokens_details") or {}
        counters = [
            raw.get("input_tokens", 0),
            raw.get("output_tokens", 0),
            details.get("cached_tokens", 0),
            details.get("cache_write_tokens", 0),
        ]
        if any(not isinstance(value, int) or value < 0 for value in counters):
            raise ValueError("Provider reported invalid raw token counters.")
        if (
            min(
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.cache_write_tokens,
            )
            < 0
            or usage.cache_read_tokens + usage.cache_write_tokens > usage.input_tokens
        ):
            raise ValueError("Provider cache buckets do not fit inside input tokens.")
        turn = TurnUsage(
            usage.input_tokens,
            usage.cache_read_tokens,
            usage.cache_write_tokens,
            usage.output_tokens,
        )
        tool_calls = [
            ToolCall(id=c["id"], name=c["name"], arguments=json.loads(c["arguments"] or "{}"))
            for c in calls.values()
        ]
        record["tool_calls"] = [call.model_dump() for call in tool_calls]
        record["public_list_input_equivalent_tokens_estimate"] = turn.input_price_equivalent_tokens
        return turn, Message.assistant(
            text, tool_calls=tool_calls, usage=usage, provider_payload=terminal.provider_payload
        )
    finally:
        record["actual_response_text"] = text
        record["total_s"] = time.monotonic() - start


async def _run_scenario(
    args: argparse.Namespace, name: str, access: OAuthAccess | None, output: Any
) -> ScenarioResult:
    result = ScenarioResult(name)
    provenance = _source_identity()
    namespace = f"{args.seed}:{args.model}:{name}"
    tools, blocks = _build_prefix(namespace, args.prefix_rows)
    spec = _resolve_spec(args.model, "medium" if name == "reasoning_on" else "low")
    history: list[Message] = []
    transport = _CaptureTransport()
    async with httpx.AsyncClient(transport=transport, timeout=90) as http:
        client = OpenAICompatClient("https://api.openai.com/v1", http_client=http)
        for index in range(args.turns):
            if name == "stable_prefix":
                history = [
                    Message.user(f"Synthetic question {index}: name a primary color." + TERSE)
                ]
            elif name == "repeat_then_append" and index < 3:
                history = [Message.user("Name a primary color." + TERSE)]
            elif not history or history[-1].role != "tool":
                history.append(
                    Message.user(
                        "Perform the synthetic_lookup now."
                        if name == "tool_heavy" and index % 2 == 0
                        else f"Synthetic question {index}: acknowledge with OK." + TERSE
                    )
                )
            request = _make_request(spec, blocks, tools, history, namespace)
            if name == "tool_heavy" and index % 2 == 0:
                request = request.model_copy(update={"tool_choice": "required"})
            record: dict[str, Any] = dict(
                scenario=name,
                turn=index,
                seed=args.seed,
                auth_mode="chatgpt_oauth",
                retries=0,
                source_root=str(REPO),
                prefix_rows=args.prefix_rows,
                public_list_estimate_not_subscription_charge=True,
            )
            record.update(provenance)
            record["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            transport.record = record
            try:
                if args.dry_run:
                    body = client._build_codex_responses_body(request)
                    record.update(
                        endpoint=CODEX_RESPONSES_URL,
                        body_shape=_redact_body(body),
                        requested_model=args.model,
                    )
                else:
                    assert access is not None
                    turn, assistant = await _stream_turn(client, request, access, record)
                    result.turns.append(turn)
                    history.append(assistant)
                    for call in assistant.tool_calls:
                        # The real inventory is present only to measure its schema
                        # footprint. Never execute a model-selected filesystem/shell tool.
                        if call.name != "synthetic_lookup" or call.arguments:
                            raise ValueError(
                                "Unexpected tool call; benchmark executes only its "
                                "empty-argument synthetic lookup."
                            )
                        reply = await _lookup(call.id)
                        history.append(
                            Message(
                                role="tool",
                                tool_call_id=call.id,
                                tool_name=call.name,
                                content=reply.content,
                            )
                        )
            except Exception as error:
                message = str(error) or type(error).__name__
                if access:
                    for secret in (access.access_token, access.org_id):
                        if secret:
                            message = message.replace(secret, "<redacted>")
                result.error = message
                record["error"] = message
            output.write(json.dumps(record) + "\n")
            output.flush()
            if result.error or args.dry_run:
                break
            if index + 1 < args.turns:
                await asyncio.sleep(args.gap)
    return result


def _print_scenario_table(results: list[ScenarioResult]) -> None:
    print("scenario         calls      input     cached     writes   hit rate", file=sys.stderr)
    for result in results:
        print(
            f"{result.name:<16} {len(result.turns):>5} {result.total_input:>10} "
            f"{result.total_cache_read:>10} {result.total_cache_write:>10} "
            f"{result.cache_rate:>9.1%}",
            file=sys.stderr,
        )
        if result.error:
            print(f"ERROR: {result.error}", file=sys.stderr)
    denominator = sum(result.denom for result in results)
    rate = sum(result.total_cache_read for result in results) / denominator if denominator else 0
    warm = [turn for result in results for turn in result.turns[1:]]
    warm_input = sum(turn.input_tokens for turn in warm)
    warm_rate = sum(turn.cache_read_tokens for turn in warm) / warm_input if warm_input else None
    cold_label = f"{rate:.1%}" if denominator else "unknown"
    warm_label = f"{warm_rate:.1%}" if warm_rate is not None else "unknown"
    print(
        f"Reported-call weighted hit rate: {cold_label}; warm-only: {warm_label} "
        "(excluding each scenario's first call). Errors remain failures.",
        file=sys.stderr,
    )


async def _run(args: argparse.Namespace, access: OAuthAccess | None, output: Any) -> int:
    results = []
    for name in [args.scenario] if args.scenario else SCENARIOS:
        result = await _run_scenario(args, name, access, output)
        results.append(result)
        if result.error:
            break
    if not args.dry_run:
        _print_scenario_table(results)
    return int(any(result.error for result in results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Show adapter bodies without credentials or requests.",
    )
    mode.add_argument("--live", action="store_true", help="Spend the explicit bounded POST budget.")
    parser.add_argument("--model", choices=(MODEL_ID, "gpt-6-astra"), default=MODEL_ID)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--turns", type=int, choices=range(1, 9), default=3)
    parser.add_argument(
        "--prefix-rows", type=int, choices=range(0, 1801), default=0, metavar="0..1800"
    )
    parser.add_argument("--gap", type=float, default=5, help="Seconds between requests, minimum 4.")
    parser.add_argument(
        "--seed",
        default=uuid.uuid4().hex,
        help="Fresh by default; reuse deliberately for warm-cache controls.",
    )
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
    if not 4 <= args.gap <= 60:
        parser.error("--gap must be between 4 and 60 seconds")
    if not args.seed or len(args.seed) > 80:
        parser.error("--seed must contain 1..80 characters")
    try:
        access = None if args.dry_run else _oauth_access(args.auth_db)
        output = args.output.open("x") if args.output else sys.stdout
        saved = {key: os.environ.get(key) for key in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR")}
        try:
            with tempfile.TemporaryDirectory(prefix="lop-openai-cache-") as home:
                os.environ["HOME"] = home
                os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(Path(home) / ".local-operator")
                return asyncio.run(_run(args, access, output))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if output is not sys.stdout:
                output.close()
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
