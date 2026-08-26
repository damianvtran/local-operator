#!/usr/bin/env python3
"""Live prompt-cache benchmark for the OpenAI OAuth (ChatGPT / Codex) path.

This drives REAL multi-turn conversations through the harness' actual wire
client (``OpenAICompatClient``) against the operator's ChatGPT-subscription
OAuth credential for ``gpt-5.6-sol``, and reports the server-side prompt-cache
hit rate the Codex Responses backend actually delivers. Its purpose is a
before/after measurement: run the SAME script on ``origin/main`` and on a fix
branch and compare the numbers. It deliberately hardcodes nothing about any
fix -- it exercises whatever request the current code builds, so the delta is
attributable to the code under test rather than to the harness.

Why this exists
---------------
The Codex Responses path (``_build_codex_responses_body``) currently strips
``prompt_cache_key`` from the body (it is popped as a "public-API-only" key).
OpenAI's Responses cache keys off request-prefix identity, and a stable
``prompt_cache_key`` is what pins diverging sessions that share a big system +
tools prefix onto the same cache bucket (the failure mode OpenAI Codex issue
#35300 targets). Without it, a large stable prefix reused across sessions that
diverge at the first user turn caches far worse than it should. We need a live
BEFORE number to prove the gap and an AFTER number to prove a fix.

The cache-read rate and its subset semantics
--------------------------------------------
On OpenAI, ``usage.input_tokens_details.cached_tokens`` (mapped onto
``Usage.cache_read_tokens``) is a SUBSET of ``usage.input_tokens`` -- the cached
tokens are counted INSIDE the input total, not in addition to it. The Codex
backend does not report a separate cache-WRITE count, so ``cache_write_tokens``
is 0 on this path. To keep the rate comparable to the other providers' rate in
``bench_cache_rate.py`` (where cached is reported OUTSIDE input), and to keep
the denominator equal to the true prompt-side token volume, we compute:

    cache_read_rate = sum(cache_read) / sum(input + cache_read + cache_write)

Because ``cached ⊆ input`` here, ``input + cache_read`` double-counts the cached
slice on purpose: the denominator is then "prompt tokens billed at full price
(input - cached) + cached tokens counted once as input + cached counted again",
which for OpenAI reduces to ``input + cached`` and makes the ratio directly the
fraction of the prompt that was served from cache relative to a
cached-outside-input convention. In practice ``cache_write`` is 0 here, so the
denominator is ``input + cached`` and the rate is ``cached / (input + cached)``.
The per-scenario table prints the raw sums so the semantics are auditable.

What each scenario probes
-------------------------
1. ``long_session`` -- one session, 6-8 append-only turns that build on each
   other. This is the operator's dominant usage; the whole prefix (system +
   tools + growing history) is stable turn to turn, so it should show the
   HIGHEST baseline cache rate even without the fix, because a single session
   naturally reuses its own prefix.
2. ``stable_prefix`` -- the same large system + tools prefix, but 5 SEPARATE
   short sessions that diverge at the very first user turn. This is the issue
   #35300 failure mode: the big shared prefix should hit cache across sessions,
   and the ``prompt_cache_key`` is what makes that happen. Expect the biggest
   before/after gap here.
3. ``tool_heavy`` -- one session where each turn issues a tool call and feeds
   the ``function_call`` / ``function_call_output`` back, 5-6 turns. Probes how
   the cache behaves when the tail of the prefix is tool traffic rather than
   plain user/assistant text (breakpoint-after-tool-result behavior).
4. ``reasoning_on`` -- like scenario 1 but with reasoning effort set to
   ``medium`` so the encrypted-reasoning-content path is exercised. Reasoning
   items ride the Responses prefix; this checks caching still lands with them.

Cost discipline
---------------
Model OUTPUT is the expensive part and it is capped hard: every prompt asks for
a terse (<=15 word) answer, and turn counts are small and flag-tunable. The
PREFIX (system + tools) is deliberately realistic (~10-15k tokens) so the cached
prefix clears OpenAI's 1024-token cache floor -- a toy prefix would never cache
and the measurement would be meaningless. Net: a full run spends a few thousand
input tokens per turn (cached after the first) and a few hundred output tokens.

Usage
-----
    .venv/bin/python scripts/bench_openai_oauth_cache.py --dry-run
    .venv/bin/python scripts/bench_openai_oauth_cache.py --scenario long_session --turns 6
    .venv/bin/python scripts/bench_openai_oauth_cache.py            # all scenarios, live

``--dry-run`` prints the redacted request-body shape WITHOUT any network call so
the codex routing (posts to the Codex URL, and -- on origin/main -- carries NO
``prompt_cache_key``) can be inspected. Exit code is ALWAYS 0: this is a
measurement tool, not a gate. A missing OAuth credential prints a skip line and
exits 0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.harness.types import (  # noqa: E402
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamUsageEvent,
    ToolCall,
    ToolContext,
    Usage,
)
from local_operator.model.configure import build_model_spec  # noqa: E402
from local_operator.prompts_api import build_system_blocks  # noqa: E402
from local_operator.providers.auth_store import AuthStore, OAuthAccess  # noqa: E402
from local_operator.providers.clients import (  # noqa: E402
    CODEX_RESPONSES_URL,
    OpenAICompatClient,
    client_for_spec,
)
from local_operator.tools.registry import create_tools  # noqa: E402

# The model under test. Resolved through the real registry so
# supports_prompt_cache / supports_responses_api / context_window are the exact
# values the app runs with, not guesses.
MODEL_ID = "gpt-5.6-sol"

# A terse-answer instruction appended to every user prompt. Output tokens are
# the costly ones; the whole benchmark is about INPUT-side caching, so we cap
# generation to a handful of tokens per turn without changing the prefix.
TERSE = " Answer in 15 words or fewer, no code."

# A small, realistic skills block so the frozen skills tail matches what a live
# session carries. Kept byte-stable for a session exactly as the harness freezes
# its selected skills block for the conversation.
SKILLS_BLOCK = "<skills>\nminerva-observability: Datadog incident playbooks\n</skills>"

ENV_DETAILS = "Platform: Darwin (arm64). Shell: /bin/zsh. Working on a Python project."
DATE_STR = "2026-08-26"


@dataclass
class TurnUsage:
    """One turn's prompt-side token accounting from the provider's usage event."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0


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
        # See the module docstring: cached ⊆ input on OpenAI, so the denominator
        # counts input + cache_read (+ cache_write, 0 here) to express the rate
        # on the same "cached outside input" convention the other bench uses.
        return self.total_input + self.total_cache_read + self.total_cache_write

    @property
    def cache_rate(self) -> float:
        return self.total_cache_read / self.denom if self.denom else 0.0


# ---------------------------------------------------------------------------
# Prefix construction (identical big prefix for every scenario)
# ---------------------------------------------------------------------------


def _build_prefix() -> tuple[list[AgentTool], list[str]]:
    """The realistic (tools, system_blocks) prefix shared by every scenario.

    The tool inventory and system blocks are what dominate the cached prefix
    (~10-15k tokens together), so they are built once and reused: two scenarios
    that share this prefix byte-for-byte are exactly what a server-side prefix
    cache is supposed to reward.
    """
    tools = create_tools(ToolContext(cwd=str(REPO), session_id="bench-oauth-cache"))
    system_blocks = build_system_blocks(tools, SKILLS_BLOCK, ENV_DETAILS, DATE_STR)
    return tools, system_blocks


def _resolve_spec(reasoning_effort: str | None) -> ModelSpec:
    """The real ModelSpec for the model, optionally pinned to a reasoning level.

    Resolved via ``build_model_spec`` (the same path the session uses) so
    ``supports_prompt_cache`` and ``supports_responses_api`` are authoritative.
    A reasoning level is only applied when the model's own ladder accepts it, so
    the request never carries a level the provider would 400 on.
    """
    spec = build_model_spec("openai", MODEL_ID)
    if reasoning_effort and reasoning_effort in spec.reasoning_efforts:
        spec = spec.model_copy(update={"reasoning_effort": reasoning_effort})
    return spec


def _make_request(
    spec: ModelSpec,
    system_blocks: list[str],
    tools: list[AgentTool],
    messages: list[Message],
    session_id: str,
) -> ChatRequest:
    """Build a ChatRequest exactly as the session host does.

    Critically, ``prompt_cache_key`` is set to the session id, mirroring
    ``SessionStreamFn.stream`` in ``model/configure.py`` (which copies the
    session id onto the request when the field is unset). This is the realistic
    behavior; whether the Codex body then FORWARDS it is the code under test.
    """
    return ChatRequest(
        model=spec,
        system_blocks=system_blocks,
        messages=messages,
        tools=tools,
        prompt_cache_key=session_id,
    )


# ---------------------------------------------------------------------------
# Scenario message builders
# ---------------------------------------------------------------------------

# Scenario 1 & 4: an append-only coding conversation. Each prompt builds on the
# last so the whole history is a stable growing prefix.
_LONG_PROMPTS = [
    "I'm building a CLI todo app in Python. What single module layout do you suggest?",
    "Name the dataclass fields for a Todo item.",
    "What method signatures should the TodoList class expose?",
    "How should completed todos be marked in storage?",
    "Suggest one edge case my remove() method must handle.",
    "What's a good name for the JSON persistence file?",
    "One sentence: how should I test the add() path?",
    "Summarize the design in one line.",
]

# Scenario 2: five short sessions that share the big prefix but DIVERGE at the
# very first user turn. Each is a different domain question, so nothing but the
# system + tools prefix is shared -- which is precisely what a stable
# prompt_cache_key is supposed to keep warm across sessions.
_DIVERGING_FIRST_TURNS = [
    "In one line, what does a load balancer do?",
    "In one line, what is a Python context manager?",
    "In one line, what is TCP backpressure?",
    "In one line, what is a database index?",
    "In one line, what is idempotency?",
]

# Scenario 3: tool-heavy. Each turn asks for a lookup that the model answers
# with a function_call; we synthesize the function_call_output and feed it back,
# so the tail of the prefix is tool traffic rather than plain prose.
_TOOL_PROMPTS = [
    "Read the file config.yaml and tell me the port in one line.",
    "Now read app.log and report the last error in one line.",
    "Read version.txt and report the version in one line.",
    "Read hosts.txt and report the first host in one line.",
    "Read status.json and report the state in one line.",
    "Read notes.md and report the title in one line.",
]

_TOOL_OUTPUTS = [
    "port: 8080",
    "ERROR connection reset by peer",
    "v2.3.1",
    "web-01.internal",
    '{"state": "healthy"}',
    "# Deployment notes",
]


def _long_messages(turn_idx: int, history: list[Message], prompt: str) -> list[Message]:
    """Append-only: add a user turn, keep all prior assistant/user turns."""
    history.append(Message.user(prompt + TERSE))
    return list(history)


def _tool_call_message(name: str, path: str) -> Message:
    """An assistant turn requesting a single read tool call.

    Uses a real ToolCall shape (id, name, arguments) so the Responses converter
    emits a proper ``function_call`` input item, exercising the tool-result
    breakpoint path rather than plain prose.
    """
    call = ToolCall(name=name, arguments={"path": path})
    return Message(role="assistant", tool_calls=[call])


def _tool_result_message(call_id: str, output: str) -> Message:
    """A tool result answering a specific call id (=> ``function_call_output``)."""
    from local_operator.harness.types import TextContent

    return Message(
        role="tool",
        content=[TextContent(text=output)],
        tool_call_id=call_id,
        tool_name="read",
    )


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------


async def _stream_turn(
    client: Any,
    request: ChatRequest,
    oauth_access: OAuthAccess | None,
) -> TurnUsage:
    """Drive one real provider turn and return its prompt-side usage.

    Usage arrives on ``StreamUsageEvent`` and/or the terminal ``StreamEndEvent``;
    we take the last non-empty usage seen. ``api_key`` is None because the OAuth
    bearer is carried on ``oauth_access`` and injected by the client's headers.
    """
    last_usage: Usage | None = None
    async for event in client.stream(request, None, oauth_access=oauth_access):
        if isinstance(event, StreamUsageEvent) and event.usage is not None:
            last_usage = event.usage
        elif isinstance(event, StreamEndEvent) and event.usage is not None:
            last_usage = event.usage
    if last_usage is None:
        return TurnUsage()
    return TurnUsage(
        input_tokens=last_usage.input_tokens,
        cache_read_tokens=last_usage.cache_read_tokens,
        cache_write_tokens=last_usage.cache_write_tokens,
        output_tokens=last_usage.output_tokens,
    )


async def run_long_session(
    client: Any,
    oauth_access: OAuthAccess | None,
    turns: int,
    reasoning_effort: str | None,
    scenario_name: str,
) -> ScenarioResult:
    """Scenario 1/4: one session, append-only turns."""
    spec = _resolve_spec(reasoning_effort)
    tools, system_blocks = _build_prefix()
    session_id = f"bench-{scenario_name}"
    result = ScenarioResult(name=scenario_name)
    history: list[Message] = []
    for i, prompt in enumerate(_LONG_PROMPTS[:turns]):
        messages = _long_messages(i, history, prompt)
        request = _make_request(spec, system_blocks, tools, messages, session_id)
        turn = await _stream_turn(client, request, oauth_access)
        result.turns.append(turn)
        # Keep the conversation growing so the next turn's prefix extends this
        # one. A terse canned assistant reply keeps output cost near zero while
        # preserving a realistic append-only history shape.
        history.append(Message.assistant("ok"))
    return result


async def run_stable_prefix(
    client: Any,
    oauth_access: OAuthAccess | None,
    turns: int,
) -> ScenarioResult:
    """Scenario 2: N separate sessions sharing the prefix, diverging at turn 1.

    Each session gets its OWN prompt_cache_key (session id) because that is how
    the harness assigns them -- the point of the fix is that a STABLE key per
    session still lets the shared system+tools prefix hit the server cache
    across sessions. ``turns`` here means "number of diverging sessions".
    """
    spec = _resolve_spec(None)
    tools, system_blocks = _build_prefix()
    result = ScenarioResult(name="stable_prefix")
    count = min(turns, len(_DIVERGING_FIRST_TURNS)) or len(_DIVERGING_FIRST_TURNS)
    for i in range(count):
        prompt = _DIVERGING_FIRST_TURNS[i]
        session_id = f"bench-stable-prefix-{i}"
        messages = [Message.user(prompt + TERSE)]
        request = _make_request(spec, system_blocks, tools, messages, session_id)
        turn = await _stream_turn(client, request, oauth_access)
        result.turns.append(turn)
    return result


async def run_tool_heavy(
    client: Any,
    oauth_access: OAuthAccess | None,
    turns: int,
) -> ScenarioResult:
    """Scenario 3: each turn carries a tool call + its result in the history."""
    spec = _resolve_spec(None)
    tools, system_blocks = _build_prefix()
    session_id = "bench-tool-heavy"
    result = ScenarioResult(name="tool_heavy")
    history: list[Message] = []
    count = min(turns, len(_TOOL_PROMPTS))
    for i in range(count):
        prompt = _TOOL_PROMPTS[i]
        history.append(Message.user(prompt + TERSE))
        # Model "asks" for a read; we answer with a synthesized tool result so
        # the NEXT turn's prefix ends in function_call + function_call_output.
        call_msg = _tool_call_message("read", f"./file_{i}.txt")
        history.append(call_msg)
        call_id = call_msg.tool_calls[0].id
        history.append(_tool_result_message(call_id, _TOOL_OUTPUTS[i]))
        request = _make_request(spec, system_blocks, tools, list(history), session_id)
        turn = await _stream_turn(client, request, oauth_access)
        result.turns.append(turn)
        history.append(Message.assistant("ok"))
    return result


# ---------------------------------------------------------------------------
# Dry-run: show the redacted body shape WITHOUT any network call
# ---------------------------------------------------------------------------


def _redact_body(body: dict[str, Any]) -> dict[str, Any]:
    """A compact, secret-free view of a request body for inspection.

    Large arrays (instructions, input, tools) are summarized by shape/size
    rather than dumped, and no credential is present in a body anyway (auth is
    header-only). This exists to CONFIRM routing and key presence, not to leak.
    """
    out: dict[str, Any] = {}
    for k, v in body.items():
        if k == "instructions":
            out[k] = f"<str {len(v)} chars>"
        elif k == "input":
            kinds: dict[str, int] = {}
            for item in v:
                key = item.get("type") or item.get("role") or "?"
                kinds[key] = kinds.get(key, 0) + 1
            out[k] = f"<{len(v)} items: {kinds}>"
        elif k == "tools":
            out[k] = f"<{len(v)} tool schemas>"
        else:
            out[k] = v
    return out


def _dry_run() -> None:
    """Print the redacted codex body shape for a representative turn per scenario.

    Builds the client and the exact ChatRequest each scenario's first turn would
    send, then renders the body via the client's own codex builder -- so what is
    printed is what would actually be POSTed. The absence of a
    ``prompt_cache_key`` key here on origin/main is the current bug made visible.
    """
    spec = _resolve_spec("medium")
    tools, system_blocks = _build_prefix()
    # ``client_for_spec`` is typed as the ``WireClient`` protocol; for an OpenAI
    # spec it is concretely an ``OpenAICompatClient``, whose codex body builder
    # and ``aclose`` this benchmark introspects. Assert the concrete type so the
    # introspection type-checks (CI runs whole-tree pyright).
    client = client_for_spec(spec, openai_api="responses")
    assert isinstance(client, OpenAICompatClient)

    print(f"Codex Responses URL: {CODEX_RESPONSES_URL}")
    print("prompt_cache_key set on ChatRequest: yes (= session id)\n")

    # Representative first-turn requests for each scenario.
    samples: list[tuple[str, ChatRequest]] = []
    samples.append(
        (
            "long_session",
            _make_request(
                spec,
                system_blocks,
                tools,
                [Message.user(_LONG_PROMPTS[0] + TERSE)],
                "bench-long_session",
            ),
        )
    )
    samples.append(
        (
            "stable_prefix",
            _make_request(
                spec,
                system_blocks,
                tools,
                [Message.user(_DIVERGING_FIRST_TURNS[0] + TERSE)],
                "bench-stable-prefix-0",
            ),
        )
    )
    # tool_heavy: a second-turn request whose history already carries a tool call.
    th_history: list[Message] = [Message.user(_TOOL_PROMPTS[0] + TERSE)]
    call_msg = _tool_call_message("read", "./file_0.txt")
    th_history.append(call_msg)
    th_history.append(_tool_result_message(call_msg.tool_calls[0].id, _TOOL_OUTPUTS[0]))
    samples.append(
        ("tool_heavy", _make_request(spec, system_blocks, tools, th_history, "bench-tool-heavy"))
    )

    for name, request in samples:
        # Use the client's OWN codex body builder so the printed shape is the
        # real wire body, not a reconstruction.
        body = client._build_codex_responses_body(request)  # noqa: SLF001 (bench introspection)
        redacted = _redact_body(body)
        print(f"--- scenario: {name} ---")
        print(f"  posts to: {CODEX_RESPONSES_URL}")
        print(f"  body keys: {sorted(body.keys())}")
        print(f"  has prompt_cache_key in body: {'prompt_cache_key' in body}")
        print(f"  redacted body: {json.dumps(redacted, indent=2)}")
        print()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_scenario_table(results: list[ScenarioResult]) -> None:
    header = (
        f"{'scenario':<16} {'turns':>5} {'input':>10} {'cache_read':>11} "
        f"{'cache_write':>11} {'rate %':>8}"
    )
    print(header)
    print("-" * len(header))
    total_read = 0
    total_denom = 0
    for r in results:
        if r.error:
            print(f"{r.name:<16} ERROR: {r.error}")
            continue
        total_read += r.total_cache_read
        total_denom += r.denom
        print(
            f"{r.name:<16} {len(r.turns):>5} {r.total_input:>10} "
            f"{r.total_cache_read:>11} {r.total_cache_write:>11} "
            f"{r.cache_rate * 100:>7.1f}%"
        )
    print("-" * len(header))
    overall = (total_read / total_denom * 100) if total_denom else 0.0
    print(f"{'OVERALL':<16} {'':>5} {'':>10} {total_read:>11} {'':>11} {overall:>7.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_SCENARIO_RUNNERS: dict[str, Callable[..., Any]] = {
    "long_session": lambda c, o, t: run_long_session(c, o, t, None, "long_session"),
    "stable_prefix": run_stable_prefix,
    "tool_heavy": run_tool_heavy,
    "reasoning_on": lambda c, o, t: run_long_session(c, o, t, "medium", "reasoning_on"),
}


async def _run_live(scenario: str | None, turns: int) -> int:
    store = AuthStore()
    oauth_access = await store.get_oauth_access("openai", read_only=True)
    # The Codex Responses route only engages for an OAuth grant carrying an
    # org_id (see OpenAICompatClient._codex_responses_mode). Anything else means
    # there is no ChatGPT subscription credential to measure.
    if oauth_access is None or oauth_access.kind != "oauth" or not oauth_access.org_id:
        print("skipped: no OpenAI OAuth credential")
        return 0

    spec = _resolve_spec(None)
    client = client_for_spec(spec, openai_api="responses")
    assert isinstance(client, OpenAICompatClient)

    names = [scenario] if scenario else list(_SCENARIO_RUNNERS.keys())
    results: list[ScenarioResult] = []
    try:
        for name in names:
            runner = _SCENARIO_RUNNERS.get(name)
            if runner is None:
                print(f"unknown scenario: {name}")
                continue
            try:
                result = await runner(client, oauth_access, turns)
            except Exception as exc:  # noqa: BLE001 - measurement must not crash
                # A 400 from the Codex backend (e.g. when a new field is added)
                # is itself important signal; capture it redacted rather than
                # letting it abort the whole run.
                result = ScenarioResult(name=name, error=f"{type(exc).__name__}: {exc}"[:400])
            results.append(result)
    finally:
        await client.aclose()

    _print_scenario_table(results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live OAuth prompt-cache benchmark.")
    parser.add_argument(
        "--scenario",
        choices=sorted(_SCENARIO_RUNNERS.keys()),
        default=None,
        help="Run one scenario (default: all).",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=6,
        help="Turns per session (scenario 1/3/4) or number of sessions (scenario 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the redacted request body shape without any network call.",
    )
    args = parser.parse_args()

    if args.dry_run:
        _dry_run()
        return 0

    return asyncio.run(_run_live(args.scenario, args.turns))


if __name__ == "__main__":
    # Exit 0 always: this is measurement, not a gate.
    raise SystemExit(main())
