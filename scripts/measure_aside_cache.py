"""Live Anthropic measurement of the aside prompt-cache fix.

Makes three REAL streamed calls against the configured Anthropic OAuth
credential on ``claude-opus-4-8`` and reads the provider-reported cache tokens:

1. WARM: a working-turn-shaped request (system + a real conversation + the
   full tool schema). This writes the cache prefix (tools -> system -> messages).
2. ASIDE-OLD: the same conversation with tools=[] and tool_choice="none"
   (the regression). Its prefix diverges at position 0, so cache_read should be
   low relative to the prompt.
3. ASIDE-NEW: the same conversation with the live tools restored and
   tool_choice="none" (the fix). Its tools+system head matches the warm turn,
   so cache_read should be high.

Prints cache_read / cache_write / input for each so the before/after is a
measured number, not an assertion.
"""

from __future__ import annotations

import asyncio

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    StreamUsageEvent,
)
from local_operator.model.configure import build_model_spec
from local_operator.providers.auth_store import AuthStore, default_db_path
from local_operator.providers.clients import client_for_spec


def _noop(*_a, **_k):
    raise AssertionError("tool must not run")


def _tools() -> list[AgentTool]:
    # A handful of realistic schemas so the tools block is a meaningful chunk of
    # the prefix (the real session ships ~a dozen core tools).
    names = ["bash", "read", "write", "edit", "grep", "glob", "eval", "task"]
    return [
        AgentTool(
            name=n,
            description=f"{n} tool: does {n} things with useful parameters",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "target path"},
                    "content": {"type": "string", "description": "payload"},
                },
            },
            execute=_noop,
        )
        for n in names
    ]


def _conversation() -> list[Message]:
    # A long-ish exchange so cache read/write token counts are clearly visible.
    convo: list[Message] = []
    for i in range(6):
        convo.append(
            Message.user((f"Step {i}: please explain concept number {i} in detail. " * 8).strip())
        )
        convo.append(Message.assistant((f"Concept {i} explained thoroughly. " * 40).strip()))
    # End on a user message so the WARM request is a legal working turn (the
    # model generates the reply). The asides append their own user question
    # after this, so all three requests share the same long prefix.
    convo.append(Message.user(("Now summarise everything above concisely. " * 6).strip()))
    return convo


SYSTEM = [
    "You are a helpful coding agent. " * 30,
    "Environment: macOS, python 3.14. " * 20,
]


async def _run(client, spec, oauth, request, label):
    usage = None
    async for event in client.stream(request, None, oauth_access=oauth):
        if isinstance(event, StreamUsageEvent):
            usage = event.usage
    if usage is None:
        print(f"{label}: no usage reported")
        return None
    print(
        f"{label}: cache_read={usage.cache_read_tokens} "
        f"cache_write={usage.cache_write_tokens} input={usage.input_tokens} "
        f"output={usage.output_tokens} context={usage.context_tokens}"
    )
    return usage


async def main() -> None:
    spec = build_model_spec("anthropic", "claude-opus-4-8")
    store = AuthStore(default_db_path())
    oauth = await store.get_oauth_access("anthropic")
    client = client_for_spec(spec)

    tools = _tools()
    convo = _conversation()

    warm = ChatRequest(model=spec, system_blocks=list(SYSTEM), messages=list(convo), tools=tools)
    aside_old = ChatRequest(
        model=spec,
        system_blocks=list(SYSTEM),
        messages=[*convo, Message.user("why did you pick that?")],
        tools=[],
        tool_choice="none",
    )
    aside_new = ChatRequest(
        model=spec,
        system_blocks=list(SYSTEM),
        messages=[*convo, Message.user("why did you pick that?")],
        tools=tools,
        tool_choice="none",
    )

    # 1) Warm the cache with a working-turn shape.
    await _run(client, spec, oauth, warm, "WARM (turn, tools)")
    await asyncio.sleep(1)
    # 2) Old aside behaviour: tools dropped -> prefix diverges at position 0.
    old = await _run(client, spec, oauth, aside_old, "ASIDE-OLD (tools=[])")
    await asyncio.sleep(1)
    # Re-warm so the aside-new measurement is not polluted by aside-old having
    # written a no-tools prefix.
    await _run(client, spec, oauth, warm, "RE-WARM (turn, tools)")
    await asyncio.sleep(1)
    # 3) New aside behaviour: tools restored -> tools+system head matches turn.
    new = await _run(client, spec, oauth, aside_new, "ASIDE-NEW (tools restored)")

    if old and new:
        print(
            f"\nRESULT: cache_read old={old.cache_read_tokens} "
            f"new={new.cache_read_tokens} "
            f"(delta={new.cache_read_tokens - old.cache_read_tokens})"
        )


if __name__ == "__main__":
    asyncio.run(main())
