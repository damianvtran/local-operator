"""Live Anthropic measurement: does an aside's ``tool_choice`` break the cache?

``scripts/measure_advisor_cache.py`` established that the advisor / ``/btw``
request shape rides the working turn's cached prefix — but it did so on a
~14k-token toy conversation whose tools+system HEAD was most of the prompt.
Fleet analytics on real sessions (average context 184k) then showed a
large share of daily cache-WRITE tokens on calls with the signature
``cache_read ≈ head only, cache_write ≈ the whole message tail``, and
Anthropic's prompt-caching docs ("What invalidates the cache") offered a
candidate mechanism: ``tool_choice`` is rendered into the MESSAGES level of
the tools -> system -> messages hierarchy, so a request that sends
``{"type": "none"}`` against a prefix written with ``{"type": "auto"}``
WOULD keep the head and re-write every message block. The toy measurement
could not have seen that because its tail was ~10% of the prompt.

Hypothesis under test: the aside's ``none`` splits the cache at the messages
level, and sending the turn's own ``auto`` closes the split. The script builds
a ~35k-token conversation of tool-result-shaped pairs (the thing a real
session is made of) and makes four REAL streamed calls, printing the
provider's own cache counters for each:

1. WARM        — a working-turn-shaped request: tools + ``tool_choice=auto``.
                 Writes the prefix.
2. ASIDE-OLD   — the advisor/aside shape with the PRE-FIX wire body
                 (``tool_choice: {"type": "none"}``). If the hypothesis holds,
                 ``cache_read`` is roughly the tools+system head and
                 ``cache_write`` is the rest of the conversation.
3. RE-WARM     — re-writes the turn's prefix so (4) is measured against the
                 turn, not against (2)'s ``none`` variant.
4. ASIDE-FIXED — the same request through the SHIPPED client, which sends the
                 turn's own ``{"type": "auto"}`` when tools are present. If the
                 hypothesis holds, ``cache_read`` is the whole conversation and
                 ``cache_write`` is only the appended question.

Result: the hypothesis did NOT hold. Full output, both runs, in
``docs/evidence/compaction-advisor/aside-tool-choice-measurement.txt``. Arm 2
(``none``) read the full 36,578-token prefix and wrote only its 674-token
appended question — identical to arm 4 — on ``claude-opus-5`` and
``claude-fable-5-1``. Do not expect a cache split from this script on a warm
account; if arm 2 reads 0, suspect the settle gap first (the script's first
run did exactly that with a 1s settle, and arm 3 read 0 too — hence
``SETTLE_SECONDS``). The fleet's head-only signature was later root-caused to
per-account cache isolation under reserve-verdict account rotation (PR #537).
The ``none`` -> ``auto`` mapping ships as hygiene (aside body byte-identical
to the turn's), not as a measured saving.

Deliberately small: four calls of ~35k tokens each, ``max_tokens`` capped and
the lowest reasoning effort, because the shared OAuth accounts run close to
their five-hour caps. The request shapes are the real ones (``ChatRequest``
through ``AnthropicClient``), not hand-built JSON, so what is measured is what
``Session.complete_aside`` / ``Session.advise_compaction`` send.

Run (needs the configured Anthropic OAuth credential):
    .venv/bin/python scripts/measure_aside_tool_choice_cache.py
    .venv/bin/python scripts/measure_aside_tool_choice_cache.py --model claude-fable-5-1 --arms 2

``--arms 2`` stops after the pre-fix arm: two calls instead of four, for
checking a second model against the same rule without spending the re-warm.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.advisor import ADVISOR_SYSTEM_PROMPT  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    AgentTool,
    ChatRequest,
    Message,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    ToolResult,
    Usage,
)
from local_operator.model.configure import build_model_spec  # noqa: E402
from local_operator.providers.auth_store import AuthStore, default_db_path  # noqa: E402
from local_operator.providers.clients import AnthropicClient  # noqa: E402

#: Cheaper than the default TUI model and on the same cache rules; the effect
#: is a property of the wire, not of the model.
MODEL_ID = "claude-opus-5"

#: Tool-result pairs in the conversation. Twelve pairs of ~6 KB code-shaped results land
#: around 35k tokens, which is the smallest size at which a head-only hit and a
#: full hit are unmistakably different numbers.
PAIRS = 12

#: Pause between arms. The docs say a cache entry "only becomes available
#: after the first response begins", and in practice a one-second gap after a
#: warm call that produced no output was NOT enough: the next arm read 0 and
#: wrote its own entry, which a later arm then read in full — a result that
#: looks like the docs being wrong and is really the entry still propagating.
SETTLE_SECONDS = 5

#: Output cap on every call: this measures the PROMPT side, and every output
#: token is paid for on accounts that are near their caps.
MAX_TOKENS = 200


async def _noop(*_a: Any, **_k: Any) -> ToolResult:
    raise AssertionError("tool must not run")


def _tools() -> list[AgentTool]:
    """A realistic core-tool surface, so the tools block is a real chunk of the
    cache prefix (it sits at position 0 on every wire)."""
    names = ["bash", "read", "write", "edit", "grep", "glob", "eval", "task", "todo"]
    return [
        AgentTool(
            name=name,
            description=(
                f"{name}: {name} tool with the usual parameters. Use it when the task "
                f"calls for {name}-shaped work and read the result before acting on it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "target path"},
                    "content": {"type": "string", "description": "payload"},
                    "i": {"type": "string", "description": "intent"},
                },
            },
            execute=_noop,
        )
        for name in names
    ]


def _fake_tool_output(index: int) -> str:
    """~6 KB of code-shaped text: what a ``read`` or ``grep`` actually returns.

    Deterministic, so re-runs produce the same prefix and can be compared
    across days; varied per index, so no two results are the same bytes and
    the provider cannot collapse them.
    """
    lines = []
    for line in range(40):
        lines.append(
            f"{line + 1:4d}| def handler_{index}_{line}(request, *, retries={line % 5}):  "
            f"# step {index}: validate then dispatch; see ticket LOP-{index * 100 + line}"
        )
        lines.append(
            f"{line + 1:4d}|     return dispatch(request, key='k{index}-{line}', "
            f"budget={(line * 37 + index) % 1000})"
        )
    return "\n".join(lines)


def _conversation() -> list[Message]:
    """A tool-heavy exchange: user asks, assistant calls a tool, tool answers.

    The shape matters as much as the size. Real sessions are mostly tool
    results, which Anthropic renders as ``tool_result`` blocks inside user
    messages, and those are exactly the "message blocks" the docs say a
    ``tool_choice`` change invalidates.
    """
    convo: list[Message] = [Message.user("Port the request handlers to the new dispatcher.")]
    for i in range(PAIRS):
        call_id = f"call_{i:02d}"
        convo.append(
            Message(
                role="assistant",
                content=[TextContent(text=f"Reading handler module {i}.")],
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name="read",
                        arguments={"path": f"src/handlers/module_{i}.py"},
                    )
                ],
            )
        )
        convo.append(
            Message.tool_result(
                ToolResult(
                    tool_call_id=call_id,
                    tool_name="read",
                    content=[TextContent(text=_fake_tool_output(i))],
                )
            )
        )
    convo.append(Message.assistant("All twelve modules read. Ready to port them."))
    convo.append(Message.user("Go ahead, but summarise what you will change first, briefly."))
    return convo


SYSTEM = [
    "You are a helpful coding agent working in a large repository. " * 30,
    "Environment: macOS, python 3.14, uv-managed venv. " * 20,
]

#: The advisor's appended turn, as ``build_advisor_prompt`` shapes it: the
#: instructions ride INSIDE the user turn (a system block would break the
#: prefix ahead of the messages — measured in ``measure_advisor_cache.py``).
ADVISOR_QUESTION = (
    f"{ADVISOR_SYSTEM_PROMPT}\n\n"
    "A compaction decision is pending for the conversation above.\n\n"
    "Context size: 480,000 tokens.\nAutomatic compaction threshold: 600,000 tokens.\n\n"
    "Candidate anchors, oldest first — `preserve_from` must be one of these ids:\n"
    "- m1 (user): Port the request handlers to the new dispatcher.\n\n"
    "Answer with the fenced JSON block described in your instructions and nothing else — "
    "as text, without calling any tool."
)


class PreFixAnthropicClient(AnthropicClient):
    """The wire body as shipped BEFORE this fix: ``none`` goes out literally.

    Overriding the body builder rather than hand-writing JSON keeps every other
    byte identical to the fixed client, so the only difference between arms 2
    and 4 is the ``tool_choice`` value — which is the thing being measured.
    """

    def _build_body(self, request: ChatRequest, *, oauth: bool = False) -> dict[str, Any]:
        body = super()._build_body(request, oauth=oauth)
        if request.tools and request.tool_choice == "none":
            body["tool_choice"] = {"type": "none"}
        return body


async def _run(
    client: AnthropicClient, oauth: Any, request: ChatRequest, label: str
) -> Usage | None:
    usage = None
    async for event in client.stream(request, None, oauth_access=oauth):
        if isinstance(event, StreamUsageEvent):
            usage = event.usage
    if usage is None:
        print(f"{label}: no usage reported")
        return None
    total = (
        (usage.cache_read_tokens or 0) + (usage.cache_write_tokens or 0) + (usage.input_tokens or 0)
    )
    rate = (usage.cache_read_tokens or 0) / total * 100 if total else 0.0
    print(
        f"{label}: cache_read={usage.cache_read_tokens} "
        f"cache_write={usage.cache_write_tokens} input={usage.input_tokens} "
        f"output={usage.output_tokens} context={usage.context_tokens} "
        f"cache_hit={rate:.1f}%"
    )
    return usage


async def main(model_id: str = MODEL_ID, arms: int = 4) -> None:
    base = build_model_spec("anthropic", model_id)
    # Lowest effort on EVERY arm: effort is rendered into the prompt too, so it
    # must not differ between the turn and the aside, and the bottom rung
    # spends the fewest output tokens.
    efforts = base.reasoning_efforts
    spec = base.model_copy(update={"reasoning_effort": efforts[0]}) if efforts else base
    store = AuthStore(default_db_path())
    oauth = await store.get_oauth_access("anthropic")

    tools = _tools()
    convo = _conversation()
    fixed = AnthropicClient()
    prefix = PreFixAnthropicClient()

    warm = ChatRequest(
        model=spec,
        system_blocks=list(SYSTEM),
        messages=list(convo),
        tools=tools,
        tool_choice="auto",
        max_tokens=MAX_TOKENS,
    )
    aside = ChatRequest(
        model=spec,
        system_blocks=list(SYSTEM),
        messages=[*convo, Message.user(ADVISOR_QUESTION)],
        tools=tools,
        tool_choice="none",
        max_tokens=MAX_TOKENS,
        replayable=True,
    )
    assert prefix._build_body(aside, oauth=True)["tool_choice"] == {"type": "none"}
    assert fixed._build_body(aside, oauth=True)["tool_choice"] == {"type": "auto"}

    print(f"model={spec.model_id} effort={spec.reasoning_effort} pairs={PAIRS}")
    await _run(fixed, oauth, warm, "1 WARM   (turn, tools, auto)        ")
    await asyncio.sleep(SETTLE_SECONDS)
    old = await _run(prefix, oauth, aside, "2 ASIDE  pre-fix  (tool_choice none)")
    if arms <= 2:
        return
    await asyncio.sleep(SETTLE_SECONDS)
    await _run(fixed, oauth, warm, "3 RE-WARM (turn, tools, auto)       ")
    await asyncio.sleep(SETTLE_SECONDS)
    new = await _run(fixed, oauth, aside, "4 ASIDE  fixed    (tool_choice auto)")

    if old and new:
        print(
            f"\nRESULT: aside cache_read pre-fix={old.cache_read_tokens} "
            f"fixed={new.cache_read_tokens}; cache_write pre-fix={old.cache_write_tokens} "
            f"fixed={new.cache_write_tokens}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--arms", type=int, default=4, choices=(2, 4))
    args = parser.parse_args()
    asyncio.run(main(args.model, args.arms))
