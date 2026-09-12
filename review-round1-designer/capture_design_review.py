"""Design-review captures for PR #993 (worktree lo-switch-leak).

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/dshot.py MODE out.svg

Modes:
  after     the committed after-state (4 leaked hops hidden, receipt posted)
  control   the SAME receipt over a history with NO leaked hop rows at all
  hops4     the same hidden rows, but four REAL retry notices (one per hop)
  firstrow  the resumed transcript whose FIRST row is a leaked notice
  tooladj   a tool card immediately before/after the leaked rows
  live      a real turn: prompt, tool card, retry notice, assistant line
"""

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

REPO = "/Users/damian/workspace/repos/lo-switch-leak"
sys.path.insert(0, REPO)

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.compaction.cutpoint import RENDERED_INJECTION_KEY  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    Message,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.incidents import format_model_switch_message  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import (  # noqa: E402
    EffectiveModelChanged,
    NoticePosted,
    RetryStarted,
)
from local_operator.tui.widgets.transcript import (  # noqa: E402
    TranscriptView,
    UserBlock,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

HOPS = (
    ("zai/glm-5.3", "anthropic quota exhausted (0% remaining)"),
    ("kimi/k3", "provider failure"),
    ("alibaba-token-plan/qwen3.8-max", "provider failure"),
    ("xai/grok-4.6", "provider failure"),
)


def _leaked_hops() -> list[Message]:
    return [
        Message(
            role="user",
            content=[
                TextContent(
                    text=format_model_switch_message(
                        new_label, "anthropic/claude-opus-5", reason=reason, transient=True
                    )
                )
            ],
            provider_payload={RENDERED_INJECTION_KEY: True},
        )
        for new_label, reason in HOPS
    ]


class _FallbackSession(FakeSession):
    @property
    def model_label(self) -> str:
        return "anthropic/claude-opus-5"

    @property
    def model(self):
        return SimpleNamespace(
            provider="anthropic",
            model_id="claude-opus-5",
            display_name="Claude Opus 5",
            context_window=200_000,
            reasoning_effort=None,
            reasoning_efforts=(),
            reasoning=False,
        )

    @property
    def effective_model(self):
        return SimpleNamespace(
            provider="xai",
            model_id="grok-4.6",
            display_name="Grok 4.6",
            context_window=256_000,
            reasoning_effort=None,
            reasoning_efforts=(),
            reasoning=False,
        )

    @property
    def effective_model_label(self):
        return "xai/grok-4.6"


async def _pump_until(pilot: Any, predicate: Any, attempts: int = 300) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("the resumed replay never painted")


PROMPT = "check the MiniMax subset state before the next batch"
ANSWER = "Checking the MiniMax subset state after the model switch."


def _history(mode: str) -> list[Any]:
    if mode == "control":
        return [Message.user(PROMPT), Message.assistant(ANSWER)]
    if mode == "firstrow":
        return [*_leaked_hops(), Message.user(PROMPT), Message.assistant(ANSWER)]
    if mode == "tooladjcontrol":
        # A tool card sits between the prompt and the notices, and the notices
        # sit between the tool card and the assistant line: the adjacency a
        # hidden row could orphan.
        return [
            Message.user(PROMPT),
            Message.assistant(
                "", tool_calls=[ToolCall(id="call_1", name="bash", arguments={"command": "ls"})]
            ),
            Message.tool_result(
                ToolResult(
                    tool_call_id="call_1",
                    tool_name="bash",
                    content=[TextContent(text="wait_reasoning.txt\nlocal_operator\n")],
                )
            ),
            Message.assistant(ANSWER),
        ]
    if mode == "tooladj":
        # A tool card sits between the prompt and the notices, and the notices
        # sit between the tool card and the assistant line: the adjacency a
        # hidden row could orphan.
        return [
            Message.user(PROMPT),
            Message.assistant(
                "", tool_calls=[ToolCall(id="call_1", name="bash", arguments={"command": "ls"})]
            ),
            Message.tool_result(
                ToolResult(
                    tool_call_id="call_1",
                    tool_name="bash",
                    content=[TextContent(text="wait_reasoning.txt\nlocal_operator\n")],
                )
            ),
            *_leaked_hops(),
            Message.assistant(ANSWER),
        ]
    if mode == "realreceipt":
        return [Message.user(PROMPT)]
    if mode == "onlyinjections":
        return [*_leaked_hops()]
    if mode == "deliberate":
        from local_operator.harness.types import CustomMessage

        return [
            Message.user(PROMPT),
            CustomMessage(
                custom_type="session_model_switch",
                attribution="system",
                details={
                    "text": format_model_switch_message(
                        "zai/glm-5.3", "anthropic/claude-opus-5", reason="", transient=False
                    ),
                    "new_label": "zai/glm-5.3",
                    "previous_label": "anthropic/claude-opus-5",
                    "transient": False,
                },
            ),
            Message.assistant(ANSWER),
        ]
    if mode in ("liveorder", "resume"):
        return [Message.user(PROMPT)] if mode == "liveorder" else [
            Message.user(PROMPT), *_leaked_hops(), Message.assistant(ANSWER)
        ]
    if mode == "live":
        return [
            Message.user(PROMPT),
            Message.assistant(
                "", tool_calls=[ToolCall(id="call_1", name="bash", arguments={"command": "ls"})]
            ),
            Message.tool_result(
                ToolResult(
                    tool_call_id="call_1",
                    tool_name="bash",
                    content=[TextContent(text="wait_reasoning.txt\nlocal_operator\n")],
                )
            ),
            Message.assistant(ANSWER),
        ]
    return [Message.user(PROMPT), *_leaked_hops(), Message.assistant(ANSWER)]


def _receipt(app: OperatorApp, mode: str) -> None:
    if mode == "hops4":
        # The live path's shape for a four-hop fallover: one retry notice per
        # hop, all of them attempt 1 (a per-request retry counter restarts).
        for label, reason in HOPS:
            app.post_message(RetryStarted(1, reason, label))
        app.post_message(EffectiveModelChanged("xai", "grok-4.6", None, "provider failure", True))
        return
    if mode in (
        "firstrow", "tooladj", "tooladjcontrol", "resume", "onlyinjections", "deliberate"
    ):
        return
    if mode == "realreceipt":
        # The REAL live narration: the route-change notice per hop, then the
        # recovery edge (configure.py::_on_route_change / _on_route_settle).
        for label, reason in HOPS:
            app.post_message(NoticePosted(f"{reason} — falling back to {label}", "warning"))
        app.post_message(EffectiveModelChanged("xai", "grok-4.6", None, "provider failure", True))
        app.post_message(NoticePosted("back to anthropic/claude-opus-5", "info"))
        return
    if mode == "liveorder":
        for label, reason in HOPS:
            app.post_message(RetryStarted(1, reason, label))
        app.post_message(EffectiveModelChanged("xai", "grok-4.6", None, "provider failure", True))
        return
    if mode == "realreceipt":
        # The REAL live narration: the route-change notice per hop, then the
        # recovery edge (configure.py::_on_route_change / _on_route_settle).
        for label, reason in HOPS:
            app.post_message(NoticePosted(f"{reason} — falling back to {label}", "warning"))
        app.post_message(EffectiveModelChanged("xai", "grok-4.6", None, "provider failure", True))
        app.post_message(NoticePosted("back to anthropic/claude-opus-5", "info"))
        return
    app.post_message(RetryStarted(1, "anthropic quota exhausted (0% remaining)", "zai/glm-5.3"))
    app.post_message(NoticePosted("provider failure — falling back to zai/glm-5.3", "warning"))
    app.post_message(EffectiveModelChanged("xai", "grok-4.6", None, "provider failure", True))


async def main() -> None:
    mode = sys.argv[1]
    session = _FallbackSession()
    session._history = _history(mode)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 46)) as pilot:
        await pilot.pause()
        transcript = app.query_one(TranscriptView)
        if mode == "onlyinjections":
            for _ in range(20):
                await pilot.pause()
        else:
            await _pump_until(
                pilot, lambda: any(isinstance(b, UserBlock) for b in transcript.blocks())
            )
        _receipt(app, mode)
        for _ in range(4):
            await pilot.pause()
        if mode == "liveorder":
            # The answer lands AFTER the failing request, so this is the real
            # adjacency: the receipt sits between the prompt and the reply.
            app._ensure_streaming_block().update_text(ANSWER)
        for _ in range(8):
            await pilot.pause()
        save_capture(app, sys.argv[2])
        # The settle check: does another frame move anything?
        await pilot.pause()
        save_capture(app, sys.argv[2].replace(".svg", "-settled.svg"))
    print("MODE", mode, "blocks", len(app.query_one(TranscriptView).blocks()))


asyncio.run(main())