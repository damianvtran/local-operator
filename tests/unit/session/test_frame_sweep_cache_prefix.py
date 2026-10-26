"""Consecutive Anthropic requests across a new-frame event keep the cached prefix.

WHY THIS EXISTS. Subagent prompt-cache prefix breaks were the largest single
cache cost on this fleet: 54% of all Anthropic cache-write tokens over 7 days
(analytics ``calls`` table, breaks = a call <5 min after the previous one whose
cache write is >= 20k tokens and >= 50% of context). Replaying the surviving
transcripts through ``AnthropicClient._build_body`` put the first divergent
byte of those breaks on a blank the session's boundary sweep had just written
over an EARLIER message: a stale screenshot or a superseded re-read, which the
previous request had already carried under a cache marker. Mutating it
invalidates every entry written since. A screenshot loop blanks one such frame
per call, so the provider's ~20-block lookback quickly runs out of reachable
entries and the read collapses to the system head.

Two defects combined, and this test pins both through the real path (Session
-> tool loop -> mid-turn boundary hook -> ``_plan_compaction`` sweep -> the
request the next call sends):

* ``_last_provider_request_ms`` was stamped only when a RUN ended, so for a
  subagent's single long run it stayed 0 and every boundary looked 90 minutes
  idle, which let the idle flush blank everything;
* the warm guard was a token budget (suffix < 8,000 tokens may be rewritten),
  which a frame directly behind the newest one always satisfies.

The assertion is the acceptance property itself: for every pair of
consecutive requests, the second one's encoded messages are byte-identical to
the first one's through the first one's FINAL cache marker. On the old code it
fails at the first new-frame boundary after two frames are resident.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    ImageContent,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
    Usage,
)
from local_operator.providers.clients import AnthropicClient
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

#: Big enough that the context-size gate never fires a summary: this test is
#: about the prune sweep that runs at every boundary, not about compaction.
MODEL = ModelSpec(
    provider="anthropic",
    model_id="claude-opus-5-5",
    context_window=1_000_000,
    supports_images=True,
)

#: Screens the subagent cycles through, re-reading each, the shape of the
#: designer runs that produced the measured breaks (0d598f9ee38e re-read four
#: PNGs in rotation). DISTINCT bytes per screen so a blank is observable.
SCREENS = [f"/scratch/shot-{index}.png" for index in range(3)]
_PIXELS = {path: base64.b64encode(f"png:{path}".encode() * 64).decode() for path in SCREENS}

#: A production-sized brief (~480 KB of text). Load-bearing: the mid-turn
#: boundary reaches the sweep only when the cheap pre-gate cannot rule
#: compaction out, and that gate's ``messages_tokens_upper_bound`` counts text
#: at one token per BYTE. The measured subagents carried 150-280k-token
#: histories (~0.6-1.1 MB), which always cross the 400k trigger, so they swept
#: at every boundary. The exact estimate (~4 bytes/token, ~120k) stays under
#: the trigger, so no compaction pass runs: only the sweep is exercised. A
#: small fixture skips the sweep and passes on any code.
BRIEF = "Audit these screens. " + "context for the audit, " * 21_000


def _read_frame_tool() -> AgentTool:
    """A ``read`` tool returning an image, shaped like the real image read."""

    async def execute(tool_call_id, args, signal, on_update, context):
        path = args["path"]
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="read",
            content=[
                TextContent(text=f"Image {path} (image/png, 1024x576)"),
                ImageContent(data=_PIXELS[path], mime_type="image/png"),
            ],
            details={"path": path, "mime_type": "image/png"},
        )

    return AgentTool(
        name="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )


def _browser_tool() -> AgentTool:
    """A ``browser`` screenshot that WRITES a capture file, as both hosts do.

    Load-bearing: the stale-frame sweep is gated on the history showing a
    browser capture (``_history_captured_browser_views``: ``details`` carries
    the file ``path`` and its ``bytes``). The measured designer subagents
    screenshotted with ``browser`` and then ``read`` the PNGs back.
    """

    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="browser",
            content=[TextContent(text=f"Screenshot saved to {SCREENS[0]} (226612 bytes).")],
            details={"path": SCREENS[0], "bytes": 226_612},
        )

    return AgentTool(
        name="browser",
        parameters={"type": "object", "properties": {"action": {"type": "string"}}},
        execute=execute,
    )


class RecordingStream:
    """Scripted provider: one frame read per call, then a closing answer.

    Each call reports usage, exactly as a real Anthropic stream does: that
    usage completing a call is what must advance the session's provider clock.
    """

    def __init__(self, reads: list[str]) -> None:
        self.reads = reads
        self.requests: list[ChatRequest] = []
        #: Each turn request ENCODED AT SEND TIME. Load-bearing: the sweep
        #: blanks messages IN PLACE, and a recorded ChatRequest holds the same
        #: Message objects, so encoding it afterwards would show the blank in
        #: the earlier request too and hide exactly the divergence under test.
        self.sent: list[tuple[list[str], int]] = []
        #: Image-bearing messages per turn request, also counted at send time.
        self.frames_sent: list[int] = []

    def __call__(self, request: ChatRequest, signal: Any):
        self.requests.append(request)
        if request.purpose == "turn":
            self.sent.append(_encoded_messages(request))
            self.frames_sent.append(
                sum(
                    1
                    for message in request.messages
                    if any(isinstance(block, ImageContent) for block in message.content)
                )
            )
        index = len(self.requests) - 1
        usage = Usage(input_tokens=10, output_tokens=5, context_tokens=50_000 + index)

        async def gen():
            if request.purpose != "turn":
                yield StreamTextDelta(delta="summary")
                yield StreamEndEvent(stop_reason="stop", usage=usage)
                return
            if index == 0:
                yield StreamToolCallDelta(
                    index=0,
                    id="call-capture",
                    name="browser",
                    argument_delta=json.dumps({"action": "screenshot"}),
                )
                yield StreamEndEvent(stop_reason="toolUse", usage=usage)
            elif index <= len(self.reads):
                arguments = json.dumps({"path": self.reads[index - 1]})
                yield StreamToolCallDelta(
                    index=0, id=f"call-{index}", name="read", argument_delta=arguments
                )
                yield StreamEndEvent(stop_reason="toolUse", usage=usage)
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop", usage=usage)

        return gen()


def _encoded_messages(request: ChatRequest) -> tuple[list[str], int]:
    """The request's ``messages`` exactly as the Anthropic wire encodes them.

    Returns one canonical JSON string per message with ``cache_control``
    removed (markers MOVE between requests by design; the bytes they cover
    must not change), plus the index of the message holding the request's
    final message marker.
    """
    body = AnthropicClient()._build_body(request)
    encoded: list[str] = []
    final_marker = -1
    for index, message in enumerate(body["messages"]):
        content = message["content"]
        if isinstance(content, list) and any("cache_control" in block for block in content):
            final_marker = index
        stripped = {
            "role": message["role"],
            "content": (
                [
                    {key: value for key, value in block.items() if key != "cache_control"}
                    for block in content
                ]
                if isinstance(content, list)
                else content
            ),
        }
        encoded.append(json.dumps(stripped, sort_keys=True))
    return encoded, final_marker


@pytest.mark.asyncio
async def test_new_frame_never_rewrites_the_previous_cached_prefix(tmp_path, monkeypatch):
    reads = [SCREENS[index % len(SCREENS)] for index in range(8)]
    stream = RecordingStream(reads)
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[_browser_tool(), _read_frame_tool()],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable instructions", "inventory", "env", "skills"],
        compaction_settings=CompactionSettings(),
    )
    sweeps: list[int] = []
    from local_operator.compaction import api as compaction_api

    real_sweep = compaction_api.prune_stale_frames_in_place

    def counting_sweep(*args: Any, **kwargs: Any):
        sweeps.append(1)
        return real_sweep(*args, **kwargs)

    monkeypatch.setattr(compaction_api, "prune_stale_frames_in_place", counting_sweep)
    try:
        await session.prompt(BRIEF)
    finally:
        await session.dispose()
    assert len(sweeps) >= len(reads), f"the frame sweep ran only {len(sweeps)} times"

    assert len(stream.sent) == len(reads) + 2
    for (before, final_marker), (after, _) in zip(stream.sent, stream.sent[1:]):
        assert final_marker >= 0
        covered = final_marker + 1
        assert after[:covered] == before[:covered], (
            "an earlier message changed between consecutive requests: the "
            "cached prefix through the previous final marker was rewritten"
        )

    # Negative control: the sweep RAN at every boundary (asserted above) with
    # stale frames available to elide. Every read stays resident while the
    # cache is warm (newest-two retention would otherwise leave 2), so the
    # append-only history is the guard holding, not a sweep that found nothing.
    assert stream.frames_sent[-1] == len(reads)
