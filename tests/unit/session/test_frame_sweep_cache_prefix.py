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


def _divergences(sent: list[tuple[list[str], int]]) -> list[int]:
    """Request indices whose cached prefix (through the previous final
    marker) differs from the request before them."""
    broken: list[int] = []
    for index in range(1, len(sent)):
        before, final_marker = sent[index - 1]
        after, _ = sent[index]
        covered = final_marker + 1
        if after[:covered] != before[:covered]:
            broken.append(index)
    return broken


def _session(tmp_path, stream, model: ModelSpec = MODEL) -> Session:
    return Session(
        model=model,
        stream_fn=stream,
        tools=[_browser_tool(), _read_frame_tool()],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable instructions", "inventory", "env", "skills"],
        compaction_settings=CompactionSettings(),
    )


@pytest.mark.asyncio
async def test_stale_frames_flush_in_one_batch_at_the_ceiling(tmp_path):
    """Warm stale frames are carried until STALE_FLUSH_MAX_FRAMES, then ALL go
    in one batch: exactly one prefix divergence, and byte-stable again after.

    Without the ceiling the guard let a screenshot loop climb to 58 resident
    frames and get summarized mid-task (PR #3004 review round 1, M1).
    """
    from local_operator.compaction.pruning import (
        SESSION_KEEP_RECENT_FRAMES,
        STALE_FLUSH_MAX_FRAMES,
    )

    reads = [SCREENS[index % len(SCREENS)] for index in range(16)]
    stream = RecordingStream(reads)
    session = _session(tmp_path, stream)
    try:
        await session.prompt(BRIEF)
    finally:
        await session.dispose()

    broken = _divergences(stream.sent)
    assert len(broken) == 1, f"expected one batch flush, got divergences at {broken}"
    flush_at = broken[0]
    # The batch fired when the stale set reached the ceiling. The sweep runs
    # BEFORE the request that would carry the ceiling-th stale frame, so the
    # most ever sent is one stale frame short of it, plus the retained window.
    assert (
        max(stream.frames_sent[:flush_at])
        == STALE_FLUSH_MAX_FRAMES - 1 + SESSION_KEEP_RECENT_FRAMES
    )
    assert stream.frames_sent[flush_at] <= SESSION_KEEP_RECENT_FRAMES + 1
    # And the history is append-only again after the flush.
    assert all(
        later >= earlier
        for earlier, later in zip(stream.frames_sent[flush_at:], stream.frames_sent[flush_at + 1 :])
    )


@pytest.mark.asyncio
async def test_a_run_after_a_91_minute_gap_flushes_before_its_first_request(tmp_path, monkeypatch):
    """The user's wait between runs is judged before the next run sends
    anything: stale frames go then, while the cache is provably cold, not
    after the cold request (which would cost a second rewrite) and not never
    (PR #3004 review M2, QA Q2)."""
    import local_operator.session.session as session_module

    clock = {"offset": 0.0}
    real_time = session_module.time.time
    monkeypatch.setattr(session_module.time, "time", lambda: real_time() + clock["offset"])

    reads = [SCREENS[index % len(SCREENS)] for index in range(4)]
    stream = RecordingStream(reads)
    session = _session(tmp_path, stream)
    try:
        await session.prompt(BRIEF)
        first_run = len(stream.sent)
        assert stream.frames_sent[-1] == len(reads)  # warm: all carried
        assert _divergences(stream.sent) == []

        clock["offset"] += 91 * 60
        await session.prompt("Anything else?")
    finally:
        await session.dispose()

    # The first request of the second run already carries only the retained
    # frames: the flush happened before it was built.
    from local_operator.compaction.pruning import SESSION_KEEP_RECENT_FRAMES

    assert stream.frames_sent[first_run] == SESSION_KEEP_RECENT_FRAMES
    assert _divergences(stream.sent) == [first_run]


@pytest.mark.asyncio
async def test_a_short_gap_between_runs_keeps_the_prefix(tmp_path, monkeypatch):
    """Negative control for the pre-run sweep: a user who answers inside the
    idle window keeps the cached prefix byte-identical."""
    import local_operator.session.session as session_module

    clock = {"offset": 0.0}
    real_time = session_module.time.time
    monkeypatch.setattr(session_module.time, "time", lambda: real_time() + clock["offset"])

    reads = [SCREENS[index % len(SCREENS)] for index in range(4)]
    stream = RecordingStream(reads)
    session = _session(tmp_path, stream)
    try:
        await session.prompt(BRIEF)
        clock["offset"] += 4 * 60
        await session.prompt("Anything else?")
    finally:
        await session.dispose()
    assert _divergences(stream.sent) == []


@pytest.mark.asyncio
async def test_a_long_run_on_a_200k_model_stays_under_the_image_limit(tmp_path):
    """One long run of small distinct frames on a 200k-window model, with a
    small brief: every compaction trigger stays closed (140 frames is ~170k
    estimated tokens only near the end, and a few MB), so the boundary sweep
    inside ``_plan_compaction`` is never reached. The batch-flush ceiling must
    be judged at the tool boundary anyway, or requests climb past Anthropic's
    100-images-per-request limit (PR #3004 QA round 2, Q5: peak 112 images,
    the frame sweep reached in 2 of 142 boundaries).
    """
    paths = [f"/scratch/small-{index}.png" for index in range(140)]
    pixels = {path: base64.b64encode(f"png:{path}".encode() * 8).decode() for path in paths}
    _PIXELS.update(pixels)
    small_model = ModelSpec(
        provider="anthropic",
        model_id="claude-haiku-4-5",
        context_window=200_000,
        supports_images=True,
    )
    stream = RecordingStream(paths)
    session = _session(tmp_path, stream, model=small_model)
    try:
        await session.prompt("Audit these screens.")
    finally:
        await session.dispose()
        for path in paths:
            _PIXELS.pop(path, None)

    assert len(stream.frames_sent) == len(paths) + 2
    # 90 is the provider limit (100) less the margin; the frame ceiling holds
    # the run far below it (8 stale + 2 retained at most).
    assert max(stream.frames_sent) <= 90
    assert max(stream.frames_sent) <= 10
