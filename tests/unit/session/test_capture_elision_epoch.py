"""Live-side capture elision: superseded captures leave the projection; resume matches.

The shared-image-compaction contract: at a REAL compaction epoch (and only
there), captures carrying harness provenance whose stream has a newer capture
leave the SENT projection; user uploads, unknown images and every stream's
latest frame never do. The count is recorded on the pass payload so
``build_llm_history`` re-applies exactly the same frontier -- live and resumed
contexts stay equal, the property the preserved-turn machinery is already
pinned by. Real compaction throughout (no stubbed ``compaction.api``),
following the harness pattern of ``test_compaction_preserves_user_turns.py``.
"""

from __future__ import annotations

import base64

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.compaction.marker import (
    capture_elision_frontier,
    capture_elision_identities,
)
from local_operator.harness.types import (
    CapturedFrame,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

VISION_MODEL = ModelSpec(provider="test", model_id="sees", context_window=100_000)

#: Keeps the capture turns and the doc image inside the kept window (the
#: backwards walk prices each image at ``IMAGE_TOKEN_ESTIMATE`` = 1200) while
#: leaving the filler turns on the summarized side.
KEEP_RECENT = 3_000

#: Canonical base64 (the attachment pipeline decodes and re-encodes images,
#: so an invalid payload like a repeated "aGVsbG8=" would be normalised and no
#: longer match itself on resume). Distinct decoded bytes let the doc image be
#: found by data on resume, where snapcompact archive frames also render as
#: (unmarked) images.
_CAPTURE_DATA = base64.b64encode(b"cap!" * 12_000).decode()
_DOC_DATA = base64.b64encode(b"doc?" * 12_000).decode()


class ScriptedStream:
    """Replays one reply per call; summaries get the same inert string."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[object] = []

    def __call__(self, request, signal):
        self.requests.append(request)
        index = len(self.requests) - 1
        reply = self.replies[index] if index < len(self.replies) else "reply"

        async def gen():
            yield StreamTextDelta(delta=reply)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream) -> Session:
    return Session(
        model=VISION_MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=KEEP_RECENT),
    )


def _capture(*, revision: str) -> ImageContent:
    return ImageContent(
        data=_CAPTURE_DATA,
        mime_type="image/png",
        captured_frame=CapturedFrame(
            stream_id="screen", revision_id=revision, producer_scope="mcp:sess:cam"
        ),
    )


def _plain() -> ImageContent:
    return ImageContent(data=_DOC_DATA, mime_type="image/png")


def _stream_capture(*, stream: str, revision: str) -> ImageContent:
    """A capture on a named stream with revision-DISTINCT bytes (see above)."""
    return ImageContent(
        data=base64.b64encode(f"{stream}-{revision}!".encode() * 9_000).decode(),
        mime_type="image/png",
        captured_frame=CapturedFrame(
            stream_id=stream, revision_id=revision, producer_scope="mcp:sess:cam"
        ),
    )


def _pairs(messages) -> list[tuple[str, str]]:
    return sorted(
        (image.captured_frame.stream_id, image.captured_frame.revision_id)
        for image in _images(messages)
        if image.captured_frame
    )


def _images(messages) -> list[ImageContent]:
    return [
        block
        for message in messages
        if isinstance(message, Message)
        for block in message.content
        if isinstance(block, ImageContent)
    ]


def _marked(messages) -> list[str]:
    return sorted(
        image.captured_frame.revision_id for image in _images(messages) if image.captured_frame
    )


def _has_doc(messages) -> bool:
    return any(
        image.captured_frame is None and image.data == _DOC_DATA for image in _images(messages)
    )


async def _talk(session: Session) -> None:
    """Filler first (summarized side), then the captures and doc (kept side).

    Built under the DEFAULT trigger so no pass fires mid-talk; the tests lower
    the trigger only AFTER, right before the manual pass under test.
    """
    await session.prompt("filler " + "detail " * 300)
    await session.prompt("filler two " + "detail " * 300)
    await session.prompt("older view", images=[_capture(revision="1")])
    await session.prompt("newer view", images=[_capture(revision="2")])
    await session.prompt("doc", images=[_plain()])


@pytest.mark.asyncio
async def test_an_epoch_elides_the_superseded_capture_and_resume_matches(tmp_path):
    """The money property: live projection and resumed history are the same."""
    session = make_session(tmp_path, ScriptedStream(["reply"] * 12))
    await _talk(session)
    # The pass under test is the manual one; the small trigger only defines
    # the rung's band (0.8 x 4000), far under the capture payloads.
    session._compaction_settings = CompactionSettings(
        keep_recent_tokens=KEEP_RECENT, wire_bytes_trigger=4_000
    )

    outcome = await session.compact_now()
    assert outcome.ran is True

    live = _images(session._context.messages)
    assert _marked(session._context.messages) == ["2"], "the superseded capture stays elided"
    assert len(live) == 2  # the unannotated doc image survives beside the latest capture
    assert _has_doc(session._context.messages)

    record = session._transcript.latest_entry("compaction")
    assert record is not None
    assert capture_elision_frontier(record.payload) == 1
    assert capture_elision_identities(record.payload) == (("mcp:sess:cam", "screen", "1"),)

    directory = session._transcript.directory
    await session.dispose()
    replayed = Transcript(directory).build_llm_history()
    assert _marked(replayed) == ["2"], "resume must re-apply exactly the recorded frontier"
    assert _has_doc(replayed)


@pytest.mark.asyncio
async def test_resume_re_elides_the_recorded_capture_after_a_retro_supersede(tmp_path):
    """The reviewer's C1 case, end to end: a later turn supersedes a capture
    the pass protected.

    Pass time: B's first capture was B's LATEST (protected, kept); A's first
    was superseded by A's second (the only eligible elision). A later turn
    adds B's second capture, retro-superseding the block the pass protected --
    which now PRECEDES A's first in canonical arrival order, so a
    count-re-derived replay would elide the wrong block. Resume must re-elide
    exactly the recorded address (A/1) and keep B/1.
    """
    session = make_session(tmp_path, ScriptedStream(["reply"] * 12))
    await session.prompt("start " + "detail " * 300)
    await session.prompt("b one", images=[_stream_capture(stream="b", revision="1")])
    await session.prompt("a one", images=[_stream_capture(stream="a", revision="1")])
    await session.prompt("a two", images=[_stream_capture(stream="a", revision="2")])
    session._compaction_settings = CompactionSettings(
        keep_recent_tokens=KEEP_RECENT, wire_bytes_trigger=4_000
    )
    outcome = await session.compact_now()
    assert outcome.ran is True
    session._compaction_settings = CompactionSettings(keep_recent_tokens=KEEP_RECENT)

    record = session._transcript.latest_entry("compaction")
    assert record is not None
    assert capture_elision_identities(record.payload) == (("mcp:sess:cam", "a", "1"),)
    assert _pairs(session._context.messages) == [("a", "2"), ("b", "1")]

    await session.prompt("b two", images=[_stream_capture(stream="b", revision="2")])

    directory = session._transcript.directory
    await session.dispose()
    replayed = Transcript(directory).build_llm_history()
    assert _pairs(replayed) == [
        ("a", "2"),
        ("b", "1"),
        ("b", "2"),
    ], "resume must re-elide exactly the recorded address and keep the retro-superseded block"


@pytest.mark.asyncio
async def test_a_pass_without_byte_pressure_never_elides(tmp_path):
    """Default (16 MB) trigger: no elision, no frontier key -- old behaviour."""
    session = make_session(tmp_path, ScriptedStream(["reply"] * 12))
    await _talk(session)

    outcome = await session.compact_now()
    assert outcome.ran is True

    assert _marked(session._context.messages) == ["1", "2"]
    assert len(_images(session._context.messages)) == 3

    record = session._transcript.latest_entry("compaction")
    assert record is not None
    assert capture_elision_frontier(record.payload) is None, "no pressure -> no elision recorded"

    directory = session._transcript.directory
    await session.dispose()
    replayed = Transcript(directory).build_llm_history()
    assert _marked(replayed) == ["1", "2"]
    assert _has_doc(replayed)


@pytest.mark.asyncio
async def test_a_disabled_byte_trigger_keeps_token_only_behaviour(tmp_path):
    """Trigger 0 disables the rung entirely, as it disables every byte gate."""
    session = make_session(tmp_path, ScriptedStream(["reply"] * 12))
    await _talk(session)
    session._compaction_settings = CompactionSettings(
        keep_recent_tokens=KEEP_RECENT, wire_bytes_trigger=0
    )

    outcome = await session.compact_now()
    assert outcome.ran is True

    assert _marked(session._context.messages) == ["1", "2"]
    assert len(_images(session._context.messages)) == 3
    record = session._transcript.latest_entry("compaction")
    assert record is not None
    assert capture_elision_frontier(record.payload) is None

    await session.dispose()
