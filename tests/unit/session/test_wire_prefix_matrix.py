"""Wire-prefix matrix for the shared-image-compaction contract.

The acceptance proof at the WIRE itself -- the bytes the provider encoder
builds, with no Message.id normalization anywhere:

- old->old: an identical request encodes byte-identically; deliberate text,
  image, system and field mutations each change the bytes (so the measurement
  is not vacuous);
- old->new within one epoch: consecutive requests share the settled prefix
  through the previous turn's last message and differ only by the appended
  content;
- new epoch: the elided capture is absent from the wire and the latest one
  present, and the next request keeps the new prefix settled.

Encoded with the real ``OpenAICompatClient._build_body`` under the two
counter-shapes the campaign runs (OpenRouter and direct DeepSeek), serialized
deterministically to bytes for prefix math. A byte-prefix equality is a wire
property only -- it is not a provider cache-hit assertion.
"""

from __future__ import annotations

import base64
import json

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    CapturedFrame,
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.providers.clients import OpenAICompatClient
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

OR_BASE = "https://openrouter.ai/api/v1"
DS_BASE = "https://api.deepseek.com/v1"
OR_MODEL = ModelSpec(
    provider="openrouter", model_id="deepseek/deepseek-v4.1-flash", context_window=1_000_000
)
DS_MODEL = ModelSpec(provider="deepseek", model_id="deepseek-chat", context_window=1_000_000)

#: Keeps the capture turns and the doc image inside the kept window.
KEEP_RECENT = 3_000

#: Canonical base64, like every capture fixture: the attachment pipeline
#: decodes and re-encodes, so only a canonical payload matches itself.
#: DISTINCT payloads per revision: the wire carries image bytes only -- the
#: capture metadata is bookkeeping and encoders never emit it -- so
#: "the superseded capture left the wire" is only measurable with bytes that
#: differ per revision.
_CAPTURE_DATA = {
    "1": base64.b64encode(b"old-capture!" * 9_000).decode(),
    "2": base64.b64encode(b"new-capture!" * 9_000).decode(),
    "9": base64.b64encode(b"mut-capture!" * 9_000).decode(),
}
_DOC_DATA = base64.b64encode(b"doc?" * 12_000).decode()

SHAPES = pytest.mark.parametrize(
    ("model", "base"),
    [(OR_MODEL, OR_BASE), (DS_MODEL, DS_BASE)],
    ids=["openrouter", "deepseek"],
)


class ScriptedStream:
    """Replays one reply per call and records every request verbatim."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    def __call__(self, request, signal):
        self.requests.append(request)
        index = len(self.requests) - 1
        reply = self.replies[index] if index < len(self.replies) else "reply"

        async def gen():
            yield StreamTextDelta(delta=reply)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream, model) -> Session:
    return Session(
        model=model,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable instructions"],
        compaction_settings=CompactionSettings(keep_recent_tokens=KEEP_RECENT),
    )


def _capture(*, revision: str) -> ImageContent:
    return ImageContent(
        data=_CAPTURE_DATA[revision],
        mime_type="image/png",
        captured_frame=CapturedFrame(
            stream_id="screen", revision_id=revision, producer_scope="test:wire-prefix"
        ),
    )


def _plain() -> ImageContent:
    return ImageContent(data=_DOC_DATA, mime_type="image/png")


def _encode(request: ChatRequest, base: str) -> bytes:
    """The exact built payload as deterministic bytes.

    ``_build_body`` is the real encoder; this serialization is the byte form
    its JSON is compared in (compact separators, no key reordering), so the
    prefix math below runs on the provider-visible fields exactly as built.
    """
    body = OpenAICompatClient(base)._build_body(request)
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()


def _lcp(first: bytes, second: bytes) -> int:
    limit = min(len(first), len(second))
    index = 0
    while index < limit and first[index] == second[index]:
        index += 1
    return index


def _escaped(text: str) -> bytes:
    """The bytes a JSON body carries for ``text`` (without the quotes)."""
    return json.dumps(text)[1:-1].encode()


async def _talk(session: Session) -> None:
    await session.prompt("filler " + "detail " * 300)
    await session.prompt("filler two " + "detail " * 300)
    await session.prompt("older view", images=[_capture(revision="1")])
    await session.prompt("newer view", images=[_capture(revision="2")])
    await session.prompt("doc", images=[_plain()])


@SHAPES
@pytest.mark.asyncio
async def test_old_to_old_encodes_byte_identically_and_mutations_change_the_prefix(
    tmp_path, model, base
):
    """The measurement is deterministic and not vacuous."""
    stream = ScriptedStream(["reply"] * 4)
    session = make_session(tmp_path, stream, model)
    await session.prompt("hello", images=[_capture(revision="1")])
    request = stream.requests[0]

    first = _encode(request, base)
    assert first == _encode(request, base), "an identical request must encode byte-identically"

    mutated_text = request.model_copy(deep=True)
    mutated_text.messages[-1] = Message.user("hello, changed")
    assert _encode(mutated_text, base) != first, "a text mutation must change the bytes"

    mutated_image = request.model_copy(deep=True)
    mutated_image.messages[-1] = Message.user("hello", images=[_capture(revision="9")])
    assert _encode(mutated_image, base) != first, "an image mutation must change the bytes"

    mutated_system = request.model_copy(deep=True)
    mutated_system.system_blocks = [*request.system_blocks, "an extra rule"]
    assert _encode(mutated_system, base) != first, "a system mutation must change the bytes"

    # The session runs with no tools, so the schema-shaped control is a body
    # field instead (a tools mutation would need an AgentTool fixture; the
    # body still carries this one provider-visible).
    mutated_field = request.model_copy(deep=True)
    mutated_field.max_tokens = 123
    assert _encode(mutated_field, base) != first, "a field mutation must change the bytes"

    await session.dispose()


@SHAPES
@pytest.mark.asyncio
async def test_encoders_never_emit_capture_bookkeeping(tmp_path, model, base):
    """The metadata is harness bookkeeping; the wire carries image bytes only.

    The transport contract the protection rules rest on: a producer can
    replace only its own earlier capture claims, and no reader could be
    misled by a provider-visible copy of the scope/revision fields. Pinned
    against the real encoder under both counter-shapes, beside the positive
    control that the image itself is still a normal image block.
    """
    stream = ScriptedStream(["reply"] * 4)
    session = make_session(tmp_path, stream, model)
    await session.prompt("look", images=[_capture(revision="1")])

    encoded = _encode(stream.requests[-1], base)

    assert b"captured_frame" not in encoded
    assert b"producer_scope" not in encoded
    assert _CAPTURE_DATA["1"].encode() in encoded

    await session.dispose()


@SHAPES
@pytest.mark.asyncio
async def test_consecutive_requests_share_the_settled_prefix_and_differ_by_the_turn(
    tmp_path, model, base
):
    """One epoch, two turns: the prefix through turn one is byte-stable."""
    stream = ScriptedStream(["reply"] * 6)
    session = make_session(tmp_path, stream, model)
    await session.prompt("first turn " + "detail " * 60)
    first_request = stream.requests[-1]
    await session.prompt("second turn " + "detail " * 60)
    second_request = stream.requests[-1]

    first = _encode(first_request, base)
    second = _encode(second_request, base)
    assert first != second

    frontier = first.rfind(_escaped("first turn ")) + len(_escaped("first turn "))
    assert frontier > 0, "the anchor must exist in the built body"
    shared = _lcp(first, second)
    assert shared >= frontier, "the settled prefix through the previous turn must not move"
    # The differing suffix carries the appended turn and NOT a rewrite of the
    # previously settled text.
    assert _escaped("second turn ") in second[shared:], "the appended turn must be the change"
    assert _escaped("first turn ") not in second[shared:], "settled content was rewritten"

    await session.dispose()


@SHAPES
@pytest.mark.asyncio
async def test_an_epoch_elides_on_the_wire_and_the_new_prefix_stays_settled(tmp_path, model, base):
    """The epoch's wire change, then stability of the new prefix afterwards."""
    stream = ScriptedStream(["reply"] * 12)
    session = make_session(tmp_path, stream, model)
    await _talk(session)
    before = _encode(stream.requests[-1], base)

    # Manual pass, so no mid-talk auto-fire; the small trigger only defines
    # the elision band (0.8 x 4000), far under the capture payloads.
    session._compaction_settings = CompactionSettings(
        keep_recent_tokens=KEEP_RECENT, wire_bytes_trigger=4_000
    )
    outcome = await session.compact_now()
    assert outcome.ran is True
    # Restore: the later prompts must not fire another pass (the wire is far
    # over the small trigger by construction of this fixture).
    session._compaction_settings = CompactionSettings(keep_recent_tokens=KEEP_RECENT)

    await session.prompt("after the epoch")
    epoch = _encode(stream.requests[-1], base)

    assert epoch != before, "the epoch must move the wire"
    assert _CAPTURE_DATA["1"].encode() not in epoch, "the superseded capture is still on the wire"
    assert _CAPTURE_DATA["2"].encode() in epoch, "the latest capture must remain on the wire"
    assert _DOC_DATA.encode() in epoch, "the unannotated doc image must survive"
    shared = _lcp(before, epoch)
    assert 0 < shared < len(before), "the change must be a real region of the payload"

    await session.prompt("steady turn")
    following = _encode(stream.requests[-1], base)
    anchor = epoch.rfind(_escaped("after the epoch")) + len(_escaped("after the epoch"))
    assert anchor > 0
    assert _lcp(epoch, following) >= anchor, "the new settled prefix must stay stable"

    await session.dispose()
