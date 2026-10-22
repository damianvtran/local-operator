"""No blank text block on disk or on any wire — the #2460 hardening matrix.

#2461 stopped ``Message.user`` emitting ``TextContent(text="")`` and taught the
Anthropic and OpenAI wires to drop a blank block, which recovered sessions
whose transcripts already held one. A session still wedged on a 0.73.1
runtime (pre-#2461) with Anthropic's ``messages: text content blocks must be
non-empty`` after an image-only desktop send stored ``[{}, {attachment...}]``.
This module pins the gaps that remained on main after #2461, for every stored
shape such a row can take:

* PERSISTENCE: the transcript encoder normalises a user row, so no producer —
  ``Message.user`` or a hand-built ``Message`` — can write ``{}`` or a blank
  text block beside media, and the real ``Session.prompt`` path writes none;
* REHYDRATE: an already-poisoned row decodes without the blank block, so every
  transcript consumer is healed, not only the wire serializers;
* WIRES: Anthropic, OpenAI chat, OpenAI Responses and Google each render a
  stored row (decoded WITHOUT the transcript's healing, so the wire's own
  guard is what is tested) with no empty/whitespace text part, and a user turn
  left with nothing keeps its place via ``EMPTY_USER_TEXT`` rather than being
  skipped or shipped blank;
* ASIDE: wrapping an image-only question keeps every image.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.harness.types import (
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
)
from local_operator.providers.clients import (
    EMPTY_USER_TEXT,
    AnthropicClient,
    GoogleClient,
    OpenAICompatClient,
    _message_to_openai,
)
from local_operator.session.aside import wrap_aside_turns
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript, encode_message_payload

from .test_session import MODEL, ScriptedStream

IMAGE = {"data": "aGk=", "mime_type": "image/png"}
IMAGE_BLOCK = ImageContent(data="aGk=", mime_type="image/png")

#: Every shape a blank text block takes on a stored user row. ``{}`` is the
#: one the wedged session held: ``TextContent(text="")`` dumped with
#: ``exclude_defaults``.
WITH_IMAGE = {
    "keyless-block": [{}, IMAGE],
    "empty-text": [{"text": ""}, IMAGE],
    "whitespace-text": [{"text": "  \n"}, IMAGE],
    "typed-empty-text": [{"type": "text", "text": ""}, IMAGE],
}
ONLY_BLANK = {
    "only-keyless-block": [{}],
    "only-empty-text": [{"text": ""}],
    "only-whitespace-text": [{"text": " \t"}],
}
ALL_CASES = {**WITH_IMAGE, **ONLY_BLANK}


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    # The encoder externalises media through ``config_dir()``; keep it in the
    # test's own tree.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))


def _blank_text_blocks(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stored blocks that are text-shaped (no media key) and carry no text."""
    return [
        block
        for block in content
        if not ({"data", "attachment", "mime_type"} & block.keys())
        and not str(block.get("text", "")).strip()
    ]


def _history(content: list[dict[str, Any]]) -> list[Message]:
    """A stored row, decoded the way the WIRE used to see it: raw validate."""
    return [
        Message.model_validate({"role": "user", "content": content}),
        Message.assistant("ok"),
        Message.user("next"),
    ]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(WITH_IMAGE))
async def test_append_message_never_persists_a_blank_text_block_beside_media(tmp_path, name):
    """Any producer, not only ``Message.user``: a hand-built row with a blank
    block reaches the encoder unchanged, and the encoder is the last place the
    shape can be stopped before it is replayed on every later turn."""
    transcript = Transcript(tmp_path / "sess")
    message = Message.model_validate({"role": "user", "content": WITH_IMAGE[name]})
    await transcript.append_message(message)

    payload = json.loads(transcript.path.read_text().splitlines()[-1])["payload"]
    assert _blank_text_blocks(payload["content"]) == []
    assert len(payload["content"]) == 1, "the image itself must survive"


def test_encoder_keeps_a_whitespace_only_text_only_turn() -> None:
    """The one blank shape that is NOT normalised on disk: a text-only turn of
    whitespace is the whole of what the user sent. The wires drop and backfill
    it (below); the stored row stays a faithful record."""
    payload = encode_message_payload(Message.user(" "))
    assert payload["content"] == [{"text": " "}]
    only_empty = encode_message_payload(Message(role="user", content=[TextContent(text="")]))
    assert "content" not in only_empty, "an empty block goes even when it is the only block"


def test_encoder_leaves_assistant_rows_alone() -> None:
    """User rows only: an assistant row's emptiness is ``_is_empty_assistant``'s
    rule at the wire, and rewriting it here would change bytes nobody asked."""
    message = Message(role="assistant", content=[TextContent(text="")])
    assert encode_message_payload(message)["content"] == [{}]


@pytest.mark.asyncio
async def test_session_prompt_with_whitespace_and_an_image_persists_only_the_image(tmp_path):
    """The real send path: ``Session.prompt`` → ``Message.user`` → transcript."""
    stream = ScriptedStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )
    await session.prompt("  \n", [IMAGE_BLOCK])

    rows = [json.loads(line) for line in session.transcript.path.read_text().splitlines()]
    users = [
        row["payload"]
        for row in rows
        if row["type"] == "message" and row["payload"].get("role") == "user"
    ]
    assert len(users) == 1
    assert _blank_text_blocks(users[0]["content"]) == []
    assert len(users[0]["content"]) == 1
    sent = stream.requests[0].messages
    assert not any(
        isinstance(block, TextContent) and not block.text.strip()
        for message in sent
        for block in message.content
    ), "the request the provider was handed must not carry the blank block either"


# ---------------------------------------------------------------------------
# Rehydrate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(WITH_IMAGE))
def test_a_poisoned_row_already_on_disk_rehydrates_healed(tmp_path, name):
    """Rows written by a pre-guard build are still on disk; the transcript's
    row decoder drops their blank block so every consumer reads it healed."""
    directory = tmp_path / "sess"
    directory.mkdir()
    row = {
        "id": "row-1",
        "ts": 1.0,
        "type": "message",
        "payload": {"role": "user", "content": WITH_IMAGE[name]},
    }
    (directory / "transcript.jsonl").write_text(json.dumps(row) + "\n")

    (message,) = Transcript(directory).build_llm_history()
    assert isinstance(message, Message)
    assert [type(block).__name__ for block in message.content] == ["ImageContent"]
    assert message.id == "row-1"


# ---------------------------------------------------------------------------
# Wires: no blank text part, role order preserved
# ---------------------------------------------------------------------------


def _anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    request = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-opus-5-5"), messages=messages
    )
    return AnthropicClient()._build_body(request)["messages"]


def _openai_chat(messages: list[Message]) -> list[dict[str, Any]]:
    return [_message_to_openai(message) for message in messages]


def _openai_responses(messages: list[Message]) -> list[dict[str, Any]]:
    request = ChatRequest(model=ModelSpec(provider="openai", model_id="gpt-5"), messages=messages)
    body = OpenAICompatClient("https://api.openai.com/v1")._build_responses_body(request)
    return [item for item in body["input"] if item.get("role") in ("user", "assistant")]


def _google(messages: list[Message]) -> list[dict[str, Any]]:
    request = ChatRequest(
        model=ModelSpec(provider="google", model_id="gemini-3-pro"), messages=messages
    )
    return GoogleClient()._build_body(request)["contents"]


def _texts(wire: str, turn: dict[str, Any]) -> list[str]:
    """Every text a rendered turn carries, in that wire's own spelling."""
    if wire == "google":
        return [part["text"] for part in turn["parts"] if "text" in part]
    content = turn["content"]
    if isinstance(content, str):
        return [content]
    return [
        part["text"]
        for part in content
        if part.get("type") in ("text", "input_text", "output_text")
    ]


WIRES = {
    "anthropic": _anthropic,
    "openai-chat": _openai_chat,
    "openai-responses": _openai_responses,
    "google": _google,
}


@pytest.mark.parametrize("wire", sorted(WIRES))
@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_no_wire_ships_a_blank_text_part_from_a_stored_row(wire, name):
    turns = WIRES[wire](_history(ALL_CASES[name]))

    roles = [turn["role"] for turn in turns]
    assert roles == ["user", "model" if wire == "google" else "assistant", "user"], (
        "a blank user turn is backfilled, never skipped: skipping opens the body "
        "with the assistant role"
    )
    for turn in turns:
        for text in _texts(wire, turn):
            assert text.strip(), f"{wire} shipped a blank text part for {name}: {turn}"
    first = _texts(wire, turns[0])
    if name in ONLY_BLANK:
        assert first == [EMPTY_USER_TEXT]
    else:
        assert first == [], "an image-only turn carries its image and no invented text"


def test_google_keeps_real_text_beside_an_image() -> None:
    """The blank test is on the joined text: real words are still sent."""
    message = Message.model_validate(
        {"role": "user", "content": [{"text": "look"}, {"text": " "}, IMAGE]}
    )
    (turn,) = _google([message])
    assert turn["parts"][0] == {"text": "look "}


# ---------------------------------------------------------------------------
# Aside
# ---------------------------------------------------------------------------


def test_aside_wrap_keeps_every_image_of_an_image_only_question() -> None:
    """``content[1:]`` assumed a leading text block; since #2461 an image-only
    turn has none, and the slice dropped its first image."""
    first = ImageContent(data="Zmlyc3Q=", mime_type="image/png")
    second = ImageContent(data="c2Vjb25k", mime_type="image/png")
    (wrapped,) = wrap_aside_turns([Message.user("", images=[first, second])])
    assert wrapped.content[1:] == [first, second]
    assert isinstance(wrapped.content[0], TextContent)


def test_aside_wrap_still_replaces_the_leading_text() -> None:
    image = IMAGE_BLOCK
    (wrapped,) = wrap_aside_turns([Message.user("what is this", images=[image])])
    assert wrapped.content[1:] == [image]
    assert "what is this" in wrapped.text
    assert len([b for b in wrapped.content if isinstance(b, TextContent)]) == 1
