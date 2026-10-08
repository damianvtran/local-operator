"""``ask-attachments-v1``: the owner's half of image answers.

WHY THESE CELLS EXIST. An answer is TERMINAL, so an image that an owner silently
ignores is a loss the user can neither see nor retry. Three owner-side rules keep
that from happening, and each is pinned here against the real dispatch:

* the capability is advertised only by a handle whose ``ask_respond`` literally
  takes ``attachments`` (a ``**kwargs``-only acceptor would swallow the images);
* a frame that carries images is decoded per question and REFUSED -- never
  dropped -- when any one fails to decode, names a question the answer does not
  have, rides a revision, or reaches a handle that cannot keep it;
* a text-only frame calls the handle with the exact keyword set it always did.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest

from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.types import (
    ASK_ATTACHMENTS_CAPABILITY,
    ASK_ATTACHMENTS_UNSUPPORTED,
)
from tests.unit.session.runtime.test_server import FakeHandle


def _png_b64() -> str:
    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.new("RGB", (8, 8), (200, 30, 60)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class KeepingHandle(FakeHandle):
    """The production shape: ``ask_respond`` takes the ``attachments`` keyword."""

    def __init__(self) -> None:
        super().__init__()
        self.ask_calls: list[dict[str, Any]] = []

    async def ask_respond(self, ask_id, answers, by="", attachments=None) -> str:  # noqa: ANN001
        self.ask_calls.append(
            {"ask_id": ask_id, "answers": answers, "by": by, "attachments": attachments}
        )
        return "answered"

    async def ask_revise(self, ask_id, answers, by="") -> str:  # noqa: ANN001
        self.ask_calls.append({"op": "revise", "ask_id": ask_id})
        return "revised"


class LegacyHandle(FakeHandle):
    """An owner built before the keyword: it would IGNORE images if handed any."""

    def __init__(self) -> None:
        super().__init__()
        self.ask_calls: list[tuple[Any, ...]] = []

    async def ask_respond(self, ask_id, answers, by="") -> str:  # noqa: ANN001
        self.ask_calls.append((ask_id, answers, by))
        return "answered"


class KwargsHandle(FakeHandle):
    """A ``**kwargs`` acceptor: it would swallow the images without keeping them."""

    async def ask_respond(self, ask_id, answers, **kwargs) -> str:  # noqa: ANN001
        return "answered"


def _server(handle: FakeHandle) -> RuntimeServer:
    return RuntimeServer(handle, kind="tui")


def _frame(op: str = "ask_respond", **extra: Any) -> dict[str, Any]:
    """A whole control frame: ``_dispatch`` validates it, so ``op`` rides inside it."""
    return {"op": op, "ask_id": "a-1", "answers": {"q1": ["see this"]}, "by": "desktop", **extra}


def _image(question_id: str = "q1", data: str | None = None) -> dict[str, str]:
    return {"question_id": question_id, "data_b64": data or _png_b64(), "mime_type": "image/png"}


# --- the advertisement -------------------------------------------------------


def test_the_token_is_advertised_by_a_handle_that_takes_the_keyword() -> None:
    assert ASK_ATTACHMENTS_CAPABILITY in _server(KeepingHandle())._record.capabilities


@pytest.mark.parametrize("handle_class", [LegacyHandle, KwargsHandle, FakeHandle])
def test_the_token_is_absent_when_the_handle_would_not_keep_the_images(handle_class) -> None:
    """Absent for a legacy signature, a ``**kwargs`` swallower, and no ask op at all."""
    assert ASK_ATTACHMENTS_CAPABILITY not in _server(handle_class())._record.capabilities


# --- the dispatch ------------------------------------------------------------


@pytest.mark.asyncio
async def test_images_are_decoded_per_question_and_handed_over_as_attachments() -> None:
    handle = KeepingHandle()
    server = _server(handle)

    detail = await server._dispatch("ask_respond", _frame(images=[_image(), _image()]))

    assert detail == "answered"
    (call,) = handle.ask_calls
    blocks = call["attachments"]["q1"]
    assert len(blocks) == 2
    assert all(block.type == "image" and block.data for block in blocks)
    # The wire's ``images`` key is consumed here, not forwarded as a bare kwarg.
    assert call["answers"] == {"q1": ["see this"]}


@pytest.mark.asyncio
async def test_a_text_only_frame_calls_the_handle_exactly_as_it_always_did() -> None:
    """OLD UI/relay -> NEW owner: no ``attachments`` keyword, even to a handle that takes it."""
    handle = KeepingHandle()
    server = _server(handle)

    await server._dispatch("ask_respond", _frame())

    assert handle.ask_calls[0]["attachments"] is None

    legacy = LegacyHandle()
    await _server(legacy)._dispatch("ask_respond", _frame())
    assert legacy.ask_calls == [("a-1", {"q1": ["see this"]}, "desktop")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "images, fragment",
    [
        ([_image(data="bm90LWFuLWltYWdl")], "could not be read"),
        (
            [{"question_id": "q1", "data_b64": "!!!not base64!!!", "mime_type": "image/png"}],
            "could not be read",
        ),
        ([_image(), _image(data="bm90LWFuLWltYWdl")], "could not be read"),  # one bad of two
        ([_image("q9")], "must name a question"),
        (["a string"], "list of image objects"),
    ],
)
async def test_an_image_that_fails_to_decode_refuses_the_answer_instead_of_dropping(
    images, fragment: str
) -> None:
    """The opposite of a prompt: ``image_blocks`` DROPS a bad entry, an answer cannot."""
    handle = KeepingHandle()
    server = _server(handle)

    with pytest.raises(ValueError, match=fragment):
        await server._dispatch("ask_respond", _frame(images=images))

    assert handle.ask_calls == [], "a refused answer must not reach the handle (the ask stays open)"


@pytest.mark.asyncio
async def test_a_handle_that_cannot_keep_images_refuses_in_the_authored_sentence() -> None:
    legacy = LegacyHandle()

    with pytest.raises(ValueError) as refused:
        await _server(legacy)._dispatch("ask_respond", _frame(images=[_image()]))

    assert str(refused.value) == ASK_ATTACHMENTS_UNSUPPORTED
    assert legacy.ask_calls == []


@pytest.mark.asyncio
async def test_a_revision_frame_with_images_is_refused_at_the_owner_too() -> None:
    """D5 holds even for a sender that skipped its own check."""
    handle = KeepingHandle()

    with pytest.raises(ValueError, match="does not carry images"):
        await _server(handle)._dispatch("ask_revise", _frame("ask_revise", images=[_image()]))
    assert handle.ask_calls == []

    # ...and the decoder's own guard, for a caller that reaches it past the validator.
    from local_operator.session.runtime.server import _answer_attachment_kwargs

    with pytest.raises(ValueError, match="text only"):
        await _answer_attachment_kwargs("ask_revise", handle.ask_revise, [_image()], {"q1": ["x"]})
