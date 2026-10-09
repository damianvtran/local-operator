"""Output artifacts resolve to transcript images at the TUI's mount adapter.

``tool_result_image_blocks`` is the seam between the output-attachment
contract and the transcript's ``ImageBlock`` mount: inline images pass
through, artifact images resolve from the content-addressed store, everything
else is skipped so a video cannot be drawn as a picture. These tests pin the
adapter without booting the app — the rendered evidence belongs to the PR's
end-to-end run, not to a unit suite.
"""

from __future__ import annotations

import base64

import pytest

from local_operator.harness.types import AttachmentContent, ImageContent
from local_operator.session.attachments import cache_media
from local_operator.tui.session_presentation import tool_result_image_blocks


@pytest.fixture(autouse=True)
def _isolate_attachments(tmp_path, monkeypatch):
    store_root = tmp_path / "attachments"
    store_root.mkdir()
    monkeypatch.setattr("local_operator.session.attachments.attachments_dir", lambda: store_root)
    return store_root


#: A 73-byte 1x1 PNG — real bytes, so resolution is exercised end to end.
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd4"
    "0000000049454e44ae426082"
)


def test_inline_images_pass_through_untouched():
    inline = ImageContent(data="BBBB", mime_type="image/jpeg")
    assert tool_result_image_blocks([inline]) == [inline]
    assert tool_result_image_blocks(None) == []


def test_artifact_image_resolves_bytes_from_the_store():
    block = cache_media(PNG_1X1, "image/png")
    assert block is not None

    resolved = tool_result_image_blocks([block])

    assert len(resolved) == 1
    assert base64.b64decode(resolved[0].data) == PNG_1X1
    assert resolved[0].mime_type == "image/png"


def test_artifact_with_missing_bytes_mounts_as_unavailable():
    """A digest the store no longer holds must still mount — as the
    ImageBlock's unavailable receipt (empty data), never an exception that
    would take the tool row down with it."""
    block = AttachmentContent(kind="image", content_type="image/png", attachment="0" * 32)
    resolved = tool_result_image_blocks([block])
    assert len(resolved) == 1
    assert resolved[0].data == ""


def test_non_image_artifacts_contribute_nothing():
    """Video/audio have no transcript mount yet; drawing their bytes as an
    image would be the exact class of lie the unavailable receipt exists to
    avoid."""
    video = AttachmentContent(kind="video", content_type="video/mp4", attachment="a" * 32)
    audio = AttachmentContent(kind="audio", content_type="audio/wav", attachment="b" * 32)
    assert tool_result_image_blocks([video, audio]) == []


def test_order_is_content_order():
    artifact = cache_media(PNG_1X1, "image/png")
    assert artifact is not None
    inline = ImageContent(data="AAAA", mime_type="image/jpeg")

    resolved = tool_result_image_blocks([inline, artifact])

    assert len(resolved) == 2
    assert resolved[0].data == "AAAA"
    assert base64.b64decode(resolved[1].data) == PNG_1X1
