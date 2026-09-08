"""The bound every image entering a session over the wire passes through.

The phone portal is the one image seam that used to forward whatever it was
given. A camera roll produces 4032x3024 photos and a phone screenshot 2206x266,
and a provider refuses an image over 2000 pixels on its long edge as soon as a
request carries more than twenty of them — so an unbounded block was not
"lossless", it was a delayed fault that lands in the HISTORY and re-fires on
every later request, including the compaction meant to be the escape hatch.

The session render seam did repair oversize blocks afterwards, which is why
this was degraded rather than broken. What is asserted here is that the repair
has nothing left to do: the bound happens once, at entry.
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from local_operator.imaging import IMAGE_INGEST_MAX_EDGE
from local_operator.media import sniff_image
from local_operator.session.runtime.server import image_blocks


def _wire_image(size: tuple[int, int]) -> dict[str, str]:
    """A wire entry carrying a real PNG of ``size``, the shape the phone sends."""
    import os

    # Noise rather than a flat fill: a flat image compresses to a few KB at any
    # dimension and reads as line art, which takes the ladder's bilevel
    # exemption and would not exercise the ingest bound at all.
    buffer = io.BytesIO()
    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(buffer, format="PNG")
    return {
        "data_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mime_type": "image/png",
    }


def test_mobile_images_are_bounded_on_entry() -> None:
    """A phone screenshot arrives at the ingest edge, not at its native size.

    2206x266 is the real shape that motivated this: comfortably under the 2000
    px refusal line on its SHORT edge and over it on its long one, so it sat
    harmlessly in a history for a hundred turns and then wedged the session the
    moment the twenty-first image arrived.
    """
    blocks = image_blocks([_wire_image((2206, 266))])

    assert len(blocks) == 1
    payload = base64.b64decode(blocks[0].data)
    info = sniff_image(payload)
    assert info is not None
    assert info.width is not None and info.height is not None
    assert max(info.width, info.height) <= IMAGE_INGEST_MAX_EDGE
    # The declared mime must describe the bytes: the ladder is free to switch
    # to JPEG, and a stale "image/png" label would be a 400 at the provider.
    assert blocks[0].mime_type == info.mime_type


def test_a_small_image_keeps_its_bytes() -> None:
    """Inside the bound means untouched — no gratuitous PNG round-trip.

    Re-encoding an in-bounds image routinely makes it BIGGER, which is why the
    ladder's first rung is verbatim; asserting it here keeps the mobile seam on
    that rung rather than quietly paying for every small paste.
    """
    entry = _wire_image((320, 200))

    blocks = image_blocks([entry])

    assert len(blocks) == 1
    assert blocks[0].data == entry["data_b64"]


def test_undecodable_mobile_image_is_dropped_not_fatal() -> None:
    """One bad image costs that image, never the whole prompt.

    The pre-existing contract of this helper ("bad entries are dropped") has to
    survive the bound, which introduces new ways to fail: bytes that are not
    base64, and base64 that is not an image.
    """
    good = _wire_image((320, 200))

    blocks = image_blocks(
        [
            {"data_b64": "not base64 at all!!", "mime_type": "image/png"},
            {"data_b64": base64.b64encode(b"not an image").decode("ascii")},
            good,
        ]
    )

    assert len(blocks) == 1
    assert blocks[0].data == good["data_b64"]


def test_no_images_is_an_empty_list() -> None:
    """``None`` and ``[]`` both mean "no images", which _submit_prompt relies on."""
    assert image_blocks(None) == []
    assert image_blocks([]) == []
