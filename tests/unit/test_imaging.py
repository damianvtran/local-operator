"""The bound every image block passes through, tested as a shared contract.

The ladder's rungs are exercised in depth through the ``read`` tool
(``tests/unit/tools/test_builtin_tools.py``) and through the composer
(``tests/unit/tui/test_paste_images.py``), which is where they were written and
where the interesting fixtures live. What is tested HERE is the property those
two callers depend on and neither can state alone: that a bounded image is
inside the provider's limits, whatever it looked like on the way in.

That property is the whole reason this module exists. It used to hold on the
``read`` path and not on the composer path, and the resulting 400 does not
merely fail a turn — the block is in the conversation HISTORY, so every later
request re-sends it and fails identically, including the compaction that is
supposed to be the escape hatch.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from local_operator.imaging import (
    IMAGE_MAX_EDGE,
    IMAGE_MAX_PIXELS,
    bound_image_for_model,
)
from local_operator.media import ImageInfo, sniff_image

#: The stricter per-image dimension limit a provider applies once a request
#: carries more than twenty images. NOT imported from the module under test:
#: this is the provider's number, and writing it out is what makes the
#: assertions below a statement about the API rather than a tautology about our
#: own constant.
MANY_IMAGE_PIXEL_LIMIT = 2000


def _png(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 60, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


def _sniffed(data: bytes) -> ImageInfo:
    """``sniff_image`` with its ``None`` asserted away.

    Every caller in this file sniffs bytes it just wrote, so ``None`` would be a
    broken fixture rather than a case under test — and asserting it here keeps
    that failure legible instead of surfacing as an attribute error inside the
    function being tested.
    """
    info = sniff_image(data)
    assert info is not None, "fixture bytes were not recognised as an image"
    return info


@pytest.mark.parametrize(
    ("size", "label"),
    [
        ((2206, 266), "the paste that wedged a real session"),
        ((2560, 1440), "a retina screenshot"),
        ((3456, 2234), "a native-resolution laptop capture"),
        ((6016, 3384), "a 6K display"),
        ((1600, 1000), "just over on one edge only"),
        ((800, 600), "already inside the bounds"),
        ((1, 4000), "a degenerate sliver"),
    ],
)
def test_a_bounded_image_is_inside_the_many_image_limit(size, label: str) -> None:
    """The one invariant every caller relies on.

    An image over 2000 pixels on its long edge is accepted right up until the
    request carries its twenty-first image, and is then refused forever. So
    "small enough" cannot be judged per image or per turn — it has to hold for
    every image that could ever be attached, which is what this pins.
    """
    source = _png(size)
    payload, _mime, _summary = bound_image_for_model(source, _sniffed(source))
    width, height = Image.open(io.BytesIO(payload)).size
    assert max(width, height) <= IMAGE_MAX_EDGE
    assert max(width, height) < MANY_IMAGE_PIXEL_LIMIT, label


def test_the_bound_preserves_aspect_ratio() -> None:
    """A squashed screenshot is an unreadable screenshot, and the model has no
    way to know it was distorted."""
    source = _png((2206, 266))
    payload, _mime, _summary = bound_image_for_model(source, _sniffed(source))
    width, height = Image.open(io.BytesIO(payload)).size
    assert width / height == pytest.approx(2206 / 266, rel=0.01)


def test_an_image_already_inside_the_bounds_is_returned_verbatim() -> None:
    """No re-encode can improve an image the model sees at its original size,
    and PNG round-tripping routinely makes files BIGGER. The common case must
    stay lossless and free."""
    source = _png((800, 600))
    payload, mime, _summary = bound_image_for_model(source, _sniffed(source))
    assert payload == source
    assert mime == "image/png"


def test_the_summary_names_the_source_whenever_the_bytes_changed() -> None:
    """The caption is the model's only evidence of what it is looking at. When
    the delivered image is not what is on disk, a model comparing this against
    ``ls -l`` or a later re-read has no way to reconcile the two unless the
    summary says so."""
    source = _png((2560, 1440))
    _payload, _mime, summary = bound_image_for_model(source, _sniffed(source))
    assert "1568x882" in summary
    assert "source 2560x1440" in summary


def test_a_decompression_bomb_is_refused_before_it_is_decoded() -> None:
    """A bomb is small on disk by construction, so no byte cap can see it
    coming. The refusal has to come from the header dimensions, BEFORE the
    decode allocates ~4 bytes per pixel."""
    # A forged IHDR rather than a real file: the point is that nothing decodes
    # it, so writing 3.6 GB of RGBA to build the fixture would defeat the test.
    seed = _png((8, 8))
    header = bytearray(seed)
    header[16:20] = (30000).to_bytes(4, "big")
    header[20:24] = (30000).to_bytes(4, "big")
    info = sniff_image(bytes(header))
    assert info is not None and info.width == 30000

    with pytest.raises(ValueError) as excinfo:
        bound_image_for_model(bytes(header), info)
    assert f"{IMAGE_MAX_PIXELS:,}" in str(excinfo.value)


def test_undecodable_bytes_raise_rather_than_becoming_an_image_block() -> None:
    """Forwarding bytes that will not decode earns a 400 that no retry clears,
    because by then the bad block is already in the transcript. Raising here is
    what keeps that out of the history."""
    truncated = _png((100, 100))[:60]
    # The header still parses; only the body is gone.
    with pytest.raises(ValueError):
        bound_image_for_model(truncated, _sniffed(truncated))
