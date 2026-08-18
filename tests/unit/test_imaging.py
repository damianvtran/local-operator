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


def _oriented_jpeg(size: tuple[int, int], orientation: int | None) -> bytes:
    """A JPEG with an asymmetric mark, optionally carrying an EXIF rotation.

    The mark is what makes rotation OBSERVABLE: a uniform fixture would pass
    every assertion below whether or not the pixels were ever turned.
    """
    image = Image.new("RGB", size, (20, 20, 20))
    for x in range(size[0] // 4):
        for y in range(size[1] // 4):
            image.putpixel((x, y), (255, 0, 0))
    buffer = io.BytesIO()
    if orientation is None:
        image.save(buffer, format="JPEG")
    else:
        exif = Image.Exif()
        exif[274] = orientation
        image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def _red_corner(payload: bytes) -> str:
    """Which corner the fixture's bright mark ended up in.

    How the rotation is OBSERVED. ``Orientation`` 6 turns the stored frame 90°
    clockwise, so a mark stored top-left belongs top-right once the tag has been
    honoured — which makes this the difference between "upright" and "sideways"
    without asserting on a whole image.
    """
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    width, height = image.size

    def red_at(x: int, y: int) -> bool:
        pixel = image.getpixel((x, y))
        assert isinstance(pixel, tuple), "RGB conversion should give a tuple"
        return pixel[0] > 128

    if red_at(width // 8, height // 8):
        return "top-left"
    if red_at(width - width // 8, height // 8):
        return "top-right"
    return "elsewhere"


def test_orientation_does_not_depend_on_how_big_the_photo_was() -> None:
    """Review round 1, F1.

    A camera stores the sensor's raw frame plus an ``Orientation`` tag saying
    how to turn it. Re-encoding drops the tag but not the pixels, so the resize
    rung delivered a sideways photo while the verbatim rung — same photo, just
    small enough to skip the resize — delivered an upright one. Two identical
    images differing only in resolution arrived in different orientations.

    Pinned as AGREEMENT between the two rungs rather than against one expected
    orientation, because that is the actual defect: whichever way a provider
    reads EXIF, one of the two answers was wrong.
    """
    small = _oriented_jpeg((1200, 900), orientation=6)
    large = _oriented_jpeg((4000, 3000), orientation=6)

    small_payload, _mime, _summary = bound_image_for_model(small, _sniffed(small))
    large_payload, _mime, _summary = bound_image_for_model(large, _sniffed(large))

    assert _red_corner(small_payload) == _red_corner(large_payload)
    # And upright, not merely consistent: orientation 6 turns the frame 90° CW,
    # so the mark that was top-left in the stored pixels belongs top-right.
    assert _red_corner(small_payload) == "top-right"


def test_a_rotated_photo_is_delivered_without_the_tag_that_rotated_it() -> None:
    """Once the pixels are upright the tag must NOT survive, or a provider that
    honours EXIF would rotate an already-rotated image a second time."""
    source = _oriented_jpeg((1200, 900), orientation=6)
    payload, _mime, _summary = bound_image_for_model(source, _sniffed(source))
    assert Image.open(io.BytesIO(payload)).getexif().get(274) is None


@pytest.mark.parametrize("orientation", [None, 1])
def test_an_unrotated_photo_still_takes_the_verbatim_rung(orientation) -> None:
    """The orientation check must not cost the cheap path.

    ``ImageOps.exif_transpose`` returns a ``copy()`` when there is nothing to
    do, so testing "did the object change?" reports every tagless image as
    rotated and re-encodes it for nothing. Orientation ``1`` means "already
    upright" and has to be read the same way as no tag at all.
    """
    source = _oriented_jpeg((800, 600), orientation=orientation)
    payload, mime, _summary = bound_image_for_model(source, _sniffed(source))
    assert payload == source
    assert mime == "image/jpeg"


def test_unreadable_exif_metadata_does_not_fail_the_image() -> None:
    """Orientation is a refinement, not the payload. A corrupt EXIF block raises
    from deep inside Pillow, and an unrotated image beats a failed attachment."""
    source = _oriented_jpeg((800, 600), orientation=6)

    class Exploding:
        def __getattr__(self, name):
            raise OSError("corrupt EXIF")

    from local_operator import imaging

    assert imaging._needs_exif_rotation(Exploding()) is False
    # And the real path still delivers the image.
    payload, _mime, _summary = bound_image_for_model(source, _sniffed(source))
    assert payload


def test_the_source_clause_reports_the_size_on_disk_not_the_rotated_size() -> None:
    """Review round 2, F7.

    ``source WxH`` means "what you would see from ``ls`` or a re-read". Reading
    it after the transpose swapped the axes reported 3000x4000 for a file every
    other tool calls 4000x3000 — and this is the caption the MODEL reads, so a
    model deriving crop coordinates from it gets them transposed.
    """
    source = _oriented_jpeg((4000, 3000), orientation=6)
    info = _sniffed(source)
    assert info.dimensions == "4000x3000"

    _payload, _mime, summary = bound_image_for_model(source, info)
    assert "source 4000x3000" in summary
    assert "source 3000x4000" not in summary


def test_a_rotation_alone_still_declares_that_the_bytes_changed() -> None:
    """Review round 2, F7, second shape.

    An in-bounds image that is only ROTATED can come back with the same size
    and the same mime, so both of the old triggers went false and the clause
    vanished entirely — for the one case where the pixels were most rearranged.
    A PNG makes it exact: no format change to fall back on.

    Orientation ``3`` — a 180° flip — and not ``6``, which is what makes this
    test exercise the ``rotated`` trigger AT ALL. A quarter turn swaps the axes,
    so the size comparison alone already catches it and the disjunct under test
    could be deleted with every assertion still passing (review round 3, F9).
    A half turn leaves the size identical, so nothing but ``rotated`` can fire.
    """
    buffer = io.BytesIO()
    image = Image.new("RGB", (1200, 900), (10, 60, 120))
    exif = Image.Exif()
    exif[274] = 3
    image.save(buffer, format="PNG", exif=exif)
    source = buffer.getvalue()

    _payload, wire_mime, summary = bound_image_for_model(source, _sniffed(source))
    assert wire_mime == "image/png", "the fixture must not change format"
    assert "1200x900, " in summary, "the fixture must come back the same SIZE"
    assert "source 1200x900" in summary
    assert "EXIF-rotated" in summary


def test_an_untouched_image_still_says_nothing_about_a_source() -> None:
    """The clause is for reconciling a difference. With nothing to reconcile it
    must stay absent, or every caption carries noise."""
    source = _oriented_jpeg((800, 600), orientation=None)
    _payload, _mime, summary = bound_image_for_model(source, _sniffed(source))
    assert "source" not in summary


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
