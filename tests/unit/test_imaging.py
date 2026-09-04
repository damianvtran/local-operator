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

import base64
import hashlib
import io
import random
import statistics

import pytest
from PIL import Image

from local_operator import imaging
from local_operator.imaging import (
    _REBOUND_CACHE,
    _REBOUND_CACHE_MAX_BYTES,
    IMAGE_INGEST_MAX_EDGE,
    IMAGE_MAX_BYTES,
    IMAGE_MAX_EDGE,
    IMAGE_MAX_PIXELS,
    IMAGE_REFUSAL_MAX_B64_BYTES,
    _is_line_art,
    bound_image_for_model,
    rebound_oversize_image,
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


def _noise_png(size: tuple[int, int]) -> bytes:
    """An INCOMPRESSIBLE PNG. A flat fill compresses to a few KB whatever its
    dimensions, so it cannot exercise anything that is measured in bytes."""
    import os

    buffer = io.BytesIO()
    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def _bilevel_png(size: tuple[int, int]) -> bytes:
    """Black-and-white line art: horizontal one-pixel rules, two values only.

    Stands in for the pixel-font renderings this path actually protects, without
    depending on snapcompact's geometry.
    """
    image = Image.new("L", size, 0)
    for y in range(0, size[1], 3):
        image.paste(255, (0, y, size[0], y + 1))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
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
    assert max(width, height) <= IMAGE_INGEST_MAX_EDGE
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
    assert f"{IMAGE_INGEST_MAX_EDGE}x576" in summary
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
    # Inside IMAGE_INGEST_MAX_EDGE on both edges, deliberately: a fixture the
    # ingest bound would RESIZE brings the size trigger back into play, and the
    # `rotated` disjunct under test could then be deleted with every assertion
    # still passing — the same hole review round 3, F9 closed for orientation 6.
    image = Image.new("RGB", (900, 700), (10, 60, 120))
    exif = Image.Exif()
    exif[274] = 3
    image.save(buffer, format="PNG", exif=exif)
    source = buffer.getvalue()

    _payload, wire_mime, summary = bound_image_for_model(source, _sniffed(source))
    assert wire_mime == "image/png", "the fixture must not change format"
    assert "900x700, " in summary, "the fixture must come back the same SIZE"
    assert "source 900x700" in summary
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


# ---------------------------------------------------------------------------
# Repairing history that an older build already poisoned
# ---------------------------------------------------------------------------


def test_an_oversized_block_from_an_older_build_is_rebound() -> None:
    """The reported bug, at the unit level.

    Bounding on the way in cannot reach a block that was written before it
    shipped, and a transcript is replayed verbatim on every resume — so the
    2206x266 paste that wedged a real session kept earning the many-image
    refusal on a build whose composer could no longer produce it.
    """
    source = _png((2206, 266))
    rebound = rebound_oversize_image(base64.b64encode(source).decode("ascii"))
    assert rebound is not None, "the oversized block was left as it was"
    data, mime = rebound
    width, height = Image.open(io.BytesIO(base64.b64decode(data))).size
    assert max(width, height) <= IMAGE_MAX_EDGE
    assert max(width, height) < MANY_IMAGE_PIXEL_LIMIT
    assert width / height == pytest.approx(2206 / 266, rel=0.01), "aspect ratio was not preserved"
    assert mime == "image/png"


def test_an_in_bounds_block_is_left_completely_alone() -> None:
    """``None`` means "do not touch it", and it is the answer for almost every
    block in every conversation. Re-encoding an image the provider already
    accepts would spend CPU on every turn to make the picture worse."""
    source = _png((800, 600))
    assert rebound_oversize_image(base64.b64encode(source).decode("ascii")) is None


def test_a_block_that_cannot_be_decoded_is_kept_rather_than_dropped() -> None:
    """A repair pass must never be able to destroy context on its own.

    Bytes this cannot read may still be perfectly acceptable to the provider,
    and the ``is_image_rejection`` degrade is the net for the ones that are not.
    """
    assert rebound_oversize_image("not base64 at all!!") is None

    truncated = _png((3000, 400))[:60]  # header parses, body is gone
    assert rebound_oversize_image(base64.b64encode(truncated).decode("ascii")) is None


def test_the_repair_is_memoized_so_a_resize_is_not_paid_every_turn() -> None:
    """The rendered history is rebuilt on every turn and again for every token
    count, so an uncached repair would re-decode and re-resize the same frame
    several times a turn for the life of the session."""
    source = _png((2206, 266))
    encoded = base64.b64encode(source).decode("ascii")
    _REBOUND_CACHE.clear()

    first = rebound_oversize_image(encoded)
    assert len(_REBOUND_CACHE) == 1, "an oversized block was not memoized"
    second = rebound_oversize_image(encoded)
    assert first == second, "the memo returned a different image than the resize did"

    # In-bounds blocks are settled by a header read, so they must not consume
    # cache entries: a long session of ordinary screenshots would otherwise
    # evict the one entry that actually costs something to recompute.
    small = base64.b64encode(_png((800, 600))).decode("ascii")
    rebound_oversize_image(small)
    assert len(_REBOUND_CACHE) == 1


def test_the_memo_cannot_grow_without_bound() -> None:
    """Bounded in BYTES, because the values are whole images whose sizes differ
    by orders of magnitude — an entry count would bound the wrong quantity and
    let the map retain tens of MB for the life of the process."""
    _REBOUND_CACHE.clear()
    # Derived, not the flat 40 this used to run. The cap saturates at 16
    # entries / ~30.5 MB by roughly the 17th image; every iteration past that
    # re-proves the same invariant while allocating another ~12 MB of
    # incompressible RGB, and at 40 this was the heaviest test in the suite
    # (1047 MB peak, 30 s). The count below still overshoots the saturation
    # point comfortably (27 images, 11 evictions at the current cap), so it
    # exercises eviction rather than merely filling the map, and the assertions
    # are unchanged.
    #
    # Derived from the cap rather than hard-coded so the two cannot drift:
    # "enough entries to fill the cap, plus a margin that forces evictions".
    # Raising ``_REBOUND_CACHE_MAX_BYTES`` raises the workload with it instead
    # of silently leaving the eviction assertion non-binding.
    #
    # The assumed entry size is rounded DOWN from the measured ~1.91 MB. Round
    # it up and the derived count grows more slowly than the number of entries
    # the cap can hold, so a large enough cap stops evicting and the assertion
    # goes quiet again — the very failure this derivation removes. The margin
    # scales with the cap for the same reason.
    approx_entry_bytes = 3 * 1024 * 1024 // 2
    entries_to_fill = _REBOUND_CACHE_MAX_BYTES // approx_entry_bytes
    iterations = entries_to_fill + max(6, entries_to_fill // 4)
    for index in range(iterations):
        # Distinct sizes, so every one is a distinct cache key. Photographic
        # noise, so each result is genuinely large rather than a flat PNG that
        # compresses to nothing and would never reach the cap.
        source = _noise_png((2100 + index, 2000))
        rebound_oversize_image(base64.b64encode(source).decode("ascii"))
    retained = sum(len(data) for data, _ in _REBOUND_CACHE.values())
    assert retained <= _REBOUND_CACHE_MAX_BYTES
    assert _REBOUND_CACHE, "the cap evicted so aggressively that nothing is ever cached"
    # The point of the bound is that it EVICTS. Without this the test would
    # still pass if the cap were raised high enough to hold everything, which
    # is the regression it exists to catch.
    assert len(_REBOUND_CACHE) < iterations, "nothing was ever evicted; the cap is not binding"


# ---------------------------------------------------------------------------
# Snapcompact archive frames
#
# These are the one image source that does NOT come through
# ``bound_image_for_model`` on the way in: compaction renders them to a
# per-provider geometry that is a deliberate billing decision. The repair walks
# over them like any other block, so it must be measured against a REAL frame —
# a synthetic ``Image.new("RGB", ...)`` shares neither their mode nor their
# content, and it was exactly that gap that let a 55x size regression and a
# silent rewrite of the high-res shape pass a green suite (review round 1, F4).
# ---------------------------------------------------------------------------


def _archive_frame(provider: str, model: str) -> bytes:
    from local_operator.compaction.snapcompact import render_frame, resolve_shape

    page = ("the quick brown fox jumps over the lazy dog 0123456789 " * 40 + "\n") * 40
    return render_frame(page, resolve_shape(provider, model))


def test_an_in_spec_high_res_archive_frame_is_left_alone() -> None:
    """The Anthropic high-res shape is 1932px: over ``IMAGE_MAX_EDGE`` but UNDER
    the 2000px ceiling, so no provider would refuse it.

    1932 is a costed choice ("sweet spot under the 4,784 visual-token cap").
    Repairing it would silently re-decide a billing trade that belongs to the
    compaction layer, which is the trade this module's docstring promises to
    leave alone.
    """
    frame = _archive_frame("anthropic", "claude-opus-4.7")
    assert max(Image.open(io.BytesIO(frame)).size) == 1932, "the fixture is not the high-res shape"
    assert rebound_oversize_image(base64.b64encode(frame).decode("ascii")) is None


def test_a_repaired_archive_frame_stays_a_readable_pixel_font() -> None:
    """A bilevel frame must be resampled with NEAREST and shrunk only as far as
    the refusal ceiling demands.

    The glyphs are one-pixel strokes, so a smooth filter turns each into a grey
    ramp — which destroys the legibility the font exists for and, by replacing 2
    distinct values with 256, makes the "shrunk" frame ~21x LARGER than the
    source it was shrinking. Both failure modes are asserted here because either
    one alone would look like a reasonable result.
    """
    frame = _archive_frame("google", "gemini-2.5-pro")
    source = Image.open(io.BytesIO(frame))
    source.load()
    rebound = rebound_oversize_image(base64.b64encode(frame).decode("ascii"))
    assert rebound is not None
    repaired = Image.open(io.BytesIO(base64.b64decode(rebound[0])))
    repaired.load()

    colors = repaired.getcolors(maxcolors=256)
    assert colors is not None and len(colors) <= 2, "the pixel font was antialiased into a ramp"

    # Stated against the alternative rather than against a bare ratio. Some
    # growth is unavoidable — a downscale destroys the horizontal run lengths
    # PNG was compressing — so the meaningful claim is not "it got smaller" but
    # "it did not get catastrophically bigger the way a smooth filter makes it".
    smooth = source.resize(repaired.size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    smooth.save(buffer, format="PNG")
    assert len(base64.b64decode(rebound[0])) * 4 < len(
        buffer.getvalue()
    ), "the repair is no better than the antialiasing path it exists to avoid"
    # Shrunk to the refusal ceiling, not all the way to IMAGE_MAX_EDGE: every
    # pixel taken off a 1px stroke is a stroke that may vanish.
    assert max(repaired.size) <= MANY_IMAGE_PIXEL_LIMIT
    assert max(repaired.size) > IMAGE_MAX_EDGE, "line art was shrunk further than it had to be"


def test_a_repair_lands_under_the_refusal_line_not_exactly_on_it() -> None:
    """The boundary is inferred from an error message, so it must not be sat on.

    The provider says dimensions that "exceed max allowed size ... 2000 pixels",
    which reads as making 2000 itself legal — but this is the one number whose
    failure mode is a permanently wedged session, and the module already refuses
    to sit on it for the ingest cap. A repair that landed exactly on 2000 would
    be betting the fix on a strict inequality nobody has tested.
    """
    source = _bilevel_png((2400, 900))
    rebound = rebound_oversize_image(base64.b64encode(source).decode("ascii"))
    assert rebound is not None
    repaired = Image.open(io.BytesIO(base64.b64decode(rebound[0])))
    assert max(repaired.size) < MANY_IMAGE_PIXEL_LIMIT, "a repair landed on the refusal line"

    # And the gate itself is exclusive: a block already at the limit is legal
    # and must not be rewritten.
    at_limit = _bilevel_png((MANY_IMAGE_PIXEL_LIMIT, 900))
    assert rebound_oversize_image(base64.b64encode(at_limit).decode("ascii")) is None


def test_a_dithered_halftone_is_not_treated_as_line_art() -> None:
    """Two distinct values is NOT the same predicate as line art, and the right
    resampler is opposite for the two.

    A halftone encodes tone as the RATIO of alternating black and white pixels,
    so dropping every other pixel destroys the tone it was encoding — the exact
    thing NEAREST does. This runs on the INGEST path too (every paste, every
    ``read``), so getting it wrong silently degrades scans and dithered images
    that have nothing to do with the archive frames the branch exists for.
    """
    dithered = Image.linear_gradient("L").resize((3000, 1200)).convert("1").convert("L")
    buffer = io.BytesIO()
    dithered.save(buffer, format="PNG")
    source = buffer.getvalue()

    assert dithered.getcolors(maxcolors=2) is not None, "the fixture is not two-valued"
    assert not _is_line_art(Image.open(io.BytesIO(source))), "a halftone was called line art"

    payload, _mime, _summary = bound_image_for_model(source, _sniffed(source))
    got = Image.open(io.BytesIO(payload)).convert("L")
    reference = (
        Image.linear_gradient("L")
        .resize((3000, 1200))
        .resize(got.size, Image.Resampling.LANCZOS)
        .convert("L")
    )
    # Both images are single-band ``L``, so the samples are ints; the stub types
    # these accessors wide enough to include the multi-band tuple form, hence
    # the explicit narrowing rather than arithmetic on an ``int | tuple``.
    reference_band = [int(value) for value in reference.tobytes()]
    got_band = [int(value) for value in got.tobytes()]
    error = statistics.fmean(abs(a - b) for a, b in zip(reference_band, got_band))
    # A smooth downscale reconstructs the gradient (~11); NEAREST shreds it (~85).
    assert error < 30, f"the halftone lost its tone (mean error {error:.1f})"


@pytest.mark.parametrize(
    ("height", "solid_period"),
    [
        # Defeats a ROUND `height // 64` stride: at 2048 that stride is 32, and
        # every row it reads is one of the solid ones.
        (2048, 32),
        (1280, 20),
        # Defeats a PRIME stride of 31, which collapses whenever the height is a
        # multiple of 31 — 930 reads 30 distinct rows, 155 reads 5, 62 reads 2,
        # and every one of them is solid.
        (930, 31),
        (155, 31),
        (62, 31),
    ],
)
def test_a_periodic_dither_is_never_misread_as_line_art(height: int, solid_period: int) -> None:
    """The density must be measured over every pixel, not over sampled rows.

    Any stride can be aliased by content whose vertical period shares a factor
    with it, and the misclassification is not a near miss: a halftone called
    line art is downscaled with NEAREST, which destroys the tone the alternating
    pixels encode (~85 mean error against ~11 for LANCZOS).

    Two strides have been tried in this function and each is defeated by a
    different case below, which is why the fixture is parametrised over the
    content period rather than pinned to one shape. A sampled implementation of
    any stride fails at least one of these, so a future return to sampling fails
    here rather than in a user's transcript.
    """
    width = 600
    image = Image.new("L", (width, height), 0)
    for y in range(height):
        # Dithered everywhere EXCEPT rows on the period, which are left solid
        # black — those are the rows an aliased sample reads.
        if y % solid_period:
            for x in range(0, width, 2):
                image.putpixel((x, y), 255)

    assert image.getcolors(maxcolors=2) is not None, "the fixture is not two-valued"
    assert not _is_line_art(
        image
    ), f"a period-{solid_period} dither at height {height} was misread as line art"


def test_line_art_is_still_recognised_at_a_stride_aligned_height() -> None:
    """The exact measurement must not have cost us the true positive.

    The companion to the test above: genuine line art at one of the same
    collapsing heights still has to reach the NEAREST path, or the fix for the
    aliasing would have been bought by disabling the feature.
    """
    width, height = 600, 930
    image = Image.new("L", (width, height), 255)
    for y in range(0, height, 16):  # flat horizontal rules — solid runs
        for x in range(width):
            image.putpixel((x, y), 0)

    assert _is_line_art(image), "flat rules were not recognised as line art"


def test_a_photograph_is_still_resized_smoothly_to_the_cheap_bound() -> None:
    """The NEAREST path is for line art only. Photographic content has no
    strokes to lose, is what the 1568 cost argument was measured on, and would
    look aliased under a nearest-neighbour downscale."""
    source = _noise_png((2600, 1200))
    rebound = rebound_oversize_image(base64.b64encode(source).decode("ascii"))
    assert rebound is not None
    repaired = Image.open(io.BytesIO(base64.b64decode(rebound[0])))
    assert max(repaired.size) == IMAGE_MAX_EDGE, "a photo was not taken to the cheap bound"


def test_line_art_keeps_the_correctness_bound_on_ingest() -> None:
    """PR #603 review round 1, F1.

    The ingest bound is a BILLING argument measured on photographic content,
    which has no one-pixel strokes to lose. Bilevel renderings of a small pixel
    font do, and the repair path has carried a carve-out for exactly this since
    review round 2, F8 — ingest took the sharper reduction with none of the
    protection.

    Not hypothetical: a snapcompact archive frame is a 1568px bilevel rendering
    of a 5x7 pixel font, and it reaches this function whenever such a frame is
    re-read. At 1024 its glyphs lose strokes; measured on a real
    ``render_frame`` output, "The session was compacted" is legible at 1568 and
    is not at 1024.
    """
    from local_operator.compaction import snapcompact

    text = "\n".join(
        [
            "The session was compacted at 2026-09-04T01:12Z after the context",
            "reached 412,336 tokens against a 600,000 ceiling. Below is the",
            "archived middle of the conversation, rendered as pixel-font frames.",
        ]
        * 6
    )
    frame = snapcompact.render_frame(text, snapcompact.resolve_shape("anthropic", "claude-opus-5"))
    info = _sniffed(frame)
    assert (
        max(info.width or 0, info.height or 0) > IMAGE_INGEST_MAX_EDGE
    ), "the fixture must be over the ingest bound or it proves nothing"

    payload, _mime, _summary = bound_image_for_model(frame, info)

    # Byte-identical passthrough, not merely "not shrunk to 1024": a frame this
    # function rewrites is also a frame a prompt cache has to re-write.
    assert payload == frame
    width, height = Image.open(io.BytesIO(payload)).size
    assert max(width, height) <= IMAGE_MAX_EDGE


def test_photographic_content_still_takes_the_cheaper_ingest_bound() -> None:
    """The mirror of the carve-out above: it must not swallow the saving.

    A photographic image has no strokes to lose, so it takes
    :data:`IMAGE_INGEST_MAX_EDGE` and the billed-area reduction the bound
    exists for.
    """
    source = _noise_png((2560, 1440))
    payload, _mime, _summary = bound_image_for_model(source, _sniffed(source))
    width, height = Image.open(io.BytesIO(payload)).size
    assert max(width, height) == IMAGE_INGEST_MAX_EDGE


def test_a_repair_keeps_png_when_jpeg_would_be_bigger() -> None:
    """The lossy rung must MEASURE rather than assume it wins.

    Sharp noise over an 8-colour palette at 1568x1176 measures ~1.12 MiB as PNG
    (over :data:`IMAGE_MAX_BYTES`, so the lossy rung fires) against ~1.70 MiB as
    quality-85 JPEG. Taking JPEG on the way past the budget would be worse on
    BOTH axes, bigger and lossy, so the rung declines itself.

    Exercised at the REPAIR bound rather than through ``read``: bounding ingest
    to :data:`IMAGE_INGEST_MAX_EDGE` caps the delivered area at 1024x1024, and
    at that area the rung's two conditions never hold together. A palette-noise
    sweep walks three regimes in order — PNG smaller and under budget, then
    JPEG smaller and still under budget, then JPEG smaller with PNG over
    budget — and the rung would need a fourth between the first and the last.

    Figures are deliberately NOT restated here: this comment carried three
    different wrong mechanisms across three review rounds before the generator
    was checked in. Run ``scripts/measure_ingest_lossy_rung.py`` for the table;
    the crossover colour counts are fixture-specific, the regime ORDERING is
    not, and the ordering is the whole argument.

    Still reachable whenever a caller passes an explicit ``max_edge``
    (``rebound_oversize_image`` does), which is where the branch needs cover.
    """
    rng = random.Random(1234)
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
    ]
    image = Image.new("RGB", (1568, 1176))
    pixels = image.load()
    assert pixels is not None
    for y in range(1176):
        for x in range(1568):
            pixels[x, y] = palette[rng.randrange(len(palette))]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=9)
    source = buffer.getvalue()

    payload, mime, _summary = bound_image_for_model(
        source, _sniffed(source), max_edge=IMAGE_MAX_EDGE
    )
    assert mime == "image/png", "the lossy rung took a JPEG that was BIGGER"
    assert len(payload) > IMAGE_MAX_BYTES, "the fixture no longer reaches the rung"


def test_a_block_under_every_pixel_ceiling_can_still_be_too_heavy() -> None:
    """Dimensions cannot predict the byte wall.

    Providers refuse an image block over 5 MB of base64, and an incompressible
    1900x1900 PNG clears every pixel ceiling while encoding to ~14.5 MB — so a
    dimension-only gate leaves a block that is certain to be refused.
    """
    source = _noise_png((1900, 1900))
    encoded = base64.b64encode(source).decode("ascii")
    assert max(Image.open(io.BytesIO(source)).size) <= MANY_IMAGE_PIXEL_LIMIT
    assert len(encoded) > IMAGE_REFUSAL_MAX_B64_BYTES, "the fixture is not actually too heavy"

    rebound = rebound_oversize_image(encoded)
    assert rebound is not None, "a block that would be refused on size was left in the history"
    assert len(rebound[0]) < len(encoded), "the repair did not make it lighter"


def test_the_memo_survives_a_working_set_larger_than_the_cap(monkeypatch) -> None:
    """Eviction must be oldest-first, not clear-on-overflow.

    The access pattern here is a full WALK of the same history on every render,
    not random lookups, and that is what makes the difference matter: clearing
    on overflow empties the cache part-way through each walk, so every later
    frame in that same walk misses and the memo never warms at all — measured at
    3,858 ms per warm walk against ~5 ms when the set fits.

    The cap is monkeypatched down rather than fed enough real images to reach
    32 MB. Feeding it the real thing would make this a slow test that only
    exercises eviction by accident, and the earlier version of this test did
    exactly that and proved nothing: it used four renders of the SAME text,
    which are byte-identical and collapse to one cache entry occupying 0.36% of
    the cap, so eviction never ran and the test passed against the very
    clear-on-overflow code it was written to forbid (review round 3, F10).
    """
    sources = [_noise_png((2100 + index * 40, 2000)) for index in range(6)]
    encoded = [base64.b64encode(source).decode("ascii") for source in sources]
    assert len({len(item) for item in encoded}) == len(encoded), "fixtures are not distinct"

    _REBOUND_CACHE.clear()
    for item in encoded:
        rebound_oversize_image(item)
    assert len(_REBOUND_CACHE) == len(encoded), "distinct images shared a cache entry"

    # Force a cap that fits THREE entries, then walk the whole set. Three is
    # the smallest size at which the two policies diverge: at a two-entry cap
    # both leave the final pair, so a smaller cap would make this test unable to
    # tell them apart no matter what it asserted.
    entries = sorted(len(data) for data, _ in _REBOUND_CACHE.values())
    monkeypatch.setattr(imaging, "_REBOUND_CACHE_MAX_BYTES", sum(entries[:3]))
    _REBOUND_CACHE.clear()
    for item in encoded:
        rebound_oversize_image(item)

    retained = sum(len(data) for data, _ in _REBOUND_CACHE.values())
    assert retained <= imaging._REBOUND_CACHE_MAX_BYTES, "the cap was exceeded"

    # The discriminating assertion, and it has to be about SIZE as well as
    # identity: clear-on-overflow refills from empty and so ends the walk
    # holding only the two entries added since it last cleared, while
    # oldest-first keeps the cache full at three. Both end with a newest-suffix,
    # so asserting identity alone would pass against either.
    assert len(_REBOUND_CACHE) == 3, (
        "eviction did not keep the cache full; a clearing cache refills from "
        f"empty and ends the walk holding fewer ({len(_REBOUND_CACHE)})"
    )
    expected = [hashlib.sha256(source).hexdigest() for source in sources]
    survivors = list(_REBOUND_CACHE)
    assert survivors == expected[-len(survivors) :], (
        "eviction did not keep the newest entries; a clearing cache leaves a "
        f"different set ({survivors} against {expected[-len(survivors):]})"
    )


def test_an_oversized_archive_frame_is_repaired_without_going_lossy() -> None:
    """The Google shape is 2048px and IS refusable, so it must be repaired —
    but archive frames are grayscale renderings of a 5x7 bitmap font, and the
    whole point of that font is that it stays crisp and deterministic.

    Widening the mode to RGB tripled the PNG, blew ``IMAGE_MAX_BYTES`` and
    dropped the frame onto the lossy JPEG rung, putting ringing artifacts on
    pixel-font text and inflating one measured frame 55x.
    """
    frame = _archive_frame("google", "gemini-2.5-pro")
    source = Image.open(io.BytesIO(frame))
    assert max(source.size) > MANY_IMAGE_PIXEL_LIMIT, "the fixture is not refusable"
    assert source.mode == "L", "the fixture is not the grayscale render"

    rebound = rebound_oversize_image(base64.b64encode(frame).decode("ascii"))
    assert rebound is not None, "a refusable frame was left in the history"
    data, mime = rebound
    assert mime == "image/png", "a pixel-font frame was re-encoded lossily"
    repaired = Image.open(io.BytesIO(base64.b64decode(data)))
    assert repaired.mode == "L", "grayscale was widened, which is what forces the JPEG rung"
    assert max(repaired.size) <= MANY_IMAGE_PIXEL_LIMIT
