"""Header sniffing, checked against an independent decoder.

Pillow is the oracle here for the same reason ``tests/unit/compaction`` uses it
against our PNG encoder: a hand-rolled parser agreeing with itself proves
nothing. Every dimension this module reports is compared with what Pillow reads
from the same bytes.
"""

from __future__ import annotations

import builtins
import io

import pytest
from PIL import Image

from local_operator.media import (
    SUPPORTED_IMAGE_MIME_TYPES,
    ImageInfo,
    sniff_image,
    sniff_image_file,
)

#: Sizes chosen for their edges, not their realism: 1x1 is the degenerate case,
#: 7x3 catches a width/height swap that square samples hide, and 1568x200 is a
#: real screenshot shape from the report that prompted this.
SIZES = [(1, 1), (7, 3), (640, 480), (1568, 200)]

FORMATS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


def _encode(fmt: str, width: int, height: int, **options: object) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buffer, format=fmt, **options)
    return buffer.getvalue()


@pytest.mark.parametrize(("fmt", "mime"), sorted(FORMATS.items()))
@pytest.mark.parametrize(("width", "height"), SIZES)
def test_every_supported_format_reports_the_dimensions_pillow_reads(
    fmt: str, mime: str, width: int, height: int
) -> None:
    """Width and height, not height and width.

    A square fixture cannot tell those apart, and the two formats here that
    store them in the opposite order (JPEG's SOF is height-first, GIF's screen
    descriptor is little-endian) are exactly where the mistake would land.
    """
    data = _encode(fmt, width, height)
    info = sniff_image(data)
    assert info is not None, fmt
    assert info.mime_type == mime
    assert (info.width, info.height) == Image.open(io.BytesIO(data)).size
    assert info.dimensions == f"{width}x{height}"
    assert info.sendable


def test_a_jpeg_frame_header_is_found_behind_a_large_metadata_segment() -> None:
    """JPEG has no fixed dimension offset — the frame sits behind however many
    APPn segments the encoder wrote, and a phone photo's EXIF thumbnail alone
    runs to tens of kilobytes. A parser that peeks at a fixed offset passes
    every synthetic fixture and fails on every real photograph.
    """
    raw = _encode("JPEG", 800, 601)
    exif = b"\xff\xe1" + (2 + 40_000).to_bytes(2, "big") + b"\x00" * 40_000
    info = sniff_image(raw[:2] + exif + raw[2:])
    assert info is not None
    assert (info.width, info.height) == (800, 601)


@pytest.mark.parametrize("options", [{"lossless": True}, {"lossless": False}])
def test_both_webp_encodings_are_read(options: dict[str, object]) -> None:
    """A clipboard hands over whichever encoding the source app produced, and
    VP8 and VP8L pack their dimensions completely differently."""
    data = _encode("WEBP", 333, 222, **options)
    info = sniff_image(data)
    assert info is not None
    assert (info.width, info.height) == (333, 222)


@pytest.mark.parametrize(
    ("label", "data"),
    [
        ("html", b"<!doctype html><html>hi</html>"),
        ("empty", b""),
        ("truncated png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 4),
        ("png magic only", b"\x89PNG\r\n\x1a\n"),
        ("jpeg magic only", b"\xff\xd8\xff"),
        ("riff but not webp", b"RIFF\x00\x00\x00\x00WAVEfmt "),
        ("plain text", b"just some notes about a screenshot.png"),
    ],
)
def test_what_is_not_a_readable_image_is_refused(label: str, data: bytes) -> None:
    """The point of sniffing by content: a `.png` that is actually HTML must
    not reach a provider as an image. A truncated header is refused for the
    same reason — dimensions we cannot read are dimensions we would invent."""
    assert sniff_image(data) is None, label


def test_heif_is_recognised_but_not_sendable() -> None:
    """An iPhone screenshot needs transcoding, and "convert this" is a
    different answer from "I do not know what this is" — only the first tells
    the caller what to do. Dimensions are absent by design: HEIF keeps them in
    a `meta`-nested `ispe` box, so they cost a container walk rather than a
    field read.
    """
    heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"
    info = sniff_image(heic)
    assert info is not None
    assert info.mime_type == "image/heic"
    assert not info.sendable
    assert info.width is None and info.height is None
    # The composer degrades on this rather than printing a zero it invented.
    assert info.dimensions == ""


def test_an_unlabelled_image_still_reads(tmp_path) -> None:
    """Extension-free is the normal case for a clipboard temp file, which is
    the whole reason this is content-sniffed."""
    path = tmp_path / "no-suffix-at-all"
    path.write_bytes(_encode("PNG", 42, 17))
    info = sniff_image_file(str(path))
    assert info is not None
    assert (info.mime_type, info.width, info.height) == ("image/png", 42, 17)


def test_sniffing_a_file_reads_only_its_header(tmp_path, monkeypatch) -> None:
    """Cost has to be independent of file size: this runs on every paste and
    every read, and pointing it at a huge file must not read the file.

    Patched on ``builtins.open`` rather than ``io.open`` — they are the same
    object, but the module resolved the name from builtins, so rebinding the
    ``io`` attribute intercepts nothing and the test passes vacuously.
    """
    path = tmp_path / "big.png"
    path.write_bytes(_encode("PNG", 64, 64) + b"\x00" * (8 * 1024 * 1024))

    reads: list[int] = []
    real_open = builtins.open

    def counting_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        real_read = handle.read

        def read(size=-1):
            reads.append(size)
            return real_read(size)

        handle.read = read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(builtins, "open", counting_open)
    info = sniff_image_file(str(path))
    monkeypatch.undo()

    assert info is not None and (info.width, info.height) == (64, 64)
    assert reads, "the patch never intercepted a read - the test proves nothing"
    assert all(0 < size <= 65_536 for size in reads), reads


def test_a_missing_or_unreadable_path_is_not_an_error(tmp_path) -> None:
    """A pasted path can point at nothing — the composer must get an answer,
    not an exception, on a keystroke."""
    assert sniff_image_file(str(tmp_path / "nope.png")) is None
    assert sniff_image_file(str(tmp_path)) is None  # a directory


def test_the_sendable_set_is_what_providers_actually_take() -> None:
    """Guards the set against a well-meaning widening: anything added here is
    something the wire layer must already serialize, or the failure moves from
    a local refusal to a provider 400 mid-turn."""
    assert SUPPORTED_IMAGE_MIME_TYPES == {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }
    assert not ImageInfo("image/heic").sendable
    assert ImageInfo("image/png", 1, 1).sendable
