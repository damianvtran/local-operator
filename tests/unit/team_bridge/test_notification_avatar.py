"""The notification composite: geometry, the source-kind rule and decode limits.

Pure renderer tests (no HTTP); the route and its authority fences are covered
over loopback in ``tests/unit/server/test_desktop_team_bridge_notification_avatar.py``.
"""

from __future__ import annotations

import asyncio
import io
import struct
import zlib

import pytest
from PIL import Image

from local_operator.team_bridge import avatar_cache
from local_operator.team_bridge import notification_avatar as na

ORIGIN = "https://api.radienthq.com/v1"


def _jpeg(colour=(200, 40, 40), size=(256, 256)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format="JPEG")
    return out.getvalue()


def _render(kind="person", theme="light", size=256, initials="BT", photo=None) -> Image.Image:
    data = na.compose(
        na.Variant(kind, theme, size, initials), origin=ORIGIN, account_id="acc_ben", photo=photo
    )
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    image = Image.open(io.BytesIO(data))
    assert image.size == (size, size)
    assert image.mode == "RGBA"
    return image


@pytest.mark.parametrize("size", [64, 256])
@pytest.mark.parametrize("kind", ["person", "person_agent"])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_precropped_with_transparent_corners_and_badge_inside(size, kind, theme):
    image = _render(kind=kind, theme=theme, size=size, photo=_jpeg())
    alpha = image.getchannel("A")
    # Pre-cropped: the top-left corner is outside the circle, so Windows'
    # square-or-circle crop has nothing of ours to cut there.
    assert alpha.getpixel((0, 0)) == 0
    assert alpha.getpixel((size - 1, size - 1)) == 0  # badge disc is a circle too
    # The LO badge is opaque INSIDE the canvas at the bottom-right.
    badge_centre = int(size * (1 - na.BADGE_RATIO / 2))
    assert alpha.getpixel((badge_centre, badge_centre)) == 255
    assert image.getpixel((badge_centre, badge_centre))[:3] != (200, 40, 40)


def test_source_kind_rule_agent_marker_only_for_person_agent():
    photo = _jpeg()
    person = _render(kind="person", photo=photo)
    agent = _render(kind="person_agent", photo=photo)
    marker = (int(256 * (1 - na.AGENT_MARKER_RATIO / 2)), int(256 * na.AGENT_MARKER_RATIO / 2))
    # Without a marker that point is the red photo; with one it is not.
    assert person.getpixel(marker)[:3] == pytest.approx((200, 40, 40), abs=12)
    assert agent.getpixel(marker)[:3] != pytest.approx((200, 40, 40), abs=40)
    # The rest of the picture is the same photo.
    centre = (100, 100)
    assert person.getpixel(centre) == agent.getpixel(centre)


def test_placeholder_colour_follows_account_not_name():
    one = _render(initials="BT").getpixel((60, 60))
    renamed = _render(initials="XY").getpixel((60, 60))
    assert one == renamed
    assert na.palette_colour(ORIGIN, "acc_ben", "light") == na.palette_colour(
        ORIGIN, "acc_ben", "light"
    )
    colours = {na.palette_colour(ORIGIN, f"acc_{i}", "light") for i in range(64)}
    assert len(colours) > 4  # actually spread across the palette


@pytest.mark.parametrize("initials", ["BT", "", "李", "ÉL", "9"])
def test_placeholder_never_fails_on_any_drawable_initials(initials):
    _render(initials=initials)


@pytest.mark.parametrize("value", ["BTX", "B ", "\u202eB", "B\n", "<b"])
def test_initials_vocabulary_is_closed(value):
    assert not na.valid_initials(value)


def _png_declaring(width: int, height: int) -> bytes:
    """A tiny, well-formed PNG whose IHDR claims ``width`` x ``height``."""

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\0" * 16))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize(
    "data",
    [
        _png_declaring(30000, 30000),  # decompression bomb by declared size
        _png_declaring(na.MAX_SOURCE_SIDE + 1, 8),
        b"<svg xmlns='http://www.w3.org/2000/svg'/>",
        b"GIF89a" + b"\0" * 32,
        b"\x89PNG\r\n\x1a\n" + b"\0" * 20,  # truncated
        b"",
    ],
)
def test_decode_limits_refuse_before_allocating(data):
    with pytest.raises(na.AvatarImageRejected):
        na.decode_source(data)


def test_gif_bytes_are_refused_even_when_valid():
    out = io.BytesIO()
    Image.new("RGB", (8, 8)).save(out, format="GIF")
    with pytest.raises(na.AvatarImageRejected):
        na.decode_source(out.getvalue())


def test_output_strips_source_metadata():
    out = io.BytesIO()
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "secret-location")
    Image.new("RGB", (64, 64), (1, 2, 3)).save(out, format="PNG", pnginfo=info)
    data = na.compose(
        na.Variant("person", "light", 64, ""), origin=ORIGIN, account_id="acc_ben",
        photo=out.getvalue(),
    )
    assert b"secret-location" not in data


def test_render_caches_per_key_and_renders_once(monkeypatch):
    monkeypatch.setattr(avatar_cache, "COMPOSITE_CACHE", avatar_cache.AvatarCache())
    calls = []
    real = na.compose
    monkeypatch.setattr(na, "compose", lambda *a, **k: calls.append(1) or real(*a, **k))
    variant = na.Variant("person", "light", 64, "BT")
    key = (("root", ORIGIN), (1, "acc_1", "login-a"), "notification", 1, "acc_ben")

    async def burst():
        return await asyncio.gather(
            *[
                na.render(variant, origin=ORIGIN, account_id="acc_ben", photo=None, cache_key=key)
                for _ in range(5)
            ]
        )

    results = asyncio.run(burst())
    assert len(set(results)) == 1
    assert len(calls) == 1
    # An auth transition on that scope purges the composite with the raw cache.
    avatar_cache.admit(("root", ORIGIN), None)
    assert avatar_cache.COMPOSITE_CACHE.get(key) is None
