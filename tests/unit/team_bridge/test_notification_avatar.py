"""The notification composite: geometry, the source-kind rule and decode limits.

Pure renderer tests (no HTTP); the route and its authority fences are covered
over loopback in ``tests/unit/server/test_desktop_team_bridge_notification_avatar.py``.
"""

from __future__ import annotations

import asyncio
import io
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from local_operator.team_bridge import avatar_cache
from local_operator.team_bridge import notification_avatar as na

ORIGIN = "https://api.radienthq.com/v1"


def _jpeg(colour=(200, 40, 40), size=(256, 256)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, colour).save(out, format="JPEG")
    return out.getvalue()


def _rgb(image: Image.Image, xy: tuple[int, int]) -> tuple[int, ...]:
    """RGB of one RGBA pixel; narrows Pillow's ``float | tuple | None`` result."""

    pixel = image.getpixel(xy)
    assert isinstance(pixel, tuple)
    return pixel[:3]


def _render(
    kind: na.Kind = "person",
    theme: na.Theme = "light",
    size: int = 256,
    initials: str = "BT",
    photo: bytes | None = None,
) -> Image.Image:
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
    assert _rgb(image, (badge_centre, badge_centre)) != (200, 40, 40)


def test_source_kind_rule_agent_marker_only_for_person_agent():
    photo = _jpeg()
    person = _render(kind="person", photo=photo)
    agent = _render(kind="person_agent", photo=photo)
    marker = (int(256 * (1 - na.AGENT_MARKER_RATIO / 2)), int(256 * na.AGENT_MARKER_RATIO / 2))
    # Without a marker that point is the red photo; with one it is not.
    assert _rgb(person, marker) == pytest.approx((200, 40, 40), abs=12)
    assert _rgb(agent, marker) != pytest.approx((200, 40, 40), abs=40)
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
        na.Variant("person", "light", 64, ""),
        origin=ORIGIN,
        account_id="acc_ben",
        photo=out.getvalue(),
    )
    assert b"secret-location" not in data


def _no_composite_state():
    """Core keeps no composite: no cache attribute, no in-flight table."""

    assert not hasattr(avatar_cache, "COMPOSITE_CACHE")
    assert not any("CACHE" in name or "IN_FLIGHT" in name for name in vars(na))


def test_render_is_stateless_and_bounded(monkeypatch):
    calls = []
    real = na.compose
    monkeypatch.setattr(na, "compose", lambda *a, **k: calls.append(1) or real(*a, **k))
    variant = na.Variant("person", "light", 64, "BT")

    async def burst():
        return await asyncio.gather(
            *[na.render(variant, origin=ORIGIN, account_id="acc_ben", photo=None) for _ in range(5)]
        )

    results = asyncio.run(burst())
    # Deterministic output, but every call renders: nothing is retained to hit.
    assert len(set(results)) == 1
    assert len(calls) == 5
    _no_composite_state()


def _client(tmp_path, store, headers=None, body=None):
    import httpx

    from local_operator.team_bridge.client import TeamBridgeClient

    transport = None
    if headers is not None:
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, content=body or _jpeg(), headers=headers)
        )
    return TeamBridgeClient(tmp_path, store=store, transport=transport)


def _call(client, v: str | None = "0123456789abcdef"):
    return client.notification_avatar(
        "acc_2",
        tenant_id="tenant_a",
        kind="person",
        theme="light",
        size=64,
        initials="BT",
        v=v,
        expected_account_id="acc_1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_control,etag", [("private", '"one"'), ("no-store", None)])
async def test_photo_render_leaves_only_the_raw_entity_rules(
    tmp_path, monkeypatch, cache_control, etag
):
    """QA Q1 condition: no derived variant can outlive (or out-retain) the raw entity."""

    from tests.unit.team_bridge.test_avatar import AvatarStore

    monkeypatch.setattr(avatar_cache, "AVATAR_CACHE", avatar_cache.AvatarCache())
    headers = {"content-type": "image/jpeg", "cache-control": cache_control}
    if etag:
        headers["etag"] = etag
    client = _client(tmp_path, AvatarStore(), headers)
    for theme in ("light", "dark"):
        result = await client.notification_avatar(
            "acc_2",
            tenant_id="tenant_a",
            kind="person",
            theme=theme,
            size=64,
            initials="BT",
            v="0123456789abcdef",
            expected_account_id="acc_1",
        )
        assert result.startswith(b"\x89PNG")
    # The raw cache follows upstream (validator kept; no-store kept nothing),
    # and no composite exists anywhere to diverge from it.
    assert len(avatar_cache.AVATAR_CACHE._entries) == (1 if cache_control == "private" else 0)
    _no_composite_state()


@pytest.mark.asyncio
async def test_auth_transition_during_render_returns_nothing(tmp_path, monkeypatch):
    from local_operator.team_bridge.client import BridgeRefusal
    from tests.unit.team_bridge.test_avatar import AvatarStore

    store = AvatarStore()
    real = na.compose

    def switch(*args, **kwargs):
        result = real(*args, **kwargs)
        store.incarnation = "login-a-again"
        return result

    monkeypatch.setattr(na, "compose", switch)
    with pytest.raises(BridgeRefusal) as failure:
        await _call(_client(tmp_path, store), v=None)
    assert failure.value.http_status == 409
    _no_composite_state()


@pytest.mark.asyncio
async def test_cancelled_caller_during_auth_transition_leaves_nothing(tmp_path, monkeypatch):
    """Review R1 condition: cancel the caller, switch login, let the worker finish.

    Before, the shielded worker could insert an old-login composite after the
    purge. Now the worker's bytes have nowhere to go: assert that the worker
    really ran to completion and that no module state grew.
    """

    from tests.unit.team_bridge.test_avatar import AvatarStore

    store = AvatarStore()
    real = na.compose
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def held(*args, **kwargs):
        entered.set()
        release.wait(5)
        try:
            return real(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(na, "compose", held)
    before = {k: v for k, v in vars(na).items() if isinstance(v, (dict, list, set))}
    task = asyncio.ensure_future(_call(_client(tmp_path, store), v=None))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    store.incarnation = "login-a-again"
    release.set()
    assert await asyncio.to_thread(finished.wait, 5)
    with pytest.raises(asyncio.CancelledError):
        await task
    # Give the shielded thread's completion a loop turn to settle.
    await asyncio.sleep(0.05)
    _no_composite_state()
    after = {k: v for k, v in vars(na).items() if isinstance(v, (dict, list, set))}
    assert after == before
    # The pending slot was released, so the renderer is not left wedged.
    assert (
        len(
            await na.render(
                na.Variant("person", "light", 64, "BT"),
                origin=ORIGIN,
                account_id="acc_ben",
                photo=None,
            )
        )
        > 0
    )


def _exclusion_mask(side=256):
    """The pixels compose() clears or covers: badge + marker clearances, outside the circle."""

    from PIL import ImageChops

    gap = max(1, int(side * na.GAP_RATIO))
    badge = int(side * na.BADGE_RATIO)
    marker = int(side * na.AGENT_MARKER_RATIO)
    mask = Image.new("L", (side, side), 0)
    mask.paste(na._circle_mask(badge + 2 * gap), (side - badge - gap, side - badge - gap))
    top = Image.new("L", (side, side), 0)
    top.paste(na._circle_mask(marker + 2 * gap), (side - marker - gap, -gap))
    return ImageChops.lighter(
        ImageChops.lighter(mask, top), ImageChops.invert(na._circle_mask(side))
    )


def test_initials_ink_never_meets_the_badge_or_marker():
    """Design D1: 0 % of any A-Z/0-9 pair's ink falls in an overlay clearance."""

    import itertools
    import string

    from PIL import ImageChops

    exclusion = _exclusion_mask()
    chars = string.ascii_uppercase + string.digits
    lost = []
    for pair in ("".join(p) for p in itertools.product(chars, repeat=2)):
        glyphs = na._placeholder(256, (0, 0, 0), pair).convert("L")
        if ImageChops.multiply(glyphs, exclusion).getbbox() is not None:
            lost.append(pair)
    assert lost == []


def test_initials_survive_in_the_actual_composite():
    """End-to-end control for D1: the rendered PNG keeps every glyph pixel opaque."""

    from PIL import ImageChops

    for kind in ("person", "person_agent"):
        image = _render(kind=kind, theme="light", size=256, initials="WQ")
        ink = na._placeholder(256, (0, 0, 0), "WQ").convert("L")
        # Every pixel where the glyph draws must still be white and opaque.
        white = image.convert("L").point([255 if v > 200 else 0 for v in range(256)])
        alpha = image.getchannel("A").point([255 if v == 255 else 0 for v in range(256)])
        kept = ImageChops.multiply(white, alpha)
        solid_ink = ink.point([255 if v == 255 else 0 for v in range(256)])
        assert solid_ink.getbbox() is not None
        assert ImageChops.subtract(solid_ink, kept).getbbox() is None, kind


def _contrast(fill):
    def channel(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in fill)
    return 1.05 / (0.2126 * r + 0.7152 * g + 0.0722 * b + 0.05)


@pytest.mark.parametrize("palette", [na._PALETTE_LIGHT, na._PALETTE_DARK])
def test_every_palette_fill_holds_white_initials_at_normal_text_contrast(palette):
    """Design D2: >= 4.5:1, the small-text bar, since 32/48 px appearances are small."""

    assert all(_contrast(fill) >= 4.5 for fill in palette), [
        (f"#{r:02x}{g:02x}{b:02x}", round(_contrast((r, g, b)), 2)) for r, g, b in palette
    ]


def test_worker_post_decode_owner_check_refuses_on_its_own():
    """The worker's check AFTER ``compose`` is a fence by itself (review R2-2).

    The client brackets ``render`` with its own checks, which overlap the
    worker's, so only a direct call proves this one: the owner is still current
    when the decode starts and gone when it finishes, and the bytes are refused.
    """

    calls = []

    def check_owner() -> None:
        calls.append(1)
        if len(calls) == 2:
            raise PermissionError("login changed during the decode")

    with pytest.raises(PermissionError):
        asyncio.run(
            na.render(
                na.Variant("person", "light", 64, "BT"),
                origin=ORIGIN,
                account_id="acc_ben",
                photo=None,
                check_owner=check_owner,
            )
        )
    assert len(calls) == 2


def _semaphore_free(semaphore: threading.BoundedSemaphore, capacity: int) -> bool:
    """True when all ``capacity`` permits are free; leaves them free again."""

    taken = 0
    while taken < capacity and semaphore.acquire(blocking=False):
        taken += 1
    for _ in range(taken):
        semaphore.release()
    return taken == capacity


def test_render_concurrency_bounds_and_cancel_release(monkeypatch):
    """2 renders at once, 8 pending, a 9th refused busy; cancelling frees both.

    Pins the reviewer's round-2 probe (R2-3) against the REAL module semaphores:
    the memory bound is the slot count, the closure-retention bound is the
    pending count, and a cancelled HTTP caller must not leak either permit
    (``asyncio.shield`` + the worker's ``finally``).
    """

    gate = threading.Event()
    guard = threading.Lock()
    active = [0]
    peak = [0]
    real = na.compose

    def held(*args, **kwargs):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            # Bounded so a regression fails the test instead of hanging the suite.
            assert gate.wait(timeout=10)
            return real(*args, **kwargs)
        finally:
            with guard:
                active[0] -= 1

    monkeypatch.setattr(na, "compose", held)
    variant = na.Variant("person", "light", 64, "BT")

    def call():
        return na.render(variant, origin=ORIGIN, account_id="acc_ben", photo=None)

    async def scenario():
        # Three executor workers, one more than the slots: a third render could
        # still enter (so a widened slot count shows as peak 3), while callers
        # 4-8 sit QUEUED in the executor, which is the only place an unshielded
        # cancel can drop a worker before its ``finally`` releases ``_PENDING``.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=3))
        tasks = [asyncio.ensure_future(call()) for _ in range(8)]
        for _ in range(200):
            if active[0] == 2:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)  # give a third worker the chance to (wrongly) enter
        assert peak[0] == 2
        with pytest.raises(na.RendererBusy):
            await call()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert all(task.cancelled() for task in tasks)
        gate.set()
        for _ in range(500):
            if _semaphore_free(na._PENDING, 8):
                break
            await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert peak[0] == 2
    assert _semaphore_free(na._PENDING, 8)
    assert _semaphore_free(na._RENDER_SLOTS, 2)
    monkeypatch.setattr(na, "compose", real)
    assert asyncio.run(call()).startswith(b"\x89PNG")
