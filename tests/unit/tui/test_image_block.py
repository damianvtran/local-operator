"""Inline transcript images: fit arithmetic, protocol encoding, the widget's
three modes, the live-image budget, and the app wiring (prompt, tool result,
resume replay, missing-image receipt).

The kitty escapes themselves are asserted as STRINGS (the encoders are pure),
and the end-to-end proof that a real terminal receives them lives in the MR
evidence (a PTY run against the real driver) — a unit test cannot own a
terminal. What the tests here pin is everything that decides WHAT is written:
mode detection, grid fits, transmit-once/replace-on-resize, budget eviction,
and the unavailable receipt.
"""

from __future__ import annotations

import base64
import io

import pytest
from rich.text import Text

from local_operator.harness.types import ImageContent
from local_operator.tui import images as images_mod
from local_operator.tui.images import (
    MAX_COLS,
    MAX_ROWS,
    PLACEHOLDER,
    CellSize,
    encode_delete,
    encode_placement,
    encode_transmit,
    fit_cells,
    placeholder_grid,
    register_live,
    release_live,
    set_escape_writer,
)
from local_operator.tui.widgets.image_block import ImageBlock

CELL = CellSize(8, 16)


@pytest.fixture(autouse=True)
def _reset():
    images_mod.reset_for_tests()
    yield
    images_mod.reset_for_tests()


def _png(width: int, height: int, color=(200, 60, 90)) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _plain(block: ImageBlock) -> str:
    """The block's rendered text, TYPE-ASSERTED as the ``Text`` every mode
    builds (the same narrowing ``test_user_block`` does): a block that starts
    returning something else should fail here and say so."""
    renderable = block.renderable
    assert isinstance(renderable, Text)
    return renderable.plain


# ---------------------------------------------------------------------------
# fit_cells: aspect ratio in pixels, caps, no upscaling
# ---------------------------------------------------------------------------


def test_fit_preserves_aspect_ratio_in_pixels() -> None:
    # 1600x1000 at 8x16 cells: height caps at 12 rows = 192px -> scale .192,
    # width 307px -> 38 cells. The cell being twice as tall as wide is the
    # whole reason rows != cols/aspect.
    cols, rows = fit_cells(1600, 1000, 72, 12, CELL)
    assert rows == 12
    assert cols == 38


def test_fit_small_image_never_upscales() -> None:
    cols, rows = fit_cells(48, 24, 72, 12, CELL)
    # 48px wide is 6 cells; 24px tall is 1.5 -> 2 rows. No inflation to caps.
    assert cols == 6
    assert rows == 2


def test_fit_degenerate_dimensions_take_one_cell() -> None:
    assert fit_cells(0, 0, 72, 12, CELL) == (1, 1)
    assert fit_cells(10000, 1, 72, 12, CELL)[1] >= 1


def test_fit_narrow_terminal_clamps_width() -> None:
    cols, rows = fit_cells(1600, 1000, 20, 12, CELL)
    assert cols <= 20
    # Width became the binding constraint, so the height drops with it.
    assert rows < 12


# ---------------------------------------------------------------------------
# Escape encoders (pure string builders)
# ---------------------------------------------------------------------------


def test_transmit_is_chunked_and_pinned_to_png() -> None:
    payload = "A" * 9000  # forces three 4096 chunks
    sequence = encode_transmit(7, payload)
    assert sequence.startswith("\x1b_Ga=t,i=7,f=100,q=2,m=1;")
    assert sequence.count("\x1b_G") == 3
    assert sequence.count("m=1") == 2  # all but the last chunk continue
    assert "m=0;" in sequence  # the final chunk closes the stream
    # The full payload survives the chunking byte for byte.
    assert sequence.count("A") == 9000


def test_placement_replaces_via_fixed_placement_id() -> None:
    sequence = encode_placement(7, cols=38, rows=12)
    assert "a=p" in sequence and "U=1" in sequence
    assert "p=1" in sequence  # same id => re-place, never stack
    assert "c=38" in sequence and "r=12" in sequence


def test_delete_targets_the_image_id() -> None:
    assert "a=d,d=I,i=9" in encode_delete(9)


def test_placeholder_grid_shape_and_codepoints() -> None:
    grid = placeholder_grid(rows=2, cols=3)
    assert len(grid) == 2
    for row in grid:
        assert row.count(PLACEHOLDER) == 3
        # placeholder + row diacritic + column diacritic per cell
        assert len(row) == 9


# ---------------------------------------------------------------------------
# Live-image budget
# ---------------------------------------------------------------------------


def test_budget_evicts_oldest_and_demotes() -> None:
    written: list[str] = []
    set_escape_writer(written.append)
    demoted: list[int] = []
    for image_id in range(1, images_mod.MAX_LIVE_KITTY_IMAGES + 2):
        register_live(image_id, lambda i=image_id: demoted.append(i))
    assert demoted == [1]  # oldest only
    assert any("a=d,d=I,i=1," in seq for seq in written)  # store reclaimed


def test_release_is_idempotent_and_deletes_once() -> None:
    written: list[str] = []
    set_escape_writer(written.append)
    register_live(5, lambda: None)
    release_live(5)
    release_live(5)
    assert sum("a=d,d=I,i=5," in seq for seq in written) == 1


# ---------------------------------------------------------------------------
# ImageBlock modes (forced via the env override the module documents)
# ---------------------------------------------------------------------------


def test_halfcell_paints_pixels_as_half_blocks(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    block = ImageBlock(_png(160, 100))
    text = _plain(block)
    assert "▀" in text
    # Every row leads with the spine indent, like every other block.
    for row in text.split("\n"):
        assert row.startswith("  ")


def test_text_mode_is_a_receipt_with_dimensions(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "text")
    block = ImageBlock(_png(160, 100))
    assert "image attached (160x100)" in _plain(block)


def test_kitty_mode_transmits_once_and_replaces_on_resize(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "kitty")
    written: list[str] = []
    set_escape_writer(written.append)
    block = ImageBlock(_png(1600, 1000))
    transmits = sum("a=t,i=" in seq for seq in written)
    placements = sum("a=p,i=" in seq for seq in written)
    assert transmits == 1 and placements == 1
    assert PLACEHOLDER in _plain(block)
    # A repaint at the same grid re-sends nothing.
    block._repaint()
    assert sum("a=t,i=" in seq for seq in written) == 1
    assert sum("a=p,i=" in seq for seq in written) == 1


def test_kitty_without_writer_falls_back_to_halfcell(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "kitty")
    set_escape_writer(None)  # headless: no driver installed
    block = ImageBlock(_png(160, 100))
    assert "▀" in _plain(block)
    assert PLACEHOLDER not in _plain(block)


def test_kitty_demotion_keeps_the_picture(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "kitty")
    written: list[str] = []
    set_escape_writer(written.append)
    block = ImageBlock(_png(160, 100))
    assert PLACEHOLDER in _plain(block)
    block._demote_to_halfcell()
    assert "▀" in _plain(block)
    assert PLACEHOLDER not in _plain(block)


def test_unavailable_receipt_for_missing_bytes(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    block = ImageBlock(None)
    plain = _plain(block)
    assert "unavailable" in plain and "no longer in the transcript" in plain


def test_unavailable_receipt_for_corrupt_bytes(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    corrupt = base64.b64encode(b"not an image").decode("ascii")
    block = ImageBlock(corrupt)
    assert "could not be decoded" in _plain(block)


def test_selection_treats_every_row_as_chrome(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    block = ImageBlock(_png(160, 100))
    assert block.copy_row_is_chrome(0)
    assert block.copy_row_is_chrome(3)


def test_retained_copy_is_capped(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    block = ImageBlock(_png(3200, 2000))
    # Original dimensions are remembered for the fit and the receipt…
    assert (block._px_width, block._px_height) == (3200, 2000)
    # …but the resident pixels are bounded by the cap grid's pixel size.
    cell = images_mod.cell_size()
    assert block._pil is not None
    assert block._pil.width <= MAX_COLS * cell.width
    assert block._pil.height <= MAX_ROWS * cell.height


# ---------------------------------------------------------------------------
# App wiring: prompt path, tool-result path, resume replay
# ---------------------------------------------------------------------------


async def _pilot_app():
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    session = FakeSession()
    return OperatorApp(lambda: _factory(session)), session


@pytest.mark.asyncio
async def test_prompt_images_render_under_the_user_block(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    app, _ = await _pilot_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        image = ImageContent(data=_png(160, 100), mime_type="image/png")
        app._submit_prompt("look [Image #1]", [image])
        await pilot.pause()
        blocks = list(app.query(ImageBlock))
        assert len(blocks) == 1
        assert "▀" in _plain(blocks[0])


@pytest.mark.asyncio
async def test_tool_result_images_render_after_the_card(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    from local_operator.harness.types import (
        TextContent,
        ToolExecutionEndEvent,
        ToolExecutionStartEvent,
        ToolResult,
    )
    from local_operator.tui.events import ToolEnded, ToolStarted

    app, _ = await _pilot_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        start = ToolExecutionStartEvent(tool_call_id="c1", tool_name="read", args={})
        app.on_tool_started(ToolStarted(start))
        await pilot.pause()
        result = ToolResult(
            tool_call_id="c1",
            tool_name="read",
            content=[
                TextContent(text="Image shot.png (image/png, 160x100)"),
                ImageContent(data=_png(160, 100), mime_type="image/png"),
            ],
        )
        end = ToolExecutionEndEvent(tool_call_id="c1", tool_name="read", result=result)
        app.on_tool_ended(ToolEnded(end))
        await pilot.pause()
        assert len(list(app.query(ImageBlock))) == 1


@pytest.mark.asyncio
async def test_resume_replays_prompt_and_tool_images(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    from local_operator.harness.types import Message, ToolCall

    app, session = await _pilot_app()
    image_b64 = _png(160, 100)
    call = ToolCall(id="c9", name="read", arguments={"path": "shot.png"})
    session._history = [
        Message(
            role="user",
            content=[
                # `Message.user` stores text + image blocks exactly like this.
                *_user_content("check this [Image #1]", image_b64),
            ],
        ),
        Message(role="assistant", content=[], tool_calls=[call]),
        Message(
            role="tool",
            tool_call_id="c9",
            content=[ImageContent(data=image_b64, mime_type="image/png")],
        ),
    ]
    async with app.run_test(size=(100, 50)) as pilot:
        await pilot.pause()
        blocks = list(app.query(ImageBlock))
        # One from the user prompt replay, one from the tool-result replay.
        assert len(blocks) == 2
        for block in blocks:
            assert "▀" in _plain(block)


def _user_content(text: str, image_b64: str):
    from local_operator.harness.types import TextContent

    return [
        TextContent(text=text),
        ImageContent(data=image_b64, mime_type="image/png"),
    ]


@pytest.mark.asyncio
async def test_resume_replays_missing_image_as_receipt(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    from local_operator.harness.types import Message

    app, session = await _pilot_app()
    session._history = [
        Message(role="user", content=_user_content("gone [Image #1]", "")),
    ]
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        blocks = list(app.query(ImageBlock))
        assert len(blocks) == 1
        assert "unavailable" in _plain(blocks[0])
