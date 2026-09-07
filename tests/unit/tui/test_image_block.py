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


def test_parked_kitty_image_keeps_grid_without_protocol_writes(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "kitty")
    written: list[str] = []
    set_escape_writer(written.append)
    block = ImageBlock(_png(1600, 1000), navigation_visible=False)
    grid = block._grid()
    assert not written and PLACEHOLDER not in _plain(block)
    assert grid[0] > 0 and grid[1] > 0
    block.set_navigation_visible(True)
    block._repaint()
    assert sum("a=t,i=" in seq for seq in written) == 1
    assert sum("a=p,i=" in seq for seq in written) == 1
    assert PLACEHOLDER in _plain(block)
    writes = len(written)
    block.set_navigation_visible(False)
    block._repaint()
    assert len(written) == writes and block._grid() == grid
    assert PLACEHOLDER not in _plain(block)


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
    # A repaint at a CHANGED grid of the SAME aspect re-places — one short
    # escape — without retransmitting (review round 1, F6). The stale grid
    # here keeps the real grid's aspect so only the placement is stale.
    real = block._placed
    assert real is not None
    block._placed = (real[0] // 2, real[1] // 2)  # same aspect, wrong size
    block._repaint()
    assert sum("a=t,i=" in seq for seq in written) == 1
    assert sum("a=p,i=" in seq for seq in written) == 2


def test_kitty_retransmits_when_the_grid_aspect_moves(monkeypatch) -> None:
    """A placement grid whose ASPECT differs from the transmitted frame's
    letterbox would stretch the old bars into the picture — measured 22%
    aspect error for a wide image in a 44-column terminal (round 2, F8).
    The block must retransmit padded for the new grid, and release the old
    terminal-store image."""
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "kitty")
    written: list[str] = []
    set_escape_writer(written.append)
    block = ImageBlock(_png(3200, 400))  # 8:1, the shape that exposed F8
    first_id = block._kitty_id
    assert sum("a=t,i=" in seq for seq in written) == 1
    # Simulate the fit at a much narrower terminal: half the columns, same
    # rows — a materially different grid aspect.
    placed = block._placed
    assert placed is not None
    monkeypatch.setattr(block, "_grid", lambda: (placed[0] // 2, placed[1]))
    block._repaint()
    assert sum("a=t,i=" in seq for seq in written) == 2  # retransmitted
    assert any(f"a=d,d=I,i={first_id}," in seq for seq in written)  # old freed
    assert block._kitty_id != first_id


def test_transmit_frame_matches_the_placement_grid_aspect(monkeypatch) -> None:
    """The letterboxed frame's pixel box must equal the placement grid's
    pixel rectangle — the property whose violation was F8."""
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "kitty")
    set_escape_writer(lambda s: None)
    block = ImageBlock(_png(3200, 400))
    cell = images_mod.cell_size()
    for cols, rows in ((20, 12), (38, 12), (6, 2)):
        frame = block._transmit_frame(cols, rows)
        assert (frame.width, frame.height) == (cols * cell.width, rows * cell.height)


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


def test_kitty_placement_failure_releases_the_store_entry(monkeypatch) -> None:
    """A transmit that lands whose placement cannot be written must not
    strand the image in the terminal store or a budget slot (F1)."""
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "kitty")
    written: list[str] = []

    def writer(sequence: str) -> None:
        if "a=p," in sequence:
            raise OSError("driver gone")
        written.append(sequence)

    set_escape_writer(writer)
    block = ImageBlock(_png(160, 100))
    # Demoted to half-cells, id cleared, and the transmitted image deleted.
    assert "▀" in _plain(block)
    assert block._kitty_id is None
    assert any("a=d,d=I" in seq for seq in written)


def test_no_color_forces_text_mode(monkeypatch) -> None:
    """NO_COLOR strips the fg color that carries the kitty image id, so the
    honest mode is the receipt (F2)."""
    monkeypatch.delenv("LOCAL_OPERATOR_IMAGES", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-kitty")
    images_mod.reset_for_tests()
    assert images_mod.detect_mode() == "text"


def test_receipt_truncates_to_one_row_with_ellipsis(monkeypatch) -> None:
    """Receipts must be one row at ANY width, ellipsized in the string
    itself: Content.from_rich_text drops Text's overflow flags, so a
    wrapped receipt measures two rows while painting one (D1)."""
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    block = ImageBlock(None)
    row = block._receipt_row("image unavailable — no longer in the transcript")
    # At the default 80-col guess the full reason fits.
    assert "transcript" in row.plain
    # Narrow: the row ends in an ellipsis and never carries a newline.
    # `size` is a live property over the content region, so the narrow
    # width is injected by patching it on the class (monkeypatch restores).
    from textual.geometry import Size

    narrow = ImageBlock(None)
    monkeypatch.setattr(ImageBlock, "size", property(lambda self: Size(44, 1)))
    clipped = narrow._receipt_row("image unavailable — no longer in the transcript")
    assert "\n" not in clipped.plain
    assert clipped.plain.rstrip().endswith("…")


def test_halfcell_letterboxes_small_images(monkeypatch) -> None:
    """A 2:1 icon must paint a 2:1 footprint, not stretch to fill its grid
    (D2). Asserted from the PAINTED spans: a solid red 48x24 source in a
    6x2-cell grid (6x4 half-rows) must leave at least one half-row as pure
    theme background — the letterbox bar. The old stretch-to-fill code
    painted red into every half-row, so this fails on it by construction.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    from local_operator.tui import theme as theme_mod

    block = ImageBlock(_png(48, 24, color=(255, 0, 0)))
    cols, rows = block._grid()
    assert (cols, rows) == (6, 2)  # the fit at 8x16 cells
    frame = block._build_halfcell(cols, rows)
    background = theme_mod.semantic_color("bg").lstrip("#")
    bg = f"rgb({int(background[0:2], 16)},{int(background[2:4], 16)},{int(background[4:6], 16)})"
    # Walk the styled segments: each carries "<top> on <bottom>". 48x24 at
    # 8x16 cells occupies 3 of the grid's 4 half-rows, so SOME cell must
    # pair red pixels with a background half — the bar's edge. Stretch-to-
    # fill paints red into all 4 half-rows and has no such pair.
    pairs: list[tuple[str, str]] = []
    for span in frame.spans:
        style = str(span.style)
        if " on " not in style:
            continue
        top, _, bottom = style.partition(" on ")
        pairs.append((top.strip(), bottom.strip()))
    assert pairs, "no styled half-cell spans painted"
    assert any("255,0,0" in top for top, _ in pairs), "icon pixels missing"
    assert any(
        "255,0,0" in top and bottom == bg for top, bottom in pairs
    ), "no letterbox bar: the icon was stretched to fill its grid"


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
async def test_prompt_labels_follow_marker_numbers_not_positions(monkeypatch) -> None:
    """Marker numbers are max-derived, not positional: delete #1 and paste
    twice and the prompt reads [Image #2] [Image #3]. Receipts must name the
    text's own numbers (round 2, F9)."""
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    app, _ = await _pilot_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        missing = ImageContent(data="", mime_type="image/png")
        corrupt = ImageContent(
            data=base64.b64encode(b"junk").decode("ascii"), mime_type="image/png"
        )
        app._submit_prompt("renumbered [Image #2] [Image #3]", [missing, corrupt])
        await pilot.pause()
        blocks = list(app.query(ImageBlock))
        assert len(blocks) == 2
        assert "'#2'" in _plain(blocks[0])
        assert "'#3'" in _plain(blocks[1])


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
async def test_live_peer_event_mounts_user_receipt_and_real_image(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_IMAGES", "halfcell")
    from local_operator.harness.types import ImageContent
    from local_operator.tui.events import UserMessageStart
    from local_operator.tui.widgets.transcript import UserBlock

    app, _ = await _pilot_app()
    image = ImageContent(data=_png(160, 100), mime_type="image/png")
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app.post_message(UserMessageStart("from peer [Image #1]", [image]))
        await pilot.pause()
        assert len(list(app.query(UserBlock))) == 1
        blocks = list(app.query(ImageBlock))
        assert len(blocks) == 1
        assert "▀" in _plain(blocks[0])


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
