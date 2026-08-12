"""Frozen-prefix boundary tests — splice preconditions and hot-path hygiene.

Covers TUI-010 (refusal/deferral preconditions) and TUI-011 (append-only
updates must not re-lex the frozen prefix: fence scanning is incremental,
and the cached renderable is reused).
"""

from __future__ import annotations

from unittest.mock import patch

from markdown_it import MarkdownIt

from local_operator.tui.widgets.assistant import AssistantBlock, find_stable_boundary

# --- fence awareness -------------------------------------------------------


def test_blank_line_inside_fence_is_not_a_boundary() -> None:
    text = "```python\nx = 1\n\ny = 2\n```\n"
    # The only blank line sits INSIDE the fence: nothing settles.
    assert find_stable_boundary(text) == 0


def test_blank_line_after_closed_fence_settles() -> None:
    text = "```\ncode\n```\n\npara tail"
    boundary = find_stable_boundary(text)
    assert boundary > 0
    # The prefix carries the whole fenced block; the tail starts at "para".
    assert text[:boundary].rstrip().endswith("```")
    assert text[boundary:] == "para tail"


def test_unclosed_fence_defers_settlement() -> None:
    text = "intro\n\n```\ncode still streaming\n"
    # The settled boundary is the blank line BEFORE the fence opener.
    boundary = find_stable_boundary(text)
    assert boundary == len("intro\n\n")
    assert text[boundary:].startswith("```")


# --- trailing-blank deferral -----------------------------------------------


def test_trailing_blank_line_defers_boundary() -> None:
    """End-of-text blanks are deferred: more content may arrive."""
    text = "settled block\n\n"
    assert find_stable_boundary(text) == 0


def test_settles_when_next_line_has_content() -> None:
    text = "settled block\n\nnext"
    boundary = find_stable_boundary(text)
    assert boundary == len("settled block\n\n")
    assert text[boundary:] == "next"


# --- reference-definition refusal (TUI-010) --------------------------------


def test_reference_definition_refuses_freeze() -> None:
    """A `[label]: url` line in the prefix can pair with a tail link — the
    freeze is REFUSED so the definition never renders dangling."""
    text = "see [docs][d] later\n\n[d]: https://example.com\n\nmore text"
    # The best candidate prefix would contain the ref def; refuse entirely.
    assert find_stable_boundary(text) == 0


def test_reference_definition_in_later_prefix_still_refused() -> None:
    text = "first\n\nsecond\n\n[ref]: http://x\n\nthird"
    # Any settled prefix here includes the definition line -> refuse.
    assert find_stable_boundary(text) == 0


def test_no_reference_definition_settles_normally() -> None:
    text = "first\n\nsecond\n\nthird"
    boundary = find_stable_boundary(text)
    assert boundary == len("first\n\nsecond\n\n")


# --- list-continuation deferral (TUI-010) ----------------------------------


def test_list_item_tail_defers_boundary_one_block() -> None:
    """Last frozen block is a list item and the tail continues list syntax:
    the boundary backs off one block so the list stays one structure."""
    text = "lead\n\n- one\n- two\n\n- three"
    boundary = find_stable_boundary(text)
    # Deferred to the blank after "lead": the tail now carries the whole
    # list, so "- three" never splits off from its siblings.
    assert boundary == len("lead\n\n")
    assert text[boundary:].startswith("- one")


def test_boundary_advances_past_an_earlier_list() -> None:
    """A list inside the frozen prefix is harmless: once the boundary sits
    after the list's closing blank, render(prefix)+render(tail) ==
    render(prefix+tail). The old "any list above pins the boundary" rule
    re-rendered the whole tail on every flush for any message opening with
    bullets — the boundary must advance into later prose instead."""
    text = "lead\n\n- one\n- two\n\nplain paragraph\n\nafter"
    boundary = find_stable_boundary(text)
    assert text[boundary:] == "after"


def test_list_continuing_tail_defers_one_block() -> None:
    """The ONLY list deferral left: the frozen block is a list item and the
    tail starts with list syntax — freezing would split the list, so the
    boundary backs off to before the list."""
    text = "lead\n\n- one\n- two\n\n- three"
    boundary = find_stable_boundary(text)
    assert text[boundary:].startswith("- one")


def test_text_without_lists_settles_at_last_blank() -> None:
    """With no list anywhere, the LAST settled blank wins (no pinning)."""
    text = "alpha\n\nbeta\n\ngamma\n\ndelta"
    boundary = find_stable_boundary(text)
    assert text[boundary:] == "delta"


# --- append-only no-relex (TUI-011) ----------------------------------------


def test_append_only_update_does_not_relex_frozen_prefix() -> None:
    """Count markdown-it parse calls: an append-only update must NOT parse
    the frozen prefix again — only the new tail is lexed."""
    block = AssistantBlock()
    block.update_text("settled paragraph\n\ntail start")
    # The prefix "settled paragraph\n\n" is now frozen.
    assert block._frozen_text == "settled paragraph\n\n"

    real_parse = MarkdownIt.parse
    parse_count = {"n": 0}

    def counting_parse(self, src, *args, **kwargs):
        parse_count["n"] += 1
        return real_parse(self, src, *args, **kwargs)

    with patch.object(MarkdownIt, "parse", counting_parse):
        block.update_text("settled paragraph\n\ntail start grows")

    # Append-only: only the NEW tail is lexed; the frozen prefix is reused.
    assert parse_count["n"] == 1


def test_append_only_fence_scan_stays_incremental() -> None:
    """Fence coverage for append-only updates scans only the new suffix."""
    block = AssistantBlock()
    block.update_text("para\n\n```\ncode")
    assert block.in_fence
    block.update_text("para\n\n```\ncode\n```\n\nafter")
    assert not block.in_fence
    # The fence rows from the FIRST update are still marked covered — the
    # coverage set was carried forward, not rebuilt from scratch.
    assert 2 in block._covered
    # The settled boundary sits after the closed fence.
    assert block._frozen_text == "para\n\n```\ncode\n```\n\n"


def test_equality_guard_is_free() -> None:
    block = AssistantBlock()
    block.update_text("hello\n\nworld")
    real_parse = MarkdownIt.parse
    parse_count = {"n": 0}

    def counting_parse(self, src, *args, **kwargs):
        parse_count["n"] += 1
        return real_parse(self, src, *args, **kwargs)

    with patch.object(MarkdownIt, "parse", counting_parse):
        block.update_text("hello\n\nworld")  # identical re-emit
    assert parse_count["n"] == 0


def test_theme_epoch_invalidates_frozen_renderable() -> None:
    """TUI-016: bumping the epoch drops the cached frozen renderable."""
    from local_operator.tui import theme as theme_mod

    block = AssistantBlock()
    block.update_text("frozen prefix\n\nlive tail")
    assert block._frozen_rendered is not None
    cached = block._frozen_rendered

    epoch_before = theme_mod.get_theme_epoch()
    try:
        # Switch ramps to bump the epoch.
        theme_mod.set_theme("light")
        block.update_text("frozen prefix\n\nlive tail grows")
        assert block._frozen_rendered is not cached
        assert block._frozen_epoch == theme_mod.get_theme_epoch()
    finally:
        theme_mod.set_theme("dark")
    assert theme_mod.get_theme_epoch() > epoch_before


# --- repaint vs reflow (the layout pass a pinned block does not need) ------


def _layout_flags(block: AssistantBlock) -> list[bool]:
    """Record the ``layout`` argument of every content update on ``block``."""
    seen: list[bool] = []
    original = type(block).set_content

    def recording(self, renderable, *, layout: bool = True) -> None:
        seen.append(layout)
        original(self, renderable, layout=layout)

    block.set_content = recording.__get__(block)  # type: ignore[assignment]
    return seen


def test_a_delta_inside_the_same_rows_repaints_without_a_layout_pass() -> None:
    """The height pin is the block's whole footprint, so a delta that lands
    in the rows already reserved has nothing for the container to re-place.

    Asking anyway reflowed every widget in the transcript — measured at 7.8 ms
    across 173 widgets — thirty times a second for the length of an answer.
    """
    block = AssistantBlock()
    block.update_text("one short line")
    rows = block._pinned_rows
    flags = _layout_flags(block)

    block.update_text("one short line plus")

    assert block._pinned_rows == rows, "precondition: the row count did not move"
    assert flags == [False]


def test_a_delta_that_adds_a_row_still_asks_for_the_layout_pass() -> None:
    """The other half of the guard: a pin that MOVES must reflow, or the
    container reserves the old height and the block paints into a hole."""
    block = AssistantBlock()
    block.update_text("first line")
    flags = _layout_flags(block)

    block.update_text("first line\n\nsecond paragraph\n\nthird paragraph")

    assert block._pinned_rows > 1, "precondition: the row count moved"
    assert flags == [True]


def test_a_resize_that_did_not_change_the_width_rebuilds_nothing() -> None:
    """The rows are a pure function of the text and the width, and pinning the
    height raises a Resize of its own — so an unguarded handler re-flattened
    the message to reproduce the rows it had just been given."""
    block = AssistantBlock()
    block.update_text("settled paragraph\n\ntail")
    width = block._built_width
    applied: list[object] = []
    original = AssistantBlock._apply_rows
    block._apply_rows = (  # type: ignore[assignment]
        lambda text: (applied.append(text), original(block, text))[1]
    )

    block.on_resize(object())

    assert block._built_width == width
    assert applied == []


def test_a_resize_to_a_new_width_does_rebuild() -> None:
    """The guard must not swallow the case it exists to let through: a real
    width change is a different set of rows, and a stale fold clips or holes."""
    block = AssistantBlock()
    block.update_text("settled paragraph\n\ntail")
    applied: list[object] = []
    original = AssistantBlock._apply_rows
    block._apply_rows = (  # type: ignore[assignment]
        lambda text: (applied.append(text), original(block, text))[1]
    )

    # `_flat_width` reads `self.size.width`, which is 0 for this unmounted
    # block and falls back to FALLBACK_WIDTH. Move the recorded width instead:
    # the handler compares the two, and either side moving is a real change.
    block._built_width = 40
    block.on_resize(object())

    assert len(applied) == 1
    assert block._built_width != 40
