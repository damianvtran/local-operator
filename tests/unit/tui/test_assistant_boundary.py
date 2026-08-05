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


def test_non_list_tail_keeps_late_boundary() -> None:
    """The frozen prefix may end exactly at a list's CLOSING blank: the tail
    starts with a paragraph, so the list stays whole in the frozen render."""
    text = "lead\n\n- one\n- two\n\nplain paragraph\n\nafter"
    boundary = find_stable_boundary(text)
    # Tail starts with a paragraph, not list syntax: no deferral — the
    # boundary sits right before "plain paragraph".
    assert text[boundary:] == "plain paragraph\n\nafter"


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
