"""``split_leading_marker``'s shapes: archive text over caption, summary, plain head.

The fold-text choice is the least obvious part of the helper — a snapcompact
marker's own ``summary`` is reading instructions for frames the fold does not
carry, so the branch prefers the archive's accumulated ``text`` (the same text
``_previous_archive_text`` folds for the archive path); a context-full marker
folds its plain summary; a plain or empty head comes back untouched. Both hosts
(the session's summarize path and ``run_compaction_pass``) depend on all three,
so they are pinned here rather than through either caller.
"""

from __future__ import annotations

from local_operator.compaction.marker import (
    build_compaction_marker,
    render_compaction_marker,
    split_leading_marker,
)
from local_operator.harness.types import CustomMessage, Message


def test_a_snapcompact_marker_folds_the_archive_text_not_the_frame_caption() -> None:
    marker = CustomMessage(
        custom_type="compaction_summary",
        details={
            "summary": "Resume prior conversation. Reading the archive: 3 image frames...",
            "preserve_data": {
                "snapcompact": {
                    "frames": [],
                    "text": "OLDEST: asked about parsers\nNEWEST: fix landed in abc123",
                    "text_head": "OLDEST: asked about parsers",
                    "text_tail": "NEWEST: fix landed in abc123",
                }
            },
        },
    )
    rendered = render_compaction_marker(marker)
    new_turn = Message.user("the next question")

    previous, remaining = split_leading_marker([rendered, new_turn])

    assert previous == "OLDEST: asked about parsers\nNEWEST: fix landed in abc123"
    assert "Reading the archive" not in previous
    assert remaining == [new_turn]


def test_a_context_full_marker_folds_its_summary() -> None:
    rendered = render_compaction_marker(build_compaction_marker("THE OLD SUMMARY", None))
    new_turn = Message.user("the next question")

    previous, remaining = split_leading_marker([rendered, new_turn])

    assert previous == "THE OLD SUMMARY"
    assert remaining == [new_turn]


def test_plain_and_empty_heads_come_back_untouched() -> None:
    plain = [Message.user("a"), Message.user("b")]
    previous, remaining = split_leading_marker(plain)
    assert previous is None
    assert remaining == plain

    assert split_leading_marker([]) == (None, [])


def test_a_marker_with_no_text_is_still_removed_with_no_fold() -> None:
    """Lift XOR keep holds even when there is nothing to fold: an empty marker
    must not ride the conversation as an ordinary user turn."""
    rendered = render_compaction_marker(build_compaction_marker("", None))
    new_turn = Message.user("the next question")

    previous, remaining = split_leading_marker([rendered, new_turn])

    assert previous is None
    assert remaining == [new_turn]
