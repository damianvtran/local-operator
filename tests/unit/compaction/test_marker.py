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
    capture_elision_frontier,
    capture_elision_identities,
    render_compaction_marker,
    split_leading_marker,
    with_capture_elision_frontier,
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


def test_capture_elision_frontier_round_trips_through_a_marker() -> None:
    """The record persists on the marker, beside any strategy payload."""
    identities = (
        ("mcp:sess:cam", "screen", "1"),
        ("mcp:sess:cam", "camera", "5"),
        ("mcp:sess:cam", "screen", "2"),
    )
    preserve = with_capture_elision_frontier({"snapcompact": {"text": "x"}}, identities)
    assert preserve is not None
    marker = build_compaction_marker("A SUMMARY", preserve)

    assert marker.details["preserve_data"]["snapcompact"] == {"text": "x"}
    assert capture_elision_frontier(marker.details) == 3
    assert capture_elision_identities(marker.details) == identities


def test_capture_elision_frontier_is_omitted_when_unset_or_zero() -> None:
    """Legacy and elision-free payloads keep their exact old shape."""
    bare = build_compaction_marker("A SUMMARY")
    assert "preserve_data" not in bare.details
    assert capture_elision_frontier(bare.details) is None

    # A zero-elision pass commits NOTHING rather than spelling a zero.
    assert with_capture_elision_frontier(None, ()) is None
    kept = with_capture_elision_frontier({"snapcompact": {"text": "x"}}, ())
    assert kept == {"snapcompact": {"text": "x"}}
    assert "capture_elisions" not in kept
    zero = build_compaction_marker("A SUMMARY", with_capture_elision_frontier(None, ()))
    assert "preserve_data" not in zero.details


def test_capture_elision_frontier_malformed_values_degrade_to_none() -> None:
    """Unknown or malformed counts must read as no record."""
    for frontier in ("3", -1, True, 2.5, {}, None):
        details = {"preserve_data": {"capture_elisions": frontier}}
        assert capture_elision_frontier(details) is None
    assert capture_elision_frontier({"preserve_data": "nope"}) is None
    assert capture_elision_frontier(None) is None


def test_capture_elision_identities_degrade_to_empty() -> None:
    """A record without usable addresses elides NOTHING on replay."""
    assert capture_elision_identities(None) == ()
    assert capture_elision_identities({"preserve_data": "nope"}) == ()
    assert capture_elision_identities({"preserve_data": {"capture_elisions": {"applied": 3}}}) == ()
    assert (
        capture_elision_identities({"preserve_data": {"capture_elisions": {"identities": "nope"}}})
        == ()
    )
    record = {
        "preserve_data": {
            "capture_elisions": {
                "applied": 3,
                "identities": [
                    ["mcp:s:c", "screen", "1"],
                    ["short"],
                    [1, 2, 3],
                    ("mcp:s:c", "screen", "2"),
                ],
            }
        }
    }
    assert capture_elision_identities(record) == (
        ("mcp:s:c", "screen", "1"),
        ("mcp:s:c", "screen", "2"),
    )


def test_with_capture_elision_frontier_copies_its_input() -> None:
    original = {"snapcompact": {"text": "x"}}
    identities = (("mcp:s:c", "screen", "1"),) * 4
    merged = with_capture_elision_frontier(original, identities)

    assert merged == {
        "snapcompact": {"text": "x"},
        "capture_elisions": {
            "applied": 4,
            "identities": [["mcp:s:c", "screen", "1"]] * 4,
        },
    }
    assert original == {"snapcompact": {"text": "x"}}


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
