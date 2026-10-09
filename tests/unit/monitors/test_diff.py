"""Normalization, hashing and the bounded delta (contract §7)."""

from __future__ import annotations

from local_operator.monitors import diff


def test_line_endings_and_trailing_whitespace_are_normalized() -> None:
    assert diff.normalize("a  \r\nb\r\n") == "a\nb"
    assert diff.normalize("a\t \nb") == "a\nb"


def test_a_trailing_newline_is_not_a_change() -> None:
    assert diff.normalize("a\nb") == diff.normalize("a\nb\n")


def test_timestamps_are_scrubbed_and_ordinary_numbers_are_not() -> None:
    line = "updated 2026-09-28T12:00:03Z and 2026-09-28 12:00 at 1759003200 / 1759003200000"
    scrubbed = diff.normalize(line)
    assert "2026-09-28" not in scrubbed
    assert "<ts>" in scrubbed
    # Below the floors, the token is data: a port, a small count, a millis
    # value past 2100. (A row count that HAPPENS to be a plausible epoch is
    # scrubbed — the rule trades that false positive for the whole
    # `updated_at` noise class; §7.1.)
    data = "rows=54321 port=8080 watch-millis=9999999999999"
    assert diff.normalize(data) == data
    # ... and the ISO half too: a bare date without a clock time is data.
    assert diff.normalize("release 2026-09-28") == "release 2026-09-28"


def test_ignored_lines_are_dropped_before_compare() -> None:
    first = diff.normalize("keep\nnoise 1\nnoise 2", ignore=("^noise",))
    second = diff.normalize("keep\nnoise 9", ignore=("^noise",))
    assert first == second == "keep"


def test_sort_lines_makes_equality_a_multiset_compare() -> None:
    first = diff.normalize("b\na\n", sort_lines=True)
    second = diff.normalize("a\nb\n", sort_lines=True)
    assert first == second
    assert diff.normalize("b\na") != diff.normalize("a\nb")


def test_content_hash_is_prefix_tagged_and_exact() -> None:
    digest = diff.content_hash("x")
    assert digest.startswith("sha256:")
    assert digest == diff.content_hash("x")
    assert digest != diff.content_hash("x ")


def test_render_delta_counts_and_previews() -> None:
    old = "a\nb\nc"
    new = "a\nB\nc\nd"
    text, changed = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=1200)
    assert changed == 3  # -b +B +d
    assert "+2/-1 changed lines" in text
    assert "- b" in text and "+ B" in text and "+ d" in text


def test_render_delta_truncates_previews_and_names_the_remainder() -> None:
    old = "\n".join(f"line {i}" for i in range(40))
    new = "\n".join(f"line {i}!" for i in range(40))
    text, changed = diff.render_delta(old, new, max_delta_lines=2, delta_max_chars=10_000)
    assert changed == 80
    assert "… and 78 more changed lines" in text


def test_render_delta_respects_the_char_budget_ever_tightened() -> None:
    old = "a"
    new = "\n".join(["x" * 200] * 12)
    text, changed = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=300)
    assert len(text) <= 300
    assert changed == 13  # one removed + twelve added
    assert "more changed lines" in text


def test_beyond_window_text_names_the_checksum() -> None:
    text, changed = diff.beyond_window_text("sha256:abc")
    assert changed == 1
    assert "beyond the stored snapshot window" in text
    assert "sha256:abc" in text


def test_has_line_difference_is_a_sequence_compare() -> None:
    # The fast-path difference check mirrors the stored form: a SEQUENCE
    # compare (sort_lines rewrites the stored text instead).
    assert diff.has_line_difference("a\nb", "b\na")
    assert diff.has_line_difference("a\nb", "a\nc")
    assert not diff.has_line_difference("", "")
    assert not diff.has_line_difference("a\nb", "a\nb")


def test_is_pure_addition_from_empty_output() -> None:
    """``grep``/``ls`` printing nothing and then a first match (review R3)."""
    from local_operator.monitors.diff import is_pure_addition

    assert is_pure_addition("", "ERROR x")
    assert not is_pure_addition("", "")
