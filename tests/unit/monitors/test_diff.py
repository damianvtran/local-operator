"""Normalization, hashing and the bounded delta (contract §7)."""

from __future__ import annotations

import json

import pytest

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
    rendered = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=1200)
    assert rendered.changed_lines == 3  # -b +B +d
    assert "+2/-1 changed lines" in rendered.text
    assert "- b" in rendered.text and "+ B" in rendered.text and "+ d" in rendered.text
    assert rendered.hidden is False


def test_render_delta_truncates_previews_and_names_the_remainder() -> None:
    old = "\n".join(f"line {i}" for i in range(40))
    new = "\n".join(f"line {i}!" for i in range(40))
    rendered = diff.render_delta(old, new, max_delta_lines=2, delta_max_chars=10_000)
    assert rendered.changed_lines == 80
    assert "… and 78 more changed lines" in rendered.text


def test_render_delta_respects_the_char_budget_ever_tightened() -> None:
    old = "a"
    new = "\n".join(["x" * 200] * 12)
    rendered = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=300)
    assert len(rendered.text) <= 300
    assert rendered.changed_lines == 13  # one removed + twelve added
    assert "more changed lines" in rendered.text


def _thread_result(replies: int) -> str:
    """One ``slack_read_thread``-shaped result: the whole thread, one line.

    The live shape that reproduced the blind suppression: MCP wraps the thread
    as ``{"messages": [...], "pagination_info": ...}`` and serializes it
    compactly, so a 13.7 KB thread is ONE line whose only line-level change is
    a new message object near the end.
    """
    messages = [
        {
            "type": "message",
            "user": f"U{index:04d}",
            "text": f"reply {index}: " + "status update " * 90,
            "ts": f"17560000{index:02d}.000100",
        }
        for index in range(replies)
    ]
    return json.dumps(
        {"messages": messages, "pagination_info": {"has_more": False, "next_cursor": ""}},
        separators=(",", ":"),
    )


def test_a_single_line_json_change_is_visible_in_the_delta() -> None:
    """The 2026-10-20 failure, reproduced at the diff layer.

    Two prefix clips of a 13.7 KB single-line result are byte-identical (both
    are the thread header), so the only visible difference was the spill
    footer. The delta must instead show the characters that changed — the new
    reply — for the reader AND for the materiality gate.
    """
    old = _thread_result(10)
    new = _thread_result(11)
    assert "\n" not in old
    assert len(old) > 13_000, len(old)

    rendered = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=2000)
    assert rendered.changed_lines == 2  # one line replaced by one line
    assert rendered.hidden is False
    previews = [line for line in rendered.text.split("\n") if line[:2] in ("+ ", "- ")]
    assert len(previews) == 2, rendered.text
    old_preview, new_preview = previews
    assert old_preview != new_preview, "the two previews must not be the same prefix clip"
    # A window, not a prefix: each side is marked where it was cut.
    assert old_preview.startswith("- …") and new_preview.startswith("+ …")
    assert "reply 10:" in new_preview, new_preview


def test_a_hidden_preview_is_reported_as_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fail-open signal is derived from what the previews actually show.

    No current renderer path can hide a long pair's difference (the window is
    anchored on it), so this pins the detector itself against a renderer that
    regresses to identical clips: the gate must be told the evidence is not
    there rather than judging it.
    """
    monkeypatch.setattr(diff, "_clip", lambda line: "…")
    rendered = diff.render_delta("a", "b", max_delta_lines=12, delta_max_chars=1200)
    assert rendered.hidden is True


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


def test_is_pure_addition_survives_a_line_count_crossing_a_power_of_ten() -> None:
    """QA round 1, Q1: ``read`` re-pads every gutter at 9->10 lines."""
    from local_operator.monitors.diff import is_pure_addition

    # Faithful to ``_number_lines``: the column is right-aligned to the widest
    # line number, so 9 lines use ``N| `` and 10 lines `` N| `` on EVERY line.
    old = "\n".join(f"{i:>1}| line {i}" for i in range(1, 10))
    new = "\n".join(f"{i:>2}| line {i}" for i in range(1, 11))
    assert is_pure_addition(old, new)


def test_is_pure_addition_still_refuses_a_gutter_only_change() -> None:
    from local_operator.monitors.diff import is_pure_addition

    assert not is_pure_addition("3| x", "4| x")


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("a", "a\nb", True),  # append
        ("a\nc", "a\nb\nc", True),  # insert-mid (spot-checked, review R6)
        ("a", "x\na", True),  # prepend (newest-first listing)
        ("", "ERROR x", True),  # empty old: grep/ls first match
        ("a\nb", "a", False),  # delete
        ("a\nb", "a\nc", False),  # replace
        ("a\nb", "b\na", False),  # reorder without sort_lines
        ("a", "a\n2| <ts>", False),  # timestamp-only add
        ("a", "a\n\n", False),  # blank-only add
    ],
)
def test_is_pure_addition_table(old: str, new: str, expected: bool) -> None:
    from local_operator.monitors.diff import is_pure_addition

    assert is_pure_addition(old, new) is expected


def test_replace_previews_keep_the_minus_then_plus_order() -> None:
    """Review round 1, F2: the gate's input must not be reordered.

    Windowing pairs the two sides to find the first difference, but the
    rendered order stays the pre-change one — every ``-`` line of a hunk, then
    every ``+`` line — so an all-short replace hunk is byte-identical to the
    old renderer.
    """
    assert diff.render_delta(
        "a\nb\nc", "A\nB\nc", max_delta_lines=12, delta_max_chars=1200
    ).text == ("+2/-2 changed lines\n- a\n- b\n+ A\n+ B")


def test_a_windowed_replace_pair_also_keeps_minus_before_plus() -> None:
    old = "x" * 300 + "OLD"
    new = "x" * 300 + "NEW"
    rendered = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=1200)
    previews = [line for line in rendered.text.split("\n") if line[:2] in ("+ ", "- ")]
    assert previews[0].startswith("- …") and previews[1].startswith("+ …")
    assert "OLD" in previews[0] and "NEW" in previews[1]


# ---------------------------------------------------------------------------
# The record split: a JSON envelope over a text record log (§7.1)
# ---------------------------------------------------------------------------

#: The channel header an MCP read tool puts on top of its message payload.
_CHANNEL_HEADER = "Channel: #product-feedback (C06TXN4AD9R)"

#: What a record's fields are joined with once its internal newlines collapse.
_FIELD_SEPARATOR = " · "


def _message_block(
    body: str,
    *,
    user: str = "U058BN13Q0Y",
    ts: str = "1762400000.000100",
    thread: str | None = None,
) -> str:
    """One message as the MCP read tool renders it: a blank-line-separated block."""
    lines = [
        f"=== Message from Gong Support Bot ({user}) at 2026-11-05T20:10:00Z ===",
        f"Message TS: {ts}",
        body,
    ]
    if thread is not None:
        lines.append(f"Thread: {thread}")
    return "\n".join(lines)


def _channel_envelope(*blocks: str, extra: dict[str, object] | None = None) -> str:
    """The measured ``slack_read_channel`` result: one logged string, JSON-escaped.

    The escaping IS the defect: the whole message log is one line of JSON, so a
    new message can only replace that line.
    """
    payload = "\n".join([_CHANNEL_HEADER, ""] + [block + "\n" for block in blocks])
    envelope: dict[str, object] = {"messages": payload}
    envelope.update(extra or {})
    return json.dumps(envelope)


def _record_lines(normalized: str) -> list[str]:
    """The record lines of a normalized envelope result (no header, no blanks)."""
    return [line for line in normalized.split("\n") if line and line != _CHANNEL_HEADER]


def test_a_json_envelope_over_a_message_log_becomes_one_line_per_message() -> None:
    """The measured 2026-11-05 Slack shape: records, not one escaped line."""
    raw = _channel_envelope(
        _message_block("Victor: the risk report needs a second look", user="U0VICTOR"),
        _message_block("Mike: acknowledged", user="U0MIKE"),
        _message_block("Sara: on it", user="U0SARA"),
    )
    assert "\n" not in raw  # the defect: the payload's newlines arrive escaped

    normalized = diff.normalize(raw)

    assert normalized.split("\n")[0] == _CHANNEL_HEADER
    records = _record_lines(normalized)
    assert len(records) == 3, normalized
    assert "U0VICTOR" in records[0] and "needs a second look" in records[0]
    assert "U0MIKE" in records[1] and "acknowledged" in records[1]
    # No envelope is left behind: no escaped newline, no JSON key, no quotes.
    assert "\\n" not in normalized
    assert '"messages"' not in normalized and '"' not in normalized


def test_a_records_fields_keep_their_source_order() -> None:
    normalized = diff.normalize(_channel_envelope(_message_block("the body text")))
    record = _record_lines(normalized)[0]
    author, ts_field, body = record.split(_FIELD_SEPARATOR)
    assert author.startswith("=== Message from Gong Support Bot (U058BN13Q0Y) at ")
    assert ts_field.startswith("Message TS: ")
    assert body == "the body text"


def test_a_thread_token_change_is_a_one_line_change_on_the_parent_record() -> None:
    """Thread replies surface on the parent's record line, so the gate sees them.

    Buried in the envelope this information belongs to the one long line every
    message rewrites; on its own record line it is a single changed line whose
    preview carries the new reply count.
    """
    before = diff.normalize(
        _channel_envelope(_message_block("parent"), _message_block("other", ts="1762400100.0"))
    )
    after = diff.normalize(
        _channel_envelope(
            _message_block("parent", thread="2 replies (latest: 2026-11-05T20:17:00Z)"),
            _message_block("other", ts="1762400100.0"),
        )
    )
    assert diff.count_changes(before, after) == (1, 1)
    # The reply count rides the PARENT's record line — one line per record is
    # what puts a thread's new reply where the reader and the gate can see it.
    parent_before, parent_after = _record_lines(before)[0], _record_lines(after)[0]
    assert "Thread:" not in parent_before
    assert "Thread: 2 replies" in parent_after
    previews = [
        line
        for line in diff.render_delta(
            before, after, max_delta_lines=12, delta_max_chars=1200
        ).text.split("\n")
        if line[:2] in ("+ ", "- ")
    ]
    assert len(previews) == 2, previews
    # The ``-`` side of a replace hunk comes first (the pre-change order); the
    # new reply count is on the ``+`` side, which is what the gate reads.
    assert previews[0].startswith("- ") and "Thread:" not in previews[0]
    assert previews[1].startswith("+ ") and "2 replies" in previews[1]


def test_ignore_drops_a_whole_record_for_the_watchers_own_echo() -> None:
    """``ignore`` is per line, so a record line is the unit it can drop.

    This is what makes the watcher's own posted reply invisible: on the
    unsplit envelope the author id matches the one line every record shares, so
    the pattern would drop the whole channel (or nothing).
    """
    raw = _channel_envelope(
        _message_block("a reply the watcher itself posted", user="U0SELF"),
        _message_block("someone else's message", user="U0OTHER"),
    )
    kept = diff.normalize(raw, ignore=("U0SELF",))
    assert len(_record_lines(kept)) == 1
    assert "U0OTHER" in kept and "U0SELF" not in kept


def test_a_records_clock_and_message_id_survive_scrubbing() -> None:
    """The wake must say WHEN and WHAT, or it cannot be acted on (design D1).

    A record log's own clock is content, not the churn class the timestamp rule
    exists for: the operator's report was a timing report. The record clock is
    rendered in a readable slash form (outside the ISO churn pattern) and a
    dotted ``<epoch>.<fraction>`` id is an IDENTIFIER, so both survive while a
    bare epoch in the body is still scrubbed.
    """
    normalized = diff.normalize(
        _channel_envelope(_message_block("row 1759003200 was written", ts="1793908014.147179"))
    )
    record = _record_lines(normalized)[0]
    assert "at 2026/11/05 20:10:00Z" in record, normalized
    assert "Message TS: 1793908014.147179" in record, normalized
    # A bare 10-digit epoch in the body is still the churn class: scrubbed.
    assert "1759003200" not in normalized
    assert "<ts>" in normalized


def test_a_non_record_source_still_scrubs_its_timestamps() -> None:
    """The clock exemption is scoped to record lines, not to normalize()."""
    assert diff.normalize("updated 2026-11-05 14:46:54 EST") == "updated <ts> EST"


def test_a_plain_text_result_is_returned_unchanged() -> None:
    text = "line one\nline two\n\nkeep {not json}\n"
    assert diff.unpack_records(text) == text
    assert diff.normalize(text) == "line one\nline two\n\nkeep {not json}"


def test_a_truncated_json_envelope_is_returned_unchanged() -> None:
    """The snapshot cap can cut the envelope mid-string; that is not the trigger.

    Falling back to the previous behaviour is the documented contract: a
    monitor never fails, and a result it cannot decode is compared as text.
    """
    raw = _channel_envelope(_message_block("a body " * 500))
    truncated = raw[: len(raw) // 2]
    assert "\n" not in truncated
    assert diff.unpack_records(truncated) == truncated
    # One line, i.e. today's behaviour: no record split happened.
    assert "\n" not in diff.normalize(truncated)


def test_a_compact_json_document_of_message_objects_is_not_unpacked() -> None:
    """``{"messages": [{…}]}`` has no payload STRING, so the split must not fire.

    That shape is a compact array-of-objects result already; it is rendered by
    the delta's character window and its test above must keep its input.
    """
    raw = _thread_result(3)
    assert "\n" not in raw
    # The split is what this test pins; the timestamp-like tokens inside the
    # JSON keys are scrubbed regardless, exactly as before this change.
    assert diff.unpack_records(raw) == raw
    assert "\n" not in diff.normalize(raw)


def test_envelope_bookkeeping_never_reaches_the_compared_text() -> None:
    """A page cursor is page state, and it churned a line on every tick (Q-1)."""
    raw = _channel_envelope(
        _message_block("body"),
        extra={
            "pagination_info": "There are more messages available. cursor: `bmV4dF90`\n",
            "next_cursor": "bmV4dF90",
            "has_more": True,
            "count": 3,
        },
    )
    normalized = diff.normalize(raw)
    assert "pagination_info" not in normalized
    assert "cursor" not in normalized and "has_more" not in normalized
    assert "bmV4dF90" not in normalized
    # A non-bookkeeping field keeps its key and its own line.
    assert normalized.split("\n")[-1] == "count: 3"
    assert "body" in normalized


def test_an_envelope_of_single_line_fields_is_not_a_record_log() -> None:
    """The trigger needs a BLANK LINE (review MIN-1/Q-1, design D2).

    A prose field ends in one newline; treating it as a record split a sentence
    into records and dropped the key. Such an envelope keeps the previous
    behaviour exactly — one line, compared as text.
    """
    raw = json.dumps({"messages": "a\n", "note": "one line\nof prose"})
    assert diff.is_record_log(raw) is False
    assert diff.unpack_records(raw) == raw
    assert diff.normalize(raw) == raw


def test_a_container_field_renders_as_compact_json_with_sorted_keys() -> None:
    """Deterministic output: an envelope that reorders a container is not a change."""
    first = diff.normalize(
        _channel_envelope(_message_block("body"), extra={"meta": {"b": 1, "a": 2}})
    )
    second = diff.normalize(
        _channel_envelope(_message_block("body"), extra={"meta": {"a": 2, "b": 1}})
    )
    assert first == second
    assert first.split("\n")[-1] == 'meta: {"a":2,"b":1}'


def test_unpacking_is_idempotent() -> None:
    """Re-normalizing the stored form (the scheduler's baseline re-derivation)."""
    once = diff.unpack_records(_channel_envelope(_message_block("body")))
    assert diff.unpack_records(once) == once


def test_a_new_message_is_an_appended_record_line() -> None:
    """The acceptance shape: ``+1/-0``, and the preview IS the message text."""
    first = _message_block("first")
    second = _message_block("second", ts="1762400100.0")
    old = diff.normalize(_channel_envelope(second, first))
    new = diff.normalize(
        _channel_envelope(_message_block("brand new feedback", ts="1762400200.0"), second, first)
    )
    assert diff.is_pure_addition(old, new) is True
    rendered = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=1200)
    assert rendered.changed_lines == 1
    lines = rendered.text.split("\n")
    assert lines[0] == "+1/-0 changed lines"
    assert lines[1].startswith("+ ")
    assert "brand new feedback" in lines[1]


def test_a_shifted_window_shows_the_new_and_dropped_records_separately() -> None:
    """A fixed ``limit`` drops the oldest too: ``+1/-1``, both sides legible."""
    third = _message_block("third", ts="1762400200.0")
    second = _message_block("second", ts="1762400100.0")
    first = _message_block("first", ts="1762400000.0")
    old = diff.normalize(_channel_envelope(third, second, first))
    new = diff.normalize(
        _channel_envelope(_message_block("newest feedback", ts="1762400300.0"), third, second)
    )
    assert diff.is_pure_addition(old, new) is False
    assert diff.count_changes(old, new) == (1, 1)
    rendered = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=1200)
    previews = [line for line in rendered.text.split("\n") if line[:2] in ("+ ", "- ")]
    assert len(previews) == 2, rendered.text
    assert previews[0].startswith("+ ") and "newest feedback" in previews[0]
    assert previews[1].startswith("- ") and "first" in previews[1]


def test_a_multi_paragraph_message_stays_one_record() -> None:
    """A message body with blank lines is still ONE record (design D2).

    Measured on real traffic: a message split into a header plus four
    unattributed fragments (``+2/-7`` where one message had arrived), which also
    defeated ``ignore`` — the fragment carrying the author id was dropped and
    the prose survived.
    """
    body = (
        "*Recap* for the week\n\nWe shipped the faster feed searches.\n\n"
        "| | 2h before | 3h after |\n\n*Cleanup:* the worktrees are removed."
    )
    raw = _channel_envelope(
        _message_block(body, user="U0POSTER"),
        _message_block("a second message", user="U0OTHER", ts="1762400100.0"),
    )
    records = _record_lines(diff.normalize(raw))
    assert len(records) == 2, records
    assert records[0].startswith("=== Message from Gong Support Bot (U0POSTER)")
    assert "*Recap* for the week" in records[0]
    assert "*Cleanup:* the worktrees are removed." in records[0]
    assert "| | 2h before | 3h after |" in records[0]


def test_a_payload_without_a_repeated_lead_keeps_the_block_behaviour() -> None:
    """No repeated record prefix: blocks stay lines (the documented fallback)."""
    raw = json.dumps({"log": "alpha one\n\nbeta two\n\ngamma three"})
    assert diff.normalize(raw) == "alpha one\nbeta two\ngamma three"


def test_a_long_added_record_is_delivered_whole_not_clipped_at_200() -> None:
    """The added line IS the wake: it gets the budget, not a 200-char clip.

    Measured (design D1/MAJ-1): an 823-char record was delivered as a link
    preamble and a ``…``, so the actual feedback never arrived.
    """
    body = ":telephone_receiver: *Call with Certn* — *Recap*: " + (
        "the report needs a second look. " * 24
    )
    assert len(body) > 700
    old = diff.normalize(_channel_envelope(_message_block("older", ts="1762400100.0")))
    new = diff.normalize(
        _channel_envelope(
            _message_block(body, ts="1762400200.0", user="U0VICTOR"),
            _message_block("older", ts="1762400100.0"),
        )
    )
    rendered = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=1200)
    assert rendered.text.split("\n")[0] == "+1/-0 changed lines"
    assert "*Recap*" in rendered.text and rendered.text.rstrip().endswith("look.")
    assert len(rendered.text) <= 1200


def test_a_long_added_record_is_cut_at_the_budget_it_has_left() -> None:
    """Bounded, not dropped: a record past the budget arrives as a marked cut."""
    body = "x" * 2000
    old = "a"
    new = "a\n" + body
    rendered = diff.render_delta(old, new, max_delta_lines=12, delta_max_chars=300)
    assert len(rendered.text) <= 300
    assert rendered.text.split("\n")[1].startswith("+ ")
    assert rendered.text.split("\n")[1].endswith("…")


def test_an_indentation_only_change_is_still_a_change() -> None:
    """``rstrip`` per field, not ``strip``: leading indentation is content (MIN-2).

    A line-mode envelope (``{"stdout": …}``) re-indenting a block used to
    normalize identically, so the change never surfaced.
    """
    before = json.dumps({"stdout": "def f():\n    indent 4\n\ndef g():\n    indent 4"})
    after = json.dumps({"stdout": "def f():\n    indent 4\n\ndef g():\n        indent 8"})
    assert diff.normalize(before) != diff.normalize(after)
    # The indentation must survive INSIDE the record line: ``strip()`` removed
    # it, so a re-indent normalised identically and the change never surfaced.
    assert diff.normalize(after).count("\n") == 1
    assert f"{diff._RECORD_FIELD_SEPARATOR}        indent 8" in diff.normalize(after)


def test_a_body_timestamp_stays_masked_while_the_header_clock_is_readable() -> None:
    """The readable rendering is for HEADER fields only (review R2-MIN-A).

    Body text is content the source wrote: an ``updated_at`` inside it is the
    churn class the timestamp rule exists for, and leaving it readable let a
    changing body stamp ride out ungated (a record watch skips the gate).
    """
    raw = _channel_envelope(_message_block("first para\n\nupdated 2026-11-05T21:00:00Z by the bot"))
    record = _record_lines(diff.normalize(raw))[0]
    assert "at 2026/11/05 20:10:00Z ===" in record, record
    assert "updated <ts> by the bot" in record, record
    assert "2026-11-05T21:00:00Z" not in record


def test_a_single_record_with_paragraphs_is_still_one_record() -> None:
    """A ONE-message log has no repeated lead, and must not fragment (R2-MIN-D).

    The block fallback is for payloads with no record structure at all; a single
    record's body may contain blank lines and is still one record.
    """
    raw = json.dumps(
        {
            "messages": _CHANNEL_HEADER
            + "\n\n"
            + _message_block("*Recap*\n\nSecond paragraph of the same message.")
            + "\n"
        }
    )
    lines = diff.normalize(raw).split("\n")
    assert lines[0] == _CHANNEL_HEADER
    assert len(lines) == 2, lines
    assert "*Recap*" in lines[1] and "Second paragraph" in lines[1]
