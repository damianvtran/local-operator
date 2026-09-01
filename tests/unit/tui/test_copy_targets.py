"""The `/copy` picker's target tree, pinned without an app.

The tree is a pure text transform, so these tests build messages as strings
and assert the nodes directly: a broken fence rule names the construct that
broke rather than surfacing three calls away as a wrong clipboard.

The fence tests are the ones to keep. They are where this port deliberately
does NOT follow oh-my-pi, whose closing rule is a bare ``/^```/``.
"""

from __future__ import annotations

from local_operator.tui.copy_targets import (
    MAX_MESSAGES,
    CopyTarget,
    build_copy_targets,
    extract_blocks,
    first_line,
    flatten_targets,
    plural_lines,
)


class _Assistant:
    """A settled assistant answer — the minimal shape the tree admits."""

    def __init__(self, text: str, finalized: bool = True, truncated: bool = False) -> None:
        self._text = text
        self._finalized = finalized
        self._truncated = truncated

    def text(self) -> str:
        return self._text

    def is_finalized(self) -> bool:
        return self._finalized

    def is_truncated(self) -> bool:
        return self._truncated


class _NotAnAnswer:
    """A user/notice-shaped block: text and finalized, but never truncated."""

    def __init__(self, text: str = "what about the stale rows?") -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def is_finalized(self) -> bool:
        return True


def _by_id(targets: list[CopyTarget], node_id: str) -> CopyTarget:
    return next(node.target for node in flatten_targets(targets) if node.target.id == node_id)


# --- block extraction -------------------------------------------------------


def test_a_fence_masks_its_body_so_an_inner_quote_is_not_a_quote() -> None:
    """The `>` inside a code block is code, and copying it as a quote would
    hand the user a fragment of a program with its first character eaten."""
    blocks = extract_blocks("```sh\n> redirected output\n```\n> a real quote")
    assert [(block.kind, block.body) for block in blocks] == [
        ("code", "> redirected output"),
        ("quote", "a real quote"),
    ]


def test_an_unclosed_fence_is_ordinary_text() -> None:
    """A truncated or still-streaming answer very often ends mid-fence. If the
    opener alone made a block, its body would be the whole rest of the
    message — a code target that is not code."""
    assert extract_blocks("intro\n```python\ndef f():\n    pass") == []


def test_a_tilde_fence_is_not_closed_by_backticks() -> None:
    """The divergence from oh-my-pi, whose `CLOSE_FENCE_RE = /^```/` closes any
    fence on any backtick run and does not open on `~~~` at all. Matching the
    fence character is what `_copy_markdown.classify` already does for the
    drag-select path; the two are the only readers of this grammar and they
    must agree, or the same message copies differently by route."""
    blocks = extract_blocks("~~~\ninside\n```\nstill inside\n~~~\nafter")
    assert [(block.kind, block.body) for block in blocks] == [("code", "inside\n```\nstill inside")]


def test_a_backtick_fence_is_not_closed_by_tildes() -> None:
    """The same rule in the other direction."""
    blocks = extract_blocks("```\ninside\n~~~\nstill inside\n```")
    assert [block.body for block in blocks] == ["inside\n~~~\nstill inside"]


def test_a_marker_carrying_an_info_string_opens_and_never_closes() -> None:
    """`classify` requires a closer to be marker characters alone. Treating
    ```` ```python ```` as a closer would end the block at the next fence that
    happens to name a language and split one block into two wrong ones."""
    blocks = extract_blocks("```\nfirst\n```python\nsecond\n```")
    assert [block.body for block in blocks] == ["first\n```python\nsecond"]


def test_the_info_string_becomes_the_language_and_is_stripped() -> None:
    blocks = extract_blocks("```  python  \nx = 1\n```")
    assert blocks[0].lang == "python"
    assert blocks[0].body == "x = 1"


def test_a_fence_with_no_info_string_has_no_language() -> None:
    """Which is what leaves the preview plain rather than guessing a lexer."""
    assert extract_blocks("```\nx = 1\n```")[0].lang == ""


def test_a_quote_keeps_its_inner_indentation() -> None:
    """The marker plus ONE optional space comes off. Stripping all leading
    whitespace would flatten a quoted code sample or list."""
    blocks = extract_blocks("> outer\n>     indented\n>no space")
    assert blocks[0].body == "outer\n    indented\nno space"


def test_a_blank_line_ends_a_quote_run() -> None:
    """Two quoted passages separated by prose are two targets, not one."""
    blocks = extract_blocks("> first\n\n> second")
    assert [block.body for block in blocks] == ["first", "second"]


def test_blocks_come_back_in_document_order() -> None:
    blocks = extract_blocks("> q1\n\n```\ncode\n```\n\n> q2")
    assert [block.kind for block in blocks] == ["quote", "code", "quote"]


# --- labels and hints -------------------------------------------------------


def test_the_label_is_the_first_non_empty_line_whitespace_collapsed() -> None:
    assert first_line("\n\n   Here   is    the answer  \nmore") == "Here is the answer"


def test_line_counts_are_pluralised_and_empty_text_is_zero() -> None:
    assert (plural_lines(""), plural_lines("a"), plural_lines("a\nb")) == (
        "0 lines",
        "1 line",
        "2 lines",
    )


def test_the_message_hint_counts_whole_message_lines_and_omits_absent_kinds() -> None:
    """`3 code`, not `3 code blocks`, and the line count is the message's —
    not the blocks'."""
    targets = build_copy_targets([_Assistant("intro\n\n```\na\n```\n\ntail")])
    # Seven lines: the blank separators and both fence markers count. The code
    # block itself is one line — this hint is the message's shape, not the
    # block's, which is what `1 code` beside it reports.
    assert targets[0].hint == "7 lines · 1 code"


def test_a_message_with_no_blocks_shows_only_its_line_count() -> None:
    assert build_copy_targets([_Assistant("one\ntwo")])[0].hint == "2 lines"


def test_a_code_child_hint_leads_with_its_language() -> None:
    targets = build_copy_targets([_Assistant("```python\nx = 1\n```")])
    assert _by_id(targets, "msg:1:code:0").hint == "python · 1 line"


def test_a_code_child_without_a_language_shows_lines_alone() -> None:
    targets = build_copy_targets([_Assistant("```\nx = 1\n```")])
    assert _by_id(targets, "msg:1:code:0").hint == "1 line"


# --- the tree ---------------------------------------------------------------


def test_a_message_with_no_blocks_is_a_leaf_that_copies_itself() -> None:
    targets = build_copy_targets([_Assistant("just prose")])
    assert targets[0].children == ()
    assert targets[0].content == "just prose"


def test_a_group_node_still_copies_the_whole_message() -> None:
    """Drilling in must never be the ONLY way to get the answer out."""
    text = "answer\n\n```\ncode\n```"
    targets = build_copy_targets([_Assistant(text)])
    assert targets[0].children
    assert targets[0].content == text


def test_children_are_the_blocks_then_the_all_rows() -> None:
    text = "intro\n\n```py\na\n```\n\n> q1\n\n```py\nb\n```\n\n> q2"
    targets = build_copy_targets([_Assistant(text)])
    assert [child.label for child in targets[0].children] == [
        "Block 1",
        "Quote 1",
        "Block 2",
        "Quote 2",
        "All 2 blocks",
        "All 2 quotes",
    ]


def test_a_single_block_gets_no_all_row() -> None:
    """It would copy byte-for-byte what the row above it copies."""
    targets = build_copy_targets([_Assistant("```\na\n```\n\n> q")])
    assert [child.label for child in targets[0].children] == ["Block 1", "Quote 1"]


def test_the_all_rows_join_their_bodies_with_a_blank_line() -> None:
    targets = build_copy_targets([_Assistant("```\na\n```\n\n```\nb\n```")])
    assert _by_id(targets, "msg:1:all").content == "a\n\nb"


def test_ids_are_stable_and_name_their_position() -> None:
    targets = build_copy_targets([_Assistant("```\na\n```\n\n> q1\n\n> q2")])
    assert [node.target.id for node in flatten_targets(targets)] == [
        "msg:1",
        "msg:1:code:0",
        "msg:1:quote:0",
        "msg:1:quote:1",
        "msg:1:all-quotes",
    ]


def test_only_the_code_language_reaches_the_preview_lexer() -> None:
    """Quotes and combined rows are prose; highlighting them as a language
    would colour English as syntax."""
    targets = build_copy_targets([_Assistant("```py\na\n```\n\n```py\nb\n```\n\n> q")])
    assert _by_id(targets, "msg:1:code:0").language == "py"
    assert _by_id(targets, "msg:1:all").language is None
    assert _by_id(targets, "msg:1:quote:0").language is None


# --- the walk ---------------------------------------------------------------


def test_messages_are_listed_most_recent_first() -> None:
    """Append order, because a resumed conversation replays its history into
    the same column and that is the order the reader sees."""
    targets = build_copy_targets([_Assistant("older"), _Assistant("newer")])
    assert [target.label for target in targets] == ["newer", "older"]


def test_a_block_that_is_not_an_assistant_answer_is_skipped() -> None:
    """`UserBlock`, `NoticeBlock` and `PeerMessageBlock` all carry `text()`
    and all inherit `is_finalized`. `is_truncated` is what separates them, and
    `/copy` listing the user's own words back is the failure this pins."""
    targets = build_copy_targets([_NotAnAnswer(), _Assistant("the answer")])
    assert [target.label for target in targets] == ["the answer"]


def test_a_streaming_block_is_skipped() -> None:
    """Still mutable: a clipboard that then grows is one the user cannot
    trust. The previous settled answer is offered instead of nothing."""
    targets = build_copy_targets([_Assistant("settled"), _Assistant("live", finalized=False)])
    assert [target.label for target in targets] == ["settled"]


def test_a_blank_message_is_skipped_and_does_not_stop_the_walk() -> None:
    targets = build_copy_targets([_Assistant("real"), _Assistant("   \n  ")])
    assert [target.label for target in targets] == ["real"]


def test_a_truncated_message_is_listed_and_marked_in_its_hint() -> None:
    """`is_finalized` means IMMUTABLE, not COMPLETE. Skipping the aborted
    answer would hide the message on screen — the one the user was reading and
    most likely meant. The marker LEADS because the hint is right-aligned, so
    a trailing one is the first thing a narrow terminal cuts off."""
    targets = build_copy_targets([_Assistant("half an ans", truncated=True)])
    assert targets[0].truncated is True
    assert targets[0].hint == "truncated · 1 line"


def test_a_truncated_messages_children_are_not_marked() -> None:
    """A closed fence inside a cut-off answer is itself complete: the
    truncation ended the message, not that block. Marking the child would be a
    false claim about what is on the clipboard."""
    targets = build_copy_targets([_Assistant("intro\n```\na\n```", truncated=True)])
    assert targets[0].truncated is True
    assert all(child.truncated is False for child in targets[0].children)


def test_the_walk_caps_at_fifty_messages() -> None:
    targets = build_copy_targets([_Assistant(f"answer {index}") for index in range(80)])
    assert len(targets) == MAX_MESSAGES
    assert targets[0].label == "answer 79"


def test_blank_and_non_assistant_blocks_do_not_consume_the_cap() -> None:
    """The budget is spent on messages the picker can actually list."""
    blocks: list[object] = []
    for index in range(60):
        blocks.append(_NotAnAnswer())
        blocks.append(_Assistant(""))
        blocks.append(_Assistant(f"answer {index}"))
    assert len(build_copy_targets(blocks)) == MAX_MESSAGES


def test_the_most_recent_message_names_itself_the_last_one() -> None:
    """Rank 1 is the message a bare `/copy` used to take."""
    targets = build_copy_targets([_Assistant("older"), _Assistant("newer")])
    assert targets[0].copy_message == "Copied last message to clipboard"
    assert targets[1].copy_message == "Copied message to clipboard"


def test_an_empty_transcript_builds_no_targets() -> None:
    """The caller's cue to show a notice instead of an empty overlay."""
    assert build_copy_targets([]) == []


# --- flattening -------------------------------------------------------------


def test_flattening_records_the_geometry_that_indents_each_row() -> None:
    """`ancestor_has_next` is per level, and it is what decides whether a `│`
    guide is drawn in a child's gutter: True while that ancestor still has a
    following sibling, False beneath the last root. Both states appear here,
    and note the ranks are reversed — the LAST block appended is `msg:1`."""
    targets = build_copy_targets([_Assistant("```\nolder\n```"), _Assistant("```\nnewer\n```")])
    rows = [
        (node.target.id, node.depth, node.is_last, node.ancestor_has_next)
        for node in flatten_targets(targets)
    ]
    assert rows == [
        ("msg:1", 0, False, ()),  # the newer message; another root follows it
        ("msg:1:code:0", 1, True, (True,)),  # so its child draws the guide
        ("msg:2", 0, True, ()),  # the last root
        ("msg:2:code:0", 1, True, (False,)),  # nothing below it: no guide
    ]


def test_every_target_is_copyable_today() -> None:
    """`content: str | None` is the seam the unported command targets would
    slot into, not a state this module can currently produce. The picker's
    Enter guard is written against the type, so this pins what it guards."""
    targets = build_copy_targets([_Assistant("a\n```\nb\n```\n\n```\nc\n```\n\n> q")])
    assert all(node.target.content is not None for node in flatten_targets(targets))


# --- robustness of the walk -------------------------------------------------


def test_a_crlf_answer_is_normalised_before_the_split() -> None:
    """A pasted Windows transcript arrives as `\r\n`. Splitting on `\n` alone
    leaves a `\r` on every line — invisible in a diff, and it breaks a shell
    script pasted out of the picker."""
    targets = build_copy_targets([_Assistant("Intro\r\n```py\r\nx = 1\r\n```\r\n")])
    assert targets[0].children[0].content == "x = 1"
    assert "\r" not in targets[0].children[0].content


def test_line_counts_agree_with_the_clipboard_receipt() -> None:
    """`splitlines()`, not `split("\n")`: the latter reads a trailing newline
    as a whole further line, so an answer ending in `\n` — most of them — was
    hinted one line longer than `_put_on_clipboard` reported for the same
    text. That receipt already settled this (app.py, review round 1, F3)."""
    shapes = (
        "Here is the answer.\n",
        "\nHi\n\n",
        "answer body\n\n\n   \n",
        "a\nb\n",
        "x\n\n\n",
    )
    for text in shapes:
        hint = plural_lines(text)
        assert int(hint.split()[0]) == len(text.splitlines()), (text, hint)


def test_a_block_that_raises_while_inspected_costs_only_its_own_row() -> None:
    """A `runtime_checkable` protocol tests attribute PRESENCE, not that the
    call works: a property whose getter raises satisfies `isinstance` and then
    raises when invoked. A designer saw that shape once as
    `'UserBlock' object has no attribute 'is_truncated'` raised out of this
    walk into a live `/copy`. Whatever the transient was, a clipboard command
    must not take the turn down with it."""

    class _Raises:
        def text(self) -> str:
            return "should never be listed"

        def is_finalized(self) -> bool:
            return True

        @property
        def is_truncated(self):  # type: ignore[no-untyped-def]
            raise AttributeError("half-built widget")

    targets = build_copy_targets([_Assistant("a real answer"), _Raises()])
    assert [target.label for target in targets] == ["a real answer"]


def test_a_block_whose_methods_are_not_callable_is_skipped() -> None:
    """The other half of the same gap: the names exist, so `isinstance`
    passes, but they are attributes rather than methods."""

    class _NotCallable:
        text = "not a method"
        is_finalized = True
        is_truncated = False

    assert build_copy_targets([_NotCallable()]) == []
