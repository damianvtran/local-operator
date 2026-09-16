"""What counts as a link, and which links are listed.

The load-bearing properties here are the two the picker's usefulness rests on
and the one the app's safety rests on:

* **A bare URL is found.** Rich's markdown renderer sets ``link=`` on
  ``[text](url)`` and on ``<url>`` and nothing at all on a plain
  ``https://example.com`` — verified against rich 15.0.0 in this tree (one span,
  one hyperlink, for a message holding a markdown link and a bare URL). A
  picker built from the rendered spans would miss the shape users paste most,
  so the extraction reads the message SOURCE and these tests assert it on
  shapes Rich would never mark.
* **The message, not the tool output.** ``build_link_targets`` walks the
  conversation's own blocks: a URL a tool card printed is not something the user
  was reading, and listing it buries the three they were.
* **Only http(s) exists as far as the app is concerned.** The block text can
  come from a model or a tool result, so a scheme the app forwards is a scheme
  an attacker can choose.

No timing assertions: nothing here is about how long anything takes (AGENTS.md,
"Timing, flakes, and how to assert that something is fast").
"""

from __future__ import annotations

import pytest

from local_operator.tui.link_targets import (
    LinkTarget,
    build_link_targets,
    extract_links,
    is_openable,
)


def _answer(text: str, *, finalized: bool = True):
    from local_operator.tui.widgets.assistant import AssistantBlock

    block = AssistantBlock()
    block.update_text(text)
    if finalized:
        block.finalize_text()
    return block


def _prompt(text: str):
    """A prompt row. It finalizes itself in ``__init__``, like a notice."""
    from local_operator.tui.widgets.transcript import UserBlock

    return UserBlock(text, fold_width=80)


# --- extraction ----------------------------------------------------------------


def test_a_markdown_link_is_found_by_its_target_not_its_label() -> None:
    assert extract_links("Read [the docs](https://example.com/docs) now") == [
        "https://example.com/docs"
    ]


def test_a_bare_url_is_found_though_rich_never_marks_one() -> None:
    """The reason this module reads the source instead of the painted spans."""
    assert extract_links("deployed at https://bare.test/x yesterday") == ["https://bare.test/x"]


def test_an_autolink_is_found_without_its_angle_brackets() -> None:
    assert extract_links("see <https://auto.test/y>") == ["https://auto.test/y"]


def test_both_shapes_in_one_message_keep_the_order_the_reader_sees() -> None:
    """Position, not pattern order: the two are merged on where they appear.

    Asserted on a message whose markdown link comes SECOND, because the naive
    concatenation (every markdown link, then every bare URL) passes the same
    test written the other way round.
    """
    text = "plain https://a.test/1 first, then [labelled](https://b.test/2)."
    assert extract_links(text) == ["https://a.test/1", "https://b.test/2"]


def test_a_url_repeated_in_one_message_is_listed_once() -> None:
    """A markdown link's target is inside the bare pattern's reach as well."""
    assert extract_links("[t](https://a.test/x) and https://a.test/x") == ["https://a.test/x"]


def test_sentence_punctuation_is_not_part_of_the_url() -> None:
    """A URL at the end of a clause arrives as ``…/docs.``

    Opening the period is a 404 the user cannot see the cause of, on the most
    common shape in prose.
    """
    assert extract_links("See https://example.com/docs. Then stop.") == ["https://example.com/docs"]
    assert extract_links("Is it https://a.test/x?") == ["https://a.test/x"]
    assert extract_links("(https://a.test/x)") == ["https://a.test/x"]


def test_balanced_parentheses_stay_in_the_url() -> None:
    """They are legal path characters, and Wikipedia-shaped links are common.

    The naive trim cut the URL AND left the stray closer in the row, which is
    how this was found.
    """
    assert extract_links("https://en.wikipedia.org/wiki/Foo_(bar)") == [
        "https://en.wikipedia.org/wiki/Foo_(bar)"
    ]
    assert extract_links("see https://en.wikipedia.org/wiki/Foo_(bar).") == [
        "https://en.wikipedia.org/wiki/Foo_(bar)"
    ]


def test_text_without_a_link_yields_nothing() -> None:
    assert extract_links("nothing to see here") == []


# --- the scheme guard ----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vscode://file/etc/passwd",
        "chrome://settings",
        "ftp://example.com/x",
        "localhost:8080/x",
    ],
)
def test_only_http_and_https_are_openable(url: str) -> None:
    """The app hands this string to a browser, so the guard is a refusal list.

    Checked against the SCHEME rather than by parsing: ``urlparse`` accepts
    almost anything as a scheme, so a parser here would be the hole it looks
    like a fix for.
    """
    assert is_openable(url) is False


@pytest.mark.parametrize("url", ["http://x.test", "https://x.test/a", "HTTPS://X.test/a"])
def test_http_and_https_are_openable_in_any_case(url: str) -> None:
    """``HTTPS://`` is a legal URL; a lowercase-only test is a guard with a hole."""
    assert is_openable(url) is True


def test_a_non_http_scheme_is_never_extracted() -> None:
    """The guard runs at extraction too, so the row cannot exist to be chosen."""
    text = "secret file:///etc/passwd and js javascript:alert(1) and ok https://ok.test/a"
    assert extract_links(text) == ["https://ok.test/a"]


# --- the conversation walk -----------------------------------------------------


def test_the_newest_message_is_listed_first() -> None:
    """Append order is reading order, and a resumed conversation replays into
    the same column, so "most recent" means last-appended."""
    blocks = [_answer("old https://old.test/a"), _answer("new https://new.test/b")]
    assert [t.url for t in build_link_targets(blocks)] == [
        "https://new.test/b",
        "https://old.test/a",
    ]


def test_the_prompt_counts_as_a_message() -> None:
    """Users paste raw URLs into their own prompts constantly."""
    targets = build_link_targets([_prompt("look at https://pasted.test/x")])
    assert targets == [LinkTarget(url="https://pasted.test/x", sender="you", rank=1)]


def test_both_sides_are_labelled_and_ranked_by_age() -> None:
    blocks = [
        _answer("agent https://a.test/1"),
        _prompt("me https://u.test/2"),
        _answer("newest https://n.test/3"),
    ]
    assert build_link_targets(blocks) == [
        LinkTarget(url="https://n.test/3", sender="agent", rank=1),
        LinkTarget(url="https://u.test/2", sender="you", rank=2),
        LinkTarget(url="https://a.test/1", sender="agent", rank=3),
    ]


def test_a_url_in_two_messages_is_listed_once_at_its_newest() -> None:
    blocks = [_answer("https://shared.test/x"), _answer("again https://shared.test/x")]
    assert build_link_targets(blocks) == [
        LinkTarget(url="https://shared.test/x", sender="agent", rank=1)
    ]


def test_a_streaming_answer_is_not_scanned() -> None:
    """A half-written URL is not a link, and choosing it fails visibly.

    ``is_finalized`` means IMMUTABLE: an aborted answer is frozen exactly as a
    clean one is, so it is listed — only the live tail is held back.
    """
    live = _answer("still writing https://half.test/x", finalized=False)
    assert build_link_targets([live]) == []


def test_tool_output_is_not_scanned() -> None:
    """A URL a tool printed is not a link the user was reading."""
    from local_operator.tui.widgets.transcript import NoticeBlock

    notice = NoticeBlock("curl https://tool.test/x failed", fold_width=80)
    assert build_link_targets([notice, _answer("real https://real.test/x")]) == [
        LinkTarget(url="https://real.test/x", sender="agent", rank=1)
    ]


def test_a_block_that_cannot_answer_is_skipped_rather_than_raising() -> None:
    """Presence is not callability — the live shape ``copy_targets`` records.

    A ``runtime_checkable`` protocol passes a property whose getter raises, so a
    walk that trusted the check would take the turn down. This block raises from
    ``text``, which is exactly what must not reach the opener.
    """

    class Hostile:
        def is_finalized(self) -> bool:
            return True

        @property
        def text(self):
            raise RuntimeError("boom")

    assert build_link_targets([Hostile(), _answer("https://ok.test/x")]) == [
        LinkTarget(url="https://ok.test/x", sender="agent", rank=1)
    ]
