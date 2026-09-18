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
"Timing, flakes, and how to assert that something is fast"). The single deadline
in this module is the hang guard's, and it is not a speed assertion: it exists to
turn a NON-TERMINATING regression into a failure that carries this test's name.
"""

from __future__ import annotations

import subprocess
import sys

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


def test_a_markdown_link_whose_url_has_parentheses_stays_whole() -> None:
    """Review round 1, MAJOR-1 — the most user-visible defect this layer had.

    The markdown pattern's own terminator is ``)``, so a capture that stopped at
    the first one truncated the target — and because the bare pattern then found
    the correct form at the SAME offset, the list held TWO entries for ONE link
    with the truncated one first. That row is the one the picker's cursor starts
    on, so a plain ``enter`` opened a 404 with the correct link visible one row
    below it.
    """
    text = "See [Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar)) for the case."
    assert extract_links(text) == ["https://en.wikipedia.org/wiki/Foo_(bar)"]


@pytest.mark.parametrize(
    "text,url",
    [
        ("**https://a.test/x**", "https://a.test/x"),
        ("*https://b.test/y*", "https://b.test/y"),
        ("~~https://c.test/z~~", "https://c.test/z"),
    ],
)
def test_emphasis_marks_are_not_part_of_the_url(text: str, url: str) -> None:
    """Review round 1, MAJOR-2: a model writes a bold link, and the delimiter
    rode into the URL — ``…/x**`` is a 404 with nothing on screen to explain it.
    """
    assert extract_links(text) == [url]


def test_a_markdown_target_is_trimmed_like_a_bare_one() -> None:
    """ONE trim rule for both paths. The same characters must produce the same
    string whichever pattern found them, which is what the dedupe rests on."""
    assert extract_links("see [docs](https://a.test/x.)") == ["https://a.test/x"]
    assert extract_links("see https://a.test/x.") == ["https://a.test/x"]


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


def _nested_path(depth: int) -> str:
    """``a_(b)_e`` for ``depth`` 1, ``a_(b_(c))_e`` for 2, and so on.

    Built from the inside out, because that is the only way to write a nesting
    depth that is a parameter rather than a case: the shape the defect lived in
    was a property of the DEPTH, so a fix has to be checked at depths nobody
    wrote a literal for.
    """
    inner = "b"
    for offset in range(1, depth):
        inner = f"{chr(ord('b') + offset)}({inner})"
    return f"a_({inner})_e"


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 8])
def test_a_url_is_whole_at_every_nesting_depth(depth: int) -> None:
    """Review round 2, BLOCKER — the round-1 fix's one-level paren pattern.

    That fix spelled the balance as ``\\([^\\s()]*\\)`` inside the body, which
    matches ONE flat run, so a URL nested one level deeper was not refused: the
    capture stopped and the truncated string — still a legal ``https://`` URL —
    was painted as a row and opened by ``enter``. The text in the round-2 report
    was ``https://a.test/a_(b_(c))_d``, which the PREVIOUS head got right.

    Depths 1 through 8 are asserted, on both the bare and the markdown path,
    because the failure arrived exactly one level past whatever bound was
    written: a fix checked only at the reported depth is the same fix again.
    """
    path = _nested_path(depth)
    url = f"https://a.test/{path}"
    assert extract_links(url) == [url]
    assert extract_links(f"see {url} in prose") == [url]
    assert extract_links(f"[docs]({url})") == [url]
    assert extract_links(f"see [docs]({url}) now") == [url]


def test_an_unclosed_parenthesis_is_not_part_of_the_url() -> None:
    """Balance cuts the run, and the run is what the prose owns.

    ``…/x_(y`` is a URL followed by an opening bracket the text never closes,
    so the bracket is prose — not a target truncated mid-run, and not a row
    that ends on a ``(`` for ``enter`` to open as a 404.
    """
    assert extract_links("see https://a.test/x_(y") == ["https://a.test/x_"]
    assert extract_links("see https://a.test/x_(y_(z)") == ["https://a.test/x_"]
    assert extract_links("see [t](https://a.test/x_(y") == ["https://a.test/x_"]


def test_a_closer_that_closes_nothing_ends_the_url() -> None:
    """The prose's own closer is not the URL's, at depth 0.

    This is the rule ``[label](url)`` depends on: its terminator IS a ``)``, so
    a body that kept every closer would swallow the link. The same rule then
    decides the ambiguous case the round-2 sweep asked about — a ``)`` inside a
    query or a fragment — and decides it toward the closer, which is why the
    trailing punctuation trimmer does NOT carry a ``)``.
    """
    assert extract_links("see (https://a.test/x) done") == ["https://a.test/x"]
    assert extract_links("see ((https://a.test/x)) done") == ["https://a.test/x"]
    assert extract_links("https://a.test/x?q=1)") == ["https://a.test/x?q=1"]
    assert extract_links("https://a.test/x#frag)") == ["https://a.test/x#frag"]
    assert extract_links("https://a.test/x?a=(b)") == ["https://a.test/x?a=(b)"]


def test_a_stop_character_inside_a_parenthesised_run_ends_the_url() -> None:
    """ONE character set for the body, inside a run and outside it.

    The round-2 head's parenthesised alternative used ``[^\\s()]``, so a run
    could hold the ``<>``/quote characters its own bare alternative forbade:
    ``…/x_(a<b)`` came back with the ``<`` in it — a target no browser resolves
    and nothing on screen explains. The scan applies :data:`_BODY_STOP`
    wherever it is, which is the smaller behaviour change of the two.
    """
    assert extract_links("https://a.test/x_(a<b)") == ["https://a.test/x_"]
    assert extract_links("https://a.test/x_(a`b)") == ["https://a.test/x_"]
    assert extract_links('https://a.test/x_(a"b)') == ["https://a.test/x_"]
    assert extract_links("https://a.test/x_(a'b)") == ["https://a.test/x_"]


def test_a_markdown_label_does_not_swallow_its_target() -> None:
    """Markdown's ``](`` seam ends a body, and the label's URL is still a URL.

    ``[https://a.test/x](https://a.test/x)`` — a citation whose label IS the URL
    — is a shape a model writes, and ``]`` is not in :data:`_BODY_STOP` (it must
    not be: an IPv6 literal is legal URL text), so the scan used to balance the
    label's URL straight across the seam and return
    ``https://a.test/x](https://a.test/x)``. It begins with ``https://``, so
    :func:`is_openable` cannot refuse it: the card painted that string under
    ``❯`` and ``enter`` handed it to the browser, which is the round-1 MAJOR's
    failure class — a silently wrong target, painted and opened — on a second
    shape. Review round 3, MAJOR-1.

    The first assertion is the one the round pinned; the last is the reason the
    fix is a lookahead on the PAIR and not ``]`` in the character set.
    """
    assert extract_links("[see https://a.test/x](https://b.test/y)") == [
        "https://a.test/x",
        "https://b.test/y",
    ]
    assert extract_links("[https://a.test/x](https://a.test/x)") == ["https://a.test/x"]
    assert extract_links("![https://a.test/alt](https://b.test/i.png) more") == [
        "https://a.test/alt",
        "https://b.test/i.png",
    ]
    assert extract_links("See also: https://a.test/x](https://b.test/y)") == [
        "https://a.test/x",
        "https://b.test/y",
    ]
    # Why `]` is not a stop character: this is a legal URL, and it is whole.
    assert extract_links("https://[2001:db8::1]/path") == ["https://[2001:db8::1]/path"]


def test_a_bracket_wrapped_around_the_url_is_not_part_of_it() -> None:
    """A citation's closing bracket is prose, and the URL it wrapped is whole.

    The seam rule above covers the ``]`` that closes a markdown LABEL. It does
    not cover the ``]`` a reader put AROUND the URL, and the two shapes it
    leaves are ones a model writes: a bracketed citation (``[Source: <url>]``)
    and a wiki link (``[[<url>]]``). Every one of them came back with the
    bracket still on the end — ``https://a.test/x]`` — which begins
    ``https://``, so :func:`is_openable` cannot refuse it: the card painted it
    under ``❯`` and ``enter`` handed it to the browser as a 404. The same
    silently-wrong-target class as the seam, on a third shape. QA round 3 (Q1)
    and review round 4 (MINOR-1) found it independently on the same head.

    The rule is that ``[`` and ``]`` are a PAIR like ``(`` and ``)``, so a
    ``]`` closing nothing ends the body. Both cheaper fixes are wrong:
    ``]`` in :data:`_BODY_STOP` cuts the IPv6 authority (the last assertion
    below, which is the specimen both rounds pinned), and a trimmer cannot
    tell a closing ``]`` from the specimen's own — the balance is what knows.

    The last two assertions are the residual cost, recorded rather than
    implied: a raw ``]`` inside a query or fragment now ends the body, which
    RFC 3986 requires percent-encoded anyway and is the trade the depth-0
    ``)`` rule already makes. The seam's precedence over this rule is pinned
    by the third-to-last assertion: ``https://[2001:db8::1](x)`` keeps the
    seam's cut, because that rule was settled in round 3 and this one is
    additive to it.
    """
    assert extract_links("See [https://a.test/x] for docs.") == ["https://a.test/x"]
    assert extract_links("[[https://a.test/x]]") == ["https://a.test/x"]
    assert extract_links("[Source: https://a.test/x]") == ["https://a.test/x"]
    assert extract_links("[Source: https://a.test/wiki/Foo_(bar)]") == [
        "https://a.test/wiki/Foo_(bar)"
    ]
    assert extract_links("see https://[::1]] more") == ["https://[::1]"]
    # A `]` the body's OWN `[` opened is URL text, so it stays.
    assert extract_links("https://[2001:db8::1] end") == ["https://[2001:db8::1]"]
    assert extract_links("https://a.test/x[a]") == ["https://a.test/x[a]"]
    assert extract_links("https://[2001:db8::1]/path") == ["https://[2001:db8::1]/path"]
    # The seam is settled and keeps precedence: a `](` still cuts here.
    assert extract_links("https://[2001:db8::1](x)") == ["https://[2001:db8::1"]
    assert extract_links("https://a.test/x]?q=1") == ["https://a.test/x"]
    assert extract_links("https://a.test/x?q=]") == ["https://a.test/x?q="]


#: The hang guard's probe. It is run by that test in a CHILD process, because a
#: hang inside pytest is reported by the JOB's timeout with no test name on it;
#: see the test's docstring. The text arrives as ``argv[1]`` so the shape stays
#: visible in the test that owns it rather than buried in this string.
_BACKTRACK_PROBE = (
    "import sys\n"
    "from local_operator.tui.link_targets import extract_links\n"
    "print(extract_links(sys.argv[1]))\n"
)


#: How long that child may run before it is killed and this test fails by name.
#: The fix needs milliseconds on this text and ~1.1 s of that is interpreter and
#: import startup; the deadline is sized for a machine arbitrarily slower than
#: this one, not for the fix's cost.
_BACKTRACK_DEADLINE_S = 30


def test_the_body_is_scanned_not_backtracked() -> None:
    """The round-2 head's body pattern was a nested quantifier.

    ``(?:[^\\s()<>"'`]+|\\([^\\s()]*\\))+`` repeats a repetition, so the
    characters of one run can be split between iterations of the outer ``+`` in
    exponentially many ways — and the markdown head forces that whole search to
    FAIL whenever the link's own ``)`` is absent. Measured on this 52-character
    text, the round-2 head had run for **over 5 s** when the probe gave up,
    where the scan returns in ~30 µs; a 1,215-character text of runs took 6.5 ms
    against 0.07 ms.

    The text is a SHAPE, not prose: an unterminated markdown head whose target
    holds a second ``http://``. It is here as the guard for the pattern, and its
    failure mode is a HANG rather than a red assertion — so if this test hangs,
    the body has gone back to a repetition of a repetition.

    AND THE FAILURE HAS TO CARRY THIS NAME, which is why the probe runs in a
    child with a deadline rather than in this process. A hung test is reported
    by the JOB's timeout, as a job that ran out of time with nothing pointing at
    the shape that did it; ``subprocess.run(timeout=…)`` kills the child and
    raises ``TimeoutExpired`` in this frame, so the regression arrives as
    ``test_the_body_is_scanned_not_backtracked`` — review round 3, NIT-1. The
    cost is one interpreter start (~1.1 s measured here) against ~30 µs in
    process, paid on every green run to name a failure that is otherwise
    anonymous.
    """
    text = "**[Foo (bar)](https://a.test/a_#fraghttp://b.test/**"
    finished = subprocess.run(
        [sys.executable, "-c", _BACKTRACK_PROBE, text],
        capture_output=True,
        text=True,
        timeout=_BACKTRACK_DEADLINE_S,
    )
    assert finished.stdout.strip() == "['https://a.test/a_#fraghttp://b.test/']", finished.stderr


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
