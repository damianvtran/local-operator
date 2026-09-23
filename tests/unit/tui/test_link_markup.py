"""``autolink_bare_urls``: what gets promoted to a link, and what must not be.

Rich's markdown renderer sets ``link=`` for ``[label](target)`` and for the
explicit autolink ``<target>`` and NOTHING for a URL written plainly — verified
against rich 15.0.0 in this tree (``test_link_targets`` records the same
measurement from the extraction side). So the shape users and models write most
has no link span at all, and a click handler on top of it has nothing under the
pointer to find. The promotion this module performs is therefore half the fix
for "hyperlinks don't work in the TUI", and the seed of every test here is the
property the other routes depend on:

* **A bare URL becomes an autolink.** Otherwise the painted URL is
  indistinguishable from prose: no colour, no underline, no OSC-8, and no cell
  for ``TranscriptBlock.link_at`` to resolve.
* **The promotion is IDEMPOTENT.** The streaming path re-renders the same source
  on every flush and the resize path re-renders per width, so a transform that
  re-ran on its own output would have to be applied to a string it had already
  rewritten. ``<url>`` is masked on the next pass precisely so the brackets
  cannot accumulate; the test asserts the composition of the function with
  itself on the shapes those two paths actually re-render.
* **Prose punctuation stays outside the brackets.** ``see https://a.test/x.``
  renders as a sentence and the link still resolves to the URL. Bracketing the
  period would change what the reader sees and what the browser is asked for.
* **Code and existing links are left alone.** A URL inside a code sample is an
  argument being shown, not an address being offered, and wrapping it puts
  literal angle brackets in text meant to be copied; an existing link's target
  is already a link, and bracketing it would nest brackets inside a target and
  break the parse.

Both modules must also agree about masks that are SCOPED: leaving code alone
must not leave the whole message alone, so the prose after a closed fence is
promoted, while an UNTERMINATED fence — a half-arrived streamed sample — keeps
everything that follows it masked.

No timing assertions: nothing here is about how long anything takes (AGENTS.md,
"Timing, flakes, and how to assert that something is fast").
"""

from __future__ import annotations

import pytest

from local_operator.tui.link_markup import autolink_bare_urls

#: Shapes the re-rendering paths really feed the function: a plain sentence, a
#: pair of URLs in one message, emphasis, a code span and a fenced sample, the
#: two already-linked forms, a Wikipedia-shaped URL — and an UNTERMINATED fence,
#: which is what a streaming flush hands over before the closing marker arrives.
_ALREADY_RENDERED_SHAPES = [
    "deployed at https://a.test/x yesterday",
    "See https://a.test/x. Then https://b.test/y?",
    "**https://a.test/x** in bold",
    "`curl https://a.test/x`",
    "```\ncurl https://a.test/x\n```",
    "```\ncurl https://a.test/x",
    "[label](https://a.test/x) and https://b.test/y",
    "<https://a.test/x>",
    "https://en.wikipedia.org/wiki/Foo_(bar)",
]


# --- promotion -----------------------------------------------------------------


def test_a_bare_url_in_prose_becomes_an_autolink() -> None:
    """The whole point: the shape rich marks with nothing at all."""
    assert (
        autolink_bare_urls("deployed at https://a.test/x yesterday")
        == "deployed at <https://a.test/x> yesterday"
    )


def test_sentence_punctuation_stays_outside_the_brackets() -> None:
    """A period is the sentence's, not the address's.

    Bracketing it would both change the rendered sentence and ask the browser
    for a URL with a ``.`` it does not have.
    """
    assert autolink_bare_urls("See https://a.test/x.") == "See <https://a.test/x>."
    assert autolink_bare_urls("Is it https://a.test/x?") == "Is it <https://a.test/x>?"
    assert autolink_bare_urls("See https://example.com/docs. Then stop.") == (
        "See <https://example.com/docs>. Then stop."
    )


@pytest.mark.parametrize("text", _ALREADY_RENDERED_SHAPES)
def test_promoting_a_promoted_message_changes_nothing(text: str) -> None:
    """Idempotence, on the shapes the two re-rendering paths re-feed it.

    The streaming path re-renders on every flush, so the second pass would give
    ``<<https://a.test/x>>`` — and the renderer would then treat the inner
    brackets as the link and the outer pair as text. Asserted as composition
    rather than as a fixed expected string: "running it again changes nothing"
    is the property, and it is the one a fix for a missing mask breaks.
    """
    once = autolink_bare_urls(text)
    assert autolink_bare_urls(once) == once


@pytest.mark.parametrize(
    "text,expected",
    [
        ("**https://a.test/x**", "**<https://a.test/x>**"),
        ("*https://b.test/y*", "*<https://b.test/y>*"),
        ("~~https://c.test/z~~", "~~<https://c.test/z>~~"),
    ],
)
def test_emphasis_delimiters_stay_outside_the_brackets(text: str, expected: str) -> None:
    """A model writes a bold link, and the ``**`` belongs to the sentence.

    Inside the brackets it is a 404 the reader cannot see the cause of, the
    defect ``link_targets._TRAILING_JUNK`` records from the other route. The
    marks are asserted to land OUTSIDE the brackets rather than merely to
    survive somewhere on the line.
    """
    assert autolink_bare_urls(text) == expected


def test_a_url_with_balanced_parentheses_survives_whole() -> None:
    """They are legal path characters, and the whole URL must be the target."""
    assert autolink_bare_urls("https://en.wikipedia.org/wiki/Foo_(bar)") == (
        "<https://en.wikipedia.org/wiki/Foo_(bar)>"
    )
    assert autolink_bare_urls("see https://en.wikipedia.org/wiki/Foo_(bar).") == (
        "see <https://en.wikipedia.org/wiki/Foo_(bar)>."
    )


# --- what is left alone --------------------------------------------------------


def test_text_without_a_url_is_returned_unchanged() -> None:
    assert autolink_bare_urls("nothing to see here") == "nothing to see here"


def test_a_markdown_link_is_left_alone() -> None:
    """Its target is already a link; bracketing it would nest the delimiters."""
    text = "[the docs](https://a.test/x)"
    assert autolink_bare_urls(text) == text


def test_an_existing_autolink_is_left_alone() -> None:
    text = "read <https://a.test/x> first"
    assert autolink_bare_urls(text) == text


def test_a_reference_definition_is_left_alone() -> None:
    """The definition's target is the link, and the label on the line before it
    resolves to it — brackets around the target break both halves."""
    text = "[ref]: https://a.test/x\n\nsee [ref] for the detail"
    assert autolink_bare_urls(text) == text


def test_an_inline_code_span_is_left_alone() -> None:
    """``curl <https://a.test/x>`` is a sample that no longer runs."""
    text = "run `curl https://a.test/x` to check"
    assert autolink_bare_urls(text) == text


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_a_fenced_code_block_is_left_alone(fence: str) -> None:
    text = f"{fence}\ncurl https://a.test/x\n{fence}\n"
    assert autolink_bare_urls(text) == text


def test_a_fenced_code_block_does_not_mask_the_prose_after_it() -> None:
    """The mask is SCOPED to the fence, or one code sample would make every
    later URL in the message unclickable."""
    text = "```\ncurl https://a.test/x\n```\n\nsee https://a.test/y\n"
    assert autolink_bare_urls(text) == "```\ncurl https://a.test/x\n```\n\nsee <https://a.test/y>\n"


def test_an_unterminated_fence_masks_everything_after_it() -> None:
    """A streaming message whose closing marker has not arrived yet.

    Reading the sample as prose would autolink it mid-flush and then take the
    brackets back when the fence closed — a link that appears and disappears.
    """
    text = "```\ncurl https://a.test/x\nand https://a.test/y\n"
    assert autolink_bare_urls(text) == text
