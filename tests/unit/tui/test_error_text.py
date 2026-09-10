"""The blank-message fallback, in isolation.

The companion to ``test_aside.py::test_an_error_with_no_message_still_names_``
``itself_on_the_card``, not a substitute for it: that test proves the card
calls this, and this one proves the three inputs it has to survive.
"""

from __future__ import annotations

from local_operator.tui.error_text import error_text


def test_error_text_falls_back_to_the_class_name() -> None:
    """A message when there is one; the class name when there is not.

    The whitespace case is separate from the empty case because
    ``str(exc).strip()`` is what makes ``ValueError("   ")`` — which renders as
    a warning glyph followed by blank cells — take the fallback too.
    """
    assert error_text(ValueError("the port is already in use")) == "the port is already in use"
    assert error_text(TimeoutError()) == "TimeoutError"
    assert error_text(ValueError("   ")) == "ValueError"
