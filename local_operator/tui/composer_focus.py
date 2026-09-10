"""One authority for "may this gesture move the keyboard to the composer?".

THE DEFECT THIS MODULE EXISTS TO PREVENT. The composer-focus change added four
routes that hand focus back to the input — a dock click, a transcript click, a
row's ``tab`` binding, and a printable key pressed on a row — and each one
guarded itself with ``editor.can_focus``. That is a strictly WEAKER test than
:meth:`OperatorApp._focus_is_claimed`: ``can_focus`` is False only while the
composer is READ-ONLY (the subagent page, the login prompt), and it is True
while an approval, an ask picker, the aside or the focused sidebar owns the
keyboard. So all four routes were open doors onto a live question, and the MR
shipped with the predicate that answers this question already written and
simply never consulted from any of them.

They are one rule, so they get one implementation. Three independently-guarded
doors is precisely how two of them stayed open through a review round: the
guard was added to the door someone was looking at. A future surface hosted in
the dock is protected here by default rather than by whoever adds it
remembering to copy a predicate check into a fifth handler.

WHY A SEPARATE MODULE, AND WHY IT MUST STAY ONE. ``app.py`` imports
``widgets/transcript.py`` at import time (``app.py:291``), so a helper living
on ``OperatorApp`` — or anywhere else in ``app.py`` — cannot be imported back
from ``transcript.py`` without a circular import. Moving these functions onto
the app class is the obvious-looking tidy-up and it does not work; this module
imports nothing from either side and is safe from both. Leave it here.

WHY ``getattr`` RATHER THAN A TYPED CALL. These handlers run inside widgets
that are mounted in stripped harnesses which host a transcript and no
``OperatorApp`` at all (``tests/unit/tui/conftest.py``, ``test_tool_card.py``,
``test_spacing.py``). The predicate is therefore optional by construction, and
its absence means "this harness has no claim to respect" rather than an error
in a key or mouse path.
"""

from __future__ import annotations

from typing import Any

__all__ = ["focus_is_claimed", "composer_may_take_focus", "return_focus_to_composer"]


def focus_is_claimed(app: Any) -> bool:
    """Whether some surface has a claim on the keyboard the composer must not take.

    Delegates to :meth:`OperatorApp._focus_is_claimed`, which is the real
    definition — approval, ask picker, aside, the three full-page modes, the
    login prompt, the focused sidebar, any pushed screen, and a read-only
    composer. This is a thin, defensive accessor for the widget layer, NOT a
    second copy of that logic: a second copy would drift from the first, and
    the drift would be silent.

    Degrades to ``True`` — "something might be claiming it" — exactly as the
    predicate itself does. Refusing to steal focus is always the safe failure:
    a wrong ``True`` costs the user a gesture that does nothing, while a wrong
    ``False`` costs them a live question they can no longer answer.

    Returns ``False`` when the app has no predicate at all, which is the
    stripped-harness case described in the module docstring.
    """
    predicate = getattr(app, "_focus_is_claimed", None)
    if not callable(predicate):
        return False
    try:
        return bool(predicate())
    except Exception:  # noqa: BLE001 — degrade the way the predicate does
        return True


def composer_may_take_focus(app: Any, editor: Any) -> bool:
    """Whether a "put me back in the input" gesture may act, right now.

    The two halves of the one rule, in the order that makes the cheap, local
    test first:

    ``editor.can_focus`` — refuse rather than steal while the composer is
    inert. Read-only (the subagent page, the login prompt) means it answers no
    key, so focusing it would hand the keyboard to a field that swallows every
    keystroke — the same trap this work exists to remove.

    :func:`focus_is_claimed` — refuse while a surface that took focus ON
    PURPOSE still needs it. A ``multi=True`` ask picker is answered by Space
    and Enter, which the composer would swallow, so it holds the caret
    deliberately; taking it back makes that question unanswerable.
    """
    try:
        if not editor.can_focus:
            return False
    except Exception:  # noqa: BLE001 — a widget mid-teardown is not a claim to steal from
        return False
    return not focus_is_claimed(app)


def return_focus_to_composer(app: Any, editor: Any) -> bool:
    """Focus the composer if this gesture is allowed to. True when it moved.

    Focus and NOTHING ELSE: no text inserted, no caret moved. These gestures
    mean "put me back in the input", and a padding cell or a transcript row has
    no document position to map a caret onto — moving it would cost the user
    the place they were editing to buy them nothing.

    Already-focused is reported as ``False``: nothing moved, so a caller that
    forwards a key after focusing can tell a real transition from a no-op.
    """
    if not composer_may_take_focus(app, editor):
        return False
    try:
        if editor.has_focus:
            return False
        editor.focus()
    except Exception:  # noqa: BLE001 — never raise out of a key or mouse path
        return False
    return True
