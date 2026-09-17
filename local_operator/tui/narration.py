"""Mid-turn narration classification — ONE rule for the live and replay paths.

A single agentic turn is many model calls. Every call that ends in tool calls
streamed prose first ("Let me check the config first."), and that prose is
rendered by exactly the same block as the final answer — so a settled
transcript interleaves thinking with the outcome and the user cannot tell which
paragraph is the answer. The ``display.narration`` flag lets a user drop the
mid-turn prose once its tools have run, leaving ``user -> tools -> answer``.

The classification lives HERE, in its own module, because two separate surfaces
have to make it identically: the live path removes the block at finalize
(``tui/app.py::on_assistant_message_end``) and the replay path never mounts it
(``tui/session_presentation.py::project_settled_rows``). A second copy of the
rule is how a resumed session comes to disagree with the live one about what it
showed — the live/replay divergence this repo has already paid for in review.

This module is deliberately TUI-local and import-cheap: no session, no config,
no harness import. ``harness/rows.py`` owns row decisions both the TUI and the
phone projection must make identically; this is a user DISPLAY PREFERENCE read
from ``tui/settings.py``, which ``rows.py`` must never import.
"""

#: Default for ``display.narration``. True = show narration = today's
#: behaviour, so the toggle is opt-in and an untouched config renders exactly
#: as it shipped. Read by the live path when config is unavailable; the
#: registry in ``settings_io`` is pinned against this constant by test.
DEFAULT_NARRATION = True


def is_intermediate_narration(*, stop_reason: str | None, has_tool_calls: bool) -> bool:
    """Is this finalized assistant message mid-turn narration rather than the answer?

    ``tool_calls`` is the whole rule, and ``stop_reason`` deliberately does NOT
    corroborate it on its own. Hiding narration is only justified because tool
    activity FOLLOWS it and supersedes it on screen; with no calls there is
    nothing to supersede it, so the prose is the only thing the turn produced
    and it must stay.

    ``stop_reason == "toolUse"`` with an EMPTY ``tool_calls`` is reachable, not
    hypothetical: ``providers/clients.py`` maps ``finish_reason`` to
    ``stop_reason`` before the calls are assembled, and ``harness/loop.py``
    assigns the two independently, so a provider that reports
    ``finish_reason=tool_calls`` whose arguments then fail to assemble produces
    exactly this pair. Accepting either signal removed the prose, the
    ``for call in tool_calls`` loop mounted nothing, and
    ``assistant_stop_notice`` returns None for ``toolUse`` — leaving the user's
    prompt followed by SILENCE (review MAJOR-1).

    Every stop reason is therefore irrelevant here, and the terminal ones
    (``stop``, ``length``, ``refusal``, ``error``, ``aborted`` —
    ``harness/types.py``) stay False for the same underlying reason: this
    message is the last thing the user sees, and removing it would erase the
    outcome of the turn.
    """
    return bool(has_tool_calls)
