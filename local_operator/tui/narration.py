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

    Either signal alone is sufficient. ``tool_calls`` is the direct evidence
    that the turn continues after this message; ``stop_reason == "toolUse"`` is
    the provider saying the same thing, and providers exist that report one
    without the other.

    Every OTHER stop reason is FINAL and must classify as False: ``stop``,
    ``length``, ``refusal``, ``error`` and ``aborted`` (the vocabulary is at
    ``harness/types.py``) each mean this message is the last thing the user
    sees. Treating one of them as narration would erase the outcome of the
    turn — a refusal, or a half sentence the user needs to see was cut off.
    """
    return bool(has_tool_calls) or stop_reason == "toolUse"
