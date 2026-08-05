"""Transcript widgets for the Local Operator TUI.

The transcript is the product's centre: assistant markdown, one-line tool
cards, and notices all stack into a single scrollable column. The widgets in
this package share two invariants:

- Blocks own NO outer margin; the container owns inter-block gaps (exactly
  one separator behavior, never blank filler rows).
- Finalized blocks are immutable: once ``is_finalized()`` is true the
  container never re-renders them.
"""

from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import (
    Editor,
    EditorQuit,
    EditorSubmitted,
    InterruptRequested,
)
from local_operator.tui.widgets.status_line import StatusLine
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptBlock,
    TranscriptView,
    UserBlock,
)

__all__ = [
    "AssistantBlock",
    "Editor",
    "EditorQuit",
    "EditorSubmitted",
    "InterruptRequested",
    "NoticeBlock",
    "StatusLine",
    "ToolCard",
    "TranscriptBlock",
    "TranscriptView",
    "UserBlock",
]
