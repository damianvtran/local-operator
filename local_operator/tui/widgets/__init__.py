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
from local_operator.tui.widgets.command_picker import CommandPicker
from local_operator.tui.widgets.editor import (
    Editor,
    EditorQuit,
    EditorSubmitted,
    InterruptRequested,
    ShellModeChanged,
)
from local_operator.tui.widgets.status_line import StatusLine
from local_operator.tui.widgets.toast import Toast
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptBlock,
    TranscriptView,
    UserBlock,
)

__all__ = [
    "AssistantBlock",
    "CommandPicker",
    "Editor",
    "EditorQuit",
    "EditorSubmitted",
    "InterruptRequested",
    "ShellModeChanged",
    "NoticeBlock",
    "StatusLine",
    "Toast",
    "ToolCard",
    "TranscriptBlock",
    "TranscriptView",
    "UserBlock",
]
