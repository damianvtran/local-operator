"""The whole community post an agent asks to publish, before the owner may Allow it.

WHY (review R1-5, RFC T6: the card "never approves text you haven't read in
full"). A chief of staff's ``community_post`` card can carry up to 4000
characters (``agent_ops.MAX_TEXT_CHARS``), and the approval card bounds its
question to whatever the dock can spare: at 80x24 that is five rows, which is
the header alone. So the card cannot be where a long post is read. This screen
is: the card's header (title, audience, "shown as your own post", why it asks)
is PINNED at the top, the post text SCROLLS beneath it, and the keys line is
pinned at the bottom — the header can never be scrolled away from the text it
describes.

THE RULE IT FEEDS. The screen dismisses with whether the reader reached the
END of the text (a body that fits counts as reached on its first frame).
``ApprovalPrompt`` accepts ``y`` on a cut post card only after that; until
then ``y`` opens this screen instead of approving. It never answers the gate
itself: Allow / Deny stay on the card, so there is one answer path.

Keys: ``↑↓``/``pgup``/``pgdn``/``home``/``end`` scroll; ``esc``, ``q`` or ``v``
go back to the card. ``y``/``n`` do nothing here (a habitual double-press of
``y`` must not approve through a screen that is still showing the top).
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.i18n import keys as i18n_keys
from local_operator.tui.i18n_status import display_tr

_k = i18n_keys.tui_community
_t = display_tr("tui.community")

#: The card's body rows carry this prefix (``agent_ops.post_card``); every
#: other row is header. Splitting on it keeps the card text the single source.
BODY_PREFIX = "┊ "


def split_card(card: str) -> tuple[list[str], list[str]]:
    """``(header lines, body lines)`` of a post card's question text."""

    header: list[str] = []
    body: list[str] = []
    for line in card.splitlines():
        (body if line.startswith(BODY_PREFIX) or body else header).append(line)
    return header, body


class CommunityPostScreen(ModalScreen[bool]):
    """The full post: pinned header, scrolling text, pinned keys."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("q", "back", "Back", show=False),
        Binding("v", "back", "Back", show=False),
        Binding("up", "scroll_by(-1)", "Up", show=False),
        Binding("down", "scroll_by(1)", "Down", show=False),
        Binding("pageup", "page(-1)", "Page up", show=False),
        Binding("pagedown", "page(1)", "Page down", show=False),
        Binding("space", "page(1)", "Page down", show=False),
        Binding("home", "edge(0)", "Top", show=False),
        Binding("end", "edge(1)", "Bottom", show=False),
    ]

    def __init__(self, card: str) -> None:
        super().__init__()
        self._header, self._body_lines = split_card(card)
        #: Whether the end of the text has been on screen. Monotonic: scrolling
        #: back up after reading to the end does not un-read it.
        self.read_to_end = False

    def compose(self) -> ComposeResult:
        with Container(classes="analytics-panel"):
            yield Static(Text("\n".join(self._header), style="bold"), id="post-view-head")
            with VerticalScroll(id="post-view-scroll") as scroll:
                self._scroll = scroll
                yield Static(Text("\n".join(self._body_lines)), id="post-view-body")
            self._hint = Static(self._hint_text(), id="post-view-hint")
            yield self._hint

    def on_mount(self) -> None:
        self.watch(self._scroll, "scroll_y", self._note_position, init=False)
        # A body that fits is read on its first laid-out frame.
        self.call_after_refresh(self._note_position)

    def _note_position(self, *_: object) -> None:
        scroll = self._scroll
        if not scroll.is_mounted or not scroll.size.height:
            return
        if scroll.scroll_y >= scroll.max_scroll_y and not self.read_to_end:
            self.read_to_end = True
            self._hint.update(self._hint_text())

    def _hint_text(self) -> Text:
        maker = _k.post_view_keys_read if self.read_to_end else _k.post_view_keys
        return Text(_t(maker()), style="dim")

    def action_scroll_by(self, rows: int) -> None:
        self._scroll.scroll_relative(y=rows, animate=False)

    def action_page(self, direction: int) -> None:
        height = max(1, self._scroll.size.height - 1)
        self._scroll.scroll_relative(y=direction * height, animate=False)

    def action_edge(self, end: int) -> None:
        if end:
            self._scroll.scroll_end(animate=False)
        else:
            self._scroll.scroll_home(animate=False)

    def action_back(self) -> None:
        self.dismiss(self.read_to_end)


__all__ = ["BODY_PREFIX", "CommunityPostScreen", "split_card"]
