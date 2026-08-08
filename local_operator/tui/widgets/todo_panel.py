"""The dock-band todo panel (item 8).

Renders the session's todo list in the shared ``#band`` above the input, so
progress survives compaction AND stays visible without scrolling the
transcript. The todo tool (``local_operator.tools.builtin.execute_todo``) is
the only writer, and it writes to ``builtin.TODO_STORE`` keyed by session id
when the host attaches no ``todos`` dict to the ToolContext — which this
session never does (``Session._build_tool_context`` leaves ``todos=None``).
Reading that module table therefore reads the LIVE state; the import is
deliberately local to the accessor so merely composing the panel cannot pull
the tool registry into a reduced host.

The tool's list is FLAT (``init``/``done``/``view``; items carry ``text`` and
``status`` only — no phases). The row model still routes every item through
one builder (:meth:`TodoPanel._item_row`) so a phase header could drop in
later without restructuring the render — but the tool schema is not extended
to get there.

The panel sits above the input and must stay SHORT: long lists collapse to a
dim ``… N more todos`` line rather than pushing the composer off screen.

Zero height when empty: the panel starts ``display: none`` and only shows
once the store for this session has at least one item; going back to empty
hides it again.
"""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from rich.console import RenderableType
from rich.style import Style
from rich.text import Text
from textual.containers import Container
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod

#: Rows shown before the list collapses to a ``… N more`` line. The panel is
#: chrome above the composer, not the transcript: a twenty-item list rendered
#: in full pushed the input down by a screenful on the exact turn the user is
#: trying to read it. Eight keeps the longest plausible plan on screen and
#: stays out of the way.
MAX_TODO_ROWS = 8


def todo_items(session_id: str) -> list[dict[str, str]]:
    """The live todo list for ``session_id`` (a copy; never the store itself).

    Reads the todo tool's own store — the one writer is the tool, so the table
    IS the state. Copies each item because the tool mutates its dicts in place
    (``done`` flips a status), and a panel holding the originals would repaint
    from a stale snapshot. A missing/empty id or a store the import cannot
    reach degrades to "no todos" — a panel must never take the app down.
    """
    if not session_id:
        return []
    try:
        from local_operator.tools.builtin import TODO_STORE
    except Exception:
        return []
    items = TODO_STORE.get(session_id)
    if not items:
        return []
    return [dict(item) for item in items]


class TodoPanel(Container):
    """The session's todo list, rendered in the dock band above the input.

    The panel is a transparent SLOT (its top padding row is the gap that
    separates it from whatever sits above — a margin would violate the sheet's
    one-margin rule) holding one filled body. Visibility is the panel's own:
    ``display: none`` whenever the store is empty, so the band collapses to
    zero rows.
    """

    def __init__(self) -> None:
        super().__init__(id="todo-panel", classes="band-slot")
        self._body = Static(classes="band-body", id="todo-body")
        #: Fingerprint of what is painted, so the 1 Hz poll repaints only when
        #: the list actually changed — an equality guard, same discipline as
        #: the assistant flush.
        self._shown: tuple[tuple[str, str], ...] | None = None
        # Hidden until the first todo exists: an empty panel is not content.
        self.display = False

    def compose(self):  # type: ignore[override]
        yield self._body

    # -- refresh -------------------------------------------------------------
    def refresh(self, session: Any) -> None:
        """Re-read the store and repaint only on change.

        Called on the todo tool's ``tool_execution_end`` (immediate) AND on the
        app's 1 Hz job poll (the belt to that event's suspenders: a ``done``
        op that settles while the card is still painting still lands here).
        Never raises: a status surface must not be able to take the app down.
        """
        try:
            session_id = getattr(session, "session_id", "") or ""
            items = todo_items(session_id)
            fingerprint = tuple(
                (str(item.get("text", "")), str(item.get("status", "pending")))
                for item in items
            )
            if fingerprint == self._shown:
                return  # equality guard — identical list = no work
            self._shown = fingerprint
            if not fingerprint:
                self.display = False
                return
            self.display = True
            self._body.update(self._build(fingerprint))
        except Exception:
            self.display = False

    # -- rendering -----------------------------------------------------------
    def _build(self, rows: tuple[tuple[str, str], ...]) -> RenderableType:
        # No width budget is threaded through: every row clips itself with
        # the no_wrap/ellipsis pair (one cell-aware model), never a
        # hand-measured count.
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))

        done = sum(1 for _text, status in rows if status == "done")
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append("Todos", style=muted)
        header.append(" · ", style=dim)
        header.append(f"{done}/{len(rows)}", style=dim)

        lines = [header]
        # Every item routes through ONE row builder: the tool's list is flat
        # today, but a phase header later drops in beside this call rather
        # than restructuring the render.
        visible = rows[:MAX_TODO_ROWS]
        for text, status in visible:
            lines.append(self._item_row(text, status))
        hidden = len(rows) - len(visible)
        if hidden > 0:
            overflow = Text(no_wrap=True, overflow="ellipsis")
            overflow.append(f"… {hidden} more todo{'s' if hidden != 1 else ''}", style=dim)
            lines.append(overflow)
        return Text("\n").join(lines)

    def _item_row(self, text: str, status: str) -> Text:
        """One ``- [ ]``/``- [x]`` row — the tool's own vocabulary (its ``view``
        op renders exactly these marks), so the panel and the transcript
        receipt read identically.

        Ink by state: pending is readable ``muted``; done drops to ``dim`` AND
        strikethrough, because a finished item is a record, not an instruction
        — the same "settled things go quiet" trade the tool ledger makes.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        done = status == "done"
        row = Text(no_wrap=True, overflow="ellipsis")
        row.append("- [x] " if done else "- [ ] ", style=dim)
        # Model-controlled text reaches a real terminal: stripped like every
        # other untrusted string this app renders (same discipline as the
        # approval prompt and the tool cards).
        row.append(
            strip_control_sequences(text),
            style=Style(color=theme_mod.semantic_color("dim"), strike=True)
            if done
            else muted,
        )
        return row
