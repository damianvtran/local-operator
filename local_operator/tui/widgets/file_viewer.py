"""Full-page read-only preview; the worker and descriptor live with the page.

Unlike an editor, this page cannot write and never interprets HTML. Rich's
Markdown and Syntax renderers are local presentation only. One scroll owner
clamps every movement, including arrows, just like the other full-page modes.
"""
from __future__ import annotations

from pathlib import Path

from rich.cells import cell_len
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.message import Message
from textual.widgets import Static
from textual.worker import Worker

from local_operator.i18n.keys import tui_viewer as msg
from local_operator.i18n.text import tr
from local_operator.tui.attachments import open_path_quietly
from local_operator.tui.file_viewer import (
    Document, Identity, ViewerError, lexer_for, read_document_async,
)


_t = tr("tui.viewer")


class FileViewerDismissed(Message):
    """Esc returns focus to the surface that opened the page."""


class FileViewer(Vertical, can_focus=True):
    DEFAULT_CSS = """
    FileViewer { height: 1fr; width: 1fr; padding: 0 1; }
    FileViewer > .viewer-header { height: auto; max-height: 3; color: $text; text-style: bold; }
    FileViewer > .viewer-status { height: auto; max-height: 3; color: $warning; }
    FileViewer > .viewer-body { height: 1fr; width: 1fr; overflow: auto auto; }
    FileViewer .viewer-content { height: auto; width: 1fr; }
    FileViewer > .viewer-legend { height: auto; color: $text-muted; }
    """
    BINDINGS = [
        Binding("escape", "dismiss", show=False, priority=True),
        Binding("w", "wrap", show=False), Binding("r", "reload", show=False),
        Binding("y", "copy_path", show=False), Binding("o", "open_external", show=False),
        Binding("up", "move('up')", show=False), Binding("down", "move('down')", show=False),
        Binding("left", "move('left')", show=False), Binding("right", "move('right')", show=False),
        Binding("home", "move('home')", show=False), Binding("end", "move('end')", show=False),
        Binding("pageup", "move('pageup')", show=False),
        Binding("pagedown", "move('pagedown')", show=False),
    ]

    def __init__(self, raw: str, *, cwd: Path, expected: Identity | None = None) -> None:
        super().__init__(id="file-viewer")
        self.raw = raw
        self.cwd = cwd
        self.expected = expected
        self.document: Document | None = None
        self.wrapped = False
        self._read_worker: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        yield Static(self.raw, classes="viewer-header", markup=False)
        yield Static(_t(msg.loading()), classes="viewer-status", markup=False)
        with ScrollableContainer(classes="viewer-body", can_focus=False):
            yield Static(classes="viewer-content", markup=False)
        yield Static(_t(msg.legend()), classes="viewer-legend", markup=False)

    def on_mount(self) -> None:
        self.focus()
        self.action_reload()

    def cancel_read(self) -> None:
        if self._read_worker is not None:
            self._read_worker.cancel()

    def on_unmount(self) -> None:
        self.cancel_read()

    def action_dismiss(self) -> None:
        self.cancel_read()
        self.post_message(FileViewerDismissed())

    def action_reload(self) -> None:
        self.cancel_read()
        self.query_one(".viewer-status", Static).update(_t(msg.loading()))
        self._read_worker = self.run_worker(self._read(), group="viewer-read", exclusive=True)

    async def _read(self) -> None:
        try:
            document = await read_document_async(self.raw, cwd=self.cwd, expected=self.expected)
        except ViewerError as exc:
            errors = {"changed": msg.changed, "binary": msg.binary,
                      "unsupported": msg.unsupported, "unreadable": msg.unreadable}
            self.document = None
            self.query_one(".viewer-content", Static).update("")
            self.query_one(".viewer-status", Static).update(_t(errors[str(exc)]()))
            return
        first_read = self.document is None
        self.document = document
        # Listing identity is a one-shot admission check. Explicit reload reads
        # the current file, but still revalidates the same original affordance.
        self.expected = None
        if first_read:
            self.wrapped = document.kind != "code"
        kinds = {"code": msg.kind_code, "markdown": msg.kind_markdown, "text": msg.kind_text}
        self.query_one(".viewer-header", Static).update(
            _t(msg.header(path=str(document.path), kind=_t(kinds[document.kind]()),
                          size=str(document.size)))
        )
        self.query_one(".viewer-status", Static).update(
            _t(msg.truncated()) if document.truncated else ""
        )
        self._paint_document()

    def _paint_document(self) -> None:
        doc = self.document
        if doc is None:
            return
        content = self.query_one(".viewer-content", Static)
        if doc.kind == "markdown":
            renderable = Markdown(doc.text)
        elif doc.kind == "code":
            renderable = Syntax(
                doc.text, lexer_for(doc.path), line_numbers=True, word_wrap=self.wrapped,
                theme="ansi_dark", background_color="default",
            )
        else:
            renderable = Text(doc.text, no_wrap=not self.wrapped, overflow="fold")
        # A nowrap Static still needs its *virtual* width. Without this the
        # viewport clips the source but has no horizontal distance to scroll.
        width = max((cell_len(row.expandtabs(4)) for row in doc.text.splitlines()), default=0)
        gutter = len(str(len(doc.text.splitlines()) or 1)) + 3 if doc.kind == "code" else 0
        content.styles.width = "1fr" if self.wrapped else max(1, width + gutter)
        content.update(renderable)

    def action_wrap(self) -> None:
        if self.document is None:
            return
        self.wrapped = not self.wrapped
        self._paint_document()
        self.query_one(".viewer-status", Static).update(
            _t(msg.truncated()) if self.document and self.document.truncated
            else _t(msg.wrap_on() if self.wrapped else msg.wrap_off())
        )

    def action_move(self, direction: str) -> None:
        body = self.query_one(".viewer-body", ScrollableContainer)
        if direction == "home":
            body.scroll_to(x=0, y=0, animate=False)
        elif direction == "end":
            body.scroll_to(x=body.max_scroll_x, y=body.max_scroll_y, animate=False)
        else:
            dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
                      "pageup": (0, -body.size.height),
                      "pagedown": (0, body.size.height)}[direction]
            body.scroll_relative(x=dx, y=dy, animate=False)

    def action_copy_path(self) -> None:
        candidate = (
            Path.home() / self.raw[1:].lstrip("/") if self.raw.startswith("~")
            else self.cwd / self.raw
        )
        path = str(self.document.path) if self.document else str(candidate)
        status = self.query_one(".viewer-status", Static)
        try:
            # App.copy_to_clipboard is the existing OSC52 path; no subprocess,
            # platform clipboard dependency or new escape-sequence spelling.
            self.app.copy_to_clipboard(path)
            driver = self.app._driver
            if driver is None or driver.is_headless:
                status.update(_t(msg.copy_failed(path=path)))
            else:
                status.update(_t(msg.copied()))
        except Exception:  # noqa: BLE001 — retain the actionable path on failure
            status.update(_t(msg.copy_failed(path=path)))

    def action_open_external(self) -> None:
        self.run_worker(self._open_external(), group="viewer-open", exclusive=True)

    async def _open_external(self) -> None:
        # Recheck even after a successful preview: a path may have changed while
        # the reader was inspecting it. The OS is never given shell text.
        from local_operator.tui.link_targets import file_affordance
        import os
        from urllib.parse import unquote, urlparse

        spelling = os.path.relpath(self.raw, self.cwd) if Path(self.raw).is_absolute() else self.raw
        uri = file_affordance(spelling, cwd=self.cwd)
        ok = uri is not None and await open_path_quietly(unquote(urlparse(uri).path))
        self.query_one(".viewer-status", Static).update(
            _t(msg.open_failed()) if not ok
            else _t(msg.truncated()) if self.document and self.document.truncated else ""
        )
