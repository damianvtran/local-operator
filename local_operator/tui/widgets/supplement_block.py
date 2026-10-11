"""The block under a final answer: file callouts, then any imageless components.

Turn supplements on the TUI (memo §2.8 block anatomy; §3.2 T1/T3/T4/T5). One
widget paints every state the block may be in, because the memo's whole
interaction story is a line that becomes content IN PLACE:

- **preparing** — the file callouts (if the decision already knows them) and
  the ``Preparing highlights…`` line, whose lead is the shimmering graphic from
  :mod:`local_operator.tui.imagegen`;
- **settled** — the file rows (or the ≤ 2 files one-liner), the "N more"
  disclosure, and nothing else this surface draws: HTML components are skipped
  SILENTLY and with no placeholder — see :meth:`_build` for why that is the
  documented design point and not an omission;
- **cancelled / failed** — the callouts, then the one quiet line.

NO header, label or title chrome, on purpose (the 2026-10-10 amendment): the
block starts directly with its content and position alone carries its
relationship to the answer above; spacing does the seam. The widget also
carries NO pointer-only action pair (the memo's ``· Adjust… · Cancel``):
``ctrl+c`` is the turn interrupt and stays unbound, and the TUI's actions are
the ``/highlights`` slash commands — a terminal line advertising a click it
cannot honour is the D10 defect in the other direction.

A file row is ONE clickable affordance: the label and the session-relative path
carry a ``file:`` link admitted by
:func:`local_operator.tui.link_targets.file_affordance` — the SAME predicate
the click re-checks before Reveal-in-Finder, so paint-time gating and
open-time gating cannot drift. Where the terminal is not known to render OSC 8
(:func:`local_operator.terminals.is_hyperlinked`), the same row paints as plain
text: the path stays visible and copyable, and no link is written for a
terminal that would drop it.

The "N more" row expands in place on a click, bounded by the stored list
(``more[]`` is capped at 20 at write time). It is the block's only click I/O
that is not a file link; every state line is inert BY DESIGN (see above).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from local_operator.supplements.contract import reader_disposition
from local_operator.terminals import is_hyperlinked
from local_operator.tui import supplements as supplement_mod
from local_operator.tui import theme as theme_mod
from local_operator.tui.link_targets import file_affordance
from local_operator.tui.widgets.transcript import (
    SPINE_INDENT,
    TranscriptBlock,
    TranscriptView,
)

#: The shimmer cadence: the image-gen card's own 33 ms frame, one constant per
#: surface so the two "working" animations cannot run at different rates.
_FRAME_MS = 33.0


class SupplementBlock(TranscriptBlock):
    """One supplement row's out on the transcript. See the module docstring."""

    #: A change of subject against the answer above: the adaptive spacing rule
    #: gives the block a gap above (and below the next block), which IS the
    #: seam the amendment describes — spacing and rhythm, never a label.
    SPACING_KIND: ClassVar[str] = "supplement"

    def __init__(
        self,
        details: Mapping[str, Any],
        *,
        job_live: bool = False,
        fold_width: int = 0,
        cwd: str | None = None,
        hyperlinks: bool | None = None,
    ) -> None:
        super().__init__()
        self.add_class("supplement-block")
        self._details: dict[str, Any] = dict(details)
        #: Whether THIS runtime's registry holds the job (the stale-row reader
        #: rule). A live event implies True; a cold read implies False and a
        #: non-terminal row then paints ``Highlights cancelled``.
        self._job_live = bool(job_live)
        #: The session cwd, for the file affordance's relative resolution.
        self._cwd = cwd
        #: Tri-state: None detects the terminal, True/False are the test and
        #: capture overrides that pin the branch without an environment dance.
        self._hyperlinks = hyperlinks
        self._expanded = False
        self._frame_ms = 0.0
        self._frame_timer: Any = None
        self._built_width = -1
        self._last_lane = -1
        self._applied_rows = -1
        #: ``{file URI: the row's own path}`` for every link this build painted:
        #: the click hands BOTH to the opener so the open-time check can re-run
        #: the affordance on the same input the paint-time check admitted
        #: (``file_affordance`` refuses a bare absolute path by contract).
        self._painted_links: dict[str, str] = {}
        #: The painted line index of the "N more" row (None when absent), for
        #: the click hit test. Line-indexed because every row is truncated to
        #: one line, so widget y maps to a line exactly.
        self._more_line: int | None = None
        self.set_fold_hint(fold_width)
        self._rebuild()
        self.finalize()

    # -- identity ----------------------------------------------------------

    @property
    def anchor_id(self) -> str:
        """The final answer's message id this block hangs under.

        Named ``anchor_id`` rather than ``anchor``: Textual's ``Widget`` already
        uses ``anchor``, and shadowing it with a different type is the override
        pyright refuses (``reportIncompatibleMethodOverride``).
        """
        return str(self._details.get("anchor", ""))

    @property
    def job(self) -> str:
        return str(self._details.get("job", ""))

    def disposition(self) -> str:
        """The frozen reader disposition for this row (§2.8, contract.py)."""
        return reader_disposition(self._details, job_live=self._job_live)

    def set_details(self, details: Mapping[str, Any], *, job_live: bool | None = None) -> None:
        """Adopt a newer version of the row (the live event / read result).

        The rows are rebuilt as a pure function of the new details, so a swap
        is idempotent: replaying the same event twice paints identically, which
        is what lets the app feed events without de-duplicating first.
        """
        self._details = dict(details)
        if job_live is not None:
            self._job_live = bool(job_live)
        self._rebuild()

    def set_job_live(self, live: bool) -> None:
        if self._job_live != live:
            self._job_live = bool(live)
            self._rebuild()

    # -- rendering ---------------------------------------------------------

    def _hyperlinks_enabled(self) -> bool:
        if self._hyperlinks is not None:
            return self._hyperlinks
        return is_hyperlinked()

    def _rebuild(self, width: int | None = None) -> None:
        """Rebuild the rows and re-pin the height (the finalized-block dance)."""
        was_finalized = self._finalized
        self._finalized = False
        try:
            content = self._build(width)
        finally:
            self._finalized = was_finalized
        rows = len(content.plain.splitlines()) if content.plain else 0
        moved = rows != self._applied_rows
        self._applied_rows = rows
        self._built_width = self._last_lane
        self._set_authored_height(rows)
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(content, layout=moved)
        finally:
            self._finalized = was_finalized
        self._sync_frame_timer()
        parent = self.parent
        if isinstance(parent, TranscriptView):
            parent.refresh_gap_around(self)

    def _build(self, width: int | None = None) -> Text:
        lane = width if width is not None and width > 0 else self.fold_width(80)
        self._last_lane = lane
        body = max(lane - SPINE_INDENT, 12)
        disposition = self.disposition()
        files = self._files()
        more_count = self._more_count(files)
        components = self._details.get("components") or []
        images = self._details.get("images") or []

        lines: list[Text] = []
        self._more_line = None
        self._painted_links = {}
        if disposition == "nothing":
            # Nothing renders: a superseded or dismissed row hides the whole
            # block, and the app removes it from the transcript.
            return Text("")

        show_line = disposition in (
            "preparing",
            "files_preparing",
            "cancelled_retry",
            "files_cancelled_retry",
            "failed_retry",
            "files_failed_retry",
        )
        # The files half first (memo §0: "files first, then components"). The
        # ≤ 2 files one-liner applies when no COMPONENT (HTML or image) rides
        # the row and the whole known set is two files or fewer; it serves the
        # preparing arm's callouts as well, which are the same callouts a
        # moment earlier.
        components_present = bool(components) or bool(images)
        total_files = len(files) + more_count
        if files or more_count:
            if total_files <= 2 and not components_present:
                names: list[str] = []
                for file in files:
                    name, path, _size = supplement_mod.file_row_parts(file)
                    if name or path:
                        names.append(name or path)
                if more_count:
                    names.extend(
                        supplement_mod.display_text(str(item).rsplit("/", 1)[-1])
                        for item in self._more_paths()[:more_count]
                    )
                lines.append(
                    self._line(
                        supplement_mod.files_summary_line([n for n in names if n][:2]),
                        style=self._muted_style(),
                    )
                )
            else:
                file_rows = self._file_rows(body, files)
                lines.extend(file_rows)
                lines.extend(self._more_rows(body, more_count, base_index=len(file_rows)))

        if show_line:
            line = self._state_line(disposition)
            if line is not None:
                lines.append(line)
        # HTML components (and CSS/JS) are SILENTLY SKIPPED with NO placeholder
        # — the operator's requirement, not an omission: this surface cannot
        # run them, a dashed box or a "component available in the app" note
        # would be chrome the block's amendment forbids, and a stub that admits
        # it cannot render the thing is worse than the thing simply not being
        # there. Image entries are NOT handled here: they mount as real
        # ImageBlocks beside this one (one transport, `append_image_blocks`).
        content = Text()
        for index, line in enumerate(lines):
            if index:
                content.append("\n")
            content.append(self._fit(line, body))
        return content

    def _state_line(self, disposition: str) -> Text | None:
        """The one quiet line for a non-settled state (§2.8 copy table)."""
        if disposition in ("preparing", "files_preparing"):
            return supplement_mod.preparing_line(self._frame_ms)
        if disposition in ("cancelled_retry", "files_cancelled_retry"):
            return self._line(supplement_mod.COPY_CANCELLED, style=self._muted_style())
        if disposition in ("failed_retry", "files_failed_retry"):
            return self._line(supplement_mod.COPY_FAILED, style=self._muted_style())
        return None

    def _file_rows(self, body: int, files: Sequence[Mapping[str, Any]]) -> list[Text]:
        """One clickable row per featured file (memo §3.2 T3)."""
        rows: list[Text] = []
        hyperlinked = self._hyperlinks_enabled()
        for file in files:
            name, path, size = supplement_mod.file_row_parts(file)
            raw_path = str(file.get("path", ""))
            uri = file_affordance(raw_path, cwd=self._cwd) if hyperlinked and raw_path else None
            link_style = self._link_style(uri) if uri else None
            row = self._line("", style=None)
            text = name or path
            row.append(text, style=link_style)
            if uri:
                self._painted_links[uri] = raw_path
            if path and path != text:
                row.append("  ")
                row.append(path, style=self._path_style(link_style))
            if size:
                row.append(f"  {size}", style=self._dim_style())
            rows.append(self._fit(row, body))
        return rows

    def _more_rows(self, body: int, more_count: int, *, base_index: int) -> list[Text]:
        """The disclosure row, or the stored paths once expanded (≤ 20).

        ``base_index`` counts the rows already authored above, so the collapsed
        row's painted line is recorded for the click hit test without rebuilding
        the file rows a second time.
        """
        if more_count <= 0:
            return []
        if not self._expanded:
            self._more_line = base_index
            text = supplement_mod.more_text(more_count)
            return [self._fit(self._line(text, style=self._more_style()), body)]
        rows: list[Text] = []
        hyperlinked = self._hyperlinks_enabled()
        for item in self._more_paths()[:20]:
            raw = str(item)
            shown = supplement_mod.display_text(raw)
            label = supplement_mod.display_text(raw.rsplit("/", 1)[-1]) or shown
            uri = file_affordance(raw, cwd=self._cwd) if hyperlinked else None
            link_style = self._link_style(uri) if uri else None
            row = self._line("", style=None)
            row.append(label, style=link_style)
            if uri:
                self._painted_links[uri] = raw
            if shown and shown != label:
                row.append("  ")
                row.append(shown, style=self._path_style(link_style))
            rows.append(self._fit(row, body))
        return rows

    def _line(self, text: str, *, style: Style | None) -> Text:
        row = Text(no_wrap=True, overflow="ellipsis")
        row.append(" " * SPINE_INDENT)
        if text:
            row.append(text, style=style)
        return row

    def _fit(self, row: Text, body: int) -> Text:
        """Truncate a row to the block's lane (indent included), styles aligned.

        Walks ``row.plain`` from character 0 with a span→style lookup rather
        than iterating ``row.spans``: an unstyled head (the indent, and the
        file NAME whenever the row carries no link) has no span at all, so a
        spans-only walk emitted just the styled tail — an overflowing row
        lost exactly the characters that identify the file (design round 1,
        D1). Spans supply style, never content: every character of ``plain``
        is emitted in order, unstyled gaps as base-styled pieces, and the
        truncation eats the row's tail.

        Row-by-row truncation rather than wrapping is deliberate: the click hit
        test maps widget ``y`` to an authored line, and one authored line must
        stay one terminal row for that mapping to hold.
        """
        from local_operator.tui.widgets.tool_card import truncate_cells

        width = body + SPINE_INDENT
        plain = row.plain
        if cell_len(plain) <= width:
            return row
        out = Text(no_wrap=True, overflow="ellipsis")
        length = len(plain)
        spans = row.spans
        used = 0
        cursor = 0
        span_index = 0
        while cursor < length and used < width:
            span = spans[span_index] if span_index < len(spans) else None
            if span is not None and span.end <= cursor:
                # Unreachable for rows built by ordered appends; the guard
                # keeps a malformed span list from rewinding the cursor.
                span_index += 1
                continue
            if span is not None and span.start <= cursor:
                end, style = min(span.end, length), span.style
                span_index += 1
            elif span is not None:
                end, style = min(span.start, length), None
            else:
                end, style = length, None
            chunk = plain[cursor:end]
            room = width - used
            if cell_len(chunk) > room:
                chunk = truncate_cells(chunk, room)
            out.append(chunk, style=style)
            used += cell_len(chunk)
            cursor = end
        return out

    # -- styles (tokens only, never a hex: the design-qa gate's token rule) --

    @staticmethod
    def _dim_style() -> Style:
        return Style(color=theme_mod.semantic_color("dim"))

    @staticmethod
    def _muted_style() -> Style:
        return Style(color=theme_mod.semantic_color("muted"))

    @staticmethod
    def _path_style(link: Style | None) -> Style:
        """The session-relative path's ink: the link's own, else ``muted``.

        ``muted`` rather than ``dim``: the path is read, not scanned, and the
        light ramp's ``dim`` path ink measured 3.77:1 — below the 4.5 AA floor
        (design round 1, D2). ``muted`` reads 7.18:1 light / 8.62:1 dark; the
        size column keeps ``dim``, inside that token's 3.4 floor for
        micro-labels.
        """
        if link is not None:
            return link
        return Style(color=theme_mod.semantic_color("muted"))

    @staticmethod
    def _link_style(uri: str) -> Style:
        """The link ink: ``signal``, the ramp's link/file-path token.

        ``accent`` is the ramp's one green, reserved for the live indicator
        and focus; every other transcript link paints ``signal`` + underline
        (design round 1, D3). Both ramps clear AA — signal 7.40:1 dark /
        4.91:1 light; accent was 8.73 / 4.84 — so the token rule decides.
        """
        return Style(color=theme_mod.semantic_color("signal"), underline=True, link=uri)

    @staticmethod
    def _more_style() -> Style:
        return Style(color=theme_mod.semantic_color("muted"), underline=True)

    # -- data readers --------------------------------------------------------

    def _files(self) -> list[Mapping[str, Any]]:
        files = self._details.get("files")
        if not isinstance(files, list):
            return []
        return [item for item in files if isinstance(item, Mapping)]

    def _more_paths(self) -> list[str]:
        more = self._details.get("more")
        if not isinstance(more, list):
            return []
        return [str(item) for item in more if isinstance(item, str)]

    def _more_count(self, files: Sequence[Mapping[str, Any]]) -> int:
        raw = self._details.get("files_more")
        count = int(raw) if isinstance(raw, int) and not isinstance(raw, bool) else 0
        return max(count, 0) or (len(self._more_paths()) if raw is None else 0)

    # -- interaction ---------------------------------------------------------

    def expand_more(self) -> None:
        """Expand the "N more" row in place (the block's one disclosure)."""
        if self._expanded or self._more_count(self._files()) <= 0:
            return
        self._expanded = True
        self._rebuild()

    def on_click(self, event: Any) -> None:  # type: ignore[no-untyped-def]
        """A click on a file link reveals it; a click on "N more" expands.

        The link case resolves through the composed style exactly as the base
        block does, but routes ``file:`` URIs to the file opener (Reveal in
        Finder) instead of the browser opener. A click the block does not claim
        is left to bubble — the container's focus rescue depends on it.
        """
        url = self.link_at(event.x, event.y)
        if url is not None:
            if url.startswith("file:"):
                opener = getattr(self.app, "open_transcript_file", None)
                if opener is not None:
                    opener(url, source_path=self._painted_links.get(url, ""))
                    event.stop()
                return
            super().on_click(event)
            return
        if self._more_line is not None:
            padding = self.styles.padding.top if self.styles.padding else 0
            if event.y - padding == self._more_line:
                self.expand_more()
                event.stop()

    # -- resize / theme / lifecycle -----------------------------------------

    def on_resize(self, event: Any) -> None:  # type: ignore[no-untyped-def]
        size = getattr(event, "size", None)
        if size is not None and size.width == self._built_width:
            return
        self._rebuild()

    def retheme(self) -> None:
        """Re-ink every row from the current theme (live theme switches)."""
        self._rebuild()

    def on_unmount(self) -> None:
        self._stop_frame_timer()

    # -- the shimmer ---------------------------------------------------------

    def _wants_frames(self) -> bool:
        if not self.is_mounted:
            return False
        return self.disposition() in ("preparing", "files_preparing")

    def _sync_frame_timer(self) -> None:
        from local_operator.tui.animation import motion_enabled

        if not motion_enabled() or not self._wants_frames():
            self._stop_frame_timer()
            return
        if self._frame_timer is not None:
            return
        self._frame_timer = self.set_interval(_FRAME_MS / 1000.0, self._tick_frame)

    def _stop_frame_timer(self) -> None:
        if self._frame_timer is not None:
            self._frame_timer.stop()
            self._frame_timer = None

    def _tick_frame(self) -> None:
        """Advance the band, repaint only the line it lives on."""
        from local_operator.tui.animation import motion_enabled

        if not motion_enabled():
            # Frozen phase, no repaint: a blurred terminal gets no output (the
            # animation module's whole point), and the timer stays armed so the
            # band resumes the moment focus returns.
            return
        self._frame_ms += _FRAME_MS
        self._rebuild()
