"""``/info`` — what this install is and what is running on this machine.

Structurally this is ``/session``'s screen (title / scrolling body / hint, a
``_Body`` accumulator of ``kv`` rows, push-the-screen-then-fill-from-a-worker)
wearing ``/analytics``' section vocabulary (``▌`` headers, ``└`` nesting, the
dim note column). Both are imported rather than reimplemented: three diagnostics
screens with three header glyphs or three ways of saying "unavailable" would be
a defect, not a variation.

**Why a full screen rather than an overlay card.** The content is ~50 rendered
lines. Against the measured card viewports (11 rows at 80x24, 16 at 100x30, 25
at 150x40) that is 4.6 / 3.2 / 2.0 screens, so it does not fit as a ``/usage``
-style popup at ANY width and must scroll everywhere. It is deliberately one
continuous scroll rather than tabs: a user copying a bug report needs one linear
document, and §7's copy would otherwise disagree with the paint about order.

**Three deliberate departures from ``/session``:**

1. No bars and no ``t`` toggle — every value here is a fact, not a magnitude,
   so there is nothing to plot.
2. A wider, flexible value column. ``/session``'s ``_VALUE_CELL = 11`` is sized
   for ``1.7M tokens``; these values are paths and model ids up to 48 cells.
3. A copy affordance, because issue reporting is this screen's stated purpose.
   ``/session`` has none.
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass, field
from typing import Sequence

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.info.collect import LiveState
from local_operator.info.model import (
    UNKNOWN,
    InfoSnapshot,
    format_bytes,
    format_duration,
    is_shadowed_install,
)
from local_operator.info.render import build_export
from local_operator.tui.widgets.analytics_panel import (
    _row_prefix,
    section_header,
    semantic_style,
)
from local_operator.tui.widgets.aside_panel import ASIDE_COPY_KEY
from local_operator.tui.widgets.session_picker import (
    ATTACHED_MARKER,
    IDLE_MARKER,
    WEDGED_MARKER,
)
from local_operator.tui.widgets.tool_card import truncate_cells
from local_operator.tui.widgets.transcript import NOTICE_GLYPHS
from local_operator.tui.widgets.welcome import _fit_tail, _shorten_home

#: The copy chord. Deliberately the SAME constant the aside already binds rather
#: than a second ``"ctrl+r"`` literal: "copy this surface out" should be one
#: gesture across the app, and importing the constant is what stops the two
#: drifting. ``ASIDE_COPY_KEY``'s own docstring carries the reason a printable
#: character cannot work here (the composer holds focus by contract and
#: ``TextArea`` consumes the character first) and why ``ctrl+y`` — the obvious
#: alternative — is unavailable, being TextArea's Redo. ``ctrl+e`` was the other
#: candidate and is already ``toggle_reveal`` on the ask picker.
INFO_COPY_KEY = ASIDE_COPY_KEY

#: Label column. ``/session`` uses 22; the longest label here is
#: ``Subagent capacity`` (17), so 20 fits every label with slack and returns two
#: cells to the value column, which is the column under pressure on this screen.
_LABEL_CELL = 20

#: Cells the label column, its leading indent and the gap before the note
#: consume, so the value TRUNCATION budget is ``card - _ROW_OVERHEAD``. Derived
#: as one constant rather than restated at each call site, following
#: ``_PCT_CELL``'s rule in ``session_panel``: a change to the label column must
#: not silently reintroduce a crop somewhere else.
_ROW_OVERHEAD = 2 + _LABEL_CELL + 2

#: MINIMUM width the value is padded to, so the note column lines up on the
#: common row — not a maximum. This is ``/session``'s ``_VALUE_CELL`` model and
#: the distinction is load-bearing: padding every value out to its full
#: truncation budget (``card - 24``) consumes the entire row, leaving a note
#: budget of ZERO at every width, so no ``notes=`` rung ever fits and every
#: qualifier silently disappears. Sized at 24 to fit the widest COMMON value
#: (``anthropic/claude-opus-5`` is 23); a longer value — a deep path — pushes
#: its own note right rather than everyone else's note away, which is the
#: ragged-but-honest behaviour ``/session`` already has.
_VALUE_MIN = 24

#: Minimum label width on a GLYPH-led row (session lines, subagent tree nodes),
#: so their meta forms a column instead of being flushed to the card's right
#: edge. Right-flushing looked fine at 65 cells and fell apart at 128, where it
#: left ~90 cells of gap between a four-word label and its own status — the two
#: halves of one row stopped reading as one row. A column keeps the association
#: local at every width, and a longer label pushes its OWN meta right rather
#: than everyone else's away, which is the same ragged-but-honest rule
#: :data:`_VALUE_MIN` applies to the ``kv`` rows above.
#:
#: 30 is sized from the narrow floor, not the wide case: at a 65-cell card
#: (80 columns) the widest realistic session meta is ``this session · 14m ·
#: 228 MB`` at 27 cells, and 2 + 2 + 30 + 27 = 61 still fits.
_MARK_LABEL_CELL = 30

#: Below this the plain ``note=`` column sheds whole and section metas fall back
#: to their short form. Same value and same reasoning as ``session_panel``'s: at
#: 50 columns a note pushes the row past the viewport and folds onto its own
#: line, where it reads as a separate record.
_NOTE_MIN = 60

_DEFAULT_CARD_WIDTH = 88
_MIN_CARD_WIDTH = 38

#: How deep the tree is drawn before the remainder folds into one summary row.
#: Matches ``info.collect.MAX_TREE_DEPTH``.
_MAX_TREE_DEPTH = 3

#: Status → (glyph, ink). Every one of these glyphs already exists in the app;
#: none is invented here. Colour is never the only carrier — each state has its
#: own glyph — because this is a screen people SCREENSHOT into bug reports, and
#: colour is what disappears under ``NO_COLOR``, a weak-dim theme, or low
#: vision. That is ``_row_prefix``'s stated rule, applied where it matters most.
_SUBAGENT_MARKS: dict[str, tuple[str, str]] = {
    "running": (IDLE_MARKER, "accent"),
    "starting": (IDLE_MARKER, "accent"),
    "queued": ("⏳", "dim"),
    "pausing": ("⏸", "muted"),
    "paused": ("⏸", "muted"),
    "completed": (NOTICE_GLYPHS["success"], "dim"),
    "failed": (NOTICE_GLYPHS["error"], "danger"),
    "cancelled": ("⊘", "dim"),
    "gone": ("⊘", "dim"),
}


def _fit_pair(label: str, metas: Sequence[str], width: int, lead: int) -> str | None:
    """The widest meta rung that fits BESIDE ``label``, or ``""`` if none does.

    ``None`` means the label itself does not fit even with the meta shed, which
    is the caller's signal to try a shorter label rung (or drop the row). The
    empty string means the label fits but no meta does — a warning without its
    qualifier still carries the fact, which is the whole point of shedding.

    NO FALLBACK CROP, inherited deliberately from the `_first_that_fits` ladder
    this replaced: returning a cropped shortest rung made the old helper always
    "succeed", so a caller could not tell whether anything actually fit — it got
    a fragment whose width equalled the budget by construction. Distinguishing
    "nothing fits" from "this fits" is what lets `marked` drop a row rather than
    paint `! not the…`, which is a way of saying nothing (UX round 1, U5).
    """
    if lead + cell_len(label) > width:
        return None
    for meta in metas:
        if lead + cell_len(label) + 2 + cell_len(meta) <= width:
            return meta
    return ""


def _path(value: str, width: int) -> str:
    """Collapse ``$HOME`` then truncate from the LEFT, keeping the tail.

    Two existing helpers, reused rather than reimplemented: the leaf directory
    is what identifies "where am I", so a path too long for the card must lose
    its root and not its name. ``_shorten_home`` alone brings
    ``~/.local/share/uv/tools/local-operator`` to 38 cells, which fits uncut at
    all three reference widths — the elision is a guard for a deep ``cwd``,
    not the common case.
    """
    if not value:
        return "—"
    return _fit_tail(_shorten_home(value), width)


def _model(value: str, width: int) -> str:
    """Truncate a model id from the RIGHT: the provider and family identify it."""
    return truncate_cells(value, width) if value else "—"


@dataclass
class _Body:
    """The report's lines, with ``/session``'s row vocabulary and two changes.

    A near-copy of ``session_panel._Body`` rather than an import: that one pins
    its value column at ``_VALUE_CELL = 11`` (sized for ``1.7M tokens``) and its
    label column at 22, and both are wrong here — six of seven representative
    ``/info`` values exceed 11 cells and ``cwd`` is 48. Widening the shared class
    would change every ``/session`` row to fix a screen it does not draw. The
    ``notes=`` ladder, the ``note()`` wrap-with-indent rule and the header
    shedding are carried over verbatim, because those are the parts that encode
    a measured failure.
    """

    width: int
    lines: list[Text] = field(default_factory=list)

    def blank(self) -> None:
        self.lines.append(Text())

    @property
    def value_cells(self) -> int:
        """The widest a value may be before it must be truncated.

        A CEILING for the path and model helpers, not the pad width — see
        :data:`_VALUE_MIN` for why conflating the two erases every note.
        """
        return max(8, self.width - _ROW_OVERHEAD)

    def kv(
        self,
        name: str,
        value: str,
        note: str = "",
        *,
        notes: Sequence[str] = (),
        value_style: str = "fg",
    ) -> None:
        """One label/value row, left-aligned.

        Values are left-aligned because they are IDENTIFIERS, not magnitudes:
        nothing is being compared down this column, and one column edge reads
        more calmly than two.

        ``notes`` is a ladder of progressively shorter spellings of the SAME
        qualifier, widest first, first-that-fits wins. It exists because the
        plain ``note`` path has exactly two outcomes on a narrow frame and both
        are wrong for a load-bearing qualifier: cropped mid-word above
        ``_NOTE_MIN``, or shed wholesale below it. Cropping a qualifier is fine;
        cropping the scope is not — the existing ``/session`` screen crops
        ``8.8k thin`` at 80 columns and this screen must not inherit that.

        ``value_style`` carries the honesty ink: a value nobody could read is
        drawn ``dim`` with the word ``unavailable``, never blank and never
        ``None``. A MISSING row would be worse than an unavailable one, since
        the reader cannot then tell whether the field does not apply or the
        screen is broken.
        """
        row = Text()
        row.append(f"  {name:<{_LABEL_CELL}}", style=semantic_style("dim"))
        row.append(f"{value:<{_VALUE_MIN}}", style=semantic_style(value_style))
        if notes:
            # Budget measured from the row as actually BUILT, not from a
            # restated literal, so a change to the label column cannot silently
            # reintroduce the crop this ladder removes.
            budget = self.width - row.cell_len - 2
            for candidate in notes:
                if len(candidate) <= budget:
                    row.append(f"  {candidate}", style=semantic_style("dim"))
                    break
        elif note and self.width >= _NOTE_MIN:
            row.append(f"  {note}", style=semantic_style("dim"))
        row.truncate(self.width, overflow="crop")
        self.lines.append(row)

    def marked(
        self,
        glyph: str,
        ink: str,
        label: str,
        meta: str = "",
        indent: int = 0,
        *,
        metas: Sequence[str] = (),
        labels: Sequence[str] = (),
    ) -> None:
        """A glyph-led row: session lines and subagent tree nodes.

        The glyph column sits AFTER the nest prefix, which is what makes depth
        legible without ``│`` continuation rules.

        ``metas`` is the same first-that-fits ladder :meth:`kv` uses, and these
        rows need it for the same measured reason: a session row's meta is a
        ``·``-joined list, so a plain crop lands MID-ITEM and produced ``· b``
        for ``busy`` at 80 columns — the ``8.8k thin`` defect the existing
        ``/session`` screen has, which this screen must not inherit. Dropping a
        whole item is legible; half of one is not.
        """
        row = Text()
        row.append("  " + " " * indent, style=semantic_style("dim"))
        row.append(f"{glyph} ", style=semantic_style(ink))
        candidates = list(metas) or ([meta] if meta else [])
        lead = row.cell_len

        # The label and the meta are negotiated TOGETHER, not one after the
        # other. Sizing the label against the WIDEST meta rung charged it for a
        # string it would never be shown beside: at card 65 the widest meta (42
        # cells) left the label 17, which defeated all three label rungs — the
        # shortest is 22 — so the whole warning row was dropped at exactly the
        # width the ladder was built to survive (design round 1, D1). Every step
        # was individually right and the budget was wrong.
        #
        # STRUCTURAL, per the manager's ruling, and not another per-row special
        # case: this is the same class as U5, which was already patched once at
        # a call site. Label rungs walk OUTER because the label carries the
        # fact; meta rungs walk inner and end with `""`, which sheds the meta
        # entirely rather than losing the row. A row is dropped only when no
        # label rung fits even with no meta at all — the U5 guarantee that a
        # fragment like `! not und…` is never painted, preserved exactly.
        shown, meta_shown = "", ""
        if labels:
            for label_rung in labels:
                fitted = _fit_pair(label_rung, candidates, self.width, lead)
                if fitted is not None:
                    shown, meta_shown = label_rung, fitted
                    break
            if not shown:
                return
        else:
            widest = max((cell_len(c) for c in candidates), default=0)
            shown = truncate_cells(label, max(8, self.width - lead - (widest + 2 if widest else 0)))
            meta_shown = _fit_pair(shown, candidates, self.width, lead) or ""

        row.append(shown, style=semantic_style("fg"))
        if meta_shown and self.width >= _NOTE_MIN:
            # Pad to the meta COLUMN, never to the card's right edge — see
            # ``_MARK_LABEL_CELL``. One space minimum, so an over-long label
            # still separates from its meta rather than running into it.
            pad = max(1, _MARK_LABEL_CELL - cell_len(shown))
            if lead + cell_len(shown) + pad + cell_len(meta_shown) > self.width:
                pad = 1  # the column would push it off; sit it right beside.
            row.append(" " * pad + meta_shown, style=semantic_style("dim"))
        row.truncate(self.width, overflow="crop")
        self.lines.append(row)

    def note(self, text: str) -> None:
        """A dim footnote, wrapped HERE with the indent preserved on every line.

        Not handed to the container's ``fold``: folding applies the ``"  "``
        indent to the first line only, so a continuation lands at column 0 and
        reads as a new record in the middle of a block.
        """
        body_width = max(1, self.width - 2)
        for line in textwrap.wrap(text, body_width) or [""]:
            row = Text(no_wrap=True, overflow="crop")
            row.append(f"  {line}", style=semantic_style("dim"))
            self.lines.append(row)

    def header(self, title: str, meta: str = "", short: str = "") -> None:
        """A section header, shedding its meta to ``short`` on a narrow frame."""
        if meta and self.width < _NOTE_MIN:
            meta = short
        self.lines.append(section_header(title, meta))

    def to_text(self) -> Text:
        out = Text(style=semantic_style("fg"), overflow="fold")
        for index, line in enumerate(self.lines):
            if index:
                out.append("\n")
            out.append_text(line)
        return out


def _install_section(body: _Body, snapshot: InfoSnapshot | None, loading: bool) -> None:
    """Which build this is — FIRST, because both audiences open with that.

    A developer debugging asks "which build is this, and is it the one I think
    it is": the editable-vs-uv-tool distinction is the single most common source
    of "I fixed it and nothing changed" in this codebase. A user filing an issue
    is asked for a version before anything else.
    """
    body.header("Install", "how this lop was installed", "install")
    if snapshot is None:
        # ``Install`` is worker-filled, so it says what it is doing rather than
        # showing a blank block. It is never left out: a missing section reads
        # as a rendering bug.
        body.kv("Version", "checking…" if loading else "unavailable", value_style="dim")
        return
    install = snapshot.install
    body.kv("Version", install.version or "unavailable", notes=_latest_notes(install))
    if install.behind and install.latest_known:
        # Composed like ``welcome.py``'s update row so the two cannot diverge,
        # and the ``— /update`` remedy is dropped WHOLE when it does not fit:
        # ``/upd…`` is an instruction nobody can follow.
        remedy = f"{NOTICE_GLYPHS['warning']} latest is v{install.latest_known} — /update"
        short = f"{NOTICE_GLYPHS['warning']} v{install.latest_known} available"
        row = Text()
        row.append("  " + (remedy if len(remedy) + 2 <= body.width else short))
        row.stylize(semantic_style("warning"))
        row.truncate(body.width, overflow="crop")
        body.lines.append(row)
    body.kv(
        "Install kind",
        install.kind or "unavailable",
        notes=_kind_notes(install.kind),
        value_style="fg" if install.kind else "dim",
    )
    body.kv("Install path", _path(install.prefix, body.value_cells))
    body.kv("Interpreter", _path(install.executable, body.value_cells))
    if install.import_path:
        # Directly under the install path, because the two are only meaningful
        # side by side: this row answers "which code is actually executing",
        # and the row above answers "which install claims to be executing it".
        body.kv(
            "Running code",
            _path(install.import_path, body.value_cells),
            notes=_import_path_notes(install),
            # A WARNING, not an unavailable: nothing failed to read. The same
            # ink and the same glyph as the update row below, because it is the
            # same kind of statement — something is true that you would want to
            # act on — and this screen's whole job is to be believed.
            value_style="warning" if is_shadowed_install(install) else "fg",
        )
        if is_shadowed_install(install):
            body.marked(
                NOTICE_GLYPHS["warning"],
                "warning",
                # LADDERED like the meta beside it: a fixed label was cropped to
                # ``! not und…`` at 45 cells, so the one row built to survive
                # narrow frames was the only one that did not (UX round 1, U5).
                # The shortest rung still names the fault.
                "not under the install path above",
                labels=(
                    "not under the install path above",
                    "not under the install path",
                    "not the installed tree",
                ),
                metas=(
                    "this runtime is executing a different tree",
                    "executing a different tree",
                    "different tree",
                ),
            )
    if install.is_git_snapshot:
        # A SHA is never truncated: a half-printed one is a wrong one.
        body.kv(
            "Source",
            f"git snapshot @ {install.source_ref[:12]}" if install.source_ref else "git snapshot",
            notes=("built by lop-update, not a PyPI wheel", "lop-update build", "lop-update"),
        )
    else:
        body.kv("Source", "PyPI wheel" if install.version else "unavailable")
    if install.build_age_s is not None:
        body.kv("Built", f"{format_duration(install.build_age_s)} ago")
    body.kv(
        "Python",
        install.python_version or "unavailable",
        notes=_python_notes(install),
    )
    body.kv("Platform", install.platform or "unavailable", note=install.machine)
    if is_shadowed_install(install):
        # AFTER the table, not between two of its rows. A four-line wrapped
        # paragraph spliced into the middle of a kv block breaks the column
        # rhythm and reads as the end of the section, so the rows below it look
        # like a new one — the "one record reads as two" fault these screens
        # shed columns to avoid. The flagged VALUE and its ``!`` row stay up in
        # the table where the reader is looking; the explanation follows the
        # block it is about.
        #
        # A passive paragraph earns its space because the failure it describes
        # is silent and total, in AGENTS.md's own words: the banner shows your
        # branch's version, the status bar shows your worktree's path, and every
        # line executing is the other tree's. Nothing errors, so the only
        # symptom is a change that "did not work" — and `/reload` does not clear
        # it, because the cwd is inherited at spawn (`runtime/launch.py`).
        body.blank()
        body.note(
            "Nothing will error: the version and paths above still describe the install, "
            "while the running code is the tree named here. A change made to one will "
            "appear to have no effect in the other. Relaunch from outside that checkout "
            "to clear it — /reload does not."
        )


def _latest_notes(install: object) -> tuple[str, ...]:
    """The PyPI staleness note — three distinct states, never collapsed.

    ``latest_known is None`` means nobody has ever asked, which on the broken
    network ``/info`` is usually opened over is emphatically NOT "up to date".
    """
    from local_operator.update import TTL_S

    if not getattr(install, "version", ""):
        # The unknown is on the INSTALLED side, and the lie is the same one:
        # `is_behind("", latest)` is False by design, so a failed version probe
        # beside a cached NEWER release asserted currency. Both halves fail
        # together on a broken install, which is the state /info is opened in
        # (review round 1, M3).
        return ("latest unknown (installed version unreadable)", "latest unknown", "unknown")
    latest = getattr(install, "latest_known", None)
    if latest is None:
        return ("latest unknown (never checked)", "latest unknown", "unknown")
    age = getattr(install, "latest_age_s", None)
    if getattr(install, "behind", False):
        return (f"behind {latest}", f"< {latest}", "behind")
    if age is None:
        return ("latest", "latest")
    stale = " (stale)" if age > TTL_S else ""
    return (
        f"latest · checked {format_duration(age)} ago{stale}",
        f"checked {format_duration(age)} ago",
        "latest",
    )


def _kind_of_runtime_notes(kind: str) -> tuple[str, ...]:
    """What each runtime kind MEANS, one entry per value of the record's Literal.

    Derived from the real records on a live machine, not from the type: the
    dominant value in practice is ``daemon`` (a runtime spawned by
    ``runtime/process.py``, reattachable and outliving its terminal), while
    ``tui`` is the in-process registrant. Getting these the wrong way round
    would tell a user their session dies with the window when it does not.
    """
    if kind == "daemon":
        return ("a spawned runtime; survives this terminal", "spawned runtime", "spawned")
    if kind == "tui":
        return ("in-process, attached to this terminal", "attached", "attached")
    if kind == "exec":
        return ("one-shot non-interactive run", "one-shot run", "one-shot")
    return ()


def _import_path_notes(install: object) -> tuple[str, ...]:
    """Say what a divergence MEANS, and distinguish the one benign case.

    An editable install's package is the checkout by design, so it diverges from
    the prefix legitimately and the note says "expected". Any other kind
    diverging is the ``launch.py:287`` trap — the runtime is spawned with ``-m``
    and no ``cwd=``, so a session started inside a checkout of this repo puts
    that checkout on ``sys.path[0]`` ahead of site-packages and runs it instead
    of the install. ``/reload`` does not fix that; only relaunching from another
    directory does, which is why the note names the remedy.
    """
    if not getattr(install, "import_path_foreign", False):
        return ()
    if getattr(install, "kind", "") == "editable":
        return ("expected for an editable checkout", "editable checkout", "editable")
    return (
        "a checkout on sys.path is shadowing the install",
        "a checkout is shadowing the install",
        "shadowed by a checkout",
    )


def _kind_notes(kind: str) -> tuple[str, ...]:
    """Say what the install kind MEANS for changing the code, at three widths."""
    if kind == "editable":
        return ("editable checkout; source edits are live", "editable · live source", "editable")
    if kind == "uv-tool":
        return (
            "non-editable; `lop-update` rebuilds it",
            "non-editable · lop-update",
            "non-editable",
        )
    if kind in ("pip", "pipx"):
        return (f"non-editable {kind} install", f"non-editable {kind}", "non-editable")
    return ()


def _python_notes(install: object) -> tuple[str, ...]:
    implementation = getattr(install, "python_implementation", "") or ""
    machine = getattr(install, "machine", "") or ""
    parts = [part for part in (implementation, machine) if part]
    if not parts:
        return ()
    return (" · ".join(parts), parts[0])


def _session_section(body: _Body, snapshot: InfoSnapshot | None, live: LiveState) -> None:
    """The runtime being typed into — SECOND, and cheap enough to paint at once.

    It is the scope everything below is relative to, and it needs no worker, so
    it is complete on the first frame while the ~900 ms fleet scan is still
    resolving. That is what stops the screen ever appearing blank.
    """
    body.header("This session", "the runtime you are typing into", "this runtime")
    process = snapshot.process if snapshot else None
    body.kv("Session id", (process.session_id if process else live.session_id) or "—")
    kind = (process.kind if process else live.kind) or ""
    body.kv(
        "Kind",
        kind or "—",
        # Per KIND, not one hardcoded phrase. Checked against the real records
        # on this machine rather than read off the Literal in ``types.py``:
        # every live record here is ``daemon`` (``runtime/process.py`` spawns
        # with ``kind="daemon"``; only an in-process registrant is ``tui``), so
        # a blanket "attached to this terminal" would have mislabelled the
        # common case as the rare one.
        notes=_kind_of_runtime_notes(kind),
        value_style="fg" if kind else "dim",
    )
    if process is None:
        body.kv("Model", _model(live.model_label, body.value_cells))
        body.kv("Working dir", "checking…", value_style="dim")
        return
    body.kv("Model", _model(process.model_label, body.value_cells))
    if process.effective_model and process.effective_model != process.model_label:
        # Only when they DIFFER: under failover the model answering is not the
        # model selected, and that gap is the whole reason to show the row.
        body.kv(
            "Answering",
            _model(process.effective_model, body.value_cells),
            notes=("failover is in force", "failover"),
        )
    body.kv("Started", f"{format_duration(process.uptime_s)} ago" if process.uptime_s else "—")
    body.kv("Working dir", _path(process.cwd, body.value_cells))
    body.kv(
        "Config dir",
        _path(process.config_dir, body.value_cells),
        notes=_redirect_notes(process.config_dir_redirected, "LOCAL_OPERATOR_CONFIG_DIR"),
    )
    # Shown NEXT to the config dir on purpose. The cache root derives from
    # ``$HOME`` independently of ``LOCAL_OPERATOR_CONFIG_DIR``, which AGENTS.md
    # records as silently producing a *plausible wrong answer* in a run that
    # believes it is isolated. Two adjacent rows are how a reader sees the
    # divergence instead of assuming it away.
    body.kv("Cache dir", _path(process.cache_dir, body.value_cells))
    body.kv(
        "Agent home",
        _path(process.agent_home, body.value_cells),
        notes=_redirect_notes(process.agent_home_redirected, "LOCAL_OPERATOR_HOME"),
    )
    body.kv("Log dir", _path(process.log_dir, body.value_cells))
    port = f"{process.control_port}" if process.control_port else "—"
    body.kv("Control port", port, note=f"protocol v{process.protocol}" if process.protocol else "")


def _approval_notes(mode: str) -> tuple[str, ...]:
    if mode == "ask":
        return ("every tool call is confirmed", "confirmed")
    if mode == "auto":
        return ("tool calls run without a prompt", "auto")
    return ()


def _redirect_notes(redirected: bool, variable: str) -> tuple[str, ...]:
    if not redirected:
        return ()
    return (f"redirected by {variable}", "redirected by env", "redirected")


def _sessions_section(body: _Body, snapshot: InfoSnapshot | None) -> None:
    """Every session on this machine, outer before inner."""
    if snapshot is None:
        body.header("Sessions on this machine", "scanning…", "scanning…")
        body.note("Reading the session registry.")
        return
    sessions = snapshot.sessions
    if not sessions.available:
        # A whole section that could not be gathered REPLACES its header rather
        # than annotating it — ``/session``'s proven ``Ledger unavailable``
        # shape. The prose says what failed and what to do, in one sentence.
        body.header("Sessions unavailable")
        body.note(
            # ``r`` re-runs exactly these probes and the footer advertises it two
            # rows below this sentence; "close and reopen" sent the reader the
            # long way round and quietly implied ``r`` would not help, which is
            # the opposite of true (UX round 1, U7).
            "Could not scan the session registry. Press r to try again."
        )
        return
    meta = f"{sessions.live} live · {sessions.total} total"
    body.header("Sessions on this machine", meta, f"{sessions.live} live")
    if not sessions.lines:
        body.note("No other lop sessions are running on this machine.")
        return
    for line in sessions.lines:
        if line.state == "wedged":
            glyph, ink = WEDGED_MARKER, "danger"
        elif line.is_self:
            glyph, ink = ATTACHED_MARKER, "muted"
        elif line.busy:
            glyph, ink = IDLE_MARKER, "accent"
        else:
            glyph, ink = IDLE_MARKER, "muted"
        # Built widest-first and shed WHOLE ITEMS from the right, because the
        # rightmost facts are the least identifying: which session it is and
        # whether it is wedged must survive to the narrowest frame, while the
        # memory figure is the one a reader can do without.
        bits = ["this session"] if line.is_self else []
        if line.state != "live":
            bits.append(line.state)
        bits.append(format_duration(line.uptime_s))
        if sessions.usage_available:
            bits.append(format_bytes(line.footprint_bytes or line.rss_bytes))
        if line.pending:
            bits.append(f"needs {line.pending}")
        elif line.busy:
            bits.append("busy")
        metas = tuple(" · ".join(bits[:count]) for count in range(len(bits), 0, -1))
        name = line.conversation_name or line.session_id or f"pid {line.pid}"
        body.marked(glyph, ink, name, metas=metas)
    if sessions.build_skew:
        body.note(
            "Live sessions are running more than one build — a change may look "
            "absent in the window that has not been restarted."
        )
    if not sessions.usage_available:
        body.note("Memory could not be measured on this host.")


def _counted(value: int, probe: str, snapshot: "InfoSnapshot | None") -> str:
    """A count, or :data:`UNKNOWN` when the probe that produced it FAILED.

    The screen's own rule — stated for ``mobile_port`` and for memory — is that
    absent is not a measured value. These counts broke it: on an unresolvable
    home the render read ``Agent profiles 0`` / ``Teams 0``, a plausible figure
    the snapshot cannot support, with the failure disclosed only in a separate
    block the reader has to cross-reference (QA round 2, Q8).

    Keyed on the probe NAME appearing in ``degraded`` rather than on the value,
    because 0 is a legitimate answer on a machine that genuinely has no teams —
    suppressing every zero would trade one lie for another.
    """
    if snapshot is not None and any(name == probe for name, _ in snapshot.degraded):
        return UNKNOWN
    return str(value)


def _agents_section(body: _Body, snapshot: InfoSnapshot | None, live: LiveState) -> None:
    """Profiles, teams, and this session's subagent tree.

    The tree is drawn from the LIVE capture, not the worker's, so it is present
    on the first frame — it is the operator's stated motivation ("a sense of the
    active runtimes in parallel") and it is free to read.
    """
    agents = snapshot.agents if snapshot else None
    running = agents.running if agents else live.running
    depth = agents.max_depth if agents else live.max_depth
    tree = agents.tree if agents else live.tree
    deeper = agents.deeper if agents else live.deeper
    # ``none running`` rather than ``0 running``: a zero in a count column reads
    # as a failed probe, a word does not.
    meta = f"{running} running · depth {depth}" if running else "none running"
    body.header("Agents and subagents", meta, f"{running} running" if running else "none running")
    if agents is not None:
        body.kv("Agent profiles", _counted(agents.profiles, "agents.profiles", snapshot))
        body.kv("Teams", _counted(agents.teams, "agents.teams", snapshot))
    queued = agents.queued if agents else live.queued
    settled = agents.settled if agents else live.settled
    if running or queued or settled:
        # Only when there is something to count. On a fresh session a
        # ``0 running · 0 queued`` row directly above "No subagents have been
        # launched" says the same thing twice, and a row of zeros reads as a
        # failed probe rather than as an idle session — the same reason the
        # header meta says ``none running`` instead of ``0 running``.
        body.kv(
            "Subagents",
            f"{running} running · {queued} queued",
            # "settled (retained)" and not "settled": ``_evict_overflow`` drops
            # settled records past a cap, so this under-reports by design, and a
            # count that silently under-reports inside a bug report is worse
            # than one that says what it counts.
            notes=(f"{settled} settled (retained)", f"{settled} settled", f"{settled} done"),
        )
    max_running = agents.max_running if agents else live.max_running
    at_capacity = agents.at_capacity if agents else live.at_capacity
    if max_running is not None:
        body.kv(
            "Subagent capacity",
            f"{max_running} concurrent",
            notes=("at capacity — new launches queue", "at capacity") if at_capacity else (),
            value_style="warning" if at_capacity else "fg",
        )
    if not tree:
        # The empty state pairs a statement of FACT with a statement of why it
        # is fine, matching ``/analytics``' own empty shape. The section header
        # above is the evidence that the screen looked.
        body.note("No subagents have been launched in this session.")
        body.note("Subagents appear here while a task is running.")
        return
    body.blank()
    for node in tree:
        glyph, ink = _SUBAGENT_MARKS.get(node.status, (NOTICE_GLYPHS["info"], "dim"))
        bits = [node.status or "unknown"]
        if node.agent_role and node.agent_role != node.label:
            bits.insert(0, node.agent_role)
        prefix = _row_prefix(node.depth)
        row = Text()
        row.append("  " + prefix, style=semantic_style("dim"))
        row.append(f"{glyph} ", style=semantic_style(ink))
        # Shed whole items, longest first: the STATUS is the fact this section
        # exists to report, so the role qualifier goes before it does.
        metas = tuple(" · ".join(bits[index:]) for index in range(len(bits)))
        budget = max(8, body.width - row.cell_len - len(metas[0]) - 2)
        shown = truncate_cells(node.label or node.job_id, budget)
        row.append(shown, style=semantic_style("fg"))
        if body.width >= _NOTE_MIN:
            # The meta column is measured from the row as BUILT, so a nested
            # node's prefix pushes its label right and its meta stays put —
            # which is what keeps the status column readable down a tree whose
            # glyph column is deliberately ragged.
            pad = max(1, _MARK_LABEL_CELL + 2 - cell_len(shown) - cell_len(prefix))
            available = body.width - row.cell_len - pad
            for candidate in metas:
                if len(candidate) <= available:
                    row.append(" " * pad + candidate, style=semantic_style("dim"))
                    break
        row.truncate(body.width, overflow="crop")
        body.lines.append(row)
    if deeper:
        # BASE indent and NO `└`. The fold was emitted at the cap's child indent
        # (6 cells) and appended after the whole tree, so it rendered under
        # whatever the last row happened to be — in the captured frame a DEPTH-0
        # node — three levels to its right, claiming hidden children that node
        # does not have (design round 1, D2). `+N deeper` counts nodes below the
        # cap ANYWHERE in the tree, so it is a section summary, not a child of
        # any row. A `└` that connects to nothing is worse than no glyph, and
        # the depth it summarises is named instead of mimed.
        row = Text()
        row.append(
            f"  +{deeper} deeper (below depth {_MAX_TREE_DEPTH})",
            style=semantic_style("dim"),
        )
        body.lines.append(row)
    body.blank()
    # The honesty boundary, stated rather than implied: a tree on screen would
    # otherwise read as a fleet-wide view. Nothing about another session's
    # subagents is observable — ``SessionRecord`` carries no subagent field —
    # and reaching for one would need a new control-socket op and a protocol
    # bump.
    body.note(
        "Subagent trees are in-process state, so only this session's is visible. "
        "Other sessions report busy, pending and memory."
    )


def _env_section(body: _Body, snapshot: InfoSnapshot | None, live: LiveState) -> None:
    """The bug-report extras — LAST, because nobody reads them until asked.

    Putting these nine rows above the tree would push the parallelism question
    — the operator's stated motivation — below two folds.
    """
    body.header("Environment", "for bug reports", "environment")
    env = snapshot.env if snapshot else None
    size = (env.terminal_size if env else live.terminal_size) or None
    body.kv(
        "Terminal",
        # ``checking…`` while the worker is out, NOT ``—``: an em-dash means "we
        # looked and could not tell" everywhere else on this screen, so a first
        # frame using it for a value still arriving told the reader their TERM
        # could not be determined — mildly alarming on the one screen about what
        # is actually running, and three loading vocabularies on one frame is
        # one too many (UX round 1, U6).
        ((env.term or "—") if env else "checking…"),
        note=f"{size[0]}x{size[1]}" if size else "",
        value_style="fg" if env else "dim",
    )
    body.kv("Theme", (env.theme if env else live.theme) or "—")
    approvals = (env.approval_mode if env else live.approval_mode) or ""
    body.kv(
        "Approvals",
        approvals or "—",
        # An UNKNOWN mode gets no note at all. Falling through to the "auto"
        # wording would tell a reader that tool calls run unconfirmed on the
        # strength of a field nobody managed to read — the most consequential
        # sentence on this screen to be wrong about.
        notes=_approval_notes(approvals),
        value_style="fg" if approvals else "dim",
    )
    mcp_configured = env.mcp_configured if env else live.mcp_configured
    mcp_connected = env.mcp_connected if env else live.mcp_connected
    mcp_settling = env.mcp_settling if env else live.mcp_settling
    mcp_failed = env.mcp_failed if env else live.mcp_failed
    if mcp_configured:
        body.kv(
            "MCP servers",
            f"{mcp_connected} of {mcp_configured} connected",
            # SETTLING suppresses the failure tally rather than reporting it:
            # OAuth servers routinely miss the 250 ms startup gate, and naming
            # one as failed mid-handshake produces a bug report about a server
            # that came up a second later.
            notes=(
                ("still connecting", "connecting")
                if mcp_settling
                else ((f"{mcp_failed} failed", "failed") if mcp_failed else ())
            ),
            value_style="warning" if mcp_failed and not mcp_settling else "fg",
        )
        if env is not None and env.mcp_failures and not mcp_settling:
            for name, message in env.mcp_failures:
                body.marked(NOTICE_GLYPHS["error"], "danger", f"{name}: {message}", indent=2)
    else:
        body.kv("MCP servers", "none configured", value_style="dim")
    if env is None:
        body.kv("Browser", "checking…", value_style="dim")
        return
    body.kv(
        "Browser",
        env.browser_backend or "none",
        notes=(env.browser_name,) if env.browser_name else (),
    )
    body.kv(
        "Mobile relay",
        "installed" if env.mobile_installed else "not installed",
        notes=("healthy", "up") if env.mobile_healthy else (),
    )
    body.kv("Multiplexer", env.multiplexer or "none")
    body.kv("Guides", str(env.guides))
    if env.skills:
        body.kv("Skills", str(env.skills))
    # NAMES, never values, lengths, prefixes or hashes. The diagnostic question
    # is "is the key even set?", which a name answers completely.
    names = list(env.credential_keys)
    keys = ", ".join(names)
    credentials_failed = snapshot is not None and any(
        name == "env.credentials" for name, _ in snapshot.degraded
    )
    if credentials_failed:
        # `0 keys · none stored` is an AFFIRMATIVE claim the snapshot cannot
        # support when the probe raised — the reader is told something false and
        # must cross-reference the degraded block to discover it (QA round 2,
        # Q8).
        body.kv("Credentials", UNKNOWN, notes=("could not read",))
    else:
        body.kv(
            "Credentials",
            f"{len(names)} keys",
            # COUNT the overflow, never crop it. The middle rung used to be
            # `truncate_cells(keys, 30)`, which produced `OPENAI_API…` — a crop
            # wearing a ladder's clothing, and precisely the `8.8k thin`
            # mid-token defect the spec forbids inheriting (design round 1, D3).
            # A half-printed key cannot be told from `OPENAI_API_KEY_2` or a
            # typo, which is the exact ambiguity this row exists to resolve.
            notes=(
                (keys, f"{names[0]} +{len(names) - 1} more", f"{len(names)} set")
                if len(names) > 1
                else (keys,) if names else ("none stored",)
            ),
        )


def build_info_report(
    snapshot: InfoSnapshot | None,
    live: LiveState,
    width: int = _DEFAULT_CARD_WIDTH,
) -> Text:
    """Render the whole screen at ``width`` content cells.

    ``snapshot is None`` is the first frame: ``Install`` and ``This session``
    are already useful from the live capture, and the worker-filled rows say
    ``checking…`` in dim. The screen is never blank and never a spinner.
    """
    width = max(_MIN_CARD_WIDTH, width)
    body = _Body(width)
    _install_section(body, snapshot, loading=snapshot is None)
    body.blank()
    _session_section(body, snapshot, live)
    body.blank()
    _sessions_section(body, snapshot)
    body.blank()
    _agents_section(body, snapshot, live)
    body.blank()
    _env_section(body, snapshot, live)
    if snapshot is not None and snapshot.degraded:
        body.blank()
        # Named, not swallowed. "cache dir: —" without a reason costs the
        # reporter a second round trip, which is the cost this screen removes.
        body.header("Could not read", f"{len(snapshot.degraded)} probes", "degraded")
        for name, reason in snapshot.degraded:
            body.note(f"{name}: {reason}")
    if snapshot is not None and snapshot.captured_at:
        body.blank()
        # A wall-clock stamp so a pasted report is unambiguous about WHEN.
        body.note("captured " + time.strftime("%H:%M:%S", time.localtime(snapshot.captured_at)))
    return body.to_text()


class InfoScreen(ModalScreen[None]):
    """The ``/info`` surface: pushed immediately, filled by a worker.

    Push-before-read is ``/session``'s pattern and is right here for a stronger
    reason than there: the sessions block measured **879.5 ms** on this host
    (``session_resource_usage`` shells ``top -l1`` for the whole system on
    macOS), three orders of magnitude past a frame. So the screen owns a
    visible, cancellable surface before any I/O starts, and a late result
    updates that surface rather than pushing over whatever the user did next.

    Deliberately NOT animated. This is a snapshot people screenshot; a spinner
    would make two consecutive captures differ, which reads as motion.
    """

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back", show=False),
        Binding("q", "dismiss_screen", "Back", show=False),
        Binding(INFO_COPY_KEY, "copy_report", "Copy", show=False),
        Binding("r", "refresh_report", "Refresh", show=False),
        Binding("up", "scroll_up", "Up", show=False),
        Binding("down", "scroll_down", "Down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(self, live: LiveState, snapshot: InfoSnapshot | None = None) -> None:
        super().__init__()
        self.live = live
        self.snapshot = snapshot
        self.presentation_cancelled = False

    def compose(self) -> ComposeResult:
        with Container(classes="analytics-panel"):
            self._title = Static(self._title_text(), id="info-title")
            yield self._title
            with VerticalScroll(id="info-scroll") as scroll:
                self._scroll = scroll
                self._body = Static(self._report_text(), id="info-body")
                yield self._body
            self._hint = Static(self._info_hint(), id="info-hint")
            yield self._hint

    def on_mount(self) -> None:
        # A fast probe can finish while this screen is still mounting; the
        # stored snapshot, not the compose-time text, must win that race.
        self._repaint()
        self.call_after_refresh(self._dismiss_if_cancelled)

    def on_unmount(self) -> None:
        self.presentation_cancelled = True

    def on_screen_resume(self) -> None:
        self._dismiss_if_cancelled()

    def invalidate(self) -> None:
        """Retire this request without ever popping a newer modal above it."""
        self.presentation_cancelled = True
        self._dismiss_if_cancelled()

    def _dismiss_if_cancelled(self) -> None:
        # Dismiss only when RESUMED: ``Screen.dismiss()`` pops the current
        # screen, not necessarily the instance it was called on, so an owner
        # switch under another modal must not pop that modal instead.
        if self.presentation_cancelled and self.is_mounted and self.app.screen is self:
            self.dismiss(None)

    def set_snapshot(self, snapshot: InfoSnapshot) -> None:
        """Publish the worker's result, while this presentation is still owned."""
        if self.presentation_cancelled:
            return
        self.snapshot = snapshot
        self._repaint()

    def _card_width(self) -> int:
        """The content cells a row may actually occupy.

        MEASURED off the mounted scroll rather than recomputed from the CSS,
        because a formula and a stylesheet drift: ``#*-scroll`` reserves a
        stable scrollbar gutter that a card-width formula does not know about,
        and building rows one cell too wide folds a value onto a second line —
        the "one record reads as two" fault these screens exist to remove.
        """
        scroll = getattr(self, "_scroll", None)
        if scroll is not None and scroll.is_mounted and scroll.size.width:
            return max(_MIN_CARD_WIDTH, scroll.size.width - 1)
        try:
            card = min(140, int(self.app.size.width * 0.9))
            return max(_MIN_CARD_WIDTH, card - 7)  # 4 padding + 2 border + 1 gutter
        except Exception:  # noqa: BLE001 — before mount there is no app size
            return _DEFAULT_CARD_WIDTH

    def _title_text(self) -> Text:
        return Text(
            "Install info\n" + "─" * max(1, self._card_width()),
            no_wrap=True,
            overflow="crop",
        )

    def _report_text(self) -> Text:
        return build_info_report(self.snapshot, self.live, self._card_width())

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is not None and body.is_mounted and not self.presentation_cancelled:
            # Was the reader pinned to the BOTTOM before this repaint? The body
            # roughly doubles when the probe lands, so an `End` pressed during
            # `checking…` left the viewport at its old absolute offset — two
            # sections above the end the user asked for, moving under them for
            # reasons they cannot see (UX round 1, U4). Only the pinned case is
            # restored: a reader parked mid-document is deliberately left where
            # they are, since re-anchoring that would be the same fault.
            scroll = getattr(self, "_scroll", None)
            was_at_end = bool(
                scroll is not None
                and scroll.is_mounted
                and scroll.max_scroll_y > 0
                and scroll.scroll_y >= scroll.max_scroll_y - 1
            )
            body.update(self._report_text())
            if was_at_end and scroll is not None:
                self.call_after_refresh(scroll.scroll_end, animate=False)
            title = getattr(self, "_title", None)
            if title is not None and title.is_mounted:
                title.update(self._title_text())
            self.call_after_refresh(self._sync_hint)

    def _info_hint(self) -> str:
        """``esc / q back · ctrl+r copy`` — key-then-verb, ``·``-separated.

        The scroll hint is appended by :meth:`_sync_hint` only when the body can
        actually scroll, so the footer never advertises a dead control. Below
        ``_NOTE_MIN`` the SCROLL hint sheds before the COPY hint: copy is the
        discoverable feature here, scroll is guessable.
        """
        if self._card_width() < _NOTE_MIN:
            return f"esc back · {INFO_COPY_KEY} copy"
        return f"esc / q back · {INFO_COPY_KEY} copy · r refresh"

    def on_resize(self) -> None:
        # Column arithmetic is a function of the card width, so a resize is a
        # re-render rather than only a hint refresh.
        self._repaint()
        self.call_after_refresh(self._sync_hint)

    def _sync_hint(self) -> None:
        scroll = getattr(self, "_scroll", None)
        hint = getattr(self, "_hint", None)
        if scroll is not None and hint is not None and hint.is_mounted:
            hint.update(self._info_hint() + (" · ↑↓ scroll" if scroll.max_scroll_y > 0 else ""))

    def render_lines_for_test(self) -> list[str]:
        """The report as plain strings — what a user actually reads."""
        out: list[str] = []
        for line in self._report_text().plain.split("\n"):
            out.append(line)
        return out

    # -- actions --------------------------------------------------------------
    def action_dismiss_screen(self) -> None:
        self.invalidate()

    def action_copy_report(self) -> None:
        """Copy the REDACTED export, built from the dataclass not the pixels.

        Composed by ``info.render.build_export`` rather than scraped off the
        painted rows, following ``AsidePanel.copy_text``: two implementations of
        one idea drift, and the one that drifts here would be the one that
        decides what is safe to publish. The write goes through the app's single
        clipboard path (OSC 52, with its receipt toast), so a copy survives ssh
        and a multiplexer and is never silent.
        """
        if self.snapshot is None:
            # Keep the bell (something was refused) but SAY so: the footer has
            # advertised `ctrl+r copy` since frame one, and on a terminal with
            # the bell disabled -- common, and the default in several -- the
            # refusal was completely silent. A user who believes the report is
            # on their clipboard pastes whatever was there before into an issue,
            # which is worse than waiting (UX round 1, U3).
            self.app.bell()
            notice = getattr(self.app, "_system_notice", None)
            if notice is not None:
                notice("still reading this install — try ctrl+r again in a moment", "warning")
            return
        payload = build_export(self.snapshot)
        put = getattr(self.app, "_put_on_clipboard", None)
        if put is None:
            self.app.copy_to_clipboard(payload)
        else:
            put(payload, self)
        # A SECOND receipt, on top of the shared clipboard one, because this
        # payload has a property the user should be told exactly once and at
        # exactly this moment: it was redacted. `render.py` and its tests do
        # thorough work that nothing in the UI ever mentioned, and the moment
        # the user is thinking about pasting into a public issue is when
        # "paths are home-relativised, no credential values" is worth knowing
        # (UX round 1, U10). The shared receipt stays generic — it is used by
        # every other copy gesture and must not claim redaction for them.
        notice = getattr(self.app, "_system_notice", None)
        if notice is not None:
            notice("/info report copied — paths relativised, no secrets included", "info")

    def action_refresh_report(self) -> None:
        """Re-run the probes on demand. NOT on a timer.

        A ``top -l1`` fork per second is a real cost on the machine being
        diagnosed, and ``/info`` is a snapshot rather than a monitor — so this
        is a manual gesture and the captured-at stamp says which moment is on
        screen.

        The snapshot is cleared FIRST so the screen acknowledges the keypress
        within a frame. Without it the probe took ~4 s during which the screen
        was byte-identical to before the press — same rows, same footer, same
        stamp — so the key read as dead and the natural response was to press it
        again (UX round 1, U2). Dropping back to ``checking…`` reuses the
        first-open vocabulary rather than inventing a second loading state, and
        it is the honest picture: those fields are, once again, unread.
        """
        handler = getattr(self.app, "refresh_info_screen", None)
        if handler is None:
            return
        self.snapshot = None
        self._repaint()
        handler(self)

    def action_scroll_up(self) -> None:
        self._scroll.scroll_up()

    def action_scroll_down(self) -> None:
        self._scroll.scroll_down()

    def action_page_up(self) -> None:
        self._scroll.scroll_page_up()

    def action_page_down(self) -> None:
        self._scroll.scroll_page_down()

    def action_scroll_home(self) -> None:
        self._scroll.scroll_home()

    def action_scroll_end(self) -> None:
        self._scroll.scroll_end()
