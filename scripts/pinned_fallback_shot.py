"""Capture the pinned-child FALLBACK state: roster dock, band badge, notice.

Run from the worktree root (or point ``--repo`` at another checkout for a
before frame; the script's own imports follow ``--repo``):

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/pinned_fallback_shot.py \\
        OUT.svg [COLSxROWS] [--notice] [--repo CHECKOUT]

WHAT IT IS FOR. A role-pinned subagent (designer/ux-reviewer: ``effort:hi``
resolved through ``subagents.models.hi``) can be moved off its pinned model by
a provider fallback. On the pre-fix tree that substitution is SILENT: the job
row keeps only the effective label, so the band under the child's page names a
model nobody chose and nothing says the pin was abandoned — the exact
self-review collapse the pin exists to prevent. This frame is the evidence
pair: the same state rendered by two checkouts.

STATE. One pinned child (``round1-designer``, role ``designer``, pinned to
``anthropic/claude-sonnet-5-5``) now serving on ``deepseek/deepseek-flash`` —
the operator's own session model, which is the first thing the fallback chain
reaches — beside one unpinned sibling. The job duck-carries
``requested_model_label`` / ``model_fallback`` as plain attributes so ONE
script renders both trees: the pre-fix tree ignores them (the silent frame),
the post-fix tree paints the badge from them.

MODES. Default: the child's page is open, so the band paints the child's
model (the surface ``job_stats``' model_label reaches). ``--dock``: the
roster dock with the child page closed. ``--notice``: the dock plus the
fallback notice the relay emits, built by the production helper when the tree
has it (the pre-fix tree does not, so only the after side exists for that
frame).

Pinned so a pair differs only where the code does: ``time.time``, the dock's
and the title's spinner frames, and the update probe, exactly as
``busy_roster_shot.py`` pins them.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def _repo_from_argv() -> Path:
    for index, arg in enumerate(sys.argv):
        if arg == "--repo" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).resolve()
    return Path(__file__).resolve().parent.parent


REPO = _repo_from_argv()
# The CAPTURE helpers come from this script's own tree (they may not exist in
# an older checkout); the APP comes from --repo, inserted first so it wins.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()
sys.path.insert(0, str(REPO))

import time  # noqa: E402

NOW = 1_790_000_000.0
time.time = lambda: NOW  # type: ignore[assignment]

import local_operator  # noqa: E402
import local_operator.update as _update  # noqa: E402


class _NotBehind:
    """The update probe pinned to "not behind" (see ``dock_band_shot.py``)."""

    behind = False
    latest: str | None = None


_update.check_latest = lambda *a, **k: _NotBehind()  # type: ignore[assignment]

from local_operator.harness.types import Usage  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The incident's shape: a review child pinned cross-family, then served by the
#: operator's own session model.
PIN = "anthropic/claude-sonnet-5-5"
EFFECTIVE = "deepseek/deepseek-flash"
PARENT_MODEL = EFFECTIVE


class _Session(FakeSession):
    """The parent: a session the child inherits its model from.

    ``FakeSession.model_label`` is a fixed "test/model" and read-only, so this
    override states the one fact the frames need — the designer's fallback
    landed on the PARENT'S OWN model, which is the collapse the pin exists to
    prevent.
    """

    @property
    def model_label(self) -> str:
        return PARENT_MODEL


class _Job:
    """One duck-typed job row, carrying the fields the panel and band read.

    Mirrors ``AsyncJob``'s shape (see ``tests/unit/tui/test_subagent_stats.py``
    for the same convention) plus the pin-integrity pair this capture exists
    for. ``requested_model_label``/``model_fallback`` are set as PLAIN
    attributes so one script runs against both trees: pre-fix code reads
    neither, post-fix code reads both through ``getattr``.
    """

    def __init__(
        self,
        job_id: str,
        label: str,
        *,
        agent_role: str,
        model_label: str,
        requested_model_label: str | None = None,
        model_fallback: bool = False,
        model_fallback_reason: str = "",
        progress: str = "",
        age: float = 120.0,
        usage: Usage | None = None,
        context_window: int = 0,
    ) -> None:
        self.id = job_id
        self.type = "task"
        self.status = "running"
        self.label = label
        self.agent_role = agent_role
        self.effort = "hi" if requested_model_label else None
        self.start_time = NOW - age
        self.started_at = self.start_time
        self.settled_at = None
        self.result_text = None
        self.error_text = None
        self.queued = False
        self.restored = False
        self.cut_off_cause = ""
        self.trajectory = None
        self.prompt = None
        self.model_label = model_label
        self.requested_model_label = requested_model_label
        self.model_fallback = model_fallback
        self.model_fallback_reason = model_fallback_reason
        self.context_window = context_window
        self.usage = usage
        self.latest_details = {"progress": progress} if progress else None


class _Jobs:
    """The slice of ``AsyncJobManager`` the app and panel read."""

    def __init__(self, jobs: list[Any]) -> None:
        self._rows = {job.id: job for job in jobs}

    def list(self, *, registrant_id: str | None = None) -> list[Any]:
        return list(self._rows.values())

    def get(self, job_id: str, *, registrant_id: str | None = None) -> Any:
        return self._rows.get(job_id)


def _state() -> tuple[FakeSession, _Job]:
    designer = _Job(
        "designer1",
        "round1-designer",
        agent_role="designer",
        model_label=EFFECTIVE,
        requested_model_label=PIN,
        model_fallback=True,
        model_fallback_reason="provider failure",
        progress="reading the diff",
        usage=Usage(input_tokens=120_000, output_tokens=4_000, context_tokens=44_000),
        context_window=160_000,
    )
    coder = _Job(
        "coder1",
        "round1-coder",
        agent_role="coder",
        model_label=PARENT_MODEL,
        progress="running tests",
        usage=Usage(input_tokens=30_000, output_tokens=1_000, context_tokens=12_000),
        context_window=160_000,
    )
    session = _Session()
    # A duck manager, not ``_FakeJobs``: this script needs rows with the
    # pin-integrity attributes the panel reads through ``getattr``, which the
    # fixture's derived rows do not carry.
    session.jobs = _Jobs([designer, coder])  # type: ignore[assignment]
    return session, designer


def _notice_text(designer: _Job) -> str | None:
    """The production notice text, when this tree carries the helper.

    Pre-fix the relay has no such notice, so the before side for the notice
    frame does not exist; the post-fix side reads the real builder rather than
    re-typing its grammar here.
    """
    try:
        from local_operator.harness.subagent import _pinned_fallback_notice
    except ImportError:
        return None
    return _pinned_fallback_notice(
        designer.label,
        str(designer.agent_role or ""),
        PIN,
        EFFECTIVE,
        "provider failure",
    )


async def main(out: str, size: tuple[int, int], *, notice: bool, dock: bool) -> None:
    session, designer = _state()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        # Boot: the app adopts the session in a worker (see #142); every read
        # below goes through that session.
        for _ in range(400):
            await pilot.pause()
            if app._session is not None and app._subagent_panel is not None:
                break
            await asyncio.sleep(0.05)
        # Retire the splash so the notice paints as a transcript row (the
        # established shape: toast/splash is only for the empty state).
        app._append_block(UserBlock("review the subtree changes for #1813"))
        await pilot.pause()

        panel = app._subagent_panel
        assert panel is not None
        if not dock and not notice:
            # Selected: the designer is the child under the band, and the
            # panel's off-thread stats read is what the band consumes. NOTE the
            # dock collapses to the open child's OWN children here; the band is
            # the TUI surface a child's model paints on, so this is the frame
            # that must show the badge.
            panel.sync(session, selected_job=designer)
            app._open_subagent_view(designer.id)
            for _ in range(20):
                await asyncio.sleep(0.15)
                await pilot.pause()
                app._refresh_subagent_view()
        else:
            # The roster dock, the child page closed: the moment the notice
            # lands and the surface the operator scans for a child's state.
            app._refresh_band()
            panel.sync(session)
            await asyncio.sleep(0.4)
        if notice:
            text = _notice_text(designer)
            if text is None:
                raise SystemExit(
                    "this tree has no _pinned_fallback_notice: the notice frame "
                    "exists only on the post-fix side"
                )
            from local_operator.harness.types import NoticeEvent

            session.emit(NoticeEvent(text=text, kind="warning"))
            await asyncio.sleep(0.5)

        # The two TIME-driven cells, pinned last so a pair differs only where
        # the code does: the splash's rotating tip and the dock's spinner.
        welcome = app._welcome
        if welcome is not None:
            welcome._stop_tip_timer()
            welcome._tip_index = 0
            welcome.refresh_info()
        panel = app._subagent_panel
        if panel is not None:
            panel._stop_spinner()
            panel._spinner_index = 0
            panel._paint_all(reread_stats=False)
        view = getattr(app, "_subagent_view", None)
        if view is not None:
            # The PAGE TITLE has its own spinner tick, separate from the dock's
            # (pinned above); unpinned, a before/after pair differs by one
            # animation frame in the title glyph — noise the pair cannot afford.
            view._stop_spinner()
            view._spinner_index = 0
            view._paint_chrome()
        await pilot.pause()
        save_capture(app, out)

        band = app.query_one("#band")
        width = int(app.size.width)
        band_text = app._status.render_text(width).plain if app._status is not None else ""
        screen = app.screen
        mode = "notice" if notice else ("dock" if dock else "page")
        print(
            json.dumps(
                {
                    "tree": local_operator.__file__,
                    "mode": mode,
                    "grid": [int(app.size.width), int(app.size.height)],
                    "band_region": list(band.region),
                    "band_text": band_text,
                    "band_cells": len(band_text),
                    "screen_size": list(screen.size),
                    "screen_virtual_size": list(screen.virtual_size),
                    "vscrollbar": bool(screen.show_vertical_scrollbar),
                    "subagent_panel": list(panel.region) if panel is not None else None,
                    "panel_rows": panel.predicted_rows() if panel is not None else None,
                    "summary": panel.summary_text() if panel is not None else None,
                    "notice_text": _notice_text(designer) if notice else None,
                }
            )
        )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--repo" in sys.argv:
        repo_value = sys.argv[sys.argv.index("--repo") + 1]
        args = [a for a in args if a != repo_value]
    target = args[0]
    cols, rows = (int(x) for x in (args[1] if len(args) > 1 else "120x40").split("x"))
    asyncio.run(
        main(target, (cols, rows), notice="--notice" in sys.argv, dock="--dock" in sys.argv)
    )
