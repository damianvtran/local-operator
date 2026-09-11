"""Rendered frames of the durable audit-history surface.

Captures the head notice in each state of the §4 copy table in
``docs/design/full-history-audit-window.md``, plus the compaction marker as it
renders mid-transcript. Drives the REAL ``OperatorApp`` against a REAL owner
runtime over the real socket, so the stylesheet is applied and the pages come
back through the same RPC the TUI uses in production — a lightweight test host
declares no ``CSS_PATH`` and would show none of this.

Usage::

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \\
        scripts/audit_history_shot.py <out-dir> [state] [COLSxROWS]

States: ``context`` (rows the model still sees), ``audit`` (pre-compaction
rows), ``exhausted`` (the journal's first row reached), ``marker`` (the
compaction boundary mid-transcript), or ``all``.

The geometry argument is not decoration. A head-notice state is a function of
the VIEWPORT as well as the history: ``_reconcile_head_notice`` chooses between
its scrollable and not-scrollable copy from ``virtual_size`` against
``container_size``, so a frame captured at one size can show a state that does
not reproduce at another. Capture the states you are claiming at more than one
geometry, and say in the evidence which one each frame is.

Two frames are written per capture, ``<state>.<COLS>x<ROWS>.svg`` and
``<state>.<COLS>x<ROWS>.settled.svg``, taken before and after a further settle.
They must be identical for a static state; a difference is a reflow the reader
sees as motion.

The GEOMETRY is in the filename deliberately. It used to name frames by state
alone, so capturing two geometries into one directory left files that were
silently all the second run — a design reviewer's settle check "passed" over
four such overwritten files before the collision was spotted. Since the whole
point of the geometry argument is to compare sizes, the output names have to
distinguish them or the script quietly destroys the comparison it exists for.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Any pilot that boots the TUI must not inherit the operator's cmux identity:
# a headless test carrying a real CMUX_WORKSPACE_ID has previously renamed his
# live workspaces. Cleared before the app is imported, alongside HOME/config
# isolation.
for _name in [key for key in os.environ if key.startswith("CMUX_")]:
    del os.environ[_name]

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.harness.types import Message, TextContent  # noqa: E402
from local_operator.session.attached import AttachedSession  # noqa: E402
from local_operator.session.runtime.server import RuntimeServer  # noqa: E402
from local_operator.session.runtime.serving import ServingSessionHandle  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.e2e.harness import ScriptedStream, build_session, text_turn  # noqa: E402
from tests.unit.session.test_remote import _never_take_over  # noqa: E402


async def _seed(directory: Path, *, compactions: int, rows_each: int) -> None:
    """A journal with real compaction cuts, as a long-running session has."""
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    for cut in range(compactions):
        batch = []
        for index in range(rows_each):
            user = index % 2 == 0
            batch.append(
                Message(
                    id=f"cut{cut}-row-{index:04}",
                    role="user" if user else "assistant",
                    content=[
                        TextContent(
                            text=(
                                f"What did we decide about phase {cut}, item {index}?"
                                if user
                                else f"In phase {cut} we settled item {index} by "
                                "recording the constraint and moving on."
                            )
                        )
                    ],
                    stop_reason="stop",
                )
            )
        for message in batch:
            await transcript.append_message(message)
        await transcript.append_compaction(f"phase {cut} summary", batch[-1].id, 500)
    for index in range(rows_each):
        user = index % 2 == 0
        await transcript.append_message(
            Message(
                id=f"tail-row-{index:04}",
                role="user" if user else "assistant",
                content=[
                    TextContent(
                        text=(
                            f"And the current question {index}?"
                            if user
                            else f"Currently we are on question {index}, still in context."
                        )
                    )
                ],
                stop_reason="stop",
            )
        )
    transcript.flush()


async def capture(out_dir: Path, state: str, size: tuple[int, int] = (100, 34)) -> None:
    root = Path(os.environ["HOME"])
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"shot-{state}"
    rows_each = 30 if state == "marker" else 400
    await _seed(directory, compactions=6 if state != "marker" else 3, rows_each=rows_each)
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=root)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    remote = await AttachedSession.connect(
        server._record,
        directory.name,
        config_dir=config,
        takeover_factory=_never_take_over,
        display_window=True,
    )

    async def factory():
        return remote

    try:
        app = OperatorApp(factory)
        async with app.run_test(size=size) as pilot:
            for _ in range(300):
                await pilot.pause()
                if app._session is not None and app._transcript_view().blocks():
                    break
            for _ in range(40):
                await pilot.pause()

            async def page_once() -> bool:
                """Fetch one older page INTO the app's deferred-head buffer.

                Mirrors what ``_fetch_older_resume_page`` does on the reader's
                scroll: rows fetched straight off the session would never enter
                the frame, which is how an earlier version of this script
                captured a tail while claiming to show the head.
                """
                if not remote.history_before_token:
                    return False
                rows = await remote.load_older_display_page()
                if not rows:
                    return False
                app._resume_pending_head = rows + app._resume_pending_head
                for message in rows:
                    if getattr(message, "role", "") == "tool" and getattr(
                        message, "tool_call_id", None
                    ):
                        app._resume_results[message.tool_call_id] = message
                await pilot.pause()
                return True

            if state in ("audit", "exhausted", "marker"):
                # Walk into the pre-compaction phase the way a reader does.
                for _ in range(1200):
                    if state == "audit" and remote.history_is_audit:
                        break
                    if not await page_once():
                        break

            if state == "marker":
                # Mount pages until the compaction boundary is on screen.
                from local_operator.tui.session_presentation import (
                    COMPACTION_MARKER_NOTICE,
                )
                from local_operator.tui.widgets.transcript import NoticeBlock

                # Drive the reader's OWN gesture rather than the transport:
                # rows only enter the view through the app's paging path, so
                # calling the session directly would fetch pages the frame
                # never shows.
                marker = None
                for _ in range(600):
                    blocks = app._transcript_view().blocks()
                    marker = next(
                        (
                            block
                            for block in blocks
                            if isinstance(block, NoticeBlock)
                            and getattr(block, "text", None)
                            and block.text() == COMPACTION_MARKER_NOTICE
                        ),
                        None,
                    )
                    if marker is not None or not app._resume_pending_head:
                        break
                    app._mount_older_resume_page()
                    await pilot.pause()
                if marker is None:
                    raise SystemExit("no compaction marker mounted")
                # Put the boundary a few rows below the top of the frame, so
                # the shot shows what a reader crossing it actually sees: live
                # conversation above, the marker, pre-compaction rows below.
                view = app._transcript_view()
                for _ in range(10):
                    await pilot.pause()
                # A few rows ABOVE the boundary, so the frame shows both sides
                # of it: pre-compaction history, the marker, then rows the
                # agent can still see.
                view.scroll_to(
                    y=max(0, marker.virtual_region.y - 10), animate=False, immediate=True
                )
                for _ in range(10):
                    await pilot.pause()
            else:
                # "exhausted" is the claim that the reader reached the JOURNAL's
                # first row, so mount everything and show that row. The other
                # states keep one page pending, because the head notice only
                # exists while rows are still deferred.
                floor = 0 if state == "exhausted" else 1
                while len(app._resume_pending_head) > floor:
                    app._mount_older_resume_page()
                    await pilot.pause()
                app._transcript_view().scroll_to(y=0, animate=False, immediate=True)
                for _ in range(10):
                    await pilot.pause()

            app._reconcile_head_notice()
            for _ in range(10):
                await pilot.pause()

            notice = app._resume_head_notice
            print(
                f"[{state}] head notice: "
                f"{notice.text() if notice is not None else '<none>'!r} "
                f"| audit={remote.history_is_audit} "
                f"| more={bool(remote.history_before_token)}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{state}.{size[0]}x{size[1]}"
            save_capture(app, str(out_dir / f"{stem}.svg"))
            # Second frame after a further settle. A static state must produce
            # a byte-identical pair; a difference is a reflow the reader sees
            # as motion, and the evidence has to be able to show that.
            for _ in range(30):
                await pilot.pause()
            save_capture(app, str(out_dir / f"{stem}.settled.svg"))
    finally:
        await remote.dispose()
        server.close()
        await handle.dispose()


async def main() -> None:
    out_dir = Path(sys.argv[1])
    requested = sys.argv[2] if len(sys.argv) > 2 else "all"
    geometry = sys.argv[3] if len(sys.argv) > 3 else "100x34"
    columns, _, rows = geometry.partition("x")
    size = (int(columns), int(rows))
    states = ["context", "audit", "exhausted", "marker"] if requested == "all" else [requested]
    for state in states:
        await capture(out_dir, state, size)


if __name__ == "__main__":
    asyncio.run(main())
