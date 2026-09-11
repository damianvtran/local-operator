"""Rendered frames of the older-page failure surface, before and after the fix.

Drives the REAL ``OperatorApp`` against a REAL owner runtime over the real
socket, then makes one backward page fail the two ways production does, and
captures what the reader sees in the notice area:

``moved``
    A canonical refresh replaces the display window WHILE the page request is
    in flight — the ``history changed while paging`` shape from the operator's
    report (a compaction or canonical refresh landing mid-scroll). Pre-fix this
    painted a red internal error; post-fix the page is re-issued against the
    fresh window silently.
``disconnected``
    The owner connection is down when the reader pages up — the ``not
    attached`` / ``owner connection lost`` shapes. Pre-fix a red internal
    error; post-fix one calm, non-error row in the user's vocabulary.

The failure is induced through REAL session machinery in both states —
``_invalidate_display_history`` + ``ensure_display_current`` for the moved
window, a genuinely closed attach client for the disconnection — never by
raising a synthetic exception, so the frames show the classification the
shipped code performs, not a fixture that cured the bug as a side effect.

Usage::

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \\
        scripts/older_page_failure_shot.py <out-dir> [state] [COLSxROWS]

States: ``moved``, ``disconnected``, or ``all``. The geometry is in the
filename on purpose (see ``scripts/audit_history_shot.py``): two captures at
different sizes must not silently overwrite each other. Each state writes a
``.top.`` and a ``.tail.`` frame — a failed page reports at the tail while a
successful one mounts rows at the top, and a single frame cannot show both.
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

from textual.events import MouseScrollUp  # noqa: E402

from local_operator.harness.types import Message, TextContent  # noqa: E402
from local_operator.session.attached import AttachedSession  # noqa: E402
from local_operator.session.runtime.server import RuntimeServer  # noqa: E402
from local_operator.session.runtime.serving import ServingSessionHandle  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.e2e.harness import ScriptedStream, build_session, text_turn  # noqa: E402
from tests.unit.session.test_remote import _never_take_over  # noqa: E402


async def _seed(directory: Path, rows: int) -> None:
    """A journal long enough that the reader pages several times to the top."""
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    for index in range(rows):
        user = index % 2 == 0
        await transcript.append_message(
            Message(
                id=f"shot-row-{index:04}",
                role="user" if user else "assistant",
                content=[
                    TextContent(
                        text=(
                            f"Question {index} about the earlier work?"
                            if user
                            else f"Answer {index}: the earlier work held, and here is why."
                        )
                    )
                ],
                stop_reason="stop",
            )
        )
    transcript.flush()


async def _settle(app, pilot, rounds: int = 300) -> None:
    for _ in range(rounds):
        await pilot.pause()
        if (
            app._session is not None
            and app._transcript_view().blocks()
            and not app._resume_paging
            and not app._resume_fill_active
            and not app._resume_check_pending
        ):
            await pilot.pause()
            return
    raise AssertionError("history presentation never settled")


async def capture(out_dir: Path, state: str, size: tuple[int, int] = (100, 34)) -> None:
    root = Path(os.environ["HOME"])
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"older-page-{state}"
    await _seed(directory, rows=600)
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
            await _settle(app, pilot)
            # Drain the mounted reserve the way a deep reader does, so the
            # next upward demand has to FETCH from the owner rather than
            # spend locally held rows. This is the state the operator was in.
            while app._resume_pending_head:
                app._mount_older_resume_page()
                await _settle(app, pilot)
            assert getattr(remote, "history_before_token", None)

            if state == "moved":
                # A canonical refresh replaces the window DURING the page
                # request: the wrapper invalidates (exactly what a compaction
                # event does), lets the real refresh install the new window,
                # then completes the real page read — so load_older's own
                # identity check fires the real `history changed while
                # paging` raise, not a stub. Move ONCE (the script captures a
                # single transient): moving on every attempt would keep
                # exhausting the retry budget by construction, which is the
                # persistent-mover shape the unit regression test drives
                # instead.
                real_history_page = remote.history_page
                moved_once = False

                async def moving_history_page(before: str, *, anchor: str = ""):
                    nonlocal moved_once
                    if not moved_once:
                        moved_once = True
                        remote._invalidate_display_history()
                        await remote.ensure_display_current()
                    return await real_history_page(before, anchor=anchor)

                remote.history_page = moving_history_page  # type: ignore[method-assign]

            elif state == "disconnected":
                # Real disconnection, not a stubbed raise: a closed attach
                # client is exactly the state a dropped owner leaves behind,
                # and it is what makes the failure class recognisable as
                # transport-gone rather than a data fault. The client is known
                # live here — the reserve drain above required it — so the
                # None-guard is only for the type checker.
                assert remote._client is not None
                remote._client.close()

            # The reader's own gesture: travel to the top of what is
            # rendered, then one wheel notch — the exact upstream event a
            # physical scroll-up delivers. ``scroll_to`` alone is not the
            # gesture: it moves the offset without emitting the scroll
            # callback the paging trigger hangs off, so a script that relied
            # on it would capture a frame in which the fetch never ran.
            view = app._transcript_view()
            view.scroll_to(y=0, animate=False, immediate=True)
            await pilot.pause()
            view.post_message(MouseScrollUp(view, 1, 1, 0, -1, 0, False, False, False))
            for _ in range(60):
                await pilot.pause()
            await _settle(app, pilot)
            # Two frames per state, because the evidence lives at opposite
            # ends. A page that FAILS appends its notice at the transcript's
            # TAIL, while a page that SUCCEEDS mounts older rows at the TOP —
            # so the top frame alone would hide a red error behind the fold
            # and the tail frame alone would hide the silently loaded rows.
            # Both are captured, named by end, and the fixed/unfixed trees
            # are told apart by which end carries the difference.
            view = app._transcript_view()
            cols, rows = size
            for _ in range(10):
                await pilot.pause()
            save_capture(app, out_dir / f"{state}.top.{cols}x{rows}.svg")
            view.scroll_to(y=max(0, view.virtual_size.height), animate=False, immediate=True)
            for _ in range(10):
                await pilot.pause()
            save_capture(app, out_dir / f"{state}.tail.{cols}x{rows}.svg")
    finally:
        await remote.dispose()
        server.close()
        await handle.dispose()


async def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    state = sys.argv[2] if len(sys.argv) > 2 else "all"
    size = (100, 34)
    if len(sys.argv) > 3:
        cols, rows = sys.argv[3].split("x")
        size = (int(cols), int(rows))
    out_dir.mkdir(parents=True, exist_ok=True)
    states = ["moved", "disconnected"] if state == "all" else [state]
    for one in states:
        await capture(out_dir, one, size)


if __name__ == "__main__":
    asyncio.run(main())
