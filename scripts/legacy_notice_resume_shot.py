"""Capture a real ``/resume`` over a transcript that carries legacy notices.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/legacy_notice_resume_shot.py <session-dir> out.svg [page_backs]

WHY THIS EXISTS
---------------
A harness notice written BEFORE the ``harness_injected`` stamp existed is a plain
``role="user"`` row with no provenance at all, and a compaction pass lifted those
rows into its marker's preserved block as "user turns". Replaying that marker
re-seats them as user rows, so the reader saw the harness's own words behind the
user gutter — the reported symptom, on the session it was reported from — and a
fix that only reads the stamp cannot see them. This script drives the assembled
application (``/resume``, then the scroll-up gesture that pages older history in)
against a COPY of such a transcript and reports what each frame actually paints.

The transcript is read from a directory you name, never from the operator's live
session: copy it first (``cp -c`` on APFS is byte-identical and cheap) and pass
the copy. ``isolate_capture()`` redirects HOME and the config dir before any app
import, so the run cannot write to a real session either way.

Output is the frame at ``out.svg`` plus a printed row census per page: the total
painted user rows and how many of them are harness notices.
"""

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.paths import config_dir as app_config_dir  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    TranscriptView,
    UserBlock,
)
from tests.e2e.harness import ScriptedStream, build_session  # noqa: E402
from tests.unit.tui.test_app_pilot import _renderable_plain  # noqa: E402

#: The switch-notice head this script measures, as a literal rather than through
#: the product's recognition rule, on purpose: the instrument has to run against a
#: tree that PREDATES the rule, which is what a before-frame is taken from.
MEASURED_NOTICE_HEAD = "[model switch] "

#: The elision notice's own phrasing. It opens with a bracketed COUNT, so it has
#: no fixed head; these are the two sentences ``elision_notice_text`` writes.
_ELISION_PHRASES = (
    "were dropped here to bound the context",
    "were also dropped; these were not authored by the user",
)


#: Frames to pump after each page-back before reading the census: the mount is a
#: worker plus a gap-settle, and a fixed pump count is what produced a
#: half-painted census the first time this was written.
FILL_CYCLES = 30


def _is_measured_notice(text: str) -> bool:
    # The user gutter (``▌ ``) rides EVERY wrapped line of the block, so it is
    # removed wherever it appears, not just at the head: a phrase split by the
    # gutter is the one shape a raw substring test misses. Whitespace is
    # flattened for the same reason — the painted text is wrapped.
    flattened = " ".join(text.replace("▌", " ").split())
    if flattened.startswith(MEASURED_NOTICE_HEAD):
        return True
    return flattened.startswith("[") and any(phrase in flattened for phrase in _ELISION_PHRASES)


def census(app: OperatorApp) -> tuple[int, int]:
    """``(painted user rows, of which harness notices)`` on the mounted frame."""
    view = app.query_one(TranscriptView)
    users = [b for b in view.blocks() if isinstance(b, UserBlock)]
    notices = [
        block
        for block in users
        if _is_measured_notice(_renderable_plain(getattr(block, "renderable", "")))
    ]
    return len(users), len(notices)


def _block_content_offset(view: TranscriptView, block: Any) -> int:
    """The content-space scroll offset that puts ``block`` at the viewport top.

    ``Widget.region`` is in SCREEN coordinates, so the content offset has to be
    reconstructed from the view's own scroll position: a block scrolled above
    the viewport reports a negative ``region.y``, and one below reports a
    positive one, both relative to the same content origin.
    """
    return int(view.scroll_y + block.region.y - view.content_region.y)


async def _pump(pilot, cycles: int) -> None:
    for _ in range(cycles):
        await pilot.pause()


async def main() -> None:
    source, out = Path(sys.argv[1]), sys.argv[2]
    page_backs = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    #: Land the viewport here (content-space) before the final capture. Passed
    #: explicitly for an after-frame, so the pair shows the SAME place in the
    #: conversation: the offset is read off the before run's own report.
    content_offset = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    #: …or land on a ROW, which is the better pairing for a frame whose subject
    #: is GONE on the after side: a numeric offset belongs to the content that
    #: used to be there, so after a removal it points at different rows. An
    #: anchor row exists in both trees, which makes the two frames a comparison
    #: of the same window rather than of the same number.
    anchor_text = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
    if source.resolve() == Path.home() / ".local-operator":
        raise SystemExit("refusing to read the live config dir; pass a copy")

    # The isolated root ``isolate_capture`` installed, so the copy lands inside
    # the sandbox the app is already pointed at.
    sandbox = Path(app_config_dir())
    target = sandbox / "sessions" / "resume-target"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("transcript.jsonl",):
        shutil.copyfile(source / name, target / name)
    for extra in ("attachments", "attachments.json"):
        if (source / extra).exists():
            if (source / extra).is_dir():
                shutil.copytree(source / extra, target / extra, dirs_exist_ok=True)
            else:
                shutil.copyfile(source / extra, target / extra)

    live_dir = sandbox / "sessions" / "live"
    live = build_session(live_dir, ScriptedStream([]), cwd=sandbox)
    resumed = build_session(target, ScriptedStream([]), cwd=sandbox)

    async def factory():
        return live

    async def resume_factory(_resume_id: str | None):
        return resumed

    app = OperatorApp(factory, resume_factory=resume_factory)
    async with app.run_test(size=(120, 44)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
        await _pump(pilot, 10)

        app._resume_session("resume-target", lambda *_a, **_k: None)
        await _pump(pilot, 60)

        users, notices = census(app)
        print(f"first resume frame: user rows={users} notice rows={notices}")
        save_capture(app, out)

        view = app.query_one(TranscriptView)
        for page in range(1, page_backs + 1):
            # The gesture the reader makes: input earns demand, then the scroll
            # lands in the trigger zone and the app mounts one older page.
            view.note_user_scroll()
            view.scroll_to(y=0, animate=False)
            await _pump(pilot, 5)
            app._check_resume_page()
            await _pump(pilot, FILL_CYCLES)
            users, notices = census(app)
            print(f"after page-back {page}: user rows={users} notice rows={notices}")
        if page_backs:
            view = app.query_one(TranscriptView)
            blocks = list(view.blocks())
            if content_offset is None and anchor_text:
                anchor = next(
                    (
                        block
                        for block in blocks
                        if anchor_text in _renderable_plain(getattr(block, "renderable", ""))
                    ),
                    None,
                )
                if anchor is None:
                    raise SystemExit(f"anchor row not mounted: {anchor_text!r}")
                content_offset = _block_content_offset(view, anchor)
                print(f"anchor {anchor_text!r} at content offset {content_offset}")
            if content_offset is None:
                # No offset asked for: land on the first notice row this frame
                # paints, which is the subject of the pair, and REPORT the
                # offset so the after run can be landed on the same place.
                notice = next(
                    (
                        block
                        for block in blocks
                        if isinstance(block, UserBlock)
                        and _is_measured_notice(_renderable_plain(getattr(block, "renderable", "")))
                    ),
                    None,
                )
                if notice is not None:
                    content_offset = _block_content_offset(view, notice)
                    # PRINT the row above it too: that row survives the fix, so
                    # it is the anchor an after-frame can actually be landed on.
                    index = blocks.index(notice)
                    for neighbour in blocks[max(0, index - 2) : index]:
                        text = _renderable_plain(getattr(neighbour, "renderable", ""))
                        print(f"row above: {text.splitlines()[0][:70] if text else text!r}")
            if content_offset is not None:
                # Twice, with a settle between: a page mount that lands during
                # the first pump re-anchors the viewport to keep the reader's
                # position, and that animation is what left the subject of the
                # frame off-screen on the first attempt.
                for _ in range(2):
                    view.scroll_to(y=content_offset, animate=False)
                    await _pump(pilot, 12)
                print(f"landed at content offset {content_offset} (scroll_y={view.scroll_y})")
            save_capture(app, out)

    await live.dispose()
    await resumed.dispose()


asyncio.run(main())
