"""Render the MISSING-STORE receipt state for the output-attachment TUI mount.

Variant of the producer's artifact_resume_shot.py: the session's transcript is
copied, but the content-addressed attachment store is deliberately NOT — so the
artifact's digest does not resolve and the ImageBlock paints its own
unavailable receipt. That is the degraded state the design doc promises
("a missing store degrades visibly... never silently"), and the state the
design review wants a frame of.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        <this-file> <rig-session-dir> out.svg
"""

import asyncio
import shutil
import sys
from pathlib import Path

WORKTREE = "/Users/damian/local-operator-worktrees/output-attachments-1008"
sys.path.insert(0, WORKTREE)

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()  # BEFORE app imports: isolate HOME, config and caches

from local_operator.paths import config_dir as app_config_dir  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.image_block import ImageBlock  # noqa: E402
from local_operator.tui.widgets.transcript import TranscriptView  # noqa: E402
from tests.e2e.harness import ScriptedStream, build_session  # noqa: E402


async def main() -> None:
    source, out = Path(sys.argv[1]).resolve(), sys.argv[2]
    if source == Path.home() / ".local-operator":
        raise SystemExit("refusing to read the live config dir; pass the rig copy")

    sandbox = Path(app_config_dir())
    target = sandbox / "sessions" / "resume-target"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "transcript.jsonl", target / "transcript.jsonl")

    # NOTE: the attachments store is intentionally NOT copied — digest unresolved.

    live = build_session(sandbox / "sessions" / "live", ScriptedStream([]), cwd=sandbox)
    resumed = build_session(target, ScriptedStream([]), cwd=sandbox)

    async def factory():
        return live

    async def resume_factory(_resume_id: str | None):
        return resumed

    app = OperatorApp(factory, resume_factory=resume_factory)
    async with app.run_test(size=(140, 90)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
        for _ in range(10):
            await pilot.pause()

        app._resume_session("resume-target", lambda *_a, **_k: None)
        for _ in range(60):
            await pilot.pause()

        view = app.query_one(TranscriptView)
        blocks = list(view.blocks())
        images = [b for b in blocks if isinstance(b, ImageBlock)]
        print(f"mounted blocks: {len(blocks)}; image blocks: {len(images)}")
        for image in images:
            info = getattr(image, "image_info", None)
            print(f"  image block: {info!r}")

        view.scroll_to(y=0, animate=False)
        for _ in range(10):
            await pilot.pause()
        save_capture(app, out)
        print(f"captured {out}")


asyncio.run(main())
