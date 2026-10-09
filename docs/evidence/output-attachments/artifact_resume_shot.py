"""Capture a /resume of the output-attachment rig session (attachment-lane evidence).

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        <this-file> <rig-sessions-dir>/<session-id> out.svg

WHY THIS EXISTS
---------------
The harness PR's TUI claim: a stored session whose transcript carries an
``AttachmentContent`` block must re-mount the picture on resume, next to the
two legacy image shapes — so the frame must be of the REAL resumed session
(read from the real transcript + store), not of a hand-assembled block list.
The console surface photographs the real pty's grid, which is the primary
instrument; this is the compositor's frame for the same state (the console's
offscreen reconstruction can be unavailable for a full-screen TUI surface),
taken by the repo's own ``save_capture`` helper.

The session and the store are COPIED into the isolated root ``isolate_capture``
installs, so the rig is never the operator's live config and the capture run
cannot write to it.
"""

import asyncio
import json
import re
import shutil
import sys
from pathlib import Path

WORKTREE = str(Path(__file__).resolve().parents[2])
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
    # Small sidecars that make an old session authentic (title, birth, origin);
    # absent ones are simply absent.
    for sidecar in ("created_at.json", "title.json", "origin.json"):
        if (source / sidecar).exists():
            shutil.copyfile(source / sidecar, target / sidecar)

    # The digest store lives at the CONFIG ROOT (shared across sessions). Copy
    # ONLY the digests this transcript references — not the whole store — so a
    # frame captured against a real old session does not drag every other
    # session's media along, and a missing digest still degrades to the
    # unavailable receipt (the honest reading for a pruned store).
    digest_re = re.compile(r'"attachment"\s*:\s*"([0-9a-f]{32})"')
    digests = set(digest_re.findall((source / "transcript.jsonl").read_text(encoding="utf-8")))
    store_src = source.parent.parent / "attachments"
    store_dst = sandbox / "attachments"
    store_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for digest in sorted(digests):
        for name in (f"{digest}.bin", f"{digest}.json"):
            src = store_src / name
            if src.exists():
                shutil.copyfile(src, store_dst / name)
                copied += 1
    print(f"store: {len(digests)} digest(s) referenced, {copied} file(s) copied")

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

        # The tool card + its artifact sit at the TOP of this short transcript;
        # land there so the frame shows them first (the app opens at the bottom).
        view.scroll_to(y=0, animate=False)
        for _ in range(10):
            await pilot.pause()
        save_capture(app, out)
        print(f"captured {out}")


asyncio.run(main())
