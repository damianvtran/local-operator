"""Render the boot frame after a startup cleanup pass removed sessions.

    .venv/bin/python scripts/cleanup_notice_shot.py out.svg [100x30]

Seeds a marked store with 15 real 40-day transcripts and 3 empty directories
under the ``isolate_capture`` sandbox, enables ``session.cleanup`` with
``remove_empty: true``, boots the REAL ``OperatorApp`` over a session built by
the REAL ``create_session`` (so the real maintenance pass runs, in the
background, after the first frame), waits for the pass, and captures. The
frame must carry the "session cleanup removed 3 sessions at launch" notice;
the same driver with ``--second-boot`` boots again and must NOT repeat it.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()


def _seed(config_dir: Path) -> None:
    from local_operator.config import ConfigManager
    from local_operator.session.cleanup import mark_store

    sessions = config_dir / "sessions"
    mark_store(sessions)
    old = time.time() - 40 * 86400
    for index in range(15):
        directory = sessions / f"s{index:02d}"
        directory.mkdir()
        (directory / "transcript.jsonl").write_text(
            '{"type":"message","payload":{"role":"user","content":[{"type":"text","text":"t"}]}}\n'
        )
        os.utime(directory / "transcript.jsonl", (old + index, old + index))
    for index in range(3):
        directory = sessions / f"e{index:02d}"
        directory.mkdir()
        os.utime(directory, (old, old))
    ConfigManager(config_dir).update_config(
        {
            "hosting": "test",
            "model_name": "test-model",
            "session": {"cleanup": {"enabled": True, "remove_empty": True}},
        }
    )


async def main() -> None:
    import argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.paths import config_dir
    from local_operator.session_factory import (
        await_store_maintenance_for_tests,
        create_session,
        reset_store_maintenance_for_tests,
    )
    from local_operator.tui.app import OperatorApp

    out = sys.argv[1]
    size = tuple(int(part) for part in (sys.argv[2] if len(sys.argv) > 2 else "100x30").split("x"))
    root = config_dir()
    root.mkdir(parents=True, exist_ok=True)
    _seed(root)

    def args() -> argparse.Namespace:
        return argparse.Namespace(
            hosting="test",
            model="test-model",
            agent_name=None,
            agent_id=None,
            yolo=True,
            train=False,
        )

    async def factory():
        return await create_session(
            args(),
            ConfigManager(root),
            CredentialManager(root),
            AgentRegistry(root),
            has_ui=True,
            defer_mcp_wiring=True,
        )

    for boot in (1, 2):
        reset_store_maintenance_for_tests()
        app = OperatorApp(factory)
        async with app.run_test(size=size) as pilot:  # type: ignore[arg-type]
            for _ in range(40):
                await pilot.pause()
                if app._session is not None:
                    break
            await await_store_maintenance_for_tests()
            await asyncio.sleep(0.2)
            app._report_startup_cleanup()  # what the 5 s timer does; not waiting 5 s here
            await pilot.pause()
            await pilot.pause()
            target = out if boot == 1 else out.replace(".svg", ".second.svg")
            save_capture(app, target)
            if boot == 1:
                # Consecutive frame: anything that settles between these two
                # is motion the user sees (AGENTS.md "Visual validation" §5).
                await pilot.pause()
                save_capture(app, out.replace(".svg", ".next.svg"))
            from local_operator.tui.widgets.transcript import NoticeBlock

            texts = [block.text() or "" for block in app.query(NoticeBlock)]
            print(f"boot {boot}: {len(texts)} notice block(s)")
            for text in texts:
                print("  ", str(text)[:160])
            remaining = sum(1 for p in (root / "sessions").iterdir() if p.is_dir())
            print(f"boot {boot}: {remaining} session dirs remain")


asyncio.run(main())
