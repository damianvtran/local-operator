"""Render the boot frame after a startup cleanup pass removed sessions.

    .venv/bin/python scripts/cleanup_notice_shot.py out.svg [100x30]

Seeds a marked store with 15 real 40-day transcripts and 3 empty directories
under the ``isolate_capture`` sandbox, enables ``session.cleanup`` with
``max_inactive_days: 30`` and ``remove_empty: true``, boots the REAL
``OperatorApp`` over a session built by the REAL ``create_session`` (so the
real maintenance pass runs, in the background, after the first frame), and
lets the app's OWN recheck timer find the record — nothing here calls
``_report_startup_cleanup`` by hand. A hand call proved the wrong delivery
path in round 3 (design D11: the timer-delivered block was 3 rows taller than
its content and scrolled the wordmark off at 80x24), so the script drives the
path the user gets and asserts its geometry: the notice's region height must
equal its content height, and the wordmark must still be in the frame.

Boot 1's frame must carry the "session cleanup removed 5 sessions at launch"
notice (5 by max_inactive_days: the 15 transcripts less the picker's recent
10; the 3 empties go too but never-active removals do not announce, N6). Boot
2 must show no cleanup notice (the record was acknowledged, and boot 1's own
idle launch directory is a never-active removal). A ``.next.svg`` is captured
one frame after the first so the pair can be diffed for motion.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# FIRST: re-homes HOME/config before the app loads (see scripts/probe_isolation.py).
import scripts.probe_isolation  # noqa: E402, F401
from scripts.visual_capture import save_capture  # noqa: E402


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
            "session": {
                "cleanup": {"enabled": True, "max_inactive_days": 30, "remove_empty": True}
            },
        }
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("size", nargs="?", default="100x30")
    args = parser.parse_args()
    width, height = (int(part) for part in args.size.split("x"))
    size = (width, height)
    out = args.out

    from local_operator.paths import config_dir
    from local_operator.session_factory import (
        await_store_maintenance_for_tests,
        create_session,
        reset_store_maintenance_for_tests,
    )
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.transcript import NoticeBlock

    root = config_dir()
    root.mkdir(parents=True, exist_ok=True)
    _seed(root)

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager

    def args_ns() -> argparse.Namespace:
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
            args_ns(),
            ConfigManager(root),
            CredentialManager(root),
            AgentRegistry(root),
            has_ui=True,
            defer_mcp_wiring=True,
        )

    failures: list[str] = []
    for boot in (1, 2):
        reset_store_maintenance_for_tests()
        app = OperatorApp(factory)
        async with app.run_test(size=size) as pilot:  # type: ignore[arg-type]
            for _ in range(40):
                await pilot.pause()
                if app._session is not None:
                    break
            await await_store_maintenance_for_tests()
            # THE REAL PATH: the app's own recheck timer. Wait for it, never
            # call the report by hand (D11).
            blocks: list[NoticeBlock] = []
            for _ in range(60):
                await asyncio.sleep(0.1)
                await pilot.pause()
                blocks = [
                    b for b in app.query(NoticeBlock) if "session cleanup" in (b.text() or "")
                ]
                if blocks:
                    break
            await pilot.pause()
            target = out if boot == 1 else out.replace(".svg", ".second.svg")
            save_capture(app, target)
            if boot == 1:
                # Consecutive frame: anything that settles between these two
                # is motion the user sees (AGENTS.md "Visual validation" §5).
                await pilot.pause()
                save_capture(app, out.replace(".svg", ".next.svg"))
            print(f"boot {boot}: {len(blocks)} cleanup notice block(s)")
            for block in blocks:
                content = block.get_content_height(
                    block.size, block.container_size, block.size.width
                )
                print(
                    f"   region.height={block.region.height} content={content} "
                    f"width={block.size.width} styles.height={block.styles.height}"
                )
                if block.region.height != content:
                    failures.append(
                        f"boot {boot}: region {block.region.height} != content {content}"
                    )
                print("  ", (block.text() or "")[:160])
            # The wordmark is the top of the splash: off-frame means the
            # notice's dead rows scrolled the composition (D11).
            from local_operator.tui.widgets.welcome import WelcomeView

            for splash in app.query(WelcomeView):
                y = splash.region.y
                print(f"   splash y={y} welcome={app._welcome_visible}")
                if boot == 1 and y < 0:
                    failures.append(f"boot {boot}: splash scrolled off (y={y})")
            remaining = sum(1 for p in (root / "sessions").iterdir() if p.is_dir())
            print(f"boot {boot}: {remaining} session dirs remain")
            if boot == 1 and not blocks:
                failures.append("boot 1: no notice via the timer path")
            if boot == 2 and blocks:
                failures.append("boot 2: the notice repeated")
    if failures:
        print("FAIL:", *failures, sep="\n  ")
        sys.exit(1)


asyncio.run(main())
