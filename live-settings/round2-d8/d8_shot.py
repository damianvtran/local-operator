"""D8 evidence: the approvals KEEP path, detached and attached.

Drives the REAL ``OperatorApp`` through the REAL ``/approvals ask`` gesture and
a REAL another-process ``config.yml`` write, in both topologies:

  detached  — the TUI owns the gate (FakeSession, is_remote False)
  attached  — a production runtime owns it, the TUI is a viewer over
              ``RemoteSession``, and BOTH carriers hold ``"ask"``

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/d8_shot.py <outdir> <cols>
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO = "/private/tmp/lop-live-settings"
sys.path.insert(0, REPO)

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()
CONFIG = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
CONFIG.mkdir(parents=True, exist_ok=True)

from local_operator import settings_io  # noqa: E402
from local_operator.config import ConfigManager  # noqa: E402
from local_operator.config_watch import _reset_for_tests, process_watcher  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

OUT = Path(sys.argv[1])
COLS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT.mkdir(parents=True, exist_ok=True)


def write_elsewhere(config_dir, key, value) -> None:
    setting = settings_io.resolve_key(key)
    settings_io._store(ConfigManager(config_dir), setting.path, value)


def notices(app) -> list[str]:
    return [b.text() or "" for b in app.query(NoticeBlock)]


async def adopted(app, pilot) -> None:
    for _ in range(300):
        if app._session is not None and app._unsubscribe_config_watch is not None:
            return
        await pilot.pause()
    raise AssertionError("never adopted")


def report(tag: str, app) -> None:
    rows = [n for n in notices(app) if "approvals" in n or "config.yml changed" in n]
    print(f"\n=== {tag} ===")
    for row in rows:
        print(f"  · {row}")
    print(f"  receipts(keep)={sum('keeping tool approvals' in r for r in rows)} "
          f"applied_lines={sum('config.yml changed' in r for r in rows)} "
          f"gate_auto={app._approve_all}")


async def detached(cols: int) -> None:
    _reset_for_tests()
    cfg = CONFIG / f"detached{cols}"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(cfg)
    ConfigManager(cfg).set_config_value("hosting", "")
    write_elsewhere(cfg, "tool_approval_mode", "ask")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(cols, 24)) as pilot:
        await adopted(app, pilot)
        app._cmd_approvals("ask", app._notice)          # the real gesture
        await pilot.pause()
        write_elsewhere(cfg, "tool_approval_mode", "auto")  # another process
        process_watcher(cfg).poll_now()
        await pilot.pause()
        await pilot.pause()
        report(f"DETACHED @{cols}", app)
        save_capture(app, OUT / f"detached-{cols}.svg")
    _reset_for_tests()


async def attached(cols: int) -> None:
    _reset_for_tests()
    from local_operator.session.remote import RemoteSession
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.owned import OwnedSessionHandle, attach_gate_config_watch
    from local_operator.session.runtime.server import RuntimeServer
    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    cfg = CONFIG / f"attached{cols}"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(cfg)
    ConfigManager(cfg).set_config_value("hosting", "")
    write_elsewhere(cfg, "tool_approval_mode", "ask")

    directory = cfg / "sessions" / "d8drive001"
    directory.mkdir(parents=True)
    session = build_session(directory, ScriptedStream([text_turn("ok")] * 4))
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    attach_gate_config_watch(handle, cfg)
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()

    record = None
    for _ in range(200):
        for rec, _state in registry.scan(cfg):
            if getattr(rec, "session_id", "") == session.session_id:
                record = rec
                break
        if record:
            break
        await asyncio.sleep(0.05)
    assert record is not None, "no record published"

    async def _never():
        raise AssertionError("a viewer must never take over")

    viewer = await RemoteSession.connect(
        record, session.session_id, config_dir=cfg, takeover_factory=_never
    )
    async def _viewer_factory():
        return viewer

    app = OperatorApp(_viewer_factory)
    try:
        async with app.run_test(size=(cols, 24)) as pilot:
            await adopted(app, pilot)
            # The real gesture through the real dispatcher: routes to the
            # runtime AND records this viewer's own explicit mode.
            app._run_slash_command("/approvals ask")
            for _ in range(60):
                await pilot.pause()
                await asyncio.sleep(0.02)
            print(f"  viewer._explicit_approvals_mode={app._explicit_approvals_mode!r} "
                  f"runtime._explicit_approvals_mode={handle._explicit_approvals_mode!r}")

            write_elsewhere(cfg, "tool_approval_mode", "auto")
            process_watcher(cfg).poll_now()
            for _ in range(60):
                await pilot.pause()
                await asyncio.sleep(0.02)
            report(f"ATTACHED @{cols}", app)
            save_capture(app, OUT / f"attached-{cols}.svg")
    finally:
        await viewer.dispose()
        await server.aclose()
        _reset_for_tests()


async def attached_double(cols: int) -> None:
    """The designer's exact topology: BOTH carriers hold ``ask``.

    Two real production gestures, in the order a human performs them:
    ``/approvals ask`` typed in the viewer ROUTES to the runtime and records
    the runtime's carrier (``owned.py:2625``); a ``/settings`` page write in
    the viewer's own pane records the VIEWER's carrier (``app.py:14621``).
    A third process then loosens the file, and both keep-branches see ``ask``.
    """
    _reset_for_tests()
    from local_operator.session.remote import RemoteSession
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.owned import OwnedSessionHandle, attach_gate_config_watch
    from local_operator.session.runtime.server import RuntimeServer
    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    cfg = CONFIG / f"double{cols}"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(cfg)
    ConfigManager(cfg).set_config_value("hosting", "")
    write_elsewhere(cfg, "tool_approval_mode", "auto")  # both boot auto

    directory = cfg / "sessions" / "d8double01"
    directory.mkdir(parents=True)
    session = build_session(directory, ScriptedStream([text_turn("ok")] * 4))
    handle = OwnedSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(directory), auto_approve=True
    )
    attach_gate_config_watch(handle, cfg)
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()

    record = None
    for _ in range(200):
        for rec, _state in registry.scan(cfg):
            if getattr(rec, "session_id", "") == session.session_id:
                record = rec
                break
        if record:
            break
        await asyncio.sleep(0.05)
    assert record is not None

    async def _never():
        raise AssertionError("a viewer must never take over")

    viewer = await RemoteSession.connect(
        record, session.session_id, config_dir=cfg, takeover_factory=_never
    )

    async def _viewer_factory():
        return viewer

    app = OperatorApp(_viewer_factory)
    try:
        async with app.run_test(size=(cols, 24)) as pilot:
            await adopted(app, pilot)
            app._run_slash_command("/approvals ask")   # routes → runtime carrier
            for _ in range(60):
                await pilot.pause()
                await asyncio.sleep(0.02)
            # EXACTLY what the /settings page does: the facade write, local notify.
            setting = settings_io.resolve_key("tool_approval_mode")
            settings_io.write_setting(ConfigManager(cfg), setting, "ask")
            for _ in range(30):
                await pilot.pause()
                await asyncio.sleep(0.02)
            print(f"  viewer carrier={app._explicit_approvals_mode!r} "
                  f"runtime carrier={handle._explicit_approvals_mode!r} "
                  f"viewer gate_auto={app._approve_all} runtime gate_auto={handle._auto_approve}")

            write_elsewhere(cfg, "tool_approval_mode", "auto")
            process_watcher(cfg).poll_now()
            for _ in range(60):
                await pilot.pause()
                await asyncio.sleep(0.02)
            report(f"ATTACHED-DOUBLE @{cols}", app)
            save_capture(app, OUT / f"attached-double-{cols}.svg")
    finally:
        await viewer.dispose()
        await server.aclose()
        _reset_for_tests()


async def main() -> None:
    await detached(COLS)
    await attached(COLS)
    await attached_double(COLS)


asyncio.run(main())
