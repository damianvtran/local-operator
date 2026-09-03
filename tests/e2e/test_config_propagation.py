"""End-to-end: an edit made by ANOTHER PROCESS reaches the running TUI.

The unit suites drive the watcher's tick by hand. This is the assembled path:
the real ``OperatorApp`` over a real ``Session`` in this process, the
production watcher started by the app on its own loop, and the edit made the
way a user makes it — ``lop config edit`` running as a separate process
against the same config directory. Nothing in this process touches the
watcher after boot; the change has to arrive through the file.

Two things are asserted, both observed in the running process: the session's
in-memory ``_compaction_settings`` moved to the value the other process wrote,
and one notice naming the key landed in the transcript.

Timing: waits on an EVENT (a test-only listener on the app's watcher sets a
signal), never on the 2 s poll cadence — the wait lasts as long as the poll
takes, and only a genuine hang reaches the deadlock guard. On macOS the kqueue
accelerator usually delivers it well under the interval; on Linux the poll
alone does, which is exactly why the CI matrix's Linux leg matters here — it
proves the mechanism without the accelerator.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.config_watch import process_watcher
from local_operator.session.session import Session
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    wait_for_adoption,
)
from tests.e2e.test_tui_e2e import BOOT_BOUND_S, SCREEN
from tests.e2e.watchdog import bounded
from tests.unit.harness.test_comms import ChangeSignal, wait_for

#: Bound on the cross-process round trip: spawn ``lop config edit`` (a full
#: interpreter start plus the CLI's imports, ~1 s), the watcher's ≤2 s poll,
#: and the app's paint. Order-of-magnitude headroom for a loaded runner, for
#: the same reason ``BOOT_BOUND_S`` carries it: this exists to catch a change
#: that NEVER arrives, not to police latency.
PROPAGATION_BOUND_S = 60.0


@pytest.mark.asyncio
async def test_a_config_edit_from_another_process_reaches_the_running_session(
    headless_tui_env: Path,
) -> None:
    from local_operator.compaction.thresholds import CompactionSettings

    # A real file to diff against, at the default threshold.
    ConfigManager(headless_tui_env).set_config_value("hosting", "")

    session = build_session(
        headless_tui_env / "sessions" / "config-watch",
        ScriptedStream([]),
        cwd=headless_tui_env,
    )
    # What the factory does for a production session; the harness's
    # ``build_session`` composes the Session directly and so does not.
    watcher = process_watcher(headless_tui_env)
    session._compaction_settings = CompactionSettings()
    session.add_dispose_hook(watcher.subscribe(session._apply_config_change))

    def threshold() -> float:
        settings = session._compaction_settings
        assert isinstance(settings, CompactionSettings)
        return settings.threshold_percent

    assert threshold() != 0.5

    async def factory() -> Session:
        return session

    app = OperatorApp(factory)
    signal = ChangeSignal()
    unsubscribe = None
    try:
        with bounded(BOOT_BOUND_S, "TUI boot"):
            async with app.run_test(size=SCREEN) as pilot:
                await wait_for_adoption(app, pilot)
                # The app started the process watcher on its loop during
                # adoption. A test-only listener is the event we wait on.
                assert watcher._task is not None and not watcher._task.done()
                unsubscribe = watcher.subscribe(signal._fire)

                with bounded(PROPAGATION_BOUND_S, "cross-process config propagation"):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "local_operator.cli",
                            "config",
                            "edit",
                            "compaction.threshold_percent",
                            "0.5",
                        ],
                        env={
                            **_inherited_env(),
                            "LOCAL_OPERATOR_CONFIG_DIR": str(headless_tui_env),
                        },
                        capture_output=True,
                        text=True,
                        timeout=PROPAGATION_BOUND_S / 2,
                    )
                    assert completed.returncode == 0, completed.stderr
                    assert "Successfully updated compaction.threshold_percent" in completed.stdout

                    await wait_for(
                        lambda: threshold() == 0.5,
                        signal=signal,
                    )
                    # The notice is appended on the same listener pass as the
                    # session apply; one pump lets the widget mount.
                    await pilot.pause()
                    notices = [block.text() or "" for block in app.query(NoticeBlock)]
                    assert any(
                        "config.yml changed: applied: compaction.threshold_percent" in text
                        for text in notices
                    ), notices
    finally:
        signal.close()
        if unsubscribe is not None:
            unsubscribe()
        await dispose_quietly(session)


def _inherited_env() -> dict[str, str]:
    """The parent's environment, minus anything that would make the child's
    CLI chatty or reach outside the sandbox. ``PATH``/``HOME`` etc. stay so the
    interpreter and its site-packages resolve exactly as they do here.

    The repository root is put FIRST on the child's ``PYTHONPATH`` so it
    imports the same tree this test runs from. In a parallel-agent worktree
    the shared editable ``.venv`` resolves ``local_operator`` to the main
    checkout, and the child would then run a ``config edit`` from a different
    revision than the app under test — a real cross-version mismatch, but not
    the one this test is about. CI installs the package, where this is a
    no-op.
    """
    import os

    env = dict(os.environ)
    env.pop("NO_COLOR", None)
    root = str(Path(__file__).resolve().parents[2])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing else os.pathsep.join((root, existing))
    return env
