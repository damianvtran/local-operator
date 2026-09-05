"""Cold viewers engage the owner before copying; destinations remain local UI policy."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import OperatorApp
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    seed_transcript,
    transcript_text,
    user_message,
    wait_for_adoption,
)
from tests.e2e.test_fork_e2e import _never_take_over, _pump
from tests.e2e.watchdog import bounded


@pytest.mark.asyncio
async def test_cold_snapshot_live_preferences_overrides_and_cancel(
    headless_tui_env: Path, workspace: Path, monkeypatch
) -> None:
    from local_operator.session.runtime import launch
    from local_operator.spawn import registry

    config = headless_tui_env
    directory = config / "sessions" / "coldparent01"
    await seed_transcript(directory, [user_message("cold saved conversation")])
    stream = ScriptedStream([])
    owner = build_session(directory, stream, cwd=workspace)
    handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(workspace))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    engaged = []

    async def engage(sid, *args, **kwargs):
        # The exact in-process runtime above substitutes only process startup;
        # ensure_bound still discovers and authenticates the real socket.
        engaged.append(sid)

    monkeypatch.setattr(launch, "engage_runtime", engage)
    # Real process discovery deliberately excludes our own PID. This owner is
    # hosted in the test's process; only that discovery boundary is supplied.
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_owner_record",
        lambda *_: (server._record, server._record.pid),
    )
    launches = []

    class GuardBackend:
        name = "guard"
        opened_place = "a guarded test window"

        def spawn(self, target, env):
            launches.append(target)
            return True

    monkeypatch.setattr(registry, "active_backend", lambda **kwargs: GuardBackend())
    viewer = await RemoteSession.cold(
        owner.session_id, config_dir=config, cwd=str(workspace), takeover_factory=_never_take_over
    )

    async def factory():
        return viewer

    app = OperatorApp(factory)
    try:
        with bounded(60, "cold fork destination and cancellation"):
            async with app.run_test(size=(60, 24)) as pilot:
                await wait_for_adoption(app, pilot)
                assert viewer.is_cold
                # Explicit preferences are read at invocation, not snapshotted
                # at boot. A switch override on this no-resume host must refuse,
                # not quietly open a window or create an unreachable fork.
                (config / "config.yml").write_text("values:\n  fork:\n    mode: window\n")
                app._cmd_fork("--switch", app._notice)
                assert len(list((config / "sessions").iterdir())) == 1
                app._cmd_fork("--bogus", app._notice)
                assert not launches
                app._cmd_fork("", app._notice)
                await _pump(pilot, lambda: not app._fork_in_progress)
                assert len(launches) == 1, transcript_text(app)
                assert engaged == [owner.session_id]
                assert not viewer.is_cold
                assert not stream.requests
                assert app._session is viewer
                assert launches[0].cwd == str(workspace)
                # An invocation override does not rewrite the saved preference.
                (config / "config.yml").write_text("values:\n  fork:\n    mode: switch\n")
                app._cmd_fork("--window -- --switch is prompt text", app._notice)
                await _pump(pilot, lambda: not app._fork_in_progress)
                assert len(launches) == 2
                from local_operator.fork import consume_boot_prompt

                assert (
                    consume_boot_prompt(config / "sessions" / launches[-1].session_id)
                    == "--switch is prompt text"
                )
                assert app._config_values()["fork"]["mode"] == "switch"
                started, release = asyncio.Event(), asyncio.Event()
                original = owner.fork_snapshot

                async def held(message=""):
                    started.set()
                    await release.wait()
                    return await original(message)

                monkeypatch.setattr(owner, "fork_snapshot", held)
                app._cmd_fork("--window", app._notice)
                await asyncio.wait_for(started.wait(), 20)
                app._cmd_fork("--window", app._notice)
                app.action_stop()
                release.set()
                await _pump(pilot, lambda: not app._fork_in_progress)
                assert len(launches) == 2
                assert len(list((config / "sessions").iterdir())) == 4
                assert app._session is viewer
                assert not stream.requests

                async def fail_resume(sid):
                    raise ConnectionError("synthetic fork startup failure")

                app._resume_factory = fail_resume
                app._cmd_fork("--switch", app._notice)
                await _pump(
                    pilot, lambda: not app._fork_in_progress and not app._session_transition_pending
                )
                assert app._session is None
                painted = " ".join(transcript_text(app).split())
                assert "fork saved but could not open: /resume" in painted
                assert "Return to original: /resume coldparent01" in painted
                assert not owner._disposed
                assert len(launches) == 2
                assert len(list((config / "sessions").iterdir())) == 5

                async def return_to_original(sid):
                    assert sid == owner.session_id
                    return await RemoteSession.connect(
                        server._record, sid, config_dir=config, takeover_factory=_never_take_over
                    )

                app._resume_factory = return_to_original
                app._resume_session(owner.session_id, app._notice)
                await _pump(pilot, lambda: app._session is not None)
                assert app._session is not None and app._session.session_id == owner.session_id
                assert not stream.requests
    finally:
        await viewer.dispose()
        await server.aclose()
        await owner.dispose()
