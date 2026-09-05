"""A fork whose transcript carries the PARENT's checkpoint still binds and switches.

The fork e2e in ``test_fork_e2e.py`` seeds a parent with messages only, so its
fork inherits no ``frontend_state_checkpoint_v1`` row and the runtime restores
a fresh state stamped with the right id. Every real fork of a session that has
run a turn inherits the parent's checkpoint, and until #573 that checkpoint's
``session_id`` rode through the restore untouched: the fork's runtime served
the parent's id in ``frontend_sync``, the viewer refused it, and the fork was
un-attachable — its own switched-to viewer half-bound (RPCs landed, no state
ever painted), every second terminal refused outright. This stage drives that
exact shape against REAL runtime subprocesses, the way ``lop`` does:

* the TUI's viewer factory (cold ``RemoteSession`` engaged into a
  ``local_operator.session.runtime.process`` child by ``engage_runtime``);
* a parent transcript whose newest checkpoint names the parent and carries a
  measured context reading;
* ``/fork`` in switch mode, then ``/model`` on the fork;
* a second follower ``RemoteSession.connect``-ing to the live fork.

Only the provider is a mock (the ``test`` hosting). Everything else — the fork
copy, the lease, the spawn, the socket, the sync, the band — is production.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.tui.app import OperatorApp
from tests.e2e.harness import wait_for_adoption
from tests.e2e.test_fork_e2e import _never_take_over, _pump
from tests.e2e.watchdog import bounded

PARENT_ID = "ckptparent01"


def _seed_parent_with_checkpoint(config: Path) -> None:
    """A parent that has run a turn: messages plus the checkpoint it wrote."""
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendModelSpec,
        FrontendSessionState,
        JobState,
    )

    directory = config / "sessions" / PARENT_ID
    directory.mkdir(parents=True)
    model = FrontendModelSpec(provider="test", model_id="mock", context_window=128_000)
    checkpoint = FrontendSessionState(
        session_id=PARENT_ID,
        epoch="parent-owner",
        conversation_title="Checkpointed parent",
        # The parent's child: a fork must not list it (#573, and the reason
        # the roster sidecar is on ``fork.EXCLUDED_SIDECARS``).
        jobs=[JobState(id="parent-child", type="task", label="auditor", status="succeeded")],
        selected_model=model,
        effective_model=model,
        context_tokens=42_000,
        context_is_estimate=False,
        context_window=128_000,
    )
    rows = [
        {
            "id": "u1",
            "ts": 1.0,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "user",
                "content": [{"type": "text", "text": "original question"}],
            },
        },
        {
            "id": "a1",
            "ts": 2.0,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "saved answer"}],
                "stop_reason": "stop",
            },
        },
        {
            "id": "c1",
            "ts": 3.0,
            "type": "custom",
            "payload": {
                "custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE,
                "details": {
                    "checkpoint_id": "cp-parent",
                    "state": checkpoint.model_dump(mode="json"),
                },
            },
        },
    ]
    (directory / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _kill_runtime_children(config: Path) -> None:
    """Reap the runtime subprocesses this test's engages spawned.

    They are started in their own session group and would otherwise live
    until their idle drain; a killed child never writes anything the
    assertions read, and the tmp config dir is removed with the test.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-f", "local_operator.session.runtime.process"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
    except OSError:
        return
    for pid in out:
        try:
            environ = subprocess.run(
                ["ps", "-o", "command=", "-p", pid], capture_output=True, text=True, check=False
            )
            # ``ps`` cannot show the child's environment portably; match on the
            # record it published instead.
            del environ
            for record_path in (config / "run" / "mobile").glob("*.json"):
                if json.loads(record_path.read_text()).get("pid") == int(pid):
                    os.kill(int(pid), 9)
        except (OSError, ValueError):
            continue


@pytest.mark.asyncio
async def test_a_fork_with_an_inherited_checkpoint_binds_switches_and_admits_followers(
    headless_tui_env: Path, workspace: Path, monkeypatch
) -> None:
    from local_operator.mobile.attach_client import find_owner_record
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController
    from local_operator.spawn import registry as spawn_registry

    config = headless_tui_env
    monkeypatch.setattr(
        spawn_registry,
        "active_backend",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("switch must not open a window")),
    )
    # The runtime children must resolve the SAME source tree as this test.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    monkeypatch.setenv("HOME", str(config.parent))
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n  fork:\n    mode: switch\n",
        encoding="utf-8",
    )
    _seed_parent_with_checkpoint(config)

    async def take_over() -> Any:
        raise RuntimeError("a viewer never takes over a session")

    async def viewer_factory(resume_id: str | None) -> RemoteSession:
        # ``cli.viewer_factory`` in miniature: attach when a live record
        # exists, else open cold and let the mount engage start a runtime.
        session_id = resume_id or "newsession01"
        record = None
        if resume_id:
            record, _ = await asyncio.to_thread(find_owner_record, config, session_id)
        if record is not None:
            return await RemoteSession.connect(
                record, session_id, config_dir=config, takeover_factory=take_over
            )
        return await RemoteSession.cold(
            session_id, config_dir=config, cwd=str(workspace), takeover_factory=take_over
        )

    controller = ProviderController(AuthStore(config / "auth.db"))
    OperatorApp._check_for_update = lambda self: None  # type: ignore[method-assign]
    app = OperatorApp(
        lambda: viewer_factory(PARENT_ID),
        provider_controller=controller,
        resume_factory=viewer_factory,
    )
    follower: RemoteSession | None = None
    try:
        with bounded(150, "fork with inherited checkpoint binds, switches, admits a follower"):
            async with app.run_test(size=(100, 30)) as pilot:
                await wait_for_adoption(app, pilot)
                # The parent's own engage: a real runtime child comes up and
                # the band paints the checkpoint's context reading.
                await _pump(pilot, lambda: not getattr(app._session, "is_cold", True))
                await _pump(pilot, lambda: app._status is not None and app._status.context_tokens)
                assert app._status is not None
                assert app._status.context_tokens == 42_000

                app._run_slash_command("/fork")
                await _pump(
                    pilot,
                    lambda: app._session is not None and app._session.session_id != PARENT_ID,
                )
                fork = app._session
                assert isinstance(fork, RemoteSession)
                fork_id = fork.session_id
                status = app._status
                # THE regression: the fork's runtime must bind. Before #573
                # this engage failed with "frontend state belongs to another
                # session" and the viewer sat half-bound forever.
                await _pump(pilot, lambda: not fork.is_cold)
                await _pump(pilot, lambda: status.context_tokens == 42_000)
                state = fork.frontend_state
                assert state.session_id == fork_id
                assert state.jobs == (), "a fork must not list its parent's children"
                assert status._context_window == 128_000
                assert status._model_label == "test/mock"

                # /model on the fork lands on the owner and repaints the band
                # from the owner's sync.
                app._run_slash_command("/model test/other")
                await _pump(pilot, lambda: fork.model_label == "test/other")
                await _pump(pilot, lambda: status._model_label == "test/other")
                assert status.context_tokens == 42_000
                journal = (config / "sessions" / fork_id / "transcript.jsonl").read_text()
                # The owner journals a genuine switch (``Session.set_model``):
                # the durable proof it landed on the runtime, not just the band.
                assert '"new_label":"test/other"' in journal.replace(
                    " ", ""
                ), "the switch never reached the owner"

                # A SECOND terminal attaches as a follower (#573's report).
                record, _ = await asyncio.to_thread(find_owner_record, config, fork_id)
                assert record is not None
                follower = await RemoteSession.connect(
                    record, fork_id, config_dir=config, takeover_factory=_never_take_over
                )
                assert follower.frontend_state.session_id == fork_id
                assert follower.model_label == "test/other"
                assert follower.frontend_state.context_tokens == 42_000
                app.exit()
    finally:
        if follower is not None:
            await follower.dispose()
        _kill_runtime_children(config)


@pytest.mark.asyncio
async def test_a_fork_of_a_fork_serves_its_own_id(headless_tui_env: Path, workspace: Path) -> None:
    """#573 observed the GRANDPARENT's id two hops down. Chain two forks on
    disk and boot a real runtime for the second: its sync names itself."""
    from local_operator.fork import fork_session
    from local_operator.mobile.attach_client import find_owner_record
    from local_operator.session.runtime import registry

    config = headless_tui_env
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )
    _seed_parent_with_checkpoint(config)
    first = fork_session(config, PARENT_ID)
    second = fork_session(config, first)
    env = {
        **os.environ,
        "HOME": str(config.parent),
        "LOCAL_OPERATOR_CONFIG_DIR": str(config),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "LOP_MOBILE_CHILD_CWD": str(workspace),
        "LOP_MOBILE_CHILD_RESUME": second,
    }
    child = subprocess.Popen(
        [sys.executable, "-m", "local_operator.session.runtime.process"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    viewer: RemoteSession | None = None
    try:
        with bounded(90, "fork-of-a-fork runtime serves its own id"):
            record = None
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and record is None:
                for candidate, state in registry.scan(config):
                    if candidate.session_id == second and state == "live":
                        record = candidate
                await asyncio.sleep(0.1)
            assert record is not None, "the fork-of-a-fork runtime never published"
            # ``find_owner_record`` needs the liveness marker the child wrote.
            found, _ = await asyncio.to_thread(find_owner_record, config, second)
            assert found is not None
            viewer = await RemoteSession.connect(
                found, second, config_dir=config, takeover_factory=_never_take_over
            )
            assert viewer.frontend_state.session_id == second
            assert viewer.frontend_state.jobs == ()
    finally:
        if viewer is not None:
            await viewer.dispose()
        child.kill()
        child.wait(timeout=10)
