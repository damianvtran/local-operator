"""A verified Radient sign-in wakes the hub update runner.

WHY: checks that ran while signed out persist ``no-credential`` on every linked
item, cleared only by the next successful check -- up to ~16.5 min away on the
schedule. The sign-in hook (``DesktopAuth.rehome``, wired by ``get_desktop_auth``
and invoked only after a verified login) is where the runner learns to check now.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.config import ConfigManager
from local_operator.server.routes import auth as auth_routes


class _Runner:
    def __init__(self) -> None:
        self.wakes = 0

    def wake(self) -> None:
        self.wakes += 1


@pytest.fixture
def host_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / ".local-operator"
    root.mkdir()
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: root)
    runner = _Runner()
    app = SimpleNamespace(state=SimpleNamespace(desktop_auth=None, hub_sync=runner))
    return app, runner, ConfigManager(root)


@pytest.mark.asyncio
async def test_a_verified_radient_sign_in_wakes_the_hub_runner(host_app) -> None:
    app, runner, manager = host_app
    host = await auth_routes.get_desktop_auth(cast(Any, SimpleNamespace(app=app)), manager)
    try:
        assert host.rehome is not None
        await host.rehome("radient")
        assert runner.wakes == 1
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_another_providers_sign_in_does_not_wake_it(host_app) -> None:
    app, runner, manager = host_app
    host = await auth_routes.get_desktop_auth(cast(Any, SimpleNamespace(app=app)), manager)
    try:
        assert host.rehome is not None
        await host.rehome("openai")
        assert runner.wakes == 0
    finally:
        await host.close()


def test_a_server_without_a_runner_is_left_alone() -> None:
    app = SimpleNamespace(state=SimpleNamespace(hub_sync=None))
    auth_routes._wake_hub_sync(app, "radient")  # no runner: nothing to wake, no raise
