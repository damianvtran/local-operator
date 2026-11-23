"""The shared app-state restore must never hand a closed auth store back.

Regression for merge-gate run 44354652248 and main run 44358942296 (shard 5):
``test_server_models`` cells failed with ``500`` and ``sqlite3.ProgrammingError:
Cannot operate on a closed database`` whenever they ran after a
``test_serve_retire`` test whose ``restore_app_state`` fixture wrapped a real
lifespan. The lifespan teardown closed the provider auth host an earlier test
had left on the singleton ``app``, and the fixture's snapshot put the closed
host back. This pins the ordering deterministically in one test instead of
relying on xdist placing two files on the same worker.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_operator.providers.auth_store import AuthStore
from local_operator.server.app import app
from local_operator.server.utils.desktop_auth import DesktopAuth
from tests.unit.server.app_state import restored_app_state


def test_a_lifespan_inside_the_restore_leaves_the_callers_auth_store_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path / "config"))
    root = tmp_path / "planted"
    root.mkdir()
    # A live host on the shared app, as a lifespan-less test that resolved a
    # provider credential leaves it (``routes/auth.get_desktop_auth``).
    planted = DesktopAuth(AuthStore(root / "auth.db", config_dir=root), root)
    had = "desktop_auth" in app.state
    before = getattr(app.state, "desktop_auth", None)
    app.state.desktop_auth = planted
    try:
        with restored_app_state(app):
            # Running the real lifespan is the step that closed the planted
            # host before the fix.
            with TestClient(app, headers={"Host": "127.0.0.1:1111"}):
                pass

        assert app.state.desktop_auth is planted
        # The exact call that raised in the failing shard (models.py →
        # get_api_key → list_credentials): it must reach an open database.
        assert planted.store.list_credentials("deepseek") == []
    finally:
        planted.store.close()
        if had:
            app.state.desktop_auth = before
        else:
            del app.state["desktop_auth"]
