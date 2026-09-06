"""The CLI viewer and runtime metadata must not load the opposite host stack."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_canonical_slash_metadata_does_not_import_textual(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    env.update(HOME=str(tmp_path), LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / "config"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from local_operator.session.frontend_state import _slash_capabilities; "
            "import sys; capabilities = _slash_capabilities(); "
            "assert any(c.command == 'model' for c in capabilities); "
            "assert 'local_operator.tui.app' not in sys.modules; "
            "assert not any(m == 'textual' or m.startswith('textual.') for m in sys.modules); "
            "print('canonical slash capabilities: headless')",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "headless" in result.stdout


def test_slash_registry_reexport_keeps_identity() -> None:
    from local_operator.slash_commands import PERSIST_HINT, SLASH_COMMANDS
    from local_operator.tui import app

    assert app.SLASH_COMMANDS is SLASH_COMMANDS
    assert app.PERSIST_HINT == PERSIST_HINT
    for command in SLASH_COMMANDS:
        for name in command.names:
            assert app.slash_command_for(f"/{name}") is command


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [True, False])
async def test_factory_import_policy_is_explicit(warm: bool, monkeypatch) -> None:
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession

    calls = []
    session = FakeSession()

    async def warm_imports():
        calls.append("warm")

    async def factory():
        calls.append("factory")
        return session

    monkeypatch.setattr(OperatorApp, "_warm_session_imports", staticmethod(warm_imports))
    app = OperatorApp(factory, warm_session_imports=warm)
    assert await app._construct_session() is session
    assert calls == (["warm", "factory"] if warm else ["factory"])
