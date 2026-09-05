"""Every frontend must discover the same command and prompt-consumption rules."""

import subprocess
import sys

from local_operator.session.frontend_state import _slash_capabilities
from local_operator.slash_commands import SLASH_COMMANDS, slash_command_for


def test_canonical_frontend_capabilities_cover_the_shared_registry() -> None:
    commands = {command.name: command for command in SLASH_COMMANDS}
    capabilities = {cap.command: cap for cap in _slash_capabilities()}
    assert capabilities.keys() == commands.keys()
    for command in commands.values():
        for alias in command.names:
            assert slash_command_for(f"/{alias} argument") is command


def test_command_metadata_import_never_loads_the_textual_app() -> None:
    # Keep this in a fresh interpreter: the suite has already imported the TUI
    # during collection, which would make an in-process module census useless.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import local_operator.slash_commands; "
            "assert 'local_operator.tui.app' not in sys.modules; print('headless registry: ok')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "headless registry: ok"
