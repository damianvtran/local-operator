"""The Saver transcript script must AWAIT the slash handler it narrates.

``saver.command.saver_slash`` became a coroutine in review M2 (the answer is
published only after the journal write lands). The script kept calling it bare,
so its first line printed nothing and it died on ``'coroutine' object has no
attribute 'text'`` — a capture tool that silently produced no transcript. Run it
the way it is used (a subprocess with the repo on ``PYTHONPATH``) and pin that it
completes and narrates every command it issues.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_the_transcript_script_runs_every_command_to_a_sentence(tmp_path: Path) -> None:
    env = {
        "HOME": str(tmp_path),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / ".local-operator"),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO),
    }
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "saver_command_transcript.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "coroutine" not in done.stderr
    lines = done.stdout.splitlines()
    commands = [line for line in lines if line.startswith("$ ")]
    assert len(commands) == 5
    # Each command is followed by its receipt — a sentence, never an empty line.
    for index, line in enumerate(lines):
        if line.startswith("$ "):
            assert lines[index + 1].strip(), line
    assert lines[-1].startswith("journal rows:")
