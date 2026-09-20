"""Where a decoded chat attachment lands, and when the directory is made (audit D17).

``attachment_utils`` had its own copy of the config-dir rule
(``Path.home() / ".local-operator"``) and created the directory as a side effect
of module import. Two consequences, both reproduced below: an isolated run that
relocated the config dir still created a directory in the operator's REAL home,
and merely importing the chat routes — which ``server/routes/chat.py`` does —
performed filesystem work in a home directory the process may not even own.

The import-time half can only be shown in a fresh interpreter: pytest has this
module imported long before any test body runs.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.paths import CONFIG_DIR_ENV
from local_operator.server.utils import attachment_utils

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_the_uploads_directory_follows_the_config_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The override relocates it, which is the whole point of the config dir rule."""
    config = tmp_path / "config"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config))
    assert attachment_utils.uploads_dir() == config / "uploads"

    encoded = base64.b64encode(b"\x89PNG not really").decode()
    saved = Path(attachment_utils.save_base64_attachment(f"data:image/png;base64,{encoded}"))
    assert saved.parent == config / "uploads", "the attachment escaped the config dir"
    assert saved.read_bytes() == b"\x89PNG not really"


def test_importing_the_module_creates_nothing(tmp_path: Path) -> None:
    """No directory in the real home as a side effect of an import.

    ``HOME`` is redirected here, so on the unfixed code the created directory is
    visible inside the sandbox — which is exactly how this defect used to leave
    artifacts in the operator's real home under an isolated run.
    """
    home = tmp_path / "home"
    config = tmp_path / "config"
    home.mkdir()
    config.mkdir()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    environment.update(
        HOME=str(home),
        LOCAL_OPERATOR_CONFIG_DIR=str(config),
        PYTHONPATH=str(REPO_ROOT),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from local_operator.server.utils import attachment_utils\n" "print('imported')",
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout
    assert not (home / ".local-operator").exists(), "the import created a directory in HOME"
    assert not (config / "uploads").exists(), "the import created the uploads directory"
    assert list(home.iterdir()) == [], f"the import left artifacts in HOME: {list(home.iterdir())}"
