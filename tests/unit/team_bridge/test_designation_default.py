"""Only an absent designation may inherit the durable chief-of-staff id.

Empty/null are deliberate badge-only choices, not a request to wake whatever
session happens to be on screen. Exercise the real config writer on isolated
files, including a second read after changing the durable chief record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.aida import state as aida_state
from local_operator.config import CONFIG_FILE_NAME, ConfigManager
from local_operator.team_bridge.settings import record_local_session_if_unset


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    root = home / ".local-operator"
    root.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.chdir(home)
    return root


def _config(root: Path, bridge: dict[str, Any]) -> Path:
    path = root / CONFIG_FILE_NAME
    path.write_text(
        yaml.safe_dump({"values": {"team_bridge": bridge, "language": "en"}}),
        encoding="utf-8",
    )
    return path


def test_absent_designation_uses_durable_cos_id_not_its_display_name(config_dir):
    path = _config(config_dir, {"enabled": False, "onboarding": "seen"})
    aida_state.write_state(config_dir, {"session_id": "sess-chief-123", "name": "Renamed CoS"})

    assert record_local_session_if_unset(config_dir) is True
    # Reopen the manager: an in-memory update is not a durable designation.
    manager = ConfigManager(config_dir)
    assert manager.get_nested_value(("team_bridge", "session")) == "sess-chief-123"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))["values"]
    assert values["team_bridge"] == {
        "enabled": False,
        "onboarding": "seen",
        "session": "sess-chief-123",
    }
    assert values["language"] == "en"

    before = path.read_bytes()
    aida_state.write_state(config_dir, {"session_id": "sess-other-chief", "name": "Another name"})
    assert record_local_session_if_unset(config_dir) is False
    assert path.read_bytes() == before


def test_no_cos_does_not_create_a_config_file(config_dir):
    assert record_local_session_if_unset(config_dir) is False
    assert not (config_dir / CONFIG_FILE_NAME).exists()


@pytest.mark.parametrize("state", [{}, {"session_id": None}, {"session_id": ""}])
def test_no_durable_cos_id_does_not_touch_existing_config(config_dir, state):
    path = _config(config_dir, {"enabled": True})
    aida_state.write_state(config_dir, state)
    before = path.read_bytes()
    assert record_local_session_if_unset(config_dir) is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("chosen", ["sess-task-selected", "", None])
def test_explicit_choice_including_empty_or_null_is_preserved(config_dir, chosen):
    path = _config(config_dir, {"enabled": True, "session": chosen})
    aida_state.write_state(config_dir, {"session_id": "sess-chief-123"})
    before = path.read_bytes()
    assert record_local_session_if_unset(config_dir) is False
    assert path.read_bytes() == before
    assert ConfigManager(config_dir).get_nested_value(("team_bridge", "session")) == chosen
