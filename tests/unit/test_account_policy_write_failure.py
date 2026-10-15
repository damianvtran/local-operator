"""Real permission rejection must not publish an unsaved effective policy."""

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.providers.account_policy import effective_account_selection_policy


def test_rejected_policy_save_and_atomic_reset_keep_effective_memory_and_disk(tmp_path):
    root = tmp_path / "synthetic-config"
    manager = ConfigManager(root)
    row = settings_io.resolve_key("retry.accountSelectionPolicy")
    assert row is not None, "the canonical routing setting must be registered"
    settings_io.write_setting(manager, row, "headroom")
    # A separate manager contributes an unrelated field; rollback must never
    # write an old snapshot over it, even after our original manager is stale.
    other = ConfigManager(root)
    other.set_config_value(
        "retry",
        {**other.get_config_value("retry"), "maxRetries": 7, "usageAwareAccountPick": False},
    )
    original = manager.config_file.read_bytes()
    root.chmod(0o500)
    try:
        with pytest.raises(OSError):
            settings_io.write_setting(manager, row, "fair")
        assert manager.config_file.read_bytes() == original
        assert settings_io.read_setting(manager, row) == "headroom"
        assert effective_account_selection_policy(manager.get_config().values) == "headroom"
        with pytest.raises(OSError):
            settings_io.reset_setting(manager, row)
        assert manager.config_file.read_bytes() == original
        assert settings_io.read_setting(manager, row) == "headroom"
        assert manager.get_config_value("retry")["usageAwareAccountPick"] is False
        assert ConfigManager(root).get_config_value("retry")["maxRetries"] == 7
    finally:
        root.chmod(0o700)
    settings_io.write_setting(manager, row, "fair")
    assert settings_io.read_setting(manager, row) == "fair"
    assert ConfigManager(root).get_config_value("retry")["maxRetries"] == 7
    settings_io.reset_setting(manager, row)
    assert settings_io.read_setting(manager, row) == "reset-soonest"
    assert "usageAwareAccountPick" not in manager.get_config_value("retry")
    assert manager.get_config_value("retry")["maxRetries"] == 7
