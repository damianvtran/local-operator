import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from local_operator.config import DEFAULT_CONFIG, Config, ConfigManager


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_config_initialization():
    """Test Config class initialization with dictionary."""
    config_dict = {
        "version": "1.0.0",
        "metadata": {
            "created_at": "",
            "last_modified": "",
            "description": "Local Operator configuration file",
        },
        "values": {
            "conversation_length": 5,
            "detail_length": 3,
            "hosting": "test_host",
            "model_name": "test_model",
        },
    }
    config = Config(config_dict)

    assert config.version == "1.0.0"
    assert config.metadata["description"] == "Local Operator configuration file"
    assert config.get_value("conversation_length") == 5
    assert config.get_value("detail_length") == 3
    assert config.get_value("hosting") == "test_host"
    assert config.get_value("model_name") == "test_model"


@patch("local_operator.config.version")
def test_config_initialization_with_default_version(mock_version):
    """Test Config class initialization with default version."""
    mock_version.return_value = "2.0.0"
    config = Config({})
    assert config.version == "2.0.0"


def test_config_manager_initialization(temp_config_dir):
    """Test ConfigManager initialization creates config file if not exists."""
    config_manager = ConfigManager(temp_config_dir)

    assert config_manager.config.version is not None
    assert isinstance(config_manager.config, Config)
    assert config_manager.get_config_value("conversation_length") == DEFAULT_CONFIG.get_value(
        "conversation_length"
    )


@patch("local_operator.config.version")
def test_config_manager_version_warning(mock_version, temp_config_dir, capsys):
    """Test ConfigManager warns about old config versions."""
    mock_version.return_value = "1.0.0"

    # Create config file with old version
    test_config = {
        "version": "2.0.0",
        "metadata": {
            "created_at": "",
            "last_modified": "",
            "description": "Local Operator configuration file",
        },
        "values": {},
    }
    config_file = temp_config_dir / "config.yml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(test_config, f)

    ConfigManager(temp_config_dir)
    captured = capsys.readouterr()
    assert (
        "Warning: Your config file version (2.0.0) is newer than the current version (1.0.0)"
        in captured.out
    )


def test_config_manager_load_existing(temp_config_dir):
    """Test ConfigManager loads existing config file."""
    test_config = {
        "version": "1.0.0",
        "metadata": {
            "created_at": "",
            "last_modified": "",
            "description": "Local Operator configuration file",
        },
        "values": {
            "conversation_length": 20,
            "detail_length": 15,
            "hosting": "custom_host",
            "model_name": "custom_model",
        },
    }

    config_file = temp_config_dir / "config.yml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(test_config, f)

    config_manager = ConfigManager(temp_config_dir)
    assert config_manager.get_config_value("conversation_length") == 20
    assert config_manager.get_config_value("hosting") == "custom_host"


def test_config_manager_load_missing_file(temp_config_dir):
    """Test ConfigManager loads default config when file doesn't exist."""
    config_file = temp_config_dir / "nonexistent.yml"

    config_manager = ConfigManager(config_file)

    # Should create file with default values
    assert config_manager.config.version == DEFAULT_CONFIG.version
    assert config_manager.get_config_value("conversation_length") == DEFAULT_CONFIG.get_value(
        "conversation_length"
    )
    assert config_manager.get_config_value("detail_length") == DEFAULT_CONFIG.get_value(
        "detail_length"
    )
    assert config_manager.get_config_value("hosting") == DEFAULT_CONFIG.get_value("hosting")
    assert config_manager.get_config_value("model_name") == DEFAULT_CONFIG.get_value("model_name")


def test_config_manager_load_empty_file(temp_config_dir):
    """Test ConfigManager loads default config when file is empty."""
    config_file = temp_config_dir / "config.yml"
    config_file.touch()  # Create empty file

    config_manager = ConfigManager(temp_config_dir)

    # Should load default values
    assert config_manager.config.version == DEFAULT_CONFIG.version
    assert config_manager.get_config_value("conversation_length") == DEFAULT_CONFIG.get_value(
        "conversation_length"
    )
    assert config_manager.get_config_value("detail_length") == DEFAULT_CONFIG.get_value(
        "detail_length"
    )
    assert config_manager.get_config_value("hosting") == DEFAULT_CONFIG.get_value("hosting")
    assert config_manager.get_config_value("model_name") == DEFAULT_CONFIG.get_value("model_name")


def test_config_manager_load_partial_values(temp_config_dir):
    """Test ConfigManager loads default values for missing fields."""
    test_config = {
        "version": "1.0.0",
        "metadata": {"created_at": "", "last_modified": "", "description": "Test config"},
        "values": {
            "conversation_length": 50,  # Only specify some values
            "hosting": "custom_host",
            # detail_length and model_name intentionally omitted
        },
    }

    config_file = temp_config_dir / "config.yml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(test_config, f)

    config_manager = ConfigManager(temp_config_dir)

    # Specified values should match test config
    assert config_manager.get_config_value("conversation_length") == 50
    assert config_manager.get_config_value("hosting") == "custom_host"

    # Missing values should use defaults
    assert config_manager.get_config_value("detail_length") == DEFAULT_CONFIG.get_value(
        "detail_length"
    )
    assert config_manager.get_config_value("model_name") == DEFAULT_CONFIG.get_value("model_name")


def test_config_manager_update_config(temp_config_dir):
    """Test updating configuration values."""
    config_manager = ConfigManager(temp_config_dir)

    updates = {"conversation_length": 25, "hosting": "new_host"}
    config_manager.update_config(updates)

    # Verify updates in memory
    assert config_manager.get_config_value("conversation_length") == 25
    assert config_manager.get_config_value("hosting") == "new_host"

    # Verify updates persisted to file
    with open(config_manager.config_file, "r", encoding="utf-8") as f:
        saved_config = yaml.safe_load(f)
    assert saved_config["values"]["conversation_length"] == 25
    assert saved_config["values"]["hosting"] == "new_host"


def test_config_manager_reset_defaults(temp_config_dir):
    """Test resetting configuration to defaults."""
    config_manager = ConfigManager(temp_config_dir)

    # First modify some values
    config_manager.update_config({"conversation_length": 30})

    # Then reset to defaults
    config_manager.reset_to_defaults()

    assert config_manager.config.version == DEFAULT_CONFIG.version
    assert config_manager.get_config_value("conversation_length") == DEFAULT_CONFIG.get_value(
        "conversation_length"
    )
    assert config_manager.get_config_value("hosting") == DEFAULT_CONFIG.get_value("hosting")


def test_config_manager_get_set(temp_config_dir):
    """Test getting and setting individual config values."""
    config_manager = ConfigManager(temp_config_dir)

    # Test get with default
    assert config_manager.get_config_value("nonexistent", "default") == "default"

    # Test set and get
    config_manager.set_config_value("hosting", "test_host")
    assert config_manager.get_config_value("hosting") == "test_host"

    # Verify persistence
    with open(config_manager.config_file, "r", encoding="utf-8") as f:
        saved_config = yaml.safe_load(f)
    assert saved_config["values"]["hosting"] == "test_host"


def test_config_manager_update_from_args(temp_config_dir):
    """Test updating config from command line arguments."""
    config_manager = ConfigManager(temp_config_dir)

    args = Namespace(hosting="cli_host", model="cli_model")
    config_manager.update_config_from_args(args)

    assert config_manager.get_config_value("hosting") == "cli_host"
    assert config_manager.get_config_value("model_name") == "cli_model"
