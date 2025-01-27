import tempfile
from pathlib import Path

import pytest

from local_operator.credentials import CredentialManager


@pytest.fixture
def temp_config():
    """Fixture to create a temporary config file for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.env"
        yield config_path


def test_credential_manager_initialization(temp_config):
    """Test that CredentialManager initializes correctly with a config file."""
    manager = CredentialManager(config_dir=temp_config.parent)
    manager._ensure_config_exists()

    assert manager.config_file.exists()
    assert manager.config_file.stat().st_mode & 0o777 == 0o600
    assert manager.config_dir == temp_config.parent


def test_get_api_key(temp_config):
    """Test retrieving an API key from the config file."""
    with open(temp_config, "w") as f:
        f.write("DEEPSEEK_API_KEY=test_key\n")

    print(f"Temp config: {temp_config}")

    manager = CredentialManager(config_dir=temp_config.parent)
    manager._ensure_config_exists()

    api_key = manager.get_api_key("DEEPSEEK_API_KEY")
    assert api_key == "test_key"


def test_prompt_for_api_key(temp_config, monkeypatch):
    """Test prompting for and saving a new API key."""
    manager = CredentialManager(config_dir=temp_config.parent)
    manager._ensure_config_exists()

    # Mock user input and print statements
    monkeypatch.setattr("builtins.input", lambda _: "new_test_key")
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    api_key = manager.prompt_for_api_key("NEW_API_KEY")
    assert api_key == "new_test_key"

    # Verify the key was saved to the config file
    with open(temp_config, "r") as f:
        content = f.read()
    assert "NEW_API_KEY=new_test_key" in content
    assert temp_config.stat().st_mode & 0o777 == 0o600


def test_missing_api_key_raises_error(temp_config, monkeypatch):
    """Test that missing API key raises ValueError."""
    manager = CredentialManager(config_dir=temp_config.parent)
    manager._ensure_config_exists()

    # Mock empty user input
    monkeypatch.setattr("builtins.input", lambda _: "")

    with pytest.raises(ValueError) as exc_info:
        manager.prompt_for_api_key("NON_EXISTENT_KEY")
    assert "is required to use this application" in str(exc_info.value)


def test_config_file_permissions(temp_config):
    """Test that config file has correct permissions."""
    manager = CredentialManager(config_dir=temp_config.parent)
    manager._ensure_config_exists()

    assert manager.config_file.stat().st_mode & 0o777 == 0o600
