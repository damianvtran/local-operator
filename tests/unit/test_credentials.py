import os
import tempfile
from pathlib import Path

import pytest

from local_operator.credentials import CREDENTIALS_FILE_NAME, CredentialManager


@pytest.fixture
def temp_config():
    initial_env = os.environ.copy()
    """Fixture to create a temporary config file for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / CREDENTIALS_FILE_NAME
        yield config_path
        # Clear any used environment variables after each test
        os.environ.clear()
        os.environ.update(initial_env)


# Skip tests that check file permissions when not on a POSIX system.
(
    pytest.skip("Skipping file permission tests on non-POSIX systems", allow_module_level=True)
    if os.name != "posix"
    else None
)


@pytest.mark.skipif(os.name != "posix", reason="File permission tests only run on Unix")
def test_credential_manager_initialization(temp_config):
    """Test that CredentialManager initializes correctly with a config file."""
    manager = CredentialManager(config_dir=temp_config.parent)

    assert manager.config_file.exists()
    # The file permissions check should only run on POSIX systems.
    assert manager.config_file.stat().st_mode & 0o777 == 0o600
    assert manager.config_dir == temp_config.parent


def test_get_credential(temp_config):
    """Test retrieving a credential from the config file."""
    with open(temp_config, "w") as f:
        f.write("DEEPSEEK_API_KEY=test_key\n")

    manager = CredentialManager(config_dir=temp_config.parent)

    credential = manager.get_credential("DEEPSEEK_API_KEY")
    assert credential.get_secret_value() == "test_key"


@pytest.mark.skipif(os.name != "posix", reason="File permission tests only run on Unix")
def test_set_credential_existing(temp_config):
    """Test setting a credential in the config file."""
    with open(temp_config, "w") as f:
        f.write("DEEPSEEK_API_KEY=test_key\n")

    manager = CredentialManager(config_dir=temp_config.parent)
    manager._ensure_config_exists()

    manager.set_credential("DEEPSEEK_API_KEY", "new_test_key")
    credential = manager.get_credential("DEEPSEEK_API_KEY")
    assert credential.get_secret_value() == "new_test_key"


def test_set_credential_new(temp_config):
    """Test setting a credential in the config file."""
    manager = CredentialManager(config_dir=temp_config.parent)

    manager.set_credential("NEW_API_KEY", "new_test_key")
    credential = manager.get_credential("NEW_API_KEY")
    assert credential.get_secret_value() == "new_test_key"


@pytest.mark.skipif(os.name != "posix", reason="File permission tests only run on Unix")
def test_prompt_for_credential(temp_config, monkeypatch):
    """Test prompting for and saving a new credential."""
    manager = CredentialManager(config_dir=temp_config.parent)

    # Mock getpass and print statements
    monkeypatch.setattr("getpass.getpass", lambda _: "new_test_key")
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    credential = manager.prompt_for_credential("NEW_API_KEY")
    assert credential.get_secret_value() == "new_test_key"

    # Verify the key was saved to the config file
    with open(temp_config, "r") as f:
        content = f.read()
    assert "NEW_API_KEY=new_test_key" in content
    assert temp_config.stat().st_mode & 0o777 == 0o600


def test_missing_credential_raises_error(temp_config, monkeypatch):
    """Test that missing credential raises ValueError."""
    manager = CredentialManager(config_dir=temp_config.parent)

    # Mock empty getpass input
    monkeypatch.setattr("getpass.getpass", lambda _: "")
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    with pytest.raises(ValueError) as exc_info:
        manager.prompt_for_credential("NON_EXISTENT_KEY")
    assert "is required for this step" in str(exc_info.value)


@pytest.mark.skipif(os.name != "posix", reason="File permission tests only run on Unix")
def test_config_file_permissions(temp_config):
    """Test that config file has correct permissions."""
    manager = CredentialManager(config_dir=temp_config.parent)

    assert manager.config_file.stat().st_mode & 0o777 == 0o600


def test_get_credential_from_env(temp_config, monkeypatch):
    """Test retrieving a credential from environment variables."""
    manager = CredentialManager(config_dir=temp_config.parent)

    # Set environment variable but not in config file
    monkeypatch.setenv("ENV_ONLY_API_KEY", "env_test_key")

    # Verify the credential is retrieved from environment
    credential = manager.get_credential("ENV_ONLY_API_KEY")
    assert credential.get_secret_value() == "env_test_key"

    # Verify it was added to the credentials dict but not written to file
    assert "ENV_ONLY_API_KEY" in manager.credentials

    # Verify it wasn't written to the config file
    with open(temp_config, "r") as f:
        content = f.read()
    assert "ENV_ONLY_API_KEY=env_test_key" not in content


# Windows-specific tests
@pytest.mark.skipif(os.name != "nt", reason="Windows-specific tests")
def test_windows_credential_manager_initialization(temp_config):
    """Test that CredentialManager initializes correctly on Windows."""
    manager = CredentialManager(config_dir=temp_config.parent)

    assert manager.config_file.exists()
    assert manager.config_dir == temp_config.parent


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific tests")
def test_windows_set_credential_existing(temp_config):
    """Test setting a credential in the config file on Windows."""
    with open(temp_config, "w") as f:
        f.write("DEEPSEEK_API_KEY=test_key\n")

    manager = CredentialManager(config_dir=temp_config.parent)
    manager._ensure_config_exists()

    manager.set_credential("DEEPSEEK_API_KEY", "new_test_key")
    credential = manager.get_credential("DEEPSEEK_API_KEY")
    assert credential.get_secret_value() == "new_test_key"


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific tests")
def test_windows_prompt_for_credential(temp_config, monkeypatch):
    """Test prompting for and saving a new credential on Windows."""
    manager = CredentialManager(config_dir=temp_config.parent)

    # Mock getpass and print statements
    monkeypatch.setattr("getpass.getpass", lambda _: "new_test_key")
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    credential = manager.prompt_for_credential("NEW_API_KEY")
    assert credential.get_secret_value() == "new_test_key"

    # Verify the key was saved to the config file
    with open(temp_config, "r") as f:
        content = f.read()
    assert "NEW_API_KEY=new_test_key" in content


# --- First-run hardening (items 4, 13, 14, 16) ------------------------------


def test_set_credential_rejects_control_chars(temp_config):
    """A newline in a value would split the flat key=value store into a bogus
    second entry, so it is rejected at the boundary (item 16)."""
    manager = CredentialManager(config_dir=temp_config.parent)
    with pytest.raises(ValueError, match="control characters"):
        manager.set_credential("BAD_KEY", "line1\nline2")
    with pytest.raises(ValueError, match="control characters"):
        manager.set_credential("BAD_KEY", "has\x00nul")


def test_write_to_file_is_atomic_and_leaves_no_temp(temp_config):
    """Writes go temp-then-replace, so a crash cannot lose every key, and no
    stray temp file is left behind (item 16)."""
    manager = CredentialManager(config_dir=temp_config.parent)
    manager.set_credential("K1", "v1")
    manager.set_credential("K2", "v2")
    content = temp_config.read_text()
    assert "K1=v1" in content and "K2=v2" in content
    # No leftover .credentials-*.tmp files in the config dir.
    leftovers = list(temp_config.parent.glob(".credentials-*.tmp"))
    assert leftovers == [], leftovers


def test_loose_credentials_file_is_retightened(temp_config):
    """A pre-existing world-readable credentials file is tightened to 0600 on
    load, but only the file — never chmod the directory (item 17)."""
    if os.name != "posix":
        pytest.skip("permission test is Unix-only")
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    temp_config.touch()
    temp_config.chmod(0o644)
    CredentialManager(config_dir=temp_config.parent)
    assert temp_config.stat().st_mode & 0o077 == 0


def test_prompt_reads_from_piped_stdin_without_getpass(temp_config, monkeypatch):
    """Non-tty stdin: the value is read as one line (the automation contract),
    NOT through getpass's echo fallback (item 13)."""
    import io

    manager = CredentialManager(config_dir=temp_config.parent)
    monkeypatch.setattr("sys.stdin", io.StringIO("piped-key-value\n"))
    # getpass must NOT be called on the non-tty path.
    monkeypatch.setattr(
        "getpass.getpass",
        lambda *_a, **_k: pytest.fail("getpass used on non-tty stdin"),
    )
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    cred = manager.prompt_for_credential("PIPED_KEY")
    assert cred.get_secret_value() == "piped-key-value"


def test_prompt_empty_piped_stdin_raises_eof(temp_config, monkeypatch):
    """A closed/empty pipe raises EOFError, which the command handler turns into
    one plain line + exit 1 (item 4/13)."""
    import io

    manager = CredentialManager(config_dir=temp_config.parent)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    with pytest.raises(EOFError):
        manager.prompt_for_credential("PIPED_KEY")


def test_prompt_ascii_banner_when_stdout_cannot_encode(temp_config, monkeypatch, capsys):
    """PYTHONIOENCODING=ascii (or a legacy code page) cannot encode the box
    drawing, so the banner falls back to ASCII rather than crashing (item 14)."""
    manager = CredentialManager(config_dir=temp_config.parent)
    # Force the encoding check to report an ASCII-only stdout. Patched in the
    # credentials module namespace (it imports the name directly), not on
    # cli_style, or the already-bound reference would win.
    monkeypatch.setattr("local_operator.credentials.can_encode", lambda *a, **k: False)
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_k: "k")
    monkeypatch.setattr("sys.stdin", _TtyStub())
    manager.prompt_for_credential("ASCII_KEY")
    out = capsys.readouterr().out
    assert "─" not in out and "╭" not in out
    assert "+" in out  # ASCII border corner
    # The success glyph also falls back — a stdout that cannot encode the box
    # cannot encode ✓ either, and crashing after the key is saved is the worst
    # place to fail.
    assert "✓" not in out and "[ok]" in out


class _TtyStub:
    """A stdin stand-in that claims to be a tty (so getpass is used)."""

    def isatty(self) -> bool:
        return True
