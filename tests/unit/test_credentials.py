import errno
import os
import subprocess
import sys
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

    # Pin the interactive branch: prompt_for_credential now reads a line from
    # stdin when stdin is NOT a tty (the scripted-automation contract). Under
    # pytest in CI stdin is not a tty, so without forcing isatty True this test
    # would silently exercise the pipe path and ignore the getpass mock below.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
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

    # Pin the interactive branch (see test_prompt_for_credential): this case
    # asserts the empty-INTERACTIVE-input ValueError. The non-tty pipe path
    # raises EOFError instead (covered separately by
    # test_prompt_empty_piped_stdin_raises_eof), so force a tty here.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
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


@pytest.mark.skipif(os.name != "posix", reason="File modes are only meaningful on POSIX")
def test_read_key_names_neither_creates_nor_tightens_the_store(tmp_path: Path) -> None:
    """The read-only construction, including the mode it must NOT change.

    ``__init__`` runs ``_ensure_config_exists()``, which creates the store and
    re-tightens a loose one to 0600. ``read_key_names`` exists to do neither: it
    is the read ``/info`` performs on a host it is describing, so a store it
    chmods is a store it has already failed to leave alone. This half of the
    property had no test at any level (review round 1, R1-F2).
    """
    store = tmp_path / CREDENTIALS_FILE_NAME
    store.write_text("DEEPSEEK_API_KEY=test_key\nEMPTY_KEY=\n")
    store.chmod(0o644)

    # The empty-valued key is dropped by ``list_credential_keys``' default, so
    # this also pins that the read goes through the class's own parser.
    assert CredentialManager.read_key_names(tmp_path) == ["DEEPSEEK_API_KEY"]

    assert store.stat().st_mode & 0o777 == 0o644, "the read must not re-tighten the store"
    assert sorted(path.name for path in tmp_path.iterdir()) == [CREDENTIALS_FILE_NAME]


def test_read_key_names_returns_empty_only_when_the_store_is_absent(tmp_path: Path) -> None:
    """Q1/Q2: ``[]`` is ``ENOENT``'s answer and nobody else's.

    The previous spelling asked ``Path.is_file()``, which answers ``False`` for
    ``ENOENT``, ``ENOTDIR``, ``EBADF`` and ``ELOOP`` and — from CPython 3.14 —
    swallows every ``OSError``, so a config root the process could not traverse
    and a store symlinked to itself both came back as "no credentials
    recorded". Each case is pinned to the raise here; the collector's
    ``degraded`` row is pinned in tests/unit/info/test_collect.py.
    """
    import os

    # ENOENT: genuinely nothing there. This is the ONE empty answer.
    assert CredentialManager.read_key_names(tmp_path) == []

    if os.name != "posix":
        pytest.skip("mode bits and symlink loops are POSIX-only")

    # ENOTDIR: the config root is a regular file, so the store cannot exist.
    root_is_a_file = tmp_path / "file-root"
    root_is_a_file.write_text("")
    with pytest.raises(NotADirectoryError):
        CredentialManager.read_key_names(root_is_a_file)

    # ELOOP: the store is a symlink to itself.
    looped = tmp_path / "loop-root"
    looped.mkdir()
    (looped / CREDENTIALS_FILE_NAME).symlink_to(CREDENTIALS_FILE_NAME)
    with pytest.raises(OSError) as loop_error:
        CredentialManager.read_key_names(looped)
    assert loop_error.value.errno == errno.ELOOP

    # EACCES: a directory on the way to the store cannot be searched.
    locked = tmp_path / "locked-root"
    locked.mkdir()
    (locked / CREDENTIALS_FILE_NAME).write_text("DEEPSEEK_API_KEY=test_key\n")
    locked.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            CredentialManager.read_key_names(locked)
    finally:
        # Restore traversal so the tmp_path tree can still be cleaned up.
        locked.chmod(0o700)


_READ_NAMES_PROBE = """
import errno
import os
import sys
from pathlib import Path
from unittest.mock import patch
from local_operator.credentials import CredentialManager

root, expected, fault = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
def read():
    try:
        result = CredentialManager.read_key_names(root)
    except OSError as exc:
        assert expected != errno.ENOENT and exc.errno == expected, repr(exc)
    else:
        assert expected == errno.ENOENT and result == [], (expected, result)

if fault:
    with patch('local_operator.credentials.os.' + fault,
               side_effect=OSError(expected, 'injected diagnostic failure')):
        read()
else:
    read()
print('PASS')
"""


def _probe_read_names(root: Path, expected: int, fault: str = "") -> None:
    # A timeout in this pytest process cannot reliably interrupt a blocked
    # syscall in a collector worker. A subprocess deadline kills and reaps just
    # our probe, so restoring the FIFO regression cannot hang the whole suite.
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CMUX_", "LOP_"))}
    env.update(HOME=str(root.parent), LOCAL_OPERATOR_CONFIG_DIR=str(root))
    result = subprocess.run(
        [sys.executable, "-c", _READ_NAMES_PROBE, str(root), str(expected), fault],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"


@pytest.mark.parametrize("kind", ["fifo", "device", "directory"])
def test_read_key_names_rejects_special_files_with_deadline(tmp_path: Path, kind: str) -> None:
    store = tmp_path / CREDENTIALS_FILE_NAME
    if kind == "fifo":
        os.mkfifo(store)
    elif kind == "device":
        # The same character-device rejection as /dev/zero, but a regressed
        # reader reaches EOF instead of allocating unbounded host memory.
        store.symlink_to("/dev/null")
    else:
        store.mkdir()
    _probe_read_names(tmp_path, errno.EISDIR if kind == "directory" else errno.EINVAL)


@pytest.mark.parametrize("fault", ["open", "fstat"])
@pytest.mark.parametrize(
    "code", [errno.ENOENT, errno.EACCES, errno.ELOOP, errno.ENOTDIR, errno.EIO]
)
def test_read_key_names_preserves_diagnostic_errnos_with_deadline(
    tmp_path: Path, fault: str, code: int
) -> None:
    (tmp_path / CREDENTIALS_FILE_NAME).write_text("SYNTHETIC_KEY=fixture\n")
    _probe_read_names(tmp_path, code, fault)


def test_regular_store_symlink_preserves_parser_and_ordinary_load(tmp_path: Path) -> None:
    target = tmp_path / "fixture-store"
    target.write_text("# comment\nSYNTHETIC_KEY=fixture=value\nEMPTY_KEY=\n")
    (tmp_path / CREDENTIALS_FILE_NAME).symlink_to(target.name)
    assert CredentialManager.read_key_names(tmp_path) == ["SYNTHETIC_KEY"]
    assert CredentialManager.read_key_names(tmp_path, non_empty=False) == [
        "SYNTHETIC_KEY",
        "EMPTY_KEY",
    ]
    manager = CredentialManager(tmp_path)
    assert manager.get_credential("SYNTHETIC_KEY").get_secret_value() == "fixture=value"
    assert manager.list_credential_keys() == ["SYNTHETIC_KEY"]


def test_regular_store_closes_descriptor_on_validation_failure(tmp_path: Path, monkeypatch) -> None:
    from local_operator.credentials import _open_regular_store

    store = tmp_path / CREDENTIALS_FILE_NAME
    store.write_text("SYNTHETIC_KEY=fixture\n")
    real_open = os.open
    opened = []

    def track_open(path, flags):
        fd = real_open(path, flags)
        opened.append(fd)
        return fd

    def fail_stat(fd):
        raise OSError(errno.EIO, "injected fstat failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "open", track_open)
        patcher.setattr(os, "fstat", fail_stat)
        with pytest.raises(OSError, match="injected fstat failure"):
            _open_regular_store(str(store), os.O_RDONLY)
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.read(opened[0], 1)
    assert closed.value.errno == errno.EBADF
