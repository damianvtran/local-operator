"""Where the app's files live, per platform and under the override.

``log_dir`` is the newest of these and the easiest to get subtly wrong: a
platform branch that is never exercised on the developer's machine is a branch
that ships broken. Each branch is forced here rather than trusted.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest

from local_operator.paths import (
    APP_DIRNAME,
    CONFIG_DIR_ENV,
    LOG_DIRNAME,
    config_dir,
    ensure_log_dir,
    log_dir,
)


def _pin_os_name(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``paths`` see ``os.name == name`` without lying to ``pathlib``.

    Patching the real ``os.name`` to ``"nt"`` makes ``Path()`` instantiate a
    ``WindowsPath``, which raises ``UnsupportedOperation`` on POSIX — the test
    would then fail for a reason that has nothing to do with the branch under
    test. Only the module's own view of ``os`` is swapped.
    """
    monkeypatch.setattr(
        "local_operator.paths.os",
        types.SimpleNamespace(name=name, environ=os.environ),
    )


def test_config_dir_honours_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    assert config_dir() == tmp_path


def test_log_dir_honours_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The override wins on every platform.

    This is what keeps a test run from writing into the developer's real
    ``~/Library/Logs``: a platform branch that ignored the override would leave
    litter that nothing cleans up and that no assertion would ever notice.
    """
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    monkeypatch.setattr("sys.platform", "darwin")
    assert log_dir() == tmp_path / LOG_DIRNAME
    monkeypatch.setattr("sys.platform", "linux")
    assert log_dir() == tmp_path / LOG_DIRNAME


def test_log_dir_macos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert log_dir() == tmp_path / "Library" / "Logs" / APP_DIRNAME


def test_log_dir_linux_defaults_to_xdg_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    _pin_os_name(monkeypatch, "posix")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert log_dir() == tmp_path / ".local" / "state" / APP_DIRNAME / LOG_DIRNAME


def test_log_dir_linux_honours_xdg_state_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr("sys.platform", "linux")
    _pin_os_name(monkeypatch, "posix")
    assert log_dir() == tmp_path / "state" / APP_DIRNAME / LOG_DIRNAME


def test_log_dir_windows_uses_local_app_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LOCALAPPDATA, not APPDATA: logs must never roam between machines."""
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    monkeypatch.setattr("sys.platform", "win32")
    _pin_os_name(monkeypatch, "nt")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    assert log_dir() == tmp_path / "Local" / APP_DIRNAME / "Logs"


def test_ensure_log_dir_creates_lazily_and_privately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "cfg"))
    target = tmp_path / "cfg" / LOG_DIRNAME
    assert not target.exists()

    created = ensure_log_dir()

    assert created == target
    assert target.is_dir()
    if os.name != "nt":
        assert target.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_ensure_log_dir_returns_none_when_uncreatable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A log directory that cannot be created must not raise.

    Logging is a diagnostic, not a startup requirement: a read-only home or a
    file squatting on the path has to degrade to "no log file", never to a CLI
    that refuses to start. A regular file where the directory belongs is the
    cheapest way to force a real OSError out of ``mkdir``.
    """
    blocked = tmp_path / "cfg"
    blocked.mkdir()
    (blocked / LOG_DIRNAME).write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(CONFIG_DIR_ENV, str(blocked))

    assert ensure_log_dir() is None


# --- Agent home dir (item 12) -----------------------------------------------


def test_agent_home_dir_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from local_operator.paths import AGENT_HOME_ENV, agent_home_dir

    monkeypatch.delenv(AGENT_HOME_ENV, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert agent_home_dir() == tmp_path / "local-operator-home"


def test_agent_home_dir_honours_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from local_operator.paths import AGENT_HOME_ENV, agent_home_dir

    monkeypatch.setenv(AGENT_HOME_ENV, str(tmp_path / "custom"))
    assert agent_home_dir() == tmp_path / "custom"


def test_default_agent_cwd_portable_without_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.paths import AGENT_HOME_ENV, default_agent_cwd

    monkeypatch.delenv(AGENT_HOME_ENV, raising=False)
    # Portable ~/-relative string when no override, so an exported agent record
    # replicates on another machine.
    assert default_agent_cwd() == "~/local-operator-home"


def test_default_agent_cwd_absolute_with_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator.paths import AGENT_HOME_ENV, default_agent_cwd

    monkeypatch.setenv(AGENT_HOME_ENV, str(tmp_path / "custom"))
    # Absolute override so the stored cwd cannot diverge from the created dir.
    assert default_agent_cwd() == str(tmp_path / "custom")


def test_ensure_agent_home_dir_creates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from local_operator.paths import AGENT_HOME_ENV, ensure_agent_home_dir

    target = tmp_path / "made" / "home"
    monkeypatch.setenv(AGENT_HOME_ENV, str(target))
    result = ensure_agent_home_dir()
    assert result == target and target.is_dir()
