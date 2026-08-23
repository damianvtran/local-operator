"""PyPI updater: version compare, cache, install detection, CLI dispatch."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from local_operator import update as update_mod
from local_operator.update import (
    TTL_S,
    InstallKind,
    UpdateError,
    check_latest,
    install_kind,
    installer_argv,
    is_behind,
    parse_version,
    perform_upgrade,
    update_command,
)


def _pypi_transport(
    status: int = 200, version: str = "0.28.0", delay: float = 0
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if delay:
            raise httpx.TimeoutException("timed out")
        if status != 200:
            return httpx.Response(status, text="nope")
        return httpx.Response(status, json={"info": {"version": version}})

    return httpx.MockTransport(handler)


def _client(transport: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=transport)


def test_parse_version_accepts_only_x_y_z() -> None:
    assert parse_version("0.27.0") == (0, 27, 0)
    assert parse_version("1.0.0") == (1, 0, 0)
    assert parse_version("0.28.0rc1") is None
    assert parse_version("not-a-version") is None
    assert parse_version("") is None


def test_is_behind_is_strict_and_unparseable_is_not() -> None:
    assert is_behind("0.27.0", "0.28.0") is True
    assert is_behind("0.28.0", "0.28.0") is False
    assert is_behind("0.28.0", "0.27.0") is False
    assert is_behind("0.28.0rc1", "0.28.0") is False
    assert is_behind("0.27.0", None) is False
    assert is_behind("", "0.28.0") is False


def test_check_latest_newer_same_older(tmp_path: Path) -> None:
    with patch.object(update_mod, "installed_version", return_value="0.27.0"):
        newer = check_latest(
            force=True, cache_dir=tmp_path, client=_client(_pypi_transport(version="0.28.0"))
        )
        assert newer.behind is True
        assert newer.latest == "0.28.0"

        same = check_latest(
            force=True, cache_dir=tmp_path, client=_client(_pypi_transport(version="0.27.0"))
        )
        assert same.behind is False
        assert same.latest == "0.27.0"

        older = check_latest(
            force=True, cache_dir=tmp_path, client=_client(_pypi_transport(version="0.26.0"))
        )
        assert older.behind is False
        assert older.latest == "0.26.0"


def test_check_latest_500_and_timeout_are_silent(tmp_path: Path) -> None:
    with patch.object(update_mod, "installed_version", return_value="0.27.0"):
        failed = check_latest(
            force=True, cache_dir=tmp_path, client=_client(_pypi_transport(status=500))
        )
        assert failed.latest is None
        assert failed.behind is False

        timed = check_latest(
            force=True, cache_dir=tmp_path, client=_client(_pypi_transport(delay=1))
        )
        assert timed.latest is None
        assert timed.behind is False


def test_corrupt_cache_is_missing(tmp_path: Path) -> None:
    cache = tmp_path / "pypi-local-operator.json"
    cache.write_text("not-json", encoding="utf-8")
    with patch.object(update_mod, "installed_version", return_value="0.27.0"):
        result = check_latest(cache_dir=tmp_path, client=_client(_pypi_transport(version="0.28.0")))
    assert result.latest == "0.28.0"
    assert result.behind is True


def test_stale_cache_used_on_failure(tmp_path: Path) -> None:
    cache = tmp_path / "pypi-local-operator.json"
    cache.write_text(
        json.dumps({"fetched_at": time.time() - TTL_S - 10, "payload": {"version": "0.28.0"}}),
        encoding="utf-8",
    )
    with patch.object(update_mod, "installed_version", return_value="0.27.0"):
        result = check_latest(cache_dir=tmp_path, client=_client(_pypi_transport(status=500)))
    assert result.latest == "0.28.0"
    assert result.behind is True


def test_fresh_cache_skips_get(tmp_path: Path) -> None:
    cache = tmp_path / "pypi-local-operator.json"
    cache.write_text(
        json.dumps({"fetched_at": time.time(), "payload": {"version": "0.28.0"}}),
        encoding="utf-8",
    )
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(200, json={"info": {"version": "9.9.9"}})

    with patch.object(update_mod, "installed_version", return_value="0.27.0"):
        result = check_latest(cache_dir=tmp_path, client=_client(httpx.MockTransport(handler)))
    assert hits["n"] == 0
    assert result.latest == "0.28.0"


def test_force_bypasses_ttl_and_rewrites(tmp_path: Path) -> None:
    cache = tmp_path / "pypi-local-operator.json"
    cache.write_text(
        json.dumps({"fetched_at": time.time(), "payload": {"version": "0.27.0"}}),
        encoding="utf-8",
    )
    with patch.object(update_mod, "installed_version", return_value="0.27.0"):
        result = check_latest(
            force=True, cache_dir=tmp_path, client=_client(_pypi_transport(version="0.28.0"))
        )
    assert result.latest == "0.28.0"
    written = json.loads(cache.read_text(encoding="utf-8"))
    assert written["payload"]["version"] == "0.28.0"


def test_install_kind_uv_receipt(tmp_path: Path) -> None:
    prefix = tmp_path / "uv" / "tools" / "local-operator"
    prefix.mkdir(parents=True)
    (prefix / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
    (prefix / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=False),
    ):
        dist.return_value = object()
        assert install_kind(prefix=prefix) is InstallKind.UV_TOOL


def test_install_kind_uv_path_without_receipt(tmp_path: Path) -> None:
    prefix = tmp_path / "share" / "uv" / "tools" / "local-operator"
    prefix.mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=False),
    ):
        dist.return_value = object()
        assert install_kind(prefix=prefix) is InstallKind.UV_TOOL


def test_install_kind_pipx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "pipx-home"
    prefix = home / "venvs" / "local-operator"
    prefix.mkdir(parents=True)
    (prefix / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    monkeypatch.setenv("PIPX_HOME", str(home))
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=False),
    ):
        dist.return_value = object()
        assert install_kind(prefix=prefix) is InstallKind.PIPX


def test_install_kind_editable(tmp_path: Path) -> None:
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=True),
    ):
        dist.return_value = object()
        assert install_kind(prefix=tmp_path) is InstallKind.EDITABLE


def test_install_kind_no_distribution(tmp_path: Path) -> None:
    from importlib.metadata import PackageNotFoundError

    with patch.object(
        update_mod, "distribution", side_effect=PackageNotFoundError("local-operator")
    ):
        assert install_kind(prefix=tmp_path) is InstallKind.EDITABLE


def test_install_kind_unknown(tmp_path: Path) -> None:
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=False),
        patch.object(update_mod.sys, "prefix", str(tmp_path)),
        patch.object(update_mod.sys, "base_prefix", str(tmp_path)),
    ):
        dist.return_value = object()
        assert install_kind(prefix=tmp_path) is InstallKind.UNKNOWN


def test_perform_upgrade_runs_detected_argv() -> None:
    seen: list[list[str]] = []

    def run(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    out = perform_upgrade(target="0.28.0", kind=InstallKind.UV_TOOL, run=run)
    assert out == "0.28.0"
    assert seen == [["uv", "tool", "upgrade", "local-operator"]]

    seen.clear()
    perform_upgrade(target="0.28.0", kind=InstallKind.PIPX, run=run)
    assert seen == [["pipx", "upgrade", "local-operator"]]

    seen.clear()
    perform_upgrade(target="0.28.0", kind=InstallKind.PIP, run=run, executable="/venv/bin/python")
    assert seen == [["/venv/bin/python", "-m", "pip", "install", "-U", "local-operator"]]


def test_perform_upgrade_refuses_editable_and_unknown() -> None:
    with pytest.raises(UpdateError, match="repo .venv"):
        perform_upgrade(target="0.28.0", kind=InstallKind.EDITABLE, run=lambda _: 0)
    with pytest.raises(UpdateError, match="cannot tell"):
        perform_upgrade(target="0.28.0", kind=InstallKind.UNKNOWN, run=lambda _: 0)


def test_perform_upgrade_nonzero_installer() -> None:
    with pytest.raises(UpdateError, match="exited 9"):
        perform_upgrade(target="0.28.0", kind=InstallKind.UV_TOOL, run=lambda _: 9)


def test_installer_argv_matches_kind() -> None:
    assert installer_argv(InstallKind.UV_TOOL) == ["uv", "tool", "upgrade", "local-operator"]


def _check(installed: str, latest: str | None, behind: bool) -> update_mod.VersionCheck:
    return update_mod.VersionCheck(installed=installed, latest=latest, behind=behind)


def test_update_command_check_behind(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.28.0", True)):
        assert update_command(check=True) == 2
    out = capsys.readouterr().out
    assert "local-operator 0.27.0" in out
    assert "latest on PyPI: 0.28.0" in out
    assert "run `lop update` to install" in out


def test_update_command_check_current(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.27.0", False)):
        assert update_command(check=True) == 0
    assert capsys.readouterr().out.strip() == "local-operator 0.27.0 is the latest"


def test_update_command_check_network_error(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(update_mod, "check_latest", return_value=_check("0.27.0", None, False)):
        assert update_command(check=True) == 1
    assert "could not reach PyPI" in capsys.readouterr().err


def test_update_command_already_latest(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.27.0", False)):
        assert update_command(check=False) == 0
    assert capsys.readouterr().out.strip() == "local-operator 0.27.0 is the latest"


def test_update_command_upgrades(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.28.0", True)),
        patch.object(update_mod, "install_kind", return_value=InstallKind.UV_TOOL),
        patch.object(update_mod, "is_git_snapshot", return_value=False),
        patch.object(update_mod, "perform_upgrade", return_value="0.28.0") as upgrade,
    ):
        assert update_command(check=False) == 0
        upgrade.assert_called_once()
    out = capsys.readouterr().out
    assert "local-operator 0.27.0 (latest is 0.28.0)" in out
    assert "upgrading via uv tool…" in out
    assert "installed 0.28.0" in out


def test_main_dispatches_update_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "update", "--check"])
    with patch("local_operator.update.update_command", return_value=2) as cmd:
        assert main() == 2
        cmd.assert_called_once_with(check=True)


def test_main_dispatches_update(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "update"])
    with patch("local_operator.update.update_command", return_value=0) as cmd:
        assert main() == 0
        cmd.assert_called_once_with(check=False)
