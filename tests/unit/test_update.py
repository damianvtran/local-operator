"""PyPI updater: version compare, cache, install detection, CLI dispatch."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from local_operator import update as update_mod
from local_operator.interpreter import SAFE_PATH_FLAG
from local_operator.update import (
    TTL_S,
    InstallKind,
    MobileRefresh,
    UpdateError,
    check_latest,
    install_kind,
    installer_argv,
    is_behind,
    parse_version,
    perform_upgrade,
    refresh_mobile_after_upgrade,
    tui_editable_refusal,
    tui_installer_failure,
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


class _MetadataDist:
    """Distribution stand-in exposing only the ``read_text`` the module calls.

    ``object()`` is enough for the layout probes, which answer before any
    metadata is read. The ``INSTALLER`` path needs a real file body, and
    ``None`` — what ``read_text`` returns for an absent file — is the case
    that must stay ``UNKNOWN``.
    """

    def __init__(self, installer: str | None = None) -> None:
        self._installer = installer

    def read_text(self, name: str) -> str | None:
        return self._installer if name == "INSTALLER" else None


@contextmanager
def _base_interpreter(prefix: Path):
    """``sys.prefix == sys.base_prefix``: the mise/pyenv/asdf layout of #396.

    Patched rather than constructed because ``_is_ordinary_pip`` reads the
    live ``sys``; the seam keeps the running process untouched.
    """
    with (
        patch.object(update_mod.sys, "prefix", str(prefix)),
        patch.object(update_mod.sys, "base_prefix", str(prefix)),
    ):
        yield


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
    """Genuinely unidentifiable: base prefix, no ``pyvenv.cfg``, no ``INSTALLER``.

    Before #396 this asserted the bug — the same layout with a pip-written
    ``INSTALLER`` was called UNKNOWN too. Refusing still has to be possible,
    so the case is kept and stripped of the one signal that identifies it.
    """
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=False),
        _base_interpreter(tmp_path),
    ):
        dist.return_value = _MetadataDist(installer=None)
        assert install_kind(prefix=tmp_path) is InstallKind.UNKNOWN


@pytest.mark.parametrize("installer", ["uv", "pipx", "pipenv", "conda"])
def test_install_kind_unknown_for_unrecognised_installer(tmp_path: Path, installer: str) -> None:
    """Refuse-don't-guess: a foreign installer is not silently upgraded with pip.

    The values are chosen to be EXECUTABLE versions of the probe's own
    argument, not merely foreign names. ``uv`` is the one the docstring
    reasons about — ``uv pip install --system`` writes it into a base
    prefix, where ``uv tool upgrade`` is the wrong command — so accepting it
    here would be exactly the guess the module refuses. ``pipx`` and
    ``pipenv`` are the substring traps: a probe written as ``"pip" in
    installer`` rather than ``== "pip"`` passes a ``conda``-only test and
    fails these.
    """
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=False),
        _base_interpreter(tmp_path),
    ):
        dist.return_value = _MetadataDist(installer=installer)
        assert install_kind(prefix=tmp_path) is InstallKind.UNKNOWN


def test_install_kind_pip_on_base_interpreter(tmp_path: Path) -> None:
    """#396: ``pip install`` under mise/pyenv/asdf is PIP, not UNKNOWN.

    The reporter's prefix equalled ``base_prefix`` and carried no
    ``pyvenv.cfg``, so both layout probes declined and ``/update`` refused an
    upgrade ``pip install -U`` would have made. ``INSTALLER`` says ``pip``.
    """
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=False),
        _base_interpreter(tmp_path),
    ):
        dist.return_value = _MetadataDist(installer="pip\n")
        assert not (tmp_path / "pyvenv.cfg").exists()
        assert install_kind(prefix=tmp_path) is InstallKind.PIP


def test_install_kind_uv_tool_not_downgraded_to_pip(tmp_path: Path) -> None:
    """A uv tool env whose ``INSTALLER`` reads ``uv`` still upgrades with uv."""
    prefix = tmp_path / "uv" / "tools" / "local-operator"
    prefix.mkdir(parents=True)
    (prefix / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=False),
        _base_interpreter(prefix),
    ):
        dist.return_value = _MetadataDist(installer="uv")
        assert install_kind(prefix=prefix) is InstallKind.UV_TOOL


def test_install_kind_pipx_not_downgraded_to_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pipx vendors pip inside its venv, so its ``INSTALLER`` can read ``pip``."""
    home = tmp_path / "pipx-home"
    prefix = home / "venvs" / "local-operator"
    prefix.mkdir(parents=True)
    monkeypatch.setenv("PIPX_HOME", str(home))
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=False),
        _base_interpreter(prefix),
    ):
        dist.return_value = _MetadataDist(installer="pip")
        assert install_kind(prefix=prefix) is InstallKind.PIPX


def test_install_kind_editable_outranks_pip_installer(tmp_path: Path) -> None:
    """The editable guard runs first: ``INSTALLER`` names the tool, not the safety.

    A contributor's checkout is installed with ``pip install -e``, so its
    ``INSTALLER`` reads ``pip`` exactly like the #396 repro. Letting that
    signal reach ``_is_ordinary_pip`` would point ``pip install -U`` at the
    repo ``.venv`` and smash the editable link.
    """
    with (
        patch.object(update_mod, "distribution") as dist,
        patch.object(update_mod, "_is_editable_direct_url", return_value=True),
        _base_interpreter(tmp_path),
    ):
        dist.return_value = _MetadataDist(installer="pip")
        assert install_kind(prefix=tmp_path) is InstallKind.EDITABLE


def test_perform_upgrade_runs_detected_argv() -> None:
    seen: list[list[str]] = []

    def run(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    out = perform_upgrade(target="0.28.0", kind=InstallKind.UV_TOOL, run=run)
    assert out == "0.28.0"
    assert seen == [["uv", "tool", "install", "--force", "local-operator"]]

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
    assert installer_argv(InstallKind.UV_TOOL) == [
        "uv",
        "tool",
        "install",
        "--force",
        "local-operator",
    ]


def test_tui_refusal_copy_is_user_facing() -> None:
    assert "repo checkout" in tui_editable_refusal()
    assert "lop-update" in tui_editable_refusal()
    assert "uv tool install --force local-operator" in tui_installer_failure(InstallKind.UV_TOOL)
    assert "pipx upgrade local-operator" in tui_installer_failure(InstallKind.PIPX)


def _check(installed: str, latest: str | None, behind: bool) -> update_mod.VersionCheck:
    return update_mod.VersionCheck(installed=installed, latest=latest, behind=behind)


def test_update_command_check_behind(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.28.0", True)),
        patch.object(update_mod, "refresh_mobile_after_upgrade") as refresh,
    ):
        assert update_command(check=True) == 2
        refresh.assert_not_called()
    out = capsys.readouterr().out
    assert "local-operator 0.27.0" in out
    assert "latest on PyPI: 0.28.0" in out
    assert "run `lop update` to install" in out
    assert "mobile" not in out.lower()


def test_update_command_check_current(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.27.0", False)),
        patch.object(update_mod, "refresh_mobile_after_upgrade") as refresh,
    ):
        assert update_command(check=True) == 0
        refresh.assert_not_called()
    assert capsys.readouterr().out.strip() == "local-operator 0.27.0 is the latest"


def test_update_command_check_network_error(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(update_mod, "check_latest", return_value=_check("0.27.0", None, False)),
        patch.object(update_mod, "refresh_mobile_after_upgrade") as refresh,
    ):
        assert update_command(check=True) == 1
        refresh.assert_not_called()
    assert "could not reach PyPI" in capsys.readouterr().err


def test_update_command_already_latest(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.27.0", False)),
        patch.object(update_mod, "refresh_mobile_after_upgrade") as refresh,
    ):
        assert update_command(check=False) == 0
        refresh.assert_not_called()
    assert capsys.readouterr().out.strip() == "local-operator 0.27.0 is the latest"


def test_update_command_upgrades(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.28.0", True)),
        patch.object(update_mod, "install_kind", return_value=InstallKind.UV_TOOL),
        patch.object(update_mod, "is_git_snapshot", return_value=False),
        patch.object(update_mod, "perform_upgrade", return_value="0.28.0") as upgrade,
        patch.object(
            update_mod, "refresh_mobile_after_upgrade", return_value=MobileRefresh(kind="skipped")
        ) as refresh,
    ):
        assert update_command(check=False) == 0
        upgrade.assert_called_once()
        refresh.assert_called_once()
    captured = capsys.readouterr()
    assert "local-operator 0.27.0 (latest is 0.28.0)" in captured.out
    assert "upgrading via uv tool…" in captured.out
    assert "installed 0.28.0" in captured.out
    assert "mobile" not in captured.out
    assert captured.err == ""


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


class _FakePlist:
    """Reports existence without touching ``~/Library/LaunchAgents``."""

    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists


@contextmanager
def _upgrade_cmd():
    """Successful ``update_command`` path with the installer already done."""
    with (
        patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.28.0", True)),
        patch.object(update_mod, "install_kind", return_value=InstallKind.UV_TOOL),
        patch.object(update_mod, "is_git_snapshot", return_value=False),
        patch.object(update_mod, "perform_upgrade", return_value="0.28.0"),
    ):
        yield


def test_refresh_skipped_when_plist_absent() -> None:
    with (
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(False)),
        patch.object(update_mod, "_mobile_healthz_answers", return_value=False),
        patch("subprocess.run") as run,
    ):
        result = refresh_mobile_after_upgrade()
    assert result == MobileRefresh(kind="skipped")
    run.assert_not_called()


def test_refresh_unsupervised_when_live_without_plist() -> None:
    with (
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(False)),
        patch.object(update_mod, "_mobile_healthz_answers", return_value=True),
        patch("subprocess.run") as run,
    ):
        result = refresh_mobile_after_upgrade()
    assert result == MobileRefresh(kind="unsupervised")
    run.assert_not_called()


def test_refresh_restarts_via_new_distribution() -> None:
    with (
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run,
    ):
        result = refresh_mobile_after_upgrade()
    assert result == MobileRefresh(kind="restarted")
    run.assert_called_once()
    # `SAFE_PATH_FLAG`, and BEFORE `-m`: this runs with no `cwd=`, so a bare
    # `-m` would restart the daemon through a checkout that merely happened to
    # be the update's working directory -- pre-upgrade code reporting success.
    assert run.call_args.args[0] == [
        sys.executable,
        SAFE_PATH_FLAG,
        "-m",
        "local_operator.cli",
        "mobile",
        "restart",
    ]


def test_refresh_failed_child_exit() -> None:
    completed = subprocess.CompletedProcess([], 1, stdout="", stderr="launchctl: no such service")
    with (
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch("subprocess.run", return_value=completed),
    ):
        result = refresh_mobile_after_upgrade()
    assert result.kind == "failed"
    assert "no such service" in result.error


def test_refresh_failed_missing_binary() -> None:
    with (
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch("subprocess.run", side_effect=FileNotFoundError("launchctl")),
    ):
        result = refresh_mobile_after_upgrade()
    assert result.kind == "failed"
    assert "launchctl" in result.error


def test_refresh_failed_timeout() -> None:
    with (
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=30)),
    ):
        result = refresh_mobile_after_upgrade()
    assert result == MobileRefresh(kind="failed", error="timed out")


def test_refresh_never_falls_back_to_path_lop() -> None:
    """M1: PATH ``lop`` is a different install, not this one.

    After ``uv tool upgrade`` the interpreter path can vanish; restarting a
    PATH ``lop`` would serve whatever build THAT install has while the
    update reports success. The refresh must fail honestly instead.
    """
    with (
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch.object(update_mod.sys, "executable", "/gone/python"),
        patch.object(Path, "exists", return_value=False),
        patch("shutil.which", return_value="/usr/local/bin/lop"),
        patch("subprocess.run") as run,
    ):
        result = refresh_mobile_after_upgrade()
    assert result.kind == "failed"
    assert "lop is not on PATH" not in result.error
    run.assert_not_called()


def test_refresh_failed_when_executable_gone() -> None:
    with (
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch.object(update_mod.sys, "executable", "/gone/python"),
        patch.object(Path, "exists", return_value=False),
        patch("subprocess.run") as run,
    ):
        result = refresh_mobile_after_upgrade()
    assert result.kind == "failed"
    assert "interpreter vanished" in result.error
    run.assert_not_called()


def test_update_command_no_plist_prints_only_install_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        _upgrade_cmd(),
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(False)),
        patch.object(update_mod, "_mobile_healthz_answers", return_value=False),
        patch("subprocess.run") as run,
    ):
        assert update_command(check=False) == 0
    captured = capsys.readouterr()
    run.assert_not_called()
    assert captured.out.splitlines() == [
        "local-operator 0.27.0 (latest is 0.28.0)",
        "upgrading via uv tool…",
        "installed 0.28.0",
    ]
    assert captured.err == ""


def test_update_command_restarted_prints_phone_line(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        _upgrade_cmd(),
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run,
    ):
        assert update_command(check=False) == 0
    captured = capsys.readouterr()
    # See `test_refresh_restarts_via_new_distribution`: the isolation flag has
    # to survive the `lop update` path too, which is one of the two callers
    # that runs with a user cwd.
    assert run.call_args.args[0] == [
        sys.executable,
        SAFE_PATH_FLAG,
        "-m",
        "local_operator.cli",
        "mobile",
        "restart",
    ]
    assert "installed 0.28.0" in captured.out
    assert "mobile daemon restarted — refresh the phone UI" in captured.out
    assert captured.err == ""


def test_update_command_refresh_fail_still_zero(capsys: pytest.CaptureFixture[str]) -> None:
    completed = subprocess.CompletedProcess([], 1, stdout="", stderr="kickstart failed")
    with (
        _upgrade_cmd(),
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch("subprocess.run", return_value=completed) as run,
    ):
        assert update_command(check=False) == 0
    captured = capsys.readouterr()
    assert run.call_count == 1
    assert "installed 0.28.0" in captured.out
    # U1: the failure names the recovery, not just the cause.
    assert "warning: mobile daemon did not restart:" in captured.err
    assert "run lop mobile restart" in captured.err
    assert "kickstart failed" in captured.err


def test_update_command_refresh_missing_binary_still_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        _upgrade_cmd(),
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch("subprocess.run", side_effect=FileNotFoundError("No such file: launchctl")),
    ):
        assert update_command(check=False) == 0
    captured = capsys.readouterr()
    assert "installed 0.28.0" in captured.out
    assert "warning: mobile daemon did not restart:" in captured.err


def test_update_command_refresh_timeout_still_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        _upgrade_cmd(),
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(True)),
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=30)),
    ):
        assert update_command(check=False) == 0
    assert "warning: mobile daemon did not restart:" in capsys.readouterr().err


def test_update_command_unsupervised_warns(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        _upgrade_cmd(),
        patch.object(update_mod, "_mobile_plist_path", return_value=_FakePlist(False)),
        patch.object(update_mod, "_mobile_healthz_answers", return_value=True),
        patch("subprocess.run") as run,
    ):
        assert update_command(check=False) == 0
    captured = capsys.readouterr()
    run.assert_not_called()
    assert "installed 0.28.0" in captured.out
    assert "warning: a mobile daemon is running unsupervised" in captured.err
    # U2: `lop mobile restart` is launchd-only and is NOT the fix here.
    assert "lop mobile serve" in captured.err
    assert "lop mobile restart" not in captured.err


def test_perform_upgrade_does_not_refresh() -> None:
    with (
        patch.object(update_mod, "refresh_mobile_after_upgrade") as refresh,
        patch("subprocess.run") as run,
    ):
        out = perform_upgrade(target="0.28.0", kind=InstallKind.UV_TOOL, run=lambda _: 0)
    assert out == "0.28.0"
    refresh.assert_not_called()
    run.assert_not_called()


def test_installed_version_trusts_a_verified_checkout_in_either_direction() -> None:
    """A checkout that IS this install wins outright -- it is the running code.

    This used to take the MAXIMUM of the two, which only ever protected against
    a stray NEWER file. A stray OLDER one dragged the number DOWN, and no
    direction of `max` covers both, because the defect was never the comparison
    -- it was trusting a file whose relationship to the running code was
    unestablished. `_editable_source_version` now proves identity, so ordering
    is irrelevant and the source side is authoritative both ways:

    * forwards, the original bug: metadata never moves with the tree, so a
      checkout at 0.49.0 reported the 0.46.23 it was installed at (QA Q3/UX U13);
    * backwards, which `max` got wrong: an editable checkout deliberately moved
      to an OLDER revision really is running older code, and reporting the newer
      metadata would hide that.
    """
    with (
        patch.object(update_mod, "_editable_source_version", return_value="0.49.0"),
        patch.object(update_mod, "version", return_value="0.46.23"),
    ):
        assert update_mod.installed_version() == "0.49.0"

    with (
        patch.object(update_mod, "_editable_source_version", return_value="0.46.23"),
        patch.object(update_mod, "version", return_value="0.49.0"),
    ):
        assert update_mod.installed_version() == "0.46.23"


def test_installed_version_falls_back_to_metadata_without_a_verified_checkout() -> None:
    """Nothing proven about the tree leaves metadata as the only real input.

    This is the packaged-install case, and now also the case that broke: a
    stray checkout in the working directory yields `""` from the source side
    rather than overriding a correct install version.
    """
    with (
        patch.object(update_mod, "_editable_source_version", return_value=""),
        patch.object(update_mod, "version", return_value="0.49.2"),
    ):
        assert update_mod.installed_version() == "0.49.2"

    with (
        patch.object(update_mod, "_editable_source_version", return_value="0.49.2"),
        patch.object(update_mod, "version", side_effect=update_mod.PackageNotFoundError),
    ):
        assert update_mod.installed_version() == "0.49.2"

    # Neither side readable: empty, never an invented number.
    with (
        patch.object(update_mod, "_editable_source_version", return_value=""),
        patch.object(update_mod, "version", side_effect=update_mod.PackageNotFoundError),
    ):
        assert update_mod.installed_version() == ""


def test_editable_source_version_ignores_a_checkout_that_is_not_the_install(
    tmp_path: Path,
) -> None:
    """The measured defect: adjacency is not identity.

    The old implementation trusted any `pyproject.toml` beside the imported
    package, arguing that a released build has no adjacent project file and so
    could never read a stray one. That was false on a real install: a spawned
    child whose cwd was a checkout of this project imported THAT checkout, so
    `Path(__file__)` pointed into a tree the install had nothing to do with,
    and a 0.51.5 install reported 0.51.0.

    Here the module resolves inside `tmp_path` while the install's editable
    origin is a different directory -- exactly that shape.
    """
    stray = tmp_path / "stray-checkout"
    (stray / "local_operator").mkdir(parents=True)
    (stray / "pyproject.toml").write_text(
        '[project]\nname = "local-operator"\nversion = "0.9.9"\n', encoding="utf-8"
    )
    module = stray / "local_operator" / "update.py"
    module.write_text("", encoding="utf-8")

    with (
        patch.object(update_mod, "__file__", str(module)),
        patch.object(
            update_mod, "_editable_install_root", return_value=tmp_path / "the-real-install"
        ),
    ):
        assert update_mod._editable_source_version() == ""

    # ... and the same tree IS trusted once it is the install's editable source.
    with (
        patch.object(update_mod, "__file__", str(module)),
        patch.object(update_mod, "_editable_install_root", return_value=stray.resolve()),
    ):
        assert update_mod._editable_source_version() == "0.9.9"


def test_editable_source_version_ignores_a_non_editable_install(tmp_path: Path) -> None:
    """A packaged wheel has no editable `direct_url.json`, so no tree is trusted.

    Covers the released-build case directly: `_editable_install_root` returns
    `None`, and a `pyproject.toml` sitting next to the imported package -- which
    is what a stray checkout looks like -- must not be read.
    """
    stray = tmp_path / "checkout"
    (stray / "local_operator").mkdir(parents=True)
    (stray / "pyproject.toml").write_text(
        '[project]\nname = "local-operator"\nversion = "0.9.9"\n', encoding="utf-8"
    )
    with (
        patch.object(update_mod, "__file__", str(stray / "local_operator" / "update.py")),
        patch.object(update_mod, "_editable_install_root", return_value=None),
    ):
        assert update_mod._editable_source_version() == ""


def test_editable_install_root_reads_pep610_direct_url() -> None:
    """`dir_info.editable` plus a `file://` URL is the only accepted evidence."""
    with patch.object(
        update_mod,
        "_direct_url_payload",
        return_value={"url": "file:///tmp/checkout", "dir_info": {"editable": True}},
    ):
        assert update_mod._editable_install_root() == Path("/tmp/checkout").resolve()

    # A non-editable install (a wheel built from a temp dir, as `lop-update`
    # produces) names a path that is NOT the running tree; it must not qualify.
    with patch.object(
        update_mod,
        "_direct_url_payload",
        return_value={"url": "file:///tmp/build-dir", "dir_info": {}},
    ):
        assert update_mod._editable_install_root() is None

    for payload in ({"dir_info": {"editable": True}}, {"url": "https://pypi.org/x"}, None):
        with patch.object(update_mod, "_direct_url_payload", return_value=payload):
            assert update_mod._editable_install_root() is None


def test_editable_source_version_never_raises() -> None:
    """ "Cheap and total": a version readout must degrade to `""`, never throw."""
    with patch.object(update_mod, "_editable_install_root", side_effect=OSError("boom")):
        assert update_mod._editable_source_version() == ""


def _metadata_dir(root: Path, name: str, version: str, direct_url: str | None = None) -> Path:
    """Write a real on-disk metadata directory `importlib.metadata` can discover."""
    path = root / name
    path.mkdir(parents=True)
    key = "METADATA" if name.endswith(".dist-info") else "PKG-INFO"
    (path / key).write_text(
        f"Metadata-Version: 2.1\nName: local-operator\nVersion: {version}\n", encoding="utf-8"
    )
    if direct_url is not None:
        (path / "direct_url.json").write_text(direct_url, encoding="utf-8")
    return path


def test_direct_url_payload_ignores_a_shadowing_egg_info(tmp_path: Path) -> None:
    """A stale `*.egg-info` must not make a real editable install look absent.

    THE REGRESSION THIS PINS. `local_operator.egg-info/` is a gitignored build
    artifact that any `pip install -e` / `setup.py` run leaves in the checkout,
    so it is present in real trees. Whenever the cwd is on `sys.path` it sorts
    AHEAD of site-packages, and `distribution("local-operator")` -- which
    returns the first name match in path order -- resolves to it. Egg-info
    metadata predates PEP 610 and carries no `direct_url.json`, so reading the
    marker through that single lookup answered "not an editable install" for an
    install that plainly was one: `_editable_install_root()` returned `None`,
    the identity check failed against its own tree, and `installed_version()`
    fell through to the stale `PKG-INFO` number.

    Measured on a genuine `uv pip install -e` with `pyproject.toml` at 0.51.7
    and a leftover `PKG-INFO` at 0.46.23, that reported **0.46.23** -- the
    Settings > Updates staleness (QA Q3 / UX U13) reintroduced by the shadow.

    Real metadata directories on disk, not a patched payload: the ordering IS
    the defect, so a fixture that hands over one distribution cannot observe it.
    """
    site = tmp_path / "site"
    # Written egg-info first so it is the earlier entry in discovery order,
    # which is the shape that shadowed the dist-info on a real checkout.
    _metadata_dir(site, "local_operator.egg-info", "0.46.23")
    _metadata_dir(
        site,
        "local_operator-0.51.7.dist-info",
        "0.51.7",
        direct_url='{"url": "file:///real/checkout", "dir_info": {"editable": true}}',
    )

    def _scan(*, name: str) -> list[object]:
        from importlib.metadata import distributions

        return list(distributions(name=name, path=[str(site)]))

    with patch.object(update_mod, "distributions", _scan):
        # The egg-info is found FIRST and has no marker; the dist-info's must
        # still be the answer, or a real editable install reads as no install.
        assert update_mod._direct_url_payload() == {
            "url": "file:///real/checkout",
            "dir_info": {"editable": True},
        }
        assert update_mod._editable_install_root() == Path("/real/checkout").resolve()


def test_installed_version_survives_a_shadowing_egg_info(tmp_path: Path) -> None:
    """End to end: the checkout's version wins over the shadowed stale number.

    `_direct_url_payload` is the unit; this is the number the user actually
    sees. `version("local-operator")` still resolves to the egg-info's stale
    0.46.23 (that shadowing is `importlib.metadata`'s behaviour and not ours to
    change), so this asserts the correction survives all the way out.
    """
    site = tmp_path / "site"
    checkout = tmp_path / "checkout"
    (checkout / "local_operator").mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        '[project]\nname = "local-operator"\nversion = "0.51.7"\n', encoding="utf-8"
    )
    _metadata_dir(site, "local_operator.egg-info", "0.46.23")
    _metadata_dir(
        site,
        "local_operator-0.51.7.dist-info",
        "0.51.7",
        direct_url=json.dumps({"url": checkout.resolve().as_uri(), "dir_info": {"editable": True}}),
    )

    def _scan(*, name: str) -> list[object]:
        from importlib.metadata import distributions

        return list(distributions(name=name, path=[str(site)]))

    with (
        patch.object(update_mod, "distributions", _scan),
        patch.object(update_mod, "__file__", str(checkout / "local_operator" / "update.py")),
        # The shadow the egg-info casts over the metadata channel, reproduced.
        patch.object(update_mod, "version", return_value="0.46.23"),
    ):
        assert update_mod.installed_version() == "0.51.7"


# ---------------------------------------------------------------------------
# Build stamps: the comparable token a viewer and a runtime skew against
# ---------------------------------------------------------------------------


def test_source_ref_reads_the_commit_lop_update_recorded(tmp_path: Path) -> None:
    """``.lop-source`` holds ``<git-sha> <tag>``; only the sha identifies a build.

    The tag half repeats across every rebuild of one release, so keying on it
    would report two genuinely different builds as identical — which is the
    exact drift this host produces most often.
    """
    (tmp_path / ".lop-source").write_text(
        "4d3ce1d1a48f4f3b799efdfabb014979e70e0630 v0.49.0\n", encoding="utf-8"
    )
    assert update_mod.source_ref(tmp_path) == "4d3ce1d1a48f4f3b799efdfabb014979e70e0630"


def test_source_ref_is_empty_without_a_marker(tmp_path: Path) -> None:
    """PyPI wheels, pipx installs and editable checkouts have no marker.

    Empty rather than an error: those installs fall back to comparing on the
    distribution version alone, and dev-tree skew is out of scope by design.
    """
    assert update_mod.source_ref(tmp_path) == ""


def test_source_ref_survives_an_unreadable_marker(tmp_path: Path) -> None:
    """A build token is decoration on a diagnostic path.

    It is read at adopt, engage and bind — seams that must never fail because
    a marker file was a directory or had its permissions changed underneath.
    """
    (tmp_path / ".lop-source").mkdir()
    assert update_mod.source_ref(tmp_path) == ""


def test_source_ref_of_an_empty_marker_is_empty(tmp_path: Path) -> None:
    (tmp_path / ".lop-source").write_text("   \n", encoding="utf-8")
    assert update_mod.source_ref(tmp_path) == ""


def test_installed_build_pairs_the_version_with_the_ref(tmp_path: Path) -> None:
    (tmp_path / ".lop-source").write_text("abc1234def v0.49.0\n", encoding="utf-8")
    with patch.object(update_mod, "installed_version", return_value="0.49.0"):
        stamp = update_mod.installed_build(tmp_path)
    assert stamp == update_mod.BuildStamp(version="0.49.0", source_ref="abc1234def")


def test_same_version_rebuilds_are_different_builds() -> None:
    """The case that motivates the ref, stated as an equality.

    ``lop-update`` builds from ``main`` while ``pyproject.toml`` still names
    the last released version, so two different builds share one version
    string. Comparing on version alone reports "no drift" for the drift this
    host actually has.
    """
    before = update_mod.BuildStamp(version="0.49.0", source_ref="aaaaaaa1111")
    after = update_mod.BuildStamp(version="0.49.0", source_ref="bbbbbbb2222")
    assert before != after
    assert before.version == after.version, "the version alone cannot tell them apart"


def test_a_build_label_names_the_ref_only_when_there_is_one() -> None:
    """Notice copy: ``0.49.0`` on a wheel, ``0.49.0@4d3ce1d`` on a snapshot."""
    assert update_mod.BuildStamp(version="0.49.0").label() == "0.49.0"
    assert (
        update_mod.BuildStamp(version="0.49.0", source_ref="4d3ce1d1a48").label()
        == "0.49.0@4d3ce1d"
    )
    assert update_mod.BuildStamp(version="").label() == "unknown"


def test_installed_version_rereads_disk_within_one_process(tmp_path: Path) -> None:
    """The empirical claim the whole design rests on, pinned as a test.

    Drift detection compares what THIS process loaded against what is on disk
    NOW, which only works if ``importlib.metadata`` actually re-reads. It
    does, because the dist-info DIRECTORY NAME carries the version, so even a
    path-keyed cache misses. Asserted on a synthetic distribution so the test
    owns both sides and never depends on the real install.
    """
    import sys as _sys
    from importlib.metadata import version as _version

    site = tmp_path / "site"
    old = site / "skewpkg-0.46.23.dist-info"
    old.mkdir(parents=True)
    (old / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: skewpkg\nVersion: 0.46.23\n", encoding="utf-8"
    )
    _sys.path.insert(0, str(site))
    try:
        assert _version("skewpkg") == "0.46.23"
        new = site / "skewpkg-0.49.0.dist-info"
        new.mkdir()
        (new / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: skewpkg\nVersion: 0.49.0\n", encoding="utf-8"
        )
        import shutil

        shutil.rmtree(old)
        assert _version("skewpkg") == "0.49.0", (
            "importlib.metadata must re-read disk in a live process, or a TUI "
            "can never notice that lop-update replaced the install under it"
        )
    finally:
        _sys.path.remove(str(site))


def test_cli_version_flag_reports_the_running_build(tmp_path: Path) -> None:
    """`--version` must answer from `installed_version()`, not raw metadata.

    THE SURFACE THE USER CHECKS FIRST. Install metadata is written once and
    never moves when the checkout's `pyproject.toml` does, so
    `importlib.metadata.version("local-operator")` reports the version an
    editable install was CREATED at rather than the one it is running — and a
    leftover `*.egg-info` shadows the real dist-info downward on top of that.
    `installed_version()` corrects both; `--version` used to bypass it, so the
    one surface a user reads to answer "which build am I on?" was also the one
    still answering from the stale channel. That was the reported symptom this
    change ships with, so shipping it uncorrected invites a duplicate report of
    the bug it fixes.

    THE FIXTURE HAS TO CREATE THE DIVERGENCE, or the test is vacuous: in a
    clean checkout both channels agree, so asserting they match passes just as
    well on the bypassing form (verified — the first version of this test did).
    So the editable source is pinned to a version the metadata does not have,
    which is the ordinary state of any editable checkout after a release bump.

    Driven as a REAL SUBPROCESS through the parser builder, because the value
    is baked in at construction time by an `argparse` `action="version"`: an
    in-process call reads whatever the already-imported `cli` module captured,
    which is exactly what a stale `.pyc` would hide.
    """
    repo = Path(__file__).resolve().parents[2]
    marker = "9.9.9"
    # Force the two channels apart: `_editable_source_version` is the input
    # `installed_version()` prefers, and no real metadata can report 9.9.9.
    probe = (
        "import local_operator.update as u;"
        f"u._editable_source_version=lambda: {marker!r};"
        "import local_operator.cli as c;"
        "p=c.build_cli_parser();"
        "a=[x for x in p._actions if '--version' in getattr(x,'option_strings',[])][0];"
        "print(a.version)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(repo)
    )
    assert out.returncode == 0, f"probe failed:\n{out.stderr[-2000:]}"
    printed = out.stdout.strip().splitlines()[-1]
    assert printed == f"v{marker}", (
        "`--version` did not read through installed_version(): "
        f"got {printed!r}, expected 'v{marker}'. A raw "
        "importlib.metadata.version() call reports the installed metadata and "
        "cannot see the running checkout's version."
    )
