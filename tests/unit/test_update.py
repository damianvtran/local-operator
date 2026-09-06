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
    assert run.call_args.args[0] == [
        sys.executable,
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
    assert run.call_args.args[0] == [
        sys.executable,
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


def test_installed_version_prefers_whichever_source_is_newer() -> None:
    """Neither the checkout file nor the install metadata is authoritative alone.

    Both are stale in opposite directions, so `installed_version` takes the
    higher of the two:

    * metadata never moves with the tree, which reported 0.46.23 for a checkout
      at 0.49.0 -- the bug the checkout lookup was added for (QA Q3 / UX U13);
    * a `pyproject.toml` naming this project but declaring an older version (a
      fixture, a scratch tree, a hand-edited `version = "0.1.0"`) overrode
      correct metadata and produced a spurious "update available" (review round
      2, MINOR-2).
    """
    with (
        patch.object(update_mod, "_editable_source_version", return_value="0.49.0"),
        patch.object(update_mod, "version", return_value="0.46.23"),
    ):
        assert update_mod.installed_version() == "0.49.0"

    with (
        patch.object(update_mod, "_editable_source_version", return_value="0.1.0"),
        patch.object(update_mod, "version", return_value="0.49.2"),
    ):
        assert update_mod.installed_version() == "0.49.2"


def test_installed_version_falls_back_when_a_side_is_missing_or_unparseable() -> None:
    """An unorderable side cannot win; the other one is the only defensible answer."""
    with (
        patch.object(update_mod, "_editable_source_version", return_value="0.28.0rc1"),
        patch.object(update_mod, "version", return_value="0.49.2"),
    ):
        assert update_mod.installed_version() == "0.49.2"

    with (
        patch.object(update_mod, "_editable_source_version", return_value="0.49.2"),
        patch.object(update_mod, "version", return_value="not-a-version"),
    ):
        assert update_mod.installed_version() == "0.49.2"

    # No adjacent project file: a packaged install reports its metadata exactly
    # as it always did.
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
