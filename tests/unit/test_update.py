"""PyPI updater: version compare, cache, install detection, CLI dispatch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from local_operator import procname
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
    installer_invocation,
    is_behind,
    parse_version,
    perform_upgrade,
    refresh_mobile_after_upgrade,
    tui_editable_refusal,
    tui_installer_failure,
    update_command,
)

_REAL_SERVICES_REFUSAL = update_mod._services_refusal


def _install_kind_double(kind: InstallKind):
    """A stand-in for `install_kind` that mirrors its REAL signature.

    NOT `lambda *a, **k`. That accepts anything, which means it cannot see a
    POSITIONAL call to a keyword-only function — and a positional call is exactly
    what shipped in serve-reload review round 6's R6-1: `install_kind(mine)` raised
    `TypeError` in production while 145 tests passed, because every double here had
    a wider signature than the function it stood in for and the guard happened to
    short-circuit before reaching the line in this venv. A double must be no more
    permissive than the thing it replaces, or it is a test that cannot fail.
    """

    def _kind(*, prefix: Any = None, executable: Any = None) -> InstallKind:
        return kind

    return _kind


@pytest.fixture(autouse=True)
def _owns_this_machines_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to "this install owns the fleet" for every test in this module.

    The ownership check has its own tests below; every OTHER test here is about
    the shape of `lop update`'s output or its failure paths, and without this each
    of them would have to fabricate an install tree under a `generations` root and
    a `sys.prefix` inside it — which is a pytest process's venv, so they would all
    be asserting against a refusal instead of against the thing they were written
    for.

    THE GUARD FIRES ON KIND HERE, which is the honest reason rather than the
    prefix one this docstring first gave (serve-reload review round 4, R4-3, and
    round 5, R5-3, which caught that the first correction still described the
    wrong half): a pytest process inside this worktree reports
    ``install_kind() == EDITABLE``, so it is refused before the membership
    question is ever reached. The prefix half would refuse it too, but that is a
    coincidence of where the venv lives rather than the reason.

    WHAT THIS HIDES, stated so it is not discovered by surprise: while it is in
    force NO test in this module can see the guard through
    `update_command`. That is why the test that must see it — the upgrade-path one
    below, which is the shape that broke in serve-reload R4-1 — restores the real
    drives `update_command` itself rather than calling `_services_stage`.

    The restore is `_REAL_SERVICES_REFUSAL`, not a re-implementation: a test's own
    `monkeypatch` runs after this fixture, so it wins.
    """
    monkeypatch.setattr(update_mod, "_services_refusal", lambda *a, **k: None)


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


@pytest.fixture
def branded_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the labelled rung, so a test never depends on the host's venv.

    The label rides with a planted image (see ``procname.spawn_identity``): a
    machine that cannot plant one — a framework interpreter, CI's Linux job —
    gets the bare interpreter and no label instead. The tests below assert the
    PAIRING, so the image is forced rather than hoped for; the rung-2 shape has
    its own tests in ``test_spawn_naming_fallback.py``.
    """
    monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: Path(sys.executable))


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


def test_perform_upgrade_runs_detected_argv(tmp_path: Path, branded_image) -> None:
    seen: list[list[str]] = []

    def run(argv: list[str]) -> int:
        seen.append(argv)
        return 0

    # ``prefix`` is passed on every call because a successful UV_TOOL upgrade
    # now writes ``.lop-source`` at the prefix; without it this test would
    # drop a marker into the developer's own ``sys.prefix``.
    out = perform_upgrade(target="0.28.0", kind=InstallKind.UV_TOOL, run=run, prefix=tmp_path)
    assert out == "0.28.0"
    assert seen == [["uv", "tool", "install", "--force", "local-operator"]]

    seen.clear()
    perform_upgrade(target="0.28.0", kind=InstallKind.PIPX, run=run, prefix=tmp_path)
    assert seen == [["pipx", "upgrade", "local-operator"]]

    seen.clear()
    perform_upgrade(
        target="0.28.0",
        kind=InstallKind.PIP,
        run=run,
        executable="/venv/bin/python",
        prefix=tmp_path,
    )
    # The pip path is the one installer this product runs ITSELF, so its argv[0]
    # is the role label and the interpreter travels beside it — see
    # `installer_invocation`. A caller that took this argv alone and spawned it
    # without `executable=` would ask the kernel to execute a file named
    # "Local Operator [install] pip".
    assert seen == [
        [
            procname.branded_argv0(procname.LABEL_INSTALL),
            "-m",
            "pip",
            "install",
            "-U",
            "local-operator",
        ]
    ]


def test_installer_invocation_pairs_the_label_with_its_image(branded_image) -> None:
    """The pairing, at the seam a spawn actually uses."""
    argv, executable = installer_invocation(InstallKind.PIP, executable="/venv/bin/python")
    assert argv[0] == procname.branded_argv0(procname.LABEL_INSTALL)
    assert executable == "/venv/bin/python"


def test_installer_invocation_leaves_third_party_binaries_named() -> None:
    """``uv`` and ``pipx`` keep their own argv[0] AND their own image.

    Labelling them would both mislabel the row (they are named already) and lose
    the binary the user's PATH resolves, which is the documented reason
    ``installer_argv`` returned them untouched before this change.
    """
    assert installer_invocation(InstallKind.UV_TOOL) == (
        ["uv", "tool", "install", "--force", "local-operator"],
        None,
    )
    assert installer_invocation(InstallKind.PIPX) == (
        ["pipx", "upgrade", "local-operator"],
        None,
    )


def test_perform_upgrade_refuses_editable_and_unknown() -> None:
    with pytest.raises(UpdateError, match="repo .venv"):
        perform_upgrade(target="0.28.0", kind=InstallKind.EDITABLE, run=lambda _: 0)
    with pytest.raises(UpdateError, match="cannot tell"):
        perform_upgrade(target="0.28.0", kind=InstallKind.UNKNOWN, run=lambda _: 0)


def test_perform_upgrade_nonzero_installer(tmp_path: Path) -> None:
    with pytest.raises(UpdateError, match="exited 9"):
        perform_upgrade(target="0.28.0", kind=InstallKind.UV_TOOL, run=lambda _: 9, prefix=tmp_path)


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


def _check_latest_double(installed: str, latest: str | None, behind: bool):
    """A stand-in for `check_latest` that mirrors its REAL signature.

    Same rule as `_install_kind_double`, and the reason it exists (serve-reload
    review round 7, R7-3): the real `check_latest` is keyword-only, so
    `lambda force=False: ...` accepts a POSITIONAL call it would reject — a double
    more permissive than the function it replaces, which is a test that cannot fail.
    It also builds the real `VersionCheck` through `_check` rather than a
    `SimpleNamespace`, so `update_command` is handed the type it actually reads.
    """

    def _latest(*, force: bool = False, cache_dir: Any = None, client: Any = None):
        return _check(installed, latest, behind)

    return _latest


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
    """Nothing to INSTALL is not nothing to do.

    This test used to assert that the daemon refresh was NOT called on this path,
    which was the bug rather than the contract (serve-reload review round 1, R1-2): the
    reporting machine printed exactly this line, returned 0, and left its backend
    on a build four releases old. The canary is now the SERVICES STAGE, and it is
    asserted to run — with the install itself untouched, which is what "is the
    latest" still means.
    """
    with (
        patch.object(update_mod, "check_latest", return_value=_check("0.27.0", "0.27.0", False)),
        patch.object(update_mod, "perform_upgrade") as perform,
        patch.object(update_mod, "_services_stage") as stage,
    ):
        assert update_command(check=False) == 0
        perform.assert_not_called()
        stage.assert_called_once_with()
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
        cmd.assert_called_once_with(
            check=True, refresh_daemons=False, from_snapshot=None, services=True
        )


def test_main_dispatches_update(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "update"])
    with patch("local_operator.update.update_command", return_value=0) as cmd:
        assert main() == 0
        cmd.assert_called_once_with(
            check=False, refresh_daemons=False, from_snapshot=None, services=True
        )


def test_main_dispatches_from_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--from-snapshot`` reaches the installer, and ``--check`` refuses it."""
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "update", "--from-snapshot", "main"])
    with patch("local_operator.update.update_command", return_value=0) as cmd:
        assert main() == 0
        cmd.assert_called_once_with(
            check=False, refresh_daemons=False, from_snapshot="main", services=True
        )

    monkeypatch.setattr("sys.argv", ["lop", "update", "--check", "--from-snapshot", "main"])
    with patch("local_operator.update.update_command", return_value=1) as refused:
        assert main() == 1
        refused.assert_called_once()


def test_main_dispatches_no_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--no-services`` reaches the installer as the escape hatch it is.

    The default is to finish the job (move the serves onto the new build); the
    flag is what a caller that will start the daemons itself uses, so it has to
    survive the CLI rather than only existing in the function's signature.
    """
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "update", "--no-services"])
    with patch("local_operator.update.update_command", return_value=0) as cmd:
        assert main() == 0
        cmd.assert_called_once_with(
            check=False, refresh_daemons=False, from_snapshot=None, services=False
        )


def test_update_runs_the_services_stage_when_nothing_needs_installing(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """serve-reload R1-2: the reported bug, as a regression test.

    `lop update` on the reporting machine printed "0.59.0 is the latest" and
    returned 0 without reaching the daemon/services stage — because `behind` is a
    version-string compare and the SERVICES are not versioned by the pointer at
    all. So the one machine this change exists for was the one machine where the
    change did nothing. The stage is idempotent (a daemon already on the current
    build is reported and not touched), so it runs on both paths.
    """
    from local_operator import update

    ran: list[str] = []
    monkeypatch.setattr(
        update,
        "check_latest",
        _check_latest_double("0.59.0", "0.59.0", False),
    )
    monkeypatch.setattr(update, "_services_stage", lambda: ran.append("services"))
    assert update.update_command() == 0
    assert ran == ["services"]
    assert "0.59.0 is the latest" in capsys.readouterr().out


def test_update_no_services_still_repairs_the_supervised_daemons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--no-services`` is the pre-change behaviour, not "do nothing"."""
    from local_operator import update

    ran: list[str] = []
    monkeypatch.setattr(
        update,
        "check_latest",
        _check_latest_double("0.59.0", "0.59.0", False),
    )
    monkeypatch.setattr(update, "_services_stage", lambda: ran.append("services"))
    monkeypatch.setattr(
        update, "refresh_daemons_after_upgrade", lambda: ran.append("daemons") or []
    )
    assert update.update_command(services=False) == 0
    assert ran == ["daemons"]


def test_the_services_stage_refuses_an_install_that_is_not_this_machines(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """serve-reload R3-2: the guard must ask ownership, not just install kind.

    Asking only "is this an installation at all" let a pip-installed `lop update`
    on a uv-tool machine reload the fleet that install owns. Harmless in
    destination — everything converges on the shared pointer — but not in
    authority, and a spurious reload cuts the app's relay for nothing.
    """
    import sys as sys_mod
    from pathlib import Path

    from local_operator import services, update
    from local_operator.update import InstallKind

    called: list[str] = []
    monkeypatch.setattr(update, "_services_refusal", _REAL_SERVICES_REFUSAL)
    monkeypatch.setattr(update, "install_kind", _install_kind_double(InstallKind.UV_TOOL))
    monkeypatch.setattr(update, "stable_root", lambda: Path("/nowhere/lop"))
    monkeypatch.setattr(sys_mod, "prefix", "/usr/local/lib/python3.12/site-packages")
    monkeypatch.setattr(services, "restart_services", lambda **k: called.append("ran"))
    update._services_stage()
    assert called == []
    assert "is not one of this machine's install generations" in capsys.readouterr().err


def test_update_command_moves_the_services_on_the_upgrade_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """serve-reload R4-1's regression: the caller that just flipped the pointer is SUPERSEDED.

    ``perform_upgrade`` installs into a new generation and flips the pointer **in
    this same process** — nothing re-execs — so the guard's first attempt, which
    compared ``sys.prefix`` with the generation the pointer names, refused the one
    caller that had actually performed the upgrade. `lop update` then moved the
    tree, moved no service, and reported success: the reported bug, restored one
    generation later.

    This drives `update_command` itself rather than `_services_stage`, because the
    module's autouse fixture hides the guard from `update_command` (serve-reload R4-3) — the
    path that broke has to be the path under test.
    """
    import sys as sys_mod

    from local_operator import update
    from local_operator.update import InstallKind

    generations = tmp_path / "lop" / "generations"
    superseded = generations / "20260101T000000Z-0.1.0" / "tools" / "local-operator"
    superseded.mkdir(parents=True)
    # The pointer has already moved on, and this process is still the old build.
    current = generations / "20260102T000000Z-0.2.0"
    current.mkdir(parents=True)
    # THE POINTER IS ACTUALLY CREATED, so the pre-fix failure is the one this
    # docstring narrates — "is not the install the pointer names" — rather than the
    # "the install pointer names no build" a missing symlink produces (serve-reload
    # serve-reload review round 5, R5-4). A regression test whose failure mode is a different
    # refusal is one that would keep passing if the real check were deleted.
    (tmp_path / "lop" / "current").symlink_to(current)

    ran: list[str] = []
    monkeypatch.setattr(update, "_services_refusal", _REAL_SERVICES_REFUSAL)
    monkeypatch.setattr(update, "install_kind", _install_kind_double(InstallKind.UV_TOOL))
    monkeypatch.setattr(update, "stable_root", lambda: tmp_path / "lop")
    monkeypatch.setattr(sys_mod, "prefix", str(superseded))
    monkeypatch.setattr(
        update,
        "check_latest",
        _check_latest_double("0.2.0", "0.2.0", False),
    )
    monkeypatch.setattr(
        "local_operator.services.restart_services", lambda **k: ran.append("ran") or []
    )
    assert update.update_command() == 0
    assert ran == ["ran"], "the stage refused the caller that performed the upgrade"


@pytest.mark.parametrize("kind", [InstallKind.EDITABLE, InstallKind.UNKNOWN])
def test_the_services_stage_refuses_a_checkout(
    kind: InstallKind, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """serve-reload R2-1's sentence: a worktree does not own this machine's services.

    Before the services stage existed this was unreachable by construction (a
    checkout that was behind hit `editable_refusal`; one that was not behind
    returned early). Wiring the stage to the "nothing to install" path is what
    opened it, and the consequence was measured in review — an editable caller
    classifies every daemon as stale and signals the serve fleet.
    """
    from local_operator import services, update

    called: list[str] = []
    monkeypatch.setattr(update, "_services_refusal", _REAL_SERVICES_REFUSAL)
    monkeypatch.setattr(update, "install_kind", _install_kind_double(kind))
    monkeypatch.setattr(services, "restart_services", lambda **k: called.append("ran"))
    update._services_stage()
    assert called == []
    assert "does not own this machine's services" in capsys.readouterr().err


def test_a_generation_of_this_install_may_proceed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not refuse the callers it exists for — either of them.

    Both shapes are real, and the second is the one serve-reload R4-1 was about:

    * steady state — `lop` invoked through `current`;
    * the superseded build — the process that has just performed the upgrade.

    Checked against the REAL install on this machine before this was written: the
    `current` generation's `lop` has `sys.prefix` inside `stable_root()/generations`.
    """
    import sys as sys_mod
    from pathlib import Path

    from local_operator import update
    from local_operator.update import InstallKind

    generations = tmp_path / "lop" / "generations"
    steady = generations / "20260102T000000Z-0.2.0" / "tools" / "local-operator"
    superseded = generations / "20260101T000000Z-0.1.0" / "tools" / "local-operator"
    steady.mkdir(parents=True)
    superseded.mkdir(parents=True)

    monkeypatch.setattr(update, "_services_refusal", _REAL_SERVICES_REFUSAL)
    monkeypatch.setattr(update, "install_kind", _install_kind_double(InstallKind.UV_TOOL))
    monkeypatch.setattr(update, "stable_root", lambda: tmp_path / "lop")
    monkeypatch.setattr(sys_mod, "prefix", str(steady))
    assert update._services_refusal() is None
    # The `generations` root itself is not an install, only what lives under it.
    assert update._services_refusal(prefix=Path(generations)) is not None
    monkeypatch.setattr(sys_mod, "prefix", str(superseded))
    assert update._services_refusal() is None


def test_main_dispatches_services_status(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """``lop services status`` prints what it finds and changes nothing."""
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "services", "status"])
    with patch("local_operator.services.status_lines", return_value=["line one"]):
        assert main() == 0
    assert capsys.readouterr().out.strip() == "line one"


def test_main_dispatches_services_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """``restart`` runs the fleet stage, and ``--wait`` reaches it."""
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "services", "restart"])
    with patch("local_operator.services.restart_services", return_value=[]) as restart:
        assert main() == 0
    restart.assert_called_once_with()

    monkeypatch.setattr("sys.argv", ["lop", "services", "restart", "--wait", "5"])
    with patch("local_operator.services.restart_services", return_value=[]) as waited:
        assert main() == 0
    waited.assert_called_once_with(wait_s=5.0)


def test_main_refuses_an_unknown_services_verb(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A bare ``lop services`` names its verbs instead of doing something.

    It returns 2 rather than raising through ``parser.error`` (design review D8):
    ``parser.error`` dumped the WHOLE program's usage here — every verb of ``lop``
    under a second ``usage:`` prefix — when the thing that was mistyped is a
    subcommand of this one group. 2 keeps a usage error distinct from the 1 the
    sibling ``install`` group returns for its own, which this deliberately mirrors.
    """
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "services"])
    assert main() == 2
    err = capsys.readouterr().err
    assert err.strip() == "usage: lop services {status, restart}"
    assert "{credential,config,agents" not in err, "the whole program's verb list"


@pytest.mark.parametrize(
    "argv,name",
    [
        (["lop", "install", "status"], "status"),
        (["lop", "install", "prune"], "prune"),
        (["lop", "install", "migrate"], "migrate"),
    ],
)
def test_main_dispatches_install_verbs(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], name: str
) -> None:
    """``lop install`` is wired to the layout's own commands.

    The verbs live in ``update`` and the dispatch lives in ``cli``, so a rename
    on either side leaves a subcommand that parses and then quietly does
    nothing — which for ``migrate`` means a machine that never adopts the layout.
    """
    from local_operator.cli import main

    seen: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "local_operator.update.install_status_command", lambda: seen.append(("status", None)) or 0
    )
    monkeypatch.setattr(
        "local_operator.update.install_prune_command",
        lambda **kwargs: seen.append(("prune", kwargs)) or 0,
    )
    monkeypatch.setattr(
        "local_operator.update.install_migrate_command",
        lambda: seen.append(("migrate", None)) or 0,
    )
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    assert [entry[0] for entry in seen] == [name]
    if name == "prune":
        # The retention policy is the DEFAULT, not a number repeated here: the
        # two must not be able to disagree about how many trees survive.
        assert seen[0][1] == {"keep": update_mod.DEFAULT_KEEP_GENERATIONS}


def test_main_refuses_a_non_numeric_keep_without_naming_a_python_symbol(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R7-2: ``--keep foo`` printed ``invalid _generation_count value: 'foo'``.

    That is argparse's own template for a ``ValueError`` escaping the ``type=``
    callable, and it renders the FUNCTION'S NAME — a Python symbol on a CLI
    surface the generation PR had just cleaned of exactly that (design review D5
    removed the neighbouring one, a quoted ``None`` in the help text). The exit
    code and the usage line are unchanged: this is a bad option, not a crash.
    """
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "install", "prune", "--keep", "foo"])
    with pytest.raises(SystemExit) as refused:
        main()
    assert refused.value.code == 2
    captured = capsys.readouterr()
    assert "_generation_count" not in captured.err, captured.err
    assert "expected a whole number, got 'foo'" in captured.err, captured.err
    assert "usage: lop install prune" in captured.err, captured.err
    assert captured.out == ""


def test_main_dispatches_install_prune_keep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.cli import main

    seen: list[dict[str, object]] = []
    monkeypatch.setattr(
        "local_operator.update.install_prune_command",
        lambda **kwargs: seen.append(kwargs) or 0,
    )
    monkeypatch.setattr("sys.argv", ["lop", "install", "prune", "--keep", "3"])
    assert main() == 0
    assert seen == [{"keep": 3}]


def test_main_refuses_an_install_verb_it_does_not_have(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.cli import main

    monkeypatch.setattr("sys.argv", ["lop", "install"])
    assert main() == 1
    assert "usage: lop install" in capsys.readouterr().err


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


def test_refresh_restarts_via_new_distribution(branded_image) -> None:
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
    # argv[0] is the role LABEL, so `executable=` must carry the interpreter:
    # a label with no image is a file the kernel would be asked to execute.
    assert run.call_args.args[0] == [
        procname.branded_argv0(procname.LABEL_MOBILE_RESTART),
        SAFE_PATH_FLAG,
        "-m",
        "local_operator.cli",
        "mobile",
        "restart",
    ]
    # The image travels BESIDE the label and is always a real file — never the
    # label itself, which the kernel would try to execute. It is the branded
    # hardlink wherever one can be planted, and the interpreter where it cannot.
    image = run.call_args.kwargs["executable"]
    assert image != run.call_args.args[0][0]
    assert os.path.basename(image) in {procname.BRAND, os.path.basename(sys.executable)}


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


class TestServiceDaemonRefresh:
    """The daemons ``lop-update`` used to leave behind, repaired by one child.

    The repair RENDERS a plist, so it cannot run in-process: the updater's own
    already-imported modules are the PREVIOUS build and would render the
    previous plist shape. Every assertion here is about that child and about
    what the summary says it did.
    """

    def test_nothing_installed_means_no_child_at_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty scan on a host the scan CAN address is silence.

        The platform is pinned to the launchd shape deliberately: since audit A24
        an empty scan on a host whose daemons are systemd units or scheduled
        tasks announces that it did not refresh them, which is the neighbouring
        case and has its own test. Without the pin this test would be asserting
        the macOS answer on a Linux runner, where the answer differs by design.
        """
        monkeypatch.setattr(update_mod, "_DAEMONS_ARE_LAUNCHD_AGENTS", True)
        with (
            patch.object(update_mod, "_installed_daemon_plists", return_value=[]),
            patch("subprocess.run") as run,
        ):
            refresh = update_mod.refresh_service_daemons_after_upgrade()
        assert refresh == update_mod.DaemonRefresh("service daemons")
        assert refresh.lines == () and refresh.warnings == ()
        run.assert_not_called()

    def test_a_platform_without_launchd_says_so_instead_of_saying_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silence reads as success, so a platform the scan cannot address speaks.

        The scan is ``~/Library/LaunchAgents``, so on Linux and Windows it finds
        nothing and the empty ``DaemonRefresh`` printed nothing at all — in the
        upgrade summary that is indistinguishable from "there was nothing to
        do", while a systemd unit or a scheduled task is still running the
        previous interpreter. Re-registering those units is a follow-up; what is
        pinned here is that the step reports what it did NOT do (audit A24).

        The scan is still CONSULTED (``return_value=[]`` is the real host's
        answer, not a bypass), because that is the function's own contract and
        the sibling macOS tests below patch the same seam to prove the child
        runs; what changes on this host is only the conclusion.
        """
        monkeypatch.setattr(update_mod, "_DAEMONS_ARE_LAUNCHD_AGENTS", False)
        with (
            patch.object(update_mod, "_installed_daemon_plists", return_value=[]),
            patch("subprocess.run") as run,
        ):
            refresh = update_mod.refresh_service_daemons_after_upgrade()

        run.assert_not_called()
        assert refresh.warnings == ()
        assert len(refresh.lines) == 1
        assert "not refreshed" in refresh.lines[0]
        assert sys.platform in refresh.lines[0], "the line names this host's platform"
        assert "installer" in refresh.lines[0], "and the way to fix it"
        # The upgrade summary is read by tests that keep each step's own lines
        # apart (``test_update_command_upgrades`` pins that a skipped mobile
        # refresh prints nothing), so this sentence must not name a daemon.
        assert "mobile" not in refresh.lines[0]

    def test_an_installed_agent_is_repaired_whatever_the_platform_says(self, branded_image) -> None:
        """The platform branch decides the empty-scan answer, not whether to work.

        Gating the scan on macOS made the function's contract untestable off it:
        a host that HAS an agent to repair must reach the child, and that is what
        the four tests below assert. This one pins the boundary between them.
        """
        plist = Path("/tmp/Library/LaunchAgents/com.local-operator.tunnel.plist")
        completed = subprocess.CompletedProcess([], 0, stdout="tunnel daemon: refreshed\n")
        with (
            patch.object(update_mod, "_DAEMONS_ARE_LAUNCHD_AGENTS", False),
            patch.object(update_mod, "_installed_daemon_plists", return_value=[plist]),
            patch("subprocess.run", return_value=completed) as run,
        ):
            refresh = update_mod.refresh_service_daemons_after_upgrade()

        run.assert_called_once()
        assert refresh.lines, "the repaired agent is reported"

    def test_the_child_is_the_new_wheel_and_is_named(self, branded_image) -> None:
        plist = Path("/tmp/Library/LaunchAgents/com.local-operator.tunnel.plist")
        completed = subprocess.CompletedProcess(
            [], 0, stdout="tunnel daemon: refreshed a stale LaunchAgent and restarted it\n"
        )
        with (
            patch.object(update_mod, "_installed_daemon_plists", return_value=[plist]),
            patch("subprocess.run", return_value=completed) as run,
        ):
            refresh = update_mod.refresh_service_daemons_after_upgrade()
        # `SAFE_PATH_FLAG` before `-m`, like the mobile bounce: this runs with no
        # `cwd=`, so a bare `-m` would load a checkout the update happened to be
        # started in. The label is argv[0], so the image travels beside it.
        assert run.call_args.args[0] == [
            procname.branded_argv0(procname.LABEL_DAEMONS_REFRESH),
            SAFE_PATH_FLAG,
            "-m",
            "local_operator.cli",
            "update",
            "--refresh-daemons",
        ]
        # The image travels BESIDE the label and is always a real file — never
        # the label itself, which the kernel would try to execute. It is the
        # branded hardlink wherever one can be planted, and the interpreter
        # where it cannot.
        image = run.call_args.kwargs["executable"]
        assert image != run.call_args.args[0][0]
        assert os.path.basename(image) in {procname.BRAND, os.path.basename(sys.executable)}
        assert refresh.lines == ("tunnel daemon: refreshed a stale LaunchAgent and restarted it",)
        assert refresh.warnings == ()

    def test_a_nonzero_child_is_a_warning_not_a_failure(self) -> None:
        plist = Path("/tmp/Library/LaunchAgents/com.local-operator.wakes.plist")
        completed = subprocess.CompletedProcess([], 3, stdout="", stderr="error: boom\n")
        with (
            patch.object(update_mod, "_installed_daemon_plists", return_value=[plist]),
            patch("subprocess.run", return_value=completed),
        ):
            refresh = update_mod.refresh_service_daemons_after_upgrade()
        assert refresh.warnings == ("warning: could not refresh installed daemons: error: boom",)
        assert refresh.lines == ()

    def test_a_timeout_is_a_warning(self) -> None:
        with (
            patch.object(update_mod, "_installed_daemon_plists", return_value=[Path("/tmp/x")]),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)),
        ):
            refresh = update_mod.refresh_service_daemons_after_upgrade()
        assert refresh.warnings == ("warning: daemon refresh timed out",)

    def test_the_services_run_before_the_mobile_bounce(self) -> None:
        """The order the plist repair makes load-bearing.

        The service child REWRITES plists; the mobile bounce must restart from a
        plist that is already current, or it restarts the previous definition and
        the second start is the only one on the new shape.
        """
        order: list[str] = []
        with (
            patch.object(
                update_mod,
                "refresh_service_daemons_after_upgrade",
                side_effect=lambda: order.append("services") or update_mod.DaemonRefresh("s"),
            ),
            patch.object(
                update_mod,
                "refresh_mobile_after_upgrade",
                side_effect=lambda: order.append("mobile")
                or update_mod.MobileRefresh(kind="restarted"),
            ),
        ):
            refreshes = update_mod.refresh_daemons_after_upgrade()
        assert order == ["services", "mobile"]
        assert [refresh.name for refresh in refreshes] == ["s", "mobile"]
        assert refreshes[1].lines == ("mobile daemon restarted — refresh the phone UI",)

    def test_only_the_installed_plists_are_probed(self, tmp_path, monkeypatch) -> None:
        """A pure filesystem probe, and one that a redirected HOME turns off."""
        directory = tmp_path / "Library" / "LaunchAgents"
        directory.mkdir(parents=True)
        (directory / "com.local-operator.tunnel.plist").write_bytes(b"")
        (directory / "com.local-operator.wakes.plist").write_bytes(b"")
        monkeypatch.setenv("HOME", str(tmp_path))
        found = update_mod._installed_daemon_plists()
        assert sorted(path.name for path in found) == [
            "com.local-operator.tunnel.plist",
            "com.local-operator.wakes.plist",
        ]

    def test_the_flag_bypasses_the_pypi_check(self, capsys) -> None:
        """``--refresh-daemons`` is a repair, not an upgrade: no network, no version."""
        with (
            patch.object(update_mod, "check_latest") as check,
            patch.object(update_mod, "daemons_refresh_command", return_value=0) as command,
        ):
            assert update_command(refresh_daemons=True) == 0
        check.assert_not_called()
        command.assert_called_once()
        assert capsys.readouterr().out == ""

    def test_the_child_refuses_to_rewrite_from_a_checkout(self, capsys) -> None:
        """The rule the upgrade path already enforces, at the point that WRITES.

        A source checkout's interpreter is not the daemon's, so a repair from
        one would point the operator's LaunchAgents at that checkout. The
        visible entry points cannot reach this (an editable install is refused
        before the refresh); the hidden flag can.
        """
        with (
            patch.object(update_mod, "install_kind", return_value=InstallKind.EDITABLE),
            patch.object(update_mod, "installer_argv", side_effect=AssertionError("must not run")),
        ):
            assert update_mod.daemons_refresh_command() == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "source checkout" in captured.err

    def test_the_child_refuses_to_rewrite_a_foreign_non_editable_install(self, capsys) -> None:
        """A hand-made venv may not repoint the operator's daemons at itself.

        THE INVARIANT: a repair may change how a daemon is NAMED, never WHICH
        INSTALL it runs. A durable install (a uv tool, pipx) IS the interpreter
        the daemons should run and may repair what it owns; a bare pip install
        in some other venv must not, because that venv can be deleted while the
        plists keep pointing at it. The comparison is by PREFIX, so the legacy
        ``<prefix>/bin/python3`` plist and the branded
        ``<prefix>/bin/Local Operator`` one count as the same install.
        """
        from local_operator import launchd

        plists = [Path("/Users/x/Library/LaunchAgents/com.local-operator.mobile.plist")]
        with (
            patch.object(update_mod, "install_kind", return_value=InstallKind.PIP),
            patch.object(update_mod, "_installed_daemon_plists", return_value=plists),
            patch.object(
                launchd,
                "load",
                return_value={"Program": "/opt/other-venv/bin/Local Operator"},
            ),
            patch.object(update_mod, "installer_argv", side_effect=AssertionError("must not run")),
        ):
            assert update_mod.daemons_refresh_command() == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "another installation" in captured.err, captured.err
        assert "/opt/other-venv" in captured.err, captured.err

    def test_the_child_repairs_a_pip_install_of_its_own_prefix(self, capsys) -> None:
        """The same guard, on the machine it must NOT block.

        Both plist shapes are exercised: the legacy ``ProgramArguments[0]``
        interpreter and the branded ``Program``. A venv pip-installed from PyPI
        whose prefix is the one the daemons already record is repairing itself,
        which is exactly what the guard must allow.
        """
        from local_operator import launchd
        from local_operator.browser_bridge import install as browser_install
        from local_operator.mobile import install as mobile_install
        from local_operator.tunnels import install as tunnel_install
        from local_operator.wakes import install as wakes_install

        plists = [
            Path("/Users/x/Library/LaunchAgents/com.local-operator.mobile.plist"),
            Path("/Users/x/Library/LaunchAgents/com.local-operator.tunnel.plist"),
        ]
        loaded = [
            {
                "ProgramArguments": [
                    f"{sys.prefix}/bin/python3",
                    "-m",
                    "local_operator.mobile.service",
                ]
            },
            {"Program": f"{sys.prefix}/bin/Local Operator"},
        ]
        with (
            patch.object(update_mod, "install_kind", return_value=InstallKind.PIP),
            patch.object(update_mod, "_installed_daemon_plists", return_value=plists),
            patch.object(launchd, "load", side_effect=loaded),
            ExitStack() as stack,
        ):
            for module in (mobile_install, browser_install, tunnel_install, wakes_install):
                stack.enter_context(
                    patch.object(
                        module,
                        "refresh_plist_if_stale",
                        return_value=launchd.PlistRefresh("d", "current"),
                    )
                )
            assert update_mod.daemons_refresh_command() == 0
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_the_child_repairs_every_daemon_and_reports_each(self, capsys) -> None:
        """One line per daemon that CHANGED; silence for one already current."""
        from local_operator import launchd
        from local_operator.browser_bridge import install as browser_install
        from local_operator.mobile import install as mobile_install
        from local_operator.tunnels import install as tunnel_install
        from local_operator.wakes import install as wakes_install

        outcomes = (
            (mobile_install, launchd.PlistRefresh("mobile", "repaired")),
            (browser_install, launchd.PlistRefresh("browser bridge", "current")),
            (tunnel_install, launchd.PlistRefresh("tunnel", "failed", "boom")),
            (wakes_install, launchd.PlistRefresh("wakes supervisor", "repaired")),
        )
        with (
            patch.object(update_mod, "install_kind", return_value=InstallKind.UV_TOOL),
            ExitStack() as stack,
        ):
            for module, outcome in outcomes:
                stack.enter_context(
                    patch.object(module, "refresh_plist_if_stale", return_value=outcome)
                )
            assert update_mod.daemons_refresh_command() == 0
        captured = capsys.readouterr()
        assert captured.out.splitlines() == [
            "mobile daemon: refreshed a stale LaunchAgent and restarted it",
            "wakes supervisor daemon: refreshed a stale LaunchAgent and restarted it",
        ]
        assert captured.err.splitlines() == ["warning: tunnel daemon was not refreshed: boom"]


def test_update_command_no_plist_prints_only_install_lines(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step under test is the MOBILE one, so the daemons step is pinned quiet.

    Since audit A24 a host whose supervised daemons are not launchd agents adds
    its own "not refreshed" line to the same summary, which would make this
    assertion about an unrelated step. Pinning the daemons branch keeps the exit
    code, the child invocation and the exact output all decided by the mobile
    refresh, which is what the test is named for.
    """
    monkeypatch.setattr(update_mod, "_DAEMONS_ARE_LAUNCHD_AGENTS", True)
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


def test_update_command_restarted_prints_phone_line(
    capsys: pytest.CaptureFixture[str], branded_image
) -> None:
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
        procname.branded_argv0(procname.LABEL_MOBILE_RESTART),
        SAFE_PATH_FLAG,
        "-m",
        "local_operator.cli",
        "mobile",
        "restart",
    ]
    # The image travels BESIDE the label and is always a real file — never the
    # label itself, which the kernel would try to execute. It is the branded
    # hardlink wherever one can be planted, and the interpreter where it cannot.
    image = run.call_args.kwargs["executable"]
    assert image != run.call_args.args[0][0]
    assert os.path.basename(image) in {procname.BRAND, os.path.basename(sys.executable)}
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


def test_perform_upgrade_does_not_refresh(tmp_path: Path) -> None:
    with (
        patch.object(update_mod, "refresh_mobile_after_upgrade") as refresh,
        patch("subprocess.run") as run,
    ):
        out = perform_upgrade(
            target="0.28.0", kind=InstallKind.UV_TOOL, run=lambda _: 0, prefix=tmp_path
        )
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


def test_source_ref_ignores_the_pypi_sentinel(tmp_path: Path) -> None:
    """``pypi <version>`` is an honest "no commit", not a ref to render.

    A PyPI upgrade has no git commit at all. The marker still records what is
    installed, so the file exists — but ``source_ref`` must not hand a caller
    the word ``pypi`` to print as ``0.51.9@pypi``.
    """
    (tmp_path / ".lop-source").write_text("pypi 0.51.9\n", encoding="utf-8")
    assert update_mod.source_ref(tmp_path) == ""
    assert update_mod.installed_build(tmp_path).source_ref == ""


def test_is_git_snapshot_follows_the_token_not_the_files_existence(tmp_path: Path) -> None:
    """The regression this whole change is about, stated as a predicate.

    Before ``/update`` wrote the marker, existence WAS the test — correct only
    while ``lop-update`` was the sole writer. Now a wheel installed over a git
    snapshot leaves a marker behind, and calling that host a git snapshot
    would keep printing "this runtime was built from git" about a wheel.
    """
    marker = tmp_path / ".lop-source"

    marker.write_text("4d3ce1d1a48f4f3b799efdfabb014979e70e0630 main\n", encoding="utf-8")
    assert update_mod.is_git_snapshot(tmp_path) is True

    marker.write_text("pypi 0.51.9\n", encoding="utf-8")
    assert update_mod.is_git_snapshot(tmp_path) is False


def test_write_source_marker_records_a_pypi_install(tmp_path: Path) -> None:
    assert update_mod.write_source_marker(tmp_path, version="0.51.9") is True
    assert (tmp_path / ".lop-source").read_text(encoding="utf-8") == "pypi 0.51.9\n"
    assert update_mod.source_ref(tmp_path) == ""
    assert update_mod.is_git_snapshot(tmp_path) is False


def test_write_source_marker_keeps_lop_updates_two_token_shape(tmp_path: Path) -> None:
    """``lop-update`` is a second writer, out of this tree; the format is the contract.

    It writes ``printf '%s %s\\n' "$COMMIT" "$REF"``. Writing the same shape
    here is what keeps the two interchangeable — a marker written by either
    reads identically through ``source_ref``.
    """
    sha = "4d3ce1d1a48f4f3b799efdfabb014979e70e0630"
    assert update_mod.write_source_marker(tmp_path, version="0.51.9", commit=sha, ref="main")
    assert (tmp_path / ".lop-source").read_text(encoding="utf-8") == f"{sha} main\n"
    assert update_mod.source_ref(tmp_path) == sha
    assert update_mod.is_git_snapshot(tmp_path) is True


def test_write_source_marker_replaces_a_stale_marker(tmp_path: Path) -> None:
    """The reported state: a marker naming the build the upgrade DISPLACED."""
    stale = "f1cd77900182616a683c4e7e58f0b0e01be580b3"
    (tmp_path / ".lop-source").write_text(f"{stale} main\n", encoding="utf-8")
    assert update_mod.source_ref(tmp_path) == stale

    update_mod.write_source_marker(tmp_path, version="0.51.9")
    assert update_mod.source_ref(tmp_path) == ""
    assert (tmp_path / ".lop-source").read_text(encoding="utf-8") == "pypi 0.51.9\n"


def test_write_source_marker_never_raises_on_an_unwritable_root(tmp_path: Path) -> None:
    """A failed marker write must not turn a SUCCESSFUL upgrade into an error."""
    missing = tmp_path / "does-not-exist"
    assert update_mod.write_source_marker(missing, version="0.51.9") is False


def test_write_source_marker_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    """Temp-and-rename: a reader must never see a partial marker.

    ``RuntimeServer.__init__`` reads this file, so a torn write would reach
    every runtime on the host (review round 1, R1-2).

    NOT SUFFICIENT ON ITS OWN. A plain ``path.write_text`` leaves no temp file
    either, so this assertion is satisfied by the very implementation it reads
    as rejecting; it passed against both non-atomic mutants in review round 1.
    The two tests below are the ones that discriminate — this one only pins
    that the temp file is cleaned up.
    """
    update_mod.write_source_marker(tmp_path, version="0.51.9")
    assert sorted(p.name for p in tmp_path.iterdir()) == [".lop-source"]


def test_write_source_marker_installs_by_rename_not_by_writing_in_place(
    tmp_path: Path,
) -> None:
    """The destination is only ever reached by a rename, never opened for write.

    THIS IS THE ATOMICITY GUARD, and it is structural rather than timed: a
    rename REPLACES the destination, so the inode a reader would open changes;
    truncating the destination in place (``path.write_text``, or a
    ``shutil.copyfile`` over it) keeps the same inode and exposes a window in
    which a concurrent ``RuntimeServer.__init__`` reads a half-written marker.
    Inode identity is a fact about how the file got there, so this cannot flake
    the way a race-the-writer test would.

    Verified to discriminate (review round 1, R1-1): both the ``write_text``
    and ``copyfile`` mutants keep the inode and fail here.
    """
    marker = tmp_path / ".lop-source"
    marker.write_text("f1cd77900182 main\n", encoding="utf-8")
    before = marker.stat().st_ino

    assert update_mod.write_source_marker(tmp_path, version="0.51.9") is True

    after = marker.stat().st_ino
    assert after != before, (
        "the marker must be installed by renaming a fully-written temp file over it; "
        f"the inode was unchanged ({before}), so the destination was written in place"
    )


def test_a_failed_marker_write_leaves_the_previous_marker_byte_intact(
    tmp_path: Path,
) -> None:
    """A write that cannot start must not damage what is already there.

    The companion to the rename guard above, covering the failure path: with
    the temp file unavailable there is nothing to rename, so the function has
    to report ``False`` and leave the existing marker exactly as it found it.
    An implementation that writes the destination directly reports success and
    overwrites a marker it never managed to replace — which is worse than not
    writing at all, because every runtime on the host then reads it.
    """
    marker = tmp_path / ".lop-source"
    stale = "f1cd77900182 main\n"
    marker.write_text(stale, encoding="utf-8")

    with patch.object(update_mod.tempfile, "mkstemp", side_effect=OSError("no temp")):
        assert update_mod.write_source_marker(tmp_path, version="0.51.9") is False

    assert marker.read_text(encoding="utf-8") == stale


def test_upgrading_a_uv_tool_records_what_it_just_installed(tmp_path: Path) -> None:
    """End to end over ``perform_upgrade``: the marker follows the payload.

    Nothing in Python wrote ``.lop-source`` before this, so an upgrade driven
    from ``lop update`` or the TUI's ``/update`` left it naming the displaced
    build for as long as the host lived.
    """
    stale = "f1cd77900182616a683c4e7e58f0b0e01be580b3"
    (tmp_path / ".lop-source").write_text(f"{stale} main\n", encoding="utf-8")

    perform_upgrade(target="0.51.9", kind=InstallKind.UV_TOOL, run=lambda _: 0, prefix=tmp_path)

    assert (tmp_path / ".lop-source").read_text(encoding="utf-8") == "pypi 0.51.9\n"
    assert update_mod.source_ref(tmp_path) == "", "no commit may be invented for a wheel"


def test_a_failed_upgrade_leaves_the_marker_alone(tmp_path: Path) -> None:
    """The marker describes what is INSTALLED, and a failed install changed nothing."""
    stale = "f1cd77900182616a683c4e7e58f0b0e01be580b3"
    (tmp_path / ".lop-source").write_text(f"{stale} main\n", encoding="utf-8")

    with pytest.raises(UpdateError):
        perform_upgrade(target="0.51.9", kind=InstallKind.UV_TOOL, run=lambda _: 1, prefix=tmp_path)

    assert update_mod.source_ref(tmp_path) == stale


def test_only_the_uv_tool_layout_gets_a_marker(tmp_path: Path) -> None:
    """pipx and pip installs never had a marker and gain nothing from one."""
    perform_upgrade(target="0.51.9", kind=InstallKind.PIPX, run=lambda _: 0, prefix=tmp_path)
    assert not (tmp_path / ".lop-source").exists()


def test_build_marker_age_reads_the_newest_write_not_the_marker(tmp_path: Path) -> None:
    """The settle guard must not be disarmed by a marker older than the payload.

    A stale marker reported a minutes-old install as ~19000 s old on the
    reporting host. Taking the most recent of marker and dist-info can only
    make the age smaller, and a smaller age makes the guard wait longer —
    the safe direction.
    """
    marker = tmp_path / ".lop-source"
    marker.write_text("pypi 0.51.9\n", encoding="utf-8")
    old = time.time() - 19_000
    os.utime(marker, (old, old))

    dist_dir = tmp_path / "local_operator-0.51.9.dist-info"
    dist_dir.mkdir()

    class _Located:
        _path = dist_dir

    with patch.object(update_mod, "distribution", return_value=_Located()):
        age = update_mod.build_marker_age_s(tmp_path)

    assert age is not None
    assert age < 60, f"the fresh dist-info must win over the stale marker, got {age}"


def test_build_marker_age_is_none_without_either_signal(tmp_path: Path) -> None:
    with patch.object(update_mod, "distribution", side_effect=PackageNotFoundError()):
        assert update_mod.build_marker_age_s(tmp_path) is None


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
