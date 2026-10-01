"""The relay's Linux supervision arm — systemd ``--user`` (remote onboarding slice (b)).

WHY THIS FILE EXISTS SEPARATELY. ``test_purge`` pins the LAUNCHD half of the
supervision surface plus the no-supervisor refusal; these cells pin the half
that did not exist until slice (b): the unit render, the write/enable path (dry
run and file half only — a test never reaches a user manager), and the
missing-unit start that installs. ``sys.platform`` and ``shutil.which`` are
patched to the Linux answer rather than stubbing ``is_supported`` itself, so
the real expression runs (the same discipline ``test_purge`` documents).

NO CELL HERE MAY REACH ``systemctl``: every one that could is either a dry run
or has ``systemctl_user`` replaced by something that raises when called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import supervisors
from local_operator.network import relay

# tmp_path fixtures -------------------------------------------------------


@pytest.fixture()
def linux_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A HOME the unit path resolves under, claiming linux + systemctl."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(relay.sys, "platform", "linux")
    monkeypatch.setattr(
        relay.shutil,
        "which",
        lambda name: "/usr/bin/systemctl" if name == "systemctl" else f"/usr/bin/{name}",
    )
    return home


def _unit(home: Path) -> Path:
    return home / ".config" / "systemd" / "user" / relay.SYSTEMD_UNIT


def _no_systemctl(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: str, **kwargs: Any) -> Any:
        raise AssertionError(f"systemctl was invoked: {args}")

    monkeypatch.setattr(supervisors, "systemctl_user", _boom)


# tmp_path render ---------------------------------------------------------


def test_render_systemd_has_the_contract_every_unit_shares(tmp_path: Path) -> None:
    text = relay.render_systemd(4097)
    assert text.startswith("[Unit]\n")
    assert "Description=Local Operator network relay" in text
    assert "-m local_operator.network.relay --port 4097" in text
    # The config-dir pin is present and QUOTED (an unquoted assignment truncates
    # at the first space — the mobile daemon's own measured lesson).
    assert 'Environment="LOCAL_OPERATOR_CONFIG_DIR=' in text
    assert "Restart=on-failure" in text
    assert "WantedBy=default.target" in text


def test_install_dry_run_names_the_unit_and_touches_nothing(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run REPORTS what it would write; it writes nothing and reaches no manager.

    The same contract ``_install_launchd``'s dry run has: the receipt names the
    unit path so an operator can see where it would land, and the file itself is
    the non-dry-run half — which the redirected-home cell below exercises (the
    file half is testable; the enable half is what refuses there).
    """
    _no_systemctl(monkeypatch)

    result = relay.install(4097, dry_run=True)

    assert result["ok"] is True, result
    assert not _unit(linux_host).exists(), "a dry run must not write the unit"
    assert any(str(_unit(linux_host)) in step for step in result["steps"])
    assert any("dry run" in step for step in result["steps"])


def test_a_redirected_home_never_reaches_the_user_manager(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The addressability guard, asserted on the REAL expression.

    The test's own HOME is redirected and therefore not the passwd home, so
    ``_install_systemd`` must refuse AFTER writing the file (the file half is
    testable) and BEFORE any enable — with ``systemctl_user`` armed to fail.
    """
    _no_systemctl(monkeypatch)

    result = relay.install(4097)

    assert result["ok"] is False
    assert result["reason"] == "isolated_home"
    assert _unit(linux_host).exists(), "the file half is still written"


def test_service_action_start_installs_a_missing_unit(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``join`` never installed a unit, so start installs one — via ``_install_systemd``."""
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    calls: list[int] = []

    def _fake_install(port: int, *, dry_run: bool = False) -> dict[str, Any]:
        calls.append(port)
        return {"ok": True, "steps": ["installed"]}

    monkeypatch.setattr(relay, "_install_systemd", _fake_install)
    _no_systemctl(monkeypatch)

    result = relay.service_action("start")

    assert result["ok"] is True, result
    assert calls, "_install_systemd must be what starts a relay that has no unit"


def test_service_action_drives_systemctl_when_a_unit_is_there(
    linux_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    unit = _unit(linux_host)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text("(unit)\n", encoding="utf-8")
    seen: list[tuple[str, ...]] = []

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def _record(*args: str, **kwargs: Any) -> Any:
        seen.append(tuple(args))
        return Done()

    monkeypatch.setattr(supervisors, "systemctl_user", _record)

    assert relay.service_action("restart")["ok"] is True
    assert seen == [("restart", relay.SYSTEMD_UNIT)]
