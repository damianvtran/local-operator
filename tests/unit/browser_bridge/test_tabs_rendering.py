"""`lop browser tabs` must never render an UNKNOWN browser as an empty one.

The live-tab merge answered a blocker (a command called `tabs` reporting no
tabs while `status` drove seven), and shipped with no guard at all: nothing in
the tree exercised this rendering, so a one-line divergence from `status`
passed CI. These tests pin the distinction that divergence erased.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli
from local_operator.browser_bridge import install as browser_install
from local_operator.browser_bridge import state as browser_state


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    *,
    health: dict[str, Any] | None,
    discovery: object | None = None,
    json_mode: bool = False,
) -> str:
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(browser_state, "read", lambda: discovery)
    monkeypatch.setattr(browser_install, "health", lambda port, **kw: health)
    argv = ["lop", "browser", "tabs"] + (["--json"] if json_mode else [])
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    return capsys.readouterr().out


def test_a_missing_discovery_file_still_finds_the_daemons_tabs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """D9/U9: `state.read()` returns None for a CORRUPT file as much as no daemon.

    Skipping the probe on None made a healthy bridge driving five tabs render
    as an empty browser, while `status` found them at the default port in the
    same state.
    """
    tabs = [{"url": "http://127.0.0.1:4199/", "title": ""}]
    out = _run(
        monkeypatch,
        capsys,
        tmp_path,
        health={"extension_connected": True, "driven_tabs": tabs},
        discovery=None,
    )
    assert "http://127.0.0.1:4199/" in out
    assert "No browser tabs" not in out


def test_an_unreachable_bridge_is_not_reported_as_zero_tabs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A probe that did not answer must read as unknown, and name the diagnosis."""
    out = _run(monkeypatch, capsys, tmp_path, health=None, discovery=None)
    assert "No browser tabs" not in out
    assert "unknown" in out
    assert "lop browser status" in out


def test_a_live_bridge_with_no_tabs_still_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The honest zero must stay distinguishable from the unknown."""
    out = _run(
        monkeypatch,
        capsys,
        tmp_path,
        health={"extension_connected": True, "driven_tabs": []},
        discovery=None,
    )
    assert "No browser tabs and no ownership records." in out


def test_json_reports_unknown_as_null_never_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A script must not be able to read an unreachable bridge as an empty one."""
    unknown = json.loads(
        _run(monkeypatch, capsys, tmp_path, health=None, discovery=None, json_mode=True)
    )
    assert unknown["live_tabs_known"] is False
    assert unknown["live_tab_count"] is None

    live = json.loads(
        _run(
            monkeypatch,
            capsys,
            tmp_path,
            health={"extension_connected": True, "driven_tabs": []},
            discovery=None,
            json_mode=True,
        )
    )
    assert live["live_tabs_known"] is True
    assert live["live_tab_count"] == 0
