"""Untrusted daemon PIDs stay inside the process-control range before rendering."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import supervisors
from local_operator.network import cli, readiness, relay, store
from tests.unit.network import test_readiness as fixtures


@pytest.fixture()
def guarded_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    attempted: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        attempted.append("host control")
        raise AssertionError("PID-readiness test attempted host control")

    # A class preserves Popen's import-time generic interface. No pipe is ever
    # created: all process launches, including text-mode ones, are refused.
    class ForbiddenPopen(subprocess.Popen[bytes]):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            attempted.append("Popen")
            raise AssertionError("PID-readiness test attempted a process launch")

    monkeypatch.setattr(subprocess, "Popen", ForbiddenPopen)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "killpg", forbidden)
    monkeypatch.setattr(supervisors, "systemctl_user", forbidden)
    monkeypatch.setattr(supervisors, "schtasks", forbidden)
    monkeypatch.setattr(relay, "_launchctl", forbidden)
    # Negative controls prove both signal and service interlocks before any
    # readiness path executes; they never reach an OS signal/service API.
    for call in (
        lambda: os.kill(424242, 0),
        lambda: supervisors.systemctl_user("status", "synthetic"),
    ):
        with pytest.raises(AssertionError, match="attempted host control"):
            call()
    assert attempted == ["host control", "host control"]
    attempted.clear()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / "config"))
    monkeypatch.chdir(home)
    # Existing fake viewer avoids credential reads. No network is stored, no
    # peer is dialled, and the relay's unrelated generation probe is isolated.
    monkeypatch.setattr(readiness, "ViewerFacts", lambda *a, **k: fixtures._FakeViewer())
    monkeypatch.setattr(store, "list_networks", lambda root: [])
    monkeypatch.setattr(relay, "generation_reading", lambda pid: {})
    monkeypatch.setattr(relay, "generation_clause", lambda reading: "")
    yield
    assert attempted == [], "even a swallowed host-control attempt must fail the test"
    print("negative controls refused; signal/service/launch attempts after controls: []")


@pytest.mark.parametrize(
    ("pid", "valid"),
    [
        pytest.param(0, False, id="zero-group-selector"),
        pytest.param(-1, False, id="negative-broadcast-selector"),
        pytest.param(2**31, False, id="outside-signed-pid-t"),
        pytest.param(10**3999, False, id="4000-digit-peer-integer"),
        pytest.param(True, False, id="bool-not-a-pid"),
        pytest.param(None, False, id="missing-pid"),
        pytest.param("123", False, id="numeric-string-not-a-pid"),
        pytest.param(1, True, id="positive-lower-bound"),
        pytest.param(2**31 - 1, True, id="signed-upper-bound"),
    ],
)
def test_ready_cli_validates_daemon_pid_before_presenting_it(
    guarded_cli: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pid: Any,
    valid: bool,
) -> None:
    fact = fixtures._serve_fact()
    fact["daemons"][0]["pid"] = pid
    # JSON round-trip is the peer answer's value grammar, not a mocked bound.
    fact = json.loads(json.dumps(fact))
    # Also exercise the peer-source entry explicitly; compose below covers the
    # CLI's local-report route through the same production decoder.
    peer_row = readiness.serve_daemon_row(
        fixtures._member(name="synthetic-peer"),
        {"serve_daemons": fact},
        peer_label="synthetic-peer",
    )
    assert peer_row["ok"] is valid
    if not valid:
        assert peer_row["code"] == readiness.CODE_UNKNOWN
    monkeypatch.setattr(readiness, "serve_daemons_fact", lambda root: fact)
    # This no-network compose route needs only the synthetic context fields;
    # no RelayServer lifecycle or host listener is constructed.
    server: Any = SimpleNamespace(
        root=tmp_path / "home" / "config", identity=SimpleNamespace(device_id=fixtures.VIEWER)
    )
    calls: list[str] = []

    def control(op: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(op)
        # Exercise production compose -> local_serve_daemon_rows -> the SAME
        # serve_daemon_row decoder used for peer facts, then the real presenter.
        return readiness.compose(server)

    monkeypatch.setattr(cli, "_relay_call", control)
    parser = argparse.ArgumentParser(prog="lop")
    cli.add_parser(parser.add_subparsers(dest="command"), help_locale="en")
    assert cli.main(parser.parse_args(["network", "ready", "--json"])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls == ["peer_readiness"]
    [row] = payload["checks"]
    assert row["capability"] == readiness.CAPABILITY_SERVE_DAEMON
    if valid:
        assert row["ok"] is True and "code" not in row
        assert row["observed"]["daemons"][0]["pid"] == pid
    else:
        assert row["ok"] is False and row["code"] == readiness.CODE_UNKNOWN
        assert "malformed" in row["detail"]
        assert row["state"] == readiness.ROW_STATE_WARN
        assert len(row["detail"]) < 1000
        assert not row.get("observed", {}).get("daemons")
    # Serve-daemon equipment remains advisory. Reject the field, not admission;
    # neither an invalid nor a valid synthetic PID authorizes a signal here.
    assert payload["ok"] is True
    print(
        f"ready CLI: valid={valid}; row_ok={row['ok']}; "
        f"state={row['state']}; detail_chars={len(row['detail'])}"
    )
