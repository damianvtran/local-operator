"""The CLI forwards explicit peer repository identity without resolving local cwd."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import cli


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lop")
    cli.add_parser(parser.add_subparsers(dest="subcommand"))
    return parser


def test_ready_forwards_task_repository_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _parser().parse_args(["network", "ready", "--peer", "synthetic-peer", "--json"])
    args.repository_path = "/srv/synthetic-task"
    args.repository_url = "https://example.invalid/synthetic/task.git"
    calls: list[tuple[str, dict[str, Any]]] = []

    def relay(op: str, **fields: Any) -> dict[str, Any]:
        calls.append((op, fields))
        return {"checks": []}

    monkeypatch.setattr(cli, "_relay_call", relay)
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert calls[0][0] == "peer_readiness"
    assert calls[0][1]["repository"] == {
        "path": "/srv/synthetic-task",
        "url": "https://example.invalid/synthetic/task.git",
    }


def test_ready_parses_explicit_peer_path_and_clone_url() -> None:
    args = _parser().parse_args(
        [
            "network",
            "ready",
            "--repository-path",
            "~/synthetic-task",
            "--repository-url",
            "https://example.invalid/synthetic/task.git",
            "--json",
        ]
    )
    # Expansion belongs to the peer's existing probe, never this CLI's HOME/cwd.
    assert args.repository_path == "~/synthetic-task"
    assert args.repository_url == "https://example.invalid/synthetic/task.git"


def test_ready_omits_repository_without_explicit_task_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(cli, "_relay_call", lambda op, **kw: captured.append(kw) or {"checks": []})
    assert cli.main(_parser().parse_args(["network", "ready", "--json"])) == 0
    assert "repository" not in captured[0]


def test_ready_offline_fallback_keeps_explicit_arguments_without_local_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _parser().parse_args(["network", "ready", "--repository-path", "~/peer-only", "--json"])
    monkeypatch.setattr(cli, "_relay_call", lambda *a, **kw: None)
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "_ready_locally", lambda arg: seen.append(arg) or {"checks": []})
    assert cli.main(args) == 0
    assert seen == [args] and seen[0].repository_path == "~/peer-only"


def test_real_cli_control_receives_context_and_surfaces_probe_and_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator import supervisors
    from local_operator.network import identity, readiness, relay
    from local_operator.network.types import MeshRefusal

    root = tmp_path / "viewer-config"
    root.mkdir()
    home = tmp_path / "viewer-home"
    home.mkdir()
    peer_home = tmp_path / "synthetic-peer-home"
    peer_home.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(
        relay, "supervision_state", lambda: {"platform": "none", "persistent": False}
    )
    monkeypatch.setattr(relay, "_port_holder_pids", lambda port: [])

    def forbidden(*a: Any, **kw: Any) -> None:
        pytest.fail("readiness CLI test attempted host service control")

    monkeypatch.setattr(relay, "_launchctl", forbidden)
    monkeypatch.setattr(supervisors, "systemctl_user", forbidden)
    monkeypatch.setattr(supervisors, "schtasks", forbidden)
    identity.mint(root, name="synthetic-viewer")
    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    frames: list[dict[str, Any]] = []

    def answer(frame: dict[str, Any]) -> dict[str, Any]:
        frames.append(frame)
        if frame.get("peer") == "synthetic-denied":
            # Inject an existing refusal at the real control boundary. This
            # proves CLI propagation, not a new peer authorization policy.
            raise MeshRefusal("not_authorised", "synthetic peer readiness role denied")
        fact = readiness.repository_fact(peer_home, repository=frame.get("repository"))
        row = readiness.repository_row(
            SimpleNamespace(device_id="synthetic-peer"),
            {"repository": fact},
            peer_label="synthetic-peer",
        )
        return {"checks": [row]}

    server.register_ops({}, {"peer_readiness": answer}, replace=True)
    server.start()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CMUX_", "LOP_")) and k != "XPC_FLAGS"
    }
    env.update(HOME=str(home), LOCAL_OPERATOR_CONFIG_DIR=str(root))
    url = "https://example.invalid/synthetic/task.git"

    def run(peer: str, path: str, *, full_process: bool = False) -> tuple[int, dict[str, Any]]:
        argv = [
            "network",
            "ready",
            "--peer",
            peer,
            "--repository-path",
            path,
            "--repository-url",
            url,
            "--json",
        ]
        if full_process:
            result = subprocess.run(
                [sys.executable, "-m", "local_operator.cli", *argv],
                cwd=home,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            rc, stdout, stderr = result.returncode, result.stdout, result.stderr
        else:
            # The remaining state cells use the shipped network parser/entry
            # and the same REAL control socket, without four redundant cold
            # imports of the entire product beside the fleet's CI workers.
            capsys.readouterr()
            rc = cli.main(_parser().parse_args(argv))
            captured = capsys.readouterr()
            stdout, stderr = captured.out, captured.err
        body = json.loads(stdout)
        with capsys.disabled():
            print(f"ready CLI: subprocess={full_process} peer={peer} rc={rc} body={body}")
        assert "synthetic-never-print" not in stdout + stderr
        return rc, body

    try:
        rc, missing = run("synthetic-peer", "~/missing-task", full_process=True)
        assert rc == 0  # repository equipment warnings do not gate onboarding
        row = missing["checks"][0]
        assert row["ok"] is False and row["observed"]["state"] == "missing"
        assert row["observed"]["path"] == str(peer_home / "missing-task")
        assert frames[-1]["repository"] == {"path": "~/missing-task", "url": url}
        assert not (home / "missing-task").exists()

        plain = peer_home / "plain-task"
        plain.mkdir()
        rc, invalid = run("synthetic-peer", "plain-task")
        assert invalid["checks"][0]["observed"]["state"] == "not_git_checkout"
        assert invalid["checks"][0]["ok"] is False
        assert frames[-1]["repository"] == {"path": "plain-task", "url": url}

        checkout = peer_home / "checkout-task"
        subprocess.run(
            ["git", "init", "--quiet", str(checkout)],
            cwd=home,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        rc, valid = run("synthetic-peer", "checkout-task")
        assert rc == 0 and valid["checks"][0]["ok"] is True
        assert valid["checks"][0]["observed"]["state"] == "checkout"
        assert valid["checks"][0]["observed"]["path"] == str(checkout)

        rc, url_only = run("synthetic-peer", "")
        assert url_only["checks"][0]["observed"]["path"] == str(peer_home / "task")
        assert frames[-1]["repository"] == {"path": "", "url": url}

        rc, refusal = run("synthetic-denied", "~/synthetic-never-print")
        assert rc == 1 and refusal["code"] == "not_authorised"
        assert refusal["message"] == "synthetic peer readiness role denied"
    finally:
        server.stop()
