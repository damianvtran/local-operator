"""The destination's reachability / prerequisite probe (docs/design/move-working-set.md §11.1).

Live finding (Mac -> node2, 2026-11-18 17:12Z): every worktree said
``clone_check: "pending_destination"``, the user confirmed, and node2 could neither
fetch the SSH remote nor push the HTTPS one. The preflight now asks the DESTINATION,
before the confirmation, and each cell pins one answer with REAL git:

* ``ok`` — the destination can ``ls-remote`` and ``push --dry-run``;
* ``destination_cannot_fetch`` — ``ls-remote`` fails;
* ``destination_cannot_push`` — fetch works, the receive-pack advertisement is
  refused (a fake ssh that serves upload-pack and denies receive-pack, the exact
  shape GitHub's SSH answer has);
* SSH unreachable but HTTPS works — ``https_rewrite`` is OFFERED (applied in PR B);
* hostile URLs (``ext::``, ``file://``, local paths, option-shaped) are refused
  WITHOUT running git;
* the dry-run push writes nothing to the remote;
* missing toolchains and Team Bridge enrolment are reported, never blocking;
* an older destination reads ``unknown_destination`` on the offload's link ask.

Isolation: ``GIT_CONFIG_SYSTEM``/``GIT_CONFIG_GLOBAL`` are ``/dev/null`` and
``GIT_TERMINAL_PROMPT=0``; remotes are local bare repositories reached through
``url.<path>.insteadOf`` (the module's test seam) or a fake ``GIT_SSH_COMMAND``.
No network, no real credential.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import mobility, workset_probe
from tests.unit.network.test_mobility import (  # noqa: F401 — `devices` is a fixture
    SESSION,
    Devices,
    _owned_session,
    devices,
)
from tests.unit.network.test_relay_e2e import _pair_settled
from tests.unit.network.workset_rig import Rig, git, make_rig, point_session_at

HTTPS = "https://git.test/owner/proj.git"
SSH = "ssh://git.test/owner/proj.git"
SCP = "git@git.test:owner/proj.git"

_FAKE_SSH = """#!/bin/sh
# argv: <host> <command>; the command is the last argument (GIT_SSH_VARIANT=simple).
for last; do :; done
case "$FAKE_SSH_MODE:$last" in
  deny_all:*) echo "ssh: connect to host git.test port 22: Connection refused" >&2; exit 255;;
  *:git-upload-pack*) exec git upload-pack "$FAKE_SSH_REPO";;
  allow:git-receive-pack*) exec git receive-pack "$FAKE_SSH_REPO";;
  *:git-receive-pack*) echo "ERROR: Permission to owner/proj.git denied to probe." >&2; exit 128;;
esac
exit 1
"""


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    built = make_rig(tmp_path / "ws")
    monkeypatch.setenv("HOME", str(built.home))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    script = tmp_path / "fake-ssh"
    script.write_text(_FAKE_SSH, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("GIT_SSH_COMMAND", str(script))
    monkeypatch.setenv("GIT_SSH_VARIANT", "simple")
    monkeypatch.setenv("FAKE_SSH_REPO", str(built.origin))
    monkeypatch.setenv("FAKE_SSH_MODE", "deny_push")
    return built


def _map_https(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """``https://git.test/owner/proj.git`` -> ``target`` for the probe's git only."""
    monkeypatch.setattr(workset_probe, "_EXTRA_CONFIG", (f"url.{target}.insteadOf={HTTPS}",))


def _probe(tmp_path: Path, *rows: dict[str, Any], team_bridge: bool = False) -> dict[str, Any]:
    frame = {"worktrees": list(rows), "team_bridge": team_bridge}
    return workset_probe.run_probe(tmp_path, frame, device="node2")


def _row(url: str, *, path: str = "~/proj/.worktrees/feat", markers: list[str] | None = None):
    return {"path": path, "remote_url": url, "branch": "feat", "markers": markers or []}


def test_a_reachable_remote_answers_ok_and_the_dry_run_writes_nothing(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _map_https(monkeypatch, rig.origin)
    refs_before = git(rig.origin, "for-each-ref")

    answer = _probe(tmp_path, _row(HTTPS))

    assert answer["worktrees"][0]["clone_check"] == "ok", answer
    # NO REMOTE WRITE: the throwaway probe branch was never created.
    assert git(rig.origin, "for-each-ref") == refs_before
    assert "lop-move-probe" not in refs_before


def test_an_unfetchable_remote_answers_cannot_fetch(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _map_https(monkeypatch, tmp_path / "does-not-exist.git")

    row = _probe(tmp_path, _row(HTTPS))["worktrees"][0]

    assert row["clone_check"] == "destination_cannot_fetch", row
    assert "https_rewrite" not in row  # already HTTPS: nothing to offer


def test_a_push_denied_remote_answers_cannot_push(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "deny_push")

    row = _probe(tmp_path, _row(SSH))["worktrees"][0]

    assert row["clone_check"] == "destination_cannot_push", row
    assert "Permission" in row.get("detail", ""), row
    # Control: the same fake ssh allowing receive-pack answers ok, so the
    # failure above is the denial and not a broken rig.
    monkeypatch.setenv("FAKE_SSH_MODE", "allow")
    assert _probe(tmp_path, _row(SSH))["worktrees"][0]["clone_check"] == "ok"


def test_ssh_unreachable_but_https_reachable_offers_the_rewrite(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SSH_MODE", "deny_all")
    _map_https(monkeypatch, rig.origin)

    row = _probe(tmp_path, _row(SCP))["worktrees"][0]

    assert row["clone_check"] == "destination_cannot_fetch", row
    assert row["https_rewrite"] == {"url": HTTPS, "fetch": True, "push": True}, row


def test_a_github_remote_names_the_lending_device_remedy(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # github.com over a fake ssh that denies receive-pack: no network involved.
    monkeypatch.setenv("FAKE_SSH_MODE", "deny_push")

    row = _probe(tmp_path, _row("git@github.com:owner/proj.git"))["worktrees"][0]

    assert row["clone_check"] == "destination_cannot_push", row
    # No share on this (isolated) device -> the share verb, run on the lender.
    assert row["remedy"] == {
        "run_on": "lending_device",
        "command": "lop network credential share github --with node2",
    }


@pytest.mark.parametrize(
    "url",
    [
        "ext::sh -c touch% /tmp/lop-probe-pwned",
        "file:///etc",
        "/srv/git/proj.git",
        "./proj.git",
        "~/proj.git",
        "--upload-pack=touch /tmp/x",
        "host:-oProxyCommand=touch /tmp/x",
        "http://git.test/owner/proj.git",
        "fd::17",
        "https://git.test/owner/proj.git\nfoo",
    ],
)
def test_a_hostile_url_is_refused_without_running_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(workset_probe, "_git", lambda args, **_kw: calls.append(args) or (True, ""))

    row = _probe(tmp_path, _row(url))["worktrees"][0]

    assert row == {
        "path": "~/proj/.worktrees/feat",
        "clone_check": "destination_cannot_clone",
        "detail": "url_refused",
        "missing_tools": [],
    }
    assert calls == []


def test_missing_toolchains_and_team_bridge_are_reported_never_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = {"git", "node", "uv"}
    monkeypatch.setattr(
        workset_probe.shutil, "which", lambda name: f"/bin/{name}" if name in present else None
    )
    monkeypatch.setattr(workset_probe, "_can_fetch", lambda url, env: (True, ""))
    monkeypatch.setattr(workset_probe, "_can_push", lambda url, env, scratch: (True, ""))

    answer = _probe(
        tmp_path,
        _row(HTTPS, markers=["go.mod", "pnpm-lock.yaml", "pyproject.toml"]),
        team_bridge=True,
    )

    assert answer["worktrees"][0]["missing_tools"] == ["go", "pnpm"]
    assert answer["worktrees"][0]["clone_check"] == "ok"
    assert answer["team_bridge"] == {"needed": True, "enrolled": False}
    block = {"worktrees": [{"path": "~/proj/.worktrees/feat", "state": "planned"}]}
    applied = workset_probe.apply_probe(block, answer)
    assert applied["worktrees"][0]["state"] == "planned"  # a flag, never a state change
    assert applied["destination_probe"] == {
        "state": "answered",
        "missing_tools": ["go", "pnpm"],
        "team_bridge": {"needed": True, "enrolled": False},
    }


# ---------------------------------------------------------------------------
# Through the relays: the preflight carries the destination's answer
# ---------------------------------------------------------------------------


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    value: Devices = request.getfixturevalue("devices")
    _server_a, server_b, _host, _port = value
    server_b.bind_control()
    server_b.start()
    return value


def _dry_run(server: Any, monkeypatch: pytest.MonkeyPatch, *, to: str) -> dict[str, Any]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server.root))
    return dict(mobility.request_move(SESSION, root=server.root, to=to, dry_run=True))


def _ssh_session(pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig) -> None:
    server_a, server_b, _host, _port = pair
    _pair_settled(pair, monkeypatch, role="admin", settings=server_b.settings)
    git(rig.primary, "remote", "set-url", "origin", SSH)
    point_session_at(_owned_session(server_a), rig.feat)


def test_the_offload_preflight_asks_the_destination_and_flags_the_row(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    server_a, server_b, _host, _port = pair
    _ssh_session(pair, monkeypatch, rig)
    asked: list[str] = []
    real = mobility._destination_workset_probe

    def spy(server: Any, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        asked.append(server.identity.device_id)
        return real(server, link, frame)

    monkeypatch.setattr(mobility, "_destination_workset_probe", spy)

    preflight = _dry_run(server_a, monkeypatch, to=server_b.identity.device_id)

    block = preflight["working_set"]
    assert asked == [server_b.identity.device_id], block  # the DESTINATION answered
    row = next(r for r in block["worktrees"] if r["path"].endswith("feat"))
    assert row["clone_check"] == "destination_cannot_push", row
    assert row["state"] == "planned"
    assert block["destination_probe"]["state"] == "answered"
    assert "pnpm" in row["markers"] or row["markers"] == ["pnpm-lock.yaml"]


def test_an_older_destination_reads_unknown_destination_not_a_failure(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    server_a, server_b, _host, _port = pair
    _ssh_session(pair, monkeypatch, rig)

    def older(server: Any, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        # What a pre-feature destination's dispatcher raises for a phase it lacks.
        raise mobility._peer_error("bad_request", "unknown move phase 'workset_probe'")

    monkeypatch.setattr(mobility, "_destination_workset_probe", older)

    preflight = _dry_run(server_a, monkeypatch, to=server_b.identity.device_id)

    assert preflight["ok"] is True, preflight
    block = preflight["working_set"]
    assert {row["clone_check"] for row in block["worktrees"]} == {"unknown_destination"}
    assert block["destination_probe"] == {"state": "unknown_destination"}
    assert preflight["confirmation_required"] is True


def test_the_recall_preflight_probes_this_device_itself(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    _server_a, server_b, _host, _port = pair
    _ssh_session(pair, monkeypatch, rig)
    monkeypatch.setenv("FAKE_SSH_MODE", "allow")

    preflight = _dry_run(server_b, monkeypatch, to="local")

    row = next(r for r in preflight["working_set"]["worktrees"] if r["path"].endswith("feat"))
    assert row["clone_check"] == "ok", row
    assert row["missing_tools"] == workset_probe.missing_tools(["pnpm-lock.yaml"])
