"""Non-interactive offload consent: ``lop sessions move <id> --to <peer> --yes/--json``.

THE v0.111.0 DEFECT (receipt C2, #4378 QA): ``--yes`` confirmed the owner's FIRST
refusal, which an offload computes before the destination is asked, so it is always
the "files stay here" scope. The move reported ``done`` with ``carries_files: false``,
``transferred: false`` and every worktree ``left_behind`` even though the destination
could take them. Interactive moves were right because only they ran the
destination-aware preflight.

Each cell drives the REAL parser and ``cli.sessions_move_command`` on the OWNER (device
A) against the in-process two-relay rig, with a real git worktree as the session's cwd
and B's ``workset_import`` grant set or not, so the preflight, the owner's gate and the
destination's import check are all on the path. The consent rules pinned here
(security ruling, 2026-11-21):

* ``--yes`` confirms ONLY the carrying preflight this invocation computed;
* ``--json`` alone is not consent: exit 1 with that carrying plan;
* an explicit narrow digest is honoured exactly, ``--yes`` or not;
* no carry possible / preflight failed → ``--yes`` refuses, nothing moves;
* in an agent's shell (advisory marker) a carrying confirmation refuses and names the
  operator's surfaces; ``--without-working-set`` still moves the conversation only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli
from local_operator.agent_shell import AGENT_SHELL_ENV
from local_operator.network import mobility
from tests.unit.network.test_mobility import SESSION, Devices
from tests.unit.network.test_mobility import devices as _devices_fixture
from tests.unit.network.test_workset_move_e2e import (
    _audit_events,
    _grant_import,
    _move,
    _setup,
)
from tests.unit.network.test_workset_move_e2e import bhome as _bhome_fixture
from tests.unit.network.test_workset_move_e2e import pair as _pair_fixture
from tests.unit.network.test_workset_move_e2e import rig as _rig_fixture
from tests.unit.network.workset_rig import Rig, git

# Expose the original fixture objects without wrapping their scope or dependencies
# (the convention in test_peer_file_read_audit.py).
devices = _devices_fixture
pair = _pair_fixture
rig = _rig_fixture
bhome = _bhome_fixture


def _cli(
    server: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *flags: str
) -> tuple[int, Any]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server.root))
    args = cli.build_cli_parser().parse_args(["sessions", "move", SESSION, *flags])
    capsys.readouterr()  # the rig's pairing prints its SAS document; not this command's
    code = cli.sessions_move_command(args)
    return code, capsys.readouterr()


def _landed(bhome: Path) -> Path:
    """B restores after ``done`` on its own pull thread, so A's answer can come first."""
    landed = bhome / "proj" / ".worktrees" / "feat"
    deadline = time.monotonic() + 30
    while not (landed / ".git").exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    return landed


def _still_on_a(server_a: Any, bhome: Path) -> None:
    assert (server_a.root / "sessions" / SESSION).is_dir()
    assert not (bhome / "proj" / ".worktrees" / "feat").exists()


def test_yes_json_offload_carries_the_files(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The C2 receipt, inverted: a capable, granted destination gets the files."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    shown = _move(server_a, monkeypatch, to=target, dry_run=True)["working_set"]
    assert shown["carries_files"] is True, shown

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes", "--json")

    receipt = json.loads(out.out)
    assert code == 0, receipt
    ws = receipt["working_set"]
    assert ws["consent"] == "confirmed" and ws["transferred"] is True, ws
    (row,) = [r for r in ws["worktrees"] if r["path"].endswith("feat")]
    assert row["state"] in ("transferred", "restored") and row["state"] != "left_behind", ws
    assert receipt["confirmed_plan"] == {
        "scope_digest": shown["scope_digest"],
        "carries_files": True,
        "honoured": True,
    }
    landed = _landed(bhome)
    assert "unpushed work" in git(landed, "log", "-1", "--format=%s")
    # The secret-shaped file never travels (security exclusion unchanged).
    assert not (landed / ".env.local").exists()
    # The owner recorded WHO consented to WHAT before the transfer started.
    (row_audit,) = [
        e for e in _audit_events(server_a.root) if e.get("event") == "session.workset.consent"
    ]
    detail = row_audit["detail"]
    assert detail["surface"] == "cli-yes", row_audit
    assert detail["digest"] == shown["scope_digest"], row_audit
    assert detail["carry_decision"] == "carry", row_audit


def test_yes_without_json_offload_carries_the_files(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes")

    assert code == 0, out
    assert (_landed(bhome) / ".git").exists()
    assert not (server_a.root / "sessions" / SESSION).exists()


def test_json_alone_is_not_consent_and_shows_the_carrying_plan(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    shown = _move(server_a, monkeypatch, to=target, dry_run=True)["working_set"]

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--json")

    document = json.loads(out.out)
    assert code == 1 and document["code"] == "working_set_unconfirmed", document
    # The plan a script would pass back is the CARRYING one the preflight showed.
    assert document["working_set"]["carries_files"] is True, document
    assert document["working_set"]["scope_digest"] == shown["scope_digest"]
    _still_on_a(server_a, bhome)


def test_an_explicit_narrow_digest_is_honoured_exactly_under_yes(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A "files stay" digest the user named stays files-stay even once B could carry."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    target = server_b.identity.device_id
    stays = _move(server_a, monkeypatch, to=target, dry_run=True)["working_set"]
    assert stays["carries_files"] is False, stays
    _grant_import(server_b, server_a)

    code, out = _cli(
        server_a,
        monkeypatch,
        capsys,
        "--to",
        target,
        "--confirm-working-set",
        stays["scope_digest"],
        "--yes",
        "--json",
    )

    receipt = json.loads(out.out)
    assert code == 0, receipt
    assert receipt["working_set"]["transferred"] is False, receipt
    assert not (bhome / "proj" / ".worktrees" / "feat").exists()


def test_yes_refuses_when_the_destination_cannot_take_the_files(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No ``workset_import`` grant on B: v0.111.0 moved with the files left behind and
    said ``done``. Leaving them is now a decision the caller states, not one --yes makes."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    target = server_b.identity.device_id

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes", "--json")

    document = json.loads(out.out)
    assert code == 1 and document["ok"] is False, document
    declined = document["consent_declined"]
    assert declined["reason"] == "carry_unavailable" and declined["files"] == "not_moved"
    assert "--without-working-set" in declined["message"]
    _still_on_a(server_a, bhome)


def test_yes_refuses_when_the_preflight_fails(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Never fall back to the owner's non-carrying digest when the plan is unknown."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id

    def unreachable(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise mobility.Moved("unreachable", "that device did not answer")

    monkeypatch.setattr(mobility, "_workset_preflight_local", unreachable)

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes", "--json")

    document = json.loads(out.out)
    assert code == 1, document
    assert document["consent_declined"]["reason"] == "preflight_failed", document
    _still_on_a(server_a, bhome)


def test_an_agent_shell_yes_refuses_to_carry_and_names_the_operator_surface(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Agent path stays files-stay (security-owner ruling): the refusal is
    machine-readable and names where the OPERATOR confirms."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes", "--json")

    document = json.loads(out.out)
    assert code == 1 and document["ok"] is False, document
    declined = document["consent_declined"]
    assert declined["reason"] == "agent_shell", declined
    assert declined["files"] == "not_moved" and declined["operator_confirmation_required"]
    assert f"lop sessions move {SESSION} --to {target}" in declined["operator_surfaces"]
    assert "move card" in declined["message"], declined
    _still_on_a(server_a, bhome)


def test_an_agent_shell_digest_that_would_carry_is_refused(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    sends = _move(server_a, monkeypatch, to=target, dry_run=True)["working_set"]
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")

    code, out = _cli(
        server_a,
        monkeypatch,
        capsys,
        "--to",
        target,
        "--confirm-working-set",
        sends["scope_digest"],
        "--json",
    )

    document = json.loads(out.out)
    assert code == 1, document
    assert document["consent_declined"]["reason"] == "agent_shell", document
    assert document["working_set"]["carries_files"] is True
    _still_on_a(server_a, bhome)


def test_an_agent_shell_can_still_move_only_the_conversation(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Files-stay is available to the agent, and its receipt says so plainly."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")

    code, out = _cli(
        server_a, monkeypatch, capsys, "--to", target, "--without-working-set", "--json"
    )

    receipt = json.loads(out.out)
    assert code == 0, receipt
    assert receipt["working_set"]["consent"] == "without"
    assert receipt["working_set"]["transferred"] is False
    assert not (bhome / "proj" / ".worktrees" / "feat").exists()


def test_a_confirmed_carry_that_did_not_happen_is_not_reported_as_honoured() -> None:
    """The owner's narrowing rule (destination changed after the preflight) is safe,
    but must never read as the plan the user consented to."""
    block = {"scope_digest": "sha256:x", "carries_files": True}
    moved = {"ok": True, "working_set": {"transferred": False}}

    plan = cli._sessions_move_bind_plan(moved, block)["confirmed_plan"]
    assert plan["honoured"] is False and plan["reason"] == "files_not_sent"

    queued = {"ok": True, "queue": {"state": "queued"}, "working_set": {"transferred": False}}
    assert cli._sessions_move_bind_plan(queued, block)["confirmed_plan"]["honoured"] is True
