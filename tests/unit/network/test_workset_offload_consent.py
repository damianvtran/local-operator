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


def _flushed_events(server: Any) -> list[dict[str, Any]]:
    """The relay's audit rows, published first: ``AuditLog.record`` buffers, so a
    read straight after the move can miss a row that IS recorded."""
    server.audit.flush()
    return _audit_events(server.root)


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
        e for e in _flushed_events(server_a) if e.get("event") == "session.workset.consent"
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
    # A digest the user named, honoured as named, is not a broken plan (security S1).
    assert "confirmed_plan" not in receipt, receipt
    (row_audit,) = [
        e for e in _flushed_events(server_a) if e.get("event") == "session.workset.consent"
    ]
    assert row_audit["detail"]["surface"] == "cli-digest", row_audit
    assert row_audit["detail"]["carry_decision"] == "stay", row_audit


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
    declined = document["consent_declined"]
    assert declined["reason"] == "preflight_failed", document
    # Machine code on the JSON; the relay's English stays out of the keyed sentence
    # (review R1-6) and nobody but the agent path waits on the operator (QA Q-3).
    assert declined["preflight_error"] == "unreachable", declined
    assert "that device did not answer" not in declined["message"], declined
    assert declined["operator_confirmation_required"] is False, declined
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
    assert "desktop_move_card" in declined["operator_surfaces"]
    # Keyed copy (cli.sessions_move_workset), English on the wire for the agent.
    assert declined["message_envelope"]["code"] == (
        "cli.sessions_move_workset.agent_needs_operator"
    )
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


# ---------------------------------------------------------------------------
# Round-1 remediation (PR #4380 ledger): Q-1, U1, S1, R1-1..R1-4, S3, U2-U6
# ---------------------------------------------------------------------------


def _exhaust_probe_budget(monkeypatch: pytest.MonkeyPatch, asker: Any) -> None:
    """Make B's REAL probe admission refuse A (``workset_probe.rate_limited``), the
    QA Q-1 repro: B is up and granted, but its answer to the probe is ``rate_limited``,
    which ``mobility._probe_destination`` turns into ``unknown_destination``."""
    from local_operator.network import workset_probe

    now = time.monotonic()
    monkeypatch.setitem(
        workset_probe._rate_seen,  # noqa: SLF001 — the probe's own admission state
        str(asker.identity.device_id),
        [now] * workset_probe.PROBE_RATE_LIMIT,
    )


def test_yes_refuses_as_preflight_failed_when_the_destination_does_not_answer(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """QA Q-1: a granted, capable B whose probe goes unanswered was refused as
    ``carry_unavailable`` ("cannot take the working files"). Unknown is not "no"."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    _exhaust_probe_budget(monkeypatch, server_a)
    # The real shape, through the real dry run: ok, but the destination never answered.
    shown = _move(server_a, monkeypatch, to=target, dry_run=True)
    assert shown["ok"] is True, shown
    assert shown["working_set"]["destination_probe"] == {"state": "unknown_destination"}

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes", "--json")

    document = json.loads(out.out)
    assert code == 1 and document["ok"] is False, document
    declined = document["consent_declined"]
    assert declined["reason"] == "preflight_failed", declined
    assert declined["preflight_error"] == "destination_unanswered", declined
    assert declined["files"] == "not_moved"
    assert "cannot take" not in declined["message"], declined
    assert "did not answer" in declined["message"], declined
    # The top-level message no longer offers the digest route next to the refusal.
    assert document["message"] == declined["message"], document
    _still_on_a(server_a, bhome)


def test_yes_text_for_an_unanswered_destination_says_only_what_is_known(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """UX U3/U5: no "they stay on device-a … Only continue if" verdict after "did not
    answer", a way to check reachability, and the device's NAME, not its id."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    _exhaust_probe_budget(monkeypatch, server_a)

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes")

    assert code == 1, out
    assert out.out == "", out
    lines = out.err.strip().splitlines()
    assert len(lines) == 1, out.err
    assert lines[0].startswith("--yes did not confirm: device-b did not answer"), out.err
    assert "lop network peers" in out.err and target not in out.err
    assert "Only continue" not in out.err and "--confirm-working-set" not in out.err
    _still_on_a(server_a, bhome)


def test_an_agent_shell_digest_cannot_carry_past_an_unanswered_probe(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-1's class on the agent path: an unanswered probe recomputes the plan as
    "files stay", so an OLD carrying digest no longer matched the preflight and went
    through to a gate that, B being granted and up, would carry it."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    sends = _move(server_a, monkeypatch, to=target, dry_run=True)["working_set"]
    assert sends["carries_files"] is True
    _exhaust_probe_budget(monkeypatch, server_a)
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
    declined = document["consent_declined"]
    assert declined["reason"] == "agent_shell", declined
    assert declined["preflight_error"] == "destination_unanswered", declined
    _still_on_a(server_a, bhome)


def test_an_agent_shell_files_stay_digest_follows_the_human_path(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review R1-4: once the preflight answers, a digest for a plan that leaves the
    files here is the agent's to give — the restriction only refuses carrying."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    target = server_b.identity.device_id
    stays = _move(server_a, monkeypatch, to=target, dry_run=True)["working_set"]
    assert stays["carries_files"] is False, stays
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")

    code, out = _cli(
        server_a,
        monkeypatch,
        capsys,
        "--to",
        target,
        "--confirm-working-set",
        stays["scope_digest"],
        "--json",
    )

    receipt = json.loads(out.out)
    assert code == 0, receipt
    assert receipt["working_set"]["transferred"] is False, receipt
    assert not (bhome / "proj" / ".worktrees" / "feat").exists()


def _revoke_after_preflight(monkeypatch: pytest.MonkeyPatch, server_a: Any, server_b: Any) -> None:
    """B stops accepting A's files between the CLI's preflight and the move (the
    owner's narrowing case, ``mobility._workset_gate_full``)."""
    real = cli._sessions_move_preflight  # noqa: SLF001

    def then_revoke(session_id: str, *, to: str) -> dict[str, Any]:
        answer = real(session_id, to=to)
        _grant_import(server_b, server_a, revoke=True)
        return answer

    monkeypatch.setattr(cli, "_sessions_move_preflight", then_revoke)


def test_a_confirmed_carry_that_did_not_happen_leads_with_not_sent(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """UX U1: "Moved … (done)" plus "stays … as confirmed for this move" read as
    success and misstated the consent. The keyed "not sent" line now leads, on stderr,
    and no row says "as confirmed"."""
    from local_operator.i18n.keys import cli_sessions_move_workset as wsh
    from local_operator.i18n.messages import render_resolved

    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    _revoke_after_preflight(monkeypatch, server_a, server_b)

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes")

    assert code == 1, out
    expected = render_resolved(wsh.plan_not_honoured(device="device-b", source="device-a"))
    assert out.err.splitlines()[0] == expected, out.err
    assert expected.startswith("The working files were not sent."), expected
    assert "as confirmed" not in out.out + out.err, out
    # The row says the real cause; the conversation did move.
    assert "the other device cannot receive it" in out.out, out.out
    assert not (bhome / "proj" / ".worktrees" / "feat").exists()


def test_an_explicit_carrying_digest_the_owner_narrowed_exits_1(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Security S1: the same narrowing under ``--confirm-working-set <carrying>``
    exited 0 with no ``confirmed_plan`` while ``--yes`` exited 1."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    sends = _move(server_a, monkeypatch, to=target, dry_run=True)["working_set"]
    assert sends["carries_files"] is True
    _grant_import(server_b, server_a, revoke=True)

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

    receipt = json.loads(out.out)
    assert code == 1, receipt
    assert receipt["ok"] is True and receipt["working_set"]["transferred"] is False
    assert receipt["confirmed_plan"] == {
        "scope_digest": sends["scope_digest"],
        "carries_files": True,
        "honoured": False,
        "reason": "files_not_sent",
    }


def test_the_prompt_in_an_agent_shell_refuses_a_carrying_plan(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review R1-2: an agent faking a tty could answer `y`. The refusal names the
    operator's commands on their own lines (U6) and offers no route it refuses (U2)."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    def no_prompt(_prompt: str) -> str:
        raise AssertionError("an agent shell must not reach the y/N prompt for a carry")

    monkeypatch.setattr("builtins.input", no_prompt)

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--queue")

    assert code == 1, out
    assert out.err.startswith("lop sessions move did not confirm:"), out.err
    lines = out.err.splitlines()
    # Each command on its own line, whole, with the caller's own move flags (U6).
    assert f"  lop sessions move {SESSION} --to {target} --queue" in lines, lines
    assert f"  lop sessions move {SESSION} --to {target} --dry-run" in lines, lines
    # What WOULD be sent is listed, but nothing invites the agent to continue (U2).
    assert "~/proj/.worktrees/feat" in out.err
    assert "Only continue" not in out.err
    assert "re-run with --confirm-working-set sha256:" not in out.err
    _still_on_a(server_a, bhome)


def test_an_agent_shell_yes_text_has_no_digest_hint(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """UX U2 / QA Q-4: the refusal was followed by "re-run with --confirm-working-set
    <carrying digest>", the exact command the same shell then refuses."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes")

    assert code == 1, out
    assert out.err.startswith("--yes did not confirm:"), out.err
    assert "sha256:" not in out.err and "Only continue" not in out.err, out.err
    _still_on_a(server_a, bhome)


def test_the_consent_row_precedes_staging_and_the_journal(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Security S3 / review R1-3: the row exists before any byte is staged and before
    the handoff journal is written, so it survives a move that later fails."""
    from local_operator.network import workset_move
    from local_operator.session import placement

    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    seen: dict[str, bool] = {}

    def consent_rows() -> bool:
        return any(e.get("event") == "session.workset.consent" for e in _flushed_events(server_a))

    def wrap(module: Any, name: str) -> None:
        real = getattr(module, name)

        def spy(*a: Any, **k: Any) -> Any:
            # Only A's own calls: B writes its own journal entry for the pull.
            if Path(a[0] if name == "write_handoff_entry" else a[0].root) == server_a.root:
                seen.setdefault(name, consent_rows())
            return real(*a, **k)

        monkeypatch.setattr(module, name, spy)

    wrap(workset_move, "source_prepare")
    wrap(placement, "write_handoff_entry")

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target, "--yes", "--json")

    assert code == 0, out
    assert seen == {"source_prepare": True, "write_handoff_entry": True}, seen


def test_a_consent_with_no_surface_is_unattributed_never_operator(
    pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    rig: Rig,
    bhome: Path,
) -> None:
    """Security S3 / review R1-3: a caller (or an older hop) that names no surface is
    recorded as ``unattributed``; nothing is ever recorded as ``operator``."""
    server_a, server_b, _host, _port = pair
    _setup(pair, monkeypatch, rig, bhome)
    _grant_import(server_b, server_a)
    target = server_b.identity.device_id
    sends = _move(server_a, monkeypatch, to=target, dry_run=True)["working_set"]

    moved = _move(server_a, monkeypatch, to=target, confirm_working_set=sends["scope_digest"])

    assert moved["ok"] is True, moved
    (row_audit,) = [
        e for e in _flushed_events(server_a) if e.get("event") == "session.workset.consent"
    ]
    assert row_audit["detail"]["surface"] == "unattributed", row_audit


def test_the_consent_frame_keeps_only_closed_surface_tokens() -> None:
    """``surface`` crosses the wire: only ``CONSENT_SURFACES`` tokens survive a hop."""
    for token in sorted(mobility.CONSENT_SURFACES):
        assert mobility._consent_frame({"surface": token})["surface"] == token  # noqa: SLF001
        # Re-stamped at the next hop, unchanged.
        again = mobility._consent_frame(mobility._consent_frame({"surface": token}))  # noqa: SLF001
        assert again["surface"] == token
    for claimed in ("operator", "OPERATOR", "cli-yes\n", "desktop; rm -rf", ""):
        assert mobility._consent_frame({"surface": claimed})["surface"] == ""  # noqa: SLF001
    assert "operator" not in mobility.CONSENT_SURFACES


def test_the_desktop_card_claims_the_desktop_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.server.utils import desktop_mesh

    calls: list[dict[str, Any]] = []

    def record(session_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(mobility, "request_move", record)
    desktop_mesh.transfer(SESSION, to="device-b", confirm_working_set="sha256:x")
    assert calls and calls[0]["consent_surface"] == "desktop"


def test_the_prompt_records_the_cli_prompt_surface(
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
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    code, out = _cli(server_a, monkeypatch, capsys, "--to", target)

    assert code == 0, out
    (row_audit,) = [
        e for e in _flushed_events(server_a) if e.get("event") == "session.workset.consent"
    ]
    assert row_audit["detail"]["surface"] == "cli-prompt", row_audit
    assert row_audit["detail"]["carry_decision"] == "carry", row_audit


def test_confirmed_plan_has_a_verdict_only_for_a_move_that_ran() -> None:
    """Review R1-1: ``honoured: true`` was stamped on a refused re-issue and on a queued
    move. A carry whose answer never arrived is not "not sent" either (the files may
    be there), so it carries ``reason: outcome_unknown`` and no verdict."""
    block = {"scope_digest": "sha256:x", "carries_files": True}

    moved = {"ok": True, "working_set": {"transferred": False}}
    plan = cli._sessions_move_bind_plan(moved, block)["confirmed_plan"]
    assert plan["honoured"] is False and plan["reason"] == "files_not_sent"

    sent = {"ok": True, "working_set": {"transferred": True}}
    assert cli._sessions_move_bind_plan(sent, block)["confirmed_plan"]["honoured"] is True

    for unrun in (
        {"ok": False, "code": "working_set_changed", "working_set": {"transferred": False}},
        {"ok": True, "queue": {"state": "queued"}, "working_set": {"transferred": False}},
    ):
        plan = cli._sessions_move_bind_plan(unrun, block)["confirmed_plan"]
        assert "honoured" not in plan, (unrun, plan)

    unknown = {"ok": True, "working_set": {"side": "source", "state": "outcome_unknown"}}
    plan = cli._sessions_move_bind_plan(unknown, block)["confirmed_plan"]
    assert "honoured" not in plan and plan["reason"] == "outcome_unknown", plan

    stays = {"scope_digest": "sha256:y", "carries_files": False}
    assert cli._sessions_move_bind_plan(moved, stays)["confirmed_plan"]["honoured"] is True


def test_the_source_receipt_of_a_carry_says_it_carried() -> None:
    """QA Q-2: a successful carry's receipt said ``carries_files: false`` with the
    "files stay" digest beside ``transferred: true``."""
    from local_operator.network import workset_move

    planned = {
        "carries_files": False,
        "scope_digest": "sha256:stays",
        "worktrees": [{"path": "~/proj/.worktrees/feat", "state": "left_behind"}],
    }
    answer = {
        "transferred": True,
        "worktrees": [{"path": "~/proj/.worktrees/feat", "state": "transferred"}],
    }

    block = workset_move.source_receipt(planned, answer, sent_digest="sha256:sends")
    assert block["transferred"] is True and block["carries_files"] is True
    assert block["scope_digest"] == "sha256:sends"
    unknown = workset_move.source_receipt(planned, None, sent_digest="sha256:sends")
    assert unknown["carries_files"] is True and unknown["state"] == "outcome_unknown"
