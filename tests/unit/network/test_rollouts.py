"""S3a of the mesh rolling updates: the record and the serial pass.

One cell per behaviour the note freezes (``mesh-update-propagation.md`` §3/§4/§6;
RU §6.1/§6.3): the record's discipline (atomic, supersede, window, reader-never-
creates), the pass's rules (serial, skip-then-retry, busy cadence, failure
backoff, no_grant parked, unreachable caught on contact, drop-after-apply
re-probe), resume/cancel, the closing rows, and — the S3a contract — the
driver's DORMANCY: with no record, it asks nobody and writes nothing, and the
only way a record appears today is an explicit ``open_record`` call (S3b's hook
is what will make that a production path; the module docstring and the PR say
so, and this file's dormancy cell observes the inert half directly).

THE ASK IS FAKED AT ITS SEAM: ``meshupdate.update_peer`` is the one entry point
a driver's ask goes through, and every cell here scripts it — the pass's logic
is what this slice owns, while the ask itself (dial, frame, member's reply)
is S1's and is exercised against real relays in ``test_mesh_update.py``.
"""

from __future__ import annotations

import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import meshupdate, rollouts


class _Audit:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def record(self, event: Any) -> None:
        self.rows.append(event)


class _Server:
    """Just enough relay: a root and an audit sink (the two the pass touches)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.audit = _Audit()
        self.ops: Any = None
        self.kw: Any = None

    def register_ops(self, ops: Any, **kw: Any) -> None:
        self.ops = ops
        self.kw = kw


def _open(
    root: Path | None, *, devices: list[str], network: str = "n1", now: float = 1000.0
) -> dict:
    return rollouts.open_record(
        network=network,
        origin="d_origin",
        target={"version": "0.99.0", "source_ref": ""},
        members=[{"device": d, "name": d} for d in devices],
        root=root,
        now=now,
    )


def _scripted(monkeypatch: pytest.MonkeyPatch, answers: list[dict[str, Any]]) -> list[str]:
    """Script ``update_peer``: one answer per call, remembering the devices asked."""
    asked: list[str] = []

    def _fake(server: Any, device: str, *, rollout: str = "") -> dict[str, Any]:
        asked.append(device)
        assert rollout.startswith("ro_"), rollout
        return answers.pop(0) if answers else {"ok": False, "code": "no_answer", "message": ""}

    monkeypatch.setattr(meshupdate, "update_peer", _fake)
    return asked


def _answer(state: str, code: str = "", reason: str = "", version: str = "") -> dict[str, Any]:
    return {
        "ok": state in ("done", "already_on_target", "ahead_of_target"),
        "state": state,
        "code": code,
        "message": reason,
        "method": "uv-tool-generation" if state == "done" else "",
        "version": version,
    }


def _member(record: dict[str, Any], device: str) -> dict[str, Any]:
    member = rollouts.member_of(record, device)
    assert member is not None
    return member


# ---------------------------------------------------------------------------
# The record's discipline (RU §6.1)
# ---------------------------------------------------------------------------


def test_readers_never_create_and_the_round_trip_holds(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    assert rollouts.read_record("ro_missing", root) is None
    assert not rollouts.rollouts_dir(root).exists(), "a reader created the rollouts dir"

    record = _open(root, devices=["d_a"])
    on_disk = rollouts.read_record(record["id"], root)
    assert on_disk is not None and on_disk["state"] == "active"
    assert on_disk["members"][0]["state"] == "pending"
    assert rollouts.rollouts_dir(root).exists()


def test_opening_supersedes_the_older_active_record_for_the_network(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    first = _open(root, devices=["d_a"], now=1000.0)
    second = _open(root, devices=["d_a", "d_b"], now=2000.0)
    assert rollouts.read_record(first["id"], root)["state"] == "superseded"
    assert rollouts.read_record(second["id"], root)["state"] == "active"

    # A DIFFERENT network's active record is NOT superseded: the record is per
    # (origin, network) (note §3, RU §6.3).
    other = _open(root, devices=["d_c"], network="n2", now=2500.0)
    assert rollouts.read_record(other["id"], root)["state"] == "active"


def test_the_window_lapse_closes_the_record_with_a_sentence(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"], now=1000.0)
    rollouts.step_record(server, record["id"], now=1000.0 + rollouts.WINDOW_S + 1.0)
    lapsed = rollouts.read_record(record["id"], root)
    assert lapsed["state"] == "done"
    rows = [r for r in server.audit.rows if r.event == rollouts.ROLLOUT_DONE]
    assert rows and "no longer active" in rows[-1].detail["counts"], rows


def test_the_summary_names_queued_and_skipped(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    record = _open(root, devices=["d_a", "d_b"])
    _member(record, "d_b").update({"state": "skipped", "code": "no_grant"})
    line = rollouts.summary(record)
    # RU §7.3 NAMES the sets (round 1, M4): counts alone hid WHO was skipped.
    assert "queued: d_a" in line and "skipped (no_grant): d_b" in line, line


def test_a_concurrent_open_does_not_retire_a_newer_record(tmp_path: Path) -> None:
    """M1 (round 1): an OLDER racer's scan must not supersede a newer record."""
    root = tmp_path / "cfg"
    newer = _open(root, devices=["d_a"], now=2000.0)
    older = _open(root, devices=["d_a"], now=1000.0)  # a stale racer's open
    assert (
        rollouts.read_record(newer["id"], root)["state"] == "active"
    ), "the newer record was retired by an older racer"
    # Both stay active — the conservative outcome — and no pass sees zero actives.
    assert rollouts.read_record(older["id"], root)["state"] == "active"


# ---------------------------------------------------------------------------
# The pass (note §4)
# ---------------------------------------------------------------------------


def test_the_pass_is_serial_and_a_done_member_is_not_re_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a", "d_b"])
    asked = _scripted(
        monkeypatch, [_answer("done", version="0.99.0"), _answer("already_on_target")]
    )

    rollouts.step_record(server, record["id"], now=1000.0)
    assert asked == ["d_a", "d_b"], asked  # in snapshot order, one at a time
    after = rollouts.read_record(record["id"], root)
    assert _member(after, "d_a")["state"] == "done"
    assert _member(after, "d_b")["state"] == "already_on_target"
    assert after["state"] == "done", "all members terminal must close the record"

    # A done member is never re-applied: a second scan asks nobody new.
    called: list[str] = []
    monkeypatch.setattr(
        meshupdate, "update_peer", lambda *a, **k: called.append("x") or _answer("busy")
    )
    rollouts.step_record(server, record["id"], now=2000.0)
    assert called == []


def test_a_busy_member_is_parked_on_the_cadence_and_writes_one_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a", "d_b"])
    _member(record, "d_b").update({"state": "done"})
    rollouts.mutate_record(record["id"], root, lambda rec: _sync(rec, record) or True)
    asked = _scripted(monkeypatch, [_answer("busy", reason="1 session is busy")])

    rollouts.step_record(server, record["id"], now=1000.0)
    member = _member(rollouts.read_record(record["id"], root), "d_a")
    assert member["state"] == "deferred" and member["code"] == "busy", member
    assert member["next_ask_at"] == 1000.0 + rollouts.CADENCE_S, member

    # ONE row for the state CHANGE, not one per poll: the cadence re-asks below
    # (inside the floor: not at all; past it: again, same answer) with no
    # further rows.
    rows = [r for r in server.audit.rows if r.event == rollouts.MEMBER_STATE]
    assert len(rows) == 1 and rows[0].cause == "busy", rows

    rollouts.step_record(server, record["id"], now=1100.0)
    assert asked == ["d_a"], "a member inside its floor must not be asked"

    _scripted(monkeypatch, [_answer("busy", reason="1 session is busy")])
    rollouts.step_record(server, record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
    rows = [r for r in server.audit.rows if r.event == rollouts.MEMBER_STATE]
    assert len(rows) == 1, "a re-ask with the same outcome must not add a row"


def _sync(rec: dict[str, Any], source: dict[str, Any]) -> None:
    """Copy one member's (test-set) state into the live record — cell glue."""
    for member in rec.get("members", []):
        for source_member in source.get("members", []):
            if member.get("device") == source_member.get("device"):
                member["state"] = source_member["state"]
                member["code"] = source_member.get("code", "")


def test_a_failed_member_backs_off_and_the_count_is_in_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    _scripted(monkeypatch, [_answer("failed", reason="installer exited 1")])
    rollouts.step_record(server, record["id"], now=1000.0)
    member = _member(rollouts.read_record(record["id"], root), "d_a")
    assert member["state"] == "failed" and member["fails"] == 1, member
    assert member["next_ask_at"] == 1000.0 + rollouts.FAIL_BACKOFF_S[0], member

    _scripted(monkeypatch, [_answer("failed", reason="installer exited 1")])
    rollouts.step_record(server, record["id"], now=1000.0 + rollouts.FAIL_BACKOFF_S[0] + 1.0)
    member = _member(rollouts.read_record(record["id"], root), "d_a")
    assert member["fails"] == 2, member
    assert member["next_ask_at"] == 1000.0 + rollouts.FAIL_BACKOFF_S[0] + 1.0 + (
        rollouts.FAIL_BACKOFF_S[1]
    ), member


def test_a_no_grant_member_is_parked_for_the_whole_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a", "d_b"])
    _member(record, "d_b").update({"state": "done"})
    rollouts.mutate_record(record["id"], root, lambda rec: _sync(rec, record) or True)
    asked = _scripted(monkeypatch, [{"ok": False, "code": "no_grant", "message": "remedy"}] * 3)

    for moment in (1000.0, 1300.0, 1600.0):  # ≥ 3 ticks past the cadence
        rollouts.step_record(server, record["id"], now=moment)
    assert asked == ["d_a"], "an ungranted member is parked: ONE refusal per rollout"
    final = rollouts.read_record(record["id"], root)
    assert _member(final, "d_a")["state"] == "skipped"
    assert _member(final, "d_a")["code"] == "no_grant"
    rows = [r for r in server.audit.rows if r.event == rollouts.MEMBER_STATE]
    assert len(rows) == 1 and rows[0].cause == "capability_denied", rows
    assert final["state"] == "done", "the parked member is terminal; the rest finished"


def test_an_unreachable_member_is_caught_on_its_next_contact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    _scripted(monkeypatch, [{"ok": False, "code": "unreachable", "message": "off"}])
    rollouts.step_record(server, record["id"], now=1000.0)
    member = _member(rollouts.read_record(record["id"], root), "d_a")
    assert member["state"] == "unreachable" and member["code"] == "unreachable", member
    assert member["next_ask_at"] == 1000.0 + rollouts.CADENCE_S, member

    _scripted(monkeypatch, [_answer("done", version="0.99.0")])
    rollouts.step_record(server, record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
    final = rollouts.read_record(record["id"], root)
    assert _member(final, "d_a")["state"] == "done" and final["state"] == "done"


def test_a_drop_after_applying_re_probes_on_the_bounded_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    seen_states: list[str] = []

    def _fake(server_: Any, device: str, *, rollout: str = "") -> dict[str, Any]:
        live = rollouts.read_record(record["id"], root)
        seen_states.append(str(_member(live, device)["state"]))
        return answers.pop(0)

    answers = [{"ok": False, "code": "no_answer", "message": ""}]
    monkeypatch.setattr(meshupdate, "update_peer", _fake)
    rollouts.step_record(server, record["id"], now=1000.0)
    assert seen_states == ["applying"], seen_states  # the in-flight marker is written first
    member = _member(rollouts.read_record(record["id"], root), "d_a")
    assert member["state"] == "deferred" and member["code"] == "no_answer", member
    assert member["next_ask_at"] == 1000.0 + rollouts.DROPPED_AFTER_APPLY_RETRY_S, member

    answers.append(_answer("already_on_target"))
    rollouts.step_record(
        server, record["id"], now=1000.0 + rollouts.DROPPED_AFTER_APPLY_RETRY_S + 1.0
    )
    assert _member(rollouts.read_record(record["id"], root), "d_a")["state"] == "already_on_target"


def test_resume_clears_the_floors_and_restarts_the_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a", "d_b"])
    _scripted(monkeypatch, [_answer("busy"), {"ok": False, "code": "unreachable", "message": ""}])
    rollouts.step_record(server, record["id"], now=1000.0)
    parked = rollouts.read_record(record["id"], root)
    assert _member(parked, "d_a")["next_ask_at"] > 0 and _member(parked, "d_b")["next_ask_at"] > 0

    resumed = rollouts.resume_record(record["id"], root)
    assert resumed is not None
    assert all(_member(resumed, d)["next_ask_at"] == 0.0 for d in ("d_a", "d_b"))

    asked = _scripted(
        monkeypatch, [_answer("done", version="0.99.0"), _answer("done", version="0.99.0")]
    )
    rollouts.step_record(server, record["id"], now=1000.0)  # SAME moment: floors are gone
    assert asked == ["d_a", "d_b"], asked


def test_the_three_member_pass_defers_offline_and_busy_then_rolls_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The S3a acceptance rig: busy + offline defer; both roll on their next contact.

    THE RIG IS THE DRIVER WITH THE ASK SEAM SCRIPTED (``update_peer``), NOT
    three live relays: the pass's rules — serial order, deferral, the cadence,
    the close — are what this slice owns, and S1's rig already exercises a real
    two-relay ask (``test_mesh_update.py``). A real three-device drill belongs
    to the QA round, which can stand it up without the driver being in its head.
    """
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_busy", "d_offline", "d_idle"])
    asked = _scripted(
        monkeypatch,
        [
            _answer("busy", reason="2 sessions busy"),
            {"ok": False, "code": "unreachable", "message": "not answering right now"},
            _answer("done", version="0.99.0"),
        ],
    )
    rollouts.step_record(server, record["id"], now=1000.0)
    assert asked == ["d_busy", "d_offline", "d_idle"], asked  # serial, snapshot order
    mid = rollouts.read_record(record["id"], root)
    assert mid["state"] == "active", "two members are still owed a roll"
    assert (
        _member(mid, "d_busy")["state"] == "deferred" and _member(mid, "d_busy")["code"] == "busy"
    )
    assert _member(mid, "d_offline")["state"] == "unreachable"
    assert _member(mid, "d_idle")["state"] == "done"

    # Nothing is due before the cadence; at it, both deferred members are asked
    # once more and both land — "caught on next contact".
    asked2 = _scripted(monkeypatch, [])
    rollouts.step_record(server, record["id"], now=1000.0 + rollouts.CADENCE_S - 1.0)
    assert asked2 == [], "a member inside its floor must not be asked"
    asked3 = _scripted(
        monkeypatch, [_answer("done", version="0.99.0"), _answer("already_on_target")]
    )
    rollouts.step_record(server, record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
    assert asked3 == ["d_busy", "d_offline"], asked3
    final = rollouts.read_record(record["id"], root)
    assert final["state"] == "done", final
    assert _member(final, "d_busy")["state"] == "done"
    assert _member(final, "d_offline")["state"] == "already_on_target"
    causes = [r.cause for r in server.audit.rows if r.event == rollouts.MEMBER_STATE]
    assert "busy" in causes and "peer_unreachable" in causes, causes


def test_cancel_abandons_the_record_and_stops_the_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    cancelled = rollouts.cancel_record(record["id"], root, server=server)
    assert cancelled is not None
    final = rollouts.read_record(record["id"], root)
    assert final["state"] == "abandoned"
    rows = [r for r in server.audit.rows if r.event == rollouts.ROLLOUT_DONE]
    assert rows and rows[-1].outcome == "partial", rows

    called: list[str] = []
    monkeypatch.setattr(
        meshupdate, "update_peer", lambda *a, **k: called.append("x") or _answer("busy")
    )
    rollouts.step_record(server, record["id"], now=2000.0)
    assert called == [], "an abandoned record must not be driven"


def test_a_fully_terminal_pass_closes_with_ok_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    _scripted(monkeypatch, [_answer("already_on_target")])
    rollouts.step_record(server, record["id"], now=1000.0)
    rows = [r for r in server.audit.rows if r.event == rollouts.ROLLOUT_DONE]
    assert rows and rows[-1].outcome == "ok" and rows[-1].detail["counts"] == "1 of 1 moved", rows


# ---------------------------------------------------------------------------
# Dormancy, registration, CLI — the S3a contract
# ---------------------------------------------------------------------------


def test_a_codeless_refusal_parks_the_member_as_no_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-1 (round 1): the authoriser's codeless `refused` IS the grant's absence.

    The live shape: `update_peer` reports the error frame's code — `refused`
    when the wire withheld which guard fired (`wire.refusal_frame`) — and the
    record must PARK the member (`skipped (no_grant)`), not defer it onto the
    failure ladder. Three scans audit one change and ask exactly once.
    """
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    asked = _scripted(
        monkeypatch,
        [{"ok": False, "code": "refused", "message": "not authorised", "state": "refused"}],
    )
    rollouts.step_record(server, record["id"], now=1000.0)
    member = _member(rollouts.read_record(record["id"], root), "d_a")
    assert member["state"] == "skipped" and member["code"] == "no_grant", member
    assert float(member["next_ask_at"]) == 0.0, "a parked member has no floor"
    causes = [r.cause for r in server.audit.rows if r.event == rollouts.MEMBER_STATE]
    assert causes == ["capability_denied"], causes
    # Parked means parked: two cadence scans later the member is still asked once.
    rollouts.step_record(server, record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
    rollouts.step_record(server, record["id"], now=1000.0 + 2 * rollouts.CADENCE_S + 2.0)
    assert asked == ["d_a"], asked
    assert rollouts.read_record(record["id"], root)["state"] == "done"


def test_the_failure_ladder_is_consecutive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N2 (round 1): a busy answer between two failures resets the rung."""
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    _scripted(monkeypatch, [_answer("failed", reason="x")])
    rollouts.step_record(server, record["id"], now=1000.0)
    member = _member(rollouts.read_record(record["id"], root), "d_a")
    assert member["fails"] == 1, member
    at = float(member["next_ask_at"]) + 1.0
    _scripted(monkeypatch, [_answer("busy", reason="2 sessions busy")])
    rollouts.step_record(server, record["id"], now=at)
    member = _member(rollouts.read_record(record["id"], root), "d_a")
    assert member["state"] == "deferred" and member["fails"] == 0, member
    at2 = float(member["next_ask_at"]) + 1.0
    _scripted(monkeypatch, [_answer("failed", reason="x")])
    rollouts.step_record(server, record["id"], now=at2)
    member = _member(rollouts.read_record(record["id"], root), "d_a")
    assert member["fails"] == 1, "the consecutive count restarted after the busy answer"
    assert float(member["next_ask_at"]) == at2 + rollouts.FAIL_BACKOFF_S[0], member


def test_cancel_without_a_server_still_lands_its_closing_row(tmp_path: Path) -> None:
    """Q-2 (round 1): the server-less writer is CLOSED, so the row reaches disk."""
    root = tmp_path / "cfg"
    record = _open(root, devices=["d_a"])
    cancelled = rollouts.cancel_record(record["id"], root)  # no server: the CLI path
    assert cancelled is not None and cancelled["state"] == "abandoned"
    audit = rollouts.rollouts_dir(root).parent / "audit.jsonl"
    assert audit.exists(), f"no audit file at {audit}"
    lines = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    rows = [row for row in lines if row.get("event") == rollouts.ROLLOUT_DONE]
    assert rows and rows[-1].get("outcome") == "partial", lines


def test_terminal_records_are_refused_by_the_verbs(tmp_path: Path) -> None:
    """M2 (round 1): a terminal record is refused, not rewritten."""
    from local_operator.network.types import MeshRefusal

    root = tmp_path / "cfg"
    record = _open(root, devices=["d_a"])
    rollouts.cancel_record(record["id"], root)  # active → abandoned
    with pytest.raises(MeshRefusal) as cancelled:
        rollouts.cancel_record(record["id"], root)
    assert cancelled.value.code == "rollout_not_active", cancelled.value.code
    with pytest.raises(MeshRefusal) as resumed:
        rollouts.resume_record(record["id"], root)
    assert resumed.value.code == "rollout_not_active", resumed.value.code
    assert rollouts.read_record(record["id"], root)["state"] == "abandoned"


def test_the_real_parser_builds_and_both_resume_spellings_parse() -> None:
    """B1 (round 1): the REAL CLI parser, built and exercised.

    The mirror-repro (an argparse fragment) proved the shape; only building the
    actual parser catches a propagated `--resume` colliding with the verb's own
    — which made every `lop` invocation die at parser build.
    """
    from local_operator.cli import build_cli_parser

    parser = build_cli_parser()  # must not raise
    verb = parser.parse_args(["network", "update", "--resume", "ro_x"])
    assert getattr(verb, "resume", None) == "ro_x", vars(verb)
    latest = parser.parse_args(["network", "update", "--cancel"])
    assert getattr(latest, "cancel", "x") == "", vars(latest)  # const: newest active
    sibling = parser.parse_args(["config", "list", "--resume", "abc"])
    assert getattr(sibling, "resume", None) == "abc", "propagation still reaches other verbs"


def test_the_slice_registers_one_start_hook_and_no_ops() -> None:
    server = _Server(Path("/nonexistent-rollouts-root"))
    rollouts.install(server)
    assert server.ops == {}, "S3a serves no ops"
    assert list(server.kw.get("on_start", {})) == ["mesh-rollout"], server.kw
    assert not server.kw.get("local_handlers") and not server.kw.get("slow"), server.kw


def test_the_driver_is_dormant_until_a_record_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE S3a CONTRACT: with no record, the driver asks nobody and writes nothing.

    Nothing in production opens a record in this half (the trigger is S3b's;
    ``grep -rn 'open_record' local_operator/`` hits only ``rollouts.py``
    itself), so this cell observes the inert half directly: a live driver on an
    empty root stays silent, and acts the moment — and only when — a record
    exists.
    """
    root = tmp_path / "cfg"
    server = _Server(root)
    monkeypatch.setattr(rollouts, "SCAN_INTERVAL_S", 0.05)
    monkeypatch.setattr(
        meshupdate, "update_peer", lambda *a, **k: pytest.fail("the dormant driver asked a peer")
    )
    driver = rollouts.ensure_driver(server)
    assert driver.name == "mesh-rollout", driver.name
    time.sleep(0.3)
    assert not rollouts.rollouts_dir(root).exists(), "a dormant driver created the rollouts dir"
    assert server.audit.rows == [], server.audit.rows

    asked: list[str] = []
    monkeypatch.setattr(
        meshupdate, "update_peer", lambda s, d, *, rollout="": asked.append(d) or _answer("busy")
    )
    record = _open(root, devices=["d_a"], now=time.time())
    time.sleep(0.4)
    assert asked and asked[0] == "d_a", asked
    assert rollouts.read_record(record["id"], root)["members"][0]["state"] in (
        "deferred",
        "applying",
        "pending",
    )


def test_the_cli_verbs_mutate_the_record(monkeypatch: pytest.MonkeyPatch) -> None:
    # The verbs act on THIS device's config root (root=None), so the record is
    # opened there — the harness isolates it — and no relay is ever touched.
    record = _open(None, devices=["d_a"])
    monkeypatch.setattr(rollouts, "latest_active", lambda: rollouts.read_record(record["id"]))
    monkeypatch.setattr(
        net_cli, "_relay_answer", lambda *a, **k: pytest.fail("--resume/--cancel are local")
    )

    ns = Namespace(resume="", cancel=None, peer="", json=False)
    assert net_cli._cmd_update(ns) == 0
    assert rollouts.read_record(record["id"])["state"] == "active"

    ns = Namespace(resume=None, cancel="", peer="", json=False)
    assert net_cli._cmd_update(ns) == 0
    assert rollouts.read_record(record["id"])["state"] == "abandoned"


def test_a_member_frame_carries_the_rollout_id_into_the_members_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The member's own rows name the rollout (note §6's correlation contract)."""
    server = _Server(tmp_path / "cfg")
    seen: dict[str, Any] = {}

    def _fake_exec(
        server_: Any, link: Any, target: Any, *, rollout: str = "", runner: Any = None
    ) -> dict:
        seen["rollout"] = rollout
        return {
            "state": "busy",
            "code": "",
            "reason": "",
            "method": "",
            "version": "",
            "sessions": {"moved": 0, "kept": 0},
            "updated_at": 0.0,
        }

    monkeypatch.setattr(meshupdate, "_execute_validated", _fake_exec)
    reply = meshupdate.member_execute(
        server,
        None,
        {"target": {"version": "0.9.0", "source_ref": ""}, "rollout": "ro_abc123"},
    )
    assert reply["state"] == "busy"
    assert seen["rollout"] == "ro_abc123", seen
    assert server.audit.rows[-1].detail["rollout"] == "ro_abc123", server.audit.rows
