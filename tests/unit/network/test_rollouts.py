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
import logging
import os
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import meshupdate, rollouts, types
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)


def _live(record_id: str, root: Path | None = None) -> dict[str, Any]:
    """The record, asserted present — the cells always just wrote it.

    ``read_record`` returns Optional (a reader over a directory that may
    not exist); in a cell the record is the setup, so the None case is a
    test bug the assert catches (round 3, R3-1: CI's pyright runs the
    test files too).
    """
    record = rollouts.read_record(record_id, root)
    assert record is not None
    return record


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
        #: The relay's own stop flag, present so the driver's stand-down guard
        #: (R6-1) reads a real Event here, exactly as it does on a RelayServer.
        self._stop = threading.Event()

    def register_ops(self, ops: Any, **kw: Any) -> None:
        self.ops = ops
        self.kw = kw


def _open(
    root: Path | None,
    *,
    devices: list[str],  # noqa: F811 — the fixture import above shares the name
    network: str = "n1",
    now: float = 1000.0,
) -> dict[str, Any]:
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
    assert _live(first["id"], root)["state"] == "superseded"
    assert _live(second["id"], root)["state"] == "active"

    # A DIFFERENT network's active record is NOT superseded: the record is per
    # (origin, network) (note §3, RU §6.3).
    other = _open(root, devices=["d_c"], network="n2", now=2500.0)
    assert _live(other["id"], root)["state"] == "active"


def test_the_window_lapse_closes_the_record_with_a_sentence(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"], now=1000.0)
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + rollouts.WINDOW_S + 1.0)
    lapsed = _live(record["id"], root)
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
    # Round 1, D1/D2: the open line is an ask in speech a person uses, and the
    # count reads "(s)" — the repo's convention — even at one.
    assert "will ask 2 peer(s) to move to this build" in line, line
    one = _open(root, devices=["d_c"], network="n2")
    assert "will ask 1 peer(s) to move to this build" in rollouts.summary(one), rollouts.summary(
        one
    )


def test_a_concurrent_open_does_not_retire_a_newer_record(tmp_path: Path) -> None:
    """M1 (round 1): an OLDER racer's scan must not supersede a newer record."""
    root = tmp_path / "cfg"
    newer = _open(root, devices=["d_a"], now=2000.0)
    older = _open(root, devices=["d_a"], now=1000.0)  # a stale racer's open
    assert (
        _live(newer["id"], root)["state"] == "active"
    ), "the newer record was retired by an older racer"
    # Both stay active — the conservative outcome — and no pass sees zero actives.
    assert _live(older["id"], root)["state"] == "active"


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

    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    assert asked == ["d_a", "d_b"], asked  # in snapshot order, one at a time
    after = _live(record["id"], root)
    assert _member(after, "d_a")["state"] == "done"
    assert _member(after, "d_b")["state"] == "already_on_target"
    assert after["state"] == "done", "all members terminal must close the record"

    # A done member is never re-applied: a second scan asks nobody new.
    called: list[str] = []
    monkeypatch.setattr(
        meshupdate, "update_peer", lambda *a, **k: called.append("x") or _answer("busy")
    )
    rollouts.step_record(cast(Any, server), record["id"], now=2000.0)
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

    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    member = _member(_live(record["id"], root), "d_a")
    assert member["state"] == "deferred" and member["code"] == "busy", member
    assert member["next_ask_at"] == 1000.0 + rollouts.CADENCE_S, member

    # ONE row for the state CHANGE, not one per poll: the cadence re-asks below
    # (inside the floor: not at all; past it: again, same answer) with no
    # further rows.
    rows = [r for r in server.audit.rows if r.event == rollouts.MEMBER_STATE]
    assert len(rows) == 1 and rows[0].cause == "busy", rows

    rollouts.step_record(cast(Any, server), record["id"], now=1100.0)
    assert asked == ["d_a"], "a member inside its floor must not be asked"

    _scripted(monkeypatch, [_answer("busy", reason="1 session is busy")])
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
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
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    member = _member(_live(record["id"], root), "d_a")
    assert member["state"] == "failed" and member["fails"] == 1, member
    assert member["next_ask_at"] == 1000.0 + rollouts.FAIL_BACKOFF_S[0], member

    _scripted(monkeypatch, [_answer("failed", reason="installer exited 1")])
    rollouts.step_record(
        cast(Any, server), record["id"], now=1000.0 + rollouts.FAIL_BACKOFF_S[0] + 1.0
    )
    member = _member(_live(record["id"], root), "d_a")
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
        rollouts.step_record(cast(Any, server), record["id"], now=moment)
    assert asked == ["d_a"], "an ungranted member is parked: ONE refusal per rollout"
    final = _live(record["id"], root)
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
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    member = _member(_live(record["id"], root), "d_a")
    assert member["state"] == "unreachable" and member["code"] == "unreachable", member
    assert member["next_ask_at"] == 1000.0 + rollouts.CADENCE_S, member

    _scripted(monkeypatch, [_answer("done", version="0.99.0")])
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
    final = _live(record["id"], root)
    assert _member(final, "d_a")["state"] == "done" and final["state"] == "done"


def test_a_drop_after_applying_re_probes_on_the_bounded_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    seen_states: list[str] = []

    def _fake(server_: Any, device: str, *, rollout: str = "") -> dict[str, Any]:
        live = _live(record["id"], root)
        seen_states.append(str(_member(live, device)["state"]))
        return answers.pop(0)

    answers = [{"ok": False, "code": "no_answer", "message": ""}]
    monkeypatch.setattr(meshupdate, "update_peer", _fake)
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    assert seen_states == ["applying"], seen_states  # the in-flight marker is written first
    member = _member(_live(record["id"], root), "d_a")
    assert member["state"] == "deferred" and member["code"] == "no_answer", member
    assert member["next_ask_at"] == 1000.0 + rollouts.DROPPED_AFTER_APPLY_RETRY_S, member

    answers.append(_answer("already_on_target"))
    rollouts.step_record(
        cast(Any, server), record["id"], now=1000.0 + rollouts.DROPPED_AFTER_APPLY_RETRY_S + 1.0
    )
    assert _member(_live(record["id"], root), "d_a")["state"] == "already_on_target"


def test_resume_clears_the_floors_and_restarts_the_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a", "d_b"])
    _scripted(monkeypatch, [_answer("busy"), {"ok": False, "code": "unreachable", "message": ""}])
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    parked = _live(record["id"], root)
    assert _member(parked, "d_a")["next_ask_at"] > 0 and _member(parked, "d_b")["next_ask_at"] > 0

    resumed = rollouts.resume_record(record["id"], root)
    assert resumed is not None
    assert all(_member(resumed, d)["next_ask_at"] == 0.0 for d in ("d_a", "d_b"))

    asked = _scripted(
        monkeypatch, [_answer("done", version="0.99.0"), _answer("done", version="0.99.0")]
    )
    rollouts.step_record(
        cast(Any, server), record["id"], now=1000.0
    )  # SAME moment: floors are gone
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
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    assert asked == ["d_busy", "d_offline", "d_idle"], asked  # serial, snapshot order
    mid = _live(record["id"], root)
    assert mid["state"] == "active", "two members are still owed a roll"
    assert (
        _member(mid, "d_busy")["state"] == "deferred" and _member(mid, "d_busy")["code"] == "busy"
    )
    assert _member(mid, "d_offline")["state"] == "unreachable"
    assert _member(mid, "d_idle")["state"] == "done"

    # Nothing is due before the cadence; at it, both deferred members are asked
    # once more and both land — "caught on next contact".
    asked2 = _scripted(monkeypatch, [])
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + rollouts.CADENCE_S - 1.0)
    assert asked2 == [], "a member inside its floor must not be asked"
    asked3 = _scripted(
        monkeypatch, [_answer("done", version="0.99.0"), _answer("already_on_target")]
    )
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
    assert asked3 == ["d_busy", "d_offline"], asked3
    final = _live(record["id"], root)
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
    cancelled = rollouts.cancel_record(record["id"], root, server=cast(Any, server))
    assert cancelled is not None
    final = _live(record["id"], root)
    assert final["state"] == "abandoned"
    rows = [r for r in server.audit.rows if r.event == rollouts.ROLLOUT_DONE]
    assert rows and rows[-1].outcome == "partial", rows

    called: list[str] = []
    monkeypatch.setattr(
        meshupdate, "update_peer", lambda *a, **k: called.append("x") or _answer("busy")
    )
    rollouts.step_record(cast(Any, server), record["id"], now=2000.0)
    assert called == [], "an abandoned record must not be driven"


def test_a_fully_terminal_pass_closes_with_ok_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    _scripted(monkeypatch, [_answer("already_on_target")])
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    rows = [r for r in server.audit.rows if r.event == rollouts.ROLLOUT_DONE]
    assert rows and rows[-1].outcome == "ok" and rows[-1].detail["counts"] == "1 of 1 moved", rows


# ---------------------------------------------------------------------------
# Dormancy, registration, CLI — the S3a contract
# ---------------------------------------------------------------------------


def test_a_codeless_refusal_is_recorded_without_claiming_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-2 (round 2): the wire's codeless "not now" is never a claim of why.

    The live answer is the error frame's own shape — ``{"ok": false, "code":
    "refused", "message": …}``, NO ``state`` key (the nit the previous cell
    carried; the live error-frame path never sets one). It must record
    ``refused``/``refused`` with NO cause, retry on the cadence, and keep ONE
    row across ≥3 ticks (the rollout-scoped dedupe).
    """
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    live = {"ok": False, "code": "refused", "message": "not authorised"}
    asked: list[str] = []

    def _fake(server_: Any, device: str, *, rollout: str = "") -> dict[str, Any]:
        asked.append(device)
        return dict(live)

    monkeypatch.setattr(meshupdate, "update_peer", _fake)
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    member = _member(_live(record["id"], root), "d_a")
    assert member["state"] == "refused" and member["code"] == "refused", member
    assert float(member["next_ask_at"]) == 1000.0 + rollouts.CADENCE_S, member
    rows = [r for r in server.audit.rows if r.event == rollouts.MEMBER_STATE]
    assert len(rows) == 1 and rows[0].cause == "", rows  # non-claiming: no capability_denied

    # Inside the floor: no ask. At the cadence: asked again; still ONE row.
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + rollouts.CADENCE_S - 1.0)
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + 2 * rollouts.CADENCE_S + 2.0)
    assert asked == ["d_a", "d_a", "d_a"], asked
    rows = [r for r in server.audit.rows if r.event == rollouts.MEMBER_STATE]
    assert len(rows) == 1, rows
    assert (
        _live(record["id"], root)["state"] == "active"
    ), "a codeless refusal is not terminal — the window is the bound"


def test_the_failure_ladder_is_consecutive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N2 (round 1): a busy answer between two failures resets the rung."""
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])
    _scripted(monkeypatch, [_answer("failed", reason="x")])
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    member = _member(_live(record["id"], root), "d_a")
    assert member["fails"] == 1, member
    at = float(member["next_ask_at"]) + 1.0
    _scripted(monkeypatch, [_answer("busy", reason="2 sessions busy")])
    rollouts.step_record(cast(Any, server), record["id"], now=at)
    member = _member(_live(record["id"], root), "d_a")
    assert member["state"] == "deferred" and member["fails"] == 0, member
    at2 = float(member["next_ask_at"]) + 1.0
    _scripted(monkeypatch, [_answer("failed", reason="x")])
    rollouts.step_record(cast(Any, server), record["id"], now=at2)
    member = _member(_live(record["id"], root), "d_a")
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
    assert _live(record["id"], root)["state"] == "abandoned"


def test_the_real_parser_builds_and_both_resume_spellings_parse() -> None:
    """B1 (round 1): the REAL CLI parser, built and exercised.

    The mirror-repro (an argparse fragment) proved the shape; only building the
    actual parser catches a propagated `--resume` colliding with the verb's own
    — which made every `lop` invocation die at parser build.
    """
    from local_operator.cli import build_cli_parser

    parser = build_cli_parser()  # must not raise
    verb = parser.parse_args(["network", "update", "--resume", "ro_x"])
    assert getattr(verb, "rollout_resume", None) == "ro_x", vars(verb)
    latest = parser.parse_args(["network", "update", "--cancel"])
    assert getattr(latest, "cancel", "x") == "", vars(latest)  # const: newest active
    sibling = parser.parse_args(["config", "list", "--resume", "abc"])
    assert getattr(sibling, "resume", None) == "abc", "propagation still reaches other verbs"


def test_main_defers_the_verbs_resume_to_the_verb(tmp_path: Path) -> None:
    """Q-R2-1 (round 2): the REAL `main()` must not eat the verb's `--resume`.

    `main()` resolves a session `--resume` BEFORE dispatching, so a verb-level
    `--resume` used to arrive as "no session 'ro_missing' to resume". The verb
    owns its flag through its own dest; this drives the real entry point in a
    subprocess on an isolated root and asserts the ROLLOUT sentence comes back
    — reversing the dest rename turns this red on the "no session" assertion.
    """
    import subprocess

    root = tmp_path / "home"
    root.mkdir()
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "TERM", "LANG", "LC_ALL", "TMPDIR")
    }
    env.update({"HOME": str(root), "LOCAL_OPERATOR_CONFIG_DIR": str(root / ".local-operator")})
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from local_operator.cli import main; sys.exit(main())",
            "network",
            "update",
            "--resume",
            "ro_missing",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    text = proc.stdout + proc.stderr
    assert "no rollout record 'ro_missing'" in text, text
    assert "no session" not in text, text
    assert proc.returncode != 0, proc.returncode


def test_the_slice_registers_start_and_stop_hooks_and_no_ops() -> None:
    server = _Server(Path("/nonexistent-rollouts-root"))
    rollouts.install(cast(Any, server))
    assert server.ops == {}, "S3a serves no ops"
    assert list(server.kw.get("on_start", {})) == ["mesh-rollout"], server.kw
    assert list(server.kw.get("on_stop", {})) == ["mesh-rollout"], server.kw
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

    BOTH WAITS ARE ON EVENTS, NEVER ON CLOCKS (QA O-R6-1, then F1 — the old
    fixed ``sleep(0.3)``/``sleep(0.4)`` windows failed once in each round of
    #2204's QA; they were a bet on machine load, and the measured write->ask
    tail at load ~100 was p50 75 ms / p95 389 ms / max 557 ms, 1 in 30 over
    the 0.4 s window). The silence assertions must follow scans that actually
    RAN — counted through the real ``tick``, so a starved thread cannot pass
    them vacuously — and the ask is waited on the fake's own publication. The
    deadlines are hang backstops, not the assertions.
    """
    root = tmp_path / "cfg"
    server = _Server(root)
    monkeypatch.setattr(rollouts, "SCAN_INTERVAL_S", 0.05)
    monkeypatch.setattr(
        meshupdate, "update_peer", lambda *a, **k: pytest.fail("the dormant driver asked a peer")
    )
    scans = threading.Event()
    seen = 0
    real_tick = rollouts.tick

    def _counting_tick(srv: Any) -> None:
        nonlocal seen
        real_tick(srv)
        seen += 1
        if seen >= 2:
            scans.set()

    monkeypatch.setattr(rollouts, "tick", _counting_tick)
    driver = rollouts.ensure_driver(cast(Any, server))
    assert driver is not None
    try:
        assert driver.name == "mesh-rollout", driver.name
        assert scans.wait(timeout=10.0), "the driver never completed two scans"
        assert not rollouts.rollouts_dir(root).exists(), "a dormant driver created the rollouts dir"
        assert server.audit.rows == [], server.audit.rows

        replied = threading.Event()
        asked: list[str] = []

        def _ask(_srv: Any, device: str, *, rollout: str = "") -> dict[str, Any]:
            asked.append(device)
            replied.set()
            return _answer("busy")

        monkeypatch.setattr(meshupdate, "update_peer", _ask)
        record = _open(root, devices=["d_a"], now=time.time())
        assert replied.wait(timeout=10.0), "the driver never asked the new record's member"
        assert asked[0] == "d_a", asked
        assert _live(record["id"], root)["members"][0]["state"] in (
            "deferred",
            "applying",
            "pending",
        )
    finally:
        # A cell that starts the driver reaps it (round 5, Q-R5-1): a leaked
        # loop kept ticking into sibling tests in the same worker.
        rollouts.stop_driver(cast(Any, server))
    assert not driver.is_alive(), "the driver outlived its test's reaping"


def test_the_driver_dies_at_stop_and_rides_no_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q-R5-1: the driver stops on demand, and its cadence never rides ``time.sleep``.

    Both halves are the fix for the CI red this cell exists to keep red-proof:
    ``test_attention_lock_contention``'s WAL cell patches ``time.sleep``
    process-globally, and a driver that called ``time.sleep(15.0)`` while
    co-resident was captured by it. The loop waits on its own stop event, so
    (a) ``stop_driver`` lands within one boundary, and (b) a patched
    ``time.sleep`` sees no driver boundary AT ALL — asserted against a
    deliberately unique interval so no other thread's sleep can alias it.
    """
    root = tmp_path / "cfg"
    server = _Server(root)
    unique = 0.123456
    monkeypatch.setattr(rollouts, "SCAN_INTERVAL_S", unique)
    calls: list[float] = []
    real_sleep = time.sleep

    def _record(seconds: float) -> None:
        calls.append(seconds)
        real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", _record)
    driver = rollouts.ensure_driver(cast(Any, server))
    assert driver is not None
    real_sleep(0.4)
    assert driver.is_alive(), "the driver never came up"
    rollouts.stop_driver(cast(Any, server))
    assert not driver.is_alive(), "the driver outlived stop_driver"
    assert id(server) not in rollouts._DRIVERS
    assert unique not in calls, f"the driver's cadence rode time.sleep {calls.count(unique)} times"
    # The reap is complete: a fresh ensure starts a NEW thread that also stops.
    again = rollouts.ensure_driver(cast(Any, server))
    assert again is not None
    assert again is not driver
    real_sleep(0.2)
    rollouts.stop_driver(cast(Any, server))
    assert not again.is_alive(), "the second driver outlived stop_driver"


def test_a_start_hook_landing_after_stop_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6-1: start()'s hook thread can land after stop() ran the stop hooks.

    The relay's start hooks are fire-and-forget daemon threads, so the
    interleaving is legal: ``stop()`` runs ``stop_driver`` (nothing to reap
    yet) and the hook then calls ``ensure_driver``. Without the stand-down
    guard that starts a driver no reaper will ever visit — the relay is
    stopped and the stop hooks have already run. This cell pins the seam: a
    server whose ``_stop`` is already set must start nothing.
    """
    root = tmp_path / "cfg"
    server = _Server(root)
    server._stop.set()
    monkeypatch.setattr(rollouts, "SCAN_INTERVAL_S", 0.05)
    try:
        driver = rollouts.ensure_driver(cast(Any, server))
        assert driver is None, "a start hook landing after stop started a driver"
        assert id(server) not in rollouts._DRIVERS
    finally:
        rollouts.stop_driver(cast(Any, server))


def test_the_cli_verbs_mutate_the_record(monkeypatch: pytest.MonkeyPatch) -> None:
    # The verbs act on THIS device's config root (root=None), so the record is
    # opened there — the harness isolates it — and no relay is ever touched.
    record = _open(None, devices=["d_a"])
    monkeypatch.setattr(rollouts, "latest_active", lambda: _live(record["id"]))
    monkeypatch.setattr(
        net_cli, "_relay_answer", lambda *a, **k: pytest.fail("--resume/--cancel are local")
    )

    ns = Namespace(rollout_resume="", cancel=None, peer="", json=False)
    assert net_cli._cmd_update(ns) == 0
    assert _live(record["id"])["state"] == "active"

    ns = Namespace(rollout_resume=None, cancel="", peer="", json=False)
    assert net_cli._cmd_update(ns) == 0
    assert _live(record["id"])["state"] == "abandoned"


def test_the_receipts_count_codeless_refused_members(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R3-3 (round 3): receipts use ``member_terminal``, not bare state membership.

    A codeless-refused member is non-terminal and WILL be re-asked, so a
    receipt that counted it as parked undercounted what the verb just did
    (the reviewer's probe: "0 member(s) will be re-asked").
    """
    record = _open(None, devices=["d_a"])
    rollouts.mutate_record(
        record["id"],
        None,
        lambda rec: _member(rec, "d_a").update({"state": "refused", "code": "refused"}) or True,
    )
    monkeypatch.setattr(rollouts, "latest_active", lambda: _live(record["id"]))
    monkeypatch.setattr(net_cli, "_relay_answer", lambda *a, **k: pytest.fail("local verbs only"))

    ns = Namespace(rollout_resume="", cancel=None, peer="", json=False)
    assert net_cli._cmd_update(ns) == 0
    out = capsys.readouterr().out
    assert "1 member(s) will be re-asked" in out, out

    # The cancel receipt counts it as REMAINED for the same reason.
    monkeypatch.setattr(rollouts, "latest_active", lambda: _live(record["id"]))
    ns = Namespace(rollout_resume=None, cancel="", peer="", json=False)
    assert net_cli._cmd_update(ns) == 0
    out = capsys.readouterr().out
    assert "1 member(s) remained" in out, out


def test_a_member_frame_carries_the_rollout_id_into_the_members_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The member's own rows name the rollout (note §6's correlation contract)."""
    server = _Server(tmp_path / "cfg")
    seen: dict[str, Any] = {}

    def _fake_exec(
        server_: Any, link: Any, target: Any, *, rollout: str = "", runner: Any = None
    ) -> dict[str, Any]:
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
        cast(Any, server),
        None,
        {"target": {"version": "0.9.0", "source_ref": ""}, "rollout": "ro_abc123"},
    )
    assert reply["state"] == "busy"
    assert seen["rollout"] == "ro_abc123", seen
    assert server.audit.rows[-1].detail["rollout"] == "ro_abc123", server.audit.rows


# ---------------------------------------------------------------------------
# The trigger (S3b, note §3/§5/§6): a completed update opens the records
# ---------------------------------------------------------------------------


def _seed_mesh(
    root: Path,
    *,
    members: list[tuple[str, types.MemberKind]] = [("d_a", "device")],
    network_id: str = "n_trigger00000000000000000000001",
    name: str = "home-net",
    self_id: str = "d_origin",
    trust: types.TrustState = "active",
) -> None:
    """One network record on disk, the shape the trigger's walk reads."""
    from local_operator.network import store

    record = types.NetworkRecord(
        network_id=network_id,
        name=name,
        trust=trust,
        self_device_id=self_id,
        members=[
            types.MemberRecord(device_id=self_id, name="origin"),
            *[
                types.MemberRecord(device_id=device, name=device, kind=kind)
                for device, kind in members
            ],
        ],
    )
    store.save(record, root)


def _set_policy(root: Path, value: str) -> None:
    """Write the key the way ``/settings`` does — the one writer ``ConfigManager`` has.

    Hand-writing YAML is what this helper exists NOT to do: an unquoted ``off``
    parses as a boolean, which is the very ambiguity ``rollout_policy`` maps —
    pinned separately, through the raw-write cell below.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    settings_io.write_setting(
        ConfigManager(root),
        settings_io.BY_KEY["network.rollout_on_update"],
        value,
        notify=False,
    )


def _set_policy_raw(root: Path, text: str) -> None:
    """Write ``values.network.rollout_on_update`` verbatim, as a hand edit would."""
    from local_operator.config import CONFIG_FILE_NAME

    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / CONFIG_FILE_NAME).write_text(
        f"values:\n  network:\n    rollout_on_update: {text}\n", encoding="utf-8"
    )


def _audit_rows(root: Path, event: str) -> list[dict[str, Any]]:
    audit = rollouts.rollouts_dir(root).parent / "audit.jsonl"
    if not audit.exists():
        return []
    lines = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    return [row for row in lines if row.get("event") == event]


def test_the_trigger_opens_a_record_naming_its_authority(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    _seed_mesh(root, members=[("d_a", "device"), ("d_b", "device")])
    result = rollouts.open_after_update("0.27.0", "0.28.0", via="update", root=root, now=1000.0)
    assert result.failure == "", result.failure
    assert len(result.lines) == 1, result.lines
    record = rollouts.latest_active(root=root)
    assert record is not None, "the trigger opened no record"
    assert record["state"] == "active"
    assert record["network"] == "n_trigger00000000000000000000001"
    assert record["origin"] == "d_origin"
    assert record["target"] == {"version": "0.28.0", "source_ref": ""}, record["target"]
    # WHO DECIDED rides the record in the same step that opens it (note §3/§6).
    assert record["authority"] == {
        "authority": "standing_instruction",
        "policy": "auto",
        "via": "update",
    }, record.get("authority")
    states = [(m["device"], m["state"], m["code"]) for m in record["members"]]
    assert states == [("d_a", "pending", ""), ("d_b", "pending", "")], states
    # RU §7.3: the summary is the operator's first sight of the roll, and it
    # NAMES the queued set.
    assert "queued: d_a, d_b" in result.lines[0], result.lines
    # The durable opening row (note §6): one per record, carrying the decision.
    rows = _audit_rows(root, rollouts.ROLLOUT_STARTED)
    assert len(rows) == 1, rows
    assert rows[0]["network_id"] == record["network"]
    detail = rows[0]["detail"]
    assert detail["rollout"] == record["id"]
    assert detail["authority"] == "standing_instruction"
    assert detail["policy"] == "auto"
    assert detail["via"] == "update"
    assert detail["target"] == "0.28.0"
    assert detail["from"] == "0.27.0"
    assert detail["members"] == "2"


def test_the_trigger_opens_one_record_per_network_in_store_order(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    _seed_mesh(root, network_id="n_trigger00000000000000000000001", name="a-net")
    _seed_mesh(
        root,
        network_id="n_trigger00000000000000000000002",
        name="b-net",
        members=[("d_c", "device")],
    )
    result = rollouts.open_after_update("0.27.0", "0.28.0", via="snapshot", root=root, now=1000.0)
    assert len(result.lines) == 2, result.lines
    records = [rollouts.read_record(p.stem, root) for p in rollouts.active_record_paths(root)]
    nets = sorted(str(r["network"]) for r in records if r is not None)
    assert nets == [
        "n_trigger00000000000000000000001",
        "n_trigger00000000000000000000002",
    ], nets
    assert len(_audit_rows(root, rollouts.ROLLOUT_STARTED)) == 2


def test_the_trigger_fires_only_for_a_strictly_newer_parseable_move(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    _seed_mesh(root)
    for before, after in (
        ("0.28.0", "0.28.0"),  # a snapshot that keeps the last tag's version
        ("0.28.0", "0.27.0"),  # a deliberate rollback
        ("", "0.28.0"),  # no before to compare
        ("0.28.0", "main"),  # an unparsable after
    ):
        result = rollouts.open_after_update(before, after, via="update", root=root, now=1000.0)
        assert result == rollouts.RolloutOpen(), (before, after, result)
    assert rollouts.latest_active(root=root) is None
    assert not rollouts.rollouts_dir(root).exists(), "a fired-nothing run created the dir"
    assert _audit_rows(root, rollouts.ROLLOUT_STARTED) == []
    assert _audit_rows(root, rollouts.ROLLOUT_SUPPRESSED) == []


def test_no_roll_suppresses_with_one_row_and_no_record(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    _seed_mesh(root)
    result = rollouts.open_after_update(
        "0.27.0", "0.28.0", via="update", no_roll=True, root=root, now=1000.0
    )
    assert result.lines == () and result.failure == ""
    assert rollouts.latest_active(root=root) is None
    assert not rollouts.rollouts_dir(root).exists()
    rows = _audit_rows(root, rollouts.ROLLOUT_SUPPRESSED)
    assert len(rows) == 1, rows
    assert rows[0]["cause"] == "policy"
    assert rows[0]["detail"] == {"via": "update", "policy": "auto", "flag": "--no-roll"}, rows[0]


def test_policy_off_suppresses_and_the_row_says_which_half_stopped_it(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    _seed_mesh(root)
    _set_policy(root, "off")
    assert rollouts.rollout_policy(root) == "off"
    result = rollouts.open_after_update("0.27.0", "0.28.0", via="snapshot", root=root, now=1000.0)
    assert result.lines == () and result.failure == ""
    assert rollouts.latest_active(root=root) is None
    rows = _audit_rows(root, rollouts.ROLLOUT_SUPPRESSED)
    assert len(rows) == 1, rows
    assert rows[0]["detail"] == {"via": "snapshot", "policy": "off", "flag": ""}, rows[0]


def test_rollout_policy_defaults_and_an_unrecognised_value_reads_as_auto(tmp_path: Path) -> None:
    root = tmp_path / "cfg"
    assert rollouts.rollout_policy(root) == "auto", "an absent store is the default policy"
    _set_policy(root, "off")
    assert rollouts.rollout_policy(root) == "off"
    # The WRITE path refuses anything else (the registry's schema), and a
    # hand-edited typo must not read as a silent off either.
    with pytest.raises(ValueError, match="auto, off"):
        _set_policy(root, "banana")
    _set_policy_raw(root, "banana")
    assert rollouts.rollout_policy(root) == "auto"
    # A hand-typed unquoted `off`/`on` arrives as a YAML boolean: it reads as
    # the obvious word, and never as the unsafe fallback (auto) for the word
    # `off`.
    _set_policy_raw(root, "off")
    assert rollouts.rollout_policy(root) == "off"
    _set_policy_raw(root, "on")
    assert rollouts.rollout_policy(root) == "auto"


def test_a_pool_member_rides_skipped_unsupported_kind_and_the_summary_says_so(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cfg"
    _seed_mesh(root, members=[("d_a", "device"), ("d_pod", "pool")])
    result = rollouts.open_after_update("0.27.0", "0.28.0", via="update", root=root, now=1000.0)
    record = rollouts.latest_active(root=root)
    assert record is not None
    assert [(m["device"], m["state"], m["code"]) for m in record["members"]] == [
        ("d_a", "pending", ""),
        ("d_pod", "skipped", "unsupported_kind"),
    ], record["members"]
    assert "skipped (unsupported_kind): d_pod" in result.lines[0], result.lines


def test_a_network_with_no_other_members_and_an_untrusted_network_open_nothing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cfg"
    _seed_mesh(root, members=[])  # this device alone
    _seed_mesh(
        root,
        network_id="n_trigger00000000000000000000002",
        name="b-net",
        members=[("d_c", "device")],
        trust="untrusted",
    )
    result = rollouts.open_after_update("0.27.0", "0.28.0", via="update", root=root, now=1000.0)
    assert result == rollouts.RolloutOpen(), result
    assert rollouts.latest_active(root=root) is None
    assert _audit_rows(root, rollouts.ROLLOUT_STARTED) == []


def test_a_hook_failure_is_one_sentence_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cfg"
    _seed_mesh(root)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no space left on device")

    monkeypatch.setattr(rollouts, "open_record", _boom)
    result = rollouts.open_after_update("0.27.0", "0.28.0", via="tui", root=root, now=1000.0)
    assert result.lines == ()
    assert result.failure == (
        "updated, but the mesh roll could not be queued: no space left on device — "
        "run `lop network update` to roll by hand"
    ), result.failure
    assert rollouts.latest_active(root=root) is None


def test_a_partial_multi_network_open_keeps_the_lines_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-1 (round 1): a failure on network N loses nothing already queued.

    Every opened record is on disk and will be driven, so its summary line
    must reach the front ends WITH the failure sentence — otherwise the
    operator is told "could not be queued" for a network where it was.
    """
    root = tmp_path / "cfg"
    _seed_mesh(root, network_id="n_trigger00000000000000000000001", name="a-net")
    _seed_mesh(
        root,
        network_id="n_trigger00000000000000000000002",
        name="b-net",
        members=[("d_c", "device")],
    )
    real = rollouts.open_record
    calls: list[str] = []

    def flaky(*, network: str, **kwargs: Any) -> Any:
        calls.append(network)
        if len(calls) == 2:
            raise OSError("no space left on device")
        return real(network=network, **kwargs)

    monkeypatch.setattr(rollouts, "open_record", flaky)
    result = rollouts.open_after_update("0.27.0", "0.28.0", via="update", root=root, now=1000.0)

    assert len(calls) == 2, calls
    assert len(result.lines) == 1, result.lines
    assert "will ask 1 peer(s) to move to this build" in result.lines[0], result.lines
    assert result.failure is not None and "could not be queued" in result.failure, result.failure
    # The first network's record is on disk — the driver will find and resume it.
    assert rollouts.latest_active(root=root) is not None


def test_the_closing_nudge_is_withheld_for_an_unrollable_remainder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2 (round 1): the nudge must not name a remainder no re-run can move.

    A ``skipped (unsupported_kind)`` member — the pool shape this slice's open
    seeds — can never be rolled by a later ``lop network update``, so the
    sentence would promise an action that cannot work. The skip stays NAMED in
    the sets; only the nudge is withheld, and a rollable remainder still gets
    it (the other direction below).
    """
    root = tmp_path / "cfg"
    server = _Server(root)
    _seed_mesh(root, members=[("d_a", "device"), ("d_pod", "pool")])
    rollouts.open_after_update("0.27.0", "0.28.0", via="update", root=root, now=1000.0)
    record = rollouts.latest_active(root=root)
    assert record is not None
    _scripted(monkeypatch, [_answer("done", version="0.28.0")])
    rollouts.step_record(cast(Any, server), record["id"], now=2000.0)
    closed = _live(record["id"], root)
    assert closed["state"] == "done", closed
    rows = [r for r in server.audit.rows if r.event == rollouts.ROLLOUT_DONE]
    assert rows and "1 of 2 moved" in rows[-1].detail["counts"], rows
    assert "no longer active" not in rows[-1].detail["counts"], rows[-1].detail
    assert "skipped (unsupported_kind): d_pod" in rollouts.summary(closed), rollouts.summary(closed)

    # The other direction: a rollable remainder keeps the nudge — re-running
    # really can move a deferred member.
    other = _open(root, devices=["d_e"], network="n2")
    _member(other, "d_e").update({"state": "deferred", "code": "busy"})
    other["state"] = "done"
    assert "no longer active" in rollouts.summary(other), rollouts.summary(other)


def test_the_closing_nudge_tracks_which_refusals_a_re_run_can_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-1 (round 2): a rule-refused remainder is as unrollable as an R1-2 skip.

    ``refused/target_not_published`` and ``refused/editable_install`` are
    origin-side rule facts evaluated at ask time, terminal for the rollout via
    ``member_terminal`` — so a re-run from the same origin cannot move the
    member, and the nudge would be the same promise-that-cannot-work R1-2
    fixed for the skip codes. The member stays NAMED in the sets; only the
    nudge is withheld.

    The retained direction is pinned because the SPLIT is the point: a
    CODELESS ``refused`` (code ``refused``) is the wire's "not now", self-heals,
    and a re-run really can move it — its nudge stays.
    """
    root = tmp_path / "cfg"
    server = _Server(root)
    _seed_mesh(root, members=[("d_a", "device"), ("d_b", "device")])
    rollouts.open_after_update("0.27.0", "0.28.0", via="update", root=root, now=1000.0)
    record = rollouts.latest_active(root=root)
    assert record is not None
    _scripted(
        monkeypatch,
        [
            _answer("done", version="0.28.0"),
            _answer(
                "refused", code="editable_install", reason="editable installs cannot be rolled"
            ),
        ],
    )
    rollouts.step_record(cast(Any, server), record["id"], now=2000.0)
    closed = _live(record["id"], root)
    assert closed["state"] == "done", closed
    assert "refused: d_b" in rollouts.summary(closed), rollouts.summary(closed)
    rows = [r for r in server.audit.rows if r.event == rollouts.ROLLOUT_DONE]
    assert rows and "1 of 2 moved" in rows[-1].detail["counts"], rows
    assert "no longer active" not in rows[-1].detail["counts"], rows[-1].detail

    # The other rule code is exempt the same way...
    other = _open(root, devices=["d_c"], network="n2")
    _member(other, "d_c").update({"state": "refused", "code": "target_not_published"})
    other["state"] = "done"
    assert "no longer active" not in rollouts.summary(other), rollouts.summary(other)

    # ...while the codeless refusal KEEPS the nudge: it self-heals, so a re-run
    # really can move it.
    codeless = _open(root, devices=["d_d"], network="n3")
    _member(codeless, "d_d").update({"state": "refused", "code": "refused"})
    codeless["state"] = "done"
    assert "no longer active" in rollouts.summary(codeless), rollouts.summary(codeless)


def test_the_switch_halts_a_policy_driven_roll_at_the_next_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Note §5: the thread re-reads the key before every ask; off halts, on resumes."""
    root = tmp_path / "cfg"
    server = _Server(root)
    record = rollouts.open_record(
        network="n1",
        origin="d_origin",
        target={"version": "0.28.0", "source_ref": ""},
        members=[{"device": "d_a", "name": "a"}, {"device": "d_b", "name": "b"}],
        authority={"authority": "standing_instruction", "policy": "auto", "via": "update"},
        root=root,
        now=1000.0,
    )
    asked = _scripted(
        monkeypatch,
        [_answer("busy", reason="2 sessions"), _answer("busy"), _answer("busy"), _answer("done")],
    )
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    assert asked == ["d_a", "d_b"], asked  # one step asks every member that is due

    _set_policy(root, "off")
    # Both members sit inside their 300 s cadence floors here: this run reaches
    # no boundary, so it says nothing about the switch — the next one does.
    rollouts.step_record(cast(Any, server), record["id"], now=1000.5)
    assert asked == ["d_a", "d_b"], asked
    # Both floors have passed, d_b is due and non-terminal: with the switch off
    # the pass stops before its ask — no other command needed (note §5).
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
    assert asked == ["d_a", "d_b"], "the pass asked a member while the policy was off"

    # Nothing was lost: flipping it back on resumes at the same boundary.
    _set_policy(root, "auto")
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0 + rollouts.CADENCE_S + 1.0)
    assert asked == ["d_a", "d_b", "d_a", "d_b"], asked
    assert _member(_live(record["id"], root), "d_b")["state"] == "done"


def test_a_record_without_authority_is_not_the_switchs_to_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a ``standing_instruction`` record answers to the policy key."""
    root = tmp_path / "cfg"
    server = _Server(root)
    record = _open(root, devices=["d_a"])  # the tests' own open: no authority object
    _set_policy(root, "off")
    asked = _scripted(monkeypatch, [_answer("done")])
    rollouts.step_record(cast(Any, server), record["id"], now=1000.0)
    assert asked == ["d_a"], "a non-policy record was halted by the switch"


# ---------------------------------------------------------------------------
# The activation proof (S3b): a real update exit opens the record, and the real
# serial pass drives a real member over a real link.
# ---------------------------------------------------------------------------


def test_a_real_update_exit_opens_the_roll_and_the_pass_moves_a_real_member(
    devices: Any,  # noqa: F811 — the relay suite's fixture, imported above
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both halves of the slice, wired: ``update_command`` queues the roll; the
    driver rolls a granted member over loopback TCP and the record closes.

    THE RIG IS THE MESH SUITE'S OWN (``test_relay_e2e``'s ``devices``): two
    real relays on loopback, real identities, a real pairing ceremony, a real
    grant on the member's own row, real ``net_update`` frames. The origin is
    ``server_b`` (the joiner), the member ``server_a`` — the orientation
    ``test_mesh_update`` uses. The seams are the two install boundaries S1's
    own cells already write through (``install_build``, plus
    ``current_member_version``/``check_published``) and the pass tail; every
    frame between the two ends is the production path.

    ``update_command`` runs against the origin's root with only its install
    seam (``perform_upgrade``) and its services stage replaced: its exits, the
    hook, the record write and the summary print are the production paths this
    slice adds (the summary is asserted where the exit is driven, in
    ``tests/unit/test_update.py``).
    """
    from unittest.mock import patch

    from local_operator import update as update_mod
    from tests.unit.network.test_mesh_update import (
        _concrete_member,
        _grant,
        _set_peer_endpoints,
        _stub_pass,
    )

    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    _set_peer_endpoints(server_b, record, server_a.identity.device_id, [f"{host}:{port}"])
    _grant(server_a, record, server_b.identity.device_id)
    state = _concrete_member(monkeypatch, current="0.0.1")
    monkeypatch.setattr(meshupdate, "busy_sessions", lambda root: (0, ""))
    monkeypatch.setattr(meshupdate, "check_published", lambda version: (True, ""))
    monkeypatch.setattr(
        meshupdate, "origin_target", lambda: {"version": "0.99.0", "source_ref": ""}
    )
    _stub_pass(monkeypatch)
    calls: list[str] = []

    def _install(version: str, *, runner: Any = None) -> str:
        calls.append(version)
        state["current"] = version  # the install MOVED the member's build
        return meshupdate.METHOD_GENERATION

    monkeypatch.setattr(meshupdate, "install_build", _install)

    # The origin's own update — a strictly newer move through the real exit.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    with (
        patch.object(
            update_mod,
            "check_latest",
            return_value=update_mod.VersionCheck(installed="0.98.0", latest="0.99.0", behind=True),
        ),
        patch.object(update_mod, "install_kind", return_value=update_mod.InstallKind.UV_TOOL),
        patch.object(update_mod, "is_git_snapshot", return_value=False),
        patch.object(update_mod, "perform_upgrade", return_value="0.99.0"),
        patch.object(update_mod, "_services_stage", return_value=None),
    ):
        assert update_mod.update_command(check=False) == 0

    trigger = rollouts.latest_active(root=server_b.root)
    assert trigger is not None, "the update exit opened no record"
    assert trigger["state"] == "active"
    assert trigger["authority"] == {
        "authority": "standing_instruction",
        "policy": "auto",
        "via": "update",
    }, trigger.get("authority")
    assert [m["device"] for m in trigger["members"]] == [server_a.identity.device_id]

    # The pass runs where it runs in production — one step of the relay's
    # driver — against the real member over the real link.
    rollouts.step_record(server_b, str(trigger["id"]))
    after = _live(str(trigger["id"]), server_b.root)
    member = _member(after, server_a.identity.device_id)
    assert member["state"] == "done", member
    assert member["version"] == "0.99.0", member
    assert calls == ["0.99.0"], calls
    assert after["state"] == "done", after["state"]


class _DriverThatIgnoresTheStop:
    """A driver whose join returns at the deadline while it keeps running."""

    def __init__(self) -> None:
        self.stop_called = False
        self.join_timeout: float | None = None

    def stop(self) -> None:
        self.stop_called = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout

    def is_alive(self) -> bool:
        return True


class _DriverThatStops:
    """The clean arm: the join returns and the thread is already gone."""

    def __init__(self) -> None:
        self.stop_called = False
        self.join_timeout: float | None = None

    def stop(self) -> None:
        self.stop_called = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout

    def is_alive(self) -> bool:
        return False


def _rollout_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == rollouts.logger.name and record.levelno == logging.WARNING
    ]


def test_stop_driver_reports_a_thread_that_will_not_stop_and_keeps_it_registered(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A timed-out join must not erase the only path to the surviving scan.

    ``stop_driver`` popped the registry entry BEFORE the join, and
    ``Thread.join(timeout)`` returns nothing either way, so a driver that
    ignored its stop became unreachable — and ``ensure_driver`` could start a
    second scan beside it. The entry stays until the thread is known stopped;
    the warning is what names the survivor.
    """
    server = _Server(tmp_path / "cfg")
    driver = _DriverThatIgnoresTheStop()
    rollouts._DRIVERS[id(server)] = cast(Any, driver)
    try:
        with caplog.at_level(logging.WARNING, logger=rollouts.logger.name):
            rollouts.stop_driver(cast(Any, server), timeout=1.5)
        assert driver.stop_called, "the stop request must still go out first"
        assert driver.join_timeout == 1.5, "stop_driver must wait for its timeout"
        assert rollouts._DRIVERS[id(server)] is driver, "the survivor must stay reachable"
        warnings = _rollout_warnings(caplog)
        assert len(warnings) == 1, caplog.text
        message = warnings[0].getMessage()
        assert "1.5s" in message, message
        assert "still running" in message, message
    finally:
        rollouts._DRIVERS.pop(id(server), None)


def test_stop_driver_removes_the_entry_and_stays_silent_when_the_thread_stopped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The normal path: the driver stopped, the entry goes, no warning."""
    server = _Server(tmp_path / "cfg")
    driver = _DriverThatStops()
    rollouts._DRIVERS[id(server)] = cast(Any, driver)
    try:
        with caplog.at_level(logging.WARNING, logger=rollouts.logger.name):
            rollouts.stop_driver(cast(Any, server), timeout=1.5)
        assert driver.stop_called
        assert driver.join_timeout == 1.5
        assert id(server) not in rollouts._DRIVERS
        assert _rollout_warnings(caplog) == [], caplog.text
    finally:
        rollouts._DRIVERS.pop(id(server), None)
