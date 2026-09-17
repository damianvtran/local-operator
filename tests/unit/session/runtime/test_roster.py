"""The runtime roster: every live runtime on the machine, reachable or not.

This is the half of central reachability that does not depend on any one runtime
answering. Its contract is a bounded answer composed from four independent
sources, and each test here pins one of the ways that contract can be broken:

* a runtime with no discovery record must still be NAMED, with its port, so a
  client can dial it (34 of the reference host's 57 runtimes were in exactly that
  state from the operator's store's point of view);
* a dead pid must never be dialled, and must read ``gone`` rather than
  ``unreachable``;
* a runtime that is alive and not answering must not hold the response open — the
  budget turns it into ``unknown``, and ``unknown`` must never be reported as
  "nothing is there";
* an interface that is attached must be reported from EVIDENCE (a viewer lease, an
  established connection) rather than from the runtime's own opinion of itself;
* and the whole thing is a READER: it removes nothing from the store it reads.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.session.runtime import reclaim
from local_operator.session.runtime.reclaim import Fleet, RuntimeProcess, SocketEvidence
from local_operator.session.runtime.roster import (
    REACHABLE_GONE,
    REACHABLE_LIVE,
    REACHABLE_UNKNOWN,
    REACHABLE_UNREACHABLE,
    build_roster,
)
from local_operator.session.runtime.types import SessionRecord

MINE = os.getpid()
GONE_PID = 999_999  # above the platform's pid space, so signal-0 cannot find it


def proc(pid: int = MINE, *, age_s: float = 3600.0, cpu_s: float = 0.0) -> RuntimeProcess:
    return RuntimeProcess(
        pid=pid,
        parent_pid=1,
        age_s=age_s,
        cpu_s=cpu_s,
        command=f"/usr/bin/python3 -P -m {reclaim.RUNTIME_MODULE}",
    )


def record(pid: int, *, session_id: str = "s1", port: int | None = 5000, **extra):
    payload = {
        "pid": pid,
        "kind": "daemon",
        "session_id": session_id,
        "conversation_name": "synthetic",
        "cwd": "/tmp/synthetic",
        "model_label": "synthetic-model",
        "control_port": port or 0,
        "control_key": "k",
        "heartbeat_at": time.time(),
        "version": "0.56.11",
        "install_root": "/opt/local-operator",
    }
    payload.update(extra)
    return SessionRecord.from_json(payload)


def view(root: Path, **kwargs) -> Fleet:
    kwargs.setdefault("sockets", SocketEvidence(available=False))
    return Fleet(
        root=root,
        records={item.pid: item for item in kwargs.pop("records", ())},
        boots={int(item["pid"]): item for item in kwargs.pop("boots", ())},
        viewers=list(kwargs.pop("viewers", ())),
        sockets=kwargs.pop("sockets"),
        states=kwargs.pop("states", {}),
        own_pids=frozenset(),
    )


def build(root: Path, *, processes, fleet: Fleet, env="", connect=None, **kwargs):
    return build_roster(
        root,
        fleet=fleet,
        processes=processes,
        env_of=lambda pid: env,
        connect=connect if connect is not None else (lambda port, timeout: True),
        **kwargs,
    )


def test_a_recorded_runtime_reports_its_record_and_answers(tmp_path: Path) -> None:
    rows = build(
        tmp_path,
        processes=[proc()],
        fleet=view(tmp_path, records=[record(MINE)], states={MINE: "live"}),
    )
    assert len(rows) == 1
    row = rows[0]
    assert (row.pid, row.session_id, row.port, row.build_version) == (
        MINE,
        "s1",
        5000,
        "0.56.11",
    )
    assert row.state == "live"
    assert row.has_record is True
    assert row.reachability == REACHABLE_LIVE
    assert row.heartbeat_age_s is not None and row.heartbeat_age_s < 5.0
    assert row.install_root == "/opt/local-operator"
    # The verdict shown here is the SWEEP's own, computed by the same function: a
    # runtime with a record is never reclaimable.
    assert row.reclaimable is False
    assert row.reclaim_refusal == reclaim.REFUSAL_RECORD_PRESENT


def test_a_recordless_runtime_is_named_from_the_boot_record_and_the_socket_table(
    tmp_path: Path,
) -> None:
    # THE MEASURED GAP: this runtime was invisible to every reader in the product,
    # because each resolves a runtime through its record. A roster that cannot name
    # it is not a roster.
    rows = build(
        tmp_path,
        processes=[proc()],
        fleet=view(
            tmp_path,
            boots=[{"pid": MINE, "session_id": "orphan", "build_version": "0.56.9"}],
            sockets=SocketEvidence(ports={MINE: 6001}, available=True),
        ),
    )
    row = rows[0]
    assert (row.session_id, row.port, row.build_version, row.has_record) == (
        "orphan",
        6001,
        "0.56.9",
        False,
    )
    assert row.reachability == REACHABLE_LIVE
    # No discovery record means no heartbeat anywhere readable: reported as None,
    # never as "fresh".
    assert row.heartbeat_age_s is None
    assert row.config_root == str(tmp_path)
    assert row.reclaimable is True


def test_a_runtime_with_no_record_at_all_is_named_from_its_own_environment(
    tmp_path: Path,
) -> None:
    # The last resort, and the only way to name a runtime whose config root has been
    # deleted: its own environment.
    env = (
        f"/usr/bin/python3 -P -m {reclaim.RUNTIME_MODULE} "
        f"LOCAL_OPERATOR_CONFIG_DIR={tmp_path} LOP_MOBILE_CHILD_RESUME=fromenv"
    )
    rows = build(tmp_path, processes=[proc()], fleet=view(tmp_path), env=env)
    row = rows[0]
    assert (row.session_id, row.config_root, row.has_boot_record) == (
        "fromenv",
        str(tmp_path),
        False,
    )
    # No port from any source: the row is honest about it, and the reachability is
    # UNKNOWN rather than a dial that cannot work.
    assert row.port is None
    assert row.reachability == REACHABLE_UNKNOWN


def test_a_dead_pid_is_gone_and_is_never_dialled(tmp_path: Path) -> None:
    # A record whose process is gone is exactly the evidence a reader is asking for,
    # so the row stays — and no probe is spent on a pid that is not there. The
    # connect is the one operation that can block, which is why it must not be
    # reached for a corpse.
    dialled: list[int] = []

    def connect(port: int, timeout: float) -> bool:
        dialled.append(port)
        return True

    rows = build(
        tmp_path,
        processes=[],
        fleet=view(
            tmp_path, records=[record(GONE_PID, session_id="dead")], states={GONE_PID: "stale"}
        ),
        connect=connect,
    )
    assert len(rows) == 1
    assert rows[0].reachability == REACHABLE_GONE
    assert rows[0].state == "stale"
    assert rows[0].age_s is None
    # And the reason is NAMED: there is no process to judge, which is why the
    # sweep's ladder never runs for this row.
    assert rows[0].reclaim_refusal == reclaim.REFUSAL_GONE
    assert dialled == []


def test_a_wedged_runtime_that_does_not_answer_reads_unreachable(tmp_path: Path) -> None:
    # Alive, heartbeating long ago, and not accepting: the row must say
    # "unreachable" so a client knows to look elsewhere, not "gone" (which would
    # deny it exists) and not "live" (which would be a dial that hangs).
    stale = record(MINE)
    stale.heartbeat_at = time.time() - 10_000.0
    rows = build(
        tmp_path,
        processes=[proc()],
        fleet=view(tmp_path, records=[stale], states={MINE: "wedged"}),
        connect=lambda port, timeout: False,
    )
    row = rows[0]
    assert row.state == "wedged"
    assert row.reachability == REACHABLE_UNREACHABLE


def test_a_wedged_runtime_does_not_spend_the_whole_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # THE FAILURE THE BOUND EXISTS FOR: one runtime that is alive and not answering.
    # A probe that never returns must land as UNKNOWN for the rows it covered, and
    # the composition must still return.
    slow_rows = [proc(pid=MINE + 1), proc(pid=MINE + 2)]
    monkeypatch.setattr(reclaim.registry, "pid_alive", lambda pid, **kwargs: True)
    stale = [record(pid) for pid in (MINE + 1, MINE + 2)]
    started = time.monotonic()

    def connect(port: int, timeout: float) -> bool:
        time.sleep(0.05)
        return True

    rows = build(
        tmp_path,
        processes=slow_rows,
        fleet=view(tmp_path, records=stale, states={}),
        budget_s=0.001,
        connect=connect,
    )
    assert time.monotonic() - started < 1.0
    assert {row.reachability for row in rows} == {REACHABLE_UNKNOWN}


def test_probe_false_reports_the_inventory_without_dialling(tmp_path: Path) -> None:
    dialled: list[int] = []
    rows = build(
        tmp_path,
        processes=[proc()],
        fleet=view(tmp_path, records=[record(MINE)]),
        probe=False,
        connect=lambda port, timeout: dialled.append(port) or True,
    )
    assert rows[0].port == 5000
    assert rows[0].reachability == REACHABLE_UNKNOWN
    assert dialled == []


def test_attached_is_evidence_not_a_self_report(tmp_path: Path) -> None:
    # A viewer lease names the session; an established connection to the control
    # port is a client that is not a viewer. Either is "an interface is on it".
    rows = build(
        tmp_path,
        processes=[proc()],
        fleet=view(
            tmp_path,
            records=[record(MINE)],
            viewers=[SimpleNamespace(pid=1234, current_session="s1")],
            sockets=SocketEvidence(ports={MINE: 5000}, attached=frozenset(), available=True),
        ),
    )
    assert rows[0].attached is True
    assert rows[0].observers == 1

    other = build(
        tmp_path,
        processes=[proc()],
        fleet=view(
            tmp_path,
            records=[record(MINE)],
            sockets=SocketEvidence(ports={MINE: 5000}, attached=frozenset({5000}), available=True),
        ),
    )
    assert other[0].attached is True
    assert other[0].observers == 0


def test_a_viewer_on_another_session_does_not_attach_this_row(tmp_path: Path) -> None:
    rows = build(
        tmp_path,
        processes=[proc()],
        fleet=view(
            tmp_path,
            records=[record(MINE, session_id="s1")],
            viewers=[SimpleNamespace(pid=1234, current_session="someone-else")],
        ),
    )
    assert rows[0].attached is False
    assert rows[0].observers == 0


def test_the_roster_is_a_reader_and_removes_nothing(tmp_path: Path) -> None:
    # The composition runs against a store it does not own, so the one mutation it
    # could make — reaping a dead pid's record — must not happen. The record is the
    # evidence a later "why did this die" question is answered from.
    mobile = tmp_path / "run" / "mobile"
    mobile.mkdir(parents=True)
    path = mobile / f"{GONE_PID}.json"
    path.write_text(json.dumps(record(GONE_PID).to_json()))
    fleet = reclaim.read_fleet(tmp_path, sockets=SocketEvidence(available=False))
    rows = build_roster(tmp_path, fleet=fleet, processes=[], env_of=lambda pid: "", probe=False)
    assert [row.pid for row in rows] == [GONE_PID]
    assert path.exists()


def test_rows_are_ordered_by_pid_so_a_poller_can_diff_them(tmp_path: Path) -> None:
    rows = build_roster(
        tmp_path,
        fleet=view(tmp_path, records=[record(GONE_PID), record(MINE)]),
        processes=[proc()],
        env_of=lambda pid: "",
        probe=False,
    )
    assert [row.pid for row in rows] == sorted([GONE_PID, MINE])


def test_a_probe_that_raises_is_a_verdict_not_an_error(tmp_path: Path) -> None:
    # The roster is a diagnostic surface: an exception from one connect must not
    # take the response with it. It reads as "did not answer" for that row.
    def connect(port: int, timeout: float) -> bool:
        raise OSError("no route")

    rows = build(
        tmp_path,
        processes=[proc()],
        fleet=view(tmp_path, records=[record(MINE)]),
        connect=connect,
    )
    assert rows[0].reachability == REACHABLE_UNKNOWN


def test_the_row_json_carries_every_required_field(tmp_path: Path) -> None:
    # Deliberately NOT parametrized over the live pid: a node id containing
    # ``os.getpid()`` differs between xdist workers, and xdist refuses a run whose
    # workers collected different tests.
    rows = build(tmp_path, processes=[proc()], fleet=view(tmp_path, records=[record(MINE)]))
    payload = rows[0].to_json()
    assert {
        "session_id",
        "pid",
        "port",
        "build_version",
        "attached",
        "observers",
        "heartbeat_age_s",
        "reachability",
    } <= set(payload)
    assert json.dumps(payload)  # JSON-safe: no dataclass or Path leaks onto the wire
