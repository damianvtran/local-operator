"""The external residency sweep: the census, the refusal ladder, and the signal.

The sweep exists because the MEASURED population had no other way out — on
2026-09-17 the reference host ran 57 session runtimes, 34 of which no reader in
the product could name. Its whole risk is the mirror image of its purpose: it
holds a SIGTERM, so every test here is about a REFUSAL. A sweep that ends one
runtime it should not have is worse than a sweep that reclaims nothing.

Three layers are pinned separately, because they are separately wrong-able:

* the EVIDENCE (the census, the socket table, the config root a process reports),
  which must be read from the processes themselves rather than from a file they
  may have stopped writing;
* the VERDICT, one refusal at a time, including the two that exist to protect a
  runtime a person is using;
* the PASS, which must not signal anything on its first look (the confirm window),
  must refuse a candidate that spends CPU inside that window, and must be
  scope-able so a sweep run inside a sandbox cannot act on another store.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.session.runtime import reclaim
from local_operator.session.runtime.reclaim import (
    BUSY_CPU_FLOOR_S,
    CONFIRM_S,
    REFUSAL_BUSY_CPU,
    REFUSAL_FOREIGN_ROOT,
    REFUSAL_OBSERVED,
    REFUSAL_RECORD_PRESENT,
    REFUSAL_SELF,
    REFUSAL_UNATTRIBUTABLE,
    REFUSAL_UNCONFIRMED,
    REFUSAL_YOUNG,
    Fleet,
    RuntimeProcess,
    Sightings,
    SocketEvidence,
    ancestor_pids,
    config_root_of,
    etime_seconds,
    reclaim_runtimes,
    runtime_processes,
    session_id_of,
    socket_evidence,
    verdict,
)
from local_operator.session.runtime.types import SessionRecord

NOW = 1_800_000_000.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def proc(pid: int = 4242, *, age_s: float = 3600.0, cpu_s: float = 0.0, ppid: int = 1):
    return RuntimeProcess(
        pid=pid,
        parent_pid=ppid,
        age_s=age_s,
        cpu_s=cpu_s,
        command=f"/usr/bin/python3 -P -m {reclaim.RUNTIME_MODULE}",
    )


def record(pid: int = 4242, *, session_id: str = "s1", port: int = 5000, **extra):
    payload = {
        "pid": pid,
        "kind": "daemon",
        "session_id": session_id,
        "conversation_name": "synthetic",
        "cwd": "/tmp/synthetic",
        "model_label": "synthetic-model",
        "control_port": port,
        "control_key": "k",
        "heartbeat_at": NOW - 1.0,
        "version": "0.56.11",
    }
    payload.update(extra)
    return SessionRecord.from_json(payload)


def viewer(pid: int = 77, *, session: str = "s1"):
    return SimpleNamespace(pid=pid, current_session=session)


def fleet(
    root: Path,
    *,
    records=(),
    boots=(),
    viewers=(),
    sockets: SocketEvidence | None = None,
    own=(),
    states=None,
) -> Fleet:
    return Fleet(
        root=root,
        records={item.pid: item for item in records},
        boots={int(item["pid"]): item for item in boots},
        viewers=list(viewers),
        sockets=sockets if sockets is not None else SocketEvidence(),
        states=states or {},
        own_pids=frozenset(own),
    )


def env_text(root: str, *, session: str = "", home: str | None = None) -> str:
    parts = [f"/usr/bin/python3 -P -m {reclaim.RUNTIME_MODULE}"]
    parts.append(f"LOCAL_OPERATOR_CONFIG_DIR={root}")
    parts.append(f"HOME={home if home is not None else root + '-home'}")
    if session:
        parts.append(f"LOP_MOBILE_CHILD_RESUME={session}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# The evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1-10:04:33", 122673.0),
        ("10:04:33", 36273.0),
        ("05:12", 312.0),
        ("0:00.04", 0.04),
        ("", 0.0),
        ("garbage", 0.0),
    ],
)
def test_etime_seconds_reads_ps_elapsed_and_cpu_columns(text: str, expected: float) -> None:
    assert etime_seconds(text) == pytest.approx(expected)


def test_census_matches_the_spawn_contract_and_ignores_the_searcher() -> None:
    # THE ONE MISIDENTIFICATION THAT WOULD MAKE A SWEEP SIGNAL A STRANGER: a
    # `grep` for the module name is not a runtime, and neither is this module's own
    # test process. Only `-m <module>` in that order is the spawn contract.
    def run(command, timeout_s):
        assert command[0] == "ps"
        return (
            "  123      1  1-10:04:33  0:00.10 /usr/bin/python3 -P -m "
            "local_operator.session.runtime.process\n"
            "  456      1     05:00  0:00.02 /bin/zsh -c grep "
            "local_operator.session.runtime.process\n"
            "  789      1     00:10  0:00.00 /usr/bin/python3 -m pytest tests/unit/session -q\n"
            "garbage line\n"
        )

    found = runtime_processes(run=run)
    assert [item.pid for item in found] == [123]
    assert found[0].parent_pid == 1
    assert found[0].age_s == pytest.approx(122673.0)


def test_socket_table_gives_a_recordless_runtime_a_port_and_an_attach() -> None:
    # ONE machine-wide lsof, two facts: which pid listens where, and which of those
    # ports something is connected to right now. The second is the sweep's strongest
    # evidence that an interface is attached, and it is evidence a record cannot
    # supply for a runtime that published none.
    def run(command, timeout_s):
        assert command == ["lsof", "-nP", "-iTCP"]
        return (
            "COMMAND     PID   USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME\n"
            "python3.1 55021 damian   13u  IPv4 0x8ff026ecd85e5328      0t0  TCP"
            " 127.0.0.1:59919 (LISTEN)\n"
            "python3.1 55021 damian   14u  IPv4 0x1111111111111111      0t0  TCP"
            " 127.0.0.1:59919->127.0.0.1:51234 (ESTABLISHED)\n"
            "rapportd  55022 damian   10u  IPv6 0xfcdbe27acac2b1d5      0t0  TCP"
            " [fe80:17::a490:620c:30ea:711d]:1024 (LISTEN)\n"
            "Arc       55023 damian  121u  IPv6 0xf03de76db9ef62b5      0t0  TCP"
            " 192.168.0.153:56386->192.178.192.95:443 (ESTABLISHED)\n"
        )

    evidence = socket_evidence(run=run)
    assert evidence.available is True
    assert evidence.ports == {55021: 59919, 55022: 1024}
    # Only the ESTABLISHED row whose LOCAL side is a LISTEN port counts: a client's
    # own outbound connection says nothing about a runtime being watched.
    assert evidence.attached == frozenset({59919})


def test_a_missing_lsof_is_unknown_and_never_read_as_no_connections() -> None:
    evidence = socket_evidence(run=lambda command, timeout_s: "")
    assert evidence.available is False
    assert evidence.ports == {}
    assert evidence.attached == frozenset()


def test_config_root_comes_from_the_process_itself() -> None:
    # The root a sweep must decide against is the one the RUNTIME uses, not the one
    # the sweep is running under: a sweep in the operator's store must not treat
    # another store's runtime as its own.
    assert config_root_of(env_text("/tmp/iso/cfg")) == "/tmp/iso/cfg"
    assert config_root_of("python -P -m x HOME=/tmp/iso") == "/tmp/iso/.local-operator"
    assert config_root_of("python -P -m x") == ""
    assert session_id_of(env_text("/tmp/iso/cfg", session="abc123")) == "abc123"
    assert session_id_of(env_text("/tmp/iso/cfg")) == ""


def test_own_process_and_its_ancestors_are_never_candidates() -> None:
    # A sweep can be run FROM a session (`lop sessions reclaim` typed into a TUI, or
    # an agent's own shell). Signalling an ancestor would end the session that asked.
    chain = ancestor_pids()
    assert os.getpid() in chain
    assert len(chain) >= 1


# ---------------------------------------------------------------------------
# The verdict — one refusal at a time
# ---------------------------------------------------------------------------


def _recordless_env(tmp_path: Path):
    """A ``env_of`` that answers for exactly one root, as the census would."""

    def env_of(pid: int) -> str:
        return env_text(str(tmp_path))

    return env_of


def test_our_own_process_is_refused(tmp_path: Path) -> None:
    view = fleet(tmp_path, own=(4242,))
    assert verdict(proc(), view, env_of=_recordless_env(tmp_path)).refusal == REFUSAL_SELF


def test_a_runtime_too_young_to_have_published_is_refused(tmp_path: Path) -> None:
    # Covers the whole construction window of a spawn whose record is not out yet:
    # measured at 2.1 s warm, and the engage deadline a cold runtime is allowed is
    # 180 s.
    view = fleet(tmp_path)
    item = verdict(proc(age_s=9.0), view, env_of=_recordless_env(tmp_path), min_age_s=300.0)
    assert item.refusal == REFUSAL_YOUNG
    assert "9s" in item.detail


def test_an_unreadable_config_root_is_refused_rather_than_assumed(tmp_path: Path) -> None:
    item = verdict(proc(), fleet(tmp_path), env_of=lambda pid: "python -P -m x")
    assert item.refusal == REFUSAL_UNATTRIBUTABLE


def test_a_runtime_of_another_live_root_is_refused(tmp_path: Path) -> None:
    # The measured population, exactly: 36 of the reference host's record-less
    # runtimes were runtimes of SIBLING stores (QA cells under /tmp) whose own
    # records were heartbeat-fresh. They are reachable from their own root; a sweep
    # scoped to one store has no business ending them.
    other = tmp_path / "other"
    other.mkdir()
    item = verdict(proc(), fleet(tmp_path), env_of=lambda pid: env_text(str(other)))
    assert item.refusal == REFUSAL_FOREIGN_ROOT


def test_a_deleted_root_is_in_scope(tmp_path: Path) -> None:
    # The one class with no far side: every client, viewer, wake and supervisor of
    # that root resolves through a path that is gone.
    gone = tmp_path / "deleted-root"
    item = verdict(proc(), fleet(tmp_path), env_of=lambda pid: env_text(str(gone)))
    assert item.may_end() is True


def test_a_published_record_is_a_refusal_whatever_its_heartbeat_says(tmp_path: Path) -> None:
    # A heartbeat is authored by the runtime's own event loop, so a quiet one covers
    # a frozen process AND a healthy one starved by a long turn (measured false
    # positives: 105.8 s and 205.8 s) — which is why `wedged_runtime` refuses to end
    # a runtime on that evidence and so does this.
    view = fleet(tmp_path, records=[record()], states={4242: "wedged"})
    item = verdict(proc(), view, env_of=_recordless_env(tmp_path))
    assert item.refusal == REFUSAL_RECORD_PRESENT
    assert item.detail == "discovery record published"


def test_a_viewer_showing_the_session_is_a_refusal(tmp_path: Path) -> None:
    view = fleet(
        tmp_path,
        boots=[{"pid": 4242, "session_id": "s1"}],
        viewers=[viewer(session="s1")],
    )
    item = verdict(proc(), view, env_of=_recordless_env(tmp_path))
    assert item.refusal == REFUSAL_OBSERVED
    assert "77" in item.detail


def test_an_established_connection_on_its_control_port_is_a_refusal(tmp_path: Path) -> None:
    # The runtime an attach holds may have published nothing, so the viewer lease is
    # not enough on its own: the socket table catches a client that is connected but
    # not a viewer (a `lop send`, an attach in progress).
    view = fleet(
        tmp_path,
        sockets=SocketEvidence(ports={4242: 5050}, attached=frozenset({5050}), available=True),
    )
    item = verdict(proc(), view, env_of=_recordless_env(tmp_path))
    assert item.refusal == REFUSAL_OBSERVED
    assert "5050" in item.detail
    assert item.port == 5050


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def test_first_look_never_signals_and_second_confirms(tmp_path: Path) -> None:
    view = fleet(tmp_path)
    kill: list[tuple[int, int]] = []
    sightings = Sightings()

    def env_of(pid: int) -> str:
        return env_text(str(tmp_path))

    first = reclaim_runtimes(
        tmp_path,
        apply=True,
        sightings=sightings,
        processes=[proc()],
        env_of=env_of,
        fleet=view,
        kill=lambda pid, sig: kill.append((pid, sig)),
        now=NOW,
    )
    assert first.reclaimed == []
    assert [item.refusal for item in first.refused] == [REFUSAL_UNCONFIRMED]
    assert kill == []
    # THE SECOND LOOK IS THE DECISION. Same memory, the window elapsed: now the
    # signal goes out, and it is SIGTERM — never SIGKILL, which has no handler and
    # would destroy a turn in flight.
    second = reclaim_runtimes(
        tmp_path,
        apply=True,
        sightings=sightings,
        processes=[proc()],
        env_of=env_of,
        fleet=view,
        kill=lambda pid, sig: kill.append((pid, sig)),
        now=NOW + CONFIRM_S,
    )
    assert [item.process.pid for item in second.signalled] == [4242]
    assert kill == [(4242, signal.SIGTERM)]
    assert "1 reclaimed" in second.summary()


def test_a_dry_run_decides_but_never_signals(tmp_path: Path) -> None:
    sightings = Sightings()
    kill: list[tuple[int, int]] = []
    for now in (NOW, NOW + CONFIRM_S):
        report = reclaim_runtimes(
            tmp_path,
            apply=False,
            sightings=sightings,
            processes=[proc()],
            env_of=_recordless_env(tmp_path),
            fleet=fleet(tmp_path),
            kill=lambda pid, sig: kill.append((pid, sig)),
            now=now,
        )
    assert [item.process.pid for item in report.reclaimed] == [4242]
    assert kill == []
    assert "would reclaim (dry run)" in report.summary()


def test_a_candidate_that_spends_cpu_inside_the_window_is_refused_again(tmp_path: Path) -> None:
    # A ONE-WAY TEST: it can only ever refuse, because a runtime waiting on a model
    # API call is quiet and low CPU proves nothing. It exists so the pass never ends
    # a runtime that is visibly doing work.
    sightings = Sightings()
    kill: list[tuple[int, int]] = []
    for process, now in ((proc(cpu_s=0.0), NOW), (proc(cpu_s=30.0), NOW + CONFIRM_S)):
        report = reclaim_runtimes(
            tmp_path,
            apply=True,
            sightings=sightings,
            processes=[process],
            env_of=_recordless_env(tmp_path),
            fleet=fleet(tmp_path),
            kill=lambda pid, sig: kill.append((pid, sig)),
            now=now,
        )
    assert [item.refusal for item in report.refused] == [REFUSAL_BUSY_CPU]
    assert report.reclaimed == []
    assert kill == []


def test_a_quiet_runtime_over_a_short_window_is_confirmed(tmp_path: Path) -> None:
    # The floor: a window shortened by a caller must not turn one scheduler tick of
    # CPU into a refusal. Asserted through the public threshold rather than a literal.
    sightings = Sightings()
    assert BUSY_CPU_FLOOR_S >= 1.0
    for process, now in ((proc(cpu_s=0.0), NOW), (proc(cpu_s=0.5), NOW + 5.0)):
        report = reclaim_runtimes(
            tmp_path,
            apply=True,
            sightings=sightings,
            processes=[process],
            env_of=_recordless_env(tmp_path),
            fleet=fleet(tmp_path),
            confirm_s=5.0,
            kill=lambda pid, sig: None,
            now=now,
        )
    assert [item.process.pid for item in report.signalled] == [4242]


def test_a_recorded_runtime_is_never_signalled_even_after_the_window(tmp_path: Path) -> None:
    sightings = Sightings()
    kill: list[tuple[int, int]] = []
    view = fleet(tmp_path, records=[record()])
    for now in (NOW, NOW + 10 * CONFIRM_S):
        report = reclaim_runtimes(
            tmp_path,
            apply=True,
            sightings=sightings,
            processes=[proc()],
            env_of=_recordless_env(tmp_path),
            fleet=view,
            kill=lambda pid, sig: kill.append((pid, sig)),
            now=now,
        )
    assert kill == []
    assert report.refusals() == {REFUSAL_RECORD_PRESENT: 1}


def test_named_roots_narrow_the_pass_to_exactly_those(tmp_path: Path) -> None:
    # The seam the evidence harness and the tests drive, so a sweep run inside a
    # sandbox cannot act on another store's runtimes however the ladder reads.
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    kill: list[tuple[int, int]] = []

    def env_of(pid: int) -> str:
        return env_text(str(inside if pid == 1 else outside))

    sightings = Sightings()
    for now in (NOW, NOW + CONFIRM_S):
        report = reclaim_runtimes(
            tmp_path,
            apply=True,
            sightings=sightings,
            processes=[proc(pid=1), proc(pid=2)],
            env_of=env_of,
            fleet=fleet(tmp_path),
            roots=[inside],
            kill=lambda pid, sig: kill.append((pid, sig)),
            now=now,
        )
    assert [item.process.pid for item in report.signalled] == [1]
    assert kill == [(1, signal.SIGTERM)]


def test_a_signalled_runtime_that_leaves_is_reported_gone(tmp_path: Path, monkeypatch) -> None:
    sightings = Sightings()
    view = fleet(tmp_path)
    for now in (NOW, NOW + CONFIRM_S):
        reclaim_runtimes(
            tmp_path,
            apply=True,
            sightings=sightings,
            processes=[proc()],
            env_of=_recordless_env(tmp_path),
            fleet=view,
            kill=lambda pid, sig: None,
            now=now,
        )
    # The wait polls the real liveness call, so the test drives it: a runtime that
    # is gone on the next probe is reported as exited, which is what the CLI's
    # "gone" rows and the supervisor's log line are built from.
    monkeypatch.setattr(reclaim.registry, "pid_alive", lambda pid, **kwargs: False)
    report = reclaim_runtimes(
        tmp_path,
        apply=True,
        sightings=sightings,
        processes=[proc()],
        env_of=_recordless_env(tmp_path),
        fleet=view,
        kill=lambda pid, sig: None,
        now=NOW + 2 * CONFIRM_S,
        wait_s=1.0,
        sleep=lambda seconds: None,
    )
    assert len(report.signalled) == 1
    assert report.exited == report.signalled


def test_sightings_stay_bounded_to_the_live_census() -> None:
    sightings = Sightings()
    for pid in range(1, 11):
        sightings.confirm(reclaim.Verdict(process=proc(pid=pid), config_root="/tmp"), now=NOW)
    sightings.forget([1, 2])
    assert set(sightings._seen) == {1, 2}


# ---------------------------------------------------------------------------
# read_fleet — the reader's read
# ---------------------------------------------------------------------------


def test_read_fleet_reads_three_namespaces_and_reaps_nothing(tmp_path: Path) -> None:
    # A sweep that reaped a stale record as a side effect of deciding would be the
    # process that destroyed the evidence for the death it was looking at.
    (tmp_path / "run" / "mobile").mkdir(parents=True)
    (tmp_path / "run" / "host").mkdir(parents=True)
    (tmp_path / "run" / "viewers").mkdir(parents=True)
    live = record(pid=4242, session_id="live")
    dead = record(pid=999999, session_id="dead")
    (tmp_path / "run" / "mobile" / "4242.json").write_text(json.dumps(live.to_json()))
    (tmp_path / "run" / "mobile" / "999999.json").write_text(json.dumps(dead.to_json()))
    (tmp_path / "run" / "host" / "4242.json").write_text(
        json.dumps({"pid": 4242, "session_id": "live", "started_at": time.time()})
    )
    (tmp_path / "run" / "viewers" / "77.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "surface": "tui",
                "control_port": 1,
                "control_key": "k",
                "current_session": "live",
                "heartbeat_at": time.time(),
                "focused_at": time.time(),
            }
        )
    )

    view = reclaim.read_fleet(tmp_path, sockets=SocketEvidence(available=False))
    assert set(view.records) == {4242, 999999}
    assert view.states[999999] == "stale"
    assert 4242 in view.boots
    assert [item.current_session for item in view.viewers] == ["live"]
    assert (tmp_path / "run" / "mobile" / "999999.json").exists()


def test_the_pass_reads_the_socket_table_when_the_caller_gives_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # THE EVIDENCE A PASS MUST NOT SKIP. One of the two refusals nothing else can
    # produce — a runtime with no record, no viewer lease and a client holding its
    # control port — is read off the machine's socket table, so a pass whose default
    # was "no socket evidence" would have that refusal silently unreachable. This is
    # the call site the supervisor uses.
    seen: list[str] = []

    def fake_evidence(**kwargs):
        seen.append("read")
        return SocketEvidence(available=False)

    monkeypatch.setattr(reclaim, "socket_evidence", fake_evidence)
    report = reclaim_runtimes(
        tmp_path,
        apply=False,
        processes=[proc()],
        env_of=_recordless_env(tmp_path),
        now=NOW,
    )
    assert seen == ["read"]
    assert report.sockets_available is False


def test_the_summary_reports_the_window_the_pass_used(tmp_path: Path) -> None:
    report = reclaim_runtimes(
        tmp_path,
        apply=False,
        processes=[],
        env_of=_recordless_env(tmp_path),
        fleet=fleet(tmp_path),
        confirm_s=7.0,
        now=NOW,
    )
    assert "7s confirm window" in report.summary()


def test_the_batch_env_reader_is_one_fork_and_keys_by_pid() -> None:
    # The fleet path: one `ps -Eww -eo pid=,command=` instead of a fork per runtime.
    calls: list[list[str]] = []

    def run(command, timeout_s):
        calls.append(list(command))
        return (
            " 4242 /usr/bin/python3 -P -m local_operator.session.runtime.process "
            "LOCAL_OPERATOR_CONFIG_DIR=/tmp/one HOME=/tmp/one-home\n"
            " 4243 /bin/zsh\n"
            "broken line without a pid\n"
        )

    envs = reclaim.process_envs(run=run)
    assert len(calls) == 1
    assert set(envs) == {4242, 4243}
    assert config_root_of(envs[4242]) == "/tmp/one"
    assert config_root_of(envs[4243]) == ""
