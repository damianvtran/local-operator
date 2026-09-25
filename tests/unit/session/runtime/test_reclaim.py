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

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.session.runtime import reclaim
from local_operator.session.runtime.reclaim import (
    BUSY_CPU_FLOOR_S,
    CONFIRM_S,
    MIN_ACTIONABLE_CONFIRM_S,
    REFUSAL_BUSY_CPU,
    REFUSAL_CHANGED,
    REFUSAL_FOREIGN_ROOT,
    REFUSAL_OBSERVED,
    REFUSAL_RECORD_PRESENT,
    REFUSAL_SELF,
    REFUSAL_SOCKETS_UNKNOWN,
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
    parse_process_row,
    process_row,
    reclaim_runtimes,
    record_file,
    runtime_processes,
    session_id_of,
    socket_evidence,
    verdict,
)
from local_operator.session.runtime.types import RUN_DIRNAME, SessionRecord

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
) -> Fleet:
    # ``available=True`` BY DEFAULT, and it is the caller's job to say otherwise.
    # A fleet with no socket evidence is no longer a weaker verdict — it is a
    # REFUSAL (``REFUSAL_SOCKETS_UNKNOWN``), because a sweep that acts while its
    # strongest refusal is unevaluated is the posture QA round 1 (Q5) measured. The
    # default here therefore means "the table was read and nothing is attached"; a
    # test that means "lsof failed" passes ``SocketEvidence()`` explicitly.
    return Fleet(
        root=root,
        records={item.pid: item for item in records},
        boots={int(item["pid"]): item for item in boots},
        viewers=list(viewers),
        sockets=sockets if sockets is not None else SocketEvidence(available=True),
        own_pids=frozenset(own),
    )


def reread(pid: int, *, age_s: float = 3600.0, cpu_s: float = 0.0, root: str | Path = ""):
    """The SIGNAL-TIME re-read's seam: one row, with the environment appended to it.

    ``reclaim_runtimes`` re-reads the candidate's single row immediately before
    signalling it, in one fork carrying the process's own environment — ``ps -Eww``
    on macOS/BSD, ``ps -ww`` plus a ``/proc/<pid>/environ`` read on Linux, where
    ``-E`` is not an option (:func:`pid_environment`) — and refuses on any change. A
    test that drives a synthetic census must answer for that read too, which is what
    this is: ``runtime_processes``' row shape, with ``LOCAL_OPERATOR_CONFIG_DIR``
    appended the way both spellings append it.
    """
    command = f"/usr/bin/python3 -P -m {reclaim.RUNTIME_MODULE}"
    if root:
        command += f" LOCAL_OPERATOR_CONFIG_DIR={root}"
    return RuntimeProcess(pid=pid, parent_pid=1, age_s=age_s, cpu_s=cpu_s, command=command)


def env_text(root: str, *, session: str = "", home: str | None = None) -> str:
    parts = [f"/usr/bin/python3 -P -m {reclaim.RUNTIME_MODULE}"]
    parts.append(f"LOCAL_OPERATOR_CONFIG_DIR={root}")
    parts.append(f"HOME={home if home is not None else root + '-home'}")
    if session:
        parts.append(f"LOP_MOBILE_CHILD_RESUME={session}")
    return " ".join(parts)


#: The width ``ps`` cuts to when it cannot read a display width and its stdout is not
#: a terminal — i.e. the width every caller of this module's census gets on Linux.
#: ``ps(1)`` refuses to promise the number ("the output width is undefined (it may be
#: 80, unlimited, determined by the TERM variable, and so on)"), and 80 is the value
#: the Linux CI shard measured, twice, as the cut that cost a shard.
NARROW_WIDTH = 80

#: The interpreter an ``argv[0]`` here is forged with, at a real CI path length: a
#: runner's checkout, its ``.venv`` and ``python``. It is what makes the row below
#: long enough for the narrow cut to matter, and it is deliberately a path that is NOT
#: this host's — the row must be shaped like the one Linux prints, not like macOS's.
LONG_INTERPRETER = "/home/runner/work/local-operator/local-operator/.venv/bin/python"


@contextmanager
def live_contract_row() -> Iterator[tuple[int, str]]:
    """A REAL process wearing the spawn contract, and the REAL row a census reads for it.

    The row's SHAPE is the whole subject: a Linux ``ps`` truncates the last column to
    the display width, so the row that loses its ``-m <module>`` pair is the row whose
    command column is long — which a synthetic one-line ``ps`` output cannot show,
    because it never crosses the cut. So this forks a real child with a real long
    ``argv[0]`` (the kernel stores ``argv[0]`` verbatim, the trick
    ``test_launch_arbitration`` uses for the same reason) running ``sys.executable -c``
    — it never imports this package, and it is reaped before the caller's assertions.

    Yields ``(pid, row)``, where ``row`` is the line the census's OWN argv produced for
    that pid: the runner is real ``ps``, so a change to that argv is a change to what
    the assertions below are handed.
    """
    forged = f"{LONG_INTERPRETER} -P -m {reclaim.RUNTIME_MODULE}"
    child = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [forged, "-c", "import time; time.sleep(20)"],
        executable=sys.executable,
        text=True,
    )
    try:
        raw = ""

        def capture(command, timeout_s):
            # THE CENSUS'S OWN ARGV, run for real: this records the text those args
            # produce rather than fabricating a row, so what the cells below cut is a
            # row this module's reader actually sees.
            nonlocal raw
            raw = subprocess.run(  # noqa: S603 — fixed argv, no shell
                list(command), capture_output=True, text=True, timeout=timeout_s
            ).stdout
            return raw

        runtime_processes(run=capture)
        row = next(line for line in raw.splitlines() if line.split(None, 1)[0] == str(child.pid))
    finally:
        child.kill()
        child.wait(timeout=10)
    yield child.pid, row


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


def test_the_census_asks_for_the_whole_command_column() -> None:
    """THE WIDTH FLAG, ASSERTED ON THE ARGV RATHER THAN ON THIS HOST'S ``ps``.

    On Linux, ``command`` is the row's last column, so a piped ``ps`` extends it only
    to the display width it cannot determine — measured by this repository's own CI
    rounds at 80 columns, severing a runtime's ``-m <module>`` pair and so reporting
    a machine with runtimes on it as having none. ``ps -ww`` is unlimited width.

    The width cannot be asserted THROUGH ``ps`` here: macOS does not truncate this
    column at all (measured, same row with and without ``COLUMNS=80`` through a pipe),
    so a green run on this host says nothing about the Linux one. The argv is the
    instrument, so the argv is what is pinned — and with the real runner's real
    ordering, because a flag in the wrong place is a flag ``ps`` may not apply.
    """
    calls: list[list[str]] = []

    def run(command, timeout_s):
        calls.append(list(command))
        return ""

    assert runtime_processes(run=run) == []
    assert len(calls) == 1, "the census is ONE fork for the whole fleet"
    argv = calls[0]
    assert argv[0] == "ps"
    assert "-ww" in argv, f"the census must ask for unlimited width: {argv}"
    # The width flag belongs with ``ps``'s other display options, ahead of the format
    # that names the column it widens — the shape this module's three sibling readers
    # already use (``-Eww`` before ``-p``/``-eo``), and the shape a reader grepping for
    # the flag alone would not catch.
    assert argv.index("-ww") < argv.index("-eo"), f"the width flag comes after the format: {argv}"
    assert argv[-1] == "pid=,ppid=,etime=,time=,command="
    # AND NOT THE ENVIRONMENT. ``-Eww`` would widen the row too, but it appends each
    # process's own environment to the very column ``parse_process_row`` word-splits
    # and matches the spawn contract in, at ~4.4x the output (1.7 MB against 385 KB
    # for the whole fleet, measured for :func:`process_env`). The census has no use for
    # it: the environment is read per CANDIDATE by ``process_env``/``process_envs``.
    assert not any(flag.startswith("-E") for flag in argv), f"the census asked for env: {argv}"


def test_a_narrow_row_is_not_evidence_that_no_runtime_is_running() -> None:
    """THE SILENT ZERO ITSELF, against a real process's real row.

    Both cut rows below are THE SAME LIVE RUNTIME the census just found, seen through
    the two widths a Linux ``ps`` may pick for a piped row: the documented common 80,
    and one landing inside the module name. ``parse_process_row`` may only answer
    ``None`` for them — a severed runtime and a stranger are the same input to it, and
    it must not guess which it has — so the assertions state the consequence rather
    than papering over it: this reader CANNOT tell absence from truncation, and the
    width flag is therefore the only thing standing between a busy machine and a
    census that reports zero.
    """
    with live_contract_row() as (pid, row):
        wide = parse_process_row(row)
        assert wide is not None and wide.pid == pid, f"the wide row is not a runtime row: {row!r}"
        assert wide.command.startswith(LONG_INTERPRETER)

        cut = row[:NARROW_WIDTH]
        assert (
            reclaim.RUNTIME_MODULE not in cut
        ), f"this row is too short to model the Linux cut ({len(row)} cols): {row!r}"
        assert parse_process_row(cut) is None

        # The second width: the cut lands INSIDE the module name, so the ``-m`` this
        # reader keys on IS present and the module word is not (this is the shape the
        # spare-count helper read as "no spare" on the Linux shard).
        severed = row[: row.index(reclaim.RUNTIME_MODULE) + 20]
        assert "-m" in severed.split()
        assert severed.split()[-1] == reclaim.RUNTIME_MODULE[:20]
        assert parse_process_row(severed) is None


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
    view = fleet(tmp_path, records=[record()])
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


def test_a_recordless_runtime_the_socket_table_cannot_vouch_for_is_refused(tmp_path: Path) -> None:
    # FAIL-CLOSED ON MISSING EVIDENCE. Everything above this rung is evidence that
    # something IS there; this is the absence of the evidence that something is
    # ATTACHED, on the one rung whose job is to notice an attached client. QA round 1
    # (Q5) measured two identical record-less candidates each holding a live client on
    # the control port, differing only in whether the pass could read the socket
    # table: the readable one was ``observed`` and left alive, the unreadable one was
    # admitted and SIGTERMed. The failure mode must be refusal, not permission.
    item = verdict(
        proc(), fleet(tmp_path, sockets=SocketEvidence()), env_of=_recordless_env(tmp_path)
    )
    assert item.refusal == REFUSAL_SOCKETS_UNKNOWN
    assert item.may_end() is False
    # AND THE LADDER'S ORDER IS UNCHANGED: a stronger refusal still wins, so a
    # machine without ``lsof`` reports a recorded runtime as ``record-present``
    # rather than burying it under the weaker token.
    recorded = fleet(tmp_path, records=[record()], sockets=SocketEvidence())
    assert verdict(proc(), recorded, env_of=_recordless_env(tmp_path)).refusal == (
        REFUSAL_RECORD_PRESENT
    )


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
        row_of=lambda pid: reread(pid, root=tmp_path),
        fleet=view,
        kill=lambda pid, sig: kill.append((pid, sig)),
        now=NOW,
    )
    assert first.reclaimed == []
    # DEFERRED, NOT REFUSED: the ladder admitted it and the window is what holds it,
    # so it is reported once, in ``pending``, with the token saying why (a candidate
    # in both halves of one report was review round 1's nit 8).
    assert [item.refusal for item in first.pending] == [REFUSAL_UNCONFIRMED]
    assert first.refused == []
    assert first.refusals() == {}
    assert first.deferrals() == {REFUSAL_UNCONFIRMED: 1}
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
        row_of=lambda pid: reread(pid, root=tmp_path),
        fleet=view,
        kill=lambda pid, sig: kill.append((pid, sig)),
        now=NOW + CONFIRM_S,
    )
    assert [item.process.pid for item in second.signalled] == [4242]
    assert kill == [(4242, signal.SIGTERM)]
    assert "1 reclaimed" in second.summary()


def test_a_pass_confirms_every_candidate_and_not_just_the_first(tmp_path: Path) -> None:
    """THREE candidates through the two production passes, on the PRODUCTION shape.

    The regression test for the blocker both rounds found independently: a census of
    more than one candidate, on the two call shapes production actually uses —
    ``scope=None`` (no ``roots=``), which is what both the wake supervisor and
    ``lop sessions reclaim`` pass — and ``forget`` called with the GENERATOR
    ``reclaim_runtimes`` builds. Before the fix, ``Sightings.forget`` re-materialised
    that one-shot iterable inside its loop, so from the second iteration on it
    compared against an empty set and dropped every remaining entry: the first
    candidate in census order was the only one that could ever reach the window, and
    the other two were re-armed as ``unconfirmed`` on every pass forever (measured:
    1 of 3 signalled here, 1 of 6 at the CLI, on a real fleet 1 of 59 admitted).

    Three properties, because each one on its own was already covered by a test that
    could not see this: multiple candidates, MORE THAN ONE pass, and no ``roots=``
    (a named scope skips the ``forget`` call entirely, which is why the only existing
    multi-candidate multi-pass test passed while this shipped).
    """
    pids = [11, 22, 33]
    view = fleet(tmp_path)
    kill: list[tuple[int, int]] = []
    sightings = Sightings()
    for now in (NOW, NOW + CONFIRM_S):
        report = reclaim_runtimes(
            tmp_path,
            apply=True,
            sightings=sightings,
            processes=[proc(pid=pid) for pid in pids],
            env_of=_recordless_env(tmp_path),
            row_of=lambda pid: reread(pid, root=tmp_path),
            fleet=view,
            kill=lambda pid, sig: kill.append((pid, sig)),
            now=now,
        )
    assert [pid for pid, _sig in kill] == pids
    assert [item.process.pid for item in report.signalled] == pids
    assert "3 reclaimed" in report.summary()


def test_forget_tolerates_a_one_shot_iterable() -> None:
    # The mechanism, isolated, on the three iterables a caller could hand it. The
    # parameter is typed ``Iterable``, so the generator is a LEGAL call — and it is
    # the one production makes, which is why the fix belongs here rather than at the
    # call site: the next caller would hand it a one-shot iterable too.
    for payload in (list(range(1, 4)), tuple(range(1, 4)), (item for item in range(1, 4))):
        sightings = Sightings()
        for pid in (1, 2, 3):
            sightings.confirm(reclaim.Verdict(process=proc(pid=pid), config_root="/tmp/x"), now=NOW)
        sightings.forget(payload)
        assert sorted(sightings._seen) == [1, 2, 3], payload


def test_forget_still_drops_the_pids_that_left_the_census() -> None:
    sightings = Sightings()
    for pid in (1, 2, 3):
        sightings.confirm(reclaim.Verdict(process=proc(pid=pid), config_root="/tmp/x"), now=NOW)
    sightings.forget(pid for pid in (1, 3))
    assert sorted(sightings._seen) == [1, 3]


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
    # The token is on the PENDING row: the candidate is deferred to another window,
    # not refused by the ladder (see the ``pending`` field's comment).
    assert [item.refusal for item in report.pending] == [REFUSAL_BUSY_CPU]
    assert report.refused == []
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
            row_of=lambda pid: reread(pid, root=tmp_path),
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
            row_of=lambda pid: reread(pid, root=inside if pid == 1 else outside),
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
            row_of=lambda pid: reread(pid, root=tmp_path),
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
        row_of=lambda pid: reread(pid, root=tmp_path),
        fleet=view,
        kill=lambda pid, sig: None,
        now=NOW + 2 * CONFIRM_S,
        wait_s=1.0,
        sleep=lambda seconds: None,
    )
    assert len(report.signalled) == 1
    assert report.exited == report.signalled


def test_a_signal_is_withheld_when_the_target_is_no_longer_the_one_measured(
    tmp_path: Path,
) -> None:
    """THE SNAPSHOT AGES: the row is re-read immediately before the signal.

    The census and the fleet are read once at the top of a pass and the signal goes
    out later — measured on this host's real census at 19 ms for the first SIGTERM
    and 1161 ms for the last — so every rung that admitted a candidate was evaluated
    against a picture that is stale by the time ``os.kill`` runs. Four ways the
    target can stop being the process the pass decided about, each one a signal
    withheld rather than a stranger signalled: the pid is gone, it was recycled by
    something whose argv is not the spawn contract, it moved to another config root
    (the attribution the verdict rests on), or it published a record in the interval.
    """

    def pass_once(row_of) -> tuple[list[tuple[int, int]], reclaim.ReclaimReport]:
        kill: list[tuple[int, int]] = []
        sightings = Sightings()
        report = None
        for now in (NOW, NOW + CONFIRM_S):
            report = reclaim_runtimes(
                tmp_path,
                apply=True,
                sightings=sightings,
                processes=[proc()],
                env_of=_recordless_env(tmp_path),
                row_of=row_of,
                fleet=fleet(tmp_path),
                kill=lambda pid, sig: kill.append((pid, sig)),
                now=now,
            )
        assert report is not None
        return kill, report

    # (1) THE PID IS GONE by the time the signal would go out.
    kill, report = pass_once(lambda pid: None)
    assert kill == []
    assert report.refusals() == {REFUSAL_CHANGED: 1}

    # (2) RECYCLED by something that is not a runtime. ``process_row`` returns None
    # for it (its argv does not carry ``-m <module>``), which covers both this and
    # (1) — the point is that neither is a signal.
    kill, report = pass_once(
        lambda pid: RuntimeProcess(
            pid=pid, parent_pid=1, age_s=9000.0, cpu_s=0.0, command="/bin/zsh -c make -j8"
        )
    )
    assert kill == []
    assert report.refusals() == {REFUSAL_CHANGED: 1}

    # (3) A RUNTIME TOO YOUNG TO BE THE ONE MEASURED: the pid was recycled by
    # another runtime, whose argv passes and whose root matches, so only the age
    # separates them. The window is 60 s and this one is 3 s old.
    kill, report = pass_once(lambda pid: reread(pid, age_s=3.0, root=tmp_path))
    assert kill == []
    assert report.refusals() == {REFUSAL_CHANGED: 1}

    # (4) A DIFFERENT ROOT — the same pid, but no longer the candidate the verdict
    # attributed to this store.
    other = tmp_path / "other"
    other.mkdir()
    kill, report = pass_once(lambda pid: reread(pid, root=other))
    assert kill == []
    assert report.refusals() == {REFUSAL_CHANGED: 1}

    # (5) AND THE UNCHANGED CASE STILL SIGNALS: the check is a veto on a moved
    # target, not a second verdict that can refuse a candidate the ladder admitted.
    kill, report = pass_once(lambda pid: reread(pid, root=tmp_path))
    assert [pid for pid, _sig in kill] == [4242]
    assert report.refusals() == {}


def test_a_record_that_appears_between_the_window_and_the_signal_is_a_refusal(
    tmp_path: Path,
) -> None:
    """The other half of the race: the record comes back INSIDE the pass.

    A candidate whose record was deleted re-publishes on its next heartbeat
    (``SessionRecordWriter.heartbeat`` -> ``registry.publish``), which makes it
    reachable again — the exact race the 60 s confirm window exists to prevent, just
    moved inside the pass. The fleet a pass holds is a SNAPSHOT, so the ladder has
    already run by then; only the signal-time re-read of the record path can see it.
    """
    sightings = Sightings()
    kill: list[tuple[int, int]] = []
    report = None
    for index, now in enumerate((NOW, NOW + CONFIRM_S), start=1):
        if index == 2:
            published = tmp_path / RUN_DIRNAME
            published.mkdir(parents=True, exist_ok=True)
            (published / "4242.json").write_text(json.dumps(record(pid=4242).to_json()))
        report = reclaim_runtimes(
            tmp_path,
            apply=True,
            sightings=sightings,
            processes=[proc()],
            env_of=_recordless_env(tmp_path),
            row_of=lambda pid: reread(pid, root=tmp_path),
            fleet=fleet(tmp_path),
            kill=lambda pid, sig: kill.append((pid, sig)),
            now=now,
        )
    assert kill == []
    assert report is not None
    assert report.refusals() == {REFUSAL_RECORD_PRESENT: 1}


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
    # THE DEAD RECORD IS STILL A ROW. ``reap=False`` is the reader's read, so the
    # record of a pid that is gone comes back as evidence rather than being moved
    # aside: it is what a later "why did this die" question is answered from. (The
    # fleet used to carry ``registry.scan``'s per-record ``live|wedged|stale`` beside
    # it; the roster derives that itself now, from ``registry.classify`` with the
    # batched zombie answer supplied — one liveness rule per row instead of two.)
    assert 999999 in view.records
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
    # The fleet path: ONE ``ps`` over the process table instead of a fork per runtime
    # (``-Eww -eo pid=,command=`` where the environment comes in that fork, ``-ww``
    # where it does not — see :func:`pid_environment`, and the spelling cells at the
    # end of this file). On Linux the environment per row is a ``/proc`` read, which
    # is a file read and not a fork, so ``len(calls) == 1`` holds on both platforms.
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


# ---------------------------------------------------------------------------
# The signal-time re-read, and the window's floor
# ---------------------------------------------------------------------------


def test_process_row_reads_one_pid_and_its_environment_in_one_fork() -> None:
    # ONE FORK, for two of the four facts the re-identification rests on: the
    # environment is appended to the ``command`` column — in that fork where the
    # platform's ``ps`` can carry it (``-Eww`` on macOS/BSD), from ``/proc`` where it
    # cannot (Linux, which has no ``-E`` at all) — so the same call that says "this pid
    # is still a runtime, this old" also carries the config root the verdict
    # attributed it to. The alternative costs a second fork per candidate on the one
    # path that must not be slow enough to widen the window it is closing.
    calls: list[list[str]] = []

    def run(command, timeout_s):
        calls.append(list(command))
        return (
            " 4242      1     05:00  0:00.02 /usr/bin/python3 -P -m "
            "local_operator.session.runtime.process LOCAL_OPERATOR_CONFIG_DIR=/tmp/one "
            "HOME=/tmp/one-home\n"
        )

    row = process_row(4242, run=run)
    assert len(calls) == 1, calls
    # THE PLATFORM'S OWN SPELLING: ``-Eww`` where ``-E`` exists (it is what carries
    # the environment in this fork), ``-ww`` where it does not, and never ``-E`` on a
    # platform whose ``ps`` refuses it.
    expected_env_flag = "-ww" if reclaim._IS_LINUX else "-Eww"
    assert calls[0][:3] == ["ps", expected_env_flag, "-p"], calls[0]
    assert no_bsd_env_flag(calls[0]) or not reclaim._IS_LINUX, calls[0]
    assert calls[0][3] == "4242"
    assert row is not None
    assert row.pid == 4242
    assert row.age_s == pytest.approx(300.0)
    assert config_root_of(row.command) == "/tmp/one"


def test_process_row_does_not_recognise_a_stranger_or_an_absent_pid() -> None:
    # THE TWO SHAPES OF "the target moved", and neither may be signalled:
    # a pid recycled by something that is not a runtime (no ``-m <module>``), and a
    # pid ``ps`` does not report at all (gone between the census and the signal).
    stranger = lambda command, timeout_s: (  # noqa: E731 — a one-line seam
        " 4242      1     05:00  0:00.02 /bin/zsh -c make -j8\n"
    )
    assert process_row(4242, run=stranger) is None
    assert process_row(4242, run=lambda command, timeout_s: "") is None


def test_the_record_re_read_names_the_file_without_creating_the_directory(
    tmp_path: Path,
) -> None:
    # ``registry.record_path`` would mkdir the run directory on the way to the same
    # path, and this reader may not: the candidate's own root can be a path that is
    # GONE (the class this sweep exists for), and conjuring it back would be the
    # sweep inventing the very store the runtime could then publish into.
    root = tmp_path / "gone-store"
    assert record_file(root, 9) == root / RUN_DIRNAME / "9.json"
    assert record_file(root, 9).is_file() is False
    assert not root.exists()


def test_the_pass_refuses_when_the_socket_table_could_not_be_read(tmp_path: Path) -> None:
    """FAIL-CLOSED, END TO END: no socket table means no signal, on the real pass.

    QA round 1 (Q5) drove two identical record-less candidates with a live client on
    each control port through the pass, varying only whether the socket table could
    be read: the readable one was ``observed`` and left alive, the unreadable one was
    admitted and SIGTERMed. The failure of the strongest refusal must be a refusal.
    """
    kill: list[tuple[int, int]] = []
    sightings = Sightings()
    for now in (NOW, NOW + CONFIRM_S):
        report = reclaim_runtimes(
            tmp_path,
            apply=True,
            sightings=sightings,
            processes=[proc()],
            env_of=_recordless_env(tmp_path),
            row_of=lambda pid: reread(pid, root=tmp_path),
            fleet=fleet(tmp_path, sockets=SocketEvidence()),
            kill=lambda pid, sig: kill.append((pid, sig)),
            now=now,
        )
    assert kill == []
    assert report.sockets_available is False
    assert report.refusals() == {REFUSAL_SOCKETS_UNKNOWN: 1}
    assert "socket table unavailable" in report.summary()


def test_the_confirm_window_cannot_be_asked_for_shorter_than_the_cpu_rung(
    tmp_path: Path,
) -> None:
    """``--confirm-s`` has a floor, because a shorter window has no CPU rung.

    QA round 1 (Q2) measured the end-to-end consequence: at ``--confirm-s 0`` — which
    the flag accepted, and which skipped the watch entirely — a process that had
    burned 90.4 s of cumulative CPU (6.30 s per 60 s against a 1.2 s budget at the
    default window) was admitted and SIGTERMed. The floor is derived from the CPU
    rung's own two constants, so it cannot drift from them, and this asserts it
    through the REAL parser rather than through the helper alone.
    """
    from local_operator.cli import _confirm_window, build_cli_parser

    assert MIN_ACTIONABLE_CONFIRM_S == BUSY_CPU_FLOOR_S / reclaim.BUSY_CPU_FRACTION
    with pytest.raises(argparse.ArgumentTypeError):
        _confirm_window("0")
    with pytest.raises(argparse.ArgumentTypeError):
        _confirm_window(str(int(MIN_ACTIONABLE_CONFIRM_S) - 1))
    assert _confirm_window(str(int(MIN_ACTIONABLE_CONFIRM_S))) == int(MIN_ACTIONABLE_CONFIRM_S)
    assert _confirm_window("120") == 120

    parser = build_cli_parser()
    for bad in ("0", "5", "-1"):
        with pytest.raises(SystemExit) as raised:
            parser.parse_args(["sessions", "reclaim", "--confirm-s", bad])
        assert raised.value.code == 2, bad
    # An omitted flag keeps the default (``None``), resolved to ``CONFIRM_S`` by the
    # command itself: the window this pass documents as its safety property.
    assert parser.parse_args(["sessions", "reclaim"]).confirm_s is None
    assert parser.parse_args(["sessions", "reclaim", "--confirm-s", "90"]).confirm_s == 90


# ---------------------------------------------------------------------------
# The environment's SOURCE: ``-E`` is a BSD/macOS option, procps has none
# ---------------------------------------------------------------------------
#
# Every other environment test in this file injects ``run``, which is exactly why
# none of them could see that ``ps -Eww`` is not a procps option: the seam kept
# answering with canned text while the real reader answered nothing at all. The
# cells below therefore read a REAL child's environment through the module's REAL
# spelling, and assert the SPELLING itself per platform.

#: A variable only the children below have, so "the reader found it" cannot be an
#: inherited coincidence of the suite's own environment.
_ENV_PROBE_NAME = "LO_RECLAIM_ENV_PROBE"
_ENV_PROBE_VALUE = "probe-4f0c9a"


def no_bsd_env_flag(argv: Sequence[str]) -> bool:
    """No token in ``argv`` is the BSD-only ``-E``/``-Eww`` spelling."""
    return not any(part.startswith("-E") for part in argv)


class _EnvChild:
    """A real child process: the argv the census matches, and its own environment.

    NO RUNTIME IS STARTED. The child is this interpreter running a sleep, with the
    spawn contract's two tokens (``-m local_operator.session.runtime.process``) behind
    a first non-option word, so CPython stops parsing its OWN options and hands them
    to ``sys.argv`` instead of booting the module — booting a real runtime here would
    create a store and publish a record, which a unit test may not do. What this
    module reads is the ``ps`` command column, and those tokens are in it.

    ``sys.executable``, i.e. this suite's own python, and not a platform binary,
    because of a measured macOS 27 property: ``ps -Eww`` returned the environment of a
    python child (3318 bytes) and of NONE of ``/bin/sleep`` (14), ``/bin/sh`` (9),
    ``/bin/bash`` (9), ``/usr/bin/tail`` (27) or ``/usr/bin/yes`` (13). A live runtime
    is a python process, so the reader works where the product needs it and a test
    child that is not one would measure the platform's exception rather than the rule.
    """

    def __init__(self, root: Path, *, sleep_s: float = 30.0) -> None:
        self.root = root
        self.process = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [
                sys.executable,
                "-P",
                "-c",
                f"import time; time.sleep({sleep_s})",
                # The word that ends CPython's option parsing, so ``-m <module>``
                # lands in ``sys.argv`` (verified: ``sys.argv`` came back as
                # ``['-c', 'stand-in', '-m', 'local_operator...process']``).
                "stand-in",
                "-m",
                reclaim.RUNTIME_MODULE,
            ],
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                reclaim.CONFIG_DIR_ENV: str(root),
                "HOME": f"{root}-home",
                _ENV_PROBE_NAME: _ENV_PROBE_VALUE,
            },
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Its own session, so the reap below is a kill by the pid THIS test
            # created and can reach nothing else on the machine.
            start_new_session=True,
        )

    @property
    def pid(self) -> int:
        return self.process.pid

    def reap(self) -> None:
        try:
            os.killpg(os.getpgid(self.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover — killpg precedes
            pass


@pytest.fixture
def env_child(tmp_path: Path) -> Iterator[_EnvChild]:
    child = _EnvChild(tmp_path / "probe-root")
    try:
        yield child
    finally:
        child.reap()


@pytest.fixture
def fresh_failure_reports() -> Iterator[None]:
    """``_REPORTED_FAILURES`` is process state: a test asserting on it starts clean.

    ``getattr`` rather than an attribute read so that a run against a module WITHOUT
    the report channel fails inside the test's own assertions (the log is empty)
    instead of erroring in setup — the difference between a red behavioural proof and
    a red harness.
    """
    reported: set[tuple[str, str]] = getattr(reclaim, "_REPORTED_FAILURES", set())
    reported.clear()
    try:
        yield
    finally:
        reported.clear()


def test_a_real_childs_environment_is_read_by_the_real_reader(env_child: _EnvChild) -> None:
    """NO INJECTED ``run``: the platform's own spelling, against a real process.

    This is the cell that fails on Linux without the fix — there ``ps -Eww`` is an
    invalid option, ``_run_command`` answered ``""``, and both readers below
    returned nothing while every injected test in this file stayed green.
    """
    env = reclaim.process_env(env_child.pid)
    assert f"{_ENV_PROBE_NAME}={_ENV_PROBE_VALUE}" in env
    assert config_root_of(env) == str(env_child.root)

    batch = reclaim.process_envs()
    assert f"{_ENV_PROBE_NAME}={_ENV_PROBE_VALUE}" in batch.get(env_child.pid, "")
    assert config_root_of(batch[env_child.pid]) == str(env_child.root)


def test_the_signal_time_row_reader_reads_a_real_child_and_its_root(
    env_child: _EnvChild,
) -> None:
    """The re-read path end to end: ``process_row`` -> ``target_changed`` -> signal.

    On Linux this is the ``/proc`` append inside ``process_row``; before the fix the
    row itself came back ``None``, so no candidate could pass the re-read and the
    sweep could never signal anything at all.
    """
    row = process_row(env_child.pid)
    assert row is not None, "the real reader did not recognise a real child's argv"
    assert row.pid == env_child.pid
    assert config_root_of(row.command) == str(env_child.root)

    item = reclaim.Verdict(
        process=RuntimeProcess(
            pid=env_child.pid,
            parent_pid=os.getpid(),
            age_s=0.1,
            cpu_s=0.0,
            command=row.command,
        ),
        config_root=str(env_child.root),
    )
    # "" is "still the process the pass decided about": the one answer that lets the
    # signal go out, and the one the broken instrument could never produce.
    assert reclaim.target_changed(item, row_of=process_row, roots=[]) == ""


def test_the_spelling_is_the_one_this_platform_should_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``-E`` on Linux is a FAILURE here, not a silent zero.

    ``_IS_LINUX`` is a NAME rather than an inline ``sys.platform`` test so both arms
    are drivable from either host, and the constant is re-derived from
    ``sys.platform`` in the same breath so a constant that had drifted from the host
    cannot pass either.
    """
    assert reclaim._IS_LINUX == sys.platform.startswith("linux")

    census_row = (
        " 4242      1     05:00  0:00.02 /usr/bin/python3 -P -m "
        f"{reclaim.RUNTIME_MODULE} LOCAL_OPERATOR_CONFIG_DIR=/tmp/one HOME=/tmp/one-home\n"
    )

    # --- THE LINUX ARM, from any host: no ``-E`` anywhere, and no fork for the env.
    monkeypatch.setattr(reclaim, "_IS_LINUX", True)
    calls: list[list[str]] = []

    def run(command, timeout_s):
        calls.append(list(command))
        return census_row

    assert process_row(4242, run=run) is not None
    assert calls[0][:3] == ["ps", "-ww", "-p"], calls[0]
    assert no_bsd_env_flag(calls[0]), calls[0]
    linux_row = process_row(4242, run=run)
    assert linux_row is not None
    assert config_root_of(linux_row.command) == "/tmp/one"

    def forbidden(command, timeout_s):
        raise AssertionError(f"the Linux environment must come from /proc: {command}")

    # The environment read itself forks nothing on Linux, and its answer IS what
    # /proc says — the equality is the assertion on Linux CI and trivially true here.
    assert reclaim.pid_environment(4242, run=forbidden) == reclaim.proc_environ_text(4242)

    calls.clear()
    envs = reclaim.process_envs(run=run)
    assert set(envs) == {4242}
    # The value is the row's COMMAND COLUMN (everything after the pid), which is the
    # shape this reader has always returned; a prefix rather than an equality because
    # on Linux a /proc read for the same pid appends whatever that process holds.
    command_column = census_row.strip().split(" ", 1)[1].strip()
    assert envs[4242].startswith(command_column)
    assert config_root_of(envs[4242]) == "/tmp/one"
    assert calls[0][:2] == ["ps", "-ww"], calls[0]
    assert no_bsd_env_flag(calls[0]), calls[0]

    # --- THE BSD/macOS ARM, from any host: the environment in the SAME fork.
    monkeypatch.setattr(reclaim, "_IS_LINUX", False)
    seen: list[list[str]] = []

    def recorded(command, timeout_s):
        seen.append(list(command))
        return "row-with-env"

    assert reclaim.pid_environment(4242, run=recorded) == "row-with-env"
    assert seen == [["ps", "-Eww", "-p", "4242", "-o", "command="]]

    seen.clear()

    def recorded_row(command, timeout_s):
        seen.append(list(command))
        return census_row

    row = process_row(4242, run=recorded_row)
    assert row is not None
    assert seen[0][:3] == ["ps", "-Eww", "-p"], seen[0]
    assert len(seen) == 1, "the macOS arm must not fork a second time for the environment"
    assert config_root_of(row.command) == "/tmp/one"


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="procps and /proc are Linux; there is nothing for this cell to measure here",
)
def test_on_linux_proc_holds_the_environment_and_ps_rejects_the_bsd_flag(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """THE TWO FACTS, measured on the platform they are about when CI runs them.

    (1) ``/proc/<pid>/environ`` is where this platform's environment is, which is
    what the fix reads. (2) ``ps -Eww`` — what the module ran here before it — is
    REFUSED by procps, non-zero and with a complaint on stderr; the exit-code
    assertion is also the alarm for the day procps grows ``-E``, at which point the
    branch can be simplified rather than kept "just in case".
    """
    monkeypatch.setenv(_ENV_PROBE_NAME, _ENV_PROBE_VALUE)
    own = reclaim.proc_environ_text(os.getpid())
    assert f"{_ENV_PROBE_NAME}={_ENV_PROBE_VALUE}" in own
    assert f"HOME={os.environ['HOME']}" in own

    rejected = ["ps", "-Eww", "-p", str(os.getpid()), "-o", "command="]
    done = subprocess.run(  # noqa: S603 — fixed argv, no shell
        rejected, capture_output=True, text=True, check=False
    )
    assert done.returncode != 0, "procps accepted -E; this branch can be simplified"
    assert done.stdout == ""
    assert done.stderr.strip(), "a rejected option is refused WITH a message"

    # ...and that refusal is what the module now reports instead of swallowing.
    reclaim._REPORTED_FAILURES.clear()
    caplog.set_level(logging.WARNING, logger=reclaim.__name__)
    assert reclaim._run_command(rejected, 5.0) == ""
    assert len(caplog.records) == 1, [record.getMessage() for record in caplog.records]


def test_a_probe_that_cannot_run_reports_once_and_still_returns_empty(
    caplog: pytest.LogCaptureFixture, fresh_failure_reports: None
) -> None:
    """A tool that is not on this machine is no longer indistinguishable from one
    that found nothing — and a per-candidate reader does not reprint it per pid."""
    caplog.set_level(logging.WARNING, logger=reclaim.__name__)
    missing = ["/nonexistent/ps-probe-4f0c9a"]
    assert reclaim._run_command(missing, 5.0) == ""
    assert len(caplog.records) == 1, [record.getMessage() for record in caplog.records]
    assert "no such file" in caplog.text.lower() or "No such file" in caplog.text

    caplog.clear()
    assert reclaim._run_command(missing, 5.0) == ""
    assert caplog.records == [], "the same breakage must be one line, not one per pid"


def test_a_message_less_non_zero_exit_is_this_module_s_ordinary_answer(
    caplog: pytest.LogCaptureFixture, fresh_failure_reports: None
) -> None:
    """REPORTED: a rejection that complains. NOT reported: a quiet non-zero exit.

    Simulated with real commands because the platform that rejects ``-E`` (Linux) is
    not this host: what is simulated is the COMMAND's behaviour, never the code under
    test. The quiet arm is the module's normal answer — ``ps -p <gone>`` and ``lsof``
    with an empty table both exit non-zero in silence — and a warning on that path
    would print on every sweep, which is how a real warning stops being read.
    """
    caplog.set_level(logging.WARNING, logger=reclaim.__name__)
    rejected = ["sh", "-c", "echo 'ps: invalid option -- E' 1>&2; exit 1"]
    assert reclaim._run_command(rejected, 5.0) == ""
    assert len(caplog.records) == 1, [record.getMessage() for record in caplog.records]
    assert "invalid option" in caplog.text

    caplog.clear()
    assert reclaim._run_command(["sh", "-c", "exit 1"], 5.0) == ""
    assert caplog.records == []

    # AND THE SIX CALL SITES' CONTRACT IS UNCHANGED: a failing command that printed
    # something still hands its stdout back, which is what callers read before.
    assert reclaim._run_command(["sh", "-c", "echo partial; exit 3"], 5.0) == "partial\n"


def test_a_rejected_probe_withholds_the_signal_instead_of_signalling(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fresh_failure_reports: None
) -> None:
    """THE BUG'S CONSEQUENCE, END TO END: fail-closed, and now audible.

    The stand-in answers exactly as procps answers ``ps -Eww`` — a message on stderr,
    a non-zero exit, nothing on stdout — and the pass is the real one: the candidate
    is admitted, the signal-time re-read cannot read a row, the signal is withheld
    (``REFUSAL_CHANGED``), and the reason is on the log exactly once. NOTHING
    DANGEROUS HAPPENED while this was broken — the sweep just stopped working, which
    is what a silent instrument looks like.
    """
    rejecting = ["sh", "-c", "echo 'ps: invalid option -- E' 1>&2; exit 1"]

    def instrument(command, timeout_s):
        # The same reader, pointed at a command that refuses the argv.
        return reclaim._run_command(rejecting, timeout_s)

    def row_of(pid: int) -> RuntimeProcess | None:
        return process_row(pid, run=instrument)

    caplog.set_level(logging.WARNING, logger=reclaim.__name__)
    kill: list[tuple[int, int]] = []
    sightings = Sightings()
    report = None
    for now in (NOW, NOW + CONFIRM_S):
        report = reclaim_runtimes(
            tmp_path,
            apply=True,
            sightings=sightings,
            processes=[proc()],
            env_of=_recordless_env(tmp_path),
            row_of=row_of,
            fleet=fleet(tmp_path),
            kill=lambda pid, sig: kill.append((pid, sig)),
            now=now,
        )
    assert kill == [], "a probe that cannot read the row must never end a runtime"
    assert report is not None
    assert report.refusals() == {REFUSAL_CHANGED: 1}
    assert len(caplog.records) == 1, [record.getMessage() for record in caplog.records]
