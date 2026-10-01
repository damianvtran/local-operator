"""Bulk resume: the set selector and the bounded runner (2026-10-01).

The selector's contract is the listing's own — membership is the attention
store's latest recorded outcome kind, the sets are ``PAUSED_OUTCOME_KINDS`` /
``FAILED_OUTCOME_KINDS``, live ids are dropped before the cap and the cap
keeps the NEWEST members. The runner's contract is what the CLI and the
sessions tool both render: one outcome per session, a failure isolated to its
own line, and a receipt-driven resolution with an honest ``unresolved`` at the
bound rather than a silent hang.

Stub children substitute the module's own seam (``resume_child_argv``), so
these tests exercise the REAL runner — the semaphore, the receipt parse, the
follow-up poll and the group-kill — without spawning real sessions. The
follow-up is driven against a real ledger under a tmp config root.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import bulk_resume
from local_operator.session.attention import AttentionStore
from local_operator.session.bulk_resume import (
    DEFAULT_RESUME_MESSAGE,
    RESUME_DEFAULT_LIMIT,
    live_session_ids,
    resume_child_argv,
    resume_sessions,
    select_resume_candidates,
)

pytestmark = pytest.mark.asyncio

# --- fixtures ---------------------------------------------------------------


def _session(root: Path, session_id: str, *, age_s: float = 0.0, kind: str | None = None) -> Path:
    """One user session directory; optionally with a recorded outcome kind."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = directory / "transcript.jsonl"
    transcript.write_text('{"type":"message"}\n', encoding="utf-8")
    stamp = time.time() - age_s
    os.utime(transcript, (stamp, stamp))
    if kind is not None:
        AttentionStore(root / "attention.db").publish(
            f"session/{session_id}", str(uuid.uuid4()), f"e-{session_id}", kind, reason="seed"
        )
    return directory


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


# --- the selector -----------------------------------------------------------


def test_the_sets_are_the_products_own_vocabulary(store: Path) -> None:
    """A member is the LAST recorded kind, and the sets map to the constants.

    The mapping lives in ``info.collect`` — these tests consume it rather than
    restate it, which is the point: a resume path that grew its own definition
    of "failed" is exactly the drift the constants exist to prevent.
    """
    _session(store, "s-paused", kind="interrupted")
    _session(store, "s-retired", kind="retired")
    _session(store, "s-failed", kind="error")
    _session(store, "s-complete", kind="complete")
    _session(store, "s-none")

    paused = {sid for sid, _ in select_resume_candidates(store, paused=True).sessions}
    failed = {sid for sid, _ in select_resume_candidates(store, failed=True).sessions}
    union = {sid for sid, _ in select_resume_candidates(store, paused=True, failed=True).sessions}
    assert paused == {"s-paused", "s-retired"}
    assert failed == {"s-failed"}
    assert union == paused | failed
    # complete and no-outcome are in NEITHER set; --all still sees them.
    everything = {sid for sid, _ in select_resume_candidates(store, all_sessions=True).sessions}
    assert {"s-complete", "s-none"} <= everything


def test_live_ids_are_dropped_before_the_cap(store: Path) -> None:
    """A running session is not a member, and dropping it must free a slot."""
    _session(store, "newest", age_s=1, kind="error")
    _session(store, "running", age_s=2, kind="error")
    _session(store, "oldest", age_s=3, kind="error")

    selection = select_resume_candidates(store, failed=True, limit=2, exclude_ids={"running"})
    ids = [sid for sid, _ in selection.sessions]
    assert ids == ["newest", "oldest"]
    assert selection.matched == 2  # the exclusion happens before the cap counts


def test_an_old_member_beyond_the_cap_still_appears(store: Path) -> None:
    """Filter-before-limit, observed at this layer: the target is FOUND.

    Twenty-five newer complete sessions sit in front of one old error; a
    selection that capped the candidates first would answer "no failures"
    for a store that has one, which is the failure mode the rule prevents.
    """
    for index in range(25):
        _session(store, f"filler-{index:02d}", age_s=100 + index, kind="complete")
    _session(store, "old-error", age_s=10_000, kind="error")
    selection = select_resume_candidates(store, failed=True)
    assert [sid for sid, _ in selection.sessions] == ["old-error"]


def test_the_cap_keeps_the_newest_members_and_reports_the_pre_cap_count(store: Path) -> None:
    for index in range(5):
        _session(store, f"f{index}", age_s=10 + index, kind="error")
    selection = select_resume_candidates(store, failed=True, limit=2)
    assert [sid for sid, _ in selection.sessions] == ["f0", "f1"]
    assert selection.matched == 5


def test_the_default_cap_is_a_named_constant_and_applies_when_none_is_given(store: Path) -> None:
    """An ACTION with no limit must not be one flag away from a thousand runs."""
    assert RESUME_DEFAULT_LIMIT == 20
    for index in range(RESUME_DEFAULT_LIMIT + 3):
        _session(store, f"f{index:03d}", age_s=1000 + index, kind="interrupted")
    selection = select_resume_candidates(store, paused=True)
    assert len(selection.sessions) == RESUME_DEFAULT_LIMIT
    assert selection.matched == RESUME_DEFAULT_LIMIT + 3


def test_all_is_every_stored_session_newest_first(store: Path) -> None:
    _session(store, "older", age_s=50, kind="complete")
    _session(store, "newer", age_s=1, kind=None)
    selection = select_resume_candidates(store, all_sessions=True)
    assert [sid for sid, _ in selection.sessions] == ["newer", "older"]


def test_live_session_ids_reads_the_registry_scan(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exclude set is the registry's own scan — any published record.

    Same rule the stored listing uses (``_stored_lines`` excludes every
    session id the scan produced), so a session the fleet can see is never
    offered for a second resume.
    """

    class _Rec:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id

    from local_operator.session.runtime import registry

    monkeypatch.setattr(
        registry,
        "scan",
        lambda root=None: [
            (_Rec("live-1"), "live"),
            (_Rec("wedged-1"), "wedged"),
            (_Rec("stale-1"), "stale"),
        ],
    )
    # live and wedged name an existing pid and are excluded; a stale record's
    # pid is gone, so that session stays resumable (the crashed-runtime shape).
    assert live_session_ids(store) == {"live-1", "wedged-1"}


def test_the_child_argv_is_the_single_command(store: Path) -> None:
    """One place spells the child; this pins its exact shape."""
    assert resume_child_argv("abc123", "continue") == [
        "-m",
        "local_operator.cli",
        "exec",
        "--background",
        "--resume",
        "abc123",
        "--",
        "continue",
    ]


# --- the runner -------------------------------------------------------------


def _stub(monkeypatch: pytest.MonkeyPatch, build: Any) -> None:
    monkeypatch.setattr(bulk_resume, "resume_child_argv", build)


def _ok_stub(job_id: str = "aaaa11112222", status: str = "running") -> Any:
    code = (
        "import sys\n"
        f"print('Background job {job_id}: {status} (execution receipt)', file=sys.stderr)\n"
        f"print('Status: lop exec --status {job_id}', file=sys.stderr)\n"
    )
    return lambda session_id, message: ["-c", code]


async def test_every_resolved_child_reports_ok_with_its_receipt(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub(monkeypatch, _ok_stub())
    outcomes = await resume_sessions(
        [("s1", "one"), ("s2", "two")],
        message=DEFAULT_RESUME_MESSAGE,
        env=dict(os.environ),
    )
    assert [o.ok for o in outcomes] == [True, True]
    assert {o.status for o in outcomes} == {"running"}
    assert all(o.job_id == "aaaa11112222" for o in outcomes)


async def test_one_failure_is_one_failed_outcome_not_an_aborted_batch(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-session failure isolation — the shape a refusal must take."""

    def build(session_id: str, message: str) -> list[str]:
        if session_id == "bad":
            code = (
                "import sys\n"
                "print('\\033[31msession bad is already open in another process (pid 1) "
                "— watch and steer it there\\033[0m', file=sys.stderr)\n"
                "sys.exit(1)\n"
            )
        else:
            code = "import sys\n" "print('Background job bbbb11112222: running', file=sys.stderr)\n"
        return ["-c", code]

    _stub(monkeypatch, build)
    outcomes = await resume_sessions(
        [("bad", "refused"), ("good", "fine")],
        message="go",
        env=dict(os.environ),
    )
    by_id = {o.session_id: o for o in outcomes}
    assert by_id["bad"].ok is False
    assert "already open in another process" in by_id["bad"].detail
    assert by_id["good"].ok is True


async def test_the_bound_holds_and_still_runs_children_in_parallel(
    store: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Never more than N in flight, and more than one at a time.

    The trace records each child's start and end in spawn order; a prefix
    simulation (started - ended) is exact about overlap regardless of how the
    machine schedules, so the bound is asserted structurally rather than by
    timing.
    """
    trace = tmp_path / "trace.txt"

    def build(session_id: str, message: str) -> list[str]:
        code = (
            "import os\n"
            "import time\n"
            f"path = {str(trace)!r}\n"
            f"sid = {session_id!r}\n"
            "with open(path, 'a') as handle:\n"
            "    handle.write('S' + sid + chr(10))\n"
            "time.sleep(0.4)\n"
            "with open(path, 'a') as handle:\n"
            "    handle.write('E' + sid + chr(10))\n"
            "import sys\n"
            f"print('Background job {{0}}{{1}}: running'.format(sid[0], sid[1]), file=sys.stderr)\n"
        )
        return ["-c", code]

    _stub(monkeypatch, build)
    sessions = [(f"c{i}", f"child {i}") for i in range(6)]
    outcomes = await resume_sessions(sessions, message="go", env=dict(os.environ), concurrency=2)
    assert all(o.ok for o in outcomes)
    events = [line.strip() for line in trace.read_text().splitlines()]
    in_flight = 0
    peak = 0
    for event in events:
        in_flight += 1 if event.startswith("S") else -1
        peak = max(peak, in_flight)
    assert peak <= 2
    assert peak == 2


async def test_a_child_that_overruns_its_bound_is_killed_and_reported(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bulk_resume, "RESUME_CHILD_TIMEOUT_S", 0.4)

    def build(session_id: str, message: str) -> list[str]:
        return ["-c", "import time; time.sleep(30)"]

    _stub(monkeypatch, build)
    started = time.monotonic()
    outcomes = await resume_sessions([("slow", "slow")], message="go", env=dict(os.environ))
    assert time.monotonic() - started < 10
    assert outcomes[0].ok is False
    assert outcomes[0].status == "timeout"
    assert "did not finish" in outcomes[0].detail


async def test_a_starting_receipt_is_resolved_by_the_shared_follow_up(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child exits 'starting'; the ledger later says running — ok."""
    job = "cccc11112222"
    log_dir = store / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "exec-jobs.jsonl").write_text(
        json.dumps({"id": job, "status": "starting", "started_at": "2026-10-01T00:00:00"})
        + "\n"
        + json.dumps({"id": job, "status": "running"})
        + "\n",
        encoding="utf-8",
    )

    _stub(monkeypatch, _ok_stub(job_id=job, status="starting"))
    outcomes = await resume_sessions([("s1", "one")], message="go", env=dict(os.environ))
    assert outcomes[0].ok is True
    assert outcomes[0].status == "running"


async def test_a_starting_receipt_that_never_goes_live_is_unresolved_not_a_hang(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bulk_resume, "RESUME_READY_FOLLOWUP_S", 0.4)
    job = "dddd11112222"
    log_dir = store / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "exec-jobs.jsonl").write_text(
        json.dumps({"id": job, "status": "starting", "started_at": "2026-10-01T00:00:00"}) + "\n",
        encoding="utf-8",
    )

    _stub(monkeypatch, _ok_stub(job_id=job, status="starting"))
    outcomes = await resume_sessions([("s1", "one")], message="go", env=dict(os.environ))
    assert outcomes[0].ok is False
    assert outcomes[0].status == "unresolved"
    assert "--status" in outcomes[0].detail
