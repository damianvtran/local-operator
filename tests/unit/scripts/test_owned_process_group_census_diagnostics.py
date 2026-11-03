"""Synthetic Linux acquire churn: observe failed scans without granting authority."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from local_operator.procstate import ProcessSample
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401
from tests.unit.scripts.test_run_bounded_run_record import _install


class ProcWorld:
    """Only supplied namespace/row facts; no OS PID read or signal is permitted."""

    def __init__(self, change):
        self.change = change
        self.calls = []
        self.scans = 0

    def listdir(self, path):
        self.calls.append(("listdir", path))
        self.scans += 1
        if self.change == "birth":
            return ["10", "20"] + ([str(100 + self.scans)] if self.scans % 2 == 0 else [])
        return ["10", "20"] if self.scans % 2 else ["10"]

    def read(self, path):
        self.calls.append(("read", path))
        pid = int(path.split("/")[-2])
        if pid > 100:
            # Since strict_linux reconciles its closing delta to a fixpoint, a
            # foreign arrival is classified and the census completes. A birth
            # that still refuses must be one nobody can classify: a hidepid-style
            # EACCES row, which stays unknown on every attempt.
            raise PermissionError(13, "synthetic hidepid")
        group = 10 if pid == 10 or self.change == "owned-death" else 90
        return f"{pid} (synthetic-secret-must-not-leak) R 1 {group}"

    def stat(self, path):
        pytest.fail("these synthetic rows were read; no extra stat probe is justified")

    def census(self, diagnostics=None):
        return owned.strict_linux(
            10,
            5,
            listdir=self.listdir,
            read=self.read,
            stat=self.stat,
            clock=lambda: 0,
            **({"diagnostics": diagnostics} if diagnostics is not None else {}),
        )


@pytest.mark.parametrize("change", ["birth"])
def test_six_existing_scans_refuse_with_distinct_observations(change):
    world = ProcWorld(change)
    detail: dict[str, object] = {"stale": True}
    with pytest.raises(owned.Unknown, match="^unstable /proc census$") as raised:
        world.census(detail)
    assert world.scans == 12
    assert detail == raised.value.census_failure
    assert "stale" not in detail
    assert detail["phase"] == "attempt-limit"
    assert detail["attempt_count"] == 6
    attempts = detail["attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 6
    assert all(row["opening_count"] == 2 for row in attempts)
    assert all(row["closing_count"] == 3 and row["added_count"] == 1 for row in attempts)
    assert attempts[0]["added"] == [{"pid": 102, "relation": "unobserved"}]
    assert all(row["unclassified_count"] == 1 for row in attempts)
    assert all(row["confirmed_disappeared_count"] == 0 for row in attempts)
    assert "synthetic-secret" not in json.dumps(detail)
    assert len(json.dumps(detail, separators=(",", ":")).encode()) <= 4096


@pytest.mark.parametrize("change", ["other-death", "owned-death"])
def test_departure_of_a_parsed_pid_is_classified_and_completes(change):
    # A PID read in the opening scan and gone by the closing list was already
    # classified by that read, so this churn no longer exhausts the attempts.
    # An owned departure stays in the members for refresh() to sample.
    world = ProcWorld(change)
    detail: dict[str, object] = {}
    members = [10, 20] if change == "owned-death" else [10]
    assert world.census(detail) == (members, True, True)
    assert world.scans == 2
    assert detail["phase"] == "complete"
    relation = "same-group" if change == "owned-death" else "different-group"
    attempts = detail["attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["removed"] == [{"pid": 20, "relation": relation}]


@pytest.mark.parametrize("change", ["birth", "other-death", "owned-death"])
def test_diagnostics_do_not_add_probes_or_change_refusal(change):
    traces = []
    for sink in (None, {}):
        world = ProcWorld(change)
        if change == "birth":
            with pytest.raises(owned.Unknown, match="^unstable /proc census$"):
                world.census(sink)
        else:
            world.census(sink)
        traces.append(world.calls)
    assert traces[0] == traces[1]


def test_broken_sink_does_not_replace_original_unknown():
    class BrokenSink(dict[str, object]):
        def clear(self):
            raise RuntimeError("synthetic broken sink")

    with pytest.raises(owned.Unknown, match="^unstable /proc census$"):
        ProcWorld("birth").census(BrokenSink())


def test_deadline_replaces_old_attempt_evidence_without_reading():
    detail: dict[str, object] = {"stale": True}
    with pytest.raises(owned.Unknown, match="^strict /proc census deadline$") as raised:
        owned.strict_linux(
            10,
            0,
            clock=lambda: 0,
            diagnostics=detail,
            listdir=lambda path: pytest.fail("deadline must preclude reads"),
        )
    assert detail == raised.value.census_failure
    assert detail == {"phase": "opening-deadline", "attempt_count": 0, "attempts": []}


def test_capture_before_witness_retains_acquire_detail_and_wrapper_status(
    request, monkeypatch, tmp_path
):
    world = ProcWorld("birth")

    def census(pgid, deadline):
        return world.census()

    cap = owned.OwnedGroup.capture(
        10,
        census=census,
        # A retained corpse cannot take live-positive acquisition. Preserve the
        # strict census-before-receipt failure this diagnostic test exercises.
        samples=lambda pids: {10: ProcessSample(True, "corpse")},
        identity=lambda pid: pytest.fail("no identity after failed census"),
        clock=lambda: 0,
        deadline=5,
        signal_group=lambda *args: pytest.fail("no signal"),
    )
    assert cap.last.status == "REFUSED"
    assert cap.last.reason == "Unknown:unstable /proc census"
    assert not cap.receipts
    detail = cap.probe["census_failure"]
    assert detail["attempt_count"] == 6
    runner = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    _install(runner, monkeypatch, cap)
    output = io.StringIO()
    assert (
        runner.run(
            ["synthetic"],
            timeout=1,
            grace=0,
            memory_mb=0,
            stream=output,
            run_id="synthetic-acquire-churn",
        )
        == 126
    )
    assert "acquire refused: Unknown:unstable /proc census" in output.getvalue()
    line = next(line for line in output.getvalue().splitlines() if "census_failure=" in line)
    assert json.loads(line.split("census_failure=", 1)[1]) == detail
    record = next((tmp_path / "run-bounded-runs").glob("*.acquire.json"))
    assert json.loads(record.read_text())["ownership"]["census_failure"] == detail


@pytest.mark.parametrize("sink", [[], object(), None])
def test_malformed_optional_sink_does_not_mask_refusal(sink):
    with pytest.raises(owned.Unknown, match="^unstable /proc census$") as raised:
        ProcWorld("birth").census(sink)
    assert raised.value.census_failure is not None


def test_sink_and_subsequent_calls_cannot_rewrite_refusal_evidence():
    detail: dict[str, object] = {}
    with pytest.raises(owned.Unknown) as raised:
        ProcWorld("birth").census(detail)
    saved = json.dumps(raised.value.census_failure, sort_keys=True)
    attempts = detail["attempts"]
    assert isinstance(attempts, list)
    attempts[0]["added"].clear()
    ProcWorld("owned-death").census(detail)
    assert json.dumps(raised.value.census_failure, sort_keys=True) == saved


def test_six_attempts_share_a_total_sixteen_id_budget():
    world = ProcWorld("birth")

    def listdir(path):
        world.scans += 1
        return ["10"] + ([str(pid) for pid in range(100, 200)] if world.scans % 2 == 0 else [])

    with pytest.raises(owned.Unknown) as raised:
        owned.strict_linux(10, 5, listdir=listdir, read=world.read, clock=lambda: 0)
    detail = raised.value.census_failure
    assert detail is not None
    attempts = detail["attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 6
    assert all(row["added_count"] == 100 for row in attempts)
    assert (
        sum(
            len(row[key])
            for row in attempts
            for key in ("added", "removed", "confirmed_disappeared")
        )
        == 16
    )
    assert len(json.dumps(detail, separators=(",", ":")).encode()) <= 4096


def test_confirmed_disappearance_is_not_classified_as_foreign():
    detail: dict[str, object] = {}
    listings = iter([["10", "20"], ["10"]])

    def read(path):
        if path == "/proc/20/stat":
            raise FileNotFoundError
        return "10 (synthetic) Z 1 10"

    def stat(path):
        raise FileNotFoundError

    result = owned.strict_linux(
        10,
        5,
        listdir=lambda path: next(listings),
        read=read,
        stat=stat,
        clock=lambda: 0,
        diagnostics=detail,
    )
    assert result == ([10], True, True)
    attempts = detail["attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["confirmed_disappeared_count"] == 1
    assert attempts[0]["confirmed_disappeared"] == [{"pid": 20, "relation": "unobserved"}]
    assert detail["phase"] == "complete"


def test_next_members_attempt_does_not_reuse_failed_census_evidence():
    world = ProcWorld("birth")
    cap = owned.OwnedGroup.capture(
        10,
        census=lambda pgid, deadline: world.census(),
        clock=lambda: 0,
        deadline=5,
        samples=lambda pids: {10: ProcessSample(True, "corpse")},
    )
    assert "census_failure" in cap.probe
    with pytest.raises(owned.Unknown, match="ownership deadline exhausted"):
        cap._members(0)
    assert "census_failure" not in cap.probe


def test_failed_churn_archive_preserves_filtered_diagnostics(tmp_path):
    from tests.unit.scripts.test_run_bounded_linux_spawn_load import (
        _retain_failure_evidence,
    )

    records = tmp_path / "records"
    records.mkdir()
    detail: dict[str, object] = {}
    with pytest.raises(owned.Unknown):
        ProcWorld("birth").census(detail)
    (records / "synthetic.acquire.json").write_text(
        json.dumps(
            {
                "phase": "acquire",
                "command": "not-in-archive",
                "ownership": {"census_failure": detail},
            }
        )
    )
    destination = tmp_path / "artifact.json"
    _retain_failure_evidence("census_failure=" + json.dumps(detail), 126, records, destination)
    archived = json.loads(destination.read_text())
    assert archived["rc"] == 126
    assert archived["records"][0]["census_failure"] == detail
    assert "command" not in archived["records"][0]
    assert len(destination.read_bytes()) <= 96 * 1024


def test_row_deadline_retains_partial_attempt_without_extra_clock_reads():
    calls = []
    ticks = iter([0, 5])

    def clock():
        calls.append("clock")
        return next(ticks)

    with pytest.raises(owned.Unknown, match="^strict /proc row deadline$") as raised:
        owned.strict_linux(
            10,
            5,
            listdir=lambda path: ["10"],
            clock=clock,
            read=lambda path: pytest.fail("deadline forbids row read"),
        )
    detail = raised.value.census_failure
    assert detail is not None
    assert detail["phase"] == "row-deadline"
    assert detail["attempt_count"] == 1
    assert calls == ["clock", "clock"]


def test_wrapped_enumeration_failure_reaches_capture_before_witness(monkeypatch):
    original = owned.strict_linux

    def unreadable(path):
        raise OSError(5, "synthetic exception text must not be copied")

    monkeypatch.setattr(owned, "sys", NS(platform="linux"))
    monkeypatch.setattr(
        owned,
        "strict_linux",
        lambda pgid, deadline: original(pgid, deadline, listdir=unreadable, clock=lambda: 0),
    )
    cap = owned.OwnedGroup.capture(
        10,
        clock=lambda: 0,
        deadline=5,
        samples=lambda pids: {10: ProcessSample(True, "corpse")},
    )
    assert cap.last.reason == "Unknown:strict membership probe failed"
    assert not cap.receipts
    assert cap.probe["census_failure"]["phase"] == "opening-scan"
    assert "synthetic exception" not in json.dumps(cap.probe["census_failure"])


def test_update_raising_sink_preserves_exact_read_trace():
    class BrokenSink(dict[str, object]):
        def update(self, *args, **kwargs):
            raise RuntimeError("synthetic broken update")

    baseline, broken = ProcWorld("birth"), ProcWorld("birth")
    for world, sink in ((baseline, None), (broken, BrokenSink())):
        with pytest.raises(owned.Unknown, match="^unstable /proc census$"):
            world.census(sink)
    assert baseline.calls == broken.calls


def test_artifact_write_failure_does_not_replace_original_gate_failure(tmp_path, capsys):
    from tests.unit.scripts.test_run_bounded_linux_spawn_load import (
        _retain_failure_evidence,
    )

    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("synthetic")
    _retain_failure_evidence("acquire refused", 126, tmp_path, blocked_parent / "artifact.json")
    assert "failure evidence unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("character", ["\U0001f600", "\x00"])
def test_artifact_byte_bound_includes_json_text_escaping(tmp_path, character):
    from tests.unit.scripts.test_run_bounded_linux_spawn_load import (
        _retain_failure_evidence,
    )

    records = tmp_path / "records"
    records.mkdir()
    (records / "synthetic.acquire.json").write_text(
        json.dumps(
            {"phase": "acquire", "ownership": {"census_failure": {"oversize": "x" * 100000}}}
        )
    )
    destination = tmp_path / "artifact.json"
    _retain_failure_evidence(character * 20000, 126, records, destination)
    assert len(destination.read_bytes()) <= 96 * 1024
    evidence = json.loads(destination.read_text())
    assert evidence["records"] == [{"diagnostic_truncated": True}]
    assert len(evidence["stderr"].encode("utf-8")) <= 16384
