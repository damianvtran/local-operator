"""The run record is published atomically: a reader never sees an empty/partial file.

QA root cause (session 378e07f7f782, report 6d487accf301; gate run 42785622232
job 129731536531): `_record_run` created the FINAL `<run_id>.exit.json` with
O_CREAT|O_EXCL and filled it afterwards. A deadline check between create and
write raised TimeoutError, swallowed as OSError, and left a ZERO-BYTE final file
with rc 0 -- `json.loads` then died with "Expecting value: line 1 column 1".

Every test drives the budget with an injected clock (no sleeps, no real time), so
the verdict does not depend on host load.
"""

from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace as NS

import pytest

from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401

_BUDGET_STEP = 1.0  # far larger than the 0.25 s record budget


class StepClock:
    """`time.monotonic` that jumps past the budget when `trip` says so.

    `trip` is consulted with the number of the next call; this keeps the jump at
    an exact, named step instead of at "whenever real time happens to pass".
    """

    def __init__(self, trip):
        self.trip = trip
        self.calls = 0
        self.now = 1_000.0

    def __call__(self):
        self.calls += 1
        if self.trip(self.calls):
            self.now += _BUDGET_STEP
        return self.now


@pytest.fixture
def wrapper(memory_wrapper, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    return memory_wrapper


def _record(wrapper, stream=None):
    owner = NS(pgid=10, receipts={}, probe={})
    return wrapper._record_run(
        owner, stream or io.StringIO(), "atomic", "exit", command=["x"], exit_code=0
    )


def _dir(tmp_path):
    return tmp_path / "run-bounded-runs"


def test_normal_run_publishes_a_complete_record_and_no_temp(wrapper, monkeypatch, tmp_path):
    clock = StepClock(lambda n: False)
    monkeypatch.setattr(wrapper.time, "monotonic", clock)
    path = _record(wrapper)
    assert path == _dir(tmp_path) / "atomic.exit.json"
    assert json.loads(path.read_text())["run_id"] == "atomic"
    assert [p.name for p in _dir(tmp_path).iterdir()] == ["atomic.exit.json"]


# The monotonic call that follows each step of the old create-then-fill code: 1 is
# the deadline's own start, 2 pre-mkdir, 3 pre-open, 4 pre-write, 5 pre-flush.
# Tripping ANY of 2..5 must leave no final file; 4 is the QA repro (after the
# final name was created, before its bytes were written).
@pytest.mark.parametrize("call", [2, 3, 4, 5])
def test_deadline_exceeded_mid_write_leaves_no_final_file(wrapper, monkeypatch, tmp_path, call):
    clock = StepClock(lambda n: n == call)
    monkeypatch.setattr(wrapper.time, "monotonic", clock)
    stream = io.StringIO()
    assert _record(wrapper, stream) is None
    # Same reporting contract as before: a warning, never an exception.
    assert "record write failed (exit)" in stream.getvalue()
    leftovers = list(_dir(tmp_path).iterdir()) if _dir(tmp_path).exists() else []
    assert leftovers == []  # neither a final file nor a stray temp


def test_oserror_mid_write_leaves_no_final_file_and_is_reported(wrapper, monkeypatch, tmp_path):
    def broken(_fd):
        raise OSError(5, "simulated fsync failure")

    monkeypatch.setattr(wrapper.os, "fsync", broken)
    stream = io.StringIO()
    assert _record(wrapper, stream) is None
    assert "record write failed (exit)" in stream.getvalue()
    assert "errno=5" in stream.getvalue()
    assert list(_dir(tmp_path).iterdir()) == []


def test_existing_final_record_is_kept_and_the_retry_gets_a_pid_name(
    wrapper, monkeypatch, tmp_path
):
    first = _record(wrapper)
    second = _record(wrapper)
    assert first.name == "atomic.exit.json"
    assert second.name == f"atomic.exit.{os.getpid()}.json"
    assert json.loads(first.read_text())["run_id"] == json.loads(second.read_text())["run_id"]
    assert sorted(p.name for p in _dir(tmp_path).iterdir()) == sorted([first.name, second.name])
