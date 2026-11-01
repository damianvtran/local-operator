"""The retained run record: every run binds itself to the group it drove.

A refused run needs this to name residuals. A SUCCESSFUL run needs it for
admission — the consuming lanes are expected to pass, and a green gate that
leaves no ownership record gives them nothing to bind.

Asserted here: the CLI contract, the on-disk schema, the success path, the
acquisition-failure path (believed identity, empty actors, still a record), and
that a broken filesystem degrades to a recorded warning rather than a hang.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
from types import SimpleNamespace as NS

import pytest

from local_operator import owned_process_group as owned
from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401


class World:
    def __init__(self, birth="leader"):
        # The creator row IS this process: the leader's kernel parent must be us
        # or C1 refuses the capture and no receipt exists.
        self.rows = {
            os.getpid(): (0, os.getpid(), os.getpid(), "creator"),
            10: (os.getpid(), 10, 10, birth),
        }
        #: Set when the faked child has exited; a corpse until it is reaped.
        self.exited = False
        self.now = 0.0
        self.calls = []
        self.sample_calls = 0

    def clock(self):
        self.now += 0.001
        return self.now

    def census(self, pgid, deadline):
        return sorted(pid for pid, row in self.rows.items() if row[1] == pgid)

    def samples(self, pids):
        # THE CORPSE IS AN EVENT, NOT A CALL COUNT. Counting samples couples the
        # fixture to the implementation's call sequence: one added witness read
        # turned a live leader into a corpse during capture and the receipt was
        # never recorded. `self.exited` says the child has exited; nothing else.
        self.sample_calls += 1
        corpse = bool(getattr(self, "exited", False))
        return {
            pid: owned.procstate.ProcessSample(corpse and pid == 10, self.rows[pid][3])
            for pid in pids
            if pid in self.rows
        }

    def identity(self, pid):
        return self.rows[pid][:3]

    def send(self, pgid, signum):
        """What the kernel does with a live owned group: it goes away."""
        self.calls.append((pgid, signum))
        self.rows.pop(pgid, None)

    def capture(self, send=None, identity=None):
        # The leader is OUR pid's child: C1 refuses anything else, and a refused
        # capture leaves no receipt, which every later refresh reports as
        # "no leader receipt" and the wrapper as an unproven cleanup.
        return owned.OwnedGroup.capture(
            10,
            deadline=self.clock() + 5.0,
            parent_pid=os.getpid(),
            census=self.census,
            samples=self.samples,
            identity=identity or self.identity,
            scheme=lambda: "fixture",
            clock=self.clock,
            signal_group=send or self.send,
        )


@pytest.fixture
def wrapper(request, monkeypatch, tmp_path):
    module = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    monkeypatch.delenv(module.RUN_ID_ENV, raising=False)
    return module


#: The world the current test built; the fake Popen marks its child as exited.
_CURRENT_WORLD = None


def _record_exit():
    """The faked child has exited: from here it is a corpse until reaped."""
    if _CURRENT_WORLD is not None:
        _CURRENT_WORLD.exited = True


def _install(wrapper, monkeypatch, cap, status=0):
    monkeypatch.setattr(
        wrapper,
        "owned_process_group",
        NS(OwnedGroup=NS(capture=lambda *a, **k: cap), Verdict=owned.Verdict),
    )
    monkeypatch.setattr(
        wrapper,
        "subprocess",
        NS(
            Popen=lambda *a, **k: NS(
                pid=10,
                poll=lambda: (_record_exit(), status)[1],
                wait=lambda **_: (_record_exit(), status)[1],
            ),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    monkeypatch.setattr(
        wrapper, "_memory_budget", lambda _: (NS(ceiling_mb=0, reason="fixture"), "fixture")
    )


def _records(tmp_path, suffix):
    return sorted((tmp_path / "run-bounded-runs").glob("*.%s.json" % suffix))


class FrozenClock:
    """An explicit, non-advancing clock: elapsed time is 0.0, not "approximately".

    Pinning the elapsed assertion to a frozen clock keeps the test deterministic
    on a loaded host without inventing a tolerance constant.
    """

    def monotonic(self):
        return 1_000.0

    def sleep(self, _seconds):
        return None


def test_a_successful_run_writes_spawn_and_exit_records(wrapper, monkeypatch, tmp_path):
    world = World()
    _install(wrapper, monkeypatch, world.capture())
    monkeypatch.setattr(wrapper, "time", FrozenClock())
    output = io.StringIO()
    assert (
        wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=output, run_id="qa-1234") == 0
    ), output.getvalue()

    spawn = json.loads(_records(tmp_path, "spawn")[0].read_text())
    assert spawn["schema"] == "run-bounded-run/v1"
    assert spawn["run_id"] == "qa-1234"
    assert spawn["wrapper_pid"] > 0
    assert spawn["pgid"] == 10
    assert spawn["phase"] == "spawn"
    assert spawn["wall_clock_start_utc"].endswith("Z")
    assert [actor["pid"] for actor in spawn["actors"]] == [10]
    assert spawn["actors"][0]["birth"] == "leader"
    assert spawn["actors"][0]["scheme"] == "fixture"

    exit_record = json.loads(_records(tmp_path, "exit")[0].read_text())
    assert exit_record["run_id"] == "qa-1234"
    assert exit_record["exit_code"] == 0
    assert exit_record["wall_clock_end_utc"].endswith("Z")
    assert exit_record["elapsed_seconds"] == 0.0
    assert exit_record["verdict"]["status"] in {"VERIFIED", "OBSERVED_EMPTY", "SIGNALED"}
    assert "span run_id=qa-1234 wrapper_pid=" in output.getvalue()
    assert "record (exit)" in output.getvalue()


def test_acquisition_failure_still_leaves_a_record_with_the_run_id(wrapper, monkeypatch, tmp_path):
    """`actors` is necessarily empty here — `believed` must not be."""
    world = World()

    def unusable(_pid):
        raise PermissionError(1, "Operation not permitted")

    _install(wrapper, monkeypatch, world.capture(identity=unusable))
    output = io.StringIO()
    rc = wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=output, run_id="qa-fail")
    assert rc == 126, output.getvalue()

    spawn = json.loads(_records(tmp_path, "spawn")[0].read_text())
    assert spawn["run_id"] == "qa-fail"
    assert spawn["actors"] == []
    assert spawn["believed"]["leader"] == 10
    # The recorded parent is THIS process: the fixture now models a leader that
    # really is our child, which is what C1 requires before a receipt exists.
    assert spawn["believed"]["parent"] == os.getpid()
    assert "error" in spawn["believed"]
    # A record exists, so this is distinguishable from "the wrapper never ran".
    assert _records(tmp_path, "exit"), "a recorded run must also record its exit"


def test_run_id_comes_from_the_flag_then_the_environment(wrapper, monkeypatch):
    assert wrapper._resolve_run_id("qa-7") == "qa-7"
    generated = wrapper._resolve_run_id(None)
    assert generated.startswith("run-") and len(generated) == 16
    monkeypatch.setenv(wrapper.RUN_ID_ENV, "env-9")
    assert wrapper._resolve_run_id(None) == "env-9"
    # Path traversal and separators cannot reach the filename.
    assert wrapper._resolve_run_id("../../etc/passwd") == "etc-passwd"
    assert wrapper._resolve_run_id("x" * 200) == "x" * 64
    # N1: "phase" is a legal caller id — the old placeholder sentinel ate it.
    assert wrapper._resolve_run_id("phase") == "phase"


def test_the_flag_reaches_the_run_contract(wrapper, monkeypatch):
    seen = []

    def fake_run(argv, **kwargs):
        seen.append((list(argv), kwargs))
        return 0

    monkeypatch.setattr(wrapper, "run", fake_run)
    assert wrapper.main(["--run-id", "qa-cli", "--", "true"]) == 0
    assert seen == [
        (
            ["true"],
            {
                "timeout": None,
                "grace": wrapper.DEFAULT_GRACE_SECONDS,
                "memory_mb": None,
                "run_id": "qa-cli",
            },
        )
    ]


def test_a_broken_record_directory_warns_and_does_not_change_the_gate(
    wrapper, monkeypatch, tmp_path
):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    monkeypatch.setattr(wrapper, "_record_dir", lambda: blocker / "run-bounded-runs")
    world = World()
    _install(wrapper, monkeypatch, world.capture())
    output = io.StringIO()
    assert (
        wrapper.run(["synthetic"], timeout=None, grace=0.05, stream=output, run_id="qa-broken") == 0
    )
    assert "record write failed (spawn)" in output.getvalue()
    assert "record write failed (exit)" in output.getvalue()
