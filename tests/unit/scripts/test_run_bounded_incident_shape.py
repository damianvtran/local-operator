"""THE INCIDENT SHAPE, as a real process on a real kernel.

2026-10-30: a leader exited and left live descendants in its group; the cleanup
path had already reaped the leader, so the group number was no longer ours and
the sweep that exists to kill those descendants refused instead. This test is
that shape, run for real: the leader exits on purpose the moment it has forked a
long-lived grandchild, and the wrapper must reclaim the grandchild anyway — with
the run record naming how ownership was established.

POSIX, not Linux-only, and deliberately so: the pin needs the corpse visible in
the group census, and that was MEASURED on both platforms this ships for (Darwin
`strict_census -> [pid]`, `ps` row `Z <defunct>`). This file therefore runs on
the macOS leg of `tui-e2e` as well as the Linux unit shard, so the shape this PR
exists to fix is exercised on the operator's own platform rather than argued
about. A platform without `fork` skips; a POSIX platform that cannot produce the
shape FAILS with its reason named.

Bounded three ways so it cannot hang CI: the wrapper's own `timeout`, the
explicit wrapper deadline inside the test, and a bounded wait for the grandchild
to disappear. It spawns only inside the group it creates and signals nothing but
that group.
"""

from __future__ import annotations

import io
import json
import os
import sys
import textwrap
import time

import pytest

from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork") or os.name != "posix",
    reason="no fork: this shape cannot be produced and there is nothing to measure",
)

#: Fork a long-lived grandchild, announce its pid, then exit immediately. The
#: leader is gone before the wrapper's first poll tick, so the corpse — not a
#: live leader — is the only thing holding the group number.
INCIDENT = textwrap.dedent("""
    import subprocess, sys
    child = subprocess.Popen([sys.executable, "-S", "-c", "import time; time.sleep(120)"])
    print(child.pid, flush=True)
    sys.exit(0)
    """)


@pytest.fixture
def wrapper(request, monkeypatch, tmp_path):
    module = request.getfixturevalue("memory_wrapper")
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    return module


def _exit_record(tmp_path):
    records = sorted((tmp_path / "run-bounded-runs").glob("*.exit.json"))
    assert records, "a spawned run must leave an exit record"
    return json.loads(records[-1].read_text())


def test_the_incident_shape_is_reclaimed_and_the_record_shows_the_pin(wrapper, tmp_path):
    output = io.StringIO()
    rc = wrapper.run(
        [sys.executable, "-S", "-c", INCIDENT],
        timeout=60,
        grace=2.0,
        memory_mb=0,
        stream=output,
        run_id="incident-shape",
    )
    if rc != 0:
        # Quote the RECORD rather than guess: the owner reads the refusal class and
        # the phase before anything else, and a bare "rc != 0" would hide both.
        record = _exit_record(tmp_path)
        ownership = record.get("ownership") or {}
        raise AssertionError(
            f"rc={rc} refusal_class={ownership.get('refusal_class')!r} "
            f"anchor={ownership.get('anchor')!r} phase={record.get('phase')!r}\n"
            + output.getvalue()
        )

    record = _exit_record(tmp_path)
    ownership = record["ownership"]
    assert ownership["schema"] == "run-bounded-ownership/v1"
    assert ownership["anchor"] == "leader-reservation", ownership
    assert ownership["proof"], "the signal was proven against named identities"
    proven = {row["pid"] for row in ownership["proof"]}
    assert record["pgid"] in proven, (record["pgid"], proven)
    assert len(proven) >= 2, "the grandchild must be named in the proof set"

    # The grandchildren that were proven are GONE — the point of the whole slice.
    deadline = time.monotonic() + 10.0
    for pid in proven:
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)  # signal 0: delivers nothing, only asks
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            pytest.fail(f"proven member {pid} still exists after a clean rc=0 run")
