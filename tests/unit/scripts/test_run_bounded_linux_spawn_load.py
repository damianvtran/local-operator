"""Real Linux gauge: a HEALTHY gate must survive ordinary spawn churn.

The strict census is exact by design — a PID that appears mid-scan invalidates
the scan — so on Linux the cadence matters: refreshing on every 50 ms poll meant
~40 full ``/proc`` scans a second, and four unstable scans in a row ended a gate
that was working. This runs the real wrapper against a real owned group while
children are forked underneath it.

CI's Linux leg carries this; it skips cleanly on Darwin, where the census is one
``proc_listpids`` syscall. Bounded: the child self-limits and the wrapper carries
a timeout, so a wedged run fails rather than hangs.
"""
from __future__ import annotations

import io
import json
import os
import sys
import textwrap

import pytest

from tests.unit.scripts.test_run_bounded import memory_wrapper  # noqa: F401

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="real /proc census under spawn load is a Linux-only gauge",
)

#: Children live briefly and exit on their own: this is churn, not a cleanup
#: test, so the group empties before the final census and the run is deterministic.
CHURN = textwrap.dedent(
    """
    import subprocess, sys, time
    for _ in range(20):
        subprocess.Popen([sys.executable, "-S", "-c", "import time; time.sleep(0.05)"])
        time.sleep(0.01)
    sys.exit(0)
    """
)


def test_a_healthy_gate_survives_spawn_churn_and_records_its_group(
        memory_wrapper, monkeypatch, tmp_path):  # noqa: F811
    wrapper = memory_wrapper
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(tmp_path))
    inherited = os.environ.get(wrapper.memory_guard.RESOLVED_MEMORY_MB_ENV, "")
    memory_mb = int(inherited) if inherited.isdigit() and int(inherited) > 0 else 0

    output = io.StringIO()
    rc = wrapper.run([sys.executable, "-S", "-c", CHURN], timeout=30, grace=1.0,
                     memory_mb=memory_mb, stream=output, run_id="linux-churn")

    assert rc == 0, output.getvalue()
    record_dir = tmp_path / "run-bounded-runs"
    exit_records = sorted(record_dir.glob("*.exit.json"))
    assert exit_records, output.getvalue()
    payload = json.loads(exit_records[0].read_text())
    assert payload["run_id"] == "linux-churn"
    assert payload["exit_code"] == 0
    assert payload["pgid"] == payload["actors"][0]["pid"]
    assert "span run_id=linux-churn" in output.getvalue()
