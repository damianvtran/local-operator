"""The unit suite must not start the production broker daemon.

The autouse ``refuse_real_secret_broker_spawn`` fixture in ``tests/conftest.py``
is what enforces this. Without it, a plain store read in a test with a temp
config root reaches ``ensure_broker`` and spawns a detached brokerd rooted at
that temp directory; the daemon then outlives the test and races the fixture's
``TemporaryDirectory`` teardown (measured 2026-10-20: ``OSError: [Errno 39]
Directory not empty: '/tmp/tmp…/secrets'`` in run 39383034418, job 118496365150).

This cell is the gate's own negative control: it patches the SPAWN, so a suite
whose gate was removed starts one here and fails on ``spawned`` instead of
leaking a daemon somewhere less observable.
"""

from __future__ import annotations

import sys

import pytest

from local_operator.secrets import client

_SUPPORTED = sys.platform == "darwin" or sys.platform.startswith("linux")


@pytest.mark.skipif(not _SUPPORTED, reason="the production broker only spawns on macOS and Linux")
def test_a_unit_store_read_cannot_start_a_broker(tmp_path, monkeypatch) -> None:
    spawned: list[object] = []
    # The gate short-circuits before this is reached; if the gate is gone, the
    # real ``ensure_broker`` calls this and the assertion below names the leak.
    monkeypatch.setattr(client, "_spawn_broker", lambda base: spawned.append(base))

    assert client.ensure_broker(tmp_path, timeout=0.0) is False
    assert spawned == [], (
        "a unit test started the production secret broker. The autouse "
        "refuse_real_secret_broker_spawn fixture is missing or was bypassed; a "
        "broker here outlives the test and races its config-dir teardown."
    )
